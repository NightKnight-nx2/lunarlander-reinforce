"""REINFORCE + value baseline + entropi bonusu (annealing'li / annealing'siz)."""
import csv, json, os, time
from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn.functional as F
import gymnasium as gym
from torch.distributions import Categorical

from policy import PolicyNet, ValueNet, forward_np

OUTCOMES = ["crash", "hover", "land"]
LOG_FIELDS = ["episode", "steps", "total_steps", "return", "outcome", "ent_coef", "policy_entropy", "time_s"]


@dataclass
class Config:
    name: str = "anneal"
    episodes: int = 15000
    seed: int = 0
    anneal: bool = True            # False -> kontrol grubu (sabit entropi katsayısı)
    ent_start: float = 0.02        # başlangıç entropi katsayısı
    anneal_frac: float = 0.3       # eğitimin ilk %30'unda doğrusal olarak 0'a iner
    gamma: float = 0.99
    lr_policy: float = 1e-2
    lr_value: float = 1e-2
    batch_episodes: int = 8        # her güncellemede kullanılan bölüm sayısı
    time_penalty: float = 0.1     # YALNIZCA eğitim sinyalinde adım başına ceza (loglanan getiri ham env ödülüdür)
    ckpt_every: int = 500
    out_dir: str = "runs"

    @property
    def run_dir(self):
        return os.path.join(self.out_dir, f"{self.name}_s{self.seed}")


def entropy_coef(cfg, ep):
    """ep: o ana kadar biten bölüm sayısı. Annealing: ent_start -> 0 (doğrusal), sonra 0'da kalır."""
    if not cfg.anneal:
        return cfg.ent_start
    T = cfg.anneal_frac * cfg.episodes
    return cfg.ent_start * max(0.0, 1.0 - ep / T)


def classify(terminated, truncated, last_reward):
    if truncated and not terminated:
        return "hover"                      # zaman aşımı: iniş de çarpma da yok
    return "land" if last_reward >= 100 else "crash"   # env: dinlenme +100, çarpma/ekran dışı -100


def rollout(env, w, rng, obs):
    S, A, R = [], [], []
    while True:
        _, _, p = forward_np(w, obs)
        a = min(int(np.searchsorted(np.cumsum(p), rng.random())), 3)
        S.append(obs); A.append(a)
        obs, r, term, trunc, _ = env.step(a)
        R.append(r)
        if term or trunc:
            return np.array(S, np.float32), np.array(A), np.array(R, np.float32), classify(term, trunc, r)


def discounted(R, gamma):
    G = np.zeros_like(R); run = 0.0
    for t in range(len(R) - 1, -1, -1):
        run = R[t] + gamma * run; G[t] = run
    return G


def load_log(path):
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    d = {k: np.array([r[k] for r in rows]) for k in LOG_FIELDS}
    for k in LOG_FIELDS:
        if k != "outcome":
            d[k] = d[k].astype(float)
    return d


class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        torch.manual_seed(cfg.seed)
        self.rng = np.random.default_rng(cfg.seed)
        self.policy, self.value = PolicyNet(), ValueNet()
        self.opt_p = torch.optim.Adam(self.policy.parameters(), lr=cfg.lr_policy)
        self.opt_v = torch.optim.Adam(self.value.parameters(), lr=cfg.lr_value)
        self.rets, self.outs = [], []
        self.total_steps = 0

    def ckpt(self, tag):
        d = os.path.join(self.cfg.run_dir, "ckpt"); os.makedirs(d, exist_ok=True)
        torch.save({"policy": self.policy.state_dict(), "episode": len(self.rets)}, os.path.join(d, f"{tag}.pt"))

    def update(self, batch, beta):
        cfg = self.cfg
        S = torch.as_tensor(np.concatenate([b[0] for b in batch]))
        A = torch.as_tensor(np.concatenate([b[1] for b in batch]))
        G = torch.as_tensor(np.concatenate([discounted(b[2] - cfg.time_penalty, cfg.gamma) for b in batch])) / 100.0
        V = self.value(S)
        adv = (G - V.detach())
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        dist = Categorical(logits=self.policy(S))
        ent = dist.entropy()
        loss_p = -(dist.log_prob(A) * adv).mean() - beta * ent.mean()
        self.opt_p.zero_grad(); loss_p.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0); self.opt_p.step()
        loss_v = F.mse_loss(V, G)
        self.opt_v.zero_grad(); loss_v.backward(); self.opt_v.step()
        return np.array_split(ent.detach().numpy(), np.cumsum([len(b[1]) for b in batch])[:-1])

    def run(self, callback=None, verbose=True):
        cfg = self.cfg
        os.makedirs(cfg.run_dir, exist_ok=True)
        json.dump(asdict(cfg), open(os.path.join(cfg.run_dir, "config.json"), "w"), indent=1)
        env = gym.make("LunarLander-v3")
        obs, _ = env.reset(seed=cfg.seed)
        f = open(os.path.join(cfg.run_dir, "log.csv"), "w", newline="", encoding="utf-8")
        wr = csv.DictWriter(f, LOG_FIELDS); wr.writeheader()
        self.ckpt("ep0000"); t0 = time.time(); best_roll = -1e9
        while len(self.rets) < cfg.episodes:
            ep0 = len(self.rets); beta = entropy_coef(cfg, ep0)
            w = self.policy.numpy_weights(); batch = []
            for _ in range(min(cfg.batch_episodes, cfg.episodes - ep0)):
                S, A, R, out = rollout(env, w, self.rng, obs)
                obs, _ = env.reset()
                batch.append((S, A, R, out))
            ents = self.update(batch, beta)
            for (S, A, R, out), e in zip(batch, ents):
                self.rets.append(float(R.sum())); self.outs.append(out); self.total_steps += len(R)
                wr.writerow(dict(episode=len(self.rets), steps=len(R), total_steps=self.total_steps,
                                 **{"return": f"{R.sum():.3f}"}, outcome=out, ent_coef=f"{beta:.5f}",
                                 policy_entropy=f"{e.mean():.4f}", time_s=f"{time.time()-t0:.1f}"))
            f.flush()
            n = len(self.rets)
            if n >= 100 and np.mean(self.rets[-100:]) > best_roll:
                best_roll = np.mean(self.rets[-100:]); self.ckpt("best")
            if n // cfg.ckpt_every != ep0 // cfg.ckpt_every:
                self.ckpt(f"ep{(n // cfg.ckpt_every) * cfg.ckpt_every:04d}")
            if verbose and n // 100 != ep0 // 100:
                last = self.outs[-100:]
                print(f"[{cfg.name} s{cfg.seed}] ep {n:5d} | ret100 {np.mean(self.rets[-100:]):7.1f} | "
                      f"land {last.count('land'):3d} hover {last.count('hover'):3d} crash {last.count('crash'):3d} | "
                      f"beta {beta:.4f} | {time.time()-t0:5.0f}s", flush=True)
            if callback: callback(self, n)
        self.ckpt("final"); f.close(); env.close()
        return self
