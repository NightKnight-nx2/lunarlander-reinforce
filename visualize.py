"""Canlı görselleştirme paneli + karşılaştırmalı grafikler.

  python visualize.py compare --runs runs            # annealing vs kontrol grafiği (PNG)
  python visualize.py replay  --run runs/anneal_s0   # kayıtlı checkpoint'lerle "öğrenme hikayesi" (pencere)
  python visualize.py replay  --run runs/anneal_s0 --save hikaye.mp4   # aynısını videoya kaydet
  python visualize.py watch   --ckpt runs/anneal_s0/ckpt/final.pt      # tek modeli canlı izle
"""
import argparse, glob, os, sys, time
import numpy as np
import torch
import gymnasium as gym
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.gridspec import GridSpec

from policy import PolicyNet, forward_np, INPUT_NAMES, ACTION_NAMES, INPUT_SCALE
from reinforce import load_log

# ---- renkler (dataviz referans paleti: slot 1 mavi = annealing, slot 2 turuncu = kontrol) ----
C_ANNEAL, C_CONTROL = "#2a78d6", "#eb6834"
C_OUT = {"land": "#1baf7a", "hover": "#eda100", "crash": "#d55181"}
OUT_TR = {"land": "İNİŞ", "hover": "HAVADA KALDI (zaman aşımı)", "crash": "ÇARPTI"}
BG, PANEL, INK, INK2, DIM = "#0f1117", "#171a22", "#f0efec", "#a9a8a0", "#2b2f3a"
POS, NEG, OUTC = np.array([255, 170, 40]) / 255, np.array([70, 160, 255]) / 255, np.array([80, 230, 140]) / 255


def _rgba(img):
    """RGB -> RGBA uint8: matplotlib'in yavaş maskeli-dizi dönüşümünü atlar (kare başına ~2x hızlı)."""
    out = np.empty(img.shape[:2] + (4,), np.uint8); out[..., :3] = img; out[..., 3] = 255
    return out


class WindowClosed(Exception):
    pass


def _envelope(x, y, bins):
    """Uzun seriyi (min,max) çiftleriyle seyrelt: çizim hafif kalır, aralık bilgisi korunur."""
    n = len(y)
    if n <= 2 * bins: return x, y
    k = int(np.ceil(n / bins)); m = n // k
    yy = np.asarray(y[:m * k]).reshape(m, k); xs = np.asarray(x[:m * k]).reshape(m, k)[:, 0]
    return np.repeat(xs, 2), np.column_stack([yy.min(1), yy.max(1)]).reshape(-1)


def rolling(x, w):
    x = np.asarray(x, float)
    if len(x) < w:
        w = max(1, len(x))
    c = np.cumsum(np.insert(x, 0, 0)); out = np.empty(len(x))
    for i in range(len(x)):
        lo = max(0, i + 1 - w); out[i] = (c[i + 1] - c[lo]) / (i + 1 - lo)
    return out


def glow(v, signed=True):
    """Aktivasyon -> renk: parlaklık |v|, ton işaret (turuncu +, mavi -)."""
    v = np.asarray(v, float); base = np.array(DIM_RGB)
    if signed:
        tgt = np.where(v[:, None] >= 0, POS, NEG); a = np.clip(np.abs(v), 0, 1)[:, None]
    else:
        tgt = OUTC[None, :]; a = np.clip(v, 0, 1)[:, None]
    return np.clip(base + a * (tgt - base), 0, 1)


DIM_RGB = matplotlib.colors.to_rgb("#3a3f4d")


# ============================ Canlı panel ============================
class Dashboard:
    """(a) lander uçuşu  (b) ağ diyagramı  (c) canlı return grafiği  (d) bölüm no + en iyi sonuç."""

    def __init__(self, w, title="LunarLander · REINFORCE", record=False):
        self.w = w; self.frames = [] if record else None
        self.period = 0.04; self._bg = None; self._t = time.perf_counter(); self._animating = False
        self.fig = plt.figure(figsize=(12.8, 7.2), dpi=80, facecolor=BG)
        gs = GridSpec(2, 2, figure=self.fig, width_ratios=[1.0, 1.05], height_ratios=[1.5, 1],
                      left=0.04, right=0.98, top=0.90, bottom=0.08, wspace=0.10, hspace=0.28)
        self.ax_env, self.ax_net = self.fig.add_subplot(gs[0, 0]), self.fig.add_subplot(gs[0, 1])
        self.ax_ret, self.ax_info = self.fig.add_subplot(gs[1, 0]), self.fig.add_subplot(gs[1, 1])
        self.sup = self.fig.text(0.5, 0.955, title, color=INK, ha="center", fontsize=15, weight="bold")
        for ax in (self.ax_env, self.ax_net, self.ax_info):
            ax.set_facecolor(PANEL); ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values(): s.set_visible(False)
        self.ax_env.set_title("(a) Lander – gerçek zamanlı uçuş", color=INK2, fontsize=10, loc="left")
        self.ax_net.set_title("(b) Politika ağı aktivasyonları", color=INK2, fontsize=10, loc="left")
        self.im = self.ax_env.imshow(_rgba(np.zeros((400, 600, 3), np.uint8)), interpolation="nearest")
        self._build_net(); self._build_curve()
        self.info = self.ax_info.text(0.05, 0.93, "", transform=self.ax_info.transAxes, color=INK,
                                      family="monospace", fontsize=12, va="top", linespacing=1.6)
        self.ax_info.set_title("(d) Durum", color=INK2, fontsize=10, loc="left")
        self.curve_ep, self.best, self.best_ep, self.roll, self.coef = 0, None, 0, None, None
        self.banner = self.ax_env.text(0.5, 0.06, "", transform=self.ax_env.transAxes, ha="center",
                                       color="white", fontsize=14, weight="bold",
                                       bbox=dict(boxstyle="round,pad=0.4", fc="none", ec="none"))
        # Sadece her karede değişen sanatçılar blit ile çizilir; geri kalan (eksenler, return eğrisi) önbellekte kalır.
        self.dyn = [self.im, self.sc_in, self.sc_h, self.sc_o, self.lc1, self.lc2, *self.out_txt, self.info, self.banner]
        # Araç çubuğu pan/zoom'u diyagram panellerini kaydırmasın (sadece return grafiği gezilebilir)
        for ax in (self.ax_env, self.ax_net, self.ax_info): ax.set_navigate(False)
        self.fig.canvas.mpl_connect("draw_event", self._on_draw)
        if not record:
            plt.show(block=False); self.fig.canvas.draw()

    # ---- ağ diyagramı ----
    def _build_net(self):
        ax = self.ax_net
        ys_in, ys_h, ys_o = (3.5 - np.arange(8)) * 2.0, 7.5 - np.arange(16), (1.5 - np.arange(4)) * 4.0
        self.xs = (0.0, 1.0, 2.0)
        P = lambda x, ys: np.column_stack([np.full(len(ys), x), ys])
        self.pin, self.ph, self.po = P(0, ys_in), P(1, ys_h), P(2, ys_o)
        self.segs1 = [[self.pin[i], self.ph[j]] for j in range(16) for i in range(8)]   # idx j*8+i
        self.segs2 = [[self.ph[j], self.po[k]] for k in range(4) for j in range(16)]    # idx k*16+j
        self.lc1 = LineCollection(self.segs1, linewidths=0.8); self.lc2 = LineCollection(self.segs2, linewidths=1.0)
        ax.add_collection(self.lc1); ax.add_collection(self.lc2)
        self.sc_in = ax.scatter(*self.pin.T, s=230, edgecolors="#555b6b", linewidths=1, zorder=3)
        self.sc_h = ax.scatter(*self.ph.T, s=95, edgecolors="#555b6b", linewidths=1, zorder=3)
        self.sc_o = ax.scatter(*self.po.T, s=420, edgecolors="#555b6b", linewidths=1, zorder=3)
        for y, n in zip(ys_in, INPUT_NAMES): ax.text(-0.12, y, n, ha="right", va="center", color=INK2, fontsize=9)
        self.out_txt = [ax.text(2.07, y, n, ha="left", va="center", color=INK2, fontsize=9)
                        for y, n in zip(ys_o, ACTION_NAMES)]
        ax.text(0, 8.6, "girdi (8)", ha="center", color=INK2, fontsize=9)
        ax.text(1, 8.6, "gizli (16, tanh)", ha="center", color=INK2, fontsize=9)
        ax.text(2, 8.6, "çıktı (4, olasılık)", ha="center", color=INK2, fontsize=9)
        ax.set_xlim(-0.85, 2.8); ax.set_ylim(-8.4, 9.2)
        ax.text(0.01, 0.01, "parlaklık = |aktivasyon|   turuncu: +   mavi: −", transform=ax.transAxes,
                color=INK2, fontsize=8)
        self.update_net(np.zeros(8), np.zeros(16), np.ones(4) / 4)

    def update_net(self, obs, h, p):
        x = np.clip(np.asarray(obs) / INPUT_SCALE, -1, 1)
        self.sc_in.set_facecolors(glow(x)); self.sc_h.set_facecolors(glow(h)); self.sc_o.set_facecolors(glow(p, False))
        W1, W2 = self.w["W1"], self.w["W2"]
        s1 = (W1 * x[None, :]).reshape(-1)              # j*8+i
        s2 = (W2 * h[None, :]).reshape(-1)              # k*16+j
        for lc, s, k in ((self.lc1, s1, 0.9), (self.lc2, s2, 0.9)):
            col = np.where(s[:, None] >= 0, POS, NEG)
            rgba = np.concatenate([col, np.clip(np.abs(s) * k, 0.03, 0.9)[:, None]], 1)
            lc.set_colors(rgba)
        for t, pr, n in zip(self.out_txt, p, ACTION_NAMES): t.set_text(f"{n} {pr*100:4.0f}%")

    # ---- return grafiği ----
    def _build_curve(self):
        ax = self.ax_ret; ax.set_facecolor(PANEL)
        ax.set_title("(c) Bölüm başına getiri (return)", color=INK2, fontsize=10, loc="left")
        ax.tick_params(colors=INK2, labelsize=8)
        for s in ax.spines.values(): s.set_color(DIM)
        ax.grid(color=DIM, lw=0.5, alpha=0.6)
        ax.axhline(200, color=C_OUT["land"], lw=0.8, ls="--"); ax.axhline(0, color=DIM, lw=0.8)
        (self.l_raw,) = ax.plot([], [], color="#6f7686", lw=0.5, alpha=0.8)
        (self.l_roll,) = ax.plot([], [], color="#6ea8f0", lw=2)
        (self.l_now,) = ax.plot([], [], "o", color="white", ms=7, mec=BG, mew=2)
        ax.set_xlabel("bölüm", color=INK2, fontsize=8)

    def set_curve(self, rets, episode, best, best_ep, coef=None):
        self.rets = np.asarray(rets, float)[:max(1, episode)]
        x = np.arange(1, len(self.rets) + 1); r = rolling(self.rets, 100)
        self.l_raw.set_data(*_envelope(x, self.rets, 1200)); self.l_raw.set_rasterized(True)
        self.l_roll.set_data(x[::max(1, len(x) // 3000)], r[::max(1, len(x) // 3000)]); self.l_now.set_data([x[-1]], [r[-1]])
        self.ax_ret.set_xlim(0, max(50, self.total_eps if hasattr(self, "total_eps") else len(x)))
        lo = min(-400, float(self.rets.min()) - 20) if len(x) else -400
        self.ax_ret.set_ylim(max(lo, -700), 330)
        self.curve_ep, self.best, self.best_ep, self.roll, self.coef = episode, best, best_ep, float(r[-1]), coef
        self._bg = None   # statik içerik değişti -> önbellek yenilenecek

    # ---- kare güncelleme ----
    def frame(self, img, obs, h, p, step, ret, status="", status_col=None, hdr=""):
        self.im.set_data(_rgba(img)); self.update_net(obs, h, p)
        self.info.set_text(
            f"{hdr}"
            f"EĞİTİM BÖLÜMÜ   : {self.curve_ep}\n"
            f"EN İYİ SONUÇ    : {self.best:6.1f} (bölüm {self.best_ep})\n"
            f"SON 100 ORT.    : {self.roll:6.1f}\n"
            f"ENTROPİ KATSAYI : {self.coef:.4f}\n"
            f"──────────────────────────\n"
            f"bu uçuş: adım {step:4d}  getiri {ret:7.1f}")
        self.banner.set_text(status)
        self.banner.get_bbox_patch().set_facecolor(status_col or "none")
        self.banner.get_bbox_patch().set_edgecolor("none")
        self.refresh()

    # ---- blit altyapısı ----
    def _on_draw(self, event):
        if self._animating and self.frames is None:
            self._bg = self.fig.canvas.copy_from_bbox(self.fig.bbox)

    def begin(self):
        """Animasyon başlar: değişen sanatçılar tam çizimden çıkarılıp blit ile çizilir."""
        if self.frames is None:
            self._animating = True; self._bg = None
            for a in self.dyn: a.set_animated(True)

    def finish(self):
        """Animasyon biter: her şey normal çizime döner (araç çubuğu, yeniden boyutlandırma güvenli olur)."""
        if self.frames is None:
            self._animating = False; self._bg = None
            for a in self.dyn: a.set_animated(False)
            if plt.fignum_exists(self.fig.number): self.fig.canvas.draw_idle()

    def refresh(self):
        c = self.fig.canvas
        if self.frames is not None:                       # video kaydı: tam çizim
            c.draw(); self.frames.append(np.asarray(c.buffer_rgba())[..., :3].copy()); return
        if not plt.fignum_exists(self.fig.number):
            raise WindowClosed()
        if not self._animating: self.begin()
        if self._bg is None:
            c.draw()                                      # draw_event -> _bg önbelleğe alınır
            if self._bg is None: self._bg = c.copy_from_bbox(self.fig.bbox)
        c.restore_region(self._bg)
        for a in self.dyn: a.axes.draw_artist(a)
        c.blit(self.fig.bbox)
        # gerçek zamanlı hız: kalan süreyi GUI olay döngüsüne ver (pan/zoom/kapatma tepki verir)
        left = self.period - (time.perf_counter() - self._t)
        c.start_event_loop(max(0.001, left)); self._t = time.perf_counter()


def demo_episode(dash, env, w, seed, frame_skip=2, max_shown=1000, greedy=False, hold=0, hdr=""):
    """Bir uçuşu çalıştırıp panele çizer. Döndürür: (getiri, sonuç)."""
    from reinforce import classify
    dash.w = w; dash.period = frame_skip / 50.0; dash.begin()
    rng = np.random.default_rng(seed); obs, _ = env.reset(seed=seed); total, step, out = 0.0, 0, "hover"
    while True:
        h, z, p = forward_np(w, obs)
        if step % frame_skip == 0:
            dash.frame(env.render(), obs, h, p, step, total, hdr=hdr)
        a = int(np.argmax(p)) if greedy else min(int(np.searchsorted(np.cumsum(p), rng.random())), 3)
        obs, r, term, trunc, _ = env.step(a); total += r; step += 1
        if term or trunc:
            out = classify(term, trunc, r); break
        if step >= max_shown:
            out = "hover"; break
    h, z, p = forward_np(w, obs)
    for _ in range(max(1, hold)):
        dash.frame(env.render(), obs, h, p, step, total, OUT_TR[out], C_OUT[out] + "cc", hdr=hdr)
    dash.finish()
    return total, out


def _run_info(rets, upto):
    r = np.asarray(rets[:max(1, upto)], float)
    i = int(np.argmax(r)) if len(r) else 0
    return float(r[i]) if len(r) else 0.0, i + 1


def make_live_callback(every, total_eps, frame_skip=2):
    """train.py --live için: her `every` bölümde güncel politikayla bir uçuş gösterir."""
    env = gym.make("LunarLander-v3", render_mode="rgb_array"); st = {"dash": None, "last": 0}

    def cb(tr, n):
        if st["dash"] is not None:
            if not plt.fignum_exists(st["dash"].fig.number): return   # pencere kapatıldı: eğitim sessizce sürer
            st["dash"].fig.canvas.flush_events()                         # eğitim sırasında pencere donmasın
        if n - st["last"] < every: return
        st["last"] = n
        if st["dash"] is None:
            st["dash"] = Dashboard(tr.policy.numpy_weights(), f"EĞİTİM CANLI · {tr.cfg.name} (seed {tr.cfg.seed})")
            st["dash"].total_eps = total_eps
        d = st["dash"]; b, be = _run_info(tr.rets, n)
        d.set_curve(tr.rets, n, b, be, coef=__import__("reinforce").entropy_coef(tr.cfg, n))
        try:
            demo_episode(d, env, tr.policy.numpy_weights(), seed=n, frame_skip=frame_skip, hold=8)
        except WindowClosed:
            pass
    return cb


# ============================ Karşılaştırma ============================
def _group_runs(root):
    g = {}
    for d in sorted(glob.glob(os.path.join(root, "*_s*"))):
        if os.path.exists(os.path.join(d, "log.csv")):
            g.setdefault(os.path.basename(d).rsplit("_s", 1)[0], []).append(load_log(os.path.join(d, "log.csv")))
    return g


def success_series(outs, w=100):
    return rolling((np.asarray(outs) == "land").astype(float), w) * 100


def compare(root, out_png, w=100):
    g = _group_runs(root); style = {"anneal": (C_ANNEAL, "Annealing'li"), "control": (C_CONTROL, "Annealing'siz (kontrol)")}
    plt.switch_backend("Agg")
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#c9c8c2", "axes.labelcolor": "#52514e",
                         "xtick.color": "#52514e", "ytick.color": "#52514e"})
    fig, axs = plt.subplots(2, 2, figsize=(13, 8.2), facecolor="#fcfcfb")
    (a1, a2), (a3, a4) = axs
    for ax in axs.flat:
        ax.set_facecolor("#fcfcfb"); ax.grid(color="#e6e5e0", lw=0.7); ax.spines[["top", "right"]].set_visible(False)
    nseed = max(len(v) for v in g.values())
    for name, logs in g.items():
        col, lab = style.get(name, ("#6250d6", name)); n = min(len(l["return"]) for l in logs); x = np.arange(1, n + 1)
        R = np.array([rolling(l["return"][:n], w) for l in logs]); S = np.array([success_series(l["outcome"][:n], w) for l in logs])
        E = np.array([l["ent_coef"][:n] for l in logs]); H = np.array([rolling(l["policy_entropy"][:n], w) for l in logs])
        if len(logs) == 1:
            a1.plot(x, logs[0]["return"][:n], color=col, lw=0.4, alpha=0.18)
        for ax, M in ((a1, R), (a2, S)):
            if len(logs) > 1: ax.fill_between(x, M.min(0), M.max(0), color=col, alpha=0.15, lw=0)
            ax.plot(x, M.mean(0), color=col, lw=2, label=lab)
        a3.plot(x, E.mean(0), color=col, lw=2, label=f"{lab}: entropi katsayısı")
        a3b = a3.twinx() if False else None
        a4.plot(x, H.mean(0), color=col, lw=2, label=lab)
    a1.axhline(200, color="#1baf7a", lw=1, ls="--"); a1.text(5, 205, "200 = 'çözüldü' eşiği", color="#1baf7a", fontsize=8)
    a1.set_title(f"Bölüm başına getiri ({w} bölüm hareketli ortalama)", loc="left", weight="bold")
    a2.set_title(f"İniş başarı oranı (son {w} bölümde %, 'land' sonucu)", loc="left", weight="bold"); a2.set_ylim(-2, 102)
    a3.set_title("Entropi bonusu katsayısı (zamanlama)", loc="left", weight="bold")
    a4.set_title(f"Politikanın gerçek entropisi (nat, {w} böl. ort.; maks ln4=1.39)", loc="left", weight="bold")
    for ax in (a1, a2, a3, a4): ax.set_xlabel("bölüm")
    a1.set_ylabel("getiri"); a2.set_ylabel("% iniş"); a3.set_ylabel("β"); a4.set_ylabel("entropi")
    a1.legend(frameon=False, loc="lower right"); a2.legend(frameon=False, loc="upper left")
    fig.suptitle(f"LunarLander-v3 · REINFORCE + value baseline · {nseed} seed" + ("" if nseed == 1 else " (çizgi=ortalama, bant=min–maks)"),
                 x=0.01, ha="left", fontsize=13, weight="bold", color="#0b0b0b")
    fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(out_png, dpi=130); plt.close(fig)
    return out_png


def outcome_table(root):
    return _group_runs(root)


# ============================ Replay / watch ============================
def _weights_from_ckpt(path):
    ck = torch.load(path, map_location="cpu"); p = PolicyNet(); p.load_state_dict(ck["policy"])
    return p.numpy_weights(), ck.get("episode", 0)


def replay(run_dir, stages=6, save=None, frame_skip=2, max_shown=600, seed=123):
    if save: plt.switch_backend("Agg")
    log = load_log(os.path.join(run_dir, "log.csv")); N = len(log["return"])
    ep_of = lambda c: int(os.path.basename(c)[2:-3])
    ck = sorted(glob.glob(os.path.join(run_dir, "ckpt", "ep*.pt")), key=ep_of)
    ck = [c for c in ck if ep_of(c) < N] + [os.path.join(run_dir, "ckpt", "final.pt")]
    idx = np.unique(np.linspace(0, len(ck) - 1, min(stages, len(ck))).round().astype(int)); ck = [ck[i] for i in idx]
    env = gym.make("LunarLander-v3", render_mode="rgb_array"); w, _ = _weights_from_ckpt(ck[0])
    d = Dashboard(w, f"ÖĞRENME HİKAYESİ · {os.path.basename(run_dir)}", record=bool(save)); d.total_eps = N
    try:
        for k, c in enumerate(ck):
            w, ep = _weights_from_ckpt(c); ep = N if c.endswith("final.pt") else ep
            b, be = _run_info(log["return"], ep)
            d.set_curve(log["return"], ep, b, be, coef=float(log["ent_coef"][max(0, ep - 1)]))
            demo_episode(d, env, w, seed + k, frame_skip, max_shown, hold=20, hdr=f"AŞAMA {k+1}/{len(ck)}\n")
    except WindowClosed:
        return
    if save:
        import imageio.v2 as imageio
        if save.endswith(".gif"):
            imageio.mimsave(save, d.frames, duration=1000 * frame_skip / 50, loop=0)
        else:
            imageio.mimsave(save, d.frames, fps=25, macro_block_size=1)
        print("kaydedildi:", save, f"({len(d.frames)} kare)")
    else:
        plt.show()


def watch(ckpt, episodes=5, greedy=False):
    env = gym.make("LunarLander-v3", render_mode="rgb_array"); w, ep = _weights_from_ckpt(ckpt)
    d = Dashboard(w, f"MODEL İZLE · {os.path.basename(ckpt)}"); d.curve_ep = ep
    d.best, d.best_ep, d.roll, d.coef = 0.0, 0, 0.0, 0.0
    d.set_curve([0.0], ep, 0.0, 0)
    try:
        for i in range(episodes):
            r, o = demo_episode(d, env, w, seed=1000 + i, greedy=greedy, hold=15); print(f"uçuş {i+1}: getiri {r:.1f} sonuç {o}")
    except WindowClosed:
        return
    plt.show()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("compare"); a.add_argument("--runs", default="runs"); a.add_argument("--out", default=None)
    b = sub.add_parser("replay"); b.add_argument("--run", required=True); b.add_argument("--save"); b.add_argument("--stages", type=int, default=6)
    c = sub.add_parser("watch"); c.add_argument("--ckpt", required=True); c.add_argument("--episodes", type=int, default=5); c.add_argument("--greedy", action="store_true")
    A = ap.parse_args()
    if A.cmd == "compare": print(compare(A.runs, A.out or os.path.join(A.runs, "karsilastirma.png")))
    elif A.cmd == "replay": replay(A.run, A.stages, A.save)
    else: watch(A.ckpt, A.episodes, A.greedy)
