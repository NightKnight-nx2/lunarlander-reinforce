"""Ana script: annealing'li ve annealing'siz (kontrol) iki koşu, özet ve karşılaştırma grafiği.

  python train.py --quick            # birkaç yüz bölümlük test
  python train.py                    # tam eğitim (varsayılan 15000 bölüm, seed 0)
  python train.py --seeds 0 1        # birden çok seed
  python train.py --live             # eğitim sırasında canlı pencere (koşular sırayla)
"""
import argparse, glob, json, multiprocessing as mp, os, sys
import numpy as np
from reinforce import Config, Trainer, load_log


def run_one(cfg):
    import torch; torch.set_num_threads(1)
    Trainer(cfg).run()
    return cfg.run_dir


def run_live(cfg, every):
    import visualize
    Trainer(cfg).run(callback=visualize.make_live_callback(every, cfg.episodes))


def summarize(root, episodes):
    from visualize import _group_runs
    g = _group_runs(root); ent0 = json.load(open(sorted(glob.glob(os.path.join(root, 'anneal_s*', 'config.json')))[0]))['ent_start']; W = min(500, max(50, episodes // 5)); lines = []
    P = lambda s: (lines.append(s), print(s))
    P("=" * 78); P("ÖZET"); P("=" * 78)
    res = {}
    for name, logs in g.items():
        for i, l in enumerate(logs):
            outs = l["outcome"]; n = len(outs)
            cnt = {k: int((outs == k).sum()) for k in ("land", "hover", "crash")}
            rate = lambda a: 100 * float((a == "land").mean()); T = int(0.3 * n)
            res[f"{name}_{i}"] = dict(pre=rate(outs[T - W:T]), steps=int(l["total_steps"][-1]), counts=cnt, first=rate(outs[:W]), last=rate(outs[-W:]),
                                      ret_first=float(l["return"][:W].mean()), ret_last=float(l["return"][-W:].mean()),
                                      best=float(l["return"].max()), time=float(l["time_s"][-1]))
            r = res[f"{name}_{i}"]
            P(f"\n[{name}] seed-sırası {i}: {n} bölüm, toplam {r['steps']:,} adım, {r['time']:.0f} sn")
            P(f"  sonuç dağılımı : land {cnt['land']} (%{100*cnt['land']/n:.1f}) | hover {cnt['hover']} (%{100*cnt['hover']/n:.1f}) | crash {cnt['crash']} (%{100*cnt['crash']/n:.1f})")
            P(f"  iniş başarısı  : ilk {W} bölüm %{r['first']:.1f}  |  annealing bitiminden önceki {W} bölüm %{r['pre']:.1f}  |  son {W} bölüm %{r['last']:.1f}")
            P(f"  ort. getiri    : ilk {W} bölüm {r['ret_first']:.1f}  ->  son {W} bölüm {r['ret_last']:.1f} | en iyi tek bölüm {r['best']:.1f}")
    a = [v for k, v in res.items() if k.startswith("anneal")]; c = [v for k, v in res.items() if k.startswith("control")]
    if a and c:
        P("\n" + "-" * 78)
        P(f"Annealing'li koşu, seed ortalaması (β başlangıç={ent0:.2f}, bölümlerin ilk %30'unda 0'a iner):")
        P(f"  annealing ÖNCESİ (bitimden önceki {W} bölüm, β>0): %{np.mean([v['pre'] for v in a]):.1f}  ->  SONRASI (son {W} bölüm, β=0): %{np.mean([v['last'] for v in a]):.1f}")
        P(f"Son {W} bölümde annealing'li vs kontrol (sabit β) iniş başarısı:  %{np.mean([v['last'] for v in a]):.1f}  vs  %{np.mean([v['last'] for v in c]):.1f}")
        P(f"Son {W} bölüm ort. getiri:  {np.mean([v['ret_last'] for v in a]):.1f}  vs  {np.mean([v['ret_last'] for v in c]):.1f}")
    json.dump(res, open(os.path.join(root, "ozet.json"), "w"), indent=1)
    open(os.path.join(root, "ozet.txt"), "w", encoding="utf-8").write("\n".join(lines))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=15000); ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--out", default="runs"); ap.add_argument("--quick", action="store_true")
    ap.add_argument("--ent", type=float, default=0.02); ap.add_argument("--live", action="store_true")
    ap.add_argument("--lr", type=float, default=1e-2); ap.add_argument("--pen", type=float, default=0.1)
    ap.add_argument("--anneal-frac", type=float, default=0.3); ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--live-every", type=int, default=100); ap.add_argument("--workers", type=int, default=2)
    A = ap.parse_args()
    if A.quick: A.episodes, A.out = 300, "runs_quick"
    from policy import PolicyNet, count_params
    print(f"Politika ağı parametre sayısı: {count_params(PolicyNet())} (8*16+16 + 16*4+4)")
    cfgs = [Config(name=n, anneal=an, episodes=A.episodes, seed=s, ent_start=A.ent, out_dir=A.out, lr_policy=A.lr, time_penalty=A.pen, anneal_frac=A.anneal_frac, batch_episodes=A.batch,
                   ckpt_every=max(50, A.episodes // 8)) for s in A.seeds for n, an in (("anneal", True), ("control", False))]
    if A.live:
        for c in cfgs: run_live(c, A.live_every)
    else:
        with mp.get_context("spawn").Pool(A.workers) as pool: pool.map(run_one, cfgs, chunksize=1)
    summarize(A.out, A.episodes)
    import visualize
    print("grafik:", visualize.compare(A.out, os.path.join(A.out, "karsilastirma.png")))
