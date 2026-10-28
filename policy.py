"""Politika ağı (8 -> 16 tanh -> 4) ve baseline için küçük değer ağı."""
import numpy as np
import torch
import torch.nn as nn

OBS_DIM, HIDDEN, N_ACT = 8, 16, 4
INPUT_NAMES = ["x", "y", "vx", "vy", "açı", "açısal hız", "bacak 1", "bacak 2"]
ACTION_NAMES = ["boşta", "sol motor", "ana motor", "sağ motor"]
# Ağ diyagramında girdileri [-1, 1] aralığına oturtmak için (yalnızca görselleştirme)
INPUT_SCALE = np.array([1.0, 1.0, 2.0, 2.0, 1.0, 2.0, 1.0, 1.0])


class PolicyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(OBS_DIM, HIDDEN)
        self.fc2 = nn.Linear(HIDDEN, N_ACT)
        # çıkış katmanı küçük başlatılır -> başlangıç politikası ~uniform (yüksek entropi)
        nn.init.orthogonal_(self.fc1.weight, gain=1.0)
        nn.init.zeros_(self.fc1.bias)
        nn.init.orthogonal_(self.fc2.weight, gain=0.1)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        return self.fc2(torch.tanh(self.fc1(x)))  # aksiyon logit'leri

    def numpy_weights(self):
        return {k: v.detach().cpu().numpy().astype(np.float64) for k, v in
                dict(W1=self.fc1.weight, b1=self.fc1.bias, W2=self.fc2.weight, b2=self.fc2.bias).items()}


class ValueNet(nn.Module):
    """Varyans azaltma için ayrı küçük value (baseline) ağı: 8 -> 32 tanh -> 1."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(OBS_DIM, 32), nn.Tanh(), nn.Linear(32, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def forward_np(w, obs):
    """Hızlı numpy ileri geçiş -> (gizli aktivasyonlar, logit'ler, olasılıklar)."""
    h = np.tanh(w["W1"] @ obs + w["b1"])
    z = w["W2"] @ h + w["b2"]
    e = np.exp(z - z.max())
    return h, z, e / e.sum()


def count_params(m):
    return sum(p.numel() for p in m.parameters())


if __name__ == "__main__":
    p = PolicyNet()
    n_w = p.fc1.weight.numel() + p.fc2.weight.numel()
    n_b = p.fc1.bias.numel() + p.fc2.bias.numel()
    print(f"fc1: {OBS_DIM}x{HIDDEN} = {p.fc1.weight.numel()} ağırlık + {HIDDEN} bias")
    print(f"fc2: {HIDDEN}x{N_ACT} = {p.fc2.weight.numel()} ağırlık + {N_ACT} bias")
    print(f"Toplam: {n_w} ağırlık + {n_b} bias = {count_params(p)} parametre")
    assert count_params(p) == 212
    x = np.random.randn(OBS_DIM)
    _, z, pr = forward_np(p.numpy_weights(), x)
    zt = p(torch.tensor(x, dtype=torch.float32)).detach().numpy()
    assert np.allclose(z, zt, atol=1e-5) and abs(pr.sum() - 1) < 1e-9
    print("numpy ve torch ileri geçişleri eşleşiyor. OK (212 parametre)")
    print(f"Value ağı (baseline, ayrı): {count_params(ValueNet())} parametre")
