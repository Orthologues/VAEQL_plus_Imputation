## PROMPT
```text
If I have a z-score scaled dataset, and I amputated some values artificially,
and I fill in these values with 0 for the real-valued or count features, 1
for the positive real-valued features, and 1/abs(N_categories) for the one-
hot encoded ordinal/categorical features, and I wanna train a VAE+Q Learning
imputation model. In the cross validation pre-training phase of the VAE, I
wanna freeze the weights associated with these amputated values. How should I
do it? Give me an example code snippet in pytorch
```


## COMMAND
```bash
codex exec --skip-git-repo-check -o post_amputation_weight_freezing_EXAMPLE.py   resume 019bfd88-5043-7432-89e6-9cddaa28a4e5 -   < VAE_prompt.txt
```


## ANSWER
Great question. The key point is: you can’t freeze weights per missing *cell* directly (weights are shared), but you can make missing cells contribute zero gradient by masking input and loss.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- fill vector ----------
def make_fill_vector(D, idx_real, idx_count, idx_pos, onehot_groups, device):
    fill = torch.zeros(D, device=device)
    fill[idx_real] = 0.0
    fill[idx_count] = 0.0
    fill[idx_pos] = 1.0
    for g in onehot_groups:          # g = list of column idx for one categorical variable
        fill[g] = 1.0 / len(g)
    return fill

# ---------- VAE ----------
class MaskedVAE(nn.Module):
    def __init__(self, D, H=256, Z=32):
        super().__init__()
        self.enc1 = nn.Linear(2 * D, H)  # [masked_x, missing_mask]
        self.enc2 = nn.Linear(H, H)
        self.mu = nn.Linear(H, Z)
        self.logvar = nn.Linear(H, Z)

        self.dec1 = nn.Linear(Z, H)
        self.dec2 = nn.Linear(H, H)
        self.out = nn.Linear(H, D)

    def forward(self, x_masked, miss_mask):
        h = torch.cat([x_masked, miss_mask], dim=1)
        h = F.relu(self.enc1(h))
        h = F.relu(self.enc2(h))
        mu, logvar = self.mu(h), self.logvar(h)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)
        h = F.relu(self.dec1(z))
        h = F.relu(self.dec2(h))
        x_hat = self.out(h)
        return x_hat, mu, logvar

def kl_div(mu, logvar):
    return -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))

# ---------- one train step ----------
def train_step(model, opt, x_true, amp_mask, fill_vec, beta=1e-3):
    """
    x_true:   [B, D] complete target (before amputation), already z-scored
    amp_mask: [B, D] 1 = amputated/missing, 0 = observed
    fill_vec: [D] feature-wise fill constants
    """
    obs_mask = 1.0 - amp_mask

    # Corrupt input with your fill rules
    x_corrupt = obs_mask * x_true + amp_mask * fill_vec.unsqueeze(0)

    # IMPORTANT: zero-out missing positions before encoder -> no gradient from amputated cells
    x_for_encoder = x_corrupt * obs_mask

    x_hat, mu, logvar = model(x_for_encoder, amp_mask)

    # Pre-train imputation: reconstruction only on amputated cells
    rec = ((x_hat - x_true) ** 2 * amp_mask).sum() / amp_mask.sum().clamp_min(1.0)
    loss = rec + beta * kl_div(mu, logvar)

    opt.zero_grad()
    loss.backward()
    opt.step()
    return loss.item(), rec.item()

# ---------------- example usage ----------------
device = "cuda" if torch.cuda.is_available() else "cpu"
B, D = 64, 200
x_true = torch.randn(B, D, device=device)             # z-scored data
amp_mask = (torch.rand(B, D, device=device) < 0.2).float()

idx_real = list(range(0, 80))
idx_count = list(range(80, 120))
idx_pos = list(range(120, 150))
onehot_groups = [list(range(150, 160)), list(range(160, 175)), list(range(175, 200))]

fill_vec = make_fill_vector(D, idx_real, idx_count, idx_pos, onehot_groups, device)
model = MaskedVAE(D).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

loss, rec = train_step(model, opt, x_true, amp_mask, fill_vec)
print(loss, rec)
```

If you want, I can give the same snippet with mixed likelihoods (Gaussian for real, Poisson/NB for counts, categorical cross-entropy for one-hot) instead of plain MSE.
