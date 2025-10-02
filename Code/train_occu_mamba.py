# ===============================================================
# OCC_Mamba (no calendar claims):
# - Dual path: Mamba-lite SSM + masked attention with gated fusion
# - Channel squeeze-excitation
# - Huber loss + early stopping
# - Train-only MinMax scaling, 80/20 split
# - NOW: Full-dataset inference + shared "Full_Results" sink
# ===============================================================

import os, re, glob, json, math
from pathlib import Path
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_squared_error as mse

# ============================================================================
# Paths (aligned with preprocess.py & run_all.py)
# ============================================================================
PROJ_ROOT = Path(os.environ.get("MAMBA_PROJ_ROOT", Path(__file__).resolve().parent))
BASE_DIR  = (PROJ_ROOT / "Aggregated_Cleaned").resolve()
BASE_DIR.mkdir(parents=True, exist_ok=True)

def make_out_dirs(model_name: str):
    out_dir  = (BASE_DIR / f"{model_name}_Results").resolve()
    pred_dir = out_dir / "preds"
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)
    return out_dir, pred_dir

# Selected, windowized datasets written by preprocess.py (lowercase 'room_' like your template)
PATTERN = str(BASE_DIR / "Xy_seq_flat_selected_room_*.csv")

# Outputs for this script
OUT_DIR, PRED_DIR = make_out_dirs("OCC_Mamba")

# Central folder to collect full predictions from ALL models
FULL_RESULTS_DIR = (PROJ_ROOT / "Full_Results").resolve()
FULL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
# ============================================================================

TARGET_COL = "occupant_num"

# ---------------- Train config ----------------
SEED            = 42
TEST_SIZE       = 0.2
BATCH_SIZE      = 64
EPOCHS          = 60
LR              = 1e-3
WEIGHT_DECAY    = 1e-5
EARLY_PATIENCE  = 12
MIN_DELTA_RMSE  = 1e-4
HUBER_DELTA     = 1.0       # robust loss
USE_HETEROSCEDASTIC = False # set True to predict mean+logvar
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(SEED); np.random.seed(SEED)

# ---------------- Manifest-aware helpers for [T,F] ----------------
def try_load_manifest(fpath_csv: str):
    """Return (T, bases) from manifest_<room>.json if it exists; else (None, None)."""
    room_tag = Path(fpath_csv).stem.replace("Xy_seq_flat_selected_", "")
    mpath = BASE_DIR / f"manifest_{room_tag}.json"
    if mpath.exists():
        with open(mpath, "r") as f:
            j = json.load(f)
        T = int(j.get("T", 0)) or None
        bases = j.get("bases", None)
        if bases and T:
            return T, list(bases)
    return None, None

def infer_T(columns):
    ks = []
    for c in columns:
        m = re.search(r"_t(\d+)$", str(c))
        if m: ks.append(int(m.group(1)))
    return max(ks) if ks else 1

def build_cols_from_bases(bases, T, df_cols):
    """Create ordered [f_t1..f_tT] per base; require all columns to exist."""
    ordered = []
    exist = set(df_cols)
    for t in range(1, T+1):
        for b in bases:
            name = f"{b}_t{t}"
            if name not in exist:
                return [], 0
            ordered.append(name)
    return ordered, len(bases)

def split_by_t(columns, T):
    by_t, map_tf = {t: set() for t in range(1, T+1)}, {}
    for c in columns:
        m = re.search(r"(.*)_t(\d+)$", str(c))
        if not m: continue
        base, t = m.group(1), int(m.group(2))
        if 1 <= t <= T:
            by_t[t].add(base); map_tf[(t, base)] = c
    return by_t, map_tf

def build_order_fallback(df_cols, T):
    feats = [c for c in df_cols if c != TARGET_COL]
    by_t, map_tf = split_by_t(feats, T)
    if not by_t or any(len(s)==0 for s in by_t.values()):
        return [], 0
    common = sorted(set.intersection(*by_t.values()))
    if not common: return [], 0
    ordered = []
    for t in range(1, T+1):
        for b in common: ordered.append(map_tf[(t, b)])
    return ordered, len(common)

# ---------------- Mamba-lite pieces (CPU friendly) ----------------
class CausalDepthwiseConv1d(nn.Module):
    def __init__(self, channels, k=7):
        super().__init__()
        self.pad = k - 1
        self.conv = nn.Conv1d(channels, channels, k, groups=channels, bias=True)
    def forward(self, x):  # [B,T,C]
        x = x.transpose(1,2)
        x = nn.functional.pad(x, (self.pad, 0))
        x = self.conv(x)
        return x.transpose(1,2)

class SimpleSSM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.log_decay = nn.Parameter(torch.randn(channels))
        self.mix = nn.Linear(channels, channels)
    def forward(self, x):
        B,T,C = x.shape
        decay = torch.sigmoid(self.log_decay)
        t = torch.arange(T, device=x.device).float()
        k = torch.pow(decay.unsqueeze(0), t.unsqueeze(1))   # [T,C]
        x_d = x.transpose(1,2)                               # [B,C,T]
        k_d = k.transpose(0,1).unsqueeze(1)                  # [C,1,T]
        y = nn.functional.conv1d(nn.functional.pad(x_d, (T-1,0)), k_d, groups=C)
        return self.mix(y.transpose(1,2))

class MambaLiteBlock(nn.Module):
    def __init__(self, d_model, d_ff=2, k=7, dropout=0.1):
        super().__init__()
        hidden = d_model * d_ff
        self.in_proj = nn.Linear(d_model, hidden * 2)
        self.dw = CausalDepthwiseConv1d(hidden, k)
        self.ssm = SimpleSSM(hidden)
        self.out_proj = nn.Linear(hidden, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
    def forward(self, x):
        res = x
        x = self.norm(x)
        u = self.in_proj(x)
        g, v = u.chunk(2, dim=-1)
        g = torch.sigmoid(g)
        v = self.dw(v) + self.ssm(v)
        y = self.out_proj(g * v)
        return res + self.drop(y)

# ---------------- Attention + Gated Fusion + SE ----------------
class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, nhead=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ln = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
    def forward(self, x):
        T = x.size(1)
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        y, _ = self.attn(self.ln(x), self.ln(x), self.ln(x), attn_mask=mask)
        return x + self.drop(y)

class SqueezeExcite(nn.Module):
    def __init__(self, d_model, r=8):
        super().__init__()
        self.se = nn.Sequential(
            nn.Linear(d_model, d_model//r),
            nn.ReLU(),
            nn.Linear(d_model//r, d_model),
            nn.Sigmoid()
        )
    def forward(self, x):        # [B,T,D]
        w = self.se(x.mean(dim=1))   # [B,D]
        return x * w.unsqueeze(1)

class OccuMambaLayer(nn.Module):
    def __init__(self, d_model, nhead=4, dropout=0.1):
        super().__init__()
        self.mamba = MambaLiteBlock(d_model, d_ff=2, k=7, dropout=dropout)
        self.attn  = CausalSelfAttention(d_model, nhead=nhead, dropout=dropout)
        self.fuse_gate = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )
        self.se = SqueezeExcite(d_model)
    def forward(self, x):          # [B,T,D]
        a = self.attn(x)
        m = self.mamba(x)
        gate = self.fuse_gate(x)   # [B,T,D]
        y = gate * m + (1 - gate) * a
        y = self.se(y)
        return y

class OccuMamba(nn.Module):
    """
    No calendar/time encodings here; preprocessing already removed non-office periods.
    """
    def __init__(self, in_feats, d_model=128, depth=3, nhead=4, dropout=0.10, heterosced=False):
        super().__init__()
        self.heterosced = heterosced
        self.in_proj = nn.Linear(in_feats, d_model)
        self.layers = nn.ModuleList([OccuMambaLayer(d_model, nhead=nhead, dropout=dropout)
                                     for _ in range(depth)])
        self.norm = nn.LayerNorm(d_model)
        out_dim = 2 if heterosced else 1
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model//2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model//2, out_dim)
        )
    def forward(self, x):  # x:[B,T,F]
        x = self.in_proj(x)
        for lyr in self.layers:
            x = lyr(x)
        x = self.norm(x)
        x = x[:, -1, :]   # last step
        out = self.head(x)
        if self.heterosced:
            mean, logvar = out.chunk(2, dim=-1)
            return mean, logvar
        return out

# ---------------- Train/Eval ----------------
class HuberLoss1D(nn.Module):
    def __init__(self, delta=1.0):
        super().__init__()
        self.delta = delta
    def forward(self, pred, target):
        err = pred - target
        abs_e = torch.abs(err)
        quad = torch.minimum(abs_e, torch.tensor(self.delta, device=err.device))
        lin  = abs_e - quad
        return torch.mean(0.5 * quad**2 + self.delta * lin)

def train_one(model, train_loader, val_loader, epochs=EPOCHS):
    model.to(DEVICE)
    if USE_HETEROSCEDASTIC:
        def nll(mean, logvar, y):
            inv = torch.exp(-logvar)
            return torch.mean(0.5 * (inv * (y-mean)**2 + logvar))
        loss_fn = nll
    else:
        loss_fn = HuberLoss1D(delta=HUBER_DELTA)

    opt  = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_rmse = float("inf")
    best = None
    patience = 0

    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            if USE_HETEROSCEDASTIC:
                mean, logvar = model(xb)
                loss = loss_fn(mean, logvar, yb)
            else:
                pred = model(xb)
                loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()

        # ---- quick val
        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                if USE_HETEROSCEDASTIC:
                    mean, _ = model(xb); pred = mean
                else:
                    pred = model(xb)
                pred = pred.squeeze(-1).cpu().numpy()
                y_pred.extend(pred.tolist())
                y_true.extend(yb.squeeze(-1).cpu().numpy().tolist())
        rmse = math.sqrt(mse(y_true, y_pred))
        if best_rmse - rmse > MIN_DELTA_RMSE:
            best_rmse = rmse
            best = {k: v.cpu() for k,v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= EARLY_PATIENCE:
                break

    if best is not None:
        model.load_state_dict(best)
    return model

def evaluate(model, loader):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            if USE_HETEROSCEDASTIC:
                mean, _ = model(xb); pred = mean
            else:
                pred = model(xb)
            pred = pred.squeeze(-1).cpu().numpy()
            y_pred.extend(pred.tolist())
            y_true.extend(yb.squeeze(-1).cpu().numpy().tolist())
    y_true = np.array(y_true); y_pred = np.array(y_pred)
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mse(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    return mae, rmse, r2, y_true, y_pred

def save_preds(room, model_name, y_true, y_pred):
    dfp = pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "abs_error": np.abs(y_true - y_pred)})
    out = PRED_DIR / f"{room}_{model_name}_preds.csv"
    dfp.to_csv(out, index=False)
    return out

# ---- Batched full-dataset inference helper ----
def predict_full(model, X_full_tensor, batch_size=256):
    """Batched inference over the full dataset tensor; returns (N,) numpy."""
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, X_full_tensor.shape[0], batch_size):
            xb = X_full_tensor[i:i+batch_size].to(DEVICE)
            if USE_HETEROSCEDASTIC:
                mean, _ = model(xb)
                yb = mean.squeeze(-1).detach().cpu().numpy()
            else:
                yb = model(xb).squeeze(-1).detach().cpu().numpy()
            outs.append(yb)
    return np.concatenate(outs, axis=0)

# ---------------- Main ----------------
def main():
    files = sorted(glob.glob(PATTERN))
    if not files:
        raise SystemExit(f"No selected files found: {PATTERN}")

    results = []

    for fpath in files:
        room = Path(fpath).stem.replace("Xy_seq_flat_selected_", "")
        print(f"\n📂 Room: {room} | backend=occumamba")

        df = pd.read_csv(fpath)
        if TARGET_COL not in df.columns:
            df = df.rename(columns={df.columns[-1]: TARGET_COL})

        # Prefer manifest; fallback to name inference
        T_m, bases = try_load_manifest(fpath)
        if T_m is not None and bases:
            T = T_m
            ordered_cols, F_common = build_cols_from_bases(bases, T, df.columns)
            if F_common == 0:
                T = infer_T([c for c in df.columns if c != TARGET_COL]) or 1
                ordered_cols, F_common = build_order_fallback(df.columns, T)
        else:
            T = infer_T([c for c in df.columns if c != TARGET_COL]) or 1
            ordered_cols, F_common = build_order_fallback(df.columns, T)

        if F_common == 0 or T == 1:
            print("  ⚠️ Need a consistent lag grid (T>1 & common bases). Skipping this room.")
            continue

        X_flat = df[ordered_cols].astype(np.float32).values
        y = df[TARGET_COL].astype(np.float32).values.reshape(-1, 1)

        # split & scale (fit on train only)
        Xtr, Xte, ytr, yte = train_test_split(X_flat, y, test_size=TEST_SIZE, random_state=SEED)
        scaler = MinMaxScaler()
        Xtr = scaler.fit_transform(Xtr)
        Xte = scaler.transform(Xte)

        # [N,T,F]
        Ntr, Nte = Xtr.shape[0], Xte.shape[0]
        Xtr_seq = torch.tensor(Xtr.reshape(Ntr, T, F_common), dtype=torch.float32)
        Xte_seq = torch.tensor(Xte.reshape(Nte, T, F_common), dtype=torch.float32)
        ytr_t   = torch.tensor(ytr, dtype=torch.float32)
        yte_t   = torch.tensor(yte, dtype=torch.float32)

        train_loader = DataLoader(TensorDataset(Xtr_seq, ytr_t), batch_size=BATCH_SIZE, shuffle=True)
        val_loader   = DataLoader(TensorDataset(Xte_seq, yte_t),  batch_size=BATCH_SIZE)

        model = OccuMamba(in_feats=F_common, d_model=128, depth=3, nhead=4, dropout=0.10,
                          heterosced=USE_HETEROSCEDASTIC)
        model = train_one(model, train_loader, val_loader)
        mae, rmse, r2, y_true, y_pred = evaluate(model, val_loader)
        ppath = save_preds(room, "OCC_Mamba", y_true, y_pred)

        results.append([room, "OCC_Mamba", T, F_common, mae, rmse, r2, str(ppath)])
        print(f"  ✅ OCC_Mamba | T={T} F={F_common} | MAE={mae:.4f} RMSE={rmse:.4f} R²={r2:.4f} | preds: {ppath}")

        # ---- FULL-DATASET INFERENCE & SAVE (local + shared) ----
        X_full_flat = scaler.transform(X_flat)  # apply train-fitted scaler to full matrix
        Nfull = X_full_flat.shape[0]
        X_full_seq = torch.tensor(X_full_flat.reshape(Nfull, T, F_common), dtype=torch.float32)
        y_pred_full = predict_full(model, X_full_seq)

        # 1) Local copy under this model's preds/
        local_full = PRED_DIR / f"{room}_OCC_Mamba_preds_full.csv"
        pd.DataFrame({"y_pred": y_pred_full}).to_csv(local_full, index=False)

        # 2) Canonical shared file for post-processing
        shared_full = FULL_RESULTS_DIR / f"{room}__OCC_Mamba__full.csv"
        pd.DataFrame({
            "room": room,
            "model": "OCC_Mamba",
            "y_pred": y_pred_full,
            "T": T,
            "F": F_common
        }).to_csv(shared_full, index=False)

        print(f"  ↳ saved FULL predictions (local):  {local_full}")
        print(f"  ↳ saved FULL predictions (shared): {shared_full}")

    if results:
        out_csv = OUT_DIR / "occ_mamba_results.csv"
        pd.DataFrame(results, columns=["Room","Model","T","F","MAE","RMSE","R2","preds_path"]).to_csv(out_csv, index=False)
        print(f"\n✅ Saved results to {out_csv}")
        print(f"📁 Prediction CSVs in: {PRED_DIR}")
        print(f"📦 Full-period predictions gathered in: {FULL_RESULTS_DIR}")
    else:
        print("\nℹ️ No rooms were trained (no consistent lag grid found).")

if __name__ == "__main__":
    main()
