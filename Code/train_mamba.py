import os, re, glob, math, json
from pathlib import Path
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error as mse

# ============================================================================
# Standard paths (shared with run_all.py & other scripts)
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

# Windowized datasets written by preprocess.py
PATTERN = str(BASE_DIR / "Xy_seq_flat_selected_room_*.csv")

# Outputs for this script
OUT_DIR, PRED_DIR = make_out_dirs("Mamba")

# Central place to collect full predictions from ALL models
FULL_RESULTS_DIR = (PROJ_ROOT / "Full_Results").resolve()
FULL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
# ============================================================================

TARGET_COL = "occupant_num"

# ---------------- Train config ----------------
SEED               = 42
TEST_SIZE          = 0.2
BATCH_SIZE         = 64
EPOCHS             = 60
LR                 = 1e-3
WEIGHT_DECAY       = 1e-5
EARLY_STOP_PATIENCE = 12
MIN_DELTA_RMSE     = 1e-4
DEVICE             = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- Try the official Mamba ----------------
USE_OFFICIAL = False
try:
    from mamba_ssm import Mamba  # state-spaces/mamba (CUDA strongly recommended)
    USE_OFFICIAL = True
except Exception:
    USE_OFFICIAL = False

# ---------------- Helpers for [T,F] ----------------
def try_load_manifest(fpath_csv: str):
    """If manifest_<room>.json exists, return (T, bases) else (None, None)."""
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
        if m:
            ks.append(int(m.group(1)))
    return max(ks) if ks else 1

def split_by_t(columns, T):
    base_set_by_t = {t: set() for t in range(1, T+1)}
    base_to_col_for_t = {}
    for c in columns:
        m = re.search(r"(.*)_t(\d+)$", str(c))
        if not m:
            continue
        base, t = m.group(1), int(m.group(2))
        if 1 <= t <= T:
            base_set_by_t[t].add(base)
            base_to_col_for_t[(t, base)] = c
    return base_set_by_t, base_to_col_for_t

def build_order_fallback(df_cols, T):
    """Intersection-of-bases fallback when manifest is missing/mismatched."""
    feat_cols = [c for c in df_cols if c != TARGET_COL]
    base_set_by_t, base_to_col_for_t = split_by_t(feat_cols, T)
    if not base_set_by_t:
        return [], 0
    common_bases = set.intersection(*base_set_by_t.values())
    if not common_bases:
        return [], 0
    common_bases = sorted(list(common_bases))
    ordered_cols = []
    for t in range(1, T+1):
        for base in common_bases:
            ordered_cols.append(base_to_col_for_t[(t, base)])
    return ordered_cols, len(common_bases)

def build_cols_from_bases(bases, T, df_cols):
    """Create ordered [base_t1..base_tT] per base, only if all exist; else []"""
    ordered = []
    exist = set(df_cols)
    for t in range(1, T+1):
        for b in bases:
            name = f"{b}_t{t}"
            if name not in exist:
                return [], 0
            ordered.append(name)
    return ordered, len(bases)

# ---------------- CPU-lite Mamba-ish fallback ----------------
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
        k = torch.pow(decay.unsqueeze(0), t.unsqueeze(1))      # [T,C]
        x_d = x.transpose(1,2)                                  # [B,C,T]
        k_d = k.transpose(0,1).unsqueeze(1)                     # [C,1,T]
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

# ---------------- Mamba regressor ----------------
class MambaRegressor(nn.Module):
    """
    Token proj -> L x (Mamba/MambaLite) -> Norm -> Pool(last) -> Head
    """
    def __init__(self, in_feats, d_model=96, n_layers=3, dropout=0.1, use_official=False):
        super().__init__()
        self.proj = nn.Linear(in_feats, d_model)
        blocks = []
        for _ in range(n_layers):
            if use_official:
                blocks.append(Mamba(d_model))   # official block
            else:
                blocks.append(MambaLiteBlock(d_model, d_ff=2, k=7, dropout=dropout))
        self.blocks = nn.ModuleList(blocks)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model//2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model//2, 1)
        )
    def forward(self, x):  # x: [B,T,F]
        x = self.proj(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        x = x[:, -1, :]        # last token pooling
        return self.head(x)

# ---------------- Inference helper (FULL predictions) ----------------
def predict_full(model, X_full_tensor, batch_size=256):
    """Batched inference over the full dataset tensor; returns (N,) numpy."""
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, X_full_tensor.shape[0], batch_size):
            xb = X_full_tensor[i:i+batch_size].to(DEVICE)
            yb = model(xb).squeeze(-1).detach().cpu().numpy()
            outs.append(yb)
    return np.concatenate(outs, axis=0)

# ---------------- Train/Eval ----------------
def train_one(model, train_loader, val_loader, epochs=EPOCHS):
    model.to(DEVICE)
    crit = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_rmse = float("inf")
    best = None
    no_improve = 0

    for ep in range(1, epochs+1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()

        # val
        model.eval()
        with torch.no_grad():
            y_true, y_pred = [], []
            for xb, yb in val_loader:
                xb = xb.to(DEVICE)
                pred = model(xb).squeeze(-1).cpu().numpy()
                y_pred.extend(pred.tolist())
                y_true.extend(yb.squeeze(-1).numpy().tolist())
        rmse = math.sqrt(mse(y_true, y_pred))
        if best_rmse - rmse > MIN_DELTA_RMSE:
            best_rmse = rmse
            best = {k: v.cpu() for k,v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= EARLY_STOP_PATIENCE:
                break

    if best is not None:
        model.load_state_dict(best)
    return model

def evaluate(model, loader):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(DEVICE)
            pred = model(xb).squeeze(-1).cpu().numpy()
            y_pred.extend(pred.tolist())
            y_true.extend(yb.squeeze(-1).numpy().tolist())
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

# ---------------- Main ----------------
def main():
    files = sorted(glob.glob(PATTERN))
    if not files:
        raise SystemExit(f"No selected files found: {PATTERN}")

    results = []

    for fpath in files:
        room = Path(fpath).stem.replace("Xy_seq_flat_selected_", "")
        print(f"\n📂 Room: {room} | backend={'official' if USE_OFFICIAL else 'lite'}")

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
            print("  ⚠️ Need lag grid to train Mamba (T>1 and common bases). Skipping.")
            continue

        X_flat = df[ordered_cols].astype(np.float32).values
        y = df[TARGET_COL].astype(np.float32).values.reshape(-1, 1)

        # split & scale (fit on train only)
        Xtr, Xte, ytr, yte = train_test_split(X_flat, y, test_size=TEST_SIZE, random_state=SEED)
        scaler = MinMaxScaler()
        Xtr = scaler.fit_transform(Xtr)
        Xte = scaler.transform(Xte)

        # to [T,F]
        Ntr, Nte = Xtr.shape[0], Xte.shape[0]
        Xtr_seq = torch.tensor(Xtr.reshape(Ntr, T, F_common), dtype=torch.float32)
        Xte_seq = torch.tensor(Xte.reshape(Nte, T, F_common), dtype=torch.float32)
        ytr_t = torch.tensor(ytr, dtype=torch.float32)
        yte_t = torch.tensor(yte, dtype=torch.float32)

        train_loader = DataLoader(TensorDataset(Xtr_seq, ytr_t), batch_size=BATCH_SIZE, shuffle=True)
        val_loader   = DataLoader(TensorDataset(Xte_seq, yte_t),  batch_size=BATCH_SIZE)

        # ---- Train & evaluate
        model = MambaRegressor(in_feats=F_common, d_model=96, n_layers=3, dropout=0.10, use_official=USE_OFFICIAL)
        model = train_one(model, train_loader, val_loader)
        mae, rmse, r2, y_true, y_pred = evaluate(model, val_loader)
        ppath = save_preds(room, "Mamba", y_true, y_pred)

        results.append([room, "Mamba", T, F_common, mae, rmse, r2, str(ppath)])
        print(f"  ✅ Mamba | T={T} F={F_common} | MAE={mae:.4f} RMSE={rmse:.4f} R²={r2:.4f}")

        # ---- NEW: Full-dataset inference for post-processing ----
        X_full_flat = scaler.transform(df[ordered_cols].astype(np.float32).values)
        Nfull = X_full_flat.shape[0]
        X_full_seq = torch.tensor(X_full_flat.reshape(Nfull, T, F_common), dtype=torch.float32)
        y_pred_full = predict_full(model, X_full_seq)

        # Save a single-column y_pred (plus optional metadata) into the shared folder
        full_df = pd.DataFrame({
            "room": room,
            "model": "Mamba",
            "y_pred": y_pred_full,
            "T": T,
            "F": F_common
        })
        full_path = FULL_RESULTS_DIR / f"{room}__Mamba__full.csv"
        full_df.to_csv(full_path, index=False)
        print(f"  ↳ saved FULL predictions: {full_path}")

    if results:
        out_csv = OUT_DIR / "mamba_results.csv"
        pd.DataFrame(results, columns=["Room","Model","T","F","MAE","RMSE","R2","preds_path"]).to_csv(out_csv, index=False)
        print(f"\n✅ Saved results to {out_csv}")
        print(f"📁 Prediction CSVs in: {PRED_DIR}")
        print(f"📦 Full-period predictions gathered in: {FULL_RESULTS_DIR}")
    else:
        print("\nℹ️ No rooms were trained (likely no consistent lag grid).")

if __name__ == "__main__":
    torch.manual_seed(SEED); np.random.seed(SEED)
    main()
