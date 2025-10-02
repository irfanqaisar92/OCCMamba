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

# Where the selected, windowized datasets are written by preprocess.py
PATTERN = str(BASE_DIR / "Xy_seq_flat_selected_room_*.csv")

# Outputs for this script
OUT_DIR, PRED_DIR = make_out_dirs("Baseline")

# ---- Central "Full_Results" folder used by all models (override via env) ----
FULL_RESULTS_DIR = Path(
    os.environ.get("MAMBA_FULL_RESULTS", PROJ_ROOT / "Full_Results")
).resolve()
FULL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def save_full_preds(room_tag: str,
                    model_name: str,
                    y_pred_full: np.ndarray,
                    extra: dict | None = None) -> Path:
    """
    Save full-period predictions (one per window) for post-processing.
    Filename pattern: <room>__<model>__full.csv
    """
    df = pd.DataFrame({"room": room_tag, "model": model_name, "y_pred": y_pred_full})
    if extra:
        for k, v in extra.items():
            df[k] = v
    out = FULL_RESULTS_DIR / f"{room_tag}__{model_name}__full.csv"
    df.to_csv(out, index=False)
    return out
# ============================================================================

TARGET_COL = "occupant_num"

# ---------------- Training config ----------------
SEED        = 42
TEST_SIZE   = 0.2
BATCH_SIZE  = 64
EPOCHS      = 60
LR          = 1e-3
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- Utilities ----------------
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
    return max(ks) if ks else None

def build_cols_from_bases(bases, T, df_cols):
    """Create ordered [base_t1..base_tT] per base, only keeping columns that exist."""
    ordered = []
    exist = set(df_cols)
    for t in range(1, T+1):
        for b in bases:
            name = f"{b}_t{t}"
            if name in exist:
                ordered.append(name)
            else:
                # If any required name is missing, abort to avoid bad reshapes.
                return [], 0
    return ordered, len(bases)

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

def build_consistent_column_order(df_cols, T):
    """Fallback if no manifest: intersection of bases across all t=1..T."""
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

# ---------------- Models ----------------
class MLP(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(128, 1),
        )
    def forward(self, x):
        return self.net(x)

class LSTMModel(nn.Module):
    def __init__(self, in_feats, hidden=64, num_layers=1, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(input_size=in_feats, hidden_size=hidden,
                            num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden, 1)
    def forward(self, x):          # x: [B, T, F]
        out, _ = self.lstm(x)      # [B, T, H]
        last = out[:, -1, :]       # [B, H]
        return self.fc(last)       # [B, 1]

class TinyTransformer(nn.Module):
    def __init__(self, in_feats, d_model=64, nhead=4, num_layers=2, dim_ff=128, dropout=0.1, pool="last"):
        super().__init__()
        self.proj = nn.Linear(in_feats, d_model)
        enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                         dim_feedforward=dim_ff, dropout=dropout,
                                         batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(enc, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.pool = pool
        self.head = nn.Linear(d_model, 1)
    def forward(self, x):           # x: [B, T, F]
        x = self.proj(x)
        x = self.encoder(x)
        x = self.norm(x)
        x = x.mean(dim=1) if self.pool == "mean" else x[:, -1, :]
        return self.head(x)

# ---------------- Train/Eval helpers ----------------
def train_and_predict(model, train_loader, test_loader):
    model.to(DEVICE)
    crit = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    for _ in range(EPOCHS):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            pred = model(xb)
            loss = crit(pred, yb)
            loss.backward()
            opt.stop_gradient = False
            opt.step()

    # Eval & collect predictions
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(DEVICE)
            pred = model(xb).squeeze(-1).detach().cpu().numpy()
            y_pred.extend(pred.tolist())
            y_true.extend(yb.squeeze(-1).numpy().tolist())

    mae  = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mse(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    return mae, rmse, r2, np.array(y_true), np.array(y_pred)

def save_preds(room_tag, model_name, y_true, y_pred):
    dfp = pd.DataFrame({
        "y_true": y_true,
        "y_pred": y_pred,
        "abs_error": np.abs(y_true - y_pred)
    })
    out_path = PRED_DIR / f"{room_tag}_{model_name}_preds.csv"
    dfp.to_csv(out_path, index=False)
    return out_path

def predict_full(model: nn.Module, X_full_tensor: torch.Tensor, batch_size: int = 256) -> np.ndarray:
    """Batched inference over the full dataset tensor; returns (N,) numpy."""
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, X_full_tensor.shape[0], batch_size):
            xb = X_full_tensor[i:i+batch_size].to(DEVICE)
            yb = model(xb).squeeze(-1).detach().cpu().numpy()
            outs.append(yb)
    return np.concatenate(outs, axis=0)

# ---------------- Main loop ----------------
def main():
    files = sorted(glob.glob(PATTERN))
    if not files:
        raise SystemExit(f"No files matched: {PATTERN}")

    all_results = []

    for fpath in files:
        room_tag = Path(fpath).stem.replace("Xy_seq_flat_selected_", "")
        print(f"\n📂 Room: {room_tag}")

        # Load
        df = pd.read_csv(fpath)
        if TARGET_COL not in df.columns:
            df = df.rename(columns={df.columns[-1]: TARGET_COL})

        # Prefer manifest (exact T and bases); fallback to name inference
        T_m, bases = try_load_manifest(fpath)
        if T_m is not None and bases:
            T = T_m
            ordered_cols, F_common = build_cols_from_bases(bases, T, df.columns)
            if F_common == 0:
                T = infer_T([c for c in df.columns if c != TARGET_COL]) or 1
                ordered_cols, F_common = build_consistent_column_order(df.columns, T)
        else:
            T = infer_T([c for c in df.columns if c != TARGET_COL]) or 1
            ordered_cols, F_common = build_consistent_column_order(df.columns, T)

        # If we still can't form a consistent lag grid, do flat MLP only
        if F_common == 0 or T == 1:
            X_flat = df.drop(columns=[TARGET_COL]).astype(np.float32).values
            y = df[TARGET_COL].astype(np.float32).values.reshape(-1, 1)

            X_train, X_test, y_train, y_test = train_test_split(
                X_flat, y, test_size=TEST_SIZE, random_state=SEED
            )
            scaler = MinMaxScaler()
            X_train = scaler.fit_transform(X_train)
            X_test  = scaler.transform(X_test)

            Xtr = torch.tensor(X_train, dtype=torch.float32)
            Xte = torch.tensor(X_test,  dtype=torch.float32)
            ytr = torch.tensor(y_train, dtype=torch.float32)
            yte = torch.tensor(y_test,  dtype=torch.float32)

            tl = DataLoader(TensorDataset(Xtr, ytr), batch_size=BATCH_SIZE, shuffle=True)
            vl = DataLoader(TensorDataset(Xte, yte), batch_size=BATCH_SIZE)

            mlp = MLP(in_dim=Xtr.shape[1])
            mae, rmse, r2, y_true, y_pred = train_and_predict(mlp, tl, vl)
            all_results.append([room_tag, "MLP", mae, rmse, r2])
            ppath = save_preds(room_tag, "MLP", y_true, y_pred)
            print(f"  ✅ MLP         | MAE={mae:.4f} RMSE={rmse:.4f} R²={r2:.4f}  | preds: {ppath}")
            print("  ℹ️ Skipping LSTM/Transformer (no consistent lag grid).")

            # ---- FULL inference (flat) -> Full_Results ----
            X_full_flat = scaler.transform(df.drop(columns=[TARGET_COL]).astype(np.float32).values)
            X_full_t = torch.tensor(X_full_flat, dtype=torch.float32)
            y_pred_full = predict_full(mlp, X_full_t)
            full_path = save_full_preds(room_tag, "MLP", y_pred_full, extra=None)
            print(f"  ↳ saved FULL predictions to: {full_path}")
            continue

        # With consistent [T,F]
        X_flat = df[ordered_cols].astype(np.float32).values
        y = df[TARGET_COL].astype(np.float32).values.reshape(-1, 1)

        X_train_flat, X_test_flat, y_train, y_test = train_test_split(
            X_flat, y, test_size=TEST_SIZE, random_state=SEED
        )

        scaler = MinMaxScaler()
        X_train_flat = scaler.fit_transform(X_train_flat)
        X_test_flat  = scaler.transform(X_test_flat)

        # Flat for MLP
        Xtr_mlp = torch.tensor(X_train_flat, dtype=torch.float32)
        Xte_mlp = torch.tensor(X_test_flat,  dtype=torch.float32)

        # Sequence tensors for LSTM/Transformer
        Ntr, Nte = X_train_flat.shape[0], X_test_flat.shape[0]
        Xtr_seq = torch.tensor(X_train_flat.reshape(Ntr, T, F_common), dtype=torch.float32)
        Xte_seq = torch.tensor(X_test_flat.reshape(Nte, T, F_common), dtype=torch.float32)

        ytr = torch.tensor(y_train, dtype=torch.float32)
        yte = torch.tensor(y_test,  dtype=torch.float32)

        # Loaders
        tl_mlp = DataLoader(TensorDataset(Xtr_mlp, ytr), batch_size=BATCH_SIZE, shuffle=True)
        vl_mlp = DataLoader(TensorDataset(Xte_mlp, yte), batch_size=BATCH_SIZE)

        tl_seq = DataLoader(TensorDataset(Xtr_seq, ytr), batch_size=BATCH_SIZE, shuffle=True)
        vl_seq = DataLoader(TensorDataset(Xte_seq, yte), batch_size=BATCH_SIZE)

        # MLP
        mlp = MLP(in_dim=Xtr_mlp.shape[1])
        mae, rmse, r2, y_true, y_pred = train_and_predict(mlp, tl_mlp, vl_mlp)
        all_results.append([room_tag, "MLP", mae, rmse, r2])
        ppath = save_preds(room_tag, "MLP", y_true, y_pred)
        print(f"  ✅ MLP         | MAE={mae:.4f} RMSE={rmse:.4f} R²={r2:.4f}  | preds: {ppath}")

        # ---- FULL inference (flat) -> Full_Results
        X_full_flat = scaler.transform(df[ordered_cols].astype(np.float32).values)
        X_full_t = torch.tensor(X_full_flat, dtype=torch.float32)
        y_pred_full = predict_full(mlp, X_full_t)
        full_path = save_full_preds(room_tag, "MLP", y_pred_full, extra={"T": T, "F": F_common})
        print(f"  ↳ saved FULL predictions to: {full_path}")

        # LSTM
        lstm = LSTMModel(in_feats=F_common, hidden=64, num_layers=1, dropout=0.0)
        mae, rmse, r2, y_true, y_pred = train_and_predict(lstm, tl_seq, vl_seq)
        all_results.append([room_tag, "LSTM", mae, rmse, r2])
        ppath = save_preds(room_tag, "LSTM", y_true, y_pred)
        print(f"  ✅ LSTM        | MAE={mae:.4f} RMSE={rmse:.4f} R²={r2:.4f}  | preds: {ppath}")

        # ---- FULL inference (sequence) -> Full_Results
        Nfull = X_full_flat.shape[0]
        X_full_seq = torch.tensor(X_full_flat.reshape(Nfull, T, F_common), dtype=torch.float32)
        y_pred_full = predict_full(lstm, X_full_seq)
        full_path = save_full_preds(room_tag, "LSTM", y_pred_full, extra={"T": T, "F": F_common})
        print(f"  ↳ saved FULL predictions to: {full_path}")

        # Transformer
        trans = TinyTransformer(in_feats=F_common, d_model=64, nhead=4,
                                num_layers=2, dim_ff=128, dropout=0.1, pool="last")
        mae, rmse, r2, y_true, y_pred = train_and_predict(trans, tl_seq, vl_seq)
        all_results.append([room_tag, "Transformer", mae, rmse, r2])
        ppath = save_preds(room_tag, "Transformer", y_true, y_pred)
        print(f"  ✅ Transformer | MAE={mae:.4f} RMSE={rmse:.4f} R²={r2:.4f}  | preds: {ppath}")

        # ---- FULL inference (sequence) -> Full_Results
        y_pred_full = predict_full(trans, X_full_seq)
        full_path = save_full_preds(room_tag, "Transformer", y_pred_full, extra={"T": T, "F": F_common})
        print(f"  ↳ saved FULL predictions to: {full_path}")

    # Save aggregate results
    results_df = pd.DataFrame(all_results, columns=["Room", "Model", "MAE", "RMSE", "R2"])
    out_csv = OUT_DIR / "baseline_results.csv"
    results_df.to_csv(out_csv, index=False)
    print(f"\n✅ All results saved to {out_csv}")
    print(f"📁 Test-split prediction CSVs in: {PRED_DIR}")
    print(f"📁 Full-period predictions in:   {FULL_RESULTS_DIR}")

if __name__ == "__main__":
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    main()
