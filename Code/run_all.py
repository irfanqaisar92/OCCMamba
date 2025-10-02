# run_all.py — robust orchestrator (env-pinned project root)

import os
import sys
import time
import subprocess
from pathlib import Path
import pandas as pd

# ---------- Project root & dirs ----------
THIS_FILE = Path(__file__).resolve()
PROJ_ROOT = THIS_FILE.parent                              # where run_all.py lives
SCRIPT_DIRS = [PROJ_ROOT]                                 # look for scripts here
AGG_DIR   = PROJ_ROOT / "Aggregated_Cleaned"
ALL_DIR   = AGG_DIR / "All_Results"
ALL_DIR.mkdir(parents=True, exist_ok=True)

# ---------- Scripts to run in order ----------
SCRIPT_NAMES = [
    "preprocess.py",
    "baseline_models.py",
    "train_mamba.py",
    "train_mambaformer.py",
    "train_moe_mamba.py",
    "train_occu_mamba.py",
]

# ---------- Expected results CSVs (with tolerant fallbacks) ----------
def _first_existing(paths):
    for p in paths:
        if p.exists():
            return p
    return paths[0]  # return primary even if missing; we'll warn later

RESULT_FILES = {
    "Baseline":    _first_existing([AGG_DIR / "Baseline_Results"   / "baseline_results.csv"]),
    "Mamba":       _first_existing([AGG_DIR / "Mamba_Results"      / "mamba_results.csv"]),
    "MambaFormer": _first_existing([AGG_DIR / "MambaFormer_Results"/ "mambaformer_results.csv"]),
    "MoE_Mamba":   _first_existing([AGG_DIR / "MoE_Mamba_Results"  / "moe_mamba_results.csv"]),
    # handle both lowercase/uppercase variants transparently
    "OccuMamba":   _first_existing([
                        AGG_DIR / "OccuMamba_Results" / "occu_mamba_results.csv",
                        AGG_DIR / "OccuMamba_Results" / "OCC_mamba_results.csv",
                    ]),
}

# ---------- Post-processing ----------
POSTPROCESS_SCRIPT = PROJ_ROOT / "postprocess_schedules.py"
# default thresholds / clipping (override here if you like)
PP_ARGS = dict(
    hvac_thr = "0.2",
    occ_thr1 = "0.2",
    occ_thr2 = "0.8",
    clip_min = "0",
    clip_max = "1",
)

def find_script(name: str) -> Path | None:
    for d in SCRIPT_DIRS:
        p = d / name
        if p.exists():
            return p
    return None

def run_script(name: str) -> int:
    p = find_script(name)
    if p is None:
        print(f"⚠️  Skipping {name} (not found in {SCRIPT_DIRS})")
        return 0  # don't block the rest

    env = os.environ.copy()
    env["MAMBA_PROJ_ROOT"] = str(PROJ_ROOT)

    print(f"\n▶️  Running: {p}")
    t0 = time.time()
    proc = subprocess.run([sys.executable, str(p)], cwd=str(p.parent), env=env)
    dt = time.time() - t0

    if proc.returncode == 0:
        print(f"✅ Finished: {name}  ({dt:.1f}s)")
    else:
        print(f"❌ Failed:   {name} (exit={proc.returncode}, {dt:.1f}s)")
    return proc.returncode

def run_postprocess(model_name: str) -> int:
    """Invoke postprocess_schedules.py for a given family if the script exists."""
    if not POSTPROCESS_SCRIPT.exists():
        print(f"ℹ️  Post-processor not found: {POSTPROCESS_SCRIPT} (skipping)")
        return 0
    env = os.environ.copy()
    env["MAMBA_PROJ_ROOT"] = str(PROJ_ROOT)
    cmd = [
        sys.executable, str(POSTPROCESS_SCRIPT),
        "--model", model_name,
        "--hvac_thr", PP_ARGS["hvac_thr"],
        "--occ_thr1", PP_ARGS["occ_thr1"],
        "--occ_thr2", PP_ARGS["occ_thr2"],
        "--clip_min", PP_ARGS["clip_min"],
        "--clip_max", PP_ARGS["clip_max"],
    ]
    print(f"\n🗂  Post-processing schedules for {model_name} …")
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(PROJ_ROOT), env=env)
    dt = time.time() - t0
    if proc.returncode == 0:
        print(f"✅ Post-processed {model_name}  ({dt:.1f}s)")
    else:
        print(f"❌ Post-process failed for {model_name} (exit={proc.returncode}, {dt:.1f}s)")
    return proc.returncode

def read_results(model_name: str, path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f"⚠️  Missing results for {model_name}: {path}")
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)

        # normalize columns
        if "Room" not in df.columns and "room" in df.columns:
            df = df.rename(columns={"room": "Room"})
        if "Model" not in df.columns and "model" in df.columns:
            df = df.rename(columns={"model": "Model"})

        for c in ["T","F","preds_path"]:
            if c not in df.columns:
                df[c] = None

        df["Family"] = model_name
        cols = ["Room","Model","Family","T","F","MAE","RMSE","R2","preds_path"]
        cols = [c for c in cols if c in df.columns]
        return df[cols]
    except Exception as e:
        print(f"⚠️  Could not read {model_name} results: {e}")
        return pd.DataFrame()

def aggregate():
    tables = [read_results(name, path) for name, path in RESULT_FILES.items()]
    tables = [t for t in tables if not t.empty]
    if not tables:
        print("\n⚠️  No results to aggregate.")
        return

    master = pd.concat(tables, ignore_index=True)

    # Per-room winners (by RMSE then MAE, when both exist)
    sort_cols = [c for c in ["Room","RMSE","MAE"] if c in master.columns]
    winners = (master.sort_values(sort_cols, ascending=[True,True,True])
                     .groupby("Room", as_index=False)
                     .first())

    # Overall averages
    have_metrics = [c for c in ["MAE","RMSE","R2"] if c in master.columns]
    group_cols   = [c for c in ["Model","Family"] if c in master.columns]
    overall = (master.groupby(group_cols, as_index=False)[have_metrics]
                     .mean()
                     .sort_values([c for c in ["RMSE","MAE"] if c in have_metrics],
                                  ascending=[True, True]))

    # Save
    out_all     = ALL_DIR / "all_models_results.csv"
    out_leader  = ALL_DIR / "per_room_leaderboard.csv"
    out_overall = ALL_DIR / "overall_averages.csv"

    master.to_csv(out_all, index=False)
    winners.to_csv(out_leader, index=False)
    overall.to_csv(out_overall, index=False)

    print("\n==================== SUMMARY ====================")
    print(f"All results:        {out_all}")
    print(f"Per-room best:      {out_leader}")
    print(f"Overall averages:   {out_overall}\n")

    if not overall.empty:
        print("Top-5 overall (by RMSE then MAE):")
        print(overall.head(5).to_string(index=False))

    if not winners.empty:
        print("\nPer-room winners:")
        for _, r in winners.iterrows():
            line = f" - {r['Room']:<10}"
            if "Family" in r and "Model" in r:
                line += f" → {r['Family']}/{r['Model']}"
            if "MAE" in r and "RMSE" in r and "R2" in r:
                line += f" | MAE={r['MAE']:.4f} RMSE={r['RMSE']:.4f} R²={r['R2']:.4f}"
            print(line)

def main(run_preprocess=True, run_train=True, run_post=True):
    # 1) preprocess + training
    for name in SCRIPT_NAMES:
        if not run_preprocess and name == "preprocess.py":
            print("⏭️  Skipping preprocess.py (run_preprocess=False)")
            continue
        if not run_train and name != "preprocess.py":
            print(f"⏭️  Skipping {name} (run_train=False)")
            continue
        rc = run_script(name)
        # 2) immediately post-process that family if requested
        if run_post and rc == 0 and name != "preprocess.py":
            # map filename -> family for postprocess
            family = {
                "baseline_models.py":   "Baseline",
                "train_mamba.py":       "Mamba",
                "train_mambaformer.py": "MambaFormer",
                "train_moe_mamba.py":   "MoE_Mamba",
                "train_occu_mamba.py":  "OccuMamba",
            }.get(name)
            if family:
                run_postprocess(family)

    # 3) aggregate metrics
    aggregate()

if __name__ == "__main__":
    # Toggle any of these flags if you want to skip steps
    main(run_preprocess=True, run_train=True, run_post=True)
