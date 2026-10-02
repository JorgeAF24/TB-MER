import subprocess
import json
import glob
import os
import gc
import time
import torch

SEEDS = [0, 7, 13, 21, 42, 56, 78, 101, 123, 999]
FUSIONS = ["concat", "mlb"]

DATASET = "iemocap"
TRAIN_SCRIPT = "src/train.py"
EVAL_SCRIPT = "src/evaluate.py"
TEMPORAL_ENCODER = "none"

os.makedirs("results", exist_ok=True)
COMPLETED_FILE = os.path.join("results", "completed_runs.json")

# Load completed runs
if os.path.exists(COMPLETED_FILE):
    with open(COMPLETED_FILE, "r") as f:
        completed_runs = json.load(f)
else:
    completed_runs = []

def save_completed_runs():
    with open(COMPLETED_FILE, "w") as f:
        json.dump(completed_runs, f, indent=4)

for fusion in FUSIONS:
    for seed in SEEDS:
        run_id = {
            "dataset": DATASET,
            "fusion": fusion,
            "seed": seed
        }

        if run_id in completed_runs:
            print(
                f"\n[SKIP] dataset={DATASET}, "
                f"fusion={fusion}, seed={seed}"
            )
            continue
        print(f"\n===== Running: fusion={fusion}, seed={seed} =====")

        # -------------------
        # 1. TRAIN
        # -------------------
        train_cmd = [
            "python", TRAIN_SCRIPT,
            "--fusion", fusion,
            "--dataset", DATASET,
            "--seed", str(seed),
            "--temporal_encoder", TEMPORAL_ENCODER,
            "--gpu_resident",
            "--num_workers", "0"
        ]

        try:
            subprocess.run(train_cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Training failed for fusion={fusion}, seed={seed}")
            print(e)
            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            time.sleep(30)
            continue

        # -------------------
        # 2. FIND LAST RUN JSON
        # -------------------
        exp_name = f"temporal_{TEMPORAL_ENCODER}_masked_4Hz_{DATASET}_{fusion}_seed{seed}"
        results_dir = os.path.join("results", exp_name, "train")

        json_files = glob.glob(os.path.join(results_dir, "metrics_*.json"))
        if len(json_files) == 0:
            print(
                f"[ERROR] No metrics JSON found for "
                f"fusion={fusion}, seed={seed}"
            )

            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            time.sleep(30)
            continue
        latest_json = max(json_files, key=os.path.getctime)

        print(f"Using JSON: {latest_json}")

        # -------------------
        # 3. EVALUATE
        # -------------------
        eval_cmd = [
            "python", EVAL_SCRIPT,
            "--run_json", latest_json
        ]

        try:
            subprocess.run(eval_cmd, check=True)
            completed_runs.append(run_id)
            save_completed_runs()

            print(f"[DONE] dataset={DATASET}, "f"fusion={fusion}, seed={seed}")
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Evaluation failed for fusion={fusion}, seed={seed}")
            print(e)
        finally:
            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            time.sleep(30)
print("\nAll experiments completed.")
# Running args: python src/run_experiments.py