#!/usr/bin/env bash
set -euo pipefail

# ===== 基本設定 =====
GPU_ID=0
DATASET=office31
SOURCE=amazon
TARGET=dslr

# 実験名
BASE_EXP=office31_repro
TUNE_EXP=office31_tune

# seed 一覧
SEEDS=(1234 2024 7777)

# パラメータ候補
GAMMAS=(0.6 0.7 0.8)
LAMS=(0.05 0.1 0.2)
MUS=(0.5 0.7 0.9)

CONFIG_PATH="config/office31-config.yaml"
BACKUP_CONFIG="config/office31-config.yaml.bak"

# ===== バックアップ作成 =====
cp "${CONFIG_PATH}" "${BACKUP_CONFIG}"

restore_config() {
  cp "${BACKUP_CONFIG}" "${CONFIG_PATH}"
}
trap restore_config EXIT

mkdir -p results

echo "========================================"
echo "Step 1: 現在環境で固定条件の再実行"
echo "========================================"

for seed in "${SEEDS[@]}"; do
  echo "[RUN] baseline seed=${seed}"
  python main.py \
    --gpu_index "${GPU_ID}" \
    --exp "${BASE_EXP}_seed${seed}" \
    --dataset "${DATASET}" \
    --source "${SOURCE}" \
    --target "${TARGET}" \
    --seed "${seed}"
done

echo "========================================"
echo "Step 2: 平均とばらつきの確認"
echo "========================================"

python - <<'PY'
import os
import glob
import pandas as pd

base = "log"
target_dirs = sorted(glob.glob(os.path.join(base, "office31_repro_seed*", "amazon2dslr_*")))
rows = []

for d in target_dirs:
    csv_path = os.path.join(d, "result.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        row = df.iloc[0].to_dict()
        row["log_dir"] = d
        rows.append(row)

if not rows:
    print("result.csv が見つかりませんでした")
    raise SystemExit(1)

res = pd.DataFrame(rows)
metrics = ["cls_common_acc", "cls_tp_acc", "tp_nmi", "cls_overall_acc", "h_score", "h3_score"]

print("\n=== 各実験結果 ===")
print(res[["log_dir"] + metrics].to_string(index=False))

summary = res[metrics].agg(["mean", "std", "min", "max"])
print("\n=== 平均・標準偏差・最小・最大 ===")
print(summary)

summary.to_csv("results/baseline_summary.csv")
res.to_csv("results/baseline_all_runs.csv", index=False)
print("\n保存: results/baseline_summary.csv, results/baseline_all_runs.csv")
PY

echo "========================================"
echo "Step 3: 少数パラメータだけ調整"
echo "========================================"

run_tune () {
  local gamma="$1"
  local lam="$2"
  local mu="$3"
  local seed="$4"

  python - <<PY
import yaml

config_path = "${CONFIG_PATH}"
with open(config_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

cfg["param"]["gamma"] = ${gamma}
cfg["param"]["lam"] = ${lam}
cfg["param"]["mu"] = ${mu}

with open(config_path, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
PY

  echo "[RUN] gamma=${gamma}, lam=${lam}, mu=${mu}, seed=${seed}"
  python main.py \
    --gpu_index "${GPU_ID}" \
    --exp "${TUNE_EXP}_g${gamma}_l${lam}_m${mu}_s${seed}" \
    --dataset "${DATASET}" \
    --source "${SOURCE}" \
    --target "${TARGET}" \
    --seed "${seed}"
}

# 基準 seed だけで軽く探索
BASE_SEED=1234

for gamma in "${GAMMAS[@]}"; do
  run_tune "${gamma}" "0.1" "0.7" "${BASE_SEED}"
done

for lam in "${LAMS[@]}"; do
  run_tune "0.7" "${lam}" "0.7" "${BASE_SEED}"
done

for mu in "${MUS[@]}"; do
  run_tune "0.7" "0.1" "${mu}" "${BASE_SEED}"
done

echo "========================================"
echo "チューニング結果の集計"
echo "========================================"

python - <<'PY'
import os
import glob
import pandas as pd

base = "log"
target_dirs = sorted(glob.glob(os.path.join(base, "office31_tune_*", "amazon2dslr_*")))
rows = []

for d in target_dirs:
    csv_path = os.path.join(d, "result.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        row = df.iloc[0].to_dict()
        row["log_dir"] = d
        rows.append(row)

if not rows:
    print("tuning result.csv が見つかりませんでした")
    raise SystemExit(1)

res = pd.DataFrame(rows)
metrics = ["cls_common_acc", "cls_tp_acc", "tp_nmi", "cls_overall_acc", "h_score", "h3_score"]

print("\n=== チューニング結果一覧 ===")
print(res[["log_dir"] + metrics].sort_values("h_score", ascending=False).to_string(index=False))

res.sort_values("h_score", ascending=False).to_csv("results/tuning_results_sorted.csv", index=False)
print("\n保存: results/tuning_results_sorted.csv")
PY

echo "完了しました。"