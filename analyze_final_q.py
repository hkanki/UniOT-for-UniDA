import os
import numpy as np
import pandas as pd
import torch

from sklearn.metrics import normalized_mutual_info_score

from data import *
from utils.prototype_split import analyze_true_category_mixing


# ============================================================
# final.pkl の読み込み
# ============================================================

if parser_args.model_path is None:
    raise ValueError(
        "--model_path に final.pkl を指定してください。"
    )


data = torch.load(
    parser_args.model_path,
    map_location="cpu"
)


# ============================================================
# 保存情報
# ============================================================

q_assignment_matrix = (
    data["q_assignment_matrix"]
    .cpu()
    .numpy()
)

q_sample_ids = (
    data["q_sample_ids"]
    .cpu()
    .numpy()
)

initial_K = int(
    data["initial_K"]
)

final_K = int(
    data["final_K"]
)


print("initial_K:", initial_K)
print("final_K:", final_K)
print("Q shape:", q_assignment_matrix.shape)
print("sample count:", len(q_sample_ids))


# ============================================================
# GT label
# ============================================================

target_labels = np.asarray(
    target_train_ds.labels,
    dtype=np.int64
)

tp_classes_np = np.asarray(
    classes_set["tp_classes"],
    dtype=np.int64
)


# ============================================================
# safety check
# ============================================================

if q_assignment_matrix.shape[0] != len(q_sample_ids):
    raise ValueError(
        "Q sample数とsample ID数が一致しません。"
    )

if q_assignment_matrix.shape[1] != final_K:
    raise ValueError(
        f"Q dimension={q_assignment_matrix.shape[1]}, "
        f"final_K={final_K}"
    )

if np.max(q_sample_ids) >= len(target_labels):
    raise ValueError(
        "q_sample_ids がtarget datasetの範囲を超えています。"
    )


# ============================================================
# 各prototypeのGT mixing / Purity
# ============================================================

mixing_results = analyze_true_category_mixing(
    q_assignment_matrix=
        q_assignment_matrix,

    q_sample_ids=
        q_sample_ids,

    target_labels=
        target_labels,

    tp_classes=
        classes_set["tp_classes"]
)

mixing_df = pd.DataFrame(
    mixing_results
)


# ============================================================
# Prototype TP-NMI
# ============================================================

# 各sampleを最もQが大きいprototypeへ割り当て
hard_assignment = np.argmax(
    q_assignment_matrix,
    axis=1
)


# final Qに含まれるsampleのGT label
true_labels = target_labels[
    q_sample_ids
]


# target-private sampleのみ
tp_mask = np.isin(
    true_labels,
    tp_classes_np
)

tp_true_labels = true_labels[
    tp_mask
]

tp_proto_assignment = hard_assignment[
    tp_mask
]


prototype_tp_nmi = (
    normalized_mutual_info_score(
        tp_true_labels,
        tp_proto_assignment
    )
)


# ============================================================
# Aggregate Purity
# ============================================================

valid_df = mixing_df[
    mixing_df["tp_sample_count"] > 0
].copy()


# prototypeごとのPurity単純平均
mean_private_purity = (
    valid_df["purity"].mean()
)


# sample数による重み付きPurity
weighted_private_purity = (
    (
        valid_df["purity"]
        * valid_df["tp_sample_count"]
    ).sum()
    / valid_df["tp_sample_count"].sum()
)


active_private_prototypes = int(
    len(valid_df)
)


# ============================================================
# 保存
# ============================================================

output_dir = os.path.dirname(
    parser_args.model_path
)

mixing_path = os.path.join(
    output_dir,
    "prototype_true_mixing.csv"
)

mixing_df.to_csv(
    mixing_path,
    index=False
)


summary = {
    "initial_K":
        initial_K,

    "final_K":
        final_K,

    "prototype_tp_nmi":
        prototype_tp_nmi,

    "private_purity":
        mean_private_purity,

    "weighted_private_purity":
        weighted_private_purity,

    "active_private_prototypes":
        active_private_prototypes,

    "tp_sample_count":
        int(tp_mask.sum())
}


summary_df = pd.DataFrame(
    [summary]
)

summary_path = os.path.join(
    output_dir,
    "prototype_true_mixing_summary.csv"
)

summary_df.to_csv(
    summary_path,
    index=False
)


# ============================================================
# 表示
# ============================================================

print()
print("========================================")
print("Prototype GT analysis")
print("========================================")

print(
    "Prototype TP-NMI:",
    prototype_tp_nmi
)

print(
    "Mean private purity:",
    mean_private_purity
)

print(
    "Weighted private purity:",
    weighted_private_purity
)

print(
    "Active private prototypes:",
    active_private_prototypes
)

print(
    "TP sample count:",
    int(tp_mask.sum())
)

print()
print(
    "Saved:",
    mixing_path
)

print(
    "Saved:",
    summary_path
)