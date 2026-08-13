from data import *

import os
import numpy as np
import pandas as pd
import torch

from scipy.stats import spearmanr

from utils.prototype_split import (
    analyze_prototype_splits,
    analyze_true_category_mixing
)


seed = 1234


# ============================================================
# model path check
# ============================================================

if parser_args.model_path is None:
    raise ValueError(
        "Please specify --model_path."
    )


# ============================================================
# load final.pkl
# ============================================================

data = torch.load(
    parser_args.model_path,
    map_location="cpu",
    weights_only=False
)

required_keys = [
    "q_assignment_matrix",
    "q_sample_ids",
    "K"
]

for key in required_keys:
    if key not in data:
        raise KeyError(
            f"{key} is not included in final.pkl."
        )


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

loaded_K = int(
    data["K"]
)


# ============================================================
# basic check
# ============================================================

print(
    "=========================================="
)

print(
    "Task:",
    f"{source} -> {target}"
)

print(
    "K:",
    loaded_K
)

print(
    "Q matrix shape:",
    q_assignment_matrix.shape
)

print(
    "Number of Q sample IDs:",
    len(q_sample_ids)
)

print(
    "Target train size:",
    len(target_train_ds)
)

print(
    "Target label size:",
    len(target_train_ds.labels)
)

print(
    "Maximum Q sample ID:",
    int(q_sample_ids.max())
)

print(
    "=========================================="
)


# ============================================================
# safety check
# ============================================================

if (
    q_assignment_matrix.shape[0]
    != len(q_sample_ids)
):
    raise ValueError(
        "The number of Q vectors and sample IDs does not match."
    )


if (
    q_assignment_matrix.shape[1]
    != loaded_K
):
    raise ValueError(
        "The Q dimension does not match K."
    )


if (
    q_sample_ids.max()
    >= len(target_train_ds.labels)
):
    raise ValueError(
        "q_sample_ids cannot be mapped to target_train_ds.labels."
    )


if not np.all(
    np.isfinite(
        q_assignment_matrix
    )
):
    raise ValueError(
        "q_assignment_matrix contains NaN or Inf."
    )


# ============================================================
# 1. calculate R_k
# ============================================================

split_results = (
    analyze_prototype_splits(
        q_assignment_matrix,
        min_samples=10,
        random_state=seed
    )
)

split_df = pd.DataFrame(
    split_results
)


# ============================================================
# 2. calculate true target-private category mixing
# ============================================================

mixing_results = (
    analyze_true_category_mixing(
        q_assignment_matrix=
            q_assignment_matrix,

        q_sample_ids=
            q_sample_ids,

        target_labels=
            target_train_ds.labels,

        tp_classes=
            classes_set[
                "tp_classes"
            ]
    )
)

mixing_df = pd.DataFrame(
    mixing_results
)


# ============================================================
# 3. merge
# ============================================================

analysis_df = pd.merge(
    split_df,
    mixing_df,
    on="prototype_id",
    how="left"
)

analysis_df = (
    analysis_df
    .sort_values(
        "split_improvement",
        ascending=False,
        na_position="last"
    )
    .reset_index(
        drop=True
    )
)


# ============================================================
# output directory
# ============================================================

model_dir = os.path.dirname(
    parser_args.model_path
)

if model_dir == "":
    model_dir = "."


# ============================================================
# 4. save prototype-level result
# ============================================================

analysis_path = os.path.join(
    model_dir,
    "prototype_split_category_mixing.csv"
)

analysis_df.to_csv(
    analysis_path,
    index=False
)


# ============================================================
# 5. correlation analysis
# ============================================================

valid_df = (
    analysis_df
    .dropna(
        subset=[
            "split_improvement",
            "purity"
        ]
    )
    .copy()
)

# target-private sampleを1件以上含むprototypeのみ
valid_df = valid_df[
    valid_df["tp_sample_count"] > 0
]


if len(valid_df) >= 2:

    corr_categories, p_categories = (
        spearmanr(
            valid_df[
                "split_improvement"
            ],
            valid_df[
                "num_categories"
            ]
        )
    )

    corr_purity, p_purity = (
        spearmanr(
            valid_df[
                "split_improvement"
            ],
            valid_df[
                "purity"
            ]
        )
    )

else:

    corr_categories = np.nan
    p_categories = np.nan

    corr_purity = np.nan
    p_purity = np.nan


correlation_result = {
    "task":
        f"{source}->{target}",

    "K":
        loaded_K,

    "num_valid_prototypes":
        int(
            len(valid_df)
        ),

    "rk_num_categories_spearman":
        corr_categories,

    "rk_num_categories_pvalue":
        p_categories,

    "rk_purity_spearman":
        corr_purity,

    "rk_purity_pvalue":
        p_purity
}


correlation_df = pd.DataFrame(
    [correlation_result]
)

correlation_path = os.path.join(
    model_dir,
    "prototype_split_correlation.csv"
)

correlation_df.to_csv(
    correlation_path,
    index=False
)


# ============================================================
# print result
# ============================================================

print(
    "\n===== Top 20 prototypes by R_k ====="
)

print(
    analysis_df[
        [
            "prototype_id",
            "sample_count",
            "split_improvement",
            "tp_sample_count",
            "num_categories",
            "purity"
        ]
    ].head(20)
)


print(
    "\n===== Spearman correlation ====="
)

print(
    "R_k vs num_categories:",
    corr_categories
)

print(
    "p-value:",
    p_categories
)

print(
    "R_k vs purity:",
    corr_purity
)

print(
    "p-value:",
    p_purity
)


print(
    "\nSaved:"
)

print(
    analysis_path
)

print(
    correlation_path
)