import argparse
import os

import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve
)


# ============================================================
# Arguments
# ============================================================

parser = argparse.ArgumentParser(
    description=(
        "Evaluate before_dispersion "
        "as a mixed-prototype detection score."
    )
)


parser.add_argument(
    "--csv_path",
    type=str,
    required=True,
    help="Path to before_dispersion_ranked.csv"
)


parser.add_argument(
    "--task",
    type=str,
    required=True,
    help="Task name, e.g. Art->Clipart"
)


# ------------------------------------------------------------
# 混在prototypeの正解定義
# ------------------------------------------------------------

parser.add_argument(
    "--purity_threshold",
    type=float,
    default=0.5,
    help=(
        "Prototype is treated as mixed "
        "when purity is lower than this value."
    )
)


# ------------------------------------------------------------
# before_dispersionの判定閾値
#
# Noneの場合は、評価用として
# F1が最大になるthresholdも調べる
# ------------------------------------------------------------

parser.add_argument(
    "--dispersion_threshold",
    type=float,
    default=None,
    help=(
        "Threshold for before_dispersion. "
        "If omitted, best-F1 threshold is also reported."
    )
)


args = parser.parse_args()


# ============================================================
# Load CSV
# ============================================================

df = pd.read_csv(
    args.csv_path
)


required_columns = [
    "prototype_id",
    "before_dispersion",
    "tp_sample_count",
    "num_categories",
    "purity"
]


for column in required_columns:

    if column not in df.columns:

        raise ValueError(
            f"Required column '{column}' "
            "is not included in CSV."
        )


# ============================================================
# Select valid prototypes
# ============================================================

valid_df = (
    df
    .dropna(
        subset=[
            "before_dispersion",
            "purity"
        ]
    )
    .copy()
)


# target-private sampleを含むprototypeのみ
valid_df = valid_df[
    valid_df[
        "tp_sample_count"
    ] > 0
].copy()


if len(valid_df) < 2:

    raise ValueError(
        "Not enough valid prototypes."
    )


# ============================================================
# Ground truth
#
# purity < threshold
#     -> mixed prototype = 1
#
# purity >= threshold
#     -> non-mixed prototype = 0
# ============================================================

valid_df[
    "true_mixed"
] = (
    valid_df[
        "purity"
    ]
    < args.purity_threshold
).astype(int)


# before_dispersionを予測スコアとして利用
y_true = (
    valid_df[
        "true_mixed"
    ]
    .to_numpy()
)

y_score = (
    valid_df[
        "before_dispersion"
    ]
    .to_numpy()
)


# ============================================================
# Safety check
# ============================================================

num_positive = int(
    np.sum(
        y_true == 1
    )
)

num_negative = int(
    np.sum(
        y_true == 0
    )
)


print(
    "\n=========================================="
)

print(
    "Task:",
    args.task
)

print(
    "Purity threshold:",
    args.purity_threshold
)

print(
    "Valid prototypes:",
    len(valid_df)
)

print(
    "Mixed prototypes:",
    num_positive
)

print(
    "Non-mixed prototypes:",
    num_negative
)

print(
    "=========================================="
)


if (
    num_positive == 0
    or num_negative == 0
):

    raise ValueError(
        "ROC-AUC cannot be calculated "
        "because only one class exists. "
        "Change purity_threshold."
    )


# ============================================================
# ROC-AUC
# ============================================================

roc_auc = roc_auc_score(
    y_true,
    y_score
)


# ============================================================
# PR-AUC
#
# sklearnのaverage_precision_scoreを使用
# ============================================================

pr_auc = average_precision_score(
    y_true,
    y_score
)


# ============================================================
# ROC curve
# ============================================================

fpr, tpr, roc_thresholds = (
    roc_curve(
        y_true,
        y_score
    )
)


# ============================================================
# Precision-Recall curve
# ============================================================

precision_values, recall_values, pr_thresholds = (
    precision_recall_curve(
        y_true,
        y_score
    )
)


# ============================================================
# Find threshold maximizing F1
#
# 注意：
# これはアノテーションを使った評価上の
# 「理想的なthreshold」であり、
# 提案手法にそのまま採用するものではない。
# ============================================================

best_f1 = -1.0
best_threshold = np.nan
best_precision = np.nan
best_recall = np.nan


for threshold in np.unique(
    y_score
):

    prediction = (
        y_score
        >= threshold
    ).astype(int)

    precision = precision_score(
        y_true,
        prediction,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        prediction,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        prediction,
        zero_division=0
    )

    if f1 > best_f1:

        best_f1 = f1
        best_threshold = float(
            threshold
        )

        best_precision = float(
            precision
        )

        best_recall = float(
            recall
        )


# ============================================================
# Evaluation threshold
#
# user指定があればその値
# 指定がなければbest-F1 threshold
# ============================================================

if (
    args.dispersion_threshold
    is None
):

    evaluation_threshold = (
        best_threshold
    )

    threshold_type = (
        "best_f1"
    )

else:

    evaluation_threshold = (
        args.dispersion_threshold
    )

    threshold_type = (
        "specified"
    )


# ============================================================
# Binary prediction
# ============================================================

y_pred = (
    y_score
    >= evaluation_threshold
).astype(int)


# ============================================================
# Classification metrics
# ============================================================

precision = precision_score(
    y_true,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_true,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_true,
    y_pred,
    zero_division=0
)

accuracy = accuracy_score(
    y_true,
    y_pred
)


# ============================================================
# Confusion matrix
# ============================================================

tn, fp, fn, tp = (
    confusion_matrix(
        y_true,
        y_pred,
        labels=[
            0,
            1
        ]
    )
    .ravel()
)


# ============================================================
# Add prediction to dataframe
# ============================================================

valid_df[
    "predicted_mixed"
] = y_pred


valid_df[
    "correct_prediction"
] = (
    valid_df[
        "true_mixed"
    ]
    == valid_df[
        "predicted_mixed"
    ]
)


# ============================================================
# Save directory
# ============================================================

output_dir = os.path.dirname(
    args.csv_path
)


if output_dir == "":
    output_dir = "."


# ============================================================
# Save summary
# ============================================================

summary = {
    "task":
        args.task,

    "num_valid_prototypes":
        int(
            len(valid_df)
        ),

    "purity_threshold":
        float(
            args.purity_threshold
        ),

    "num_mixed":
        num_positive,

    "num_non_mixed":
        num_negative,

    "roc_auc":
        float(
            roc_auc
        ),

    "pr_auc":
        float(
            pr_auc
        ),

    "threshold_type":
        threshold_type,

    "dispersion_threshold":
        float(
            evaluation_threshold
        ),

    "precision":
        float(
            precision
        ),

    "recall":
        float(
            recall
        ),

    "f1_score":
        float(
            f1
        ),

    "accuracy":
        float(
            accuracy
        ),

    "true_positive":
        int(
            tp
        ),

    "false_positive":
        int(
            fp
        ),

    "false_negative":
        int(
            fn
        ),

    "true_negative":
        int(
            tn
        ),

    "best_f1_threshold":
        float(
            best_threshold
        ),

    "best_f1":
        float(
            best_f1
        ),

    "best_f1_precision":
        float(
            best_precision
        ),

    "best_f1_recall":
        float(
            best_recall
        )
}


summary_df = pd.DataFrame(
    [
        summary
    ]
)


summary_path = os.path.join(
    output_dir,
    "before_dispersion_detection_summary.csv"
)


summary_df.to_csv(
    summary_path,
    index=False
)


# ============================================================
# Save prototype-level predictions
# ============================================================

prediction_path = os.path.join(
    output_dir,
    "before_dispersion_detection_prototypes.csv"
)


valid_df = (
    valid_df
    .sort_values(
        "before_dispersion",
        ascending=False
    )
    .reset_index(
        drop=True
    )
)


valid_df.to_csv(
    prediction_path,
    index=False
)


# ============================================================
# Save ROC curve
# ============================================================

roc_df = pd.DataFrame(
    {
        "fpr":
            fpr,

        "tpr":
            tpr,

        "threshold":
            roc_thresholds
    }
)


roc_path = os.path.join(
    output_dir,
    "before_dispersion_roc_curve.csv"
)


roc_df.to_csv(
    roc_path,
    index=False
)


# ============================================================
# Save Precision-Recall curve
# ============================================================

pr_df = pd.DataFrame(
    {
        "precision":
            precision_values[:-1],

        "recall":
            recall_values[:-1],

        "threshold":
            pr_thresholds
    }
)


pr_path = os.path.join(
    output_dir,
    "before_dispersion_pr_curve.csv"
)


pr_df.to_csv(
    pr_path,
    index=False
)


# ============================================================
# Print
# ============================================================

print(
    "\n===== Threshold-independent metrics ====="
)

print(
    "ROC-AUC:",
    roc_auc
)

print(
    "PR-AUC:",
    pr_auc
)


print(
    "\n===== Best-F1 threshold (evaluation only) ====="
)

print(
    "Threshold:",
    best_threshold
)

print(
    "Precision:",
    best_precision
)

print(
    "Recall:",
    best_recall
)

print(
    "F1:",
    best_f1
)


print(
    "\n===== Evaluation at selected threshold ====="
)

print(
    "Threshold:",
    evaluation_threshold
)

print(
    "Precision:",
    precision
)

print(
    "Recall:",
    recall
)

print(
    "F1:",
    f1
)

print(
    "Accuracy:",
    accuracy
)


print(
    "\n===== Confusion matrix ====="
)

print(
    "TP:",
    tp
)

print(
    "FP:",
    fp
)

print(
    "FN:",
    fn
)

print(
    "TN:",
    tn
)


print(
    "\nSaved:"
)

print(
    summary_path
)

print(
    prediction_path
)

print(
    roc_path
)

print(
    pr_path
)