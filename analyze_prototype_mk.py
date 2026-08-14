import os
import numpy as np
import pandas as pd
import torch

from sklearn.metrics import (
    normalized_mutual_info_score,
    roc_auc_score
)

from data import *

from utils.prototype_split import (
    analyze_true_category_mixing,
    analyze_prototype_splits
)


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
# final.pkl 保存情報
# ============================================================

if "q_assignment_matrix" not in data:
    raise KeyError(
        "final.pkl に q_assignment_matrix がありません。"
    )

if "q_sample_ids" not in data:
    raise KeyError(
        "final.pkl に q_sample_ids がありません。"
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


# ============================================================
# K / method / tau_mix
# ============================================================

initial_K = int(
    data.get(
        "initial_K",
        data.get("K")
    )
)


final_K = int(
    data.get(
        "final_K",
        data.get("K")
    )
)


method = data.get(
    "method",
    "unknown"
)


tau_mix = data.get(
    "tau_mix",
    None
)


print("method:", method)
print("initial_K:", initial_K)
print("final_K:", final_K)
print("tau_mix:", tau_mix)
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

if q_assignment_matrix.ndim != 2:
    raise ValueError(
        "q_assignment_matrix は2次元である必要があります。"
    )


if q_assignment_matrix.shape[0] != len(q_sample_ids):
    raise ValueError(
        "Q sample数とsample ID数が一致しません。"
    )


if q_assignment_matrix.shape[1] != final_K:
    raise ValueError(
        f"Q dimension={q_assignment_matrix.shape[1]}, "
        f"final_K={final_K}"
    )


if len(q_sample_ids) == 0:
    raise ValueError(
        "q_sample_ids が空です。"
    )


if np.max(q_sample_ids) >= len(target_labels):
    raise ValueError(
        "q_sample_ids がtarget datasetの範囲を超えています。"
    )


if not np.all(
    np.isfinite(
        q_assignment_matrix
    )
):
    raise ValueError(
        "q_assignment_matrix にNaNまたはInfがあります。"
    )


# ============================================================
# 念のためQを確率分布として再正規化
# ============================================================

q_assignment_matrix = (
    q_assignment_matrix
    /
    np.clip(
        q_assignment_matrix.sum(
            axis=1,
            keepdims=True
        ),
        1e-12,
        None
    )
)


# ============================================================
# Hard assignment
# ============================================================

hard_assignment = np.argmax(
    q_assignment_matrix,
    axis=1
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
# 各prototypeの M_k
#
# before_dispersion = 現在split判定に用いている M_k
# ============================================================

dispersion_results = analyze_prototype_splits(
    q_assignment_matrix=
        q_assignment_matrix,

    min_samples=10,

    random_state=1234
)


dispersion_df = pd.DataFrame(
    dispersion_results
)


dispersion_df = (
    dispersion_df[
        [
            "prototype_id",
            "sample_count",
            "before_dispersion",
            "split_dispersion",
            "split_improvement"
        ]
    ]
    .rename(
        columns={
            "before_dispersion":
                "M_k"
        }
    )
)


# ============================================================
# GT mixing と M_k を統合
# ============================================================

analysis_df = pd.merge(
    mixing_df,
    dispersion_df,
    on="prototype_id",
    how="left"
)


# ============================================================
# GTカテゴリ構成に基づく代表性指標
# ============================================================

def parse_category_counts(
    category_counts
):

    if (
        pd.isna(category_counts)
        or str(category_counts).strip() == ""
    ):
        return []


    counts = []


    for item in str(
        category_counts
    ).split(","):

        item = item.strip()

        if item == "":
            continue


        _, count = item.split(":")


        counts.append(
            int(count)
        )


    return counts


def calculate_gt_representative_metrics(
    category_counts
):

    counts = parse_category_counts(
        category_counts
    )


    if len(counts) == 0:

        return pd.Series(
            {
                "gt_top1_ratio":
                    np.nan,

                "gt_top2_ratio":
                    np.nan,

                "gt_top1_top2_margin":
                    np.nan,

                "gt_category_entropy":
                    np.nan,

                "gt_normalized_category_entropy":
                    np.nan
            }
        )


    counts = np.asarray(
        counts,
        dtype=np.float64
    )


    total = counts.sum()


    if total <= 0:

        return pd.Series(
            {
                "gt_top1_ratio":
                    np.nan,

                "gt_top2_ratio":
                    np.nan,

                "gt_top1_top2_margin":
                    np.nan,

                "gt_category_entropy":
                    np.nan,

                "gt_normalized_category_entropy":
                    np.nan
            }
        )


    probs = (
        counts
        / total
    )


    sorted_probs = np.sort(
        probs
    )[::-1]


    gt_top1_ratio = float(
        sorted_probs[0]
    )


    if len(sorted_probs) >= 2:

        gt_top2_ratio = float(
            sorted_probs[1]
        )

    else:

        gt_top2_ratio = 0.0


    gt_top1_top2_margin = float(
        gt_top1_ratio
        - gt_top2_ratio
    )


    gt_category_entropy = float(
        -np.sum(
            probs
            * np.log(
                np.clip(
                    probs,
                    1e-12,
                    None
                )
            )
        )
    )


    if len(probs) > 1:

        gt_normalized_category_entropy = float(
            gt_category_entropy
            /
            np.log(
                len(probs)
            )
        )

    else:

        gt_normalized_category_entropy = 0.0


    return pd.Series(
        {
            "gt_top1_ratio":
                gt_top1_ratio,

            "gt_top2_ratio":
                gt_top2_ratio,

            "gt_top1_top2_margin":
                gt_top1_top2_margin,

            "gt_category_entropy":
                gt_category_entropy,

            "gt_normalized_category_entropy":
                gt_normalized_category_entropy
        }
    )


gt_metrics_df = (
    analysis_df[
        "category_counts"
    ]
    .apply(
        calculate_gt_representative_metrics
    )
)


analysis_df = pd.concat(
    [
        analysis_df,
        gt_metrics_df
    ],
    axis=1
)


# ============================================================
# 新しい仮説:
# prototypeごとの平均Q分布を調べる
#
# mean_q_entropy:
#   平均Qが均等なほど大きい
#
# mean_q_top1_ratio:
#   平均Qの最大値
#
# mean_q_top1_top2_margin:
#   平均Qの1位と2位の差
#
# 理想:
#
#   機能しているprototype
#       ↓
#   mean Q が1方向へ集中
#       ↓
#   entropy 小
#   top1 大
#   margin 大
#
#   機能していないprototype
#       ↓
#   mean Q が複数方向へ均等
#       ↓
#   entropy 大
#   top1 小
#   margin 小
# ============================================================

q_representative_results = []


for proto_id in range(
    final_K
):

    mask = (
        hard_assignment
        == proto_id
    )


    q_vectors = (
        q_assignment_matrix[
            mask
        ]
    )


    sample_count = int(
        q_vectors.shape[0]
    )


    if sample_count == 0:

        q_representative_results.append(
            {
                "prototype_id":
                    proto_id,

                "q_sample_count":
                    0,

                "mean_q_entropy":
                    np.nan,

                "mean_q_normalized_entropy":
                    np.nan,

                "mean_q_top1_ratio":
                    np.nan,

                "mean_q_top2_ratio":
                    np.nan,

                "mean_q_top1_top2_margin":
                    np.nan,

                "mean_q_effective_prototypes":
                    np.nan
            }
        )

        continue


    # --------------------------------------------------------
    # prototypeに属するsampleの平均Q
    # --------------------------------------------------------

    mean_q = np.mean(
        q_vectors,
        axis=0
    )


    mean_q = (
        mean_q
        /
        np.clip(
            mean_q.sum(),
            1e-12,
            None
        )
    )


    sorted_mean_q = np.sort(
        mean_q
    )[::-1]


    # --------------------------------------------------------
    # 平均QのTop1
    # --------------------------------------------------------

    mean_q_top1_ratio = float(
        sorted_mean_q[0]
    )


    # --------------------------------------------------------
    # 平均QのTop2
    # --------------------------------------------------------

    if len(
        sorted_mean_q
    ) >= 2:

        mean_q_top2_ratio = float(
            sorted_mean_q[1]
        )

    else:

        mean_q_top2_ratio = 0.0


    # --------------------------------------------------------
    # Top1 - Top2 margin
    # --------------------------------------------------------

    mean_q_top1_top2_margin = float(
        mean_q_top1_ratio
        - mean_q_top2_ratio
    )


    # --------------------------------------------------------
    # 平均Q Entropy
    # --------------------------------------------------------

    mean_q_entropy = float(
        -np.sum(
            mean_q
            * np.log(
                np.clip(
                    mean_q,
                    1e-12,
                    None
                )
            )
        )
    )


    # --------------------------------------------------------
    # Kで正規化したEntropy
    #
    # 0:
    #   1つに完全集中
    #
    # 1:
    #   全prototypeへ完全均等
    # --------------------------------------------------------

    if final_K > 1:

        mean_q_normalized_entropy = float(
            mean_q_entropy
            /
            np.log(
                final_K
            )
        )

    else:

        mean_q_normalized_entropy = 0.0


    # --------------------------------------------------------
    # Effective number of prototypes
    #
    # exp(H)
    #
    # 1に近い:
    #   ほぼ1prototype
    #
    # 大きい:
    #   多方向に分散
    # --------------------------------------------------------

    mean_q_effective_prototypes = float(
        np.exp(
            mean_q_entropy
        )
    )


    q_representative_results.append(
        {
            "prototype_id":
                proto_id,

            "q_sample_count":
                sample_count,

            "mean_q_entropy":
                mean_q_entropy,

            "mean_q_normalized_entropy":
                mean_q_normalized_entropy,

            "mean_q_top1_ratio":
                mean_q_top1_ratio,

            "mean_q_top2_ratio":
                mean_q_top2_ratio,

            "mean_q_top1_top2_margin":
                mean_q_top1_top2_margin,

            "mean_q_effective_prototypes":
                mean_q_effective_prototypes
        }
    )


q_representative_df = pd.DataFrame(
    q_representative_results
)


# ============================================================
# 新しいQ代表性指標を統合
# ============================================================

analysis_df = pd.merge(
    analysis_df,
    q_representative_df,
    on="prototype_id",
    how="left"
)


# ============================================================
# GT Purity group
# ============================================================

def purity_group(
    purity
):

    if pd.isna(
        purity
    ):
        return "no_private_sample"

    elif purity >= 0.9:
        return "purity_0.9_1.0"

    elif purity >= 0.7:
        return "purity_0.7_0.9"

    elif purity >= 0.5:
        return "purity_0.5_0.7"

    else:
        return "purity_below_0.5"


analysis_df[
    "purity_group"
] = (
    analysis_df[
        "purity"
    ]
    .apply(
        purity_group
    )
)


# ============================================================
# GT上の「prototypeとして機能していない」
# を分析用に定義
#
# 注意:
# 学習には使用しない。
#
# まず複数基準を用意し、
# 新しいQ指標がどの基準と対応するかを見る。
# ============================================================

analysis_df[
    "gt_low_purity_050"
] = (
    analysis_df[
        "purity"
    ]
    < 0.50
)


analysis_df[
    "gt_low_purity_060"
] = (
    analysis_df[
        "purity"
    ]
    < 0.60
)


analysis_df[
    "gt_low_purity_070"
] = (
    analysis_df[
        "purity"
    ]
    < 0.70
)


# ============================================================
# Prototype TP-NMI
# ============================================================

true_labels = target_labels[
    q_sample_ids
]


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


if len(
    tp_true_labels
) > 0:

    prototype_tp_nmi = (
        normalized_mutual_info_score(
            tp_true_labels,
            tp_proto_assignment
        )
    )

else:

    prototype_tp_nmi = np.nan


# ============================================================
# Aggregate Purity
# ============================================================

valid_private_df = analysis_df[
    analysis_df[
        "tp_sample_count"
    ]
    > 0
].copy()


if len(
    valid_private_df
) > 0:

    mean_private_purity = (
        valid_private_df[
            "purity"
        ].mean()
    )


    weighted_private_purity = (
        (
            valid_private_df[
                "purity"
            ]
            *
            valid_private_df[
                "tp_sample_count"
            ]
        ).sum()
        /
        valid_private_df[
            "tp_sample_count"
        ].sum()
    )

else:

    mean_private_purity = np.nan
    weighted_private_purity = np.nan


active_private_prototypes = int(
    len(
        valid_private_df
    )
)


# ============================================================
# GT Purity別に
# M_k と新Q指標を比較
# ============================================================

purity_group_order = [
    "purity_0.9_1.0",
    "purity_0.7_0.9",
    "purity_0.5_0.7",
    "purity_below_0.5"
]


purity_summary_rows = []


for group_name in purity_group_order:

    group_df = analysis_df[
        (
            analysis_df[
                "purity_group"
            ]
            == group_name
        )
    ].copy()


    row = {

        "purity_group":
            group_name,

        "prototype_count":
            int(
                len(
                    group_df
                )
            ),

        "M_k_mean":
            group_df[
                "M_k"
            ].mean(),

        "M_k_median":
            group_df[
                "M_k"
            ].median(),

        "mean_q_normalized_entropy_mean":
            group_df[
                "mean_q_normalized_entropy"
            ].mean(),

        "mean_q_normalized_entropy_median":
            group_df[
                "mean_q_normalized_entropy"
            ].median(),

        "mean_q_top1_ratio_mean":
            group_df[
                "mean_q_top1_ratio"
            ].mean(),

        "mean_q_top1_top2_margin_mean":
            group_df[
                "mean_q_top1_top2_margin"
            ].mean(),

        "mean_q_effective_prototypes_mean":
            group_df[
                "mean_q_effective_prototypes"
            ].mean(),

        "gt_entropy_mean":
            group_df[
                "gt_normalized_category_entropy"
            ].mean()
    }


    purity_summary_rows.append(
        row
    )


purity_summary_df = pd.DataFrame(
    purity_summary_rows
)


# ============================================================
# Correlation analysis
#
# 新しい提案が妥当なら:
#
# mean_q_normalized_entropy
#   vs purity                 負
#   vs GT entropy             正
#
# mean_q_top1_ratio
#   vs purity                 正
#
# mean_q_top1_top2_margin
#   vs purity                 正
#
# さらに、M_kと比較する。
# ============================================================

correlation_columns = [
    "M_k",
    "split_improvement",
    "purity",
    "gt_top1_ratio",
    "gt_top1_top2_margin",
    "gt_normalized_category_entropy",
    "num_categories",
    "mean_q_normalized_entropy",
    "mean_q_top1_ratio",
    "mean_q_top2_ratio",
    "mean_q_top1_top2_margin",
    "mean_q_effective_prototypes"
]


correlation_df = (
    analysis_df[
        correlation_columns
    ]
    .dropna()
)


if len(
    correlation_df
) >= 2:

    pearson_corr = (
        correlation_df.corr(
            method="pearson"
        )
    )


    spearman_corr = (
        correlation_df.corr(
            method="spearman"
        )
    )

else:

    pearson_corr = pd.DataFrame()
    spearman_corr = pd.DataFrame()


# ============================================================
# 重要な相関だけ抜き出す
# ============================================================

comparison_metrics = [
    "M_k",
    "mean_q_normalized_entropy",
    "mean_q_top1_ratio",
    "mean_q_top1_top2_margin",
    "mean_q_effective_prototypes"
]


target_gt_metrics = [
    "purity",
    "gt_top1_top2_margin",
    "gt_normalized_category_entropy"
]


comparison_rows = []


for metric in comparison_metrics:

    row = {
        "metric":
            metric
    }


    for gt_metric in target_gt_metrics:

        if (
            not spearman_corr.empty
            and
            metric in spearman_corr.index
            and
            gt_metric in spearman_corr.columns
        ):

            row[
                f"spearman_vs_{gt_metric}"
            ] = float(
                spearman_corr.loc[
                    metric,
                    gt_metric
                ]
            )

        else:

            row[
                f"spearman_vs_{gt_metric}"
            ] = np.nan


        if (
            not pearson_corr.empty
            and
            metric in pearson_corr.index
            and
            gt_metric in pearson_corr.columns
        ):

            row[
                f"pearson_vs_{gt_metric}"
            ] = float(
                pearson_corr.loc[
                    metric,
                    gt_metric
                ]
            )

        else:

            row[
                f"pearson_vs_{gt_metric}"
            ] = np.nan


    comparison_rows.append(
        row
    )


metric_comparison_df = pd.DataFrame(
    comparison_rows
)


# ============================================================
# ROC-AUC
#
# GT Purityが低いprototypeを
# 「機能していない」と仮定した場合、
# 各Q指標がどれだけ区別できるかを見る。
#
# 正解ラベルは分析にのみ使用。
# ============================================================

auc_rows = []


auc_score_definitions = {

    # 大きいほど悪い
    "M_k":
        analysis_df[
            "M_k"
        ],

    # 大きいほど悪い
    "mean_q_normalized_entropy":
        analysis_df[
            "mean_q_normalized_entropy"
        ],

    # 小さいほど悪いため符号反転
    "negative_mean_q_top1_ratio":
        -analysis_df[
            "mean_q_top1_ratio"
        ],

    # 小さいほど悪いため符号反転
    "negative_mean_q_top1_top2_margin":
        -analysis_df[
            "mean_q_top1_top2_margin"
        ],

    # 大きいほど悪い
    "mean_q_effective_prototypes":
        analysis_df[
            "mean_q_effective_prototypes"
        ]
}


gt_failure_definitions = {
    "purity_below_0.50":
        0.50,

    "purity_below_0.60":
        0.60,

    "purity_below_0.70":
        0.70
}


for failure_name, purity_threshold in (
    gt_failure_definitions.items()
):

    y_true_all = (
        analysis_df[
            "purity"
        ]
        < purity_threshold
    )


    for metric_name, score_series in (
        auc_score_definitions.items()
    ):

        valid_mask = (
            analysis_df[
                "purity"
            ].notna()
            &
            score_series.notna()
        )


        y_true = (
            y_true_all[
                valid_mask
            ]
            .astype(
                int
            )
            .to_numpy()
        )


        y_score = (
            score_series[
                valid_mask
            ]
            .to_numpy()
        )


        # 0と1の両方が存在する場合のみAUC
        if (
            len(
                np.unique(
                    y_true
                )
            )
            == 2
        ):

            auc_value = (
                roc_auc_score(
                    y_true,
                    y_score
                )
            )

        else:

            auc_value = np.nan


        auc_rows.append(
            {
                "gt_failure_definition":
                    failure_name,

                "metric":
                    metric_name,

                "roc_auc":
                    auc_value,

                "valid_prototypes":
                    int(
                        len(
                            y_true
                        )
                    ),

                "failure_prototypes":
                    int(
                        y_true.sum()
                    )
            }
        )


auc_df = pd.DataFrame(
    auc_rows
)


# ============================================================
# Summary
# ============================================================

summary = {

    "method":
        method,

    "initial_K":
        initial_K,

    "final_K":
        final_K,

    "tau_mix":
        tau_mix,

    "prototype_tp_nmi":
        prototype_tp_nmi,

    "private_purity":
        mean_private_purity,

    "weighted_private_purity":
        weighted_private_purity,

    "active_private_prototypes":
        active_private_prototypes,

    "tp_sample_count":
        int(
            tp_mask.sum()
        ),

    "mean_M_k":
        analysis_df[
            "M_k"
        ].mean(),

    "mean_q_normalized_entropy":
        analysis_df[
            "mean_q_normalized_entropy"
        ].mean(),

    "mean_q_top1_ratio":
        analysis_df[
            "mean_q_top1_ratio"
        ].mean(),

    "mean_q_top1_top2_margin":
        analysis_df[
            "mean_q_top1_top2_margin"
        ].mean(),

    "mean_q_effective_prototypes":
        analysis_df[
            "mean_q_effective_prototypes"
        ].mean(),

    "mean_gt_normalized_entropy":
        analysis_df[
            "gt_normalized_category_entropy"
        ].mean()
}


summary_df = pd.DataFrame(
    [
        summary
    ]
)


# ============================================================
# 保存
# ============================================================

output_dir = os.path.dirname(
    parser_args.model_path
)


# ------------------------------------------------------------
# prototype単位詳細
# ------------------------------------------------------------

analysis_path = os.path.join(
    output_dir,
    "prototype_q_functionality_analysis.csv"
)


analysis_df.to_csv(
    analysis_path,
    index=False
)


# ------------------------------------------------------------
# Purity別比較
# ------------------------------------------------------------

purity_summary_path = os.path.join(
    output_dir,
    "prototype_q_functionality_purity_summary.csv"
)


purity_summary_df.to_csv(
    purity_summary_path,
    index=False
)


# ------------------------------------------------------------
# 指標比較
# ------------------------------------------------------------

metric_comparison_path = os.path.join(
    output_dir,
    "prototype_q_metric_comparison.csv"
)


metric_comparison_df.to_csv(
    metric_comparison_path,
    index=False
)


# ------------------------------------------------------------
# ROC-AUC
# ------------------------------------------------------------

auc_path = os.path.join(
    output_dir,
    "prototype_q_metric_auc.csv"
)


auc_df.to_csv(
    auc_path,
    index=False
)


# ------------------------------------------------------------
# Pearson
# ------------------------------------------------------------

pearson_path = os.path.join(
    output_dir,
    "prototype_q_metric_pearson.csv"
)


pearson_corr.to_csv(
    pearson_path
)


# ------------------------------------------------------------
# Spearman
# ------------------------------------------------------------

spearman_path = os.path.join(
    output_dir,
    "prototype_q_metric_spearman.csv"
)


spearman_corr.to_csv(
    spearman_path
)


# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------

summary_path = os.path.join(
    output_dir,
    "prototype_q_functionality_summary.csv"
)


summary_df.to_csv(
    summary_path,
    index=False
)


# ============================================================
# 表示
# ============================================================

print()
print(
    "============================================================"
)

print(
    "Prototype Q functionality analysis"
)

print(
    "============================================================"
)


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


print()
print(
    "------------------------------------------------------------"
)

print(
    "Q-based functionality metrics"
)

print(
    "------------------------------------------------------------"
)


print(
    "Mean M_k:",
    analysis_df[
        "M_k"
    ].mean()
)


print(
    "Mean Q normalized entropy:",
    analysis_df[
        "mean_q_normalized_entropy"
    ].mean()
)


print(
    "Mean Q Top1 ratio:",
    analysis_df[
        "mean_q_top1_ratio"
    ].mean()
)


print(
    "Mean Q Top1-Top2 margin:",
    analysis_df[
        "mean_q_top1_top2_margin"
    ].mean()
)


print(
    "Mean Q effective prototypes:",
    analysis_df[
        "mean_q_effective_prototypes"
    ].mean()
)


print()
print(
    "------------------------------------------------------------"
)

print(
    "Metrics by GT Purity group"
)

print(
    "------------------------------------------------------------"
)


print(
    purity_summary_df.to_string(
        index=False
    )
)


print()
print(
    "------------------------------------------------------------"
)

print(
    "Metric comparison with GT"
)

print(
    "------------------------------------------------------------"
)


print(
    metric_comparison_df.to_string(
        index=False
    )
)


print()
print(
    "------------------------------------------------------------"
)

print(
    "ROC-AUC for detecting low-Purity prototypes"
)

print(
    "------------------------------------------------------------"
)


print(
    auc_df.to_string(
        index=False
    )
)


print()
print(
    "------------------------------------------------------------"
)

print(
    "Spearman correlation"
)

print(
    "------------------------------------------------------------"
)


if not spearman_corr.empty:

    print(
        spearman_corr.to_string()
    )

else:

    print(
        "Not enough valid prototypes."
    )


print()
print(
    "------------------------------------------------------------"
)

print(
    "Saved files"
)

print(
    "------------------------------------------------------------"
)


print(
    analysis_path
)

print(
    purity_summary_path
)

print(
    metric_comparison_path
)

print(
    auc_path
)

print(
    pearson_path
)

print(
    spearman_path
)

print(
    summary_path
)