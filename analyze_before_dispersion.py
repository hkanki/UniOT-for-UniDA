import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


parser = argparse.ArgumentParser(
    description="Analyze before_dispersion vs true category mixing"
)

parser.add_argument(
    "--csv_path",
    type=str,
    required=True,
    help="Path to prototype_split_category_mixing.csv"
)

parser.add_argument(
    "--task",
    type=str,
    required=True,
    help="Task name, e.g. Art->Clipart"
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
            "is not included in the CSV."
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
    valid_df["tp_sample_count"] > 0
]


if len(valid_df) < 2:

    raise ValueError(
        "Not enough valid prototypes "
        "for correlation analysis."
    )


# ============================================================
# Spearman correlation
# ============================================================

corr_categories, p_categories = (
    spearmanr(
        valid_df[
            "before_dispersion"
        ],
        valid_df[
            "num_categories"
        ]
    )
)

corr_purity, p_purity = (
    spearmanr(
        valid_df[
            "before_dispersion"
        ],
        valid_df[
            "purity"
        ]
    )
)


# ============================================================
# Summary statistics
# ============================================================

mean_before = float(
    valid_df[
        "before_dispersion"
    ].mean()
)

median_before = float(
    valid_df[
        "before_dispersion"
    ].median()
)

max_before = float(
    valid_df[
        "before_dispersion"
    ].max()
)


# ============================================================
# Compare upper and lower quartiles
# ============================================================

q25 = valid_df[
    "before_dispersion"
].quantile(
    0.25
)

q75 = valid_df[
    "before_dispersion"
].quantile(
    0.75
)


low_group = valid_df[
    valid_df[
        "before_dispersion"
    ] <= q25
]

high_group = valid_df[
    valid_df[
        "before_dispersion"
    ] >= q75
]


low_num_categories = float(
    low_group[
        "num_categories"
    ].mean()
)

high_num_categories = float(
    high_group[
        "num_categories"
    ].mean()
)


low_purity = float(
    low_group[
        "purity"
    ].mean()
)

high_purity = float(
    high_group[
        "purity"
    ].mean()
)


# ============================================================
# Save result
# ============================================================

result = {
    "task":
        args.task,

    "num_valid_prototypes":
        int(
            len(valid_df)
        ),

    "before_dispersion_mean":
        mean_before,

    "before_dispersion_median":
        median_before,

    "before_dispersion_max":
        max_before,

    "before_vs_num_categories_spearman":
        corr_categories,

    "before_vs_num_categories_pvalue":
        p_categories,

    "before_vs_purity_spearman":
        corr_purity,

    "before_vs_purity_pvalue":
        p_purity,

    "low25_num_categories_mean":
        low_num_categories,

    "high25_num_categories_mean":
        high_num_categories,

    "low25_purity_mean":
        low_purity,

    "high25_purity_mean":
        high_purity
}


result_df = pd.DataFrame(
    [result]
)


output_dir = os.path.dirname(
    args.csv_path
)

if output_dir == "":
    output_dir = "."


output_path = os.path.join(
    output_dir,
    "before_dispersion_correlation.csv"
)


result_df.to_csv(
    output_path,
    index=False
)


# ============================================================
# Display top prototypes
# ============================================================

sorted_df = (
    valid_df
    .sort_values(
        "before_dispersion",
        ascending=False
    )
    .reset_index(
        drop=True
    )
)


top_output_path = os.path.join(
    output_dir,
    "before_dispersion_ranked.csv"
)

sorted_df.to_csv(
    top_output_path,
    index=False
)


print(
    "\n===================================="
)

print(
    "Task:",
    args.task
)

print(
    "Valid prototypes:",
    len(valid_df)
)

print(
    "===================================="
)


print(
    "\n===== Spearman correlation ====="
)

print(
    "before_dispersion vs num_categories:",
    corr_categories
)

print(
    "p-value:",
    p_categories
)

print(
    "before_dispersion vs purity:",
    corr_purity
)

print(
    "p-value:",
    p_purity
)


print(
    "\n===== Upper vs lower 25% ====="
)

print(
    "Low 25% mean num_categories:",
    low_num_categories
)

print(
    "High 25% mean num_categories:",
    high_num_categories
)

print(
    "Low 25% mean purity:",
    low_purity
)

print(
    "High 25% mean purity:",
    high_purity
)


print(
    "\n===== Top 10 prototypes ====="
)

print(
    sorted_df[
        [
            "prototype_id",
            "before_dispersion",
            "tp_sample_count",
            "num_categories",
            "purity"
        ]
    ].head(10)
)


print(
    "\nSaved:"
)

print(
    output_path
)

print(
    top_output_path
)