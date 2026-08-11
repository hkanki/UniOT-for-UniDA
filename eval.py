from easydl import variable_to_numpy
from easydl import TrainingModeManager, Accumulator

import os

import faiss
import numpy as np
import ot
import pandas as pd
import torch
import torch.nn.functional as F

from tqdm import tqdm
from sklearn.metrics import normalized_mutual_info_score

from utils.lib import (
    seed_everything,
    ubot_CCD,
    adaptive_filling
)

from utils.util import (
    ResultsCalculator,
    calculate_private_purity,
    calculate_weighted_private_purity,
    calculate_prototypes_per_class,
    calculate_average_prototypes_per_class
)

from utils.q_assignment_merge import (
    merge_prototypes_by_q_distribution,
    apply_prototype_groups
)


# ============================================================
# K-means
# ============================================================
def run_kmeans(
    L2_feat,
    ncentroids,
    init_centroids=None,
    seed=None,
    gpu=False,
    min_points_per_centroid=1
):
    if seed is None:
        seed = int(
            os.environ["PYTHONHASHSEED"]
        )

    dim = L2_feat.shape[1]

    kmeans = faiss.Kmeans(
        d=dim,
        k=ncentroids,
        seed=seed,
        gpu=gpu,
        niter=20,
        verbose=False,
        nredo=5,
        min_points_per_centroid=min_points_per_centroid,
        spherical=True
    )

    if torch.is_tensor(L2_feat):
        L2_feat = variable_to_numpy(
            L2_feat
        )

    kmeans.train(
        L2_feat,
        init_centroids=init_centroids
    )

    _, pred_centroid = (
        kmeans.index.search(
            L2_feat,
            1
        )
    )

    pred_centroid = np.squeeze(
        pred_centroid
    )

    return (
        pred_centroid,
        kmeans.centroids
    )


# ============================================================
# Evaluation
# ============================================================
def eval(
    feature_extractor,
    classifier,
    cluster_head,
    eval_dl,
    classes_set,
    gamma=0.7,
    beta=None,
    seed=None,
    uniformed_index=None,
    prototype_groups=None
):

    if seed is None:
        seed = int(
            os.environ["PYTHONHASHSEED"]
        )

    if uniformed_index is None:
        uniformed_index = len(
            classes_set["source_classes"]
        )

    # source prototype
    source_prototype = (
        classifier.module.ProtoCLS.fc.weight
    )

    if beta is None:
        beta = ot.unif(
            source_prototype.size()[0]
        )

    # ========================================================
    # Extract target features
    # ========================================================
    with TrainingModeManager(
        [
            feature_extractor,
            classifier,
            cluster_head
        ],
        train=False
    ) as mgr, \
            Accumulator(
                [
                    "label_t",
                    "norm_feat_t"
                ]
            ) as eval_accumulator, \
            torch.no_grad():

        for i, (
            im_t,
            label_t
        ) in enumerate(
            tqdm(
                eval_dl,
                desc="testing"
            )
        ):

            im_t = im_t.cuda()
            label_t = label_t.cuda()

            feature_ex_t = (
                feature_extractor.forward(
                    im_t
                )
            )

            (
                before_lincls_feat_t,
                after_lincls_t
            ) = classifier(
                feature_ex_t
            )

            norm_feat_t = F.normalize(
                before_lincls_feat_t
            )

            val = {}

            for name in eval_accumulator.names:
                val[name] = (
                    locals()[name]
                    .cpu()
                    .data
                    .numpy()
                )

            eval_accumulator.updateData(
                val
            )

    val = {}

    for x in eval_accumulator:
        val[x] = (
            eval_accumulator[x]
        )

    label_t = val[
        "label_t"
    ]

    norm_feat_t = val[
        "norm_feat_t"
    ]

    del val

    # ========================================================
    # Target prototype prediction
    # ========================================================
    with torch.no_grad():

        norm_feat_t_tensor = (
            torch.from_numpy(
                norm_feat_t
            )
            .float()
            .cuda()
        )

        proto_logits = (
            cluster_head(
                norm_feat_t_tensor
            )
        )

    proto_pred = (
        torch.argmax(
            proto_logits,
            dim=1
        )
        .cpu()
        .numpy()
    )

    # ========================================================
    # CCD / UOT prediction
    # ========================================================
    stopThr = 1e-6

    newsim, fake_size = (
        adaptive_filling(
            torch.from_numpy(
                norm_feat_t
            ).cuda(),
            source_prototype,
            gamma,
            beta,
            0,
            stopThr=stopThr
        )
    )

    (
        _,
        __,
        pred_label,
        ___
    ) = ubot_CCD(
        newsim,
        beta,
        fake_size=fake_size,
        fill_size=0,
        mode="minibatch",
        stopThr=stopThr
    )

    pred_label = (
        pred_label
        .cpu()
        .data
        .numpy()
    )

    # ========================================================
    # Target-private samples
    # ========================================================
    private_mask = np.zeros(
        label_t.shape,
        dtype=bool
    )

    for i in range(
        label_t.size
    ):
        if (
            label_t[i]
            in classes_set["tp_classes"]
        ):
            private_mask[i] = True

    private_feat = (
        norm_feat_t[
            private_mask,
            :
        ]
    )

    private_label = (
        label_t[
            private_mask
        ]
    )

    private_proto_pred = (
        proto_pred[
            private_mask
        ]
    )

    # ========================================================
    # Before merge metrics
    # ========================================================

    # Prototype NMI
    prototype_tp_nmi = (
        normalized_mutual_info_score(
            private_label,
            private_proto_pred
        )
    )

    # Active prototype number
    active_private_prototypes = int(
        np.unique(
            private_proto_pred
        ).size
    )

    # Purity
    private_purity = (
        calculate_private_purity(
            private_label,
            private_proto_pred
        )
    )

    weighted_private_purity = (
        calculate_weighted_private_purity(
            private_label,
            private_proto_pred
        )
    )

    # Prototype number per class
    prototypes_per_class = (
        calculate_prototypes_per_class(
            private_label,
            private_proto_pred
        )
    )

    avg_prototypes_per_class = (
        calculate_average_prototypes_per_class(
            private_label,
            private_proto_pred
        )
    )

    # ========================================================
    # Q_tt based merge evaluation
    # ========================================================
    merged_results = None
    merged_prototypes_per_class = None

    if prototype_groups is not None:

        prototype_groups = np.asarray(
            prototype_groups
        )

        # prototype ID range check
        if (
            prototype_groups.ndim != 1
        ):
            raise ValueError(
                "prototype_groups must be "
                "a 1-dimensional array."
            )

        if (
            private_proto_pred.max()
            >= len(prototype_groups)
        ):
            raise ValueError(
                "prototype_groups size does not "
                "match prototype IDs."
            )

        # ----------------------------------------
        # Original prototype -> merged group
        # ----------------------------------------
        merged_private_pred = (
            apply_prototype_groups(
                private_proto_pred,
                prototype_groups
            )
        )

        # ----------------------------------------
        # Merged NMI
        # ----------------------------------------
        merged_prototype_tp_nmi = (
            normalized_mutual_info_score(
                private_label,
                merged_private_pred
            )
        )

        # ----------------------------------------
        # Active merged groups
        # ----------------------------------------
        merged_active_private_prototypes = int(
            np.unique(
                merged_private_pred
            ).size
        )

        # ----------------------------------------
        # Merged purity
        # ----------------------------------------
        merged_private_purity = (
            calculate_private_purity(
                private_label,
                merged_private_pred
            )
        )

        merged_weighted_private_purity = (
            calculate_weighted_private_purity(
                private_label,
                merged_private_pred
            )
        )

        # ----------------------------------------
        # Merged groups per class
        # ----------------------------------------
        merged_prototypes_per_class = (
            calculate_prototypes_per_class(
                private_label,
                merged_private_pred
            )
        )

        merged_avg_prototypes_per_class = (
            calculate_average_prototypes_per_class(
                private_label,
                merged_private_pred
            )
        )

        # ----------------------------------------
        # Total merged group number
        # ----------------------------------------
        merged_total_groups = int(
            np.unique(
                prototype_groups
            ).size
        )

        merged_results = {
            "merged_prototype_tp_nmi":
                merged_prototype_tp_nmi,

            "merged_total_groups":
                merged_total_groups,

            "merged_active_private_prototypes":
                merged_active_private_prototypes,

            "merged_private_purity":
                merged_private_purity,

            "merged_weighted_private_purity":
                merged_weighted_private_purity,

            "merged_avg_prototypes_per_class":
                merged_avg_prototypes_per_class
        }

    # ========================================================
    # Original UniOT K-means evaluation
    # ========================================================
    ncentroids = len(
        classes_set[
            "tp_classes"
        ]
    )

    private_pred, _ = (
        run_kmeans(
            private_feat,
            ncentroids,
            init_centroids=None,
            seed=seed,
            gpu=True
        )
    )

    results = ResultsCalculator(
        classes_set,
        label_t,
        pred_label,
        private_label,
        private_pred
    )

    results_dict = {
        "cls_common_acc":
            results.common_acc_aver,

        "cls_tp_acc":
            results.tp_acc,

        "tp_nmi":
            results.tp_nmi,

        "kmeans_tp_nmi":
            results.tp_nmi,

        "prototype_tp_nmi":
            prototype_tp_nmi,

        "active_private_prototypes":
            active_private_prototypes,

        "private_purity":
            private_purity,

        "weighted_private_purity":
            weighted_private_purity,

        "avg_prototypes_per_class":
            avg_prototypes_per_class,

        "cls_overall_acc":
            results.overall_acc_aver,

        "h_score":
            results.h_score,

        "h3_score":
            results.h3_score
    }

    if merged_results is not None:
        results_dict.update(
            merged_results
        )

    prototype_details = {
        "before_merge":
            prototypes_per_class,

        "after_merge":
            merged_prototypes_per_class
    }

    return (
        results_dict,
        prototype_details
    )


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":

    from data import *
    from utils.net import (
        ResNet50Fc,
        CLS,
        ProtoCLS
    )

    import torch.nn as nn

    # --------------------------------------------------------
    # JS divergence thresholds
    # --------------------------------------------------------
    js_thresholds = [
        0.01,
        0.02,
        0.05,
        0.10,
        0.15,
        0.20
    ]

    min_assignment_count = 5

    # --------------------------------------------------------
    # Check model path
    # --------------------------------------------------------
    if parser_args.model_path is None:
        raise ValueError(
            "NO model_path input!"
        )

    seed = 1234
    seed_everything(
        seed
    )

    # --------------------------------------------------------
    # GPU
    # --------------------------------------------------------
    if len(
        parser_args.gpu_index
    ) < 1:

        os.environ[
            "CUDA_VISIBLE_DEVICES"
        ] = ""

        gpu_ids = []

    else:

        os.environ[
            "CUDA_VISIBLE_DEVICES"
        ] = (
            parser_args.gpu_index
        )

        gpu_ids = list(
            map(
                int,
                parser_args.gpu_index
            )
        )

    # --------------------------------------------------------
    # Load checkpoint
    # --------------------------------------------------------
    data = torch.load(
        parser_args.model_path,
        map_location="cpu"
    )

    loaded_K = int(
        data["K"]
    )

    print(
        f"Loaded K = {loaded_K}"
    )

    # --------------------------------------------------------
    # Check Q information
    # --------------------------------------------------------
    if (
        "q_assignment_matrix"
        not in data
    ):
        raise ValueError(
            "このfinal.pklには"
            "q_assignment_matrixがありません。\n"
            "Q_tt収集処理を追加したmain.pyで"
            "再学習してください。"
        )

    q_assignment_data = (
        data[
            "q_assignment_matrix"
        ]
    )

    if torch.is_tensor(
        q_assignment_data
    ):

        q_assignment_matrix = (
            q_assignment_data
            .cpu()
            .numpy()
        )

    else:

        q_assignment_matrix = np.asarray(
            q_assignment_data
        )

    # --------------------------------------------------------
    # Q matrix check
    # --------------------------------------------------------
    if (
        q_assignment_matrix.ndim
        != 2
    ):
        raise ValueError(
            "q_assignment_matrix must have "
            "shape (N, K)."
        )

    if (
        q_assignment_matrix.shape[0]
        == 0
    ):
        raise ValueError(
            "q_assignment_matrix is empty."
        )

    if (
        q_assignment_matrix.shape[1]
        != loaded_K
    ):
        raise ValueError(
            "q_assignment_matrixのKと"
            "checkpointのKが一致していません。\n"
            f"matrix K = "
            f"{q_assignment_matrix.shape[1]}, "
            f"loaded K = {loaded_K}"
        )

    if not np.all(
        np.isfinite(
            q_assignment_matrix
        )
    ):
        raise ValueError(
            "q_assignment_matrix contains "
            "NaN or Inf."
        )

    print(
        "Q assignment matrix shape = "
        f"{q_assignment_matrix.shape}"
    )

    # --------------------------------------------------------
    # Network
    # --------------------------------------------------------
    cls_output_dim = len(
        source_classes
    )

    feat_dim = 256

    feature_extractor = (
        ResNet50Fc(
            pretrained_model_path
        )
    )

    classifier = CLS(
        feature_extractor.output_dim,
        cls_output_dim,
        hidden_mlp=2048,
        feat_dim=feat_dim,
        temp=temp
    )

    cluster_head = ProtoCLS(
        feat_dim,
        loaded_K,
        temp=temp
    )

    feature_extractor = (
        feature_extractor.cuda()
    )

    classifier = (
        classifier.cuda()
    )

    cluster_head = (
        cluster_head.cuda()
    )

    feature_extractor = (
        nn.DataParallel(
            feature_extractor
        ).train(False)
    )

    classifier = (
        nn.DataParallel(
            classifier
        ).train(False)
    )

    cluster_head = (
        nn.DataParallel(
            cluster_head
        ).train(False)
    )

    feature_extractor.load_state_dict(
        data[
            "feature_extractor"
        ]
    )

    classifier.load_state_dict(
        data[
            "classifier"
        ]
    )

    cluster_head.load_state_dict(
        data[
            "cluster_head"
        ]
    )

    # --------------------------------------------------------
    # beta
    # --------------------------------------------------------
    beta = data[
        "beta"
    ]

    if torch.is_tensor(
        beta
    ):
        beta = (
            beta
            .cpu()
            .numpy()
        )

    # --------------------------------------------------------
    # Save directory
    # --------------------------------------------------------
    save_dir = os.path.dirname(
        parser_args.model_path
    )

    # ========================================================
    # Before merge evaluation
    # ========================================================
    (
        results_before,
        prototype_details_before
    ) = eval(
        feature_extractor,
        classifier,
        cluster_head,
        target_test_dl,
        classes_set,
        gamma=gamma,
        beta=beta,
        prototype_groups=None
    )

    prototypes_per_class = (
        prototype_details_before[
            "before_merge"
        ]
    )

    print(
        "\n[Before Merge]"
    )

    print(
        results_before
    )

    # --------------------------------------------------------
    # Save baseline result
    # --------------------------------------------------------
    pd.DataFrame(
        [
            results_before
        ]
    ).to_csv(
        os.path.join(
            save_dir,
            "q_merge_before_result.csv"
        ),
        index=False
    )

    # --------------------------------------------------------
    # Save before prototypes per class
    # --------------------------------------------------------
    prototype_class_df = pd.DataFrame(
        [
            {
                "class_label":
                    class_label,

                "prototype_count":
                    prototype_count
            }
            for (
                class_label,
                prototype_count
            )
            in prototypes_per_class.items()
        ]
    )

    prototype_class_df.to_csv(
        os.path.join(
            save_dir,
            "prototypes_per_class_eval.csv"
        ),
        index=False
    )

    # ========================================================
    # Q-based merge experiment
    # ========================================================
    js_merge_result_rows = []

    js_distribution_saved = False
    assignment_counts_saved = False

    for js_threshold in js_thresholds:

        (
            prototype_groups,
            q_distance_matrix,
            assignment_counts
        ) = merge_prototypes_by_q_distribution(
            q_assignment_matrix,
            js_threshold=js_threshold,
            min_assignment_count=(
                min_assignment_count
            )
        )

        # ----------------------------------------------------
        # Assignment counts
        # ----------------------------------------------------
        valid_ids = np.where(
            assignment_counts
            >= min_assignment_count
        )[0]

        invalid_ids = np.where(
            assignment_counts
            < min_assignment_count
        )[0]

        if not assignment_counts_saved:

            assignment_count_df = (
                pd.DataFrame(
                    {
                        "prototype_id":
                            np.arange(
                                len(
                                    assignment_counts
                                )
                            ),

                        "assignment_count":
                            assignment_counts,

                        "is_valid_for_merge":
                            (
                                assignment_counts
                                >= min_assignment_count
                            )
                    }
                )
            )

            assignment_count_df.to_csv(
                os.path.join(
                    save_dir,
                    "q_assignment_counts.csv"
                ),
                index=False
            )

            assignment_counts_saved = True

        # ----------------------------------------------------
        # JS divergence distribution
        #
        # min_assignment_countを満たしたprototypeだけで確認
        # ----------------------------------------------------
        if (
            not js_distribution_saved
            and len(valid_ids) >= 2
        ):

            valid_distance_matrix = (
                q_distance_matrix[
                    np.ix_(
                        valid_ids,
                        valid_ids
                    )
                ]
            )

            valid_off_diagonal = (
                valid_distance_matrix[
                    ~np.eye(
                        len(valid_ids),
                        dtype=bool
                    )
                ]
            )

            print(
                "\n"
                "[Q Assignment JS Divergence]"
            )

            print(
                f"valid prototypes = "
                f"{len(valid_ids)}"
            )

            print(
                f"invalid prototypes = "
                f"{len(invalid_ids)}"
            )

            print(
                f"min    = "
                f"{valid_off_diagonal.min():.6f}"
            )

            print(
                f"p01    = "
                f"{np.percentile(valid_off_diagonal, 1):.6f}"
            )

            print(
                f"p05    = "
                f"{np.percentile(valid_off_diagonal, 5):.6f}"
            )

            print(
                f"p10    = "
                f"{np.percentile(valid_off_diagonal, 10):.6f}"
            )

            print(
                f"median = "
                f"{np.median(valid_off_diagonal):.6f}"
            )

            print(
                f"mean   = "
                f"{valid_off_diagonal.mean():.6f}"
            )

            print(
                f"max    = "
                f"{valid_off_diagonal.max():.6f}"
            )

            # JS分布もCSV保存
            pd.DataFrame(
                {
                    "js_divergence":
                        valid_off_diagonal
                }
            ).to_csv(
                os.path.join(
                    save_dir,
                    "q_js_divergence_distribution.csv"
                ),
                index=False
            )

            js_distribution_saved = True

        # ----------------------------------------------------
        # Total merged groups
        # ----------------------------------------------------
        total_groups = int(
            np.unique(
                prototype_groups
            ).size
        )

        num_reduced_groups = (
            loaded_K
            - total_groups
        )

        print(
            "\n"
            f"JS threshold="
            f"{js_threshold:.3f}, "
            f"groups="
            f"{total_groups}, "
            f"reduced="
            f"{num_reduced_groups}"
        )

        # ----------------------------------------------------
        # Evaluate merged groups
        # ----------------------------------------------------
        (
            merged_results,
            merged_details
        ) = eval(
            feature_extractor,
            classifier,
            cluster_head,
            target_test_dl,
            classes_set,
            gamma=gamma,
            beta=beta,
            prototype_groups=(
                prototype_groups
            )
        )

        # ----------------------------------------------------
        # Save summary row
        # ----------------------------------------------------
        js_merge_result_rows.append(
            {
                "js_threshold":
                    js_threshold,

                "min_assignment_count":
                    min_assignment_count,

                "valid_prototypes":
                    len(valid_ids),

                "before_total_k":
                    loaded_K,

                "before_nmi":
                    merged_results[
                        "prototype_tp_nmi"
                    ],

                "before_active_k":
                    merged_results[
                        "active_private_prototypes"
                    ],

                "before_purity":
                    merged_results[
                        "private_purity"
                    ],

                "before_weighted_purity":
                    merged_results[
                        "weighted_private_purity"
                    ],

                "before_avg_proto_per_class":
                    merged_results[
                        "avg_prototypes_per_class"
                    ],

                "after_nmi":
                    merged_results[
                        "merged_prototype_tp_nmi"
                    ],

                "after_total_groups":
                    merged_results[
                        "merged_total_groups"
                    ],

                "num_reduced_groups":
                    num_reduced_groups,

                "after_active_k":
                    merged_results[
                        "merged_active_private_prototypes"
                    ],

                "after_purity":
                    merged_results[
                        "merged_private_purity"
                    ],

                "after_weighted_purity":
                    merged_results[
                        "merged_weighted_private_purity"
                    ],

                "after_avg_proto_per_class":
                    merged_results[
                        "merged_avg_prototypes_per_class"
                    ]
            }
        )

        # ----------------------------------------------------
        # Save per-class merged group number
        # ----------------------------------------------------
        after_merge_per_class = (
            merged_details[
                "after_merge"
            ]
        )

        if after_merge_per_class is None:
            raise RuntimeError(
                "after_merge is None. "
                "prototype_groups was not "
                "applied correctly."
            )

        pd.DataFrame(
            [
                {
                    "class_label":
                        class_label,

                    "merged_group_count":
                        group_count
                }
                for (
                    class_label,
                    group_count
                )
                in after_merge_per_class.items()
            ]
        ).to_csv(
            os.path.join(
                save_dir,
                f"q_merge_prototypes_per_class_"
                f"js_{js_threshold:.2f}.csv"
            ),
            index=False
        )

    # ========================================================
    # Save all threshold results
    # ========================================================
    js_merge_result_df = pd.DataFrame(
        js_merge_result_rows
    )

    js_merge_result_df.to_csv(
        os.path.join(
            save_dir,
            "q_assignment_merge_results.csv"
        ),
        index=False
    )

    print(
        "\nQ-based prototype merge "
        "evaluation finished."
    )

    print(
        "Saved to:"
    )

    print(
        os.path.join(
            save_dir,
            "q_assignment_merge_results.csv"
        )
    )