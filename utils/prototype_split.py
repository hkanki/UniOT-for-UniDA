import numpy as np

from sklearn.cluster import KMeans
from scipy.spatial.distance import jensenshannon


def calculate_js_divergence(
    p,
    q,
    eps=1e-12
):
    """
    2つのQ分布間のJensen-Shannon divergenceを計算する。
    """

    p = np.asarray(
        p,
        dtype=np.float64
    )

    q = np.asarray(
        q,
        dtype=np.float64
    )

    p = np.clip(
        p,
        eps,
        None
    )

    q = np.clip(
        q,
        eps,
        None
    )

    # 確率分布になるように正規化
    p = p / p.sum()
    q = q / q.sum()

    # scipyのjensenshannonは
    # sqrt(JSD)を返すため2乗する
    js_divergence = (
        jensenshannon(
            p,
            q
        ) ** 2
    )

    return float(
        js_divergence
    )


def calculate_within_dispersion(
    q_vectors
):
    """
    1つのprototypeに属するQベクトル群の
    内部ばらつきを計算する。

    これが W_k_before に相当する。
    """

    q_vectors = np.asarray(
        q_vectors,
        dtype=np.float64
    )

    if q_vectors.shape[0] == 0:
        return np.nan

    # Qベクトルの平均
    mean_q = np.mean(
        q_vectors,
        axis=0
    )

    mean_q = (
        mean_q
        / np.clip(
            mean_q.sum(),
            1e-12,
            None
        )
    )

    distances = []

    for q in q_vectors:

        distance = (
            calculate_js_divergence(
                q,
                mean_q
            )
        )

        distances.append(
            distance
        )

    return float(
        np.mean(distances)
    )


def split_q_vectors(
    q_vectors,
    random_state=1234
):
    """
    Qベクトルを仮に2群へ分割する。
    """

    q_vectors = np.asarray(
        q_vectors,
        dtype=np.float64
    )

    if q_vectors.shape[0] < 2:
        return None

    kmeans = KMeans(
        n_clusters=2,
        random_state=random_state,
        n_init=10
    )

    split_labels = (
        kmeans.fit_predict(
            q_vectors
        )
    )

    return split_labels


def calculate_split_dispersion(
    q_vectors,
    split_labels
):
    """
    2群に分割した後の群内ばらつきを計算する。

    これが W_k_split に相当する。
    """

    q_vectors = np.asarray(
        q_vectors,
        dtype=np.float64
    )

    total_distance = 0.0
    total_samples = 0

    for cluster_id in [0, 1]:

        mask = (
            split_labels
            == cluster_id
        )

        cluster_q = (
            q_vectors[
                mask
            ]
        )

        if cluster_q.shape[0] == 0:
            continue

        cluster_dispersion = (
            calculate_within_dispersion(
                cluster_q
            )
        )

        total_distance += (
            cluster_dispersion
            * cluster_q.shape[0]
        )

        total_samples += (
            cluster_q.shape[0]
        )

    if total_samples == 0:
        return np.nan

    return float(
        total_distance
        / total_samples
    )


def calculate_split_improvement(
    q_vectors,
    eps=1e-12,
    random_state=1234
):
    """
    prototype k の混在度 R_k を計算する。

    R_k =
        (W_before - W_split)
        / (W_before + eps)
    """

    q_vectors = np.asarray(
        q_vectors,
        dtype=np.float64
    )

    if q_vectors.shape[0] < 2:

        return {
            "before_dispersion":
                np.nan,

            "split_dispersion":
                np.nan,

            "split_improvement":
                np.nan
        }

    # 分割前
    before_dispersion = (
        calculate_within_dispersion(
            q_vectors
        )
    )

    # 仮に2分割
    split_labels = (
        split_q_vectors(
            q_vectors,
            random_state=random_state
        )
    )

    # 分割後
    split_dispersion = (
        calculate_split_dispersion(
            q_vectors,
            split_labels
        )
    )

    # R_k
    split_improvement = (
        before_dispersion
        - split_dispersion
    ) / (
        before_dispersion
        + eps
    )

    return {
        "before_dispersion":
            float(
                before_dispersion
            ),

        "split_dispersion":
            float(
                split_dispersion
            ),

        "split_improvement":
            float(
                split_improvement
            )
    }


def analyze_prototype_splits(
    q_assignment_matrix,
    min_samples=10,
    random_state=1234
):
    """
    全prototypeについて R_k を計算する。
    """

    q_assignment_matrix = np.asarray(
        q_assignment_matrix,
        dtype=np.float64
    )

    if q_assignment_matrix.ndim != 2:
        raise ValueError(
            "q_assignment_matrix must be 2-dimensional."
        )

    # 各sampleをQ最大のprototypeへhard assignment
    hard_assignment = np.argmax(
        q_assignment_matrix,
        axis=1
    )

    num_prototypes = (
        q_assignment_matrix.shape[1]
    )

    results = []

    for proto_id in range(
        num_prototypes
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

        # サンプル数が少なすぎる場合は
        # split判定を行わない
        if sample_count < min_samples:

            results.append(
                {
                    "prototype_id":
                        proto_id,

                    "sample_count":
                        sample_count,

                    "before_dispersion":
                        np.nan,

                    "split_dispersion":
                        np.nan,

                    "split_improvement":
                        np.nan
                }
            )

            continue

        result = (
            calculate_split_improvement(
                q_vectors,
                random_state=random_state
            )
        )

        results.append(
            {
                "prototype_id":
                    proto_id,

                "sample_count":
                    sample_count,

                "before_dispersion":
                    result[
                        "before_dispersion"
                    ],

                "split_dispersion":
                    result[
                        "split_dispersion"
                    ],

                "split_improvement":
                    result[
                        "split_improvement"
                    ]
            }
        )

    return results

def analyze_true_category_mixing(
    q_assignment_matrix,
    q_sample_ids,
    target_labels,
    tp_classes
):
    """
    Q_ttのhard assignmentに基づいて、
    各prototypeに含まれるtarget-privateカテゴリの
    実際の混在状態を正解ラベルで評価する。

    正解ラベルは評価にのみ使用する。
    """

    q_assignment_matrix = np.asarray(
        q_assignment_matrix,
        dtype=np.float64
    )

    q_sample_ids = np.asarray(
        q_sample_ids,
        dtype=np.int64
    )

    target_labels = np.asarray(
        target_labels
    )

    tp_classes = set(
        int(x)
        for x in tp_classes
    )


    if (
        q_assignment_matrix.shape[0]
        != q_sample_ids.shape[0]
    ):
        raise ValueError(
            "Number of Q vectors and sample IDs does not match."
        )


    if len(q_sample_ids) == 0:
        return []


    if (
        np.max(q_sample_ids)
        >= len(target_labels)
    ):
        raise ValueError(
            "q_sample_ids exceeds target label size."
        )


    # Q最大のprototype
    hard_assignment = np.argmax(
        q_assignment_matrix,
        axis=1
    )


    # sample IDに対応する正解ラベル
    true_labels = target_labels[
        q_sample_ids
    ]


    num_prototypes = (
        q_assignment_matrix.shape[1]
    )

    results = []


    for proto_id in range(
        num_prototypes
    ):

        proto_mask = (
            hard_assignment
            == proto_id
        )

        proto_labels = (
            true_labels[
                proto_mask
            ]
        )


        # target-privateだけ抽出
        private_labels = np.asarray(
            [
                int(label)
                for label
                in proto_labels
                if int(label)
                in tp_classes
            ],
            dtype=np.int64
        )


        tp_sample_count = int(
            private_labels.size
        )


        if tp_sample_count == 0:

            results.append(
                {
                    "prototype_id":
                        proto_id,

                    "tp_sample_count":
                        0,

                    "num_categories":
                        0,

                    "majority_category":
                        np.nan,

                    "purity":
                        np.nan,

                    "categories":
                        "",

                    "category_counts":
                        ""
                }
            )

            continue


        unique_labels, counts = np.unique(
            private_labels,
            return_counts=True
        )


        num_categories = int(
            unique_labels.size
        )


        majority_index = int(
            np.argmax(
                counts
            )
        )


        majority_category = int(
            unique_labels[
                majority_index
            ]
        )


        purity = float(
            counts[
                majority_index
            ]
            / counts.sum()
        )


        categories = ",".join(
            str(int(x))
            for x
            in unique_labels
        )


        category_counts = ",".join(
            f"{int(label)}:{int(count)}"
            for label, count
            in zip(
                unique_labels,
                counts
            )
        )


        results.append(
            {
                "prototype_id":
                    proto_id,

                "tp_sample_count":
                    tp_sample_count,

                "num_categories":
                    num_categories,

                "majority_category":
                    majority_category,

                "purity":
                    purity,

                "categories":
                    categories,

                "category_counts":
                    category_counts
            }
        )


    return results