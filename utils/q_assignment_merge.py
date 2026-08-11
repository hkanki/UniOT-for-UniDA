import numpy as np

from sklearn.cluster import AgglomerativeClustering


def calculate_js_divergence(
    p,
    q,
    eps=1e-12
):
    """
    2つの確率分布間のJensen-Shannon divergenceを計算する。

    0に近い:
        2つのprototypeが似たtarget sampleを担当

    大きい:
        担当sampleが異なる
    """

    p = np.asarray(
        p,
        dtype=np.float64
    )

    q = np.asarray(
        q,
        dtype=np.float64
    )

    p = p + eps
    q = q + eps

    p = p / p.sum()
    q = q / q.sum()

    m = 0.5 * (
        p + q
    )

    kl_pm = np.sum(
        p * np.log2(
            p / m
        )
    )

    kl_qm = np.sum(
        q * np.log2(
            q / m
        )
    )

    js = 0.5 * (
        kl_pm + kl_qm
    )

    return float(js)


def calculate_q_distance_matrix(
    q_assignment_matrix
):
    """
    Q割当行列からprototype間JS divergence行列を作る。

    q_assignment_matrix:
        shape = (N, K)

        N:
            記録したtarget sample数

        K:
            target prototype数
    """

    q_assignment_matrix = np.asarray(
        q_assignment_matrix,
        dtype=np.float64
    )

    N, K = (
        q_assignment_matrix.shape
    )

    # --------------------------------
    # prototypeごとに
    # sample方向へ確率分布化する
    # --------------------------------
    prototype_distribution = (
        q_assignment_matrix.copy()
    )

    column_sum = (
        prototype_distribution.sum(
            axis=0,
            keepdims=True
        )
    )

    column_sum = np.maximum(
        column_sum,
        1e-12
    )

    prototype_distribution = (
        prototype_distribution
        / column_sum
    )

    # --------------------------------
    # pairwise JS divergence
    # --------------------------------
    distance_matrix = np.zeros(
        (K, K),
        dtype=np.float64
    )

    for i in range(K):

        for j in range(
            i + 1,
            K
        ):

            distance = (
                calculate_js_divergence(
                    prototype_distribution[:, i],
                    prototype_distribution[:, j]
                )
            )

            distance_matrix[i, j] = (
                distance
            )

            distance_matrix[j, i] = (
                distance
            )

    np.fill_diagonal(
        distance_matrix,
        0.0
    )

    return distance_matrix


def get_q_assignment_counts(
    q_assignment_matrix
):
    """
    平均Qでargmaxした場合に、
    各prototypeを担当するsample数を数える。
    """

    hard_assignment = np.argmax(
        q_assignment_matrix,
        axis=1
    )

    K = (
        q_assignment_matrix.shape[1]
    )

    counts = np.bincount(
        hard_assignment,
        minlength=K
    )

    return counts


def merge_prototypes_by_q_distribution(
    q_assignment_matrix,
    js_threshold,
    min_assignment_count=5
):
    """
    Q割当分布に基づきprototypeを統合する。

    JS divergenceが小さいprototypeほど
    同じtarget sample群を担当しているとみなす。

    Parameters
    ----------
    q_assignment_matrix:
        shape=(N,K)

    js_threshold:
        JS divergenceの統合閾値

    min_assignment_count:
        統合判断に利用するために必要な
        最低担当sample数
    """

    q_assignment_matrix = np.asarray(
        q_assignment_matrix,
        dtype=np.float64
    )

    K = (
        q_assignment_matrix.shape[1]
    )

    # Q分布間距離
    distance_matrix = (
        calculate_q_distance_matrix(
            q_assignment_matrix
        )
    )

    # 各prototypeの担当数
    assignment_counts = (
        get_q_assignment_counts(
            q_assignment_matrix
        )
    )

    # 十分なsampleを持つprototype
    valid_mask = (
        assignment_counts
        >= min_assignment_count
    )

    valid_ids = np.where(
        valid_mask
    )[0]

    # 最初は全prototypeを別groupにする
    prototype_groups = np.arange(
        K,
        dtype=np.int64
    )

    # 統合できるprototypeが2個未満
    if len(valid_ids) < 2:

        return (
            prototype_groups,
            distance_matrix,
            assignment_counts
        )

    valid_distance_matrix = (
        distance_matrix[
            np.ix_(
                valid_ids,
                valid_ids
            )
        ]
    )

    try:

        model = AgglomerativeClustering(
            n_clusters=None,
            metric="precomputed",
            linkage="average",
            distance_threshold=js_threshold
        )

    except TypeError:

        model = AgglomerativeClustering(
            n_clusters=None,
            affinity="precomputed",
            linkage="average",
            distance_threshold=js_threshold
        )

    valid_group_labels = (
        model.fit_predict(
            valid_distance_matrix
        )
    )

    # group番号を振り直す
    next_group = 0

    group_map = {}

    for proto_id, local_group in zip(
        valid_ids,
        valid_group_labels
    ):

        if local_group not in group_map:

            group_map[
                local_group
            ] = next_group

            next_group += 1

        prototype_groups[
            proto_id
        ] = group_map[
            local_group
        ]

    # sample不足prototypeは統合せず
    # それぞれ独立groupにする
    invalid_ids = np.where(
        ~valid_mask
    )[0]

    for proto_id in invalid_ids:

        prototype_groups[
            proto_id
        ] = next_group

        next_group += 1

    return (
        prototype_groups,
        distance_matrix,
        assignment_counts
    )


def apply_prototype_groups(
    proto_pred,
    prototype_groups
):
    """
    元prototype番号を統合group番号へ変換する。
    """

    proto_pred = np.asarray(
        proto_pred
    )

    prototype_groups = np.asarray(
        prototype_groups
    )

    return prototype_groups[
        proto_pred
    ]