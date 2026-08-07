import numpy as np
import torch
import torch.nn.functional as F

from sklearn.cluster import AgglomerativeClustering


def get_prototype_similarity(cluster_head):
    """
    target prototype間のコサイン類似度を計算する。

    Returns
    -------
    similarity_matrix : np.ndarray
        shape = (K, K)
    """

    # DataParallelを使用している場合
    if hasattr(cluster_head, "module"):
        prototypes = cluster_head.module.fc.weight.detach()
    else:
        prototypes = cluster_head.fc.weight.detach()

    prototypes = F.normalize(
        prototypes,
        dim=1
    )

    similarity_matrix = torch.matmul(
        prototypes,
        prototypes.t()
    )

    return similarity_matrix.cpu().numpy()


def merge_prototypes_by_threshold(
    cluster_head,
    cosine_threshold
):
    """
    プロトタイプ間コサイン類似度を基準に
    階層的クラスタリングによって統合する。

    Parameters
    ----------
    cluster_head
        UniOTのtarget prototype head

    cosine_threshold : float
        例: 0.90

    Returns
    -------
    prototype_groups : np.ndarray
        各prototypeが所属する統合後group番号

    similarity_matrix : np.ndarray
        prototype間コサイン類似度
    """

    similarity_matrix = get_prototype_similarity(
        cluster_head
    )

    similarity_matrix = np.clip(
        similarity_matrix,
        -1.0,
        1.0
    )

    distance_matrix = (
        1.0 - similarity_matrix
    )

    distance_matrix = (
        distance_matrix
        + distance_matrix.T
    ) / 2.0

    np.fill_diagonal(
        distance_matrix,
        0.0
    )

    distance_threshold = (
        1.0 - cosine_threshold
    )

    # sklearn新バージョン
    try:
        merge_model = AgglomerativeClustering(
            n_clusters=None,
            metric="precomputed",
            linkage="average",
            distance_threshold=distance_threshold
        )

    # sklearn旧バージョン
    except TypeError:
        merge_model = AgglomerativeClustering(
            n_clusters=None,
            affinity="precomputed",
            linkage="average",
            distance_threshold=distance_threshold
        )

    prototype_groups = merge_model.fit_predict(
        distance_matrix
    )

    return (
        prototype_groups,
        similarity_matrix
    )


def apply_prototype_groups(
    proto_pred,
    prototype_groups
):
    """
    元prototype番号を統合後group番号へ変換する。
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