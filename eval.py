from easydl import variable_to_numpy
from easydl import TrainingModeManager, Accumulator
import numpy as np
from utils.lib import seed_everything, ubot_CCD, adaptive_filling
import torch
import torch.nn.functional as F
from tqdm import tqdm
import ot
import faiss
import os



# 変更前
# from utils.util import ResultsCalculator
# 変更後
from utils.util import (
    ResultsCalculator,
    calculate_private_purity,
    calculate_weighted_private_purity,
    calculate_prototypes_per_class,
    calculate_average_prototypes_per_class
)

# 新たに追加
import pandas as pd
from sklearn.metrics import normalized_mutual_info_score
# 第二段階にて追加した
from utils.prototype_merge import (
    merge_prototypes_by_threshold,
    apply_prototype_groups
)

def run_kmeans(L2_feat, ncentroids, init_centroids=None, seed=None, gpu=False, min_points_per_centroid=1):
    if seed is None:
        seed = int(os.environ['PYTHONHASHSEED'])
    dim = L2_feat.shape[1]
    kmeans = faiss.Kmeans(d=dim, k=ncentroids, seed=seed, gpu=gpu, niter=20, verbose=False, \
                        nredo=5, min_points_per_centroid=min_points_per_centroid, spherical=True)
    if torch.is_tensor(L2_feat):
        L2_feat = variable_to_numpy(L2_feat)
    kmeans.train(L2_feat, init_centroids=init_centroids)
    _, pred_centroid = kmeans.index.search(L2_feat, 1)
    pred_centroid = np.squeeze(pred_centroid)
    return pred_centroid, kmeans.centroids

# def eval(feature_extractor, classifier, cluster_head,eval_dl, classes_set, 
#         gamma=0.7, beta=None, seed=None, uniformed_index=None):
# 第二段階にて変更
def eval( feature_extractor, classifier, cluster_head, eval_dl, classes_set, 
         gamma=0.7, beta=None, seed=None, uniformed_index=None, merge_threshold=None ):
        
    if seed is None:
        seed = int(os.environ['PYTHONHASHSEED'])
    if uniformed_index is None:
        uniformed_index = len(classes_set['source_classes'])
    # source_prototypeの位置を変更
    source_prototype = classifier.module.ProtoCLS.fc.weight
    if beta is None:
        beta = ot.unif(source_prototype.size()[0])


    # cluster_headの追加
    with TrainingModeManager([feature_extractor, classifier, cluster_head], train=False) as mgr, \
            Accumulator(['label_t', 'norm_feat_t']) as eval_accumulator, \
            torch.no_grad():
        for i, (im_t, label_t) in enumerate(tqdm(eval_dl, desc='testing')):
            im_t = im_t.cuda()
            label_t = label_t.cuda()
            feature_ex_t = feature_extractor.forward(im_t)
            before_lincls_feat_t, after_lincls_t = classifier(feature_ex_t)
            norm_feat_t = F.normalize(before_lincls_feat_t)

            val = dict()
            for name in eval_accumulator.names:
                val[name] = locals()[name].cpu().data.numpy()
            eval_accumulator.updateData(val)  

    for x in eval_accumulator:
        val[x] = eval_accumulator[x] 
    label_t = val['label_t']
    norm_feat_t = val['norm_feat_t']
    del val
    
    # obtain target-prototype prediction
    with torch.no_grad():
        norm_feat_t_tensor = torch.from_numpy(
            norm_feat_t
        ).float().cuda()

        proto_logits = cluster_head(norm_feat_t_tensor)

    proto_pred = torch.argmax(
        proto_logits,
        dim=1
    ).cpu().numpy()

    # Unbalanced OT
    #source_prototype = classifier.module.ProtoCLS.fc.weight

    stopThr = 1e-6
    # Adaptive filling 
    newsim, fake_size = adaptive_filling(torch.from_numpy(norm_feat_t).cuda(), 
                                        source_prototype, gamma, beta, 0, stopThr=stopThr)

    # obtain predict label
    _, __, pred_label, ___ = ubot_CCD(newsim, beta, fake_size=fake_size, fill_size=0, mode='minibatch', stopThr=stopThr)
    pred_label = pred_label.cpu().data.numpy()

    # obtain private samples
    filter = (lambda x: x in classes_set["tp_classes"])
    private_mask = np.zeros((label_t.size,), dtype=bool) 
    for i in range(label_t.size):
        if filter(label_t[i]):
            private_mask[i] = True
    private_feat = norm_feat_t[private_mask, :]
    private_label = label_t[private_mask]

    # 統合前(従来手法の場合)
    # 追加された新たな指標
    # cluster_headによる全ターゲットサンプルの割当
    private_proto_pred = proto_pred[private_mask]

    # 1. プロトタイプ割当NMI
    prototype_tp_nmi = normalized_mutual_info_score(
        private_label,
        private_proto_pred
    )

    # 2. Active Prototype数
    active_private_prototypes = int(
        np.unique(private_proto_pred).size
    )

    # 3. PrivatePurity
    private_purity = calculate_private_purity(
        private_label,
        private_proto_pred
    )

    # 4. Weighted PrivatePurity
    weighted_private_purity = calculate_weighted_private_purity(
        private_label,
        private_proto_pred
    )

    # 5. 各カテゴリの割当プロトタイプ数
    prototypes_per_class = calculate_prototypes_per_class(
        private_label,
        private_proto_pred
    )

    # 1カテゴリ当たりの平均プロトタイプ数
    avg_prototypes_per_class = calculate_average_prototypes_per_class(
        private_label,
        private_proto_pred
    )
    
    # 統合後(提案手法で追加しようとしている統合手法)
    # 第二段階にて追加
    # ==========================================
    # Stage 2: Post-hoc Prototype Merge
    # ==========================================

    merged_results = None
    merged_prototypes_per_class = None

    if merge_threshold is not None:

        # prototypeを階層的クラスタリングで統合
        prototype_groups, similarity_matrix = (
            merge_prototypes_by_threshold(
                cluster_head,
                merge_threshold
            )
        )

        # private sampleのprototype番号を
        # merged group番号へ変換
        merged_private_pred = (
            apply_prototype_groups(
                private_proto_pred,
                prototype_groups
            )
        )

        # ------------------------------
        # 統合後NMI
        # ------------------------------
        merged_prototype_tp_nmi = (
            normalized_mutual_info_score(
                private_label,
                merged_private_pred
            )
        )

        # ------------------------------
        # 統合後active group数
        # ------------------------------
        merged_active_prototypes = int(
            np.unique(
                merged_private_pred
            ).size
        )

        # ------------------------------
        # 統合後Purity
        # ------------------------------
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

        # ------------------------------
        # 統合後prototype/group per class
        # ------------------------------
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

        # 元のK個から何groupになったか
        total_merged_groups = int(
            np.unique(
                prototype_groups
            ).size
        )

        merged_results = {
            "merge_threshold":
                float(merge_threshold),

            "merged_total_groups":
                total_merged_groups,

            "merged_active_private_prototypes":
                merged_active_prototypes,

            "merged_prototype_tp_nmi":
                merged_prototype_tp_nmi,

            "merged_private_purity":
                merged_private_purity,

            "merged_weighted_private_purity":
                merged_weighted_private_purity,

            "merged_avg_prototypes_per_class":
                merged_avg_prototypes_per_class,
        }
        
    # obtain results
    ncentroids = len(classes_set["tp_classes"])
    private_pred, _ = run_kmeans(private_feat, ncentroids, init_centroids=None, seed=seed, gpu=True)
    results = ResultsCalculator(classes_set, label_t, pred_label, private_label, private_pred)
    # 変更前
    # results_dict = {
    #     'cls_common_acc': results.common_acc_aver,
    #     'cls_tp_acc': results.tp_acc,
    #     'tp_nmi': results.tp_nmi,
    #     'cls_overall_acc': results.overall_acc_aver,
    #     'h_score': results.h_score,
    #     'h3_score': results.h3_score
    # }
    
    results_dict = {
        'cls_common_acc': results.common_acc_aver,
        'cls_tp_acc': results.tp_acc,

        # 従来名を維持
        'tp_nmi': results.tp_nmi,
        'kmeans_tp_nmi': results.tp_nmi,

        'prototype_tp_nmi': prototype_tp_nmi,
        'active_private_prototypes': active_private_prototypes,
        'private_purity': private_purity,
        'weighted_private_purity': weighted_private_purity,
        'avg_prototypes_per_class': avg_prototypes_per_class,

        'cls_overall_acc': results.overall_acc_aver,
        'h_score': results.h_score,
        'h3_score': results.h3_score
    }
    
    if merged_results is not None:
        results_dict.update(
            merged_results
        )
    # 第2段階
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


if __name__ == '__main__':
    from data import *
    from utils.net import ResNet50Fc, CLS, ProtoCLS
    import torch.nn as nn

    if parser_args.model_path is None:
        raise ValueError('NO model_path input!')

    seed = 1234
    seed_everything(seed)

    if len(parser_args.gpu_index) < 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        gpu_ids = []
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = (
            parser_args.gpu_index
        )
        gpu_ids = list(
            map(int, parser_args.gpu_index)
        )

    # 先にcheckpointを読み込む
    data = torch.load(
        parser_args.model_path,
        map_location='cpu'
    )

    loaded_K = int(data['K'])

    cls_output_dim = len(source_classes)
    feat_dim = 256

    feature_extractor = ResNet50Fc(
        pretrained_model_path
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

    feature_extractor = feature_extractor.cuda()
    classifier = classifier.cuda()
    cluster_head = cluster_head.cuda()

    feature_extractor = nn.DataParallel(
        feature_extractor
    ).train(False)

    classifier = nn.DataParallel(
        classifier
    ).train(False)

    cluster_head = nn.DataParallel(
        cluster_head
    ).train(False)

    feature_extractor.load_state_dict(
        data['feature_extractor']
    )
    classifier.load_state_dict(
        data['classifier']
    )
    cluster_head.load_state_dict(
        data['cluster_head']
    )

    beta = data['beta']

    if torch.is_tensor(beta):
        beta = beta.cpu().numpy()

        # 第二段階にて変更
        # ==========================================
        # 統合前の最終評価
        # ==========================================

    # 保存先は最初に決める
    save_dir = os.path.dirname(
        parser_args.model_path
    )

    # ==========================================
    # 統合前
    # ==========================================
    results_before, prototype_details_before = eval(
        feature_extractor,
        classifier,
        cluster_head,
        target_test_dl,
        classes_set,
        gamma=gamma,
        beta=beta,
        merge_threshold=None
    )

    # 統合前結果保存
    pd.DataFrame(
        [results_before]
    ).to_csv(
        os.path.join(
            save_dir,
            "before_merge_result.csv"
        ),
        index=False
    )

    prototypes_per_class = (
        prototype_details_before[
            "before_merge"
        ]
    )

    # ==========================================
    # Threshold experiments
    # ==========================================
    merge_thresholds = [
        0.70,
        0.75,
        0.80,
        0.85,
        0.90,
        0.95
    ]

    merge_result_rows = []

    for merge_threshold in merge_thresholds:

        print(
            f"Post-hoc prototype merge: "
            f"threshold={merge_threshold:.2f}"
        )

        merged_results, merged_details = eval(
            feature_extractor,
            classifier,
            cluster_head,
            target_test_dl,
            classes_set,
            gamma=gamma,
            beta=beta,
            merge_threshold=merge_threshold
        )

        merge_result_rows.append(
            {
                "threshold":
                    merge_threshold,

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
                    ],
            }
        )

        # 各カテゴリの統合後グループ数も保存
        after_merge_per_class = (
            merged_details[
                "after_merge"
            ]
        )

        pd.DataFrame(
            [
                {
                    "class_label":
                        class_label,
                    "merged_group_count":
                        group_count,
                }
                for class_label, group_count
                in after_merge_per_class.items()
            ]
        ).to_csv(
            os.path.join(
                save_dir,
                f"merged_prototypes_per_class_"
                f"threshold_{merge_threshold:.2f}.csv"
            ),
            index=False
        )


    # 全閾値の結果
    merge_result_df = pd.DataFrame(
        merge_result_rows
    )

    merge_result_df.to_csv(
        os.path.join(
            save_dir,
            "prototype_merge_results.csv"
        ),
        index=False
    )
    # print(results)
    # print(prototypes_per_class)

    prototype_class_df = pd.DataFrame(
        [
            {
                'class_label': class_label,
                'prototype_count': prototype_count
            }
            for class_label, prototype_count
            in prototypes_per_class.items()
        ]
    )

    prototype_class_df.to_csv(
        os.path.join(
            save_dir,
            'prototypes_per_class_eval.csv'
        ),
        index=False
    )

