from data import *
from eval import eval
# from utils.net import ResNet50Fc, ProtoCLS, CLS
from utils.net import ( ResNet50Fc, ProtoCLS, DynamicProtoCLS, CLS )
from utils.lib import seed_everything, sinkhorn, ubot_CCD, adaptive_filling
from utils.visualization import draw_tsne
from utils.util import MemoryQueue
from easydl import inverseDecaySheduler, OptimWithSheduler, TrainingModeManager, OptimizerManager, AccuracyCounter
from easydl import one_hot, variable_to_numpy, clear_output
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm
from tensorboardX import SummaryWriter
import pandas as pd
import ot
import os
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
cudnn.deterministic = True

# 提案手法で追加した関数
import numpy as np

from utils.prototype_split import (
    analyze_prototype_splits,
    find_mixed_prototypes,
    build_split_children
)
seed = 1234
seed_everything(seed)

if len(parser_args.gpu_index) < 1:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    gpu_ids = []
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = parser_args.gpu_index
    gpu_ids = list(map(int, parser_args.gpu_index))

log_dir = f'{log_path}'
logger = SummaryWriter(log_dir)

# define network architecture
cls_output_dim = len(source_classes)
feat_dim = 256
feature_extractor = ResNet50Fc(pretrained_model_path)
classifier = CLS(feature_extractor.output_dim, cls_output_dim, hidden_mlp=2048, feat_dim=256, temp=temp)
# cluster_head = ProtoCLS(feat_dim, K, temp=temp)
if use_prototype_split:

    cluster_head = DynamicProtoCLS(
        feat_dim,
        K,
        temp=temp
    )

else:

    # 従来UniOT
    cluster_head = ProtoCLS(
        feat_dim,
        K,
        temp=temp
    )
    
def register_new_prototype_parameter(
    optimizer,
    new_parameter
):
    """
    splitによって新規生成したprototype parameterを
    既存optimizerに追加する。

    既存prototypeのmomentum等は維持される。

    注意: easydlのOptimWithSheduler.step()は各param_groupの
    "initial_lr"を参照してlrをスケジューリングし直すため、
    add_param_groupする際は必ず"initial_lr"も設定する。
    ("initial_lr"が無いと次回のscheduler.step()でKeyErrorになる)
    既存param_group同様、cluster_headの基準lr(args.train.lr)を
    initial_lrとして使うことで、他のprototypeと同じ減衰曲線に従う。
    """

    base_lr = (
        args.train.lr
    )


    optimizer.add_param_group(
        {
            "params": [
                new_parameter
            ],

            "lr":
                base_lr,

            "initial_lr":
                base_lr
        }
    )

feature_extractor = feature_extractor.cuda()
classifier = classifier.cuda()
cluster_head = cluster_head.cuda()

optimizer_featex = optim.SGD(feature_extractor.parameters(), lr=args.train.lr*0.1, weight_decay=args.train.weight_decay, momentum=args.train.sgd_momentum, nesterov=True)
optimizer_cls = optim.SGD(classifier.parameters(), lr=args.train.lr, weight_decay=args.train.weight_decay, momentum=args.train.sgd_momentum, nesterov=True)
optimizer_cluhead = optim.SGD(cluster_head.parameters(), lr=args.train.lr, weight_decay=args.train.weight_decay, momentum=args.train.sgd_momentum, nesterov=True)

# learning rate decay
scheduler = lambda step, initial_lr: inverseDecaySheduler(step, initial_lr, gamma=10, power=0.75, max_iter=args.train.min_step)
opt_sche_featex = OptimWithSheduler(optimizer_featex,scheduler)
opt_sche_cls = OptimWithSheduler(optimizer_cls,scheduler)
opt_sche_cluhead = OptimWithSheduler(optimizer_cluhead,scheduler)

feature_extractor = nn.DataParallel(feature_extractor).train(True)
classifier = nn.DataParallel(classifier).train(True)
cluster_head = nn.DataParallel(cluster_head).train(True)
# 現在有効なtarget prototype数 
current_K = int( K )

save_config["runtime"] = {

    "dataset":
        parser_args.dataset,

    "source":
        source,

    "target":
        target,

    "exp":
        parser_args.exp,

    "method":
        method_name,

    "initial_K":
        K,

    "tau_mix":
        (
            tau_mix
            if use_prototype_split
            else None
        ),

    "seed":
        seed
}
with open(os.path.join(log_dir, 'config.yaml'), 'w') as f:
    f.write(yaml.dump(save_config))

# Memory queue init
target_size = target_train_ds.__len__()
n_batch = int(MQ_size/batch_size)    
memqueue = MemoryQueue(feat_dim, batch_size, n_batch, temp).cuda()
cnt_i = 0
with TrainingModeManager([feature_extractor, classifier], train=False) as mgr, torch.no_grad():
    while cnt_i < n_batch:
        for i, (im_target, _, id_target) in enumerate(target_initMQ_dl):
            im_target = im_target.cuda()
            id_target = id_target.cuda()
            feature_ex = feature_extractor(im_target)
            before_lincls_feat, after_lincls = classifier(feature_ex)
            memqueue.update_queue(F.normalize(before_lincls_feat), id_target)
            cnt_i += 1
            if cnt_i > n_batch-1:
                break

# ============================================================
# Dynamic prototype split
# ============================================================

split_start_step = int(
    args.train.min_step
    * split_start_ratio
)


next_split_check = (
    split_start_step
)


# split処理がまだ有効か
split_active = (
    use_prototype_split
)


# Qの合計
split_q_sum = {}


# featureの合計
split_feature_sum = {}


# 出現回数
split_sample_count = {}


# split履歴
split_event_logs = []


# M_k判定履歴
mixedness_logs = []

# 提案手法で追加
# =====================================================
# Q_tt collection for prototype split analysis
# =====================================================

# 学習最後の何step分のQ_ttを集めるか
q_collect_steps = 1000

q_collect_start_step = max(
    0,
    args.train.min_step
    - q_collect_steps
)

# ============================================================
# split終了直前禁止
#
# 最終q_collect_steps分は最終評価用Qを安定して収集するため、
# このstep以降は新しいsplitを行わずKを固定する。
# (q_collect_start_stepと同じ閾値:
#  最終Q収集が始まると同時にsplitを止める)
# ============================================================
split_end_step = (
    q_collect_start_step
)

# sample IDごとにQを蓄積
q_assignment_sum = {}

# sample IDごとの出現回数
q_assignment_count = {}

# 最終分析用Qが何prototype次元か記録
q_collection_K = int(
    current_K
)


total_steps = tqdm(range(args.train.min_step), desc='global step')
global_step = 0
beta = None

# split判定前はまだM_kが存在しない
max_mixedness = np.nan
while global_step < args.train.min_step:
    iters = zip(source_train_dl, target_train_dl)
    for minibatch_id, ((im_source, label_source, id_source), (im_target, _, id_target)) in enumerate(iters):
        label_source = label_source.cuda()
        im_source = im_source.cuda()
        im_target = im_target.cuda()

        feature_ex_s = feature_extractor.forward(im_source)
        feature_ex_t = feature_extractor.forward(im_target)

        before_lincls_feat_s, after_lincls_s = classifier(feature_ex_s)
        before_lincls_feat_t, after_lincls_t = classifier(feature_ex_t)

        norm_feat_s = F.normalize(before_lincls_feat_s)
        norm_feat_t = F.normalize(before_lincls_feat_t)

        after_cluhead_t = cluster_head(before_lincls_feat_t)

        # =====Source Supervision=====
        criterion = nn.CrossEntropyLoss().cuda()
        loss_cls = criterion(after_lincls_s, label_source)

        # =====Private Class Discovery=====
        minibatch_size = norm_feat_t.size(0)

        # obtain nearest neighbor from memory queue and current mini-batch
        feat_mat2 = torch.matmul(norm_feat_t, norm_feat_t.t()) / temp
        mask = torch.eye(feat_mat2.size(0), feat_mat2.size(0)).bool().cuda()
        feat_mat2.masked_fill_(mask, -1/temp)

        nb_value_tt, nb_feat_tt = memqueue.get_nearest_neighbor(norm_feat_t, id_target.cuda())
        neighbor_candidate_sim = torch.cat([nb_value_tt.reshape(-1,1), feat_mat2], 1)
        values, indices = torch.max(neighbor_candidate_sim, 1)
        neighbor_norm_feat = torch.zeros((minibatch_size, norm_feat_t.shape[1])).cuda()
        for i in range(minibatch_size):
            neighbor_candidate_feat = torch.cat([nb_feat_tt[i].reshape(1,-1), norm_feat_t], 0)
            neighbor_norm_feat[i,:] = neighbor_candidate_feat[indices[i],:]
            
        neighbor_output = cluster_head(neighbor_norm_feat)
        
        # fill input features with memory queue
        #fill_size_ot = K
        fill_size_ot = current_K
        mqfill_feat_t = memqueue.random_sample(fill_size_ot)
        mqfill_output_t = cluster_head(mqfill_feat_t)

        # OT process
        # mini-batch feat (anchor) | neighbor feat | filled feat (sampled from memory queue)
        S_tt = torch.cat([after_cluhead_t, neighbor_output, mqfill_output_t], 0)
        S_tt *= temp
        Q_tt = sinkhorn(S_tt.detach(), epsilon=0.05, sinkhorn_iterations=3)
        Q_tt_tilde = Q_tt * Q_tt.size(0)
        anchor_Q = Q_tt_tilde[:minibatch_size, :]
        neighbor_Q = Q_tt_tilde[minibatch_size:2*minibatch_size, :]
        # ============================================================
        # Collect Q_tt / feature for dynamic split
        # ============================================================

        if (
            use_prototype_split
            and
            split_active
        ):

            collect_start_step = max(
                0,
                next_split_check
                - split_collect_steps
            )


            should_collect = (
                global_step
                >= collect_start_step
                and
                global_step
                < next_split_check
            )


            if should_collect:

                # Qを確率分布として正規化
                anchor_prob_split = (
                    anchor_Q
                    / anchor_Q.sum(
                        dim=1,
                        keepdim=True
                    ).clamp_min(
                        1e-12
                    )
                )


                q_np = (
                    anchor_prob_split
                    .detach()
                    .cpu()
                    .numpy()
                )


                feature_np = (
                    norm_feat_t
                    .detach()
                    .cpu()
                    .numpy()
                )


                ids_np = (
                    id_target
                    .detach()
                    .cpu()
                    .numpy()
                    .reshape(-1)
                )


                for (
                    sample_id,
                    q_vector,
                    feature_vector
                ) in zip(
                    ids_np,
                    q_np,
                    feature_np
                ):

                    sample_id = int(
                        sample_id
                    )


                    if (
                        sample_id
                        not in split_q_sum
                    ):

                        split_q_sum[
                            sample_id
                        ] = (
                            q_vector.copy()
                        )

                        split_feature_sum[
                            sample_id
                        ] = (
                            feature_vector.copy()
                        )

                        split_sample_count[
                            sample_id
                        ] = 1


                    else:

                        split_q_sum[
                            sample_id
                        ] += (
                            q_vector
                        )

                        split_feature_sum[
                            sample_id
                        ] += (
                            feature_vector
                        )

                        split_sample_count[
                            sample_id
                        ] += 1

        # 提案手法の追加コード
        # =====================================================
        # Q_tt collection
        # 学習後半のQ_ttをsample IDごとに保存
        # =====================================================

        if global_step >= q_collect_start_step:
            # ============================================================
            # current_Kが変化した場合、
            # 異なる次元のQを混ぜないためbufferをリセット
            # ============================================================

            if (
                q_collection_K
                != current_K
            ):

                q_assignment_sum = {}
                q_assignment_count = {}

                q_collection_K = int(
                    current_K
                )

                print(
                    "Final Q buffer reset because K changed:",
                    q_collection_K
                )

            # 各sampleについてQの合計が1になるように正規化
            anchor_prob = (
                anchor_Q
                / anchor_Q.sum(
                    dim=1,
                    keepdim=True
                ).clamp_min(1e-12)
            )

            anchor_prob_np = (
                anchor_prob
                .detach()
                .cpu()
                .numpy()
            )

            target_ids_np = (
                id_target
                .detach()
                .cpu()
                .numpy()
                .reshape(-1)
            )

            for sample_id, q_vector in zip(
                target_ids_np,
                anchor_prob_np
            ):

                sample_id = int(
                    sample_id
                )

                if sample_id not in q_assignment_sum:

                    q_assignment_sum[
                        sample_id
                    ] = (
                        q_vector.copy()
                    )

                    q_assignment_count[
                        sample_id
                    ] = 1

                else:

                    q_assignment_sum[
                        sample_id
                    ] += q_vector

                    q_assignment_count[
                        sample_id
                    ] += 1
                    
        # compute loss_PCD
        loss_local = 0
        for i in range(minibatch_size):
            sub_loss_local = 0
            sub_loss_local += -torch.sum(neighbor_Q[i,:] * F.log_softmax(after_cluhead_t[i,:]))
            sub_loss_local += -torch.sum(anchor_Q[i,:] * F.log_softmax(neighbor_output[i,:]))
            sub_loss_local /= 2
            loss_local += sub_loss_local
        loss_local /= minibatch_size
        loss_global = -torch.mean(torch.sum(anchor_Q * F.log_softmax(after_cluhead_t, dim=1), dim=1))
        loss_PCD = (loss_global + loss_local) / 2

        # =====Common Class Detection=====
        if global_step > 100:
            source_prototype = classifier.module.ProtoCLS.fc.weight
            if beta is None:
                beta = ot.unif(source_prototype.size()[0])

            # fill input features with memory queue
            fill_size_uot = n_batch*batch_size
            mqfill_feat_t = memqueue.random_sample(fill_size_uot)
            ubot_feature_t = torch.cat([mqfill_feat_t, norm_feat_t], 0)
            full_size = ubot_feature_t.size(0)
            
            # Adaptive filling
            newsim, fake_size = adaptive_filling(ubot_feature_t, source_prototype, gamma, beta, fill_size_uot)
        
            # UOT-based CCD
            high_conf_label_id, high_conf_label, _, new_beta = ubot_CCD(newsim, beta, fake_size=fake_size, 
                                                                    fill_size=fill_size_uot, mode='minibatch')
            # adaptive update for marginal probability vector
            beta = mu*beta + (1-mu)*new_beta

            # fix the bug raised in https://github.com/changwxx/UniOT-for-UniDA/issues/1
            # Due to mini-batch sampling, current mini-batch samples might be all target-private. 
            # (especially when target-private samples dominate target domain, e.g. OfficeHome)
            if high_conf_label_id.size(0) > 0:
                loss_CCD = criterion(after_lincls_t[high_conf_label_id,:], high_conf_label[high_conf_label_id])
            else:
                loss_CCD = 0
        else:
            loss_CCD = 0
        
        loss_all = loss_cls + lam * (loss_PCD + loss_CCD)
        
        with OptimizerManager([opt_sche_featex, opt_sche_cls, opt_sche_cluhead]):
            loss_all.backward()

        classifier.module.ProtoCLS.weight_norm() # very important for proto-classifier
        cluster_head.module.weight_norm() # very important for proto-classifier
        memqueue.update_queue(norm_feat_t, id_target.cuda())
        global_step += 1
        total_steps.update()

        # ============================================================
        # 学習終了直前はsplit禁止
        #
        # 最終q_collect_steps分は最終評価用Qを安定収集する期間のため、
        # このタイミング以降はsplit_activeをFalseにして
        # 新しいsplitが起きないようにする(Kを固定する)。
        # 学習自体はmin_stepまでそのまま継続する。
        # ============================================================
        if (
            use_prototype_split
            and split_active
            and global_step > split_end_step
        ):

            split_active = False

            print(
                "\nReached split_end_step:",
                split_end_step,
                "- prototype splitting disabled for the "
                "remainder of training. Final K:",
                current_K
            )

        # ============================================================
        # Dynamic prototype split check
        # ============================================================

        if (
            use_prototype_split
            and
            split_active
            and
            global_step >= next_split_check
        ):

            print(
                "\n========================================"
            )

            print(
                "Prototype split check"
            )

            print(
                "global_step:",
                global_step
            )

            print(
                "current_K:",
                current_K
            )

            print(
                "========================================"
            )


            # ========================================================
            # Q / Feature matrixを構築
            # ========================================================

            sample_ids = sorted(
                split_q_sum.keys()
            )


            if len(
                sample_ids
            ) == 0:

                print(
                    "No Q samples collected."
                )

            else:

                split_q_matrix = np.stack(
                    [
                        split_q_sum[
                            sample_id
                        ]
                        / split_sample_count[
                            sample_id
                        ]

                        for sample_id
                        in sample_ids
                    ],
                    axis=0
                )


                split_feature_matrix = np.stack(
                    [
                        split_feature_sum[
                            sample_id
                        ]
                        / split_sample_count[
                            sample_id
                        ]

                        for sample_id
                        in sample_ids
                    ],
                    axis=0
                )


                # feature再正規化
                feature_norm = np.linalg.norm(
                    split_feature_matrix,
                    axis=1,
                    keepdims=True
                )


                split_feature_matrix = (
                    split_feature_matrix
                    / np.clip(
                        feature_norm,
                        1e-12,
                        None
                    )
                )


                # ====================================================
                # safety check
                # ====================================================

                if (
                    split_q_matrix.shape[1]
                    != current_K
                ):

                    raise ValueError(
                        "Q dimension "
                        f"{split_q_matrix.shape[1]} "
                        f"does not match current_K="
                        f"{current_K}."
                    )


                # ====================================================
                # M_kの計算
                # ====================================================

                mixedness_results = (
                    find_mixed_prototypes(
                        q_assignment_matrix=
                            split_q_matrix,

                        tau_mix=
                            tau_mix,

                        min_samples=
                            split_min_samples
                    )
                )


                # 全prototypeのうち最大M_k
                valid_mixedness = [
                    result[
                        "before_dispersion"
                    ]
                    for result
                    in mixedness_results
                ]


                if len(
                    valid_mixedness
                ) > 0:

                    max_mixedness = max(
                        valid_mixedness
                    )

                else:

                    max_mixedness = np.nan


                mixedness_logs.append(
                    {
                        "global_step":
                            global_step,

                        "current_K":
                            current_K,

                        "max_before_dispersion":
                            max_mixedness,

                        "tau_mix":
                            tau_mix
                    }
                )


                # ====================================================
                # split候補
                # ====================================================

                candidates = [
                    result
                    for result
                    in mixedness_results
                    if result[
                        "is_mixed"
                    ]
                ]


                candidates = sorted(
                    candidates,
                    key=lambda x:
                        x[
                            "before_dispersion"
                        ],
                    reverse=True
                )


                print(
                    "max M_k:",
                    max_mixedness
                )

                print(
                    "tau_mix:",
                    tau_mix
                )

                print(
                    "split candidates:",
                    len(
                        candidates
                    )
                )


                # ====================================================
                # STOP CONDITION
                #
                # max_k M_k <= tau_mix
                # ====================================================

                if len(
                    candidates
                ) == 0:

                    split_active = False


                    print(
                        "\nNo prototype satisfies "
                        "M_k > tau_mix."
                    )

                    print(
                        "Prototype splitting stopped."
                    )

                    print(
                        "Final adaptive K:",
                        current_K
                    )


                else:

                    # =================================================
                    # candidateごとにsplit
                    # =================================================

                    successful_splits = 0


                    for candidate in (
                        candidates
                    ):

                        proto_id = int(
                            candidate[
                                "prototype_id"
                            ]
                        )


                        child_result = (
                            build_split_children(
                                q_assignment_matrix=
                                    split_q_matrix,

                                feature_matrix=
                                    split_feature_matrix,

                                prototype_id=
                                    proto_id,

                                min_group_samples=
                                    split_min_group_samples,

                                random_state=
                                    seed
                            )
                        )


                        if (
                            child_result
                            is None
                        ):

                            continue


                        child1 = (
                            torch.from_numpy(
                                child_result[
                                    "child_weight_1"
                                ]
                            )
                            .float()
                            .cuda()
                        )


                        child2 = (
                            torch.from_numpy(
                                child_result[
                                    "child_weight_2"
                                ]
                            )
                            .float()
                            .cuda()
                        )


                        old_K = (
                            current_K
                        )


                        # =============================================
                        # 動的prototype追加
                        # =============================================

                        (
                            new_parameter,
                            new_proto_id
                        ) = (
                            cluster_head
                            .module
                            .split_prototype(
                                prototype_id=
                                    proto_id,

                                child_weight_1=
                                    child1,

                                child_weight_2=
                                    child2
                            )
                        )


                        # =============================================
                        # 新prototypeをoptimizerに登録
                        # =============================================

                        register_new_prototype_parameter(
                            optimizer=
                                optimizer_cluhead,

                            new_parameter=
                                new_parameter
                        )


                        current_K = (
                            cluster_head
                            .module
                            .num_prototypes
                        )


                        successful_splits += 1


                        split_event_logs.append(
                            {
                                "global_step":
                                    global_step,

                                "parent_prototype":
                                    proto_id,

                                "new_prototype":
                                    new_proto_id,

                                "before_dispersion":
                                    candidate[
                                        "before_dispersion"
                                    ],

                                "group1_count":
                                    child_result[
                                        "group1_count"
                                    ],

                                "group2_count":
                                    child_result[
                                        "group2_count"
                                    ],

                                "K_before":
                                    old_K,

                                "K_after":
                                    current_K
                            }
                        )


                    # =============================================
                    # normalize
                    # =============================================

                    cluster_head.module.weight_norm()


                    print(
                        "Successful splits:",
                        successful_splits
                    )

                    print(
                        "New K:",
                        current_K
                    )


                    # 候補はあったが
                    # 安全条件により1つもsplitできない場合
                    if (
                        successful_splits
                        == 0
                    ):

                        print(
                            "Candidates existed, "
                            "but no valid two-group split "
                            "was found."
                        )

                        print(
                            "Splitting remains active "
                            "and will be checked again."
                        )


            # ========================================================
            # split後は必ずbufferを空にする
            #
            # Kが変わるため、古いQと新しいQを
            # 混ぜてはいけない
            # ========================================================

            split_q_sum = {}

            split_feature_sum = {}

            split_sample_count = {}


            # ========================================================
            # 次の判定時刻
            # ========================================================

            next_split_check += (
                split_check_interval
            )
        
        if global_step % args.log.log_interval == 0:
            counter = AccuracyCounter()
            counter.addOneBatch(variable_to_numpy(one_hot(label_source, len(source_classes))), variable_to_numpy(after_lincls_s))
            acc_source = torch.tensor([counter.reportAccuracy()]).cuda()
            logger.add_scalar('loss_all', loss_all, global_step)
            logger.add_scalar('loss_cls', loss_cls, global_step)
            logger.add_scalar('loss_PCD', loss_PCD, global_step)
            logger.add_scalar('loss_CCD', loss_CCD, global_step)
            logger.add_scalar('acc_source', acc_source, global_step)
            logger.add_scalar('prototype/current_K',current_K,global_step)
            if (
                use_prototype_split
                and
                not np.isnan(
                    max_mixedness
                )
            ):

                logger.add_scalar(
                    'prototype/max_mixedness',
                    max_mixedness,
                    global_step
                )
        if global_step % args.test.test_interval == 0:
            results = eval(feature_extractor, classifier, target_test_dl, classes_set, gamma=gamma, beta=beta)
            logger.add_scalar('cls_common_acc', results['cls_common_acc'], global_step)
            logger.add_scalar('cls_tp_acc', results['cls_tp_acc'], global_step)
            logger.add_scalar('tp_nmi', results['tp_nmi'], global_step)
            logger.add_scalar('cls_overall_acc', results['cls_overall_acc'], global_step)
            logger.add_scalar('h_score', results['h_score'], global_step)
            logger.add_scalar('h3_score', results['h3_score'], global_step)
            clear_output()

# 提案手法の追加コード
# =====================================================
# Build averaged Q assignment matrix
# =====================================================
if len(
    split_event_logs
) > 0:

    pd.DataFrame(
        split_event_logs
    ).to_csv(
        os.path.join(
            log_dir,
            "prototype_split_events.csv"
        ),
        index=False
    )

if len(
    mixedness_logs
) > 0:

    pd.DataFrame(
        mixedness_logs
    ).to_csv(
        os.path.join(
            log_dir,
            "prototype_mixedness_history.csv"
        ),
        index=False
    )
    
     
q_sample_ids = sorted(
    q_assignment_sum.keys()
)

if len(q_sample_ids) == 0:

    raise RuntimeError(
        "Q_tt was not collected."
    )

q_assignment_matrix = np.stack(
    [
        q_assignment_sum[
            sample_id
        ]
        / q_assignment_count[
            sample_id
        ]

        for sample_id
        in q_sample_ids
    ],
    axis=0
)

q_sample_ids = np.asarray(
    q_sample_ids,
    dtype=np.int64
)

# safety check
if (
    q_assignment_matrix.shape[1]
    != current_K
):

    raise ValueError(
        "Q matrix prototype dimension "
        f"{q_assignment_matrix.shape[1]} "
        f"does not match current_K="
        f"{current_K}."
    )

if not np.all(
    np.isfinite(
        q_assignment_matrix
    )
):

    raise ValueError(
        "Q matrix contains NaN or Inf."
    )

print(
    "Collected target samples:",
    q_assignment_matrix.shape[0]
)

print(
    "Number of prototypes:",
    q_assignment_matrix.shape[1]
)

print(
    "Target train size:",
    target_size
)

# 修正前
# save final model
# data = {
#         "feature_extractor": feature_extractor.state_dict(),
#         "classifier": classifier.state_dict(),
#         'cluster_head': cluster_head.state_dict(),
#         'K': K,
#         'beta': torch.from_numpy(beta)
#         }

# 提案手法の場合
data = {

    "feature_extractor":
        feature_extractor.state_dict(),

    "classifier":
        classifier.state_dict(),

    "cluster_head":
        cluster_head.state_dict(),

    "initial_K":
        K,

    "final_K":
        current_K,

    "K":
        current_K,

    "method":
        method_name,
        
    "tau_mix":
        (
            tau_mix
            if use_prototype_split
            else None
        ),

    "beta":
        torch.from_numpy(
            beta
        ),

    # ========================================================
    # Final Q
    # ========================================================

    "q_assignment_matrix":
        torch.from_numpy(
            q_assignment_matrix
        ).float(),

    "q_sample_ids":
        torch.from_numpy(
            q_sample_ids
        ).long(),

    "q_final_collection_K":
        q_collection_K,

    "q_final_sample_count":
        int(
            q_assignment_matrix.shape[0]
        )
}
runtime_result = {
    "method":
        method_name,

    "initial_K":
        K,

    "final_K":
        current_K,

    "tau_mix":
        tau_mix
}


pd.DataFrame(
    [runtime_result]
).to_csv(
    os.path.join(
        log_dir,
        "prototype_runtime_result.csv"
    ),
    index=False
)

with open(os.path.join(log_dir, 'final.pkl'), 'wb') as f:
    torch.save(data, f)

# 提案手法の追加コード
# =====================================================
# Prototype split analysis
# =====================================================

split_results = (
    analyze_prototype_splits(
        q_assignment_matrix,
        min_samples=10,
        random_state=seed
    )
)

split_result_df = pd.DataFrame(
    split_results
)

split_result_df = (
    split_result_df
    .sort_values(
        "before_dispersion",
        ascending=False,
        na_position="last"
    )
    .reset_index(
        drop=True
    )
)

split_result_df.to_csv(
    os.path.join(
        log_dir,
        "prototype_split_analysis.csv"
    ),
    index=False
)

# save test result in csv file
result = dict()
result.update(results)
pd.DataFrame(result, index=[0]).to_csv(f'{log_dir}/result.csv')

# visualization (only for office31/officehome/visda)
if parser_args.dataset in ['office31', 'officehome', 'visda']:
    s_label = classes_set['source_classes']
    t_label = classes_set['target_classes']
    writer = SummaryWriter(f'{log_path}/tsne')
    draw_tsne(feature_extractor, classifier, cluster_head,
            source_test_dl, target_test_dl,
            s_label, t_label,
            writer, parser_args.dataset)

