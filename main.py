from data import *
from eval import eval
from utils.net import ResNet50Fc, ProtoCLS, CLS
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
from utils.lib import seed_everything, sinkhorn, ubot_CCD, adaptive_filling
from utils.lib import seed_everything, sinkhorn, ubot_CCD, ubot_CCD_debug, adaptive_filling

cudnn.benchmark = True
cudnn.deterministic = True

# 乱数シードの固定
seed = 1234
seed_everything(seed)

def calc_ccd_eval_metrics(debug_info, label_target, source_classes, common_classes, tp_classes):
    """
    CCDにおけるUOTの共通クラス検出性能を評価する関数。

    評価指標:
    - precision_delta
    - recall_delta
    - fpr_delta
    - acc_pseudo
    """

    device = label_target.device

    delta_i = debug_info["delta_i"].bool()
    pseudo_label_idx = debug_info["pseudo_label"].long()

    # pseudo_label_idx は source_classes のindexなので、実際のクラスラベルに変換する
    source_classes_tensor = torch.tensor(source_classes, device=device, dtype=label_target.dtype)
    pseudo_label_class = source_classes_tensor[pseudo_label_idx]

    common_classes_tensor = torch.tensor(common_classes, device=device, dtype=label_target.dtype)
    tp_classes_tensor = torch.tensor(tp_classes, device=device, dtype=label_target.dtype)

    # label_target が common_classes に含まれるか
    is_true_common = (label_target.view(-1, 1) == common_classes_tensor.view(1, -1)).any(dim=1)

    # label_target が target-private classes に含まれるか
    is_true_target_private = (label_target.view(-1, 1) == tp_classes_tensor.view(1, -1)).any(dim=1)

    # 集計用
    n_delta = delta_i.sum().float()
    n_common = is_true_common.sum().float()
    n_target_private = is_true_target_private.sum().float()

    tp_delta = (delta_i & is_true_common).sum().float()
    fp_delta = (delta_i & is_true_target_private).sum().float()

    # 0除算対策
    eps = 1e-12

    # 1. delta_i=1 と判定されたもののうち、本当に共通クラスだった割合
    precision_delta = tp_delta / (n_delta + eps)

    # 2. 本当に共通クラスのものを、どれだけ delta_i=1 として拾えたか
    recall_delta = tp_delta / (n_common + eps)

    # 3. target-private を誤って delta_i=1 とした割合
    fpr_delta = fp_delta / (n_target_private + eps)

    # 4. delta_i=1 かつ真の共通クラスの中で、擬似ラベルが正しかった割合
    selected_common = delta_i & is_true_common
    n_selected_common = selected_common.sum().float()

    correct_pseudo = (
        selected_common & (pseudo_label_class == label_target)
    ).sum().float()

    acc_pseudo = correct_pseudo / (n_selected_common + eps)

    metrics = {
        "precision_delta": precision_delta.item(),
        "recall_delta": recall_delta.item(),
        "fpr_delta": fpr_delta.item(),
        "acc_pseudo": acc_pseudo.item(),
        "n_delta": int(n_delta.item()),
        "n_common": int(n_common.item()),
        "n_target_private": int(n_target_private.item()),
        "n_selected_common": int(n_selected_common.item())
    }

    return metrics

# 使用するGPUをconfig.yamlで設定した値を入力
## 注意：GPUを使用しない場合は、config.yamlのgpu_indexを空にしてください。という処理
if len(parser_args.gpu_index) < 1:
    os.environ["CUDA_VISIbetaLE_DEVICES"] = ""
    gpu_ids = []

else:
    os.environ["CUDA_VISIbetaLE_DEVICES"] = parser_args.gpu_index
    gpu_ids = list(map(int, parser_args.gpu_index))

## TensorBoard用のログ保存先を設定
log_dir = f'{log_path}'
# TensorBoardに学習ログを保存する準備
logger = SummaryWriter(log_dir)

# define network architecture
# ソースドメインで分類するクラス数をcls_output_dimに設定
cls_output_dim = len(source_classes)
# 特徴抽出後に利用する特徴ベクトルの次元数
feat_dim = 256
# ResNet50をベースにした特徴抽出器を定義
feature_extractor = ResNet50Fc(pretrained_model_path)
# ソースドメインのクラス分類に使う分類器
classifier = CLS(feature_extractor.output_dim, cls_output_dim, hidden_mlp=2048, feat_dim=256, temp=temp)
# ターゲットドメインのクラスタリング用ヘッド
## 各クラスタの代表ベクトルとの類似度を出す層を定義
### 共通クラス検出 CCD のために使う“ソースクラスのプロトタイプ”を作る役割を持つ層も定義
cluster_head = ProtoCLS(feat_dim, K, temp=temp)

feature_extractor = feature_extractor.cuda()
classifier = classifier.cuda()
cluster_head = cluster_head.cuda()

# 特徴抽出器ResNet50用の最適化器を定義
optimizer_featex = optim.SGD(feature_extractor.parameters(), lr=args.train.lr*0.1, weight_decay=args.train.weight_decay, momentum=args.train.sgd_momentum, nesterov=True)
# ソースドメインのクラス分類用の最適化器を定義
optimizer_cls = optim.SGD(classifier.parameters(), lr=args.train.lr, weight_decay=args.train.weight_decay, momentum=args.train.sgd_momentum, nesterov=True)
# ターゲットドメインのクラスタリング用の最適化器を定義
optimizer_cluhead = optim.SGD(cluster_head.parameters(), lr=args.train.lr, weight_decay=args.train.weight_decay, momentum=args.train.sgd_momentum, nesterov=True)

# learning rate decay
## 学習率を訓練ステップに応じて徐々に小さくする関数を定義
scheduler = lambda step, initial_lr: inverseDecaySheduler(step, initial_lr, gamma=10, power=0.75, max_iter=args.train.min_step)

# 3つのoptimizerに、学習率を変化させる仕組みを組み込む
## opt_sche_featex：feature_extractor 用
## opt_sche_cls：classifier 用
## opt_sche_cluhead：cluster_head 用
opt_sche_featex = OptimWithSheduler(optimizer_featex,scheduler)
opt_sche_cls = OptimWithSheduler(optimizer_cls,scheduler)
opt_sche_cluhead = OptimWithSheduler(optimizer_cluhead,scheduler)

## DataParallel は、複数GPUがある場合に、ミニバッチを分割して複数GPUで並列計算するための仕組み
## .train(True)で学習モードに設定
feature_extractor = nn.DataParallel(feature_extractor).train(True)
classifier = nn.DataParallel(classifier).train(True)
cluster_head = nn.DataParallel(cluster_head).train(True)

# 実験設定を保存
with open(os.path.join(log_dir, 'config.yaml'), 'w') as f:
    f.write(yaml.dump(save_config))

# Memory queue init
## ターゲット訓練データの総数を取得
target_size = target_train_ds.__len__()

# Memory Queueに何バッチ分の特徴量を保存するかを決める
n_batch = int(MQ_size/batch_size)    
# ターゲット特徴量を保存する MemoryQueue を作成
# feat_dim：保存する特徴量の次元数(このコードでは256)
# batch_size：1回に更新する特徴量数
# n_batch：Queueに何バッチ分保存するか
# temp：類似度計算で使うtemperature
memqueue = MemoryQueue(feat_dim, batch_size, n_batch, temp).cuda()

# Memory Queueに何バッチ分の特徴量を入れたかを数えるための変数
cnt_i = 0
# feature_extractor と classifier を一時的に評価モードにする
with TrainingModeManager([feature_extractor, classifier], train=False) as mgr, torch.no_grad():
    # target_initMQ_dl からターゲット画像を取り出して、Memory Queueが必要なバッチ数だけ埋まるまで繰り返す
    ## _ はターゲットラベルを表し、ターゲットドメインは基本的に教師なしとして扱うため、学習にはラベルを利用しない
    ## →そのため _ で受け取って無視している
    ## id_target はターゲット画像のIDで、Memory Queueで「同じサンプル自身を最近傍として選ばない」ためなどに利用する
    while cnt_i < n_batch:
        for i, (im_target, _, id_target) in enumerate(target_initMQ_dl):
            im_target = im_target.cuda()
            id_target = id_target.cuda()
            # ResNet50から画像特徴を抽出
            feature_ex = feature_extractor(im_target)
            # 分類器に通す
            ## before_lincls_feat：分類器の線形層に入る前の特徴量
            ## after_lincls：ソースクラス分類用のスコア
            before_lincls_feat, after_lincls = classifier(feature_ex)
            # 抽出したターゲット特徴量をMemory Queueに保存
            ## 正規化する理由はこの後の最近傍探索や類似度計算では、特徴量同士の内積を使うため、特徴ベクトルの長さを1にそろえている
            memqueue.update_queue(F.normalize(before_lincls_feat), id_target)
            
            # 1バッチ分保存するたびに cnt_i を増やして、Memory Queueが必要なバッチ数だけ埋まったらループを抜ける
            cnt_i += 1
            if cnt_i > n_batch-1:
                break

# 学習ステップの進捗バーを表示
total_steps = tqdm(range(args.train.min_step), desc='global step')
global_step = 0

analysis_steps = {
    200,
    args.train.min_step // 2,
    args.train.min_step - 1
}

beta = None

while global_step < args.train.min_step:
    iters = zip(source_train_dl, target_train_dl)
    for minibatch_id, ((im_source, label_source, id_source), (im_target, label_target, id_target)) in enumerate(iters):
        label_source = label_source.cuda()
        im_source = im_source.cuda()
        im_target = im_target.cuda()
        label_target = label_target.cuda()

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
        fill_size_ot = K
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
            # ===== CCD Evaluation =====
            if global_step in analysis_steps:
                with torch.no_grad():
                    debug_info = ubot_CCD_debug(
                        newsim,
                        beta,
                        fake_size=fake_size,
                        fill_size=fill_size_uot,
                        mode='minibatch'
                    )

                    ccd_metrics = calc_ccd_eval_metrics(
                        debug_info=debug_info,
                        label_target=label_target,
                        source_classes=source_classes,
                        common_classes=common_classes,
                        tp_classes=tp_classes
                    )

                    # TensorBoardへ記録
                    logger.add_scalar('ccd_eval/precision_delta', ccd_metrics['precision_delta'], global_step)
                    logger.add_scalar('ccd_eval/recall_delta', ccd_metrics['recall_delta'], global_step)
                    logger.add_scalar('ccd_eval/fpr_delta', ccd_metrics['fpr_delta'], global_step)
                    logger.add_scalar('ccd_eval/acc_pseudo', ccd_metrics['acc_pseudo'], global_step)

                    # CSVへ保存
                    ccd_metrics['global_step'] = global_step
                    ccd_metrics_path = os.path.join(log_dir, 'ccd_eval_metrics.csv')

                    pd.DataFrame([ccd_metrics]).to_csv(
                        ccd_metrics_path,
                        mode='a',
                        header=not os.path.exists(ccd_metrics_path),
                        index=False
                    )
                    

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
        
        if global_step % args.log.log_interval == 0:
            counter = AccuracyCounter()
            counter.addOneBatch(variable_to_numpy(one_hot(label_source, len(source_classes))), variable_to_numpy(after_lincls_s))
            acc_source = torch.tensor([counter.reportAccuracy()]).cuda()
            logger.add_scalar('loss_all', loss_all, global_step)
            logger.add_scalar('loss_cls', loss_cls, global_step)
            logger.add_scalar('loss_PCD', loss_PCD, global_step)
            logger.add_scalar('loss_CCD', loss_CCD, global_step)
            logger.add_scalar('acc_source', acc_source, global_step)

        if global_step % args.test.test_interval == 0:
            results = eval(feature_extractor, classifier, target_test_dl, classes_set, gamma=gamma, beta=beta)
            logger.add_scalar('cls_common_acc', results['cls_common_acc'], global_step)
            logger.add_scalar('cls_tp_acc', results['cls_tp_acc'], global_step)
            logger.add_scalar('tp_nmi', results['tp_nmi'], global_step)
            logger.add_scalar('cls_overall_acc', results['cls_overall_acc'], global_step)
            logger.add_scalar('h_score', results['h_score'], global_step)
            logger.add_scalar('h3_score', results['h3_score'], global_step)
            clear_output()

# save final model
data = {
        "feature_extractor": feature_extractor.state_dict(),
        "classifier": classifier.state_dict(),
        'cluster_head': cluster_head.state_dict(),
        'K': K,
        'beta': torch.from_numpy(beta)
        }
with open(os.path.join(log_dir, 'final.pkl'), 'wb') as f:
    torch.save(data, f)

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

