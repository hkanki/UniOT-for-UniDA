import os
import numpy as np
import pandas as pd
import torch


def build_ccd_rows(global_step, id_target, label_target_true,
                   debug_out, sim_minibatch, classes_set, source_classes):
    """
    Build a list of per-sample dicts from one mini-batch's ubot_CCD_debug output.

    Args:
        global_step       : current training step
        id_target         : CPU LongTensor (B,) - dataset indices of mini-batch samples
        label_target_true : CPU LongTensor (B,) - true class labels (used only for CSV)
        debug_out         : dict returned by ubot_CCD_debug()
        sim_minibatch     : CUDA FloatTensor (B, |S|) - raw cosine sim to source prototypes
                            = matmul(norm_feat_t.detach(), source_prototype.detach().t())
        classes_set       : dict with 'common_classes', 'tp_classes', etc. from data.py
        source_classes    : list of source class indices from data.py
    """
    wt_i_np        = debug_out['wt_i'].detach().cpu().numpy()
    pseudo_label_np = debug_out['pseudo_label'].detach().cpu().numpy()
    delta_i_np     = debug_out['delta_i'].detach().cpu().numpy()
    proto_conf_np  = torch.max(sim_minibatch, dim=1)[0].detach().cpu().numpy()

    id_np         = id_target.cpu().numpy()
    true_label_np = label_target_true.cpu().numpy()

    common_set = set(classes_set['common_classes'])
    tp_set     = set(classes_set['tp_classes'])

    rows = []
    for i in range(len(id_np)):
        tid      = int(id_np[i])
        true_lbl = int(true_label_np[i])
        ps_idx   = int(pseudo_label_np[i])
        # source_classes = [0, 1, ..., |S|-1] so source_classes[ps_idx] == ps_idx,
        # but we make the mapping explicit for correctness in all configurations.
        ps_class = int(source_classes[ps_idx])
        max_w    = float(wt_i_np[i])
        di       = int(delta_i_np[i])
        pc       = float(proto_conf_np[i])

        is_common = true_lbl in common_set
        is_tp     = true_lbl in tp_set
        pred_str  = 'common' if di == 1 else 'unknown'

        # meaningful only when is_common; TP samples can never match a source-class label
        is_ps_correct = (ps_class == true_lbl) and is_common

        # delta is correct when: common→detected OR target-private→rejected
        is_delta_ok = (di == 1 and is_common) or (di == 0 and is_tp)

        if is_common:
            if di == 1 and is_ps_correct:
                status = 'common_detected_correct'
            elif di == 1 and not is_ps_correct:
                status = 'common_detected_wrong_label'
            else:
                status = 'common_undetected'
        elif is_tp:
            status = 'private_correctly_rejected' if di == 0 else 'private_misdetected_as_common'
        else:
            status = 'other'

        rows.append({
            'global_step'               : global_step,
            'target_id'                 : tid,
            'true_label'                : true_lbl,
            'is_true_common'            : is_common,
            'is_true_target_private'    : is_tp,
            'pseudo_label_idx'          : ps_idx,
            'pseudo_label_class'        : ps_class,
            'max_Qst_weight'            : max_w,
            'delta_i'                   : di,
            'predicted_common_or_unknown': pred_str,
            'source_prototype_confidence': pc,
            'is_pseudo_label_correct'   : is_ps_correct,
            'is_delta_correct'          : is_delta_ok,
            'sample_status'             : status,
        })
    return rows


def compute_summary_row(rows, global_step):
    """
    Compute one summary row from the per-sample row list of a single analysis step.

    Returns a dict with aggregated metrics (precision/recall/FPR of delta, pseudo-label
    accuracy, and wt_i statistics split by true class membership).
    """
    df = pd.DataFrame(rows)

    n_samples = len(df)
    n_common  = int(df['is_true_common'].sum())
    n_tp      = int(df['is_true_target_private'].sum())

    delta1   = df[df['delta_i'] == 1]
    n_delta1 = len(delta1)

    # precision: fraction of delta=1 samples that are truly common
    precision = (
        float(delta1['is_true_common'].sum() / n_delta1)
        if n_delta1 > 0 else float('nan')
    )

    # recall: fraction of truly common samples detected as delta=1
    common_df = df[df['is_true_common']]
    recall = float(common_df['delta_i'].sum() / n_common) if n_common > 0 else float('nan')

    # FPR: fraction of truly target-private samples misdetected as delta=1
    tp_df = df[df['is_true_target_private']]
    fpr   = float(tp_df['delta_i'].sum() / n_tp) if n_tp > 0 else float('nan')

    # pseudo-label accuracy restricted to delta=1 AND truly common samples
    delta1_common = delta1[delta1['is_true_common']]
    n_d1c  = len(delta1_common)
    ps_acc = (
        float(delta1_common['is_pseudo_label_correct'].sum() / n_d1c)
        if n_d1c > 0 else float('nan')
    )

    mean_w_all    = float(df['max_Qst_weight'].mean())
    mean_w_common = float(common_df['max_Qst_weight'].mean()) if n_common > 0 else float('nan')
    mean_w_tp     = float(tp_df['max_Qst_weight'].mean())     if n_tp > 0    else float('nan')

    return {
        'global_step'                   : global_step,
        'n_samples'                     : n_samples,
        'n_common'                      : n_common,
        'n_target_private'              : n_tp,
        'precision_delta'               : precision,
        'recall_delta'                  : recall,
        'fpr_delta'                     : fpr,
        'pseudo_label_accuracy_on_delta': ps_acc,
        'mean_max_Qst_weight'           : mean_w_all,
        'mean_max_Qst_weight_common'    : mean_w_common,
        'mean_max_Qst_weight_private'   : mean_w_tp,
    }


def save_ccd_analysis_step(rows, analysis_dir, global_step):
    """Save per-sample CCD analysis for one step to ccd_analysis_step_{step}.csv."""
    df   = pd.DataFrame(rows)
    path = os.path.join(analysis_dir, f'ccd_analysis_step_{global_step}.csv')
    df.to_csv(path, index=False)


def append_ccd_summary(summary_row, analysis_dir):
    """Append one summary row to ccd_summary.csv (creates the file on first call)."""
    path        = os.path.join(analysis_dir, 'ccd_summary.csv')
    df          = pd.DataFrame([summary_row])
    write_header = not os.path.exists(path)
    df.to_csv(path, mode='a', header=write_header, index=False)
