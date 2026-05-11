"""
m3_eval_combined.py — Method 3: M1 + M2 combined unknown detection
===================================================================
단일 forward pass 에서 두 가지 unknown 판단 기준을 동시에 적용.

텍스트 구성:
  texts = K known classes + 10 objectiveness prompts   (M2 와 동일)

  OBJECTIVENESS_PROMPTS[0] = "object"  →  index K = M1 의 "object" 토큰
  scores_all [N, K+10] 에서:
    scores_all[:, :K]  z_known   — M2 P_u 계산용
    scores_all[:, K]   obj_score — M1 object 토큰 점수
    scores_all[:, K:]  z_attr    — M2 P_b 계산용 (10개)

Combination 모드 (--mode):
  union     (기본)  : M1 OR  M2 통과 → 재현율 우선
  intersect         : M1 AND M2 통과 → 정밀도 우선
  score             : α * obj_score + (1-α) * P_u 융합 점수으로 단일 threshold

Unknown ranking 점수:
  union / intersect → P_u 내림차순 (범위 일관)
  score             → fused_score 내림차순

Usage:
  conda activate py310
  cd /Users/jisu/Desktop/dev/cli/YOLO-World
  python owod_reproduce/m3_eval_combined.py                      # union, task 1-4
  python owod_reproduce/m3_eval_combined.py --mode intersect
  python owod_reproduce/m3_eval_combined.py --mode score --alpha 0.4
  python owod_reproduce/m3_eval_combined.py --task 1 --quickeval 30
  python owod_reproduce/m3_eval_combined.py --sweep              # unk_thr sweep
"""

import sys
import functools
import warnings
import argparse
warnings.filterwarnings("ignore")

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party" / "mmyolo"))

import numpy as np
import torch

from owod_reproduce.util import (
    TASK_SPLITS, iou_xyxy,
    setup_eval, build_gt_indices, load_image_ids,
    run_task, print_summary, positive_int,
    resolve_image_path, load_gt, accumulate_known_ap, compute_task_metrics,
)
from owod_reproduce.m2_eval_pu import (
    OBJECTIVENESS_PROMPTS,
    patch_head_for_full_scores,
    compute_pu,
    run_task_sweep as _m2_sweep,
)
from collections import defaultdict


# ── M3 고유: 텍스트 구성 ──────────────────────────────────────────────────────

def _texts_fn(known_classes):
    """K known + 10 objectiveness prompts.
    OBJECTIVENESS_PROMPTS[0] = "object" → index K = M1 object 토큰."""
    return [[c] for c in known_classes] + [[p] for p in OBJECTIVENESS_PROMPTS]


# ── M3 고유: 조합 로직 ────────────────────────────────────────────────────────

def _combined_scores(pred, p_boxes, n_known,
                      obj_thr, unk_thr, pb_min, alpha, gamma, mode):
    """
    각 box 에 대해 (is_candidate, ranking_score) 를 반환.

    Returns:
        is_cand    : np.ndarray [N] bool
        rank_score : np.ndarray [N] float  (내림차순 정렬용)
    """
    n_attr = len(OBJECTIVENESS_PROMPTS)

    if (not hasattr(pred, 'scores_all')
            or pred.scores_all.shape[-1] != n_known + n_attr
            or pred.scores_all.shape[0] == 0):
        n = len(p_boxes)
        return np.zeros(n, dtype=bool), np.zeros(n)

    sa = pred.scores_all.cpu()                # [N, K+10]

    # ── M1 기준: raw object 토큰 점수 (index K) ───────────────────────────
    obj_scores = sa[:, n_known].numpy()       # [N], sigmoid 출력

    # ── M2 기준: P_u / P_b ────────────────────────────────────────────────
    p_u, p_b, _, _ = compute_pu(sa, n_known, gamma=gamma)
    p_u_np = p_u.numpy()
    p_b_np = p_b.numpy()

    m1_pass = obj_scores >= obj_thr
    m2_pass = (p_u_np >= unk_thr) & (p_b_np >= pb_min)

    if mode == "union":
        is_cand    = m1_pass | m2_pass
        rank_score = p_u_np                   # P_u 로 정렬

    elif mode == "intersect":
        is_cand    = m1_pass & m2_pass
        rank_score = p_u_np

    else:  # "score"
        fused      = alpha * obj_scores + (1.0 - alpha) * p_u_np
        is_cand    = fused >= unk_thr         # unk_thr 을 fused score 기준으로 재사용
        rank_score = fused

    return is_cand, rank_score


# ── M3 고유: unknown detector (run_task 콜백) ─────────────────────────────────

def _unknown_detector(pred, p_boxes, p_scores, p_labels,
                       gt_boxes, gt_labels, gt_ignore,
                       known_set, n_known, iou_thr,
                       _gt_used_known, *,
                       obj_thr=0.001, unk_thr=0.25, pb_min=0.10,
                       alpha=0.5, gamma=3, mode="union"):
    """
    M1 + M2 조합 unknown detector.
    gt_used_known 은 미사용 (독립 플래그).
    """
    is_cand, rank_score = _combined_scores(
        pred, p_boxes, n_known,
        obj_thr, unk_thr, pb_min, alpha, gamma, mode)

    unk_gt_used = [False] * len(gt_boxes)
    unk_tp = 0

    for ui in np.argsort(-rank_score):
        if not is_cand[ui]:
            continue
        pbox = p_boxes[ui]

        best_iou, best_gi = 0.0, -1
        for gi, gbox in enumerate(gt_boxes):
            if gt_ignore[gi]:
                continue
            v = iou_xyxy(pbox, gbox)
            if v > best_iou:
                best_iou, best_gi = v, gi

        if (best_iou >= iou_thr and best_gi >= 0
                and gt_labels[best_gi] not in known_set
                and not unk_gt_used[best_gi]):
            unk_tp += 1
            unk_gt_used[best_gi] = True

    return unk_tp


# ── Sweep 모드 ────────────────────────────────────────────────────────────────

def run_task_sweep(task_id, model, pipeline, coco_val_idx, coco_train_idx, ids,
                   known_keep_thr=0.05, obj_thr=0.001, pb_min=0.10,
                   alpha=0.5, gamma=3, mode="union",
                   unk_thrs=(0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40),
                   iou_thr=0.50):
    """
    추론 1회 → P_u/obj 캔디데이트 수집 → threshold 변경하며 재계산.
    """
    prev_classes  = [c for split in TASK_SPLITS[:task_id - 1] for c in split]
    cur_classes   = TASK_SPLITS[task_id - 1]
    known_classes = prev_classes + cur_classes
    known_set     = set(known_classes)
    prev_set      = set(prev_classes)
    n_known       = len(known_classes)
    n_attr        = len(OBJECTIVENESS_PROMPTS)
    idx_to_cls    = {i: c for i, c in enumerate(known_classes)}

    texts = _texts_fn(known_classes)

    known_preds         = defaultdict(list)
    known_n_gt          = defaultdict(int)
    unk_n_gt            = 0
    skipped             = 0
    unk_gt_global_count = 0
    # (obj_score, p_u, p_b, global_gt_id_or_-1)
    all_unk_preds       = []

    print(f"\n{'='*65}")
    print(f"Task {task_id}  |  prev={len(prev_classes)}  cur={len(cur_classes)}  "
          f"known={n_known}  [M3 SWEEP / mode={mode}]")
    print(f"{'='*65}")

    for idx, img_id in enumerate(ids):
        if idx % 500 == 0:
            print(f"  [{idx:5d}/{len(ids)}]  unk_gt={unk_gt_global_count}")

        try:
            img_path, source = resolve_image_path(img_id)
        except FileNotFoundError:
            skipped += 1
            continue

        try:
            if source == "coco_train" and coco_train_idx is None:
                skipped += 1
                continue
            gt_boxes, gt_labels, gt_ignore = load_gt(
                img_path, source, coco_val_idx, coco_train_idx)
        except Exception as e:
            print(f"  [GT ERROR] {img_id}: {e}")
            skipped += 1
            continue

        img_unk_gt_ids = {}
        for gi, (lbl, ign) in enumerate(zip(gt_labels, gt_ignore)):
            if ign:
                continue
            if lbl in known_set:
                known_n_gt[lbl] += 1
            else:
                img_unk_gt_ids[gi] = unk_gt_global_count
                unk_gt_global_count += 1
                unk_n_gt += 1

        try:
            data_info = dict(img_id=0, img_path=str(img_path), texts=texts)
            data_info = pipeline(data_info)
            batch = dict(inputs=data_info["inputs"].unsqueeze(0),
                         data_samples=[data_info["data_samples"]])
            with torch.no_grad():
                output = model.test_step(batch)[0]
        except Exception as e:
            print(f"  [INFER ERROR] {img_id}: {e}")
            continue

        pred     = output.pred_instances
        p_boxes  = pred.bboxes.cpu().numpy()
        p_scores = pred.scores.cpu().numpy()
        p_labels = pred.labels.cpu().numpy()

        # Known branch
        order   = np.argsort(-p_scores)
        gt_used = [False] * len(gt_boxes)
        accumulate_known_ap(
            p_boxes[order], p_scores[order], p_labels[order],
            gt_boxes, gt_labels, gt_ignore, gt_used,
            n_known, idx_to_cls, known_keep_thr, iou_thr, known_preds)

        # Unknown scores
        if (not hasattr(pred, 'scores_all')
                or pred.scores_all.shape[-1] != n_known + n_attr
                or pred.scores_all.shape[0] == 0):
            continue

        sa = pred.scores_all.cpu()
        obj_scores_np = sa[:, n_known].numpy()
        p_u, p_b, _, _ = compute_pu(sa, n_known, gamma=gamma)
        p_u_np = p_u.numpy()
        p_b_np = p_b.numpy()

        if mode == "score":
            rank = alpha * obj_scores_np + (1.0 - alpha) * p_u_np
        else:
            rank = p_u_np

        # Greedy 매칭
        unk_local_used = set()
        for ui in np.argsort(-rank):
            if p_b_np[ui] < pb_min:
                continue
            pbox = p_boxes[ui]
            best_iou, best_gi = 0.0, -1
            for gi, gbox in enumerate(gt_boxes):
                if gt_ignore[gi] or gi not in img_unk_gt_ids:
                    continue
                v = iou_xyxy(pbox, gbox)
                if v > best_iou:
                    best_iou, best_gi = v, gi

            if (best_iou >= iou_thr and best_gi >= 0
                    and best_gi not in unk_local_used):
                unk_local_used.add(best_gi)
                all_unk_preds.append((obj_scores_np[ui], p_u_np[ui],
                                      p_b_np[ui], img_unk_gt_ids[best_gi]))
            else:
                all_unk_preds.append((obj_scores_np[ui], p_u_np[ui],
                                      p_b_np[ui], -1))

    # Known AP
    from owod_reproduce.util import compute_ap
    prev_aps, cur_aps, all_aps = [], [], []
    for cls in known_classes:
        ap = compute_ap(known_preds[cls], known_n_gt[cls])
        if not np.isnan(ap):
            all_aps.append(ap)
            (prev_aps if cls in prev_set else cur_aps).append(ap)
    known_map50 = float(np.mean(all_aps)) if all_aps else 0.0

    print(f"\n  Known AP50 ({n_known:2d} cls) : {known_map50:.4f}")
    print(f"  Total unknown GT       : {unk_n_gt}")
    print(f"  mode={mode}  obj_thr={obj_thr}  pb_min={pb_min}")
    print(f"\n  unk_thr | U-REC50 | H-Score | TP / GT={unk_n_gt}")
    print(f"  {'-'*45}")

    sweep_results = []
    for unk_thr in unk_thrs:
        hit_gts = set()
        for obj_s, pu_i, pb_i, gt_gid in all_unk_preds:
            if pb_i < pb_min:
                continue
            if gt_gid < 0:
                continue
            if mode == "union":
                passes = (obj_s >= obj_thr) or (pu_i >= unk_thr)
            elif mode == "intersect":
                passes = (obj_s >= obj_thr) and (pu_i >= unk_thr)
            else:  # score
                passes = (alpha * obj_s + (1.0 - alpha) * pu_i) >= unk_thr

            if passes and gt_gid not in hit_gts:
                hit_gts.add(gt_gid)

        tp      = len(hit_gts)
        u_rec   = tp / (unk_n_gt + 1e-8)
        h_score = 2 * known_map50 * u_rec / (known_map50 + u_rec + 1e-8)
        print(f"  {unk_thr:.2f}    | {u_rec:.4f}  | {h_score:.4f}  | {tp}")
        sweep_results.append(dict(unk_thr=unk_thr, u_rec50=u_rec,
                                   h_score=h_score, unk_tp=tp,
                                   unk_n_gt=unk_n_gt, known_map50=known_map50))
    return sweep_results


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="OWOD M3: M1+M2 combined")
    parser.add_argument("--task",       type=int,   choices=[1, 2, 3, 4], default=None)
    parser.add_argument("--device",     default="cpu")
    parser.add_argument("--mode",       choices=["union", "intersect", "score"],
                        default="union",
                        help="union: M1 OR M2 | intersect: M1 AND M2 | score: α 융합")
    parser.add_argument("--known-thr",  type=float, default=0.05)
    parser.add_argument("--obj-thr",    type=float, default=0.001,
                        help="M1: object 토큰 threshold")
    parser.add_argument("--unk-thr",    type=float, default=0.001, # 기존 기본값 
                        help="M2: P_u threshold (score 모드에서는 fused score threshold)")
    parser.add_argument("--pb-min",     type=float, default=0.10,
                        help="M2: P_b 최솟값 (objectiveness floor)")
    parser.add_argument("--alpha",      type=float, default=0.5,
                        help="m1 & m2 combined score 조율 값 α * obj_score + (1-α) * P_u")
    parser.add_argument("--gamma",      type=int,   default=3,
                        help="top-γ for P_b")
    parser.add_argument("--quickeval",  nargs="?",  const=50,
                        type=positive_int, default=None, metavar="N",
                        help="VOC N + COCO N 이미지로 빠른 평가 (기본 N=50)")
    parser.add_argument("--sweep",      action="store_true",
                        help="unk_thr 0.10~0.40 sweep 모드")
    args = parser.parse_args()

    task_ids = [args.task] if args.task else [1, 2, 3, 4]

    print("Loading model …")
    model, pipeline = setup_eval(device=args.device)
    patch_head_for_full_scores(model.bbox_head)
    print(f"  done. (head patched, mode={args.mode})")

    coco_val_idx, coco_train_idx = build_gt_indices()
    ids = load_image_ids(args.quickeval)

    if args.sweep:
        SWEEP_THRS = (0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40)
        for tid in task_ids:
            run_task_sweep(
                tid, model, pipeline, coco_val_idx, coco_train_idx, ids,
                known_keep_thr=args.known_thr,
                obj_thr=args.obj_thr, pb_min=args.pb_min,
                alpha=args.alpha, gamma=args.gamma, mode=args.mode,
                unk_thrs=SWEEP_THRS)
        return

    detector = functools.partial(
        _unknown_detector,
        obj_thr=args.obj_thr,
        unk_thr=args.unk_thr,
        pb_min=args.pb_min,
        alpha=args.alpha,
        gamma=args.gamma,
        mode=args.mode,
    )
    results = []
    for tid in task_ids:
        r = run_task(
            tid, model, pipeline, coco_val_idx, coco_train_idx, ids,
            texts_fn=_texts_fn,
            unknown_detector=detector,
            known_keep_thr=args.known_thr,
        )
        results.append(r)

    print_summary(results)


if __name__ == "__main__":
    main()
