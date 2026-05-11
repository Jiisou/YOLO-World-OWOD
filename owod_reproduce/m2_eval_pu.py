"""
m2_eval_pu.py — Method 2: P_u unknown detection
================================================
Unknown 탐지를 "object" 토큰 대신 아래 공식으로 교체:

    P_u = 0.5 * (P_b + P_un) * (1 - max(P_C))

    P_C   = known class 확률분포 (joint L1 정규화)
    P_un  = H(P_C) / log(K)      — normalized 엔트로피 ∈ [0,1]
    P_b   = top-γ objectiveness 프롬프트 joint prob 합산

texts = K known classes + 10 objectiveness prompts
head monkey-patch 으로 NMS 이후 scores_all [N, K+10] 보존.

Usage:
  conda activate py310
  cd /Users/jisu/Desktop/dev/cli/YOLO-World
  python owod_reproduce/m2_eval_pu.py
  python owod_reproduce/m2_eval_pu.py --task 2
  python owod_reproduce/m2_eval_pu.py --quickeval 30
  python owod_reproduce/m2_eval_pu.py --sweep
"""

import sys
import copy
import types
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
from mmengine.structures import InstanceData
from mmdet.models.utils import filter_scores_and_topk

from owod_reproduce.util import (
    TASK_SPLITS, iou_xyxy,
    setup_eval, build_gt_indices, load_image_ids,
    run_task, print_summary, positive_int,
    # GT/AP helpers (sweep 모드에서 직접 사용)
    resolve_image_path, load_gt, accumulate_known_ap, compute_task_metrics,
    build_test_pipeline,
)
from mmengine.config import Config

# ── Objectiveness prompts ─────────────────────────────────────────────────────

OBJECTIVENESS_PROMPTS = [
    "object", "thing", "item", "an object", "a thing",
    "some object", "a region", "foreground", "something", "entity",
]


# ── Head monkey-patch ─────────────────────────────────────────────────────────

def patch_head_for_full_scores(bbox_head):
    """
    predict_by_feat 을 monkey-patch 해 NMS 이후에도
    InstanceData.scores_all [N, K+10] 을 보존한다.

    원본은 argmax 이후 스칼라 1개만 남기므로 전체 분포가 소실된다.
    keep_idxs 선택 직전에 full matrix 를 복사해 scores_all 로 저장하면
    InstanceData.__getitem__ 이 임의 필드를 인덱싱으로 보존하므로
    _bbox_post_process 내 NMS 인덱싱을 통과해도 살아남는다.
    """

    def new_predict_by_feat(self, cls_scores, bbox_preds, objectnesses=None,
                             batch_img_metas=None, cfg=None,
                             rescale=True, with_nms=True):
        assert len(cls_scores) == len(bbox_preds)
        with_objectnesses = objectnesses is not None
        if with_objectnesses:
            assert len(cls_scores) == len(objectnesses)

        cfg = self.test_cfg if cfg is None else cfg
        cfg = copy.deepcopy(cfg)
        multi_label = cfg.multi_label & (self.num_classes > 1)
        cfg.multi_label = multi_label

        num_imgs      = len(batch_img_metas)
        featmap_sizes = [cs.shape[2:] for cs in cls_scores]

        if featmap_sizes != self.featmap_sizes:
            self.mlvl_priors = self.prior_generator.grid_priors(
                featmap_sizes,
                dtype=cls_scores[0].dtype,
                device=cls_scores[0].device)
            self.featmap_sizes = featmap_sizes
        flatten_priors = torch.cat(self.mlvl_priors)

        mlvl_strides = [
            flatten_priors.new_full(
                (fs.numel() * self.num_base_priors, ), stride)
            for fs, stride in zip(featmap_sizes, self.featmap_strides)
        ]
        flatten_stride = torch.cat(mlvl_strides)

        flatten_cls_scores = torch.cat([
            cs.permute(0, 2, 3, 1).reshape(num_imgs, -1, self.num_classes)
            for cs in cls_scores], dim=1).sigmoid()
        flatten_bbox_preds = torch.cat([
            bp.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
            for bp in bbox_preds], dim=1)
        flatten_decoded_bboxes = self.bbox_coder.decode(
            flatten_priors[None], flatten_bbox_preds, flatten_stride)

        if with_objectnesses:
            flatten_objectness = torch.cat([
                obj.permute(0, 2, 3, 1).reshape(num_imgs, -1)
                for obj in objectnesses], dim=1).sigmoid()
        else:
            flatten_objectness = [None] * num_imgs

        results_list = []
        for bboxes, scores, objectness, img_meta in zip(
                flatten_decoded_bboxes, flatten_cls_scores,
                flatten_objectness, batch_img_metas):

            ori_shape    = img_meta['ori_shape']
            scale_factor = img_meta['scale_factor']
            pad_param    = img_meta.get('pad_param', None)
            score_thr    = cfg.get('score_thr', -1)

            if (objectness is not None and score_thr > 0
                    and not cfg.get('yolox_style', False)):
                conf_inds  = objectness > score_thr
                bboxes     = bboxes[conf_inds, :]
                scores     = scores[conf_inds, :]
                objectness = objectness[conf_inds]

            if objectness is not None:
                scores *= objectness[:, None]

            if scores.shape[0] == 0:
                empty = InstanceData()
                empty.bboxes     = bboxes
                empty.scores     = scores[:, 0]
                empty.labels     = scores[:, 0].int()
                empty.scores_all = scores          # empty [0, K+10]
                results_list.append(empty)
                continue

            # ── argmax 전에 full distribution 복사 ───────────────────────────
            scores_full = scores.clone()           # [N_anchors, K+10]

            nms_pre = cfg.get('nms_pre', 100000)
            if cfg.multi_label is False:
                scores, labels = scores.max(1, keepdim=True)
                scores, _, keep_idxs, d = filter_scores_and_topk(
                    scores, score_thr, nms_pre, results=dict(labels=labels[:, 0]))
                labels = d['labels']
            else:
                scores, labels, keep_idxs, _ = filter_scores_and_topk(
                    scores, score_thr, nms_pre)

            results = InstanceData(
                scores     = scores,
                labels     = labels,
                bboxes     = bboxes[keep_idxs],
                scores_all = scores_full[keep_idxs],   # ← 전체 분포 보존
            )

            if rescale:
                if pad_param is not None:
                    results.bboxes -= results.bboxes.new_tensor(
                        [pad_param[2], pad_param[0], pad_param[2], pad_param[0]])
                results.bboxes /= results.bboxes.new_tensor(
                    scale_factor).repeat((1, 2))

            if cfg.get('yolox_style', False):
                cfg.max_per_img = len(results)

            results = self._bbox_post_process(
                results=results, cfg=cfg, rescale=False,
                with_nms=with_nms, img_meta=img_meta)
            results.bboxes[:, 0::2].clamp_(0, ori_shape[1])
            results.bboxes[:, 1::2].clamp_(0, ori_shape[0])
            results_list.append(results)

        return results_list

    bbox_head.predict_by_feat = types.MethodType(new_predict_by_feat, bbox_head)


# ── P_u computation ───────────────────────────────────────────────────────────

def compute_pu(scores_all: torch.Tensor, n_known: int, gamma: int = 3,
               norm: str = "l1"):
    """
    P_u = 0.5 * (P_b + P_un) * (1 - max_pc)

    Args:
        scores_all : [N, K+n_attr]  sigmoid * objectness score
        n_known    : K
        gamma      : top-γ for P_b
        norm       : 정규화 방식
                     "l1"      — joint L1 정규화 (기본값)
                                  p_joint = z / sum(z)
                                  K 증가 시 attr 비율이 감소하는 특성 있음
                     "softmax" — joint softmax 정규화
                                  p_joint = softmax(z)
                                  K 증가에도 attr 비율이 안정적으로 유지됨

    Returns:
        (p_u, p_b, p_un, max_pc) — 각각 [N] CPU float tensor
    """
    eps = 1e-8
    z = scores_all.float()                                      # [N, K+n_attr]

    # ── 정규화 ────────────────────────────────────────────────────────────────
    if norm == "softmax":
        p_joint = torch.softmax(z, dim=-1)                      # [N, K+n_attr]
    else:  # "l1"
        p_joint = z / z.sum(dim=-1, keepdim=True).clamp(min=eps)

    p_known_joint = p_joint[:, :n_known]                        # [N, K]
    p_attr_joint  = p_joint[:, n_known:]                        # [N, n_attr]

    # ── P_C : known 내부 재정규화 ─────────────────────────────────────────────
    p_c    = p_known_joint / p_known_joint.sum(dim=-1, keepdim=True).clamp(min=eps)
    max_pc = p_known_joint.max(dim=-1).values                   # joint 기준 절댓값

    # ── P_un : 정규화 엔트로피 ────────────────────────────────────────────────
    h    = -(p_c * torch.log(p_c + eps)).sum(dim=-1)
    p_un = (h / float(np.log(max(n_known, 2)))).clamp(0.0, 1.0)

    # ── P_b : top-γ objectiveness joint prob ─────────────────────────────────
    # l1:      joint 비율 합산 → K 증가 시 감소하는 구조적 편향 있음
    # softmax: joint 비율 합산 → K 무관하게 attr 쪽 비율이 상대적으로 유지됨
    topk_vals, _ = p_attr_joint.topk(min(gamma, p_attr_joint.shape[-1]), dim=-1)
    p_b = topk_vals.sum(dim=-1)

    p_u = 0.5 * (p_b + p_un) * (1.0 - max_pc)
    return p_u.cpu(), p_b.cpu(), p_un.cpu(), max_pc.cpu()


# ── M2 고유: 텍스트 구성 ──────────────────────────────────────────────────────

def _texts_fn(known_classes):
    """K known classes + 10 objectiveness prompts."""
    return [[c] for c in known_classes] + [[p] for p in OBJECTIVENESS_PROMPTS]


# ── M2 고유: unknown detector ─────────────────────────────────────────────────

def _unknown_detector(pred, p_boxes, p_scores, p_labels,
                       gt_boxes, gt_labels, gt_ignore,
                       known_set, n_known, iou_thr,
                       _gt_used_known, *, unk_thr=0.25, pb_min=0.10, gamma=3,
                       norm="l1"):
    """
    P_u 기반 unknown TP 카운트.

    - gt_used_known 은 사용하지 않음: known branch 와 독립 플래그 사용.
    - P_u 내림차순으로 순회해 greedy 매칭.
    """
    n_attr = len(OBJECTIVENESS_PROMPTS)

    if (not hasattr(pred, 'scores_all')
            or pred.scores_all.shape[-1] != n_known + n_attr):
        return 0

    sa = pred.scores_all.cpu()
    if sa.shape[0] == 0:
        return 0

    p_u, p_b, _, _ = compute_pu(sa, n_known, gamma=gamma, norm=norm)
    p_u_np = p_u.numpy()
    p_b_np = p_b.numpy()

    is_cand    = (p_u_np >= unk_thr) & (p_b_np >= pb_min)
    unk_gt_used = [False] * len(gt_boxes)
    unk_tp      = 0

    for ui in np.argsort(-p_u_np):
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
                   known_keep_thr=0.05, pb_min=0.10, gamma=3,
                   unk_thrs=(0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40),
                   iou_thr=0.50, norm="l1"):
    """
    추론은 1회만 수행하고, P_u candidate 목록을 수집한 뒤
    threshold 만 바꾸며 U-REC / H-Score 를 재계산.
    """
    from collections import defaultdict

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
    all_unk_preds       = []   # [(p_u, p_b, global_gt_id_or_-1)]

    print(f"\n{'='*65}")
    print(f"Task {task_id}  |  prev={len(prev_classes)}  cur={len(cur_classes)}  "
          f"known={n_known}  [SWEEP mode]")
    print(f"{'='*65}")

    for idx, img_id in enumerate(ids):
        if idx % 500 == 0:
            print(f"  [{idx:5d}/{len(ids)}]  unk_gt_total={unk_gt_global_count}")

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

        # GT 집계 + global ID 부여
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

        # Known branch (threshold 무관 → 한 번만)
        order   = np.argsort(-p_scores)
        gt_used = [False] * len(gt_boxes)
        accumulate_known_ap(
            p_boxes[order], p_scores[order], p_labels[order],
            gt_boxes, gt_labels, gt_ignore, gt_used,
            n_known, idx_to_cls, known_keep_thr, iou_thr, known_preds)

        # P_u 계산
        if (not hasattr(pred, 'scores_all')
                or pred.scores_all.shape[-1] != n_known + n_attr
                or pred.scores_all.shape[0] == 0):
            continue

        p_u, p_b, _, _ = compute_pu(pred.scores_all.cpu(), n_known, gamma=gamma, norm=norm)
        p_u_np = p_u.numpy()
        p_b_np = p_b.numpy()

        # Greedy 매칭으로 (p_u, p_b, global_gt_id) 수집
        unk_local_used = set()
        for ui in np.argsort(-p_u_np):
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
                all_unk_preds.append((p_u_np[ui], p_b_np[ui],
                                      img_unk_gt_ids[best_gi]))
            else:
                all_unk_preds.append((p_u_np[ui], p_b_np[ui], -1))

    # Known AP (threshold 무관)
    known_map50, prev_map50, cur_map50 = _compute_known_map(
        known_classes, prev_set, known_preds, known_n_gt)

    print(f"\n  Known AP50 ({n_known:2d} cls) : {known_map50:.4f}")
    print(f"  Total unknown GT       : {unk_n_gt}")
    print(f"\n  unk_thr | U-REC50 | H-Score | TP / GT={unk_n_gt}")
    print(f"  {'-'*45}")

    sorted_preds = sorted(all_unk_preds, key=lambda x: -x[0])
    sweep_results = []
    for unk_thr in unk_thrs:
        hit_gts = set()
        for pu_i, pb_i, gt_gid in sorted_preds:
            if pu_i < unk_thr:
                break
            if pb_i >= pb_min and gt_gid >= 0 and gt_gid not in hit_gts:
                hit_gts.add(gt_gid)
        tp      = len(hit_gts)
        u_rec   = tp / (unk_n_gt + 1e-8)
        h_score = 2 * known_map50 * u_rec / (known_map50 + u_rec + 1e-8)
        print(f"  {unk_thr:.2f}    | {u_rec:.4f}  | {h_score:.4f}  | {tp}")
        sweep_results.append(dict(unk_thr=unk_thr, u_rec50=u_rec, h_score=h_score,
                                   unk_tp=tp, unk_n_gt=unk_n_gt,
                                   known_map50=known_map50))
    return sweep_results


def _compute_known_map(known_classes, prev_set, known_preds, known_n_gt):
    """sweep 모드용 known AP 집계 (출력 없이 수치만 반환)."""
    prev_aps, cur_aps, all_aps = [], [], []
    for cls in known_classes:
        ap = _compute_ap_val(known_preds[cls], known_n_gt[cls])
        if not np.isnan(ap):
            all_aps.append(ap)
            (prev_aps if cls in prev_set else cur_aps).append(ap)
    return (float(np.mean(all_aps)) if all_aps else 0.0,
            float(np.mean(prev_aps)) if prev_aps else None,
            float(np.mean(cur_aps)) if cur_aps else 0.0)


def _compute_ap_val(preds, n_gt):
    """compute_ap 와 동일 로직 (common 의존 없이 sweep 내부용)."""
    from owod_reproduce.util import compute_ap
    return compute_ap(preds, n_gt)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="OWOD M2: P_u unknown detection")
    parser.add_argument("--task",       type=int,   choices=[1, 2, 3, 4], default=None)
    parser.add_argument("--device",     default="cpu")
    parser.add_argument("--known-thr",  type=float, default=0.05)
    parser.add_argument("--unk-thr",    type=float, default=0.25,
                        help="P_u threshold")
    parser.add_argument("--pb-min",     type=float, default=0.10,
                        help="P_b 최솟값 (objectiveness floor)")
    parser.add_argument("--gamma",      type=int,   default=3,
                        help="top-γ for P_b")
    parser.add_argument("--norm",       choices=["l1", "softmax"], default="l1",
                        help="P_u 계산 정규화 방식: l1 (기본) 또는 softmax")
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
    print("  done. (head patched for scores_all)")

    coco_val_idx, coco_train_idx = build_gt_indices()
    ids = load_image_ids(args.quickeval)

    if args.sweep:
        SWEEP_THRS = (0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40)
        for tid in task_ids:
            run_task_sweep(tid, model, pipeline, coco_val_idx, coco_train_idx,
                           ids, known_keep_thr=args.known_thr,
                           pb_min=args.pb_min, gamma=args.gamma,
                           unk_thrs=SWEEP_THRS, norm=args.norm)
        return

    detector = functools.partial(_unknown_detector,
                                  unk_thr=args.unk_thr,
                                  pb_min=args.pb_min,
                                  gamma=args.gamma,
                                  norm=args.norm)
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
