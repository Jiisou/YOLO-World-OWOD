"""
m1_eval.py — Method 1: "object" token unknown detection
=========================================================
Unknown 탐지: "object" 텍스트 프롬프트와의 CLIP 유사도 점수 사용.

texts = K known classes + ["object"]
object token score ~0.001-0.01 → unknown_thr=0.001 별도 적용.

Usage:
  conda activate py310
  cd /Users/jisu/Desktop/dev/cli/YOLO-World
  python owod_reproduce/m1_eval.py
  python owod_reproduce/m1_eval.py --task 2
  python owod_reproduce/m1_eval.py --quickeval 30
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

from owod_reproduce.util import (
    iou_xyxy,
    setup_eval, build_gt_indices, load_image_ids,
    run_task, print_summary, positive_int,
)

# ── M1 고유: 텍스트 구성 ───────────────────────────────────────────────────────

def _texts_fn(known_classes):
    """K known classes + ["object"] 토큰."""
    return [[c] for c in known_classes] + [["object"]]


# ── M1 고유: unknown detector ─────────────────────────────────────────────────

def _unknown_detector(_pred, p_boxes, p_scores, p_labels,
                       gt_boxes, gt_labels, gt_ignore,
                       known_set, n_known, iou_thr,
                       gt_used_known, *, unknown_thr=0.001):
    """
    "object" 토큰 예측으로 unknown TP 카운트.

    - object_idx = n_known (texts 마지막 인덱스)
    - gt_used_known 을 공유 → known branch 가 이미 사용한 GT 는 재사용 불가.
    - score 내림차순으로 순회.
    """
    object_idx = n_known

    obj_mask   = p_labels == object_idx
    obj_boxes  = p_boxes[obj_mask]
    obj_scores = p_scores[obj_mask]

    unk_tp = 0
    for i in np.argsort(-obj_scores):
        if obj_scores[i] < unknown_thr:
            continue
        pbox = obj_boxes[i]

        best_iou, best_gi = 0.0, -1
        for gi, gbox in enumerate(gt_boxes):
            if gt_ignore[gi]:
                continue
            v = iou_xyxy(pbox, gbox)
            if v > best_iou:
                best_iou, best_gi = v, gi

        if (best_iou >= iou_thr and best_gi >= 0
                and gt_labels[best_gi] not in known_set
                and not gt_used_known[best_gi]):
            unk_tp += 1
            gt_used_known[best_gi] = True  # known branch 와 공유 플래그 업데이트

    return unk_tp


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="OWOD M1: object token")
    parser.add_argument("--task",        type=int,   choices=[1, 2, 3, 4], default=None)
    parser.add_argument("--device",      default="cpu")
    parser.add_argument("--known-thr",   type=float, default=0.05)
    parser.add_argument("--unknown-thr", type=float, default=0.001,
                        help='"object" 토큰 confidence threshold')
    parser.add_argument("--quickeval",   nargs="?",  const=50,
                        type=positive_int, default=None, metavar="N",
                        help="VOC N + COCO N 이미지로 빠른 평가 (기본 N=50)")
    args = parser.parse_args()

    task_ids = [args.task] if args.task else [1, 2, 3, 4]

    print("Loading model …")
    model, pipeline = setup_eval(device=args.device)
    print("  done.")

    coco_val_idx, coco_train_idx = build_gt_indices()
    ids = load_image_ids(args.quickeval)

    detector = functools.partial(_unknown_detector, unknown_thr=args.unknown_thr)

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
