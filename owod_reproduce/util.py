"""
owod_reproduce/common.py
========================
M1 / M2 eval 공통 유틸리티 모듈.

제공 항목
  Constants  : DATA_ROOT, VOC_ROOT, COCO_ROOT, CHECKPOINT, CONFIG_FILE,
               IMAGE_LIST, TASK_SPLITS, CLASS_ALIAS
  Helpers    : canon, resolve_image_path, select_quickeval_ids, positive_int
  GT parsers : parse_voc_xml, build_coco_index, parse_coco_record, load_gt
  Metrics    : iou_xyxy, voc_ap_11pt, compute_ap
  Setup      : build_test_pipeline, setup_eval, build_gt_indices, load_image_ids
  Eval loop  : accumulate_known_ap, compute_task_metrics, run_task, print_summary
"""

import sys
import json
import argparse
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from mmengine.config import Config
from mmengine.dataset import Compose
from mmdet.utils import get_test_pipeline_cfg

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_ROOT   = Path("/Users/jisu/Desktop/dev/cli/OW_OVD/reproduce_owod/data")
VOC_ROOT    = DATA_ROOT / "VOCtest_devkit" / "VOC2007"
COCO_ROOT   = DATA_ROOT / "coco2017"
CHECKPOINT  = str(Path("/Users/jisu/Desktop/dev/cli/OW_OVD/reproduce_owod") /
                  "yolo_world_l_stage2-b3e3dc3f.pth")
CONFIG_FILE = str(Path(__file__).resolve().parents[1] / "configs/pretrain" /
                  "yolo_world_v2_l_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_1280ft_lvis_minival.py")
IMAGE_LIST  = DATA_ROOT / "owod_all_task_test_nodup.txt"

# ── Task class splits (각 20개) ─────────────────────────────────────────────────
TASK_SPLITS = [
    # Task 1: VOC 20
    ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
     "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
     "pottedplant", "sheep", "sofa", "train", "tvmonitor"],
    # Task 2: COCO +20
    ["truck", "traffic light", "fire hydrant", "stop sign", "parking meter",
     "bench", "elephant", "bear", "zebra", "giraffe",
     "backpack", "umbrella", "handbag", "tie", "suitcase",
     "microwave", "oven", "toaster", "sink", "refrigerator"],
    # Task 3: COCO +20
    ["frisbee", "skis", "snowboard", "sports ball", "kite",
     "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
     "banana", "apple", "sandwich", "orange", "broccoli",
     "carrot", "hot dog", "pizza", "donut", "cake"],
    # Task 4: COCO +20
    ["bed", "toilet", "laptop", "mouse",
     "remote", "keyboard", "cell phone", "book", "clock",
     "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
     "wine glass", "cup", "fork", "knife", "spoon", "bowl"],
]

CLASS_ALIAS = {
    "airplane":     "aeroplane",
    "motorcycle":   "motorbike",
    "couch":        "sofa",
    "tv":           "tvmonitor",
    "dining table": "diningtable",
    "potted plant": "pottedplant",
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def canon(name: str) -> str:
    name = name.strip().lower()
    return CLASS_ALIAS.get(name, name)


def resolve_image_path(img_id: str):
    fn = img_id + ".jpg"
    p = VOC_ROOT / "JPEGImages" / fn
    if p.exists():
        return p, "voc"
    p = COCO_ROOT / "val2017" / fn
    if p.exists():
        return p, "coco_val"
    p = COCO_ROOT / "train2017" / fn
    if p.exists():
        return p, "coco_train"
    raise FileNotFoundError(f"Cannot resolve: {img_id}")


def select_quickeval_ids(ids, voc_limit=50, coco_limit=50):
    voc_ids, coco_ids = [], []
    for img_id in ids:
        try:
            _, source = resolve_image_path(img_id)
        except FileNotFoundError:
            continue
        if source == "voc" and len(voc_ids) < voc_limit:
            voc_ids.append(img_id)
        elif source in {"coco_val", "coco_train"} and len(coco_ids) < coco_limit:
            coco_ids.append(img_id)
        if len(voc_ids) >= voc_limit and len(coco_ids) >= coco_limit:
            break
    return voc_ids + coco_ids


def positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be positive integer")
    return value


# ── GT parsers ─────────────────────────────────────────────────────────────────

def parse_voc_xml(xml_path: Path):
    root = ET.parse(xml_path).getroot()
    boxes, labels, ignore = [], [], []
    for obj in root.findall("object"):
        name = canon(obj.find("name").text)
        diff = obj.find("difficult")
        b    = obj.find("bndbox")
        boxes.append([float(b.find("xmin").text), float(b.find("ymin").text),
                      float(b.find("xmax").text), float(b.find("ymax").text)])
        labels.append(name)
        ignore.append(bool(int(diff.text)) if diff is not None else False)
    return boxes, labels, ignore


def build_coco_index(json_path: Path):
    with open(json_path) as f:
        data = json.load(f)
    catid_name = {c["id"]: canon(c["name"]) for c in data["categories"]}
    anns_by_img = defaultdict(list)
    for ann in data["annotations"]:
        anns_by_img[ann["image_id"]].append(ann)
    file_to_rec = {}
    for img in data["images"]:
        file_to_rec[img["file_name"]] = {"id": img["id"],
                                          "anns": anns_by_img[img["id"]]}
    return file_to_rec, catid_name


def parse_coco_record(filename, coco_idx):
    file_to_rec, catid_name = coco_idx
    rec = file_to_rec[filename]
    boxes, labels, ignore = [], [], []
    for ann in rec["anns"]:
        x, y, w, h = ann["bbox"]
        boxes.append([x, y, x + w, y + h])
        labels.append(catid_name[ann["category_id"]])
        ignore.append(bool(ann.get("iscrowd", 0)))
    return boxes, labels, ignore


def load_gt(img_path, source, coco_val_idx, coco_train_idx):
    if source == "voc":
        xml = img_path.parent.parent / "Annotations" / f"{img_path.stem}.xml"
        return parse_voc_xml(xml)
    elif source == "coco_val":
        return parse_coco_record(img_path.name, coco_val_idx)
    elif source == "coco_train":
        return parse_coco_record(img_path.name, coco_train_idx)
    raise ValueError(source)


# ── IoU + AP ───────────────────────────────────────────────────────────────────

def iou_xyxy(a, b) -> float:
    xi1, yi1 = max(a[0], b[0]), max(a[1], b[1])
    xi2, yi2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, xi2 - xi1) * max(0.0, yi2 - yi1)
    if inter == 0.0:
        return 0.0
    return inter / ((a[2]-a[0])*(a[3]-a[1]) +
                    (b[2]-b[0])*(b[3]-b[1]) - inter + 1e-8)


def voc_ap_11pt(recalls, precisions) -> float:
    ap = 0.0
    for thr in np.arange(0.0, 1.1, 0.1):
        p = [p for r, p in zip(recalls, precisions) if r >= thr]
        ap += max(p) if p else 0.0
    return ap / 11.0


def compute_ap(preds, n_gt: int) -> float:
    if n_gt == 0:
        return float("nan")
    if not preds:
        return 0.0
    preds = sorted(preds, key=lambda x: -x[0])
    tp, fp = 0, 0
    recalls, precisions = [], []
    for _, matched in preds:
        if matched:
            tp += 1
        else:
            fp += 1
        recalls.append(tp / n_gt)
        precisions.append(tp / (tp + fp))
    return voc_ap_11pt(recalls, precisions)


# ── Setup helpers ──────────────────────────────────────────────────────────────

def build_test_pipeline(cfg):
    raw = get_test_pipeline_cfg(cfg=cfg)
    return Compose([s for s in raw
                    if "LoadAnnotations" not in s.get("type", "")])


def setup_eval(cfg_file=CONFIG_FILE, ckpt=CHECKPOINT, device="cpu"):
    """모델 로드 + pipeline 빌드. (model, pipeline) 반환."""
    cfg = Config.fromfile(cfg_file)
    from mmyolo.registry import MODELS as MMYOLO_MODELS
    from mmengine.runner import load_checkpoint
    from mmengine.registry import init_default_scope
    init_default_scope("mmyolo")
    model = MMYOLO_MODELS.build(cfg.model)
    load_checkpoint(model, ckpt, map_location=device)
    model = model.to(device).eval()
    pipeline = build_test_pipeline(cfg)
    return model, pipeline


def build_gt_indices():
    """COCO val/train annotation index 빌드. (coco_val_idx, coco_train_idx) 반환."""
    print("Building COCO annotation indices …")
    coco_val_idx   = build_coco_index(COCO_ROOT / "annotations/instances_val2017.json")
    train_ann      = COCO_ROOT / "annotations/instances_train2017.json"
    coco_train_idx = build_coco_index(train_ann) if train_ann.exists() else None
    print("  done.")
    return coco_val_idx, coco_train_idx


def load_image_ids(quickeval_n=None):
    """IMAGE_LIST 로드. quickeval_n 지정 시 VOC N + COCO N subset 반환."""
    with open(IMAGE_LIST) as f:
        ids = [x.strip() for x in f if x.strip()]
    if quickeval_n is not None:
        ids = select_quickeval_ids(ids,
                                   voc_limit=quickeval_n, coco_limit=quickeval_n)
        print(f"Quick eval enabled: using VOC {quickeval_n} + COCO {quickeval_n} images.")
    print(f"Total images: {len(ids)}  "
          f"(VOC: {sum(1 for x in ids if len(x) <= 6)}"
          f", COCO: {sum(1 for x in ids if len(x) > 6)})")
    return ids


# ── Shared eval components ─────────────────────────────────────────────────────

def accumulate_known_ap(p_boxes_s, p_scores_s, p_labels_s,
                         gt_boxes, gt_labels, gt_ignore, gt_used,
                         n_known, idx_to_cls, known_keep_thr, iou_thr,
                         known_preds):
    """
    score 내림차순 정렬된 예측에 대해 known class AP 누적.

    - plabel >= n_known 인 박스 (objectiveness / object 토큰) 는 건너뜀.
      M1: object_idx == n_known → 동일 조건으로 처리됨.
      M2: objectiveness prompt index (K ~ K+9) → 동일 조건으로 처리됨.
    - gt_used 는 in-place 업데이트 (known branch 에 의한 GT 소비 기록).
    """
    for pbox, pscore, plabel in zip(p_boxes_s, p_scores_s, p_labels_s):
        if int(plabel) >= n_known:
            continue
        if pscore < known_keep_thr:
            continue
        cls_name = idx_to_cls[int(plabel)]

        best_iou, best_gi = 0.0, -1
        for gi, gbox in enumerate(gt_boxes):
            if gt_ignore[gi]:
                continue
            v = iou_xyxy(pbox, gbox)
            if v > best_iou:
                best_iou, best_gi = v, gi

        if best_iou < iou_thr or best_gi < 0:
            known_preds[cls_name].append((pscore, False))
            continue

        gt_cls = gt_labels[best_gi]
        if gt_cls == cls_name and not gt_used[best_gi]:
            known_preds[cls_name].append((pscore, True))
            gt_used[best_gi] = True
        else:
            known_preds[cls_name].append((pscore, False))


def compute_task_metrics(task_id, known_classes, prev_classes, prev_set,
                          known_preds, known_n_gt, unk_tp, unk_n_gt, skipped):
    """per-class AP 표 출력 후 지표 계산. result dict 반환."""
    n_known  = len(known_classes)
    n_cur    = n_known - len(prev_classes)
    prev_aps, cur_aps, all_aps = [], [], []

    print(f"\n  {'class':22s} {'GT':>6} {'AP50':>8} {'split':>6}")
    print(f"  {'-'*46}")

    for cls in known_classes:
        ap     = compute_ap(known_preds[cls], known_n_gt[cls])
        n      = known_n_gt[cls]
        tag    = f"{ap:.4f}" if not np.isnan(ap) else "  N/A "
        bucket = "prev" if cls in prev_set else "cur"
        print(f"  {cls:22s} {n:6d} {tag:>8} {bucket:>6}")
        if not np.isnan(ap):
            all_aps.append(ap)
            (prev_aps if cls in prev_set else cur_aps).append(ap)

    prev_map50  = float(np.mean(prev_aps))  if prev_aps  else None
    cur_map50   = float(np.mean(cur_aps))   if cur_aps   else 0.0
    known_map50 = float(np.mean(all_aps))   if all_aps   else 0.0
    u_rec50     = unk_tp / (unk_n_gt + 1e-8)
    h_score     = 2 * known_map50 * u_rec50 / (known_map50 + u_rec50 + 1e-8)

    print(f"\n  {'─'*52}")
    if prev_map50 is not None:
        print(f"  Prev  AP50 ({len(prev_classes):2d} cls)  : {prev_map50:.4f}")
    print(f"  Cur   AP50 ({n_cur:2d} cls)  : {cur_map50:.4f}")
    print(f"  Known AP50 ({n_known:2d} cls) : {known_map50:.4f}")
    print(f"  U-REC  @50            : {u_rec50:.4f}  (TP={unk_tp} / GT={unk_n_gt})")
    print(f"  H-Score               : {h_score:.4f}")
    print(f"  (skipped: {skipped})")

    return dict(task_id=task_id,
                prev_map50=prev_map50, cur_map50=cur_map50,
                known_map50=known_map50, u_rec50=u_rec50, h_score=h_score,
                unk_tp=unk_tp, unk_n_gt=unk_n_gt, skipped=skipped)


def run_task(task_id, model, pipeline, coco_val_idx, coco_train_idx, ids,
             texts_fn, unknown_detector,
             known_keep_thr=0.05, iou_thr=0.50):
    """
    공통 평가 루프.

    Args:
        texts_fn         : Callable[[list[str]], list[list[str]]]
                           known_classes → 모델에 넘길 texts 구성.
        unknown_detector : Callable
                           시그니처:
                             fn(pred, p_boxes, p_scores, p_labels,
                                gt_boxes, gt_labels, gt_ignore,
                                known_set, n_known, iou_thr,
                                gt_used_known) -> int
                           이미지 1장에 대한 unk_tp 증분을 반환.
                           gt_used_known 은 known branch 가 채운 GT 소비 목록.
                           M1 : 공유하여 이중 매칭 방지.
                           M2 : 무시하고 독립 unk_gt_used 사용.
    """
    prev_classes  = [c for split in TASK_SPLITS[:task_id - 1] for c in split]
    cur_classes   = TASK_SPLITS[task_id - 1]
    known_classes = prev_classes + cur_classes
    known_set     = set(known_classes)
    prev_set      = set(prev_classes)
    n_known       = len(known_classes)
    idx_to_cls    = {i: c for i, c in enumerate(known_classes)}

    texts = texts_fn(known_classes)

    known_preds = defaultdict(list)
    known_n_gt  = defaultdict(int)
    unk_n_gt, unk_tp, skipped = 0, 0, 0

    print(f"\n{'='*65}")
    print(f"Task {task_id}  |  prev={len(prev_classes)}  cur={len(cur_classes)}  "
          f"known={n_known}  unknown=everything else")
    print(f"{'='*65}")

    for idx, img_id in enumerate(ids):
        if idx % 500 == 0:
            print(f"  [{idx:5d}/{len(ids)}]  unk_tp={unk_tp}  unk_n_gt={unk_n_gt}")

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

        # GT 집계
        for lbl, ign in zip(gt_labels, gt_ignore):
            if ign:
                continue
            if lbl in known_set:
                known_n_gt[lbl] += 1
            else:
                unk_n_gt += 1

        # 추론
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

        # score 내림차순 정렬
        order   = np.argsort(-p_scores)
        gt_used = [False] * len(gt_boxes)

        # ── Known branch ─────────────────────────────────────────────────────
        accumulate_known_ap(
            p_boxes[order], p_scores[order], p_labels[order],
            gt_boxes, gt_labels, gt_ignore, gt_used,
            n_known, idx_to_cls, known_keep_thr, iou_thr,
            known_preds)

        # ── Unknown branch (method-specific) ─────────────────────────────────
        unk_tp += unknown_detector(
            pred, p_boxes, p_scores, p_labels,
            gt_boxes, gt_labels, gt_ignore,
            known_set, n_known, iou_thr, gt_used)

    return compute_task_metrics(
        task_id, known_classes, prev_classes, prev_set,
        known_preds, known_n_gt, unk_tp, unk_n_gt, skipped)


def print_summary(results):
    """전체 task 요약 테이블 출력."""
    print("\n\n" + "=" * 75)
    print("SUMMARY")
    print("=" * 75)
    print(f"{'Task':>6} | {'Prev AP50':>10} | {'Cur AP50':>9} | "
          f"{'Known AP50':>10} | {'U-REC50':>8} | {'H-Score':>8}")
    print("-" * 75)
    for r in results:
        prev = f"{r['prev_map50']:.4f}" if r['prev_map50'] is not None else "     N/A"
        print(f"  T{r['task_id']:1d}   | {prev:>10} | {r['cur_map50']:>9.4f} "
              f"| {r['known_map50']:>10.4f} | {r['u_rec50']:>8.4f} "
              f"| {r['h_score']:>8.4f}")
    print("=" * 75)
