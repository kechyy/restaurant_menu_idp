# src/tools/labelstudio_builder.py

from __future__ import annotations
import argparse, os, json, uuid
from pathlib import Path
from typing import Optional, Dict, Set   # <-- ADD THIS LINE
import cv2
from ultralytics import YOLO
import yaml

# === imports from your modules ===
from src.ocr.multi_engine_ocr import MultiEngineOCRProcessor
from src.utils.geometry import pad_box, center_xy
from src.utils.tokens import (
    y_bin, tokens_in_box, stitch_lines_with_breaks,
    extract_price_tokens, compute_description_top_clip
)

# --- config loaders ---
def load_yaml(p: str) -> dict:
    with open(p, "r") as f:
        return yaml.safe_load(f) or {}

def canon_name(raw: str, name_map: Dict[str, Optional[str]], keep: Set[str]) -> Optional[str]:
    # exact match first
    if raw in name_map:
        return name_map[raw]
    low = raw.lower()
    mapped = name_map.get(low, low)
    # drop unknowns (anything not in keep or mapped to None)
    if mapped is None:
        return None
    return mapped if mapped in keep else None

# ============== MAIN BUILDER ==============
def build_label_studio_json(
    images_dir: str,
    yolo_model_path: str,
    ocr_cfg: dict,
    detect_cfg: dict,
    pipeline_cfg: dict,
    output_json_path: str,
    save_crops_dir: str | None = None,
):
    # thresholds from detect.yaml
    DET_IMGSZ = detect_cfg["imgsz"]
    DET_CONF  = detect_cfg["conf"]
    DET_IOU   = detect_cfg["iou"]

    # class canon + mapping from pipeline.yaml
    KEEP: Set[str] = set(pipeline_cfg["keep_classes"])
    NAME_MAP: Dict[str, Optional[str]] = pipeline_cfg["name_map"]

    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    model = YOLO(str(yolo_model_path))
    names = model.names
    print("YOLO class names:", names)

    ocr = MultiEngineOCRProcessor(ocr_cfg)

    # Collect images
    paths = []
    for ext in ("*.jpg","*.jpeg","*.png","*.bmp"):
        paths += list(Path(images_dir).glob(ext))
    paths = [str(p) for p in sorted(paths)]
    if not paths:
        print("No images found."); return

    if save_crops_dir:
        os.makedirs(save_crops_dir, exist_ok=True)

    tasks=[]
    for pth in paths:
        img = cv2.imread(pth)
        if img is None:
            print("Bad image:", pth); continue
        H, W = img.shape[:2]

        # 1) Detect
        r = model(img, imgsz=DET_IMGSZ, conf=DET_CONF, iou=DET_IOU, verbose=False)[0]
        raw_dets=[]
        if r.boxes is not None and len(r.boxes):
            for xyxy, c, cid in zip(
                r.boxes.xyxy.cpu().numpy(),
                r.boxes.conf.cpu().numpy(),
                r.boxes.cls.cpu().numpy().astype(int)
            ):
                raw_name = names[int(cid)]
                cname = canon_name(raw_name, NAME_MAP, KEEP)
                if cname is None:   # drop unknowns
                    continue
                x1,y1,x2,y2 = map(int, xyxy)
                raw_dets.append({"cls":cname, "box":(x1,y1,x2,y2), "conf": float(c)})

        # 2) Padded crops
        dish_boxes=[]
        crops=[]
        for i, d in enumerate(raw_dets):
            x1,y1,x2,y2 = pad_box(*d["box"], W, H, d["cls"])
            if d["cls"] == "dish_name":
                dish_boxes.append((x1,y1,x2,y2))
            crop = img[y1:y2, x1:x2]
            if save_crops_dir:
                out_path = Path(save_crops_dir)/f"{Path(pth).stem}_crop{i}_{d['cls']}.png"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(out_path), crop)
            crops.append({"cls": d["cls"], "xyxy":(x1,y1,x2,y2), "crop": crop})

        # 3) Full-page OCR
        full_tokens = ocr.ocr_full(img)
        price_full  = extract_price_tokens(full_tokens)

        # 4) Fill each ROI
        results=[]
        for c in crops:
            cls = c["cls"]; x1,y1,x2,y2 = c["xyxy"]
            top_clip = None
            if cls == "description" and dish_boxes:
                dishes_above = [db for db in dish_boxes if db[3] <= y2]
                top_clip = compute_description_top_clip(
                    (x1,y1,x2,y2), dishes_above, min_horiz_overlap=0.25, margin_px=4
                )

            roi_tokens = tokens_in_box(full_tokens, c["xyxy"], margin=3, top_clip=top_clip)
            text_value = ""

            if cls == "price":
                rp = extract_price_tokens(roi_tokens)
                if rp:
                    rp.sort(key=lambda t: (t["conf"], t["xyxy"][2]), reverse=True)
                    text_value = rp[0]["text"].strip()
                else:
                    cx, cy = center_xy((x1,y1,x2,y2))
                    dy = y_bin(cy)
                    cands = [p for p in price_full if abs(
                        y_bin(0.5*(p["xyxy"][1]+p["xyxy"][3])) - dy
                    ) <= 1]
                    if cands:
                        cands.sort(key=lambda p: abs(
                            0.5*(p["xyxy"][0]+p["xyxy"][2]) - cx
                        ))
                        text_value = cands[0]["text"].strip()

            elif cls in {"dish_name","section_title"}:
                if roi_tokens:
                    ys = [y_bin(0.5*(t["xyxy"][1]+t["xyxy"][3])) for t in roi_tokens]
                    top = min(ys)
                    line = [t for t in roi_tokens if y_bin(0.5*(t["xyxy"][1]+t["xyxy"][3])) == top]
                    line.sort(key=lambda t: t["xyxy"][0])
                    text_value = " ".join(t["text"] for t in line).strip()

            elif cls == "description":
                if roi_tokens:
                    text_value = stitch_lines_with_breaks(roi_tokens)

            # Fallback: crop OCR if empty
            if not text_value:
                fb = ocr.ocr_crop(c["crop"], offset_xy=(x1,y1))
                if cls == "price":
                    fp = extract_price_tokens(fb)
                    if fp:
                        fp.sort(key=lambda t: (t["conf"], t["xyxy"][2]), reverse=True)
                        text_value = fp[0]["text"].strip()
                elif cls in {"dish_name","section_title"}:
                    if fb:
                        ys = [y_bin(0.5*(t["xyxy"][1]+t["xyxy"][3])) for t in fb]
                        top = min(ys)
                        line = [t for t in fb if y_bin(0.5*(t["xyxy"][1]+t["xyxy"][3])) == top]
                        line.sort(key=lambda t: t["xyxy"][0])
                        text_value = " ".join(t["text"] for t in line).strip()
                elif cls == "description":
                    if fb:
                        text_value = stitch_lines_with_breaks(fb)

            # 5) Emit Label Studio JSON using padded box
            x_pct = (x1 / W) * 100.0; y_pct = (y1 / H) * 100.0
            w_pct = ((x2-x1)/W) * 100.0; h_pct = ((y2-y1)/H) * 100.0
            region_id = str(uuid.uuid4())

            results.append({
                "id": region_id, "from_name": "label", "to_name": "image",
                "type": "rectanglelabels",
                "value": {"x": x_pct, "y": y_pct, "width": w_pct, "height": h_pct, "rotation": 0,
                          "rectanglelabels": [cls]}
            })
            results.append({
                "id": region_id, "from_name": "transcription", "to_name": "image",
                "type": "textarea",
                "value": {"x": x_pct, "y": y_pct, "width": w_pct, "height": h_pct, "rotation": 0,
                          "text": [text_value] if text_value else []}
            })

        task = {
            "data": {"image": f"/data/local-files/?d=images/{Path(pth).name}"},
            "annotations": [{"result": results}]
        }
        tasks.append(task)

    Path(output_json_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_json_path, "w") as f:
        json.dump(tasks, f, indent=2)
    print(f"✅ Saved Label Studio JSON → {output_json_path}")

# ============== CLI ==============
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir",        required=True)
    ap.add_argument("--yolo_model_path",   required=True)
    ap.add_argument("--ocr_config",        default="configs/ocr.yaml")
    ap.add_argument("--detect_config",     default="configs/detect.yaml")
    ap.add_argument("--pipeline_config",   default="configs/pipeline.yaml")
    ap.add_argument("--output_json_path",  required=True)
    ap.add_argument("--save_crops_dir",    default=None)
    args = ap.parse_args()

    ocr_cfg      = load_yaml(args.ocr_config)
    detect_cfg   = load_yaml(args.detect_config)
    pipeline_cfg = load_yaml(args.pipeline_config)

    build_label_studio_json(
        images_dir=args.images_dir,
        yolo_model_path=args.yolo_model_path,
        ocr_cfg=ocr_cfg,
        detect_cfg=detect_cfg,
        pipeline_cfg=pipeline_cfg,
        output_json_path=args.output_json_path,
        save_crops_dir=args.save_crops_dir,
    )
