from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
import cv2
from ultralytics import YOLO

# uses the helper from utils (same as we discussed)
from src.utils.geometry import pad_box


# ---- defaults (can be overridden by call site) ----
DET_IMGSZ = 1280
DET_CONF = 0.30
DET_IOU = 0.50


class YOLODetector:
    """
    OCR-friendly YOLOv8 detector.
      - imgsz=1280, conf≈0.30, iou≈0.50 by default
      - crops from ORIGINAL image (result.orig_img)
      - class-specific padding (via pad_box)
    """

    def __init__(self, model_path: str):
        print(f"[YOLODetector] Loading: {model_path}")
        self.model = YOLO(model_path)
        self.names = self.model.names
        if not isinstance(self.names, (list, dict)):
            raise RuntimeError("Unexpected model.names type")
        print(f"[YOLODetector] Classes: {self.names}")

    def detect(
        self,
        image_bgr: np.ndarray,
        *,
        conf: float = DET_CONF,
        iou: float = DET_IOU,
        imgsz: int = DET_IMGSZ,
    ):
        """
        Run detection and return (ultralytics_result, detections_list).
        detections_list items: { box: (x1,y1,x2,y2), conf: float, class_id: int, class_name: str }
        """
        r = self.model(image_bgr, imgsz=imgsz, conf=conf, iou=iou, verbose=False)[0]

        dets: List[Dict] = []
        if r.boxes is None or len(r.boxes) == 0:
            return r, dets

        boxes = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        clses = r.boxes.cls.cpu().numpy().astype(int)

        for xyxy, c, cid in zip(boxes, confs, clses):
            x1, y1, x2, y2 = map(int, xyxy)
            dets.append(
                {
                    "box": (x1, y1, x2, y2),
                    "conf": float(c),
                    "class_id": int(cid),
                    "class_name": self.names[int(cid)],
                }
            )
        return r, dets

    def get_padded_crops(
        self,
        result,
        detections: List[Dict],
        save_dir: Optional[str] = None,
    ) -> List[Dict]:
        """
        Crop from ORIGINAL (BGR) with class-specific padding.
        Returns: list of { cls, xyxy (page coords), crop (np.ndarray), path (optional) }
        """
        img = result.orig_img
        h, w = img.shape[:2]

        out: List[Dict] = []
        out_dir = Path(save_dir) if save_dir else None
        if out_dir:
            out_dir.mkdir(parents=True, exist_ok=True)

        for i, d in enumerate(detections):
            cls = d["class_name"]
            x1, y1, x2, y2 = pad_box(*d["box"], w, h, cls)
            crop = img[y1:y2, x1:x2]

            save_path = None
            if out_dir is not None:
                save_path = str(out_dir / f"crop_{i}_{cls}.png")
                cv2.imwrite(save_path, crop)

            out.append(
                {"cls": cls, "xyxy": (x1, y1, x2, y2), "crop": crop, "path": save_path}
            )

        return out
