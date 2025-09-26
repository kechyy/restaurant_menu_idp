from typing import List, Dict, Tuple, Optional
import numpy as np
import ast

# Try both engines; fall back gracefully
try:
    import easyocr  # type: ignore
except Exception:
    easyocr = None

try:
    from paddleocr import PaddleOCR  # type: ignore
    _HAS_PADDLE = True
except Exception:
    PaddleOCR = None
    _HAS_PADDLE = False

OCR_MIN_CONF = 0.40  # default threshold


def _xyxy_from_points(points) -> Tuple[int, int, int, int]:
    xs = [int(p[0]) for p in points]
    ys = [int(p[1]) for p in points]
    return (min(xs), min(ys), max(xs), max(ys))


class MultiEngineOCRProcessor:
    """
    Robust OCR wrapper that:
      - prefers PaddleOCR (angle_cls configurable),
      - falls back to EasyOCR if Paddle returns nothing,
      - normalises outputs to {text, conf, xyxy}.
    Pass a config dict like:
      {
        "easyocr": {"enabled": True, "languages": ["en"], "gpu_enabled": False},
        "paddleocr": {"enabled": True, "lang": "en", "use_angle_cls": True}
      }
    """

    def __init__(self, cfg: dict):
        self.easy = None
        self.paddle = None

        # EasyOCR
        if easyocr and cfg.get("easyocr", {}).get("enabled", False):
            langs = cfg["easyocr"].get("languages", ["en"])
            if isinstance(langs, str):
                try:
                    langs = ast.literal_eval(langs)  # allow "['en']" string from YAML
                except Exception:
                    langs = [langs]
            self.easy = easyocr.Reader(
                langs, gpu=cfg["easyocr"].get("gpu_enabled", False)
            )

        # PaddleOCR
        if _HAS_PADDLE and cfg.get("paddleocr", {}).get("enabled", True):
            self.paddle = PaddleOCR(
                lang=cfg["paddleocr"].get("lang", "en"),
                use_angle_cls=cfg["paddleocr"].get("use_angle_cls", True),
            )

    # --- Paddle helpers (handle API differences) ---
    def _paddle_call(self, img):
        if self.paddle is None:
            return None
        try:
            return self.paddle.ocr(img)  # newer
        except TypeError:
            try:
                return self.paddle.predict(img)  # some builds
            except Exception:
                return None

    @staticmethod
    def _parse_paddle_line(line):
        """
        Accepts multiple Paddle formats and returns (points, text, conf) or None:
          - [points, (text, conf)]
          - [points, (text, conf), ...]
          - {'points': [[x,y],...], 'transcription': str, 'score': float}
        """
        # dict-style
        if isinstance(line, dict):
            pts = line.get("points") or line.get("bbox") or line.get("poly")
            txt = line.get("transcription") or line.get("text")
            sc = line.get("score") or line.get("conf") or line.get("confidence")
            if pts is not None and txt is not None and sc is not None:
                return pts, txt, sc
            return None

        # list/tuple-style
        if not isinstance(line, (list, tuple)) or len(line) < 2:
            return None

        pts = line[0]
        tc = line[1]
        txt, conf = None, None

        if isinstance(tc, (list, tuple)) and len(tc) >= 2:
            txt, conf = tc[0], tc[1]
        elif len(line) >= 3 and isinstance(line[2], (list, tuple)) and len(line[2]) >= 2:
            txt, conf = line[2][0], line[2][1]

        if pts is None or txt is None or conf is None:
            return None
        return pts, txt, conf

    @staticmethod
    def _iter_paddle_lines(res):
        """Yield line entries regardless of page/flat layout."""
        if not isinstance(res, list):
            return
        is_pages = (
            res
            and isinstance(res[0], list)
            and res[0]
            and isinstance(res[0][0], (list, tuple, dict))
        )
        if is_pages:
            for page in res:
                if isinstance(page, list):
                    for line in page:
                        yield line
        else:
            for line in res:
                yield line

    # --- Public API ---
    def ocr_full(self, image_bgr: np.ndarray) -> List[Dict]:
        """OCR the full image. Return list of {text, conf, xyxy} in page coords."""
        out: List[Dict] = []

        # Prefer Paddle
        if self.paddle:
            res = self._paddle_call(image_bgr)
            for line in self._iter_paddle_lines(res or []):
                parsed = self._parse_paddle_line(line)
                if not parsed:
                    continue
                pts, text, conf = parsed
                try:
                    if text and float(conf) >= OCR_MIN_CONF:
                        out.append(
                            {"text": text.strip(), "conf": float(conf), "xyxy": _xyxy_from_points(pts)}
                        )
                except Exception:
                    continue

        # Fallback to EasyOCR if Paddle gave nothing
        if not out and self.easy:
            try:
                for bbox, text, prob in self.easy.readtext(image_bgr):
                    if prob >= OCR_MIN_CONF:
                        x1, y1 = map(int, bbox[0])
                        x2, y2 = map(int, bbox[2])
                        out.append(
                            {
                                "text": text.strip(),
                                "conf": float(prob),
                                "xyxy": (x1, y1, x2, y2),
                            }
                        )
            except Exception:
                pass

        return out

    def ocr_crop(self, crop_bgr: np.ndarray, offset_xy: Tuple[int, int]) -> List[Dict]:
        """OCR a crop and translate boxes back to page coords using the (ox, oy) offset."""
        ox, oy = offset_xy
        out: List[Dict] = []

        if self.paddle:
            res = self._paddle_call(crop_bgr)
            for line in self._iter_paddle_lines(res or []):
                parsed = self._parse_paddle_line(line)
                if not parsed:
                    continue
                pts, text, conf = parsed
                try:
                    if text and float(conf) >= OCR_MIN_CONF:
                        x1, y1, x2, y2 = _xyxy_from_points(pts)
                        out.append(
                            {
                                "text": text.strip(),
                                "conf": float(conf),
                                "xyxy": (x1 + ox, y1 + oy, x2 + ox, y2 + oy),
                            }
                        )
                except Exception:
                    continue

        if not out and self.easy:
            try:
                for bbox, text, prob in self.easy.readtext(crop_bgr):
                    if prob >= OCR_MIN_CONF:
                        x1, y1 = map(int, bbox[0])
                        x2, y2 = map(int, bbox[2])
                        out.append(
                            {
                                "text": text.strip(),
                                "conf": float(prob),
                                "xyxy": (x1 + ox, y1 + oy, x2 + ox, y2 + oy),
                            }
                        )
            except Exception:
                pass

        return out
