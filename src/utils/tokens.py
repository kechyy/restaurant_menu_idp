import re
from .geometry import horizontal_overlap_ratio

def y_bin(y, bin_px=10):
    return int(round(y / bin_px))

def tokens_in_box(tokens, box, margin=3, top_clip=None):
    x1,y1,x2,y2 = box
    if top_clip is not None: y1 = max(y1, int(top_clip))
    X1,Y1,X2,Y2 = x1-margin, y1-margin, x2+margin, y2+margin
    out=[]
    for t in tokens:
        tx1,ty1,tx2,ty2 = t["xyxy"]
        inter = max(0, min(X2,tx2) - max(X1,tx1)) * max(0, min(Y2,ty2) - max(Y1,ty1))
        if inter > 0: out.append(t)
    return out

def stitch_lines_with_breaks(tokens):
    if not tokens: return ""
    rows=[]
    for t in tokens:
        x1,y1,x2,y2 = t["xyxy"]
        rows.append((y_bin(0.5*(y1+y2)), x1, t["text"]))
    rows.sort(key=lambda z: (z[0], z[1]))
    lines, curr, buf = [], None, []
    for yb, x, txt in rows:
        if curr is None or yb == curr:
            buf.append(txt); curr = yb
        else:
            lines.append(" ".join(buf)); buf=[txt]; curr=yb
    if buf: lines.append(" ".join(buf))
    return "\n".join(lines).strip()

PRICE_RE = re.compile(r"""(?xi) ^\s*(?:£|\$|€)?\s* \d{1,3} (?:[.,]\d{2})?\s*$ """)

def join_split_numbers(tokens):
    out=[]; i=0
    while i < len(tokens):
        t = tokens[i]
        if i+2 < len(tokens):
            t1 = tokens[i+1]["text"]; t2 = tokens[i+2]["text"]
            if t["text"].isdigit() and t1 in {".", ","} and t2.isdigit():
                out.append({"text": f"{t['text']}{t1}{t2}", "xyxy": t["xyxy"], "conf": min(t["conf"], tokens[i+2]["conf"])})
                i += 3; continue
        out.append(t); i += 1
    return out

def extract_price_tokens(tokens):
    merged = join_split_numbers(tokens)
    out=[]
    for t in merged:
        if PRICE_RE.match(t["text"].replace(" ", "")):
            out.append(t)
    out.sort(key=lambda t: t["xyxy"][2])  # rightmost first
    return out

def compute_description_top_clip(desc_box, dish_boxes_above, min_horiz_overlap=0.25, margin_px=4):
    x1,y1,x2,y2 = desc_box
    clip_y = y1; best = y1
    for dbox in dish_boxes_above:
        if horizontal_overlap_ratio(desc_box, dbox) >= min_horiz_overlap:
            _, _, _, dy2 = dbox
            if dy2 <= y2 and dy2 > best:
                best = dy2
    if best > y1:
        clip_y = best + margin_px
    return clip_y