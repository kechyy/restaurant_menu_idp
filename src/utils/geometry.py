# =================== Geometry & text helpers ===================
def pad_box(x1,y1,x2,y2,w,h, cls_name: str):
    bw, bh = x2 - x1, y2 - y1
    if cls_name == "price":
        px, py = 0.06*bw, 0.08*bh
    elif cls_name == "section_title":
        px, py = 0.06*bw, 0.08*bh
    elif cls_name == "description":
        px, py = 0.05*bw, 0.06*bh
    else:  # dish_name
        px, py = 0.05*bw, 0.04*bh
    xx1 = max(0, int(round(x1 - px))); yy1 = max(0, int(round(y1 - py)))
    xx2 = min(w-1, int(round(x2 + px))); yy2 = min(h-1, int(round(y2 + py)))
    return (xx1, yy1, xx2, yy2)

def center_xy(box):  # (cx, cy)
    x1,y1,x2,y2 = box
    return (0.5*(x1+x2), 0.5*(y1+y2))

def y_bin(y, bin_px=10):
    return int(round(y / bin_px))

def horizontal_overlap_ratio(a, b):
    (ax1,_,ax2,_)=a; (bx1,_,bx2,_)=b
    inter = max(0, min(ax2,bx2) - max(ax1,bx1))
    union = (ax2-ax1) + (bx2-bx1) - inter
    return inter / max(union, 1)

