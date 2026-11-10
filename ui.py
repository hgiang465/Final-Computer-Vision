"""Vẽ chữ Unicode (tiếng Việt có dấu) lên ảnh OpenCV. cv2.putText không hỗ trợ dấu nên dùng Pillow."""
import os
import unicodedata

import cv2
import numpy as np

_FONT_CANDIDATES = [
    os.environ.get("VSL_FONT", ""),
    r"C:\Windows\Fonts\segoeui.ttf", r"C:\Windows\Fonts\arial.ttf", r"C:\Windows\Fonts\tahoma.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf", "/Library/Fonts/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/TTF/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
]
_font_path = next((p for p in _FONT_CANDIDATES if p and os.path.exists(p)), None)
_cache = {}


def _font(size):
    from PIL import ImageFont
    if size not in _cache:
        _cache[size] = ImageFont.truetype(_font_path, size)
    return _cache[size]


def strip_accents(s):
    s = s.replace("đ", "d").replace("Đ", "D")
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def draw_texts(img, items, panel=None):
    """items: list of (text, (x, y), size, (B,G,R)). panel: (x1,y1,x2,y2) vẽ nền mờ phía sau chữ."""
    if panel is not None:
        x1, y1, x2, y2 = panel
        overlay = img.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
        img[:] = cv2.addWeighted(overlay, 0.55, img, 0.45, 0)
    if _font_path is None:   # không có font -> bỏ dấu để vẫn đọc được
        for text, (x, y), size, color in items:
            cv2.putText(img, strip_accents(text), (x, y + size), cv2.FONT_HERSHEY_SIMPLEX,
                        size / 32.0, color, 2, cv2.LINE_AA)
        return img
    from PIL import Image, ImageDraw
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(pil)
    for text, (x, y), size, color in items:
        d.text((x, y), text, font=_font(size), fill=(color[2], color[1], color[0]))
    img[:] = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    return img
