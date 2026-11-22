"""Ghép chữ cái + dấu thành tiếng Việt có dấu, và ghép câu từ cử chỉ (từ) + chữ cái (đánh vần).

Chạy  python vietnamese.py  để tự kiểm tra các quy tắc ghép.

Nguyên tắc:
- Chữ cái được lưu dạng chữ cái gốc (H, O, N, G ...). Dấu KHÔNG gắn vào "ký tự trước đó" một cách máy móc,
  mà được gắn vào đúng chữ cần dấu rồi hiển thị lại toàn bộ từ. Nhờ vậy dấu ký sau từng chữ (O rồi dấu mũ)
  hay ký sau cả từ (HONG rồi dấu mũ, dấu huyền) đều ra cùng kết quả "HỒNG".
- Dấu chất lượng:  DAU_MU (Â Ê Ô) · DAU_MOC (Ơ Ư) · DAU_TRANG (Ă) · DAU_GACH (Đ)
- Dấu thanh:       DAU_SAC · DAU_HUYEN · DAU_HOI · DAU_NGA · DAU_NANG
- Vị trí dấu thanh theo quy tắc chính tả phổ biến (kiểu mới: hòa, thủy). Xem _tone_index().
"""
import unicodedata

TONES = {"DAU_SAC": "\u0301", "DAU_HUYEN": "\u0300", "DAU_HOI": "\u0309",
         "DAU_NGA": "\u0303", "DAU_NANG": "\u0323"}
QUALITY = {"DAU_MU": ("AEO", "\u0302"), "DAU_MOC": ("OU", "\u031B"),
           "DAU_TRANG": ("A", "\u0306"), "DAU_GACH": ("D", None)}
MARK_NAMES = {"DAU_SAC": "dấu sắc", "DAU_HUYEN": "dấu huyền", "DAU_HOI": "dấu hỏi", "DAU_NGA": "dấu ngã",
              "DAU_NANG": "dấu nặng", "DAU_MU": "dấu mũ", "DAU_MOC": "dấu móc", "DAU_TRANG": "dấu trăng",
              "DAU_GACH": "gạch ngang (đ)"}
VOWELS = "AEIOUY"


def is_mark(token):
    return token in TONES or token in QUALITY


class Spelling:
    """Một từ đang được đánh vần từng chữ cái."""

    def __init__(self):
        self.letters, self.quality, self.tone = [], {}, None

    def copy(self):
        s = Spelling()
        s.letters, s.quality, s.tone = list(self.letters), dict(self.quality), self.tone
        return s

    def __bool__(self):
        return bool(self.letters)

    def add_letter(self, ch):
        ch = (ch or "").strip().upper()
        if not ch:
            return False
        q = None
        if ch == "Đ":
            base, q = "D", "DAU_GACH"
        else:
            nfd = unicodedata.normalize("NFD", ch)
            base = nfd[0]
            for c in nfd[1:]:
                q = {"\u0302": "DAU_MU", "\u031B": "DAU_MOC", "\u0306": "DAU_TRANG"}.get(c, q)
        if len(base) != 1 or not base.isalpha():
            return False
        self.letters.append(base)
        if q:
            self.quality[len(self.letters) - 1] = q
        return True

    def add_mark(self, token):
        if token in TONES:
            if self._tone_index(force=True) is None:      # chưa có nguyên âm thì chưa gắn được dấu thanh
                return False
            self.tone = token
            return True
        if token in QUALITY:
            allowed = QUALITY[token][0]
            for only_unmarked in (True, False):           # ưu tiên chữ chưa có dấu, rồi mới ghi đè
                for i in range(len(self.letters) - 1, -1, -1):
                    if self.letters[i] in allowed and (not only_unmarked or i not in self.quality):
                        self.quality[i] = token
                        return True
            return False
        return False

    def _tone_index(self, force=False):
        """Chỉ số chữ cái mang dấu thanh theo quy tắc chính tả; None nếu không có nguyên âm."""
        if self.tone is None and not force:
            return None
        L = self.letters
        # cụm nguyên âm cuối cùng
        j = len(L) - 1
        while j >= 0 and L[j] not in VOWELS:
            j -= 1
        if j < 0:
            return None
        k = j
        while k - 1 >= 0 and L[k - 1] in VOWELS:
            k -= 1
        idx = list(range(k, j + 1))
        # "qu" và "gi" (khi sau còn nguyên âm khác): u, i thuộc phụ âm đầu
        if k > 0 and idx and L[k] == "U" and L[k - 1] == "Q":
            idx = idx[1:]
        elif k > 0 and len(idx) > 1 and L[k] == "I" and L[k - 1] == "G":
            idx = idx[1:]
        if not idx:
            return None
        marked = [i for i in idx if self.quality.get(i) in ("DAU_MU", "DAU_MOC", "DAU_TRANG")]
        if marked:
            return marked[-1]                      # ê, ô, ơ, â, ă, ư mang dấu thanh (ươ -> ơ)
        n = len(idx)
        if n == 1:
            return idx[0]
        if n == 2:
            has_final = idx[-1] < len(L) - 1       # còn phụ âm cuối: toán, hoàn -> nguyên âm thứ hai
            return idx[1] if has_final else idx[0]  # âm tiết mở: hòa, thủy, mái, múa -> nguyên âm đầu
        return idx[1]                              # 3 nguyên âm: oái, ngoài -> nguyên âm giữa

    def render(self):
        ti = self._tone_index()
        out = []
        for i, b in enumerate(self.letters):
            q = self.quality.get(i)
            ch = b
            if q == "DAU_GACH" and b == "D":
                ch = "Đ"
            elif q in QUALITY and QUALITY[q][1] and b in QUALITY[q][0]:
                ch = b + QUALITY[q][1]
            if self.tone and i == ti:
                ch += TONES[self.tone]
            out.append(ch)
        return unicodedata.normalize("NFC", "".join(out))


class SentenceBuilder:
    """Câu = các từ (cử chỉ) + các từ đánh vần (chữ cái + dấu), có hoàn tác."""

    def __init__(self):
        self.words, self.spelling, self._history = [], Spelling(), []

    def _push(self):
        self._history.append((list(self.words), self.spelling.copy()))

    def add_letter(self, ch):
        self._push()
        if not self.spelling.add_letter(ch):
            self._history.pop()
            return False
        return True

    def add_mark(self, token):
        self._push()
        if not self.spelling.add_mark(token):
            self._history.pop()
            return False
        return True

    def add_word(self, word):
        self._push()
        if self.spelling:
            self.words.append(self.spelling.render())
            self.spelling = Spelling()
        self.words.append(word)

    def flush(self):
        """Kết thúc từ đang đánh vần (tự gọi khi ngừng ký hiệu một lúc)."""
        if not self.spelling:
            return False
        self._push()
        self.words.append(self.spelling.render())
        self.spelling = Spelling()
        return True

    def undo(self):
        if not self._history:
            return False
        self.words, self.spelling = self._history.pop()
        return True

    def clear(self):
        self.words, self.spelling, self._history = [], Spelling(), []

    def spelling_text(self):
        return self.spelling.render() if self.spelling else ""

    def text(self):
        parts = list(self.words)
        if self.spelling:
            parts.append(self.spelling.render())
        return " ".join(parts)


def _selftest():
    def spell(tokens):
        s = SentenceBuilder()
        for t in tokens:
            (s.add_mark if is_mark(t) else s.add_letter)(t)
        return s.spelling_text()

    cases = [
        (["H", "O", "DAU_MU", "DAU_HUYEN", "N", "G"], "HỒNG"),     # dấu ký ngay sau từng chữ
        (["H", "O", "N", "G", "DAU_MU", "DAU_HUYEN"], "HỒNG"),     # dấu ký sau cả từ -> cùng kết quả
        (["T", "O", "A", "N", "DAU_SAC"], "TOÁN"),
        (["H", "O", "A", "DAU_HUYEN"], "HÒA"),
        (["T", "H", "U", "Y", "DAU_HOI"], "THỦY"),
        (["Q", "U", "Y", "DAU_SAC"], "QUÝ"),
        (["G", "I", "A", "DAU_SAC"], "GIÁ"),
        (["G", "I", "DAU_HUYEN"], "GÌ"),
        (["N", "G", "U", "O", "I", "DAU_MOC", "DAU_MOC", "DAU_HUYEN"], "NGƯỜI"),
        (["D", "U", "O", "N", "G", "DAU_GACH", "DAU_NANG"], "ĐUỌNG"),
        (["Đ", "A", "DAU_TRANG", "DAU_NANG"], "ĐẶ"),
        (["A", "DAU_MU", "DAU_SAC"], "Ấ"),
        (["Â", "DAU_NANG"], "Ậ"),
        (["T", "I", "E", "DAU_MU", "N", "DAU_SAC", "G"], "TIẾNG"),
        (["DAU_SAC"], ""),                                          # chưa có chữ: bỏ qua
    ]
    for toks, expect in cases:
        got = spell(toks)
        assert got == expect, f"{toks} -> {got!r}, mong đợi {expect!r}"
    b = SentenceBuilder()
    for w in ["tôi", "tên", "là"]:
        b.add_word(w)
    for t in ["H", "O", "DAU_MU", "DAU_HUYEN", "N", "G"]:
        (b.add_mark if is_mark(t) else b.add_letter)(t)
    assert b.text() == "tôi tên là HỒNG", b.text()
    b.undo(); b.undo()
    assert b.spelling_text() == "HỒ", b.spelling_text()       # hoàn tác 2 bước: bỏ G, bỏ N
    b.flush(); b.add_word("nhé")
    assert b.text() == "tôi tên là HỒ nhé", b.text()
    print("vietnamese.py: tất cả kiểm tra ĐẠT")


if __name__ == "__main__":
    _selftest()
