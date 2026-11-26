MOVE_HI = 0.050        # trên mức này = đang di chuyển (đang ký cử chỉ/dấu)
MOVE_LO = 0.025        # dưới mức này = đứng yên (có thể đang giữ 1 chữ cái)
STILL_FRAMES = 5       # cần đứng yên liên tục ngần này frame mới tính là "giữ yên"
LETTER_BLOCK_AFTER_GESTURE = 0.7   # giây: sau khi nhận 1 cử chỉ/dấu, tạm không đọc chữ cái

WORD_GAP_SECONDS = 1.5  # ngừng ký (không thấy tay) quá ngần này giây thì kết thúc từ đang đánh vần
