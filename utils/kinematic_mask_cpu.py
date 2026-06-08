import cv2
import numpy as np

# Cascade phát hiện khuôn mặt (có sẵn trong OpenCV)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def get_mouth_mask(frame, prev_frame, scale=0.5):
    """Tạo mask triệt tiêu vùng miệng dựa trên Optical Flow và Haar cascade."""
    h, w = frame.shape[:2]
    small_h, small_w = int(h*scale), int(w*scale)
    prev_small = cv2.resize(prev_frame, (small_w, small_h))
    curr_small = cv2.resize(frame, (small_w, small_h))

    # Optical Flow
    flow = cv2.calcOpticalFlowFarneback(
        cv2.cvtColor(prev_small, cv2.COLOR_BGR2GRAY),
        cv2.cvtColor(curr_small, cv2.COLOR_BGR2GRAY),
        None, 0.5, 3, 15, 3, 5, 1.2, 0)
    magnitude, _ = cv2.cartToPolar(flow[...,0], flow[...,1])

    # Phát hiện khuôn mặt trên ảnh xám
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    mask = np.ones((small_h, small_w), dtype=np.float32)  # mặc định giữ nguyên

    if len(faces) > 0:
        x, y, fw, fh = faces[0]  # lấy khuôn mặt đầu tiên
        # Vùng miệng ước lượng: nửa dưới của khuôn mặt, hẹp hơn một chút
        mouth_top = y + int(fh * 0.55)
        mouth_bottom = y + int(fh * 0.9)
        mouth_left = x + int(fw * 0.2)
        mouth_right = x + int(fw * 0.8)
        # Scale về kích thước ảnh nhỏ
        mouth_top_s = int(mouth_top * scale)
        mouth_bottom_s = int(mouth_bottom * scale)
        mouth_left_s = int(mouth_left * scale)
        mouth_right_s = int(mouth_right * scale)

        # Tạo mask miệng
        mouth_area = np.zeros((small_h, small_w), dtype=np.uint8)
        mouth_area[mouth_top_s:mouth_bottom_s, mouth_left_s:mouth_right_s] = 255
        # Dilation để mở rộng vùng miệng
        kernel = np.ones((7,7), np.uint8)
        mouth_area = cv2.dilate(mouth_area, kernel, iterations=2)

        # Gán trọng số thấp cho vùng có chuyển động mạnh
        norm_mag = magnitude / (magnitude.max() + 1e-6)
        weight = 1.0 - norm_mag * 0.9
        weight[mouth_area == 0] = 1.0  # ngoài vùng miệng giữ nguyên
        mask = weight.astype(np.float32)

    # Resize về kích thước gốc
    mask = cv2.resize(mask, (w, h))
    return mask
