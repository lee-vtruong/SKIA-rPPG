import cv2
import numpy as np


class SmoothKinematicMask:
    """
    Soft kinematic mask dựa trên Optical Flow.

    Ý tưởng:
    - Không hard threshold.
    - Không xoá pixel hoàn toàn.
    - Motion càng mạnh thì weight càng thấp.
    """

    def __init__(
        self,
        alpha=0.8,
        strength=2.0,
        mouth_top_ratio=0.70,
        mouth_bottom_ratio=1.00,
        mouth_left_ratio=0.20,
        mouth_right_ratio=0.80,
    ):
        self.alpha = alpha
        self.strength = strength

        self.mouth_top_ratio = mouth_top_ratio
        self.mouth_bottom_ratio = mouth_bottom_ratio
        self.mouth_left_ratio = mouth_left_ratio
        self.mouth_right_ratio = mouth_right_ratio

        self.prev_smoothed_magnitude = None

    def reset(self):
        self.prev_smoothed_magnitude = None

    def get_mouth_mask(self, frame, prev_frame, scale=0.5):
        h, w = frame.shape[:2]

        small_h = int(h * scale)
        small_w = int(w * scale)

        prev_small = cv2.resize(prev_frame, (small_w, small_h))
        curr_small = cv2.resize(frame, (small_w, small_h))

        prev_gray = cv2.cvtColor(prev_small, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_small, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(
            prev_gray,
            curr_gray,
            None,
            0.5,
            3,
            15,
            3,
            5,
            1.2,
            0,
        )

        magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        if self.prev_smoothed_magnitude is None:
            smoothed = magnitude
        else:
            smoothed = (
                self.alpha * self.prev_smoothed_magnitude
                + (1.0 - self.alpha) * magnitude
            )

        self.prev_smoothed_magnitude = smoothed

        mouth_top = int(small_h * self.mouth_top_ratio)
        mouth_bottom = int(small_h * self.mouth_bottom_ratio)
        mouth_left = int(small_w * self.mouth_left_ratio)
        mouth_right = int(small_w * self.mouth_right_ratio)

        mask = np.ones((small_h, small_w), dtype=np.float32)

        motion_region = smoothed[
            mouth_top:mouth_bottom,
            mouth_left:mouth_right,
        ]

        motion_region = motion_region / (motion_region.max() + 1e-8)

        weight = 0.85 + 0.15 * np.exp(-self.strength * motion_region)
        weight = weight.astype(np.float32)

        mask[
            mouth_top:mouth_bottom,
            mouth_left:mouth_right,
        ] = weight

        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
        mask = np.clip(mask, 0.0, 1.0).astype(np.float32)

        return mask


_smooth_mask = SmoothKinematicMask()


def reset_mask_state():
    _smooth_mask.reset()


def get_mouth_mask(frame, prev_frame, scale=0.5):
    return _smooth_mask.get_mouth_mask(frame, prev_frame, scale)