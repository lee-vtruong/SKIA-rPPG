import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import cv2
import numpy as np
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=False,
    min_detection_confidence=0.5
)

MOUTH_INDICES = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308]

def get_mouth_mask(frame, prev_frame, scale=0.5):
    h, w = frame.shape[:2]
    small_h, small_w = int(h*scale), int(w*scale)
    prev_small = cv2.resize(prev_frame, (small_w, small_h))
    curr_small = cv2.resize(frame, (small_w, small_h))
    
    flow = cv2.calcOpticalFlowFarneback(
        cv2.cvtColor(prev_small, cv2.COLOR_BGR2GRAY),
        cv2.cvtColor(curr_small, cv2.COLOR_BGR2GRAY),
        None, 0.5, 3, 15, 3, 5, 1.2, 0)
    magnitude, _ = cv2.cartToPolar(flow[...,0], flow[...,1])
    
    results = face_mesh.process(frame)
    mask = np.ones((small_h, small_w), dtype=np.float32)
    
    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0]
        pts = []
        for idx in MOUTH_INDICES:
            lm = landmarks.landmark[idx]
            pts.append([int(lm.x * small_w), int(lm.y * small_h)])
        mouth_mask = np.zeros((small_h, small_w), dtype=np.uint8)
        cv2.fillPoly(mouth_mask, [np.array(pts)], 255)
        mouth_mask = cv2.dilate(mouth_mask, np.ones((7,7), np.uint8), iterations=3)
        
        norm_mag = magnitude / (magnitude.max() + 1e-6)
        weight = 1.0 - norm_mag * 0.9
        weight[mouth_mask == 0] = 1.0
        mask = weight.astype(np.float32)
    
    mask = cv2.resize(mask, (w, h))
    return mask
