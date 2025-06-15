import cv2
import numpy as np

def extract_frames(video_path, max_frames=30, size=(64, 64)):
    cap = cv2.VideoCapture(video_path)
    frames = []
    count = 0

    while cap.isOpened() and count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, size)
        frame = frame / 255.0
        frames.append(frame)
        count += 1

    cap.release()

    # Pad if fewer than max_frames
    while len(frames) < max_frames:
        frames.append(np.zeros_like(frames[0]))

    return np.array(frames)
