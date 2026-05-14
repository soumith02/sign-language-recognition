"""
collect_data.py
Captures webcam video sequences for each sign and saves keypoints as .npy arrays.

Folder structure created:
    data/
        hello/
            0/  0.npy ... 29.npy   (one video = 30 frames)
            1/  ...
            ...
        thanks/  ...
        iloveyou/  ...
"""
import os
import cv2
import numpy as np
import mediapipe as mp

from utils import mediapipe_detection, draw_landmarks, extract_keypoints

# ----- CONFIG -----
DATA_PATH = "data"
SIGNS = np.array(["hello", "thanks", "iloveyou"])
NUM_SEQUENCES = 10      # number of videos per sign
SEQUENCE_LENGTH = 30    # frames per video
# ------------------


def setup_folders():
    for sign in SIGNS:
        for seq in range(NUM_SEQUENCES):
            os.makedirs(os.path.join(DATA_PATH, sign, str(seq)), exist_ok=True)


def collect():
    setup_folders()
    cap = cv2.VideoCapture(0)
    mp_holistic = mp.solutions.holistic

    with mp_holistic.Holistic(min_detection_confidence=0.5,
                              min_tracking_confidence=0.5) as holistic:
        for sign in SIGNS:
            for seq in range(NUM_SEQUENCES):
                for frame_num in range(SEQUENCE_LENGTH):
                    ret, frame = cap.read()
                    if not ret:
                        continue

                    image, results = mediapipe_detection(frame, holistic)
                    draw_landmarks(image, results)

                    # Pause briefly at start of each new video sequence
                    if frame_num == 0:
                        cv2.putText(image, "GET READY",
                                    (120, 200), cv2.FONT_HERSHEY_SIMPLEX,
                                    1.5, (0, 255, 0), 4, cv2.LINE_AA)
                        cv2.putText(image,
                                    f"Sign: {sign}  |  Video: {seq+1}/{NUM_SEQUENCES}",
                                    (15, 25), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7, (0, 0, 255), 2, cv2.LINE_AA)
                        cv2.imshow("Data Collection", image)
                        cv2.waitKey(2000)  # 2-second pause
                    else:
                        cv2.putText(image,
                                    f"Sign: {sign}  |  Video: {seq+1}/{NUM_SEQUENCES}  Frame: {frame_num}",
                                    (15, 25), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.6, (0, 0, 255), 2, cv2.LINE_AA)
                        cv2.imshow("Data Collection", image)

                    # Save keypoints for this frame
                    keypoints = extract_keypoints(results)
                    npy_path = os.path.join(DATA_PATH, sign,
                                            str(seq), f"{frame_num}.npy")
                    np.save(npy_path, keypoints)

                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        cap.release()
                        cv2.destroyAllWindows()
                        return

    cap.release()
    cv2.destroyAllWindows()
    print("Data collection complete!")


if __name__ == "__main__":
    collect()
