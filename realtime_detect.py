"""
realtime_detect.py
Loads the trained model and runs real-time sign recognition from webcam.
Keeps a rolling buffer of the last 30 frames and predicts when full.
Press 'q' to quit.
"""
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

from utils import mediapipe_detection, draw_landmarks, extract_keypoints
from collect_data import SIGNS, SEQUENCE_LENGTH

THRESHOLD = 0.7  # Only show predictions above 70% confidence


def visualize_probabilities(probs, signs, frame):
    """Draw colored bars showing probability of each sign."""
    colors = [(245, 117, 16), (117, 245, 16), (16, 117, 245)]
    output = frame.copy()
    for i, prob in enumerate(probs):
        cv2.rectangle(output, (0, 60 + i * 40),
                      (int(prob * 200), 90 + i * 40),
                      colors[i % len(colors)], -1)
        cv2.putText(output, signs[i], (10, 85 + i * 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2, cv2.LINE_AA)
    return output


def main():
    sequence = []
    sentence = []
    predictions = []

    model = load_model("sign_model.h5")
    cap = cv2.VideoCapture(0)
    mp_holistic = mp.solutions.holistic

    print("Starting real-time detection. Press 'q' to quit.")

    with mp_holistic.Holistic(min_detection_confidence=0.5,
                              min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image, results = mediapipe_detection(frame, holistic)
            draw_landmarks(image, results)

            # Rolling buffer of last 30 frames
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-SEQUENCE_LENGTH:]

            if len(sequence) == SEQUENCE_LENGTH:
                res = model.predict(np.expand_dims(sequence, axis=0),
                                    verbose=0)[0]
                predicted_idx = int(np.argmax(res))
                predictions.append(predicted_idx)

                # Stabilize: require last 10 predictions to agree
                if (len(predictions) >= 10 and
                        np.unique(predictions[-10:])[0] == predicted_idx and
                        res[predicted_idx] > THRESHOLD):

                    predicted_sign = SIGNS[predicted_idx]
                    if len(sentence) == 0 or predicted_sign != sentence[-1]:
                        sentence.append(predicted_sign)

                if len(sentence) > 5:
                    sentence = sentence[-5:]

                image = visualize_probabilities(res, SIGNS, image)

            # Top sentence bar
            cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
            cv2.putText(image, " ".join(sentence), (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow("Sign Language Recognition", image)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
