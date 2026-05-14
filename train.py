"""
train.py
Loads keypoint sequences, builds an LSTM model, trains it, and saves the model.
"""
import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input

from collect_data import DATA_PATH, SIGNS, NUM_SEQUENCES, SEQUENCE_LENGTH


def load_dataset():
    """Load all .npy keypoint files into X (sequences) and y (labels)."""
    label_map = {label: idx for idx, label in enumerate(SIGNS)}
    sequences, labels = [], []

    for sign in SIGNS:
        sign_path = os.path.join(DATA_PATH, sign)
        if not os.path.exists(sign_path):
            continue
        for seq in range(NUM_SEQUENCES):
            window = []
            seq_path = os.path.join(sign_path, str(seq))
            if not os.path.exists(seq_path):
                continue
            for frame_num in range(SEQUENCE_LENGTH):
                npy_path = os.path.join(seq_path, f"{frame_num}.npy")
                if os.path.exists(npy_path):
                    window.append(np.load(npy_path))
            if len(window) == SEQUENCE_LENGTH:
                sequences.append(window)
                labels.append(label_map[sign])

    X = np.array(sequences)
    y = to_categorical(labels).astype(int)
    return X, y


def build_model(num_signs):
    """LSTM architecture (matches the paper's structure)."""
    model = Sequential([
        Input(shape=(SEQUENCE_LENGTH, 1662)),
        LSTM(64, return_sequences=True, activation='relu'),
        LSTM(128, return_sequences=True, activation='relu'),
        LSTM(64, return_sequences=False, activation='relu'),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(num_signs, activation='softmax')
    ])
    model.compile(optimizer='Adam',
                  loss='categorical_crossentropy',
                  metrics=['categorical_accuracy'])
    return model


def main():
    print("Loading dataset...")
    X, y = load_dataset()
    print(f"Loaded {len(X)} sequences across {len(SIGNS)} signs.")
    print(f"X shape: {X.shape}  |  y shape: {y.shape}")

    if len(X) == 0:
        print("No data found. Run collect_data.py first.")
        return

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42)

    print("\nBuilding model...")
    model = build_model(num_signs=len(SIGNS))
    model.summary()

    print("\nTraining (this may take 1-2 minutes)...")
    model.fit(X_train, y_train, epochs=100, verbose=1)

    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest Accuracy: {acc:.2%}")

    model.save("sign_model.h5")
    print("Model saved to sign_model.h5")


if __name__ == "__main__":
    main()
