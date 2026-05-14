## Setup

```bash
git clone https://github.com/soumith02/signlang.git
cd signlang

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

## Usage

```bash
# 1. Collect training data
python collect_data.py

# 2. Train the model
python train.py

# 3. Run real-time detection
python realtime_detect.py
```

Press `q` to quit any window.

## Keypoint Breakdown (1662 features)

| Source | Landmarks | Values per landmark | Total |
|--------|-----------|---------------------|-------|
| Pose | 33 | 4 (x, y, z, visibility) | 132 |
| Face | 468 | 3 (x, y, z) | 1404 |
| Left hand | 21 | 3 (x, y, z) | 63 |
| Right hand | 21 | 3 (x, y, z) | 63 |
| **Total** | | | **1662** |

## Tech Stack

- Python 3.12
- MediaPipe 0.10.21
- TensorFlow / Keras 2.16
- OpenCV
- NumPy, Scikit-learn

## Citation

Medipally, H. P. R., **Asani, S. R.**, Ayyagari, S. R., & Jagadish, R. M. (2022). *Sign Language Recognition using CNN and Storage Optimization Algorithm.* Vardhaman College of Engineering.
