# Flag Classification - Computer Vision Project

This project classifies real-world flag images using models trained on synthetic 3D-rendered flag data. The workflow involves feature extraction, classification, and analysis of generalization challenges.

## Files Included
- `report.pdf` – Final project report
- `codebook.ipynb` – Notebook with full classification pipeline
- `crop.py` – Script used to manually crop test flag regions

## Summary
- Extracted features using HOG, HSV, LBP, and ResNet.
- Trained ML models like SVM and Random Forest.
- Achieved 97% accuracy on synthetic data; ~17% on real images.
- Emphasized challenges in domain adaptation from synthetic to real-world flags.

## Tools Used
- Python, OpenCV, scikit-learn, NumPy, ResNet
