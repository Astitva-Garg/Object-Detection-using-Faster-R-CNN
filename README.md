# Pedestrian Detection with Faster R-CNN (Penn-Fudan Dataset)

This repository implements a pedestrian detection pipeline using **Faster R-CNN** with a **ResNet-50 FPN backbone** on the [Penn-Fudan Dataset](https://www.cis.upenn.edu/~jshi/ped_html/).

The dataset contains pedestrian images with corresponding masks and bounding box annotations. This project extends the dataset with additional objects listed in `added-object-list.txt`, parses annotations, trains Faster R-CNN, and evaluates the model using precision-recall analysis.

---

## 📂 Project Structure

```
.
├── Data/
│   └── dataset/
│       ├── PNGImages/              # Images
│       ├── PedMasks/               # Segmentation masks
│       ├── Annotation/             # Text-based annotations
│       └── added-object-list.txt   # Custom object list
├── main.py                         # Training & evaluation script
└── README.md
```

---

## 🚀 Features

- Custom `PennFudanDataset` class for handling Penn-Fudan pedestrian dataset
- Integration of additional objects via **mask parsing**
- Model: **Faster R-CNN ResNet-50 FPN**
- Training with **SGD optimizer** & **StepLR scheduler**
- Evaluation using **Precision-Recall curve** and **Average Precision (AP)**
- Visualization of model predictions with confidence scores

---

## ⚙️ Installation

```bash
git clone https://github.com/your-username/pennfudan-fasterrcnn.git
cd pennfudan-fasterrcnn

# Install dependencies
pip install torch torchvision scikit-learn matplotlib pillow numpy
```

---

## 📊 Usage

### Training & Evaluation
Run the main script:

```bash
python main.py
```

This will:
1. Load and split dataset (80% train, 20% test)
2. Train Faster R-CNN for **5 epochs**
3. Evaluate on test split and display **Precision-Recall curve**
4. Visualize predictions on sample test images

---

## 📈 Results

- Model outputs **Average Precision (AP)** after evaluation
- Precision–Recall curve is plotted
- Sample predictions with bounding boxes and confidence scores are visualized

---

## 🔍 Example Output

### Precision-Recall Curve
The script automatically displays a PR curve after evaluation.

### Predictions
Bounding boxes with confidence scores are drawn on test images:

```
[ RED BOX ] = Detected pedestrian (with score label)
```

---

## 📌 Notes

- Supports GPU acceleration (`torch.device("cuda")`) if available.
- Dataset must be placed under `Data/dataset/` with required subdirectories.

---

## 📜 License

MIT License. Free to use and modify.

---
