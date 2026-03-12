# Chicken Disease Image Classification

A deep learning project for classifying chicken diseases from images using Convolutional Neural Networks (CNNs). This project builds and compares multiple CNN architectures to identify poultry diseases, helping farmers and veterinarians detect health issues early.

## Dataset

The dataset contains **8,067** chicken images across **4 classes**:

| Class | Description |
|-------|-------------|
| **Coccidiosis** | A parasitic disease affecting the intestinal tract |
| **Salmonella** | Bacterial infection causing digestive issues |
| **New Castle Disease** | Viral disease affecting respiratory and nervous systems |
| **Healthy** | Normal, healthy chickens |

### Data Structure

```
data /
├── Train/           # Training images (8067 images)
│   ├── cocci.*.jpg
│   ├── salmo.*.jpg
│   ├── ncd.*.jpg
│   └── healthy.*.jpg
└── train_data.csv   # Image filenames and labels
```

## Project Structure

```
Chicken_Disease_Image_Classification/
├── chicken disease.ipynb   # Main Jupyter notebook with all experiments
├── data /                  # Dataset directory
│   ├── Train/
│   └── train_data.csv
├── README.md
├── LICENSE
└── .gitignore
```

## Installation

### Prerequisites

- Python 3.8+
- TensorFlow 2.x
- Jupyter Notebook or Jupyter Lab

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd Chicken_Disease_Image_Classification
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Linux/Mac
# or
venv\Scripts\activate     # On Windows
```

3. Install dependencies:
```bash
pip install tensorflow numpy pandas matplotlib seaborn scikit-learn jupyter
pip install lime shap opencv-python pillow
```

4. Launch Jupyter:
```bash
jupyter notebook "chicken disease.ipynb"
```

## Project Overview

The notebook covers the complete machine learning pipeline:

### 1. Data Exploration & Preprocessing
- Loading and exploring the dataset
- Image visualization and analysis
- Class distribution analysis
- Data augmentation using `ImageDataGenerator`
- Train/validation split (80/20)

### 2. Model Building

Four CNN architectures are built and compared:

| Model | Description |
|-------|-------------|
| **Baseline CNN** | Simple CNN with 3 convolutional blocks |
| **BatchNorm CNN** | Baseline + Batch Normalization |
| **Regularized CNN** | BatchNorm + Dropout for regularization |
| **Optimized CNN** | Regularized + AdamW optimizer + Learning rate scheduling |

### 3. Training & Evaluation
- Early stopping and model checkpointing
- Training history visualization
- Confusion matrix analysis
- Classification reports (precision, recall, F1-score)

### 4. Model Interpretability
- **LIME (Local Interpretable Model-agnostic Explanations)**: Explains individual predictions
- **SHAP (SHapley Additive exPlanations)**: Deep learning explanations for model decisions

## Results

The notebook includes comprehensive comparison of all models with:
- Training/validation accuracy and loss curves
- Performance metrics comparison table
- Visual explanations of model predictions using LIME and SHAP

## Key Features

- **Multiple CNN architectures** for comparison
- **Data augmentation** to improve generalization
- **Callbacks** (EarlyStopping, ReduceLROnPlateau, ModelCheckpoint)
- **Model interpretability** with LIME and SHAP visualizations
- **Comprehensive evaluation** with confusion matrices and classification reports

## Hardware Notes

- Training is done on **CPU** (CUDA GPU support available if detected)
- Training time on CPU: ~1-2 hours for full notebook execution
- GPU training significantly reduces training time

## Dependencies

```
tensorflow>=2.10
numpy
pandas
matplotlib
seaborn
scikit-learn
jupyter
lime
shap
opencv-python
pillow
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Dataset source: Chicken disease image dataset
- Built with TensorFlow/Keras
- Interpretability powered by LIME and SHAP
