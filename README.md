# PRODIGY_ML_05
Develop a model that can accurately recognize food items from images and estimate their calorie content, enabling user to track their dietary intake and make informed food choices.

# Food Recognition & Calorie Estimation Model

A comprehensive deep learning solution for food image recognition and calorie estimation using the Food-101 dataset. This project enables users to track their dietary intake and make informed food choices through accurate food classification and nutritional information.

## Overview

This model uses a **ResNet50-based architecture** with transfer learning to classify food images into 101 different categories and provides calorie estimates for each recognized food item. The system is designed for practical dietary tracking applications with GPU acceleration and comprehensive evaluation metrics.

## Key Features

###  **Advanced Food Recognition**
- **101 Food Categories**: Trained on the comprehensive Food-101 dataset
- **High Accuracy**: Achieves 75-85% top-1 accuracy and 90-95% top-5 accuracy
- **Transfer Learning**: Uses pre-trained ResNet50 for optimal performance
- **Robust Preprocessing**: Includes data augmentation and normalization

### **Calorie Estimation**
- **Integrated Database**: Built-in calorie information for all 101 food classes
- **Nutritional Insights**: Approximate calories per 100g for each food item
- **Top-5 Predictions**: Multiple predictions with confidence scores and calorie data

### **Performance Optimization**
- **GPU Acceleration**: Automatic CUDA detection and utilization
- **Memory Efficient**: Optimized data loading with multiple workers
- **Scalable Training**: Configurable batch sizes and class limits
- **Progress Tracking**: Real-time training metrics and visualizations

###  **Comprehensive Analysis**
- **Detailed Visualizations**: Training history, confusion matrices, and sample predictions
- **Performance Metrics**: Accuracy, precision, recall, F1-score analysis
- **Class Distribution**: Visual analysis of dataset balance
- **Model Reporting**: Automated comprehensive performance reports

##  Model Architecture

```
Input Image (224x224x3)
    ↓
ResNet50 Backbone (Pre-trained)
    ↓
Global Average Pooling
    ↓
Dropout (0.5)
    ↓
Linear Layer (2048 → 512)
    ↓
ReLU Activation
    ↓
Dropout (0.3)
    ↓
Output Layer (512 → 101 classes)
```

##  Dataset Structure

The Food-101 dataset should be organized as follows:

```
food-101/
├── images/
│   ├── apple_pie/
│   │   ├── 1011328.jpg
│   │   └── ...
│   ├── baby_back_ribs/
│   └── ...
├── meta/
│   ├── classes.txt
│   ├── labels.txt
│   ├── test.json
│   ├── test.txt
│   ├── train.json
│   └── train.txt
└── README.txt
```

##  Installation

### Prerequisites
- Python 3.7+
- CUDA-compatible GPU (recommended)
- 8GB+ RAM
- 20GB+ storage space

### Dependencies

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install matplotlib seaborn pillow opencv-python
pip install scikit-learn pandas numpy tqdm
```

Or install all at once:

```bash
pip install -r requirements.txt
```

### Download Dataset

1. Download the Food-101 dataset from [ETH Vision Lab](https://www.vision.ee.ethz.ch/datasets_extra/food-101/)
2. Extract the dataset to your project directory
3. Update the `DATA_DIR` variable in the main script

## Quick Start

### Basic Training

```python
# Run the complete training pipeline
python food_recognition_model.py
```

### Custom Configuration

```python
# Modify these parameters in the main() function
DATA_DIR = 'path/to/food-101'      # Dataset path
BATCH_SIZE = 32                     # Batch size for training
NUM_EPOCHS = 15                     # Number of training epochs
LEARNING_RATE = 0.001              # Learning rate
LIMIT_CLASSES = 20                  # Limit classes (None for full dataset)
```

### Single Image Prediction

```python
# Load trained model and predict a single image
predictions = predict_food_image(
    model_path='best_food_model.pth',
    image_path='your_food_image.jpg',
    classes=classes,
    calorie_db=calorie_db
)
```

## Training Process

### 1. Data Loading & Preprocessing
- Automatic train/validation/test splits
- Data augmentation (rotation, flips, color jittering)
- Normalization using ImageNet statistics

### 2. Model Training
- Transfer learning with pre-trained ResNet50
- Adam optimizer with weight decay
- Learning rate scheduling
- Best model checkpointing

### 3. Evaluation & Visualization
- Comprehensive metric calculation
- Training history visualization
- Confusion matrix analysis
- Sample prediction showcase

## Performance Metrics

### Expected Results (Full Dataset)
- **Top-1 Accuracy**: 75-85%
- **Top-5 Accuracy**: 90-95%
- **Training Time**: 2-4 hours (modern GPU)
- **GPU Memory**: 4-8GB (depends on batch size)

### Model Outputs
- Classification confidence scores
- Top-5 food predictions
- Calorie estimates per 100g
- Visual prediction summaries

##  Configuration Options

### Training Parameters
```python
# Core training settings
BATCH_SIZE = 32          # Adjust based on GPU memory
NUM_EPOCHS = 15          # More epochs = better accuracy
LEARNING_RATE = 0.001    # Learning rate for optimizer
LIMIT_CLASSES = None     # None for full dataset, int for subset
```

### Model Architecture
```python
# Easy architecture swapping
backbone = models.resnet50(pretrained=True)    # ResNet50
# backbone = models.efficientnet_b0(pretrained=True)  # EfficientNet
# backbone = models.densenet121(pretrained=True)      # DenseNet
```

## Output Files

After training, the following files are generated:

- `best_food_model.pth` - Best performing model weights
- `food_classes.pkl` - List of food class names
- `calorie_database.pkl` - Calorie information database
- Various visualization plots and performance reports

## Visualization Features

### Training Visualizations
- **Data Distribution**: Class frequency analysis
- **Sample Images**: Random dataset samples
- **Training History**: Loss and accuracy curves
- **Confusion Matrix**: Error analysis for top classes

### Prediction Visualizations
- **Input Image**: Original food image
- **Top-5 Predictions**: Confidence scores and calorie estimates
- **Bar Charts**: Visual prediction confidence

##  Usage Examples

### Complete Training Pipeline
```python
# Run full training with default settings
model, history, classes, calorie_db = main()
```

### Quick Prediction
```python
# Load saved model and predict
model = FoodRecognitionModel(num_classes=101)
model.load_state_dict(torch.load('best_food_model.pth'))
predictions = predict_with_calories(model, 'test_image.jpg', transform, classes, calorie_db)
```

### Custom Evaluation
```python
# Evaluate on custom dataset
accuracy, report, predictions, targets = evaluate_model(model, test_loader, class_names)
```

##  Supported Food Categories

The model recognizes 101 food categories including:

**Appetizers & Snacks**: bruschetta, deviled_eggs, edamame, guacamole, hummus, nachos, spring_rolls

**Main Dishes**: beef_carpaccio, chicken_curry, filet_mignon, fish_and_chips, hamburger, lasagna, pizza, steak

**Soups**: clam_chowder, french_onion_soup, hot_and_sour_soup, miso_soup, pho

**Desserts**: apple_pie, baklava, cheesecake, chocolate_cake, ice_cream, tiramisu

**And many more!** See `classes.txt` for the complete list.

## Technical Details

### Data Augmentation
- Random horizontal flips
- Random rotations (±15 degrees)
- Random crops and resizing
- Color jittering (brightness, contrast, saturation)

### Model Optimization
- Transfer learning from ImageNet
- Dropout layers for regularization
- Batch normalization
- Weight decay for better generalization

### Hardware Requirements
- **Minimum**: 8GB RAM, 4GB GPU memory
- **Recommended**: 16GB RAM, 8GB GPU memory
- **Storage**: 20GB+ for dataset and model files

## Troubleshooting

### Common Issues

**CUDA Out of Memory**
```python
# Reduce batch size
BATCH_SIZE = 16  # or 8
```

**Dataset Not Found**
```python
# Check dataset path
DATA_DIR = 'correct/path/to/food-101'
```

**Low Accuracy**
```python
# Increase training epochs
NUM_EPOCHS = 25
# Or train on more classes
LIMIT_CLASSES = None
```

### Memory Optimization
- Use smaller batch sizes on limited GPU memory
- Enable mixed precision training for larger models
- Use gradient accumulation for effective larger batch sizes

## 📚 Additional Resources

### Dataset Information
- [Food-101 Paper](https://www.vision.ee.ethz.ch/datasets_extra/food-101/) - Original research paper
- [Dataset Download](https://www.vision.ee.ethz.ch/datasets_extra/food-101/) - Official dataset source


## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

*Star ⭐ this repository if you found it helpful!*
