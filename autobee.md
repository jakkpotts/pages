# AutoBee Vehicle Classification System

A deep-learning based vehicle classification system designed to classify vehicles from traffic camera feeds, utilizing two distinct datasets. First, it learns from high-quality vehicle images from a dataset with a surveillance/traffic camera perspective, then adapts this knowledge to work with real-world traffic camera footage, which is often lower quality and taken from different angles.

# Model Architecture
The system uses a two-stage pipeline:

1. **Detection Stage (Pre-trained, Not Modified)**
   - YOLOv8n from Ultralytics
   - Pre-trained on COCO dataset
   - Used only for vehicle detection/localization
   - Not fine-tuned, used as-is

2. **Classification Stage (Custom Model)**
   - Backbone: EfficientNet-V2-M
     * Pre-trained on ImageNet
     * Currently frozen (not fine-tuned)
     * Used for feature extraction (1280-dim features)
   - Custom Classification Heads
     * Three separate heads for make/type/color
     * Fully trained on VeriWild dataset
     * Only part of model currently trained

## Current Training Status
- Detection model: Using pre-trained YOLOv8n, not modified
- Classification backbone: Using pre-trained EfficientNet-V2-M, currently frozen
- Classification heads: Trained from scratch on VeriWild dataset

## Planned Training Updates
1. **Full Model Fine-tuning**
   - Unfreeze EfficientNet backbone
   - Fine-tune entire classification model
   - Use lower learning rate for backbone

2. **Domain Adaptation**
   - Add traffic camera augmentations
   - Collect small traffic camera dataset
   - Fine-tune on real deployment conditions

## The Learning Process
The system learns in three distinct phases:

### Phase 1 - Feature Extraction Training
- Fine-tune EfficientNet-V2-M backbone on vehicle data
- Focus on learning discriminative vehicle features
- Temperature parameter of 0.07 for contrastive learning

### Phase 2 - Classification Head Training
- Train custom classification heads for make/type/color
- Lower learning rate (0.00005) for precise tuning
- Optimize for multi-task classification performance

### Phase 3 - Semi-supervised Refinement
- Use clustering to improve classification confidence
- Progressive confidence thresholding
- Ensemble methods for robust predictions

## Pipeline Overview
1. **Frame Acquisition**
   - Download frames from traffic cameras
   - Basic preprocessing and quality checks

2. **Vehicle Detection**
   - YOLOv8n processes full frames
   - Detect and localize vehicles
   - Filter by confidence threshold (default: 0.65)

3. **Patch Extraction**
   - Crop detected vehicles from frames
   - Resize to 224x224 for classification
   - Normalize using ImageNet statistics

4. **Fine-grained Classification**
   - Process vehicle patches through YOLO11n
   - Extract features using EfficientNet-V2-M
   - Classify make, type, and color
   - Compute confidence scores for each task

5. **Result Visualization**
   - Draw detection boxes on original frames
   - Save vehicle patches with classifications
   - Log confidence scores and predictions

## 🌟 Project Overview

AutoBee is an advanced vehicle classification system that combines traffic camera data with the CompCars surveillance dataset to create a robust vehicle classification model. It uses a sophisticated semi-supervised learning approach with ensemble clustering to improve classification accuracy with limited labeled data.

## 🎯 Core Purpose & Architecture

### The Core Purpose
This system is designed to classify vehicles from traffic camera feeds by learning from two different datasets. First, it learns from high-quality vehicle images (CompCars dataset), then adapts this knowledge to work with real-world traffic camera footage, which is often lower quality and taken from different angles.

### Model Architecture
The system uses YOLO11-nano (Ultralytics implementation) as its foundation. This architecture was chosen for its optimal balance of performance and efficiency:

- Efficient feature extraction optimized for real-time processing
- Custom classification head for vehicle-specific features
- Balanced architecture that handles both high-quality CompCars data and lower-quality traffic camera footage
- Input size optimized for traffic camera streams while maintaining detail quality

### The Learning Process
The system employs a three-phase learning approach:

1. **Initial Training Phase**
   - Focuses on basic vehicle recognition from traffic camera data
   - Uses contrastive learning to understand vehicle distinctions
   - Establishes foundational feature recognition

2. **Fine-tuning Phase**
   - Refines knowledge using the CompCars surveillance dataset
   - Optimizes for specific vehicle classification tasks
   - Adapts to different viewing angles and conditions

3. **Semi-supervised Learning Phase**
   - Implements ensemble clustering with multiple algorithms (KMeans, DBSCAN, GMM)
   - Uses progressive confidence thresholding (starting at 0.7)
   - Applies stability bonuses for consistent predictions
   - Handles low confidence cases through intelligent filtering

### Special Features
The system includes several sophisticated capabilities:

- Mixed precision training for computational efficiency
- Dynamic cluster count determination using elbow method
- Ensemble voting system for robust predictions
- Advanced monitoring and visualization tools
- Real-time confidence tracking and analysis

### Performance Optimization
The system is optimized for both accuracy and speed:

- Efficient batch processing and memory management
- Real-time processing capabilities for traffic camera feeds
- Flexible CPU/GPU deployment with TensorRT support
- Comprehensive monitoring and analysis tools
- Robust handling of camera outages and stream quality variations

### 3. Semi-Supervised Learning System
- **Clustering Ensemble**:
  - Supported Algorithms:
    * KMeans (default)
    * DBSCAN
    * Gaussian Mixture Models (GMM)
  - Dynamic cluster count using elbow method
  - Ensemble voting for robust predictions

- **Progressive Confidence Thresholding**:
  - Base threshold: 0.7
  - Per-epoch increase: 0.05
  - Stability bonus: 10% for consistent predictions
  - Ensemble agreement weighting
  - Low confidence handling: marked as -1 (ignored)

### 4. Dataset Implementations
We maintain two VeriWild dataset implementations for different use cases:

#### Current Implementation (`src/datasets/veri_wild_dataset.py`)
- **Purpose**: Optimized for high-performance training with modern augmentations
- **Features**:
  * Advanced augmentation pipeline with Albumentations
  * Memory-efficient caching system
  * Support for mixup and mosaic augmentations
  * Optimized for H100/GPU training
  * Integrated with current training pipeline
- **Usage**: Used by `train_yolo11n_classifier.py` for current training

#### Deprecated Implementation (`src/data/veri_wild_dataset.py`)
- **Purpose**: Originally used for semi-supervised learning experiments
- **Features**:
  * Basic transformations
  * Camera-aware training support
  * Integrated with Lightning training
  * Used with original CompCars training strategy
- **Status**: Deprecated, kept for reference and compatibility with old checkpoints
- **Note**: Was part of the initial training strategy that was abandoned in favor of the current two-stage pipeline

The current implementation provides better performance and more sophisticated augmentation options, while the deprecated version remains for historical purposes and to support loading older model checkpoints.

## 🚀 Getting Started

1. **Environment Setup**
   ```bash
   # Install timg for image preview support
   # For Debian/Ubuntu:
   sudo apt install timg
   # For macOS:
   brew install timg

   python -m venv .venv
   source .venv/bin/activate

   # For GPU systems:
   pip install -r requirements.txt

   # For CPU-only systems:
   pip install -r cpu-requirements.txt
   ```

2. **Download Models**
   ```bash
   python -m src.utils.download_models
   ```

3. **Data Collection**
   ```bash
   # Fetch camera streams
   python -m src.utils.fetch_california_streams
   python -m src.utils.fetch_vegas_streams

   # Process camera data
   python -m src.utils.prepare_california_cam_data
   python -m src.utils.prepare_vegas_cam_data
   ```

4. **Dataset Preparation**
   ```bash
   # Download CompCars surveillance dataset
   wget https://mobilewireless.tech/sv_data.zip
   unzip sv_data.zip -d data/compcars

   # Generate dataset splits
   python -m scripts.generate_compcars_splits \
     --data-dir data/compcars \
     --val-size 0.15 \
     --test-size 0.15
   ```

5. **Training**
   ```bash
   python -m src.training.train_classifier
   ```

## 📊 Monitoring & Visualization

### Real-time Training Monitoring
The system includes comprehensive monitoring tools that track:

1. **Per-class Confidence Distribution**
   - Violin plots showing confidence score distributions
   - Updated every clustering epoch
   - Found in `monitoring/confidence_dist_*.png`

2. **Feature Space Visualization**
   - t-SNE plots showing feature space clustering
   - Color-coded by pseudo-labels
   - Found in `monitoring/tsne_*.png`

3. **Pseudo-label Quality**
   - Confusion matrices comparing with ground truth
   - Accuracy metrics and stability tracking
   - Found in `monitoring/confusion_matrix_*.png`

### Weights & Biases Integration
All metrics and visualizations are automatically logged to W&B:

```bash
# View live training progress
wandb login
wandb init
python -m src.training.train_classifier

# Access dashboard
wandb dashboard
```

Key metrics tracked in W&B:
- Per-class confidence distributions
- Feature space embeddings
- Confusion matrices
- Training metrics (loss, accuracy, etc.)
- System metrics (GPU memory, etc.)

### Custom Monitoringo
You can extend monitoring capabilities:

```python
from src.utils.monitoring import TrainingMonitor

# Initialize monitor
monitor = TrainingMonitor(save_dir="custom_monitoring")

# Track custom metrics
monitor.track_training_progress(
    metrics={
        "custom_metric": value,
        "another_metric": another_value
    },
    epoch=current_epoch,
    phase="custom_phase"
)

# Generate custom visualizations
monitor.plot_confidence_distribution(
    confidence_scores=scores,
    labels=predictions,
    epoch=current_epoch,
    phase="custom_phase"
)
```

### Monitoring Directory Structure
```
monitoring/
├── confidence_dist/          # Confidence distribution plots
│   ├── pretrain/            # Pretraining phase plots
│   ├── finetune/            # Fine-tuning phase plots
│   └── semi_supervised/     # Semi-supervised phase plots
├── feature_space/           # t-SNE visualizations
├── confusion_matrices/      # Confusion matrix plots
└── metrics/                 # Raw metric data
```

### Analyzing Confident Predictions

The system provides tools to analyze prediction confidence and stability:

```python
from src.utils.prediction_analysis import PredictionAnalyzer

# Initialize analyzer
analyzer = PredictionAnalyzer(save_dir="analysis")

# 1. Analyze confidence trends over time
metrics = analyzer.analyze_confidence_trends(
    confidence_history=trainer.confidence_history,
    label_history=trainer.label_history,
    epoch=current_epoch
)
print(f"Mean confidence change: {metrics['mean_confidence_change']:.3f}")
print(f"Mean label changes: {metrics['mean_label_changes']:.3f}")

# 2. Analyze highly confident predictions
metrics = analyzer.analyze_confident_predictions(
    predictions=pseudo_labels,
    confidence_scores=confidence_scores,
    true_labels=ground_truth,  # Optional
    confidence_threshold=0.8
)
print(f"Number of confident predictions: {metrics['n_confident']}")
print(f"Confident prediction accuracy: {metrics.get('confident_accuracy', 'N/A')}")

# 3. Find uncertain samples for further analysis
uncertain_indices = analyzer.find_uncertain_samples(
    predictions=pseudo_labels,
    confidence_scores=confidence_scores,
    confidence_threshold=0.5
)
print(f"Found {len(uncertain_indices)} uncertain samples")

# 4. Analyze prediction stability
stable_idx, unstable_idx = analyzer.analyze_prediction_stability(
    label_history=trainer.label_history,
    min_stable_epochs=3
)
print(f"Stable predictions: {len(stable_idx)}")
print(f"Unstable predictions: {len(unstable_idx)}")

# 5. Plot confidence vs accuracy relationship
analyzer.plot_confidence_vs_accuracy(
    confidence_scores=confidence_scores,
    predictions=pseudo_labels,
    true_labels=ground_truth,
    n_bins=10
)
```

The analyzer generates visualizations and metrics including:
- Confidence score evolution over time
- Distribution of confident predictions
- Confidence vs accuracy curves
- Stability analysis
- Per-class confidence statistics

All visualizations are saved to the `analysis/` directory and logged to W&B:
```
analysis/
├── confidence_evolution_epoch{N}.png    # Confidence trends
├── confident_pred_distribution.png      # Class distribution
├── confidence_vs_accuracy.png          # Confidence-accuracy relationship
└── metrics.json                        # Raw analysis metrics
```

## 🔍 Development Status

- ✅ Core classification system
- ✅ Semi-supervised learning
- ✅ Traffic camera integration
- ✅ Advanced monitoring & visualization
  - Real-time confidence tracking
  - Feature space visualization
  - Pseudo-label quality metrics
  - W&B integration
- ❌ Unit tests (planned)
- ✅ Performance optimizations
- ✅ Experiment tracking (WandB)

## 📊 Key Dependencies

- torch==2.5.1
- ultralytics==8.3.67
- albumentations==2.0.1
- opencv-python==4.11.0.86
- scikit-learn==1.6.1
- wandb==0.19.4
- matplotlib==3.8.2
- seaborn==0.13.0

## 🛠 Performance Considerations

1. **Training Optimizations**
   - Mixed precision training
   - Gradient accumulation
   - Memory-efficient batching
   - Data prefetching

2. **Inference Optimizations**
   - TensorRT support (optional)
   - Batch inference
   - CPU/GPU flexibility

## 🔒 Security Notes

- Traffic camera data is publicly accessible
- No authentication required for data sources
- Model weights are stored locally
- WandB integration for secure experiment tracking

## 📝 License

This project has not yet defined a license.

# Current Status & Known Issues

## Model Performance
Current model performance metrics from latest training:
- Make Accuracy: 44.90% (poor)
- Type Accuracy: 86.78% (good)
- Color Accuracy: 88.57% (good)

### Known Issues
1. **Poor Make Classification**
   - Current make accuracy is only 44.90%
   - Model struggles with fine-grained make classification
   - Likely due to only training classification heads without fine-tuning backbone

2. **Domain Gap**
   - Model trained on VeriWild dataset (high quality, controlled conditions)
   - Testing on traffic camera footage (low quality, varying conditions)
   - No domain adaptation implemented yet

3. **Feature Extraction Limitations**
   - EfficientNet-V2-M backbone not fine-tuned
   - Using ImageNet pre-trained weights without vehicle-specific adaptation
   - Feature extraction may not be optimal for traffic camera conditions

## Planned Improvements

### Immediate Actions
1. **Model Retraining**
   - Fine-tune entire model including EfficientNet backbone
   - Update training configuration to train all parameters
   - Implement proper feature extractor optimization

2. **Data Augmentation**
   - Add augmentations to simulate traffic camera conditions:
     * Quality degradation
     * Lighting variations
     * Perspective changes
     * Motion blur
     * Weather effects

3. **Domain Adaptation**
   - Collect small dataset of labeled traffic camera images
   - Implement domain adaptation techniques
   - Fine-tune on traffic camera data after VeriWild training

### Long-term Improvements
1. **Architecture Enhancements**
   - Consider alternative backbones optimized for surveillance
   - Implement attention mechanisms for fine-grained classification
   - Add multi-scale feature fusion

2. **Data Collection**
   - Build traffic camera vehicle dataset
   - Implement semi-supervised learning for unlabeled data
   - Create validation set from actual deployment conditions

3. **Deployment Optimizations**
   - Model quantization for faster inference
   - Batch processing for multiple streams
   - Adaptive confidence thresholds

## Training Configuration
Current training configuration needs updates:
```yaml
training:
  optimizer:
    name: AdamW
    lr: 1e-4
    weight_decay: 0.01
  scheduler:
    name: OneCycleLR
    max_lr: 1e-3
    pct_start: 0.3
  epochs: 100
  batch_size: 32
  mixed_precision: true
```

Planned updates:
```yaml
training:
  optimizer:
    name: AdamW
    lr: 1e-5  # Lower learning rate for full model
    weight_decay: 0.01
  scheduler:
    name: OneCycleLR
    max_lr: 1e-4
    pct_start: 0.3
  epochs: 150  # More epochs for full model training
  batch_size: 32
  mixed_precision: true
  freeze_backbone: false  # Important: Allow backbone training
```

## Next Steps
1. Implement model retraining with full backbone fine-tuning
2. Add comprehensive data augmentation pipeline
3. Collect and label small traffic camera dataset
4. Implement domain adaptation techniques
5. Evaluate and iterate on improvements

# Training Strategy Evolution

## Initial Approach (Deprecated)
Our initial training strategy (`src/training/train_classifier.py`) attempted to:
- Train a vehicle classifier from scratch
- Use CompCars dataset directly
- Implement complex semi-supervised learning
- Use distributed training
This approach was abandoned due to complexity and training instability.

## Current Approach
Our current implementation (`src/training/train_yolo11n_classifier.py`) uses:
1. **Two-Stage Pipeline**:
   - YOLOv8n (pre-trained) for vehicle detection
   - EfficientNet-V2-M + custom heads for classification

2. **Training Strategy**:
   - Start with pre-trained EfficientNet-V2-M backbone
   - Fine-tune entire model (backbone + classification heads)
   - Train on VeriWild dataset
   - Use strong augmentations for domain adaptation

3. **Model Architecture**:
   - Backbone: EfficientNet-V2-M (1280-dim features)
   - Classification Heads:
     * Make classification
     * Type classification
     * Color classification
   - Input size: 224x224

4. **Current Performance**:
   - Make Accuracy: 44.90% (improving with backbone fine-tuning)
   - Type Accuracy: 86.78%
   - Color Accuracy: 88.57%

## Training Command
```bash
python -m src.training.train_yolo11n_classifier \
  --config config/yolo11n_vehicle_config.yaml \
  --batch-size 256 \
  --num-workers 32 \
  --mixed-precision
```

## File Organization
```
