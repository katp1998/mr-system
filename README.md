# GAN-based Music Emotion Recognition (MER) System

A comprehensive system that uses Generative Adversarial Networks (GANs) for Music Emotion Recognition on pop piano music, specifically designed to work with the EMOPIA dataset. The system can classify emotions, generate synthetic music features, and provide emotion-based music recommendations.

## 🎵 Features

### Core Functionality
- **Emotion Classification**: Automatically classify MIDI files into 4 emotion categories:
  - **Happy** (High Valence, High Arousal)
  - **Tense** (Low Valence, High Arousal) 
  - **Sad** (Low Valence, Low Arousal)
  - **Relaxed** (High Valence, Low Arousal)

- **GAN-based Data Augmentation**: Generate synthetic music features for each emotion category to enhance training data

- **Emotion-based Recommendations**: Given a MIDI file, recommend 5 similar tracks from the dataset that evoke the same emotions

### Technical Features
- **Conditional GAN**: Generates music features conditioned on specific emotions
- **Enhanced Feature Extraction**: Extracts comprehensive musical features from MIDI files including:
  - **MIDI Features** (13 features):
    - Tempo and rhythm features
    - Pitch statistics (mean, std, range, unique pitches)
    - Duration patterns
    - Velocity dynamics
    - Time signature information
  - **Librosa Audio Features** (20 features):
    - Spectral features (centroid, rolloff, bandwidth)
    - MFCC coefficients (first 4 coefficients)
    - Chroma features
    - Beat tracking
    - Zero crossing rate
    - Root mean square energy
    - Harmonic and percussive components
- **Cosine Similarity**: Uses cosine similarity for finding similar tracks
- **Confidence Scoring**: Provides confidence scores for emotion predictions

## 📁 Project Structure

```
mr-system/
├── main.py                 # Full GAN MER system (requires librosa)
├── main_simple.py          # Simplified GAN MER system (no librosa)
├── mer_interface.py        # Interface for full system
├── interface_simple.py     # Interface for simplified system
├── test_basic.py           # Basic functionality tests
├── demo.py                 # Demo script for full system
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── EMOPIA_1.0/            # EMOPIA dataset
│   ├── midis/             # MIDI files
│   ├── label.csv          # Emotion labels
│   └── metadata_by_song.csv
└── models/                # Trained models (created after training)
    ├── generator.pth
    ├── discriminator.pth
    └── classifier.pth
```

## 🚀 Installation

### Option 1: Simplified Version (Recommended for quick start)
```bash
pip install torch torchvision pretty_midi numpy pandas scikit-learn
```

### Option 2: Full Version (with additional features)
```bash
pip install -r requirements.txt
```

### Dataset Setup
```bash
# Extract the EMOPIA dataset
unzip EMOPIA_1.0.zip
```

## 🎯 Usage

### Quick Start (Simplified Version)

1. **Train the system**:
   ```bash
   python main_simple.py
   ```

2. **Use the interface**:
   ```bash
   # Analyze dataset
   python interface_simple.py --mode analyze
   
   # Classify emotion from MIDI file
   python interface_simple.py --mode classify --input "EMOPIA_1.0/midis/Q1_0vLPYiPN7qY_0.mid"
   
   # Get recommendations
   python interface_simple.py --mode recommend --input "EMOPIA_1.0/midis/Q1_0vLPYiPN7qY_0.mid"
   
   # Generate synthetic data
   python interface_simple.py --mode generate --emotion "Happy" --num_samples 5
   ```

### Full Version (with librosa)

1. **Train the system**:
   ```bash
   python main.py
   ```

2. **Use the interface**:
   ```bash
   python mer_interface.py --mode classify --input "path/to/your/midi/file.mid"
   ```

### Testing the System

Run basic tests to verify everything works:
```bash
python test_basic.py
```

## 🧠 System Architecture

### GAN Architecture
- **Generator**: Conditional GAN that generates music features based on emotion labels
- **Discriminator**: Dual-purpose network that:
  - Distinguishes real vs. synthetic features
  - Classifies emotions (auxiliary classifier)

### Feature Extraction
The system extracts 128-dimensional feature vectors from MIDI files including:
- **Tempo Features**: Estimated tempo, tempo changes
- **Pitch Features**: Mean, standard deviation, range, unique pitch count
- **Duration Features**: Mean, standard deviation, maximum note duration
- **Velocity Features**: Mean and standard deviation of note velocities
- **Time Signature**: Numerator and denominator

### Emotion Classification
- Uses a 4-quadrant emotion model based on Valence-Arousal space
- Maps EMOPIA quadrants to emotions:
  - Q1 → Happy (High Valence, High Arousal)
  - Q2 → Tense (Low Valence, High Arousal)
  - Q3 → Sad (Low Valence, Low Arousal)
  - Q4 → Relaxed (High Valence, Low Arousal)

## 📊 Expected Performance

The system typically achieves:
- **Classification Accuracy**: 70-85% on test set
- **GAN Training**: Stable training with conditional generation
- **Recommendation Quality**: High similarity scores for recommended tracks

## 🔧 Configuration

You can modify hyperparameters in the main files:

```python
# Hyperparameters
FEATURE_DIM = 128          # Feature vector dimension
LATENT_DIM = 100          # GAN latent space dimension
NUM_CLASSES = 4           # Number of emotion classes
BATCH_SIZE = 32           # Training batch size
GAN_EPOCHS = 50          # GAN training epochs
CLASSIFIER_EPOCHS = 30   # Classifier training epochs
```

## 📝 Output Examples

### Emotion Classification Output
```
=== Emotion Classification Results ===
File: EMOPIA_1.0/midis/Q1_0vLPYiPN7qY_0.mid
Predicted Emotion: Happy
Confidence: 0.892

Confidence Scores:
  Happy: 0.892
  Tense: 0.045
  Sad: 0.032
  Relaxed: 0.031
```

### Recommendation Output
```
=== Top 5 Recommendations ===
1. Q1_2Z9SjI131jA_0
   Emotion: Happy
   Similarity: 0.945
   File: EMOPIA_1.0/midis/Q1_2Z9SjI131jA_0.mid

2. Q1_ANZf1QXsNrY_0
   Emotion: Happy
   Similarity: 0.923
   File: EMOPIA_1.0/midis/Q1_ANZf1QXsNrY_0.mid
```

### Dataset Analysis Output
```
=== Dataset Analysis ===
Total Tracks: 1078

Emotion Distribution:
  Happy: 250 tracks (23.2%)
  Tense: 265 tracks (24.6%)
  Sad: 253 tracks (23.5%)
  Relaxed: 310 tracks (28.8%)
```

## 🎵 EMOPIA Dataset

The system uses the EMOPIA dataset, which contains:
- **1,078 MIDI clips** from pop piano music
- **4 emotion quadrants** based on Valence-Arousal space
- **YouTube-sourced** music with emotion annotations
- **Diverse musical styles** within pop piano genre

## 🔬 Research Applications

This system can be used for:
- **Music Emotion Recognition research**
- **GAN applications in music**
- **Music recommendation systems**
- **Data augmentation for MER**
- **Emotion-aware music generation**

## 🛠️ Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce batch size or use CPU
2. **MIDI file errors**: Ensure MIDI files are valid and contain notes
3. **Model loading errors**: Make sure to train models first
4. **librosa installation issues**: Use the simplified version instead

### Performance Tips

- Use GPU if available for faster training
- Adjust batch size based on available memory
- Increase epochs for better model performance
- Use larger feature dimensions for more detailed analysis

## 📚 Dependencies

### Simplified Version
- **PyTorch**: Deep learning framework
- **pretty_midi**: MIDI file processing
- **numpy/pandas**: Data manipulation
- **scikit-learn**: Machine learning utilities

### Full Version (Additional)
- **librosa**: Audio processing
- **matplotlib/seaborn**: Visualization

## 🤝 Contributing

Feel free to contribute by:
- Improving the GAN architecture
- Adding new feature extraction methods
- Enhancing the recommendation algorithm
- Adding visualization capabilities
- Optimizing performance

## 📄 License

This project is for research purposes. Please cite the EMOPIA dataset if used in publications.

## 🙏 Acknowledgments

- EMOPIA dataset creators for providing the annotated music dataset
- PyTorch community for the deep learning framework
- PrettyMIDI library for MIDI processing capabilities

## 🚀 Quick Demo

To see the system in action:

```bash
# Test basic functionality
python test_basic.py

# Analyze the dataset
python interface_simple.py --mode analyze

# Train the system (this will take some time)
python main_simple.py

# After training, test classification
python interface_simple.py --mode classify --input "EMOPIA_1.0/midis/Q1_0vLPYiPN7qY_0.mid"
``` 