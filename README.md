# GAN-based Music Emotion Recognition (MER) System

A comprehensive system that uses Generative Adversarial Networks (GANs) for Music Emotion Recognition on pop piano music, specifically designed to work with the EMOPIA dataset. The system can classify emotions, generate synthetic music features, and provide emotion-based music recommendations.

## Features

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

## Project Structure

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
-- add the model paths too!!
```

## Installation

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

## Usage

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

## Quick Demo

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