#!/usr/bin/env python3
"""
Basic test script for the GAN-based MER system
Tests core functionality without requiring full training.
"""

import os
import sys
import numpy as np
import pandas as pd

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_dataset_loading():
    """Test if the EMOPIA dataset can be loaded"""
    print("🔍 Testing dataset loading...")
    
    try:
        # Check if dataset exists
        if not os.path.exists("EMOPIA_1.0"):
            print("❌ EMOPIA dataset not found!")
            return False
        
        # Check if label file exists
        label_file = "EMOPIA_1.0/label.csv"
        if not os.path.exists(label_file):
            print("❌ Label file not found!")
            return False
        
        # Load labels
        labels_df = pd.read_csv(label_file)
        print(f"✅ Loaded {len(labels_df)} labels from dataset")
        
        # Check emotion distribution
        emotion_counts = labels_df['4Q'].value_counts().sort_index()
        print("📊 Emotion distribution:")
        emotion_mapping = {1: "Happy", 2: "Tense", 3: "Sad", 4: "Relaxed"}
        for emotion_id, count in emotion_counts.items():
            emotion_name = emotion_mapping.get(emotion_id, f"Unknown_{emotion_id}")
            print(f"   {emotion_name}: {count} tracks")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return False

def test_midi_files():
    """Test if MIDI files can be found and processed"""
    print("\n🎵 Testing MIDI files...")
    
    try:
        midi_dir = "EMOPIA_1.0/midis"
        if not os.path.exists(midi_dir):
            print("❌ MIDI directory not found!")
            return False
        
        # Count MIDI files
        midi_files = [f for f in os.listdir(midi_dir) if f.endswith('.mid')]
        print(f"✅ Found {len(midi_files)} MIDI files")
        
        if len(midi_files) == 0:
            print("❌ No MIDI files found!")
            return False
        
        # Test first few files
        print("🔍 Testing first 3 MIDI files:")
        for i, filename in enumerate(midi_files[:3]):
            file_path = os.path.join(midi_dir, filename)
            file_size = os.path.getsize(file_path)
            print(f"   {i+1}. {filename} ({file_size} bytes)")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing MIDI files: {e}")
        return False

def test_feature_extraction():
    """Test MIDI feature extraction"""
    print("\n🔧 Testing feature extraction...")
    
    try:
        # Import required modules
        import pretty_midi
        
        midi_dir = "EMOPIA_1.0/midis"
        midi_files = [f for f in os.listdir(midi_dir) if f.endswith('.mid')]
        
        if len(midi_files) == 0:
            print("❌ No MIDI files to test!")
            return False
        
        # Test feature extraction on first file
        test_file = os.path.join(midi_dir, midi_files[0])
        print(f"📁 Testing: {midi_files[0]}")
        
        # Load MIDI file
        midi_data = pretty_midi.PrettyMIDI(test_file)
        
        # Extract basic features
        features = []
        
        # Tempo features
        tempo = midi_data.estimate_tempo()
        features.extend([tempo, midi_data.get_tempo_changes()[1].mean()])
        
        # Pitch features
        all_notes = []
        for instrument in midi_data.instruments:
            for note in instrument.notes:
                all_notes.append(note.pitch)
        
        if not all_notes:
            print("❌ No notes found in MIDI file!")
            return False
        
        all_notes = np.array(all_notes)
        features.extend([
            all_notes.mean(),  # Average pitch
            all_notes.std(),   # Pitch standard deviation
            all_notes.max() - all_notes.min(),  # Pitch range
            len(np.unique(all_notes))  # Unique pitches
        ])
        
        # Duration features
        durations = [note.end - note.start for instrument in midi_data.instruments for note in instrument.notes]
        if durations:
            features.extend([
                np.mean(durations),
                np.std(durations),
                np.max(durations)
            ])
        else:
            features.extend([0, 0, 0])
        
        # Velocity features
        velocities = [note.velocity for instrument in midi_data.instruments for note in instrument.notes]
        if velocities:
            features.extend([
                np.mean(velocities),
                np.std(velocities)
            ])
        else:
            features.extend([0, 0])
        
        # Time signature features
        if midi_data.time_signature_changes:
            features.extend([
                midi_data.time_signature_changes[0].numerator,
                midi_data.time_signature_changes[0].denominator
            ])
        else:
            features.extend([4, 4])  # Default 4/4
        
        features = np.array(features)
        print(f"✅ Extracted {len(features)} features")
        print(f"📊 Feature statistics:")
        print(f"   Mean: {np.mean(features):.3f}")
        print(f"   Std:  {np.std(features):.3f}")
        print(f"   Min:  {np.min(features):.3f}")
        print(f"   Max:  {np.max(features):.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in feature extraction: {e}")
        return False

def test_gan_architecture():
    """Test GAN architecture creation"""
    print("\n🧠 Testing GAN architecture...")
    
    try:
        import torch
        import torch.nn as nn
        
        # Test if we can create the models
        from main import Generator, Discriminator, EmotionClassifier
        
        # Create models
        generator = Generator(latent_dim=100, feature_dim=128, num_classes=4)
        discriminator = Discriminator(feature_dim=128, num_classes=4)
        classifier = EmotionClassifier(feature_dim=128, num_classes=4)
        
        print(f"✅ Generator created: {sum(p.numel() for p in generator.parameters())} parameters")
        print(f"✅ Discriminator created: {sum(p.numel() for p in discriminator.parameters())} parameters")
        print(f"✅ Classifier created: {sum(p.numel() for p in classifier.parameters())} parameters")
        
        # Test forward pass
        batch_size = 4
        z = torch.randn(batch_size, 100)
        labels = torch.zeros(batch_size, 4)
        labels[:, 0] = 1  # Happy emotion
        
        # Generator forward pass
        fake_features = generator(z, labels)
        print(f"✅ Generator output shape: {fake_features.shape}")
        
        # Discriminator forward pass
        validity, emotion_labels = discriminator(fake_features)
        print(f"✅ Discriminator output shapes: {validity.shape}, {emotion_labels.shape}")
        
        # Classifier forward pass
        outputs = classifier(fake_features)
        print(f"✅ Classifier output shape: {outputs.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in GAN architecture: {e}")
        return False

def main():
    """Run all tests"""
    print("🎼 GAN-based MER System - Basic Tests")
    print("=" * 50)
    
    tests = [
        ("Dataset Loading", test_dataset_loading),
        ("MIDI Files", test_midi_files),
        ("Feature Extraction", test_feature_extraction),
        ("GAN Architecture", test_gan_architecture)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} ERROR: {e}")
    
    print(f"\n{'='*50}")
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The system is ready to use.")
        print("\n💡 Next steps:")
        print("   1. Train the system: python main.py")
        print("   2. Run the demo: python demo.py")
        print("   3. Use the interface: python mer_interface.py --help")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        print("\n🔧 Troubleshooting:")
        print("   1. Make sure EMOPIA_1.0.zip is extracted")
        print("   2. Install dependencies: pip install -r requirements.txt")
        print("   3. Check if all required files are present")

if __name__ == "__main__":
    main() 