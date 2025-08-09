#!/usr/bin/env python3
"""
Demo script for the GAN-based Music Emotion Recognition System
This script demonstrates the main features of the system.
"""

import os
import sys
import numpy as np
from typing import List

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import GANMER
from mer_interface import MERInterface

def demo_emotion_classification(interface: MERInterface):
    """Demonstrate emotion classification"""
    print("=== EMOTION CLASSIFICATION DEMO ===")
    
    # Find a sample MIDI file
    midi_dir = "EMOPIA_1.0/midis"
    if not os.path.exists(midi_dir):
        print("EMOPIA dataset not found. Please extract the dataset first.")
        return
    
    sample_files = [f for f in os.listdir(midi_dir) if f.endswith('.mid')]
    if not sample_files:
        print("No MIDI files found in the dataset.")
        return
    
    # Test with first few samples
    for i, filename in enumerate(sample_files[:3]):
        midi_path = os.path.join(midi_dir, filename)
        print(f"\n Testing file {i+1}: {filename}")
        
        result = interface.classify_emotion_from_midi(midi_path)
        if result:
            print(f" Predicted Emotion: {result['emotion_name']}")
            print(f" Confidence: {result['top_confidence']:.3f}")
            print(f" Confidence Breakdown:")
            for emotion, score in result['confidence_scores'].items():
                bar = "█" * int(score * 20)
                print(f"      {emotion:8}: {score:.3f} {bar}")

def demo_recommendations(interface: MERInterface):
    """Demonstrate recommendation system"""
    print("\n === RECOMMENDATION SYSTEM DEMO ===")
    
    # Find a sample MIDI file
    midi_dir = "EMOPIA_1.0/midis"
    if not os.path.exists(midi_dir):
        print(" EMOPIA dataset not found.")
        return
    
    sample_files = [f for f in os.listdir(midi_dir) if f.endswith('.mid')]
    if not sample_files:
        print(" No MIDI files found.")
        return
    
    # Test with first sample
    midi_path = os.path.join(midi_dir, sample_files[0])
    print(f"Input file: {sample_files[0]}")
    
    # Get recommendations
    recommendations = interface.get_emotion_recommendations(midi_path, num_recommendations=5)
    
    if recommendations:
        print(f"\n Top 5 Recommendations:")
        for i, rec in enumerate(recommendations, 1):
            print(f"   {i}. {rec['track_id']}")
            print(f"      Emotion: {rec['emotion']}")
            print(f"      Similarity: {rec['similarity_score']:.3f}")
    else:
        print(" No recommendations found.")

def demo_dataset_analysis(interface: MERInterface):
    """Demonstrate dataset analysis"""
    print("\n === DATASET ANALYSIS DEMO ===")
    
    stats = interface.analyze_dataset_emotions()
    if stats:
        print(f" Total Tracks: {stats['total_tracks']}")
        print(f"\n Emotion Distribution:")
        
        for emotion, data in stats['emotion_distribution'].items():
            percentage = data['percentage']
            bar = "" * int(percentage / 2)  # Scale bar to fit
            print(f"   {emotion:8}: {data['count']:3d} tracks ({percentage:5.1f}%) {bar}")
    else:
        print(" Could not analyze dataset.")

def demo_synthetic_generation(interface: MERInterface):
    """Demonstrate synthetic data generation"""
    print("\n === SYNTHETIC DATA GENERATION DEMO ===")
    
    emotions = ["Happy", "Tense", "Sad", "Relaxed"]
    
    for emotion in emotions:
        print(f"\n Generating {emotion} music features...")
        synthetic_data = interface.generate_synthetic_samples(emotion, num_samples=3)
        
        if len(synthetic_data) > 0:
            print(f"    Generated {len(synthetic_data)} samples")
            print(f"    Feature statistics:")
            print(f"    Mean: {np.mean(synthetic_data):.3f}")
            print(f"    Std:  {np.std(synthetic_data):.3f}")
            print(f"    Min:  {np.min(synthetic_data):.3f}")
            print(f"    Max:  {np.max(synthetic_data):.3f}")
        else:
            print(f"Failed to generate {emotion} samples")

def demo_gan_mer_system():
    """Demonstrate the core GAN MER system"""
    print(" === GAN MER SYSTEM DEMO ===")
    
    # Initialize system
    gan_mer = GANMER()
    
    # Check if models exist
    models_dir = "models"
    if os.path.exists(models_dir):
        try:
            gan_mer.load_models(models_dir)
            print(" Trained models loaded successfully")
        except Exception as e:
            print(f" Warning: Could not load models: {e}")
            print("   You may need to train the models first with: python main.py")
            return
    else:
        print("Models directory not found.")
        print("   Please train the models first with: python main.py")
        return
    
    # Test feature extraction
    midi_dir = "EMOPIA_1.0/midis"
    if os.path.exists(midi_dir):
        sample_files = [f for f in os.listdir(midi_dir) if f.endswith('.mid')]
        if sample_files:
            test_file = os.path.join(midi_dir, sample_files[0])
            print(f"\n Testing feature extraction on: {sample_files[0]}")
            
            features = gan_mer.extract_features_from_midi(test_file)
            if features is not None:
                print(f"    Features extracted successfully")
                print(f"    Feature dimension: {len(features)}")
                print(f"    Feature range: [{np.min(features):.3f}, {np.max(features):.3f}]")
                
                # Test emotion classification
                emotion_id, emotion_name = gan_mer.classify_emotion(features)
                print(f"    Classified emotion: {emotion_name}")
            else:
                print(f"    Feature extraction failed")

def main():
    """Main demo function"""
    print(" GAN-based Music Emotion Recognition System - DEMO")
    print("=" * 60)
    
    # Check if dataset exists
    if not os.path.exists("EMOPIA_1.0"):
        print(" EMOPIA dataset not found!")
        print("   Please extract the EMOPIA_1.0.zip file first.")
        return
    
    # Initialize interface
    print("\n Initializing system...")
    interface = MERInterface()
    
    # Run demos
    try:
        demo_emotion_classification(interface)
        demo_recommendations(interface)
        demo_dataset_analysis(interface)
        demo_synthetic_generation(interface)
        demo_gan_mer_system()
        
        print("\n Demo completed successfully!")
        print("\n Next steps:")
        print("   1. Train the system: python main.py")
        print("   2. Use the interface: python mer_interface.py --help")
        print("   3. Explore the code in main.py and mer_interface.py")
        
    except Exception as e:
        print(f"\n Demo failed with error: {e}")
        print("   Make sure you have installed all dependencies:")
        print("   pip install -r requirements.txt")

if __name__ == "__main__":
    main() 