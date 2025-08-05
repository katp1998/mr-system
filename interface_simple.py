#!/usr/bin/env python3
"""
Simplified interface for the GAN-based Music Emotion Recognition System
This version works without librosa dependency.
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import List, Dict, Optional
import argparse

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main_simple import GANMER

class SimpleMERInterface:
    """Simplified user-friendly interface for the GAN-based MER system"""
    
    def __init__(self, models_dir: str = "models", dataset_dir: str = "EMOPIA_1.0"):
        """
        Initialize the MER interface
        
        Args:
            models_dir: Directory containing trained models
            dataset_dir: Directory containing EMOPIA dataset
        """
        self.models_dir = models_dir
        self.dataset_dir = dataset_dir
        
        # Initialize GAN MER system
        print("Initializing GAN-based MER system...")
        self.gan_mer = GANMER()
        
        # Load trained models if they exist
        if os.path.exists(models_dir):
            try:
                self.gan_mer.load_models(models_dir)
                print("✓ Trained models loaded successfully")
            except Exception as e:
                print(f"⚠ Warning: Could not load models from {models_dir}: {e}")
                print("You may need to train the models first by running main_simple.py")
        else:
            print(f"⚠ Warning: Models directory {models_dir} not found")
            print("You need to train the models first by running main_simple.py")
    
    def classify_emotion_from_midi(self, midi_path: str) -> Optional[Dict]:
        """
        Classify the emotion of a MIDI file
        
        Args:
            midi_path: Path to the MIDI file
            
        Returns:
            Dictionary containing emotion classification results
        """
        if not os.path.exists(midi_path):
            print(f"Error: MIDI file not found at {midi_path}")
            return None
        
        try:
            # Extract features from MIDI file
            features = self.gan_mer.extract_features_from_midi(midi_path)
            if features is None:
                print("Error: Could not extract features from MIDI file")
                return None
            
            # Classify emotion
            emotion_id, emotion_name = self.gan_mer.classify_emotion(features)
            
            # Get confidence scores
            self.gan_mer.classifier.eval()
            import torch
            with torch.no_grad():
                features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.gan_mer.device)
                outputs = self.gan_mer.classifier(features_tensor)
                confidence_scores = outputs.cpu().numpy()[0]
            
            result = {
                'file_path': midi_path,
                'emotion_id': emotion_id,
                'emotion_name': emotion_name,
                'confidence_scores': {
                    'Happy': float(confidence_scores[0]),
                    'Tense': float(confidence_scores[1]),
                    'Sad': float(confidence_scores[2]),
                    'Relaxed': float(confidence_scores[3])
                },
                'top_confidence': float(max(confidence_scores))
            }
            
            return result
            
        except Exception as e:
            print(f"Error classifying emotion: {e}")
            return None
    
    def get_emotion_recommendations(self, midi_path: str, num_recommendations: int = 5) -> List[Dict]:
        """
        Get emotion-based music recommendations
        
        Args:
            midi_path: Path to the input MIDI file
            num_recommendations: Number of recommendations to return
            
        Returns:
            List of recommendation dictionaries
        """
        if not os.path.exists(midi_path):
            print(f"Error: MIDI file not found at {midi_path}")
            return []
        
        try:
            recommendations = self.gan_mer.recommend_similar_tracks(
                midi_path, self.dataset_dir, num_recommendations
            )
            return recommendations
            
        except Exception as e:
            print(f"Error getting recommendations: {e}")
            return []
    
    def analyze_dataset_emotions(self) -> Dict:
        """
        Analyze emotion distribution in the dataset
        
        Returns:
            Dictionary containing emotion statistics
        """
        try:
            labels_df = pd.read_csv(os.path.join(self.dataset_dir, 'label.csv'))
            
            # Count emotions
            emotion_counts = labels_df['4Q'].value_counts().sort_index()
            
            # Map to emotion names
            emotion_mapping = {1: "Happy", 2: "Tense", 3: "Sad", 4: "Relaxed"}
            emotion_stats = {}
            
            for emotion_id, count in emotion_counts.items():
                emotion_name = emotion_mapping.get(emotion_id, f"Unknown_{emotion_id}")
                emotion_stats[emotion_name] = {
                    'count': int(count),
                    'percentage': float(count / len(labels_df) * 100)
                }
            
            return {
                'total_tracks': len(labels_df),
                'emotion_distribution': emotion_stats
            }
            
        except Exception as e:
            print(f"Error analyzing dataset: {e}")
            return {}
    
    def generate_synthetic_samples(self, emotion_name: str, num_samples: int = 5) -> np.ndarray:
        """
        Generate synthetic music features for a specific emotion
        
        Args:
            emotion_name: Name of the emotion (Happy, Tense, Sad, Relaxed)
            num_samples: Number of synthetic samples to generate
            
        Returns:
            Array of synthetic feature vectors
        """
        # Map emotion name to ID
        emotion_mapping = {"Happy": 0, "Tense": 1, "Sad": 2, "Relaxed": 3}
        
        if emotion_name not in emotion_mapping:
            print(f"Error: Unknown emotion '{emotion_name}'. Available emotions: {list(emotion_mapping.keys())}")
            return np.array([])
        
        emotion_id = emotion_mapping[emotion_name]
        
        try:
            synthetic_data = self.gan_mer.generate_synthetic_data(num_samples, emotion_id)
            print(f"Generated {len(synthetic_data)} synthetic samples for {emotion_name}")
            return synthetic_data
            
        except Exception as e:
            print(f"Error generating synthetic data: {e}")
            return np.array([])

def main():
    """Command-line interface for the simplified MER system"""
    parser = argparse.ArgumentParser(description="Simplified GAN-based Music Emotion Recognition System")
    parser.add_argument("--mode", choices=["classify", "recommend", "analyze", "generate"], 
                       required=True, help="Operation mode")
    parser.add_argument("--input", type=str, help="Input MIDI file path")
    parser.add_argument("--emotion", type=str, choices=["Happy", "Tense", "Sad", "Relaxed"],
                       help="Target emotion for generation")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of samples to generate")
    parser.add_argument("--num_recommendations", type=int, default=5, help="Number of recommendations")
    parser.add_argument("--models_dir", type=str, default="models", help="Models directory")
    parser.add_argument("--dataset_dir", type=str, default="EMOPIA_1.0", help="Dataset directory")
    
    args = parser.parse_args()
    
    # Initialize interface
    interface = SimpleMERInterface(args.models_dir, args.dataset_dir)
    
    if args.mode == "classify":
        if not args.input:
            print("Error: --input argument required for classify mode")
            return
        
        result = interface.classify_emotion_from_midi(args.input)
        if result:
            print(f"\n=== Emotion Classification Results ===")
            print(f"File: {result['file_path']}")
            print(f"Predicted Emotion: {result['emotion_name']}")
            print(f"Confidence: {result['top_confidence']:.3f}")
            print(f"\nConfidence Scores:")
            for emotion, score in result['confidence_scores'].items():
                print(f"  {emotion}: {score:.3f}")
    
    elif args.mode == "recommend":
        if not args.input:
            print("Error: --input argument required for recommend mode")
            return
        
        recommendations = interface.get_emotion_recommendations(args.input, args.num_recommendations)
        if recommendations:
            print(f"\n=== Top {len(recommendations)} Recommendations ===")
            for i, rec in enumerate(recommendations, 1):
                print(f"{i}. {rec['track_id']}")
                print(f"   Emotion: {rec['emotion']}")
                print(f"   Similarity: {rec['similarity_score']:.3f}")
                print(f"   File: {rec['file_path']}")
                print()
        else:
            print("No recommendations found")
    
    elif args.mode == "analyze":
        stats = interface.analyze_dataset_emotions()
        if stats:
            print(f"\n=== Dataset Analysis ===")
            print(f"Total Tracks: {stats['total_tracks']}")
            print(f"\nEmotion Distribution:")
            for emotion, data in stats['emotion_distribution'].items():
                print(f"  {emotion}: {data['count']} tracks ({data['percentage']:.1f}%)")
    
    elif args.mode == "generate":
        if not args.emotion:
            print("Error: --emotion argument required for generate mode")
            return
        
        synthetic_data = interface.generate_synthetic_samples(args.emotion, args.num_samples)
        if len(synthetic_data) > 0:
            print(f"\n=== Synthetic Data Generated ===")
            print(f"Emotion: {args.emotion}")
            print(f"Samples: {len(synthetic_data)}")
            print(f"Feature dimension: {synthetic_data.shape[1]}")
            print(f"Feature statistics:")
            print(f"  Mean: {np.mean(synthetic_data):.3f}")
            print(f"  Std: {np.std(synthetic_data):.3f}")
            print(f"  Min: {np.min(synthetic_data):.3f}")
            print(f"  Max: {np.max(synthetic_data):.3f}")

if __name__ == "__main__":
    main() 