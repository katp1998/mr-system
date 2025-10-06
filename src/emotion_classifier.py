"""
Simple emotion classification based on MIDI feature thresholds
"""
import pandas as pd
import numpy as np

class EmotionClassifier:
    def __init__(self):
        self.emotion_thresholds = {
            'happy': {
                'pitch_min': 60, 'pitch_max': 76,
                'velocity_min': 80, 'velocity_max': 112,
                'tempo_min': 120, 'tempo_max': 140,
                'note_overlap_ratio_min': 0.0, 'note_overlap_ratio_max': 0.4
            },
            'sad': {
                'pitch_min': 48, 'pitch_max': 60,
                'velocity_min': 40, 'velocity_max': 70,
                'tempo_min': 60, 'tempo_max': 80,
                'note_overlap_ratio_min': 0.8, 'note_overlap_ratio_max': 1.0
            },
            'excited': {
                'pitch_min': 65, 'pitch_max': 84,
                'velocity_min': 100, 'velocity_max': 127,
                'tempo_min': 140, 'tempo_max': 180,
                'note_overlap_ratio_min': 0.0, 'note_overlap_ratio_max': 0.4
            },
            'hopeful': {
                'pitch_min': 60, 'pitch_max': 76,
                'velocity_min': 70, 'velocity_max': 100,
                'tempo_min': 100, 'tempo_max': 130,
                'note_overlap_ratio_min': 0.6, 'note_overlap_ratio_max': 0.8
            },
            'tense': {
                'pitch_min': 50, 'pitch_max': 65,
                'velocity_min': 90, 'velocity_max': 127,
                'tempo_min': 100, 'tempo_max': 130,
                'note_overlap_ratio_min': 0.0, 'note_overlap_ratio_max': 0.3
            },
            'fearful': {
                'pitch_min': 45, 'pitch_max': 60,
                'velocity_min': 50, 'velocity_max': 90,
                'tempo_min': 60, 'tempo_max': 100,
                'note_overlap_ratio_min': 0.0, 'note_overlap_ratio_max': 0.3
            }
        }
    
    def classify_emotion(self, features):
        """Classify emotion based on feature thresholds (90% matching)"""
        if not features:
            return 'unknown'
        
        emotion_scores = {}
        
        for emotion, thresholds in self.emotion_thresholds.items():
            score = 0
            total_criteria = 0
            
            # Check pitch range (use pitch_mean)
            if 'pitch_mean' in features:
                total_criteria += 1
                if thresholds['pitch_min'] <= features['pitch_mean'] <= thresholds['pitch_max']:
                    score += 1
            
            # Check velocity range (use velocity_mean)
            if 'velocity_mean' in features:
                total_criteria += 1
                if thresholds['velocity_min'] <= features['velocity_mean'] <= thresholds['velocity_max']:
                    score += 1
            
            # Check tempo range
            if 'tempo_mean' in features:
                total_criteria += 1
                if thresholds['tempo_min'] <= features['tempo_mean'] <= thresholds['tempo_max']:
                    score += 1
            
            # Check note overlap ratio
            if 'note_overlap_ratio' in features:
                total_criteria += 1
                if thresholds['note_overlap_ratio_min'] <= features['note_overlap_ratio'] <= thresholds['note_overlap_ratio_max']:
                    score += 1
            
            # Calculate percentage match
            if total_criteria > 0:
                emotion_scores[emotion] = score / total_criteria
        
        # Return emotion with highest score if >= 50% match (lowered from 75%)
        if emotion_scores:
            best_emotion = max(emotion_scores, key=emotion_scores.get)
            if emotion_scores[best_emotion] >= 0.5:
                return best_emotion
        
        return 'unknown'
    
    def classify_dataset(self, df):
        """Classify entire dataset"""
        emotions = []
        for _, row in df.iterrows():
            emotion = self.classify_emotion(row.to_dict())
            emotions.append(emotion)
        
        df['emotion'] = emotions
        return df
    
    def get_emotion_distribution(self, df):
        """Get distribution of emotions in dataset"""
        return df['emotion'].value_counts()
