import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

class RecommendationSystem:
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.trained_data = None
    
    def prepare_features(self, df):
        """Prepare features for similarity calculation"""
        # Select numeric features only
        numeric_features = df.select_dtypes(include=[np.number]).drop(['emotion'], axis=1, errors='ignore')
        self.feature_columns = numeric_features.columns
        
        # Scale features
        scaled_features = self.scaler.fit_transform(numeric_features)
        
        return scaled_features
    
    def train(self, df):
        """Train recommendation system"""
        print("Training recommendation system...")
        
        # Prepare features
        features = self.prepare_features(df)
        
        # Store training data
        self.trained_data = df.copy()
        self.trained_data['scaled_features'] = list(features)
        
        print(f"Trained on {len(df)} samples")
        print(f"Feature dimensions: {features.shape[1]}")
    
    def find_similar(self, query_file_path, n_recommendations=5):
        """Find similar tracks to query file"""
        if self.trained_data is None:
            print("System not trained yet!")
            return []
        
        # Find query file in training data
        query_row = self.trained_data[self.trained_data['file_path'] == query_file_path]
        
        if len(query_row) == 0:
            print(f"Query file {query_file_path} not found in training data!")
            return []
        
        query_features = query_row.iloc[0]['scaled_features'].reshape(1, -1)
        query_emotion = query_row.iloc[0]['emotion']
        
        # Get all files with same emotion (ONLY from original dataset, not GAN-generated)
        # Filter out GAN-generated samples by checking if file_path is not NaN and exists
        same_emotion_data = self.trained_data[
            (self.trained_data['emotion'] == query_emotion) & 
            (self.trained_data['file_path'].notna()) &
            (self.trained_data['file_path'] != '') &
            (self.trained_data['file_path'].str.contains('.mid', na=False))
        ]
        
        if len(same_emotion_data) <= 1:
            print(f"Not enough samples in emotion category: {query_emotion}")
            return []
        
        # Calculate similarities
        similarities = []
        for idx, row in same_emotion_data.iterrows():
            if row['file_path'] == query_file_path:
                continue  # Skip the query file itself
            
            # Skip if file_path is NaN or empty
            if pd.isna(row['file_path']) or row['file_path'] == '':
                continue
                
            features = row['scaled_features'].reshape(1, -1)
            similarity = cosine_similarity(query_features, features)[0][0]
            similarities.append((row['file_path'], similarity))
        
        # Sort by similarity and return top N
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:n_recommendations]
    
    def get_emotion_stats(self):
        """Get statistics about trained data"""
        if self.trained_data is None:
            return {}
        
        return {
            'total_samples': len(self.trained_data),
            'emotion_distribution': self.trained_data['emotion'].value_counts().to_dict(),
            'feature_dimensions': len(self.feature_columns) if self.feature_columns is not None else 0
        }
