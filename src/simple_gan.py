import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras

class SimpleGAN:
    def __init__(self, feature_dim, noise_dim=100):
        self.feature_dim = feature_dim
        self.noise_dim = noise_dim
        self.scaler = StandardScaler()
        
        # Build generator and discriminator
        self.generator = self._build_generator()
        self.discriminator = self._build_discriminator()
        
        # Combined model for training generator
        self.combined = self._build_combined()
    
    def _build_generator(self):
        """simple generator"""
        model = keras.Sequential([
            keras.layers.Dense(256, activation='relu', input_dim=self.noise_dim),
            keras.layers.Dense(512, activation='relu'),
            keras.layers.Dense(self.feature_dim, activation='tanh')
        ])
        return model
    
    def _build_discriminator(self):
        """simple discriminator"""
        model = keras.Sequential([
            keras.layers.Dense(512, activation='relu', input_dim=self.feature_dim),
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dense(1, activation='sigmoid')
        ])
        return model
    
    def _build_combined(self):
        """combined model for training generator"""
        self.discriminator.trainable = False
        model = keras.Sequential([
            self.generator,
            self.discriminator
        ])
        return model
    
    def _compile_models(self):
        """Compile all models"""
        # discriminator
        self.discriminator.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # combined model
        self.combined.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
            loss='binary_crossentropy'
        )
    
    def prepare_data(self, df, emotion):
        """Prepare data for specific emotion"""
        emotion_data = df[df['emotion'] == emotion]
        if len(emotion_data) == 0:
            return None, None
        
        # numeric features only
        numeric_features = emotion_data.select_dtypes(include=[np.number]).drop(['emotion'], axis=1, errors='ignore')
        
        # Scale features
        scaled_features = self.scaler.fit_transform(numeric_features)
        
        return scaled_features, numeric_features.columns
    
    def generate_samples(self, n_samples, emotion_data, feature_columns):
        """Generate new samples for specific emotion"""
        if emotion_data is None:
            return pd.DataFrame()
        
        # Generate noise
        noise = np.random.normal(0, 1, (n_samples, self.noise_dim))
        
        # Generate features
        generated_features = self.generator.predict(noise, verbose=0)
        
        # Inverse transform to original scale
        generated_features = self.scaler.inverse_transform(generated_features)
        
        # Create DataFrame
        generated_df = pd.DataFrame(generated_features, columns=feature_columns)
        generated_df['emotion'] = emotion_data.iloc[0]['emotion'] if len(emotion_data) > 0 else 'unknown'
        
        return generated_df
    
    def train(self, df, epochs=100, batch_size=32, eda_analyzer=None):
        """Train GAN on dataset"""
        # Compile models first
        self._compile_models()
        
        # Get unique emotions
        emotions = df['emotion'].unique()
        emotions = [e for e in emotions if e != 'unknown']
        
        # Prepare data for each emotion
        emotion_data = {}
        for emotion in emotions:
            data, columns = self.prepare_data(df, emotion)
            if data is not None:
                emotion_data[emotion] = (data, columns)
        
        # Training loop
        d_loss = 0
        g_loss = 0
        
        for epoch in range(epochs):
            for emotion in emotions:
                if emotion not in emotion_data:
                    continue
                
                data, columns = emotion_data[emotion]
                if len(data) < batch_size:
                    continue
                
                # Train discriminator
                real_data = data[np.random.choice(len(data), batch_size, replace=True)]
                noise = np.random.normal(0, 1, (batch_size, self.noise_dim))
                fake_data = self.generator.predict(noise, verbose=0)
                
                # Combine real and fake data
                d_data = np.vstack([real_data, fake_data])
                d_labels = np.hstack([np.ones(batch_size), np.zeros(batch_size)])
                
                # Train discriminator
                d_loss = self.discriminator.train_on_batch(d_data, d_labels)
                
                # Train generator
                noise = np.random.normal(0, 1, (batch_size, self.noise_dim))
                g_labels = np.ones(batch_size)
                g_loss = self.combined.train_on_batch(noise, g_labels)
            
            # Record training step for visualization
            if eda_analyzer:
                d_loss_val = d_loss[0] if isinstance(d_loss, (list, np.ndarray)) else d_loss
                g_loss_val = g_loss[0] if isinstance(g_loss, (list, np.ndarray)) else g_loss
                eda_analyzer.record_training_step(epoch, d_loss_val, g_loss_val)
            
            if epoch % 20 == 0:
                d_loss_val = d_loss[0] if isinstance(d_loss, (list, np.ndarray)) else d_loss
                g_loss_val = g_loss[0] if isinstance(g_loss, (list, np.ndarray)) else g_loss
                print(f"Epoch {epoch}, D Loss: {d_loss_val:.4f}, G Loss: {g_loss_val:.4f}")
    
    def balance_dataset(self, df, target_samples_per_emotion=200):
        """Balance dataset by generating samples for underrepresented emotions"""
        emotion_counts = df['emotion'].value_counts()
        balanced_df = df.copy()
        
        for emotion, count in emotion_counts.items():
            if emotion == 'unknown':
                continue
                
            if count < target_samples_per_emotion:
                needed = target_samples_per_emotion - count
                print(f"Generating {needed} samples for {emotion}")
                
                # Get existing data for this emotion
                emotion_data = df[df['emotion'] == emotion]
                data, columns = self.prepare_data(df, emotion)
                
                if data is not None:
                    # Generate new samples
                    generated = self.generate_samples(needed, emotion_data, columns)
                    balanced_df = pd.concat([balanced_df, generated], ignore_index=True)
        
        return balanced_df
