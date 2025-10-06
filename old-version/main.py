import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pretty_midi
import librosa
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict, Optional
import warnings
import re

warnings.filterwarnings('ignore')

# GAN-based Music Emotion Recognition System with Generative Adversarial Networks for Data Augmentation

# Class Purpose: Load dataset and preprocess -- with 6 emotion classes
class EMOPIADataset(Dataset):
    
    def __init__(self, midi_dir: str, label_file: str, sequence_length: int = 165):
        self.midi_dir = midi_dir
        self.sequence_length = sequence_length
        self.data = []
        self.labels = []
        self.sample_metrics = []
        self.file_paths = []  # Store file paths for track ID retrieval (final data only)
        
        # Load labels
        labels_df = pd.read_csv(label_file)
        
        # Emotion mapping for 6 classes
        self.emotion_mapping = {
            0: "Happy",      # Q1
            1: "Excited",    # Q1 
            2: "Tense",      # Q2 
            3: "Fearful",    # Q2 
            4: "Sad",        # Q3
            5: "Hopeful"     # Q4
        }
        
        # allocate arousal and tension proxies for splitting Q1 and Q2 -- as two emotions are assigned per quadrant
        arousal_proxies = []
        tension_proxies = []
        
        for _, row in labels_df.iterrows():
            midi_file = f"{row['ID']}.mid"
            midi_path = os.path.join(midi_dir, midi_file)
            
            if os.path.exists(midi_path):
                try:
                    midi_data = pretty_midi.PrettyMIDI(midi_path)
                    features, metrics = self.extract_midi_features(midi_data, midi_path)
                    
                    if features is not None:
                        # Calculate arousal and tension proxies
                        arousal = (metrics.get('tempo', 120) + 120 * metrics.get('rms_mean', 0.1) + 
                                 10 * metrics.get('note_density', 0.1) + metrics.get('spectral_centroid_mean', 1000) / 1000.0)
                        tension = (metrics.get('spectral_bandwidth_mean', 1000) / 1000.0 + 
                                 10 * metrics.get('zcr_mean', 0.1) + 5 * metrics.get('chroma_std', 0.1))
                        
                        arousal_proxies.append(arousal)
                        tension_proxies.append(tension)
                        self.sample_metrics.append(metrics)
                except Exception as e:
                    print(f"Error processing {midi_path}: {e}")
                    arousal_proxies.append(0)
                    tension_proxies.append(0)
                    self.sample_metrics.append({})
        
        # Calculate medians for splitting
        q1_indices = [i for i, q in enumerate(labels_df['4Q'] - 1) if q == 0]
        q2_indices = [i for i, q in enumerate(labels_df['4Q'] - 1) if q == 1]
        
        if q1_indices:
            q1_arousal_median = np.median([arousal_proxies[i] for i in q1_indices if i < len(arousal_proxies)])
        else:
            q1_arousal_median = 0
        
        if q2_indices:
            q2_tension_median = np.median([tension_proxies[i] for i in q2_indices if i < len(tension_proxies)])
        else:
            q2_tension_median = 0
        
        # Process each MIDI file with validation
        for index, (_, row) in enumerate(labels_df.iterrows()):
            midi_file = f"{row['ID']}.mid"
            midi_path = os.path.join(midi_dir, midi_file)
            
            if os.path.exists(midi_path):
                try:
                    # Validate file naming convention
                    file_quadrant = self._extract_quadrant_from_filename(midi_file)
                    csv_quadrant = row['4Q'] - 1 # Convert to 0-based
                    
                    # Skip if file naming doesn't match CSV quadrant
                    if file_quadrant is not None and file_quadrant != csv_quadrant:
                        print(f"Warning: File {midi_file} naming ({file_quadrant+1}) doesn't match CSV quadrant ({csv_quadrant+1}). Skipping.")
                        continue
                    
                    midi_data = pretty_midi.PrettyMIDI(midi_path)
                    features, metrics = self.extract_midi_features(midi_data, midi_path)
                    
                    if features is not None:
                        self.data.append(features)
                        self.file_paths.append(midi_path)  # Store file path for this data point
                        
                        # Map 4Q to 6 emotions
                        q = csv_quadrant
                        
                        if q == 0:  # Q1
                            arousal = arousal_proxies[index] if index < len(arousal_proxies) else 0
                            label6 = 1 if arousal > q1_arousal_median else 0
                        elif q == 1:  # Q2
                            tension = tension_proxies[index] if index < len(tension_proxies) else 0
                            label6 = 3 if tension > q2_tension_median else 2
                        elif q == 2:  # Q3
                            label6 = 4
                        else:  # Q4
                            label6 = 5
                        
                        self.labels.append(label6)
                        
                except Exception as e:
                    print(f"Error processing {midi_path}: {e}")
    
    # get quadrant from filename
    def _extract_quadrant_from_filename(self, filename: str) -> Optional[int]:
        match = re.match(r'Q(\d+)_', filename)
        if match:
            return int(match.group(1)) - 1  # Convert to 0-based
        return None
    
    # get features from midi file
    def extract_midi_features(self, midi_data: pretty_midi.PrettyMIDI, midi_path: str) -> Tuple[np.ndarray, Dict]:
        features = []
        metrics = {}
        
        try:
            # Basic MIDI features
            all_notes = []
            for instrument in midi_data.instruments:
                if not instrument.is_drum:
                    all_notes.extend(instrument.notes)
            
            if all_notes:
                pitches = [note.pitch for note in all_notes]
                velocities = [note.velocity for note in all_notes]
                durations = [note.end - note.start for note in all_notes]
                
                features.extend([
                    np.mean(pitches), np.std(pitches), np.min(pitches), np.max(pitches),
                    np.mean(velocities), np.std(velocities), np.mean(durations), np.std(durations)
                ])
            else:
                features.extend([0] * 8)
            
            # Tempo
            tempo = midi_data.estimate_tempo()
            features.append(tempo)
            metrics['tempo'] = float(tempo)
            
            # Note density
            total_duration = midi_data.get_end_time()
            note_density = len(all_notes) / total_duration if total_duration > 0 else 0
            features.append(note_density)
            metrics['note_density'] = float(note_density)
            
            # Convert to audio for librosa features
            try:
                audio_data = midi_data.synthesize(fs=22050)
                librosa_features, librosa_metrics = self._extract_librosa_features(audio_data, sr=22050)
                features.extend(librosa_features)
                metrics.update(librosa_metrics)
            except Exception as e:
                print(f"Warning: Could not extract librosa features from {midi_path}: {e}")
                features.extend([0.0] * 16)  # Default librosa features
                metrics.update({
                    'spectral_centroid_mean': 0.0,
                    'spectral_bandwidth_mean': 0.0,
                    'zcr_mean': 0.0,
                    'rms_mean': 0.0,
                    'chroma_std': 0.0
                })
            
            # Pad or truncate to fixed length
            features = np.array(features)
            if len(features) < self.sequence_length:
                features = np.pad(features, (0, self.sequence_length - len(features)))
            else:
                features = features[:self.sequence_length]
            
            return features, metrics
            
        except Exception as e:
            print(f"Error extracting features from {midi_path}: {e}")
            return None, None
    
    # get librosa features from audio data
    def _extract_librosa_features(self, audio_data: np.ndarray, sr: int = 22050) -> Tuple[List[float], Dict[str, float]]:
        
        features: List[float] = []
        metrics: Dict[str, float] = {}
        
        try:
            # Ensure audio_data is 1D and limit length
            if len(audio_data.shape) > 1:
                audio_data = audio_data.flatten()
            if len(audio_data) > sr * 10:  # Limit to 10 seconds
                audio_data = audio_data[:sr * 10]
            
            # Essential spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=sr, hop_length=512)[0]
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio_data, sr=sr, hop_length=512)[0]
            features.extend([
                float(np.mean(spectral_centroids)), float(np.std(spectral_centroids)),
                float(np.mean(spectral_bandwidth)), float(np.std(spectral_bandwidth))
            ])
            metrics['spectral_centroid_mean'] = float(np.mean(spectral_centroids))
            metrics['spectral_bandwidth_mean'] = float(np.mean(spectral_bandwidth))
            
            # MFCC (first 4 coefficients)
            mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=4, hop_length=512)
            for i in range(4):
                features.extend([float(np.mean(mfccs[i])), float(np.std(mfccs[i]))])
            
            # Chroma features
            chroma = librosa.feature.chroma_stft(y=audio_data, sr=sr, hop_length=512)
            chroma_mean = np.mean(chroma, axis=1)
            chroma_std = float(np.std(chroma_mean))
            features.extend([float(np.mean(chroma_mean)), chroma_std])
            metrics['chroma_std'] = chroma_std
            
            # Tempo
            tempo, _ = librosa.beat.beat_track(y=audio_data, sr=sr, hop_length=512)
            features.append(float(tempo))
            
            # ZCR and RMS
            zcr = librosa.feature.zero_crossing_rate(audio_data, hop_length=512)[0]
            rms = librosa.feature.rms(y=audio_data, hop_length=512)[0]
            features.extend([
                float(np.mean(zcr)), float(np.std(zcr)),
                float(np.mean(rms)), float(np.std(rms))
            ])
            metrics['zcr_mean'] = float(np.mean(zcr))
            metrics['rms_mean'] = float(np.mean(rms))
            
        except Exception as e:
            print(f"Error in librosa feature extraction: {e}")
            features = [0.0] * 16
            metrics.update({
                'spectral_centroid_mean': 0.0,
                'spectral_bandwidth_mean': 0.0,
                'zcr_mean': 0.0,
                'rms_mean': 0.0,
                'chroma_std': 0.0
            })
        
        return features, metrics
    
    #  base functions for dataset
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return torch.FloatTensor(self.data[index]), int(self.labels[index])

# class purpose: emotion classifier with attention mechanism
class EmotionClassifier(nn.Module):   
    def __init__(self, feature_dim: int = 165, num_classes: int = 6):
        super(EmotionClassifier, self).__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        
        # Feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(128, num_heads=8, batch_first=True)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        features = self.feature_extractor(x)
        attended_features, _ = self.attention(features.unsqueeze(1), features.unsqueeze(1), features.unsqueeze(1))
        attended_features = attended_features.squeeze(1)
        output = self.classifier(attended_features)
        return output

# class purpose: adversarial augmentation GAN
class AdversarialAugmentation(nn.Module):
    def __init__(self, feature_dim: int, num_classes: int = 6):
        super(AdversarialAugmentation, self).__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        
        # Generator
        self.generator = nn.Sequential(
            nn.Linear(feature_dim + num_classes, 512),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256, feature_dim),
            nn.Tanh()
        )
        
        # Discriminator
        self.discriminator = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    # generate augmented features
    def generate_augmented(self, features, labels):
        labels_onehot = torch.zeros(features.size(0), self.num_classes).to(features.device)
        labels_onehot.scatter_(1, labels.unsqueeze(1), 1)
        
        input_features = torch.cat([features, labels_onehot], dim=1)
        augmented = self.generator(input_features)
        return features + 0.1 * augmented
    
    # discriminate features
    def discriminate(self, features):
        return self.discriminator(features)

# class purpose: main GANMER system
class GANMER:    
    def __init__(self, feature_dim: int = 165, num_classes: int = 6, class_weights: Optional[torch.Tensor] = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        
        # Initialize models
        self.augmentation_gan = AdversarialAugmentation(feature_dim, num_classes).to(self.device)
        self.classifier = EmotionClassifier(feature_dim, num_classes).to(self.device)
        
        # Optimizers
        self.g_optimizer = optim.Adam(self.augmentation_gan.generator.parameters(), lr=0.0001, betas=(0.5, 0.999))
        self.d_optimizer = optim.Adam(self.augmentation_gan.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.c_optimizer = optim.Adam(self.classifier.parameters(), lr=0.0003, weight_decay=1e-4)
        
        # Loss functions with class weights for imbalance
        self.adversarial_loss = nn.BCELoss()
        if class_weights is not None:
            self.classification_loss = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
        else:
            self.classification_loss = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        # Focal loss for handling class imbalance
        self.focal_loss = self._focal_loss
        
        # Emotion mapping
        self.emotion_mapping = {
            0: "Happy", 1: "Excited", 2: "Tense", 
            3: "Fearful", 4: "Sad", 5: "Hopeful"
        }
        
        # Training histories
        self.gan_history = {'g_loss': [], 'd_loss': []}
        self.classifier_history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        
        # For recommendations
        self.training_data: Optional[np.ndarray] = None
        self.training_labels: Optional[np.ndarray] = None
        
        # Class weights for focal loss
        self.class_weights = class_weights
    
    # focal loss function
    def _focal_loss(self, inputs, targets, alpha=1, gamma=2):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        if self.class_weights is not None:
            alpha_factor = self.class_weights.to(inputs.device)[targets]
        else:
            alpha_factor = alpha
        focal_loss = alpha_factor * (1-pt)**gamma * ce_loss
        return focal_loss.mean()
    
    # train gan function
    def train_gan(self, dataloader: DataLoader, epochs: int = 30):
        print("Training adversarial augmentation GAN...")
        
        for epoch in range(epochs):
            g_losses = []
            d_losses = []
            
            for features, labels in dataloader:
                features = features.to(self.device)
                labels = labels.to(self.device)
                batch_size = features.size(0)
                
                valid = torch.ones(batch_size, 1).to(self.device)
                fake = torch.zeros(batch_size, 1).to(self.device)
                
                # Train Discriminator
                self.d_optimizer.zero_grad()
                
                real_validity = self.augmentation_gan.discriminate(features)
                d_real_loss = self.adversarial_loss(real_validity, valid)
                
                augmented_features = self.augmentation_gan.generate_augmented(features, labels)
                fake_validity = self.augmentation_gan.discriminate(augmented_features.detach())
                d_fake_loss = self.adversarial_loss(fake_validity, fake)
                
                d_loss = (d_real_loss + d_fake_loss) / 2
                d_loss.backward()
                self.d_optimizer.step()
                
                # Train Generator
                self.g_optimizer.zero_grad()
                
                augmented_features = self.augmentation_gan.generate_augmented(features, labels)
                fake_validity = self.augmentation_gan.discriminate(augmented_features)
                g_loss = self.adversarial_loss(fake_validity, valid)
                
                g_loss.backward()
                self.g_optimizer.step()
                
                g_losses.append(g_loss.item())
                d_losses.append(d_loss.item())
            
            epoch_g = float(np.mean(g_losses)) if len(g_losses) else 0.0
            epoch_d = float(np.mean(d_losses)) if len(d_losses) else 0.0
            self.gan_history['g_loss'].append(epoch_g)
            self.gan_history['d_loss'].append(epoch_d)
            
            if epoch % 10 == 0:
                print(f"Epoch [{epoch}/{epochs}] - G Loss: {epoch_g:.4f}, D Loss: {epoch_d:.4f}")
    
    # train classifier function
    def train_classifier(self, train_loader: DataLoader, val_loader: DataLoader = None, epochs: int = 80):
        print("Training emotion classifier with adversarial augmentation...")
        
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(self.c_optimizer, T_0=20, T_mult=2, eta_min=1e-6)
        best_accuracy = 0
        patience_counter = 0
        patience = 25
        
        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            total = 0
            
            self.classifier.train()
            
            for features, labels in train_loader:
                features = features.to(self.device)
                labels = labels.to(self.device)
                
                # Adversarial augmentation -- balanced
                augmentation_prob = 0.4
                for i, label in enumerate(labels):
                    # Moderate augmentation probability for minority classes (Happy, Excited)
                    if label.item() in [0, 1]:  # Happy, Excited
                        augmentation_prob = 0.6
                    elif label.item() in [4, 5]:  # Sad, Hopeful (majority classes)
                        augmentation_prob = 0.3
                
                if np.random.random() < augmentation_prob:
                    with torch.no_grad():
                        augmented_features = self.augmentation_gan.generate_augmented(features, labels)
                        features = 0.7 * features + 0.3 * augmented_features
                
                # MixUp augmentation -- balanced approach
                mixup_prob = 0.2
                for i, label in enumerate(labels):
                    if label.item() in [0, 1]:  # Happy, Excited
                        mixup_prob = 0.4
                    elif label.item() in [4, 5]:  # Sad, Hopeful
                        mixup_prob = 0.15
                
                if np.random.random() < mixup_prob and features.size(0) > 1:
                    lam = np.random.beta(0.2, 0.2)
                    index = torch.randperm(features.size(0)).to(self.device)
                    mixed_x = lam * features + (1 - lam) * features[index, :]
                    mixed_labels = lam * labels.float() + (1 - lam) * labels[index].float()
                else:
                    mixed_x = features
                    mixed_labels = labels.float()
                
                self.c_optimizer.zero_grad()
                
                outputs = self.classifier(mixed_x)
                # Use focal loss for better handling of class imbalance
                loss = self.focal_loss(outputs, mixed_labels.long(), gamma=1.5)
                
                loss.backward()
                self.c_optimizer.step()
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            train_accuracy = 100 * correct / total if total else 0.0
            train_loss = total_loss / len(train_loader) if len(train_loader) else 0.0
            self.classifier_history['train_loss'].append(float(train_loss))
            self.classifier_history['train_acc'].append(float(train_accuracy))
            
            # Validation
            if val_loader is not None:
                self.classifier.eval()
                val_loss = 0
                val_correct = 0
                val_total = 0
                
                with torch.no_grad():
                    for features, labels in val_loader:
                        features = features.to(self.device)
                        labels = labels.to(self.device)
                        
                        outputs = self.classifier(features)
                        loss = self.classification_loss(outputs, labels)
                        
                        val_loss += loss.item()
                        _, predicted = torch.max(outputs.data, 1)
                        val_total += labels.size(0)
                        val_correct += (predicted == labels).sum().item()
                
                val_accuracy = 100 * val_correct / val_total if val_total else 0.0
                val_loss = val_loss / len(val_loader) if len(val_loader) else 0.0
                self.classifier_history['val_loss'].append(float(val_loss))
                self.classifier_history['val_acc'].append(float(val_accuracy))
                
                if val_accuracy > best_accuracy:
                    best_accuracy = val_accuracy
                    patience_counter = 0
                    torch.save(self.classifier.state_dict(), 'best_classifier.pth')
                else:
                    patience_counter += 1
                
                if epoch % 10 == 0:
                    print(f"Epoch [{epoch}/{epochs}] - Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%, Best: {best_accuracy:.2f}%")
                
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch} with best accuracy: {best_accuracy:.2f}%")
                    self.classifier.load_state_dict(torch.load('best_classifier.pth'))
                    break
            else:
                if epoch % 10 == 0:
                    print(f"Epoch [{epoch}/{epochs}] - Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%")
            
            scheduler.step()
    
    #  classify emotion function
    def classify_emotion(self, features: np.ndarray) -> Tuple[int, str]:
        self.classifier.eval()
        
        with torch.no_grad():
            features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
            outputs = self.classifier(features_tensor)
            _, predicted = torch.max(outputs, 1)
            emotion_id = predicted.item()
            emotion_name = self.emotion_mapping[emotion_id]
            
        return emotion_id, emotion_name
    
    # duplicate: extract features from midi file
    def extract_features_from_midi(self, midi_path: str) -> Optional[np.ndarray]:
        try:
            midi_data = pretty_midi.PrettyMIDI(midi_path)
            features, _ = self._extract_midi_features(midi_data)
            return features
        except Exception as e:
            print(f"Error extracting features from {midi_path}: {e}")
            return None
    
    #  dup: get MIDI features for inference
    def _extract_midi_features(self, midi_data: pretty_midi.PrettyMIDI) -> Tuple[np.ndarray, Dict]:
        features = []
        
        try:
            # Basic MIDI features
            all_notes = []
            for instrument in midi_data.instruments:
                if not instrument.is_drum:
                    all_notes.extend(instrument.notes)
            
            if all_notes:
                pitches = [note.pitch for note in all_notes]
                velocities = [note.velocity for note in all_notes]
                durations = [note.end - note.start for note in all_notes]
                
                features.extend([
                    np.mean(pitches), np.std(pitches), np.min(pitches), np.max(pitches),
                    np.mean(velocities), np.std(velocities), np.mean(durations), np.std(durations)
                ])
            else:
                features.extend([0] * 8)
            
            # Tempo and note density
            tempo = midi_data.estimate_tempo()
            features.append(tempo)
            
            total_duration = midi_data.get_end_time()
            note_density = len(all_notes) / total_duration if total_duration > 0 else 0
            features.append(note_density)
            
            # Librosa features
            try:
                audio_data = midi_data.synthesize(fs=22050)
                librosa_features, _ = self._extract_librosa_features(audio_data, sr=22050)
                features.extend(librosa_features)
            except:
                features.extend([0.0] * 16)
            
            # Pad or truncate to fixed length
            features = np.array(features)
            if len(features) < self.feature_dim:
                features = np.pad(features, (0, self.feature_dim - len(features)))
            else:
                features = features[:self.feature_dim]
            
            return features, {}
            
        except Exception as e:
            print(f"Error extracting MIDI features: {e}")
            return np.zeros(self.feature_dim), {}
    
    # dup: extract librosa features from audio data
    def _extract_librosa_features(self, audio_data: np.ndarray, sr: int = 22050) -> Tuple[List[float], Dict[str, float]]:
        features: List[float] = []
        metrics: Dict[str, float] = {}
        
        try:
            if len(audio_data.shape) > 1:
                audio_data = audio_data.flatten()
            if len(audio_data) > sr * 10:
                audio_data = audio_data[:sr * 10]
            
            # Essential spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=sr, hop_length=512)[0]
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio_data, sr=sr, hop_length=512)[0]
            features.extend([
                float(np.mean(spectral_centroids)), float(np.std(spectral_centroids)),
                float(np.mean(spectral_bandwidth)), float(np.std(spectral_bandwidth))
            ])
            metrics['spectral_centroid_mean'] = float(np.mean(spectral_centroids))
            metrics['spectral_bandwidth_mean'] = float(np.mean(spectral_bandwidth))
            
            # MFCC (first 4 coefficients)
            mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=4, hop_length=512)
            for i in range(4):
                features.extend([float(np.mean(mfccs[i])), float(np.std(mfccs[i]))])
            
            # Chroma features
            chroma = librosa.feature.chroma_stft(y=audio_data, sr=sr, hop_length=512)
            chroma_mean = np.mean(chroma, axis=1)
            chroma_std = float(np.std(chroma_mean))
            features.extend([float(np.mean(chroma_mean)), chroma_std])
            metrics['chroma_std'] = chroma_std
            
            # Tempo
            tempo, _ = librosa.beat.beat_track(y=audio_data, sr=sr, hop_length=512)
            features.append(float(tempo))
            
            # ZCR and RMS
            zcr = librosa.feature.zero_crossing_rate(audio_data, hop_length=512)[0]
            rms = librosa.feature.rms(y=audio_data, hop_length=512)[0]
            features.extend([
                float(np.mean(zcr)), float(np.std(zcr)),
                float(np.mean(rms)), float(np.std(rms))
            ])
            metrics['zcr_mean'] = float(np.mean(zcr))
            metrics['rms_mean'] = float(np.mean(rms))
            
        except Exception as e:
            print(f"Error in librosa feature extraction: {e}")
            features = [0.0] * 16
            metrics.update({
                'spectral_centroid_mean': 0.0,
                'spectral_bandwidth_mean': 0.0,
                'zcr_mean': 0.0,
                'rms_mean': 0.0,
                'chroma_std': 0.0
            })
        
        return features, metrics
    
    # generate synthetic data function
    def generate_synthetic_data(self, num_samples: int, emotion_label: int) -> np.ndarray:
        self.augmentation_gan.eval()
        
        with torch.no_grad():
            # Generate random features as base
            base_features = torch.randn(num_samples, self.feature_dim).to(self.device)
            labels = torch.full((num_samples,), emotion_label, dtype=torch.long).to(self.device)
            
            # Apply adversarial augmentation
            synthetic_features = self.augmentation_gan.generate_augmented(base_features, labels)
            return synthetic_features.cpu().numpy()
    
    # recommender 
    def recommend_similar_tracks(self, input_midi_path: str, dataset_dir: str, num_recommendations: int = 5, dataset: 'EMOPIADataset' = None) -> List[Dict]:        
        # Extract features from input track
        input_features = self.extract_features_from_midi(input_midi_path)
        if input_features is None:
            print("Could not extract features from input track")
            return []
        
        # Classify emotion of input track
        input_emotion_id, input_emotion_name = self.classify_emotion(input_features)
        print(f"Input track emotion: {input_emotion_name}")
        
        # Load all dataset features and labels using processed EMOPIADataset data
        if self.training_data is None:
            if dataset is not None:
                self._load_dataset_features(dataset)
            else:
                print("Error: No EMOPIADataset provided for recommendations")
                return []
        
        # Find tracks with similar emotion
        similar_indices = np.where(self.training_labels == input_emotion_id)[0]
        
        if len(similar_indices) == 0:
            print(f"No tracks found with emotion: {input_emotion_name}")
            return []
        
        # Calculate similarity scores
        similarities = []
        for index in similar_indices:
            similarity = self._calculate_similarity(input_features, self.training_data[index])
            similarities.append((index, similarity))
        
        # Sort by similarity and get top recommendations (excluding the input track itself)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Filter out the input track if it's in the dataset
        input_track_id = os.path.splitext(os.path.basename(input_midi_path))[0]
        filtered_similarities = []
        for index, similarity in similarities:
            # Get the track ID for this index
            if index < len(self.training_labels):
                # Use the dataset's file paths to get track ID
                if hasattr(self, '_dataset_file_paths') and index < len(self._dataset_file_paths):
                    track_id = os.path.splitext(os.path.basename(self._dataset_file_paths[index]))[0]
                else:
                    track_id = f"track_{index}"
                if track_id != input_track_id:  # Exclude the input track
                    filtered_similarities.append((index, similarity))
        
        # Get top recommendations (ensure we have enough)
        if len(filtered_similarities) < num_recommendations:
            print(f"Warning: Only found {len(filtered_similarities)} different tracks with emotion {input_emotion_name}")
            top_indices = [index for index, _ in filtered_similarities]
        else:
            top_indices = [index for index, _ in filtered_similarities[:num_recommendations]]
        
        # Get file paths for recommendations
        recommendations = []
        labels_df = pd.read_csv(os.path.join(dataset_dir, 'label.csv'))
        
        for i, index in enumerate(top_indices):
            if index < len(labels_df):
                track_id = labels_df.iloc[index]['ID']
                midi_file = f"{track_id}.mid"
                midi_path = os.path.join(dataset_dir, 'midis', midi_file)
                
                if os.path.exists(midi_path):
                    # Get similarity from filtered_similarities
                    similarity = filtered_similarities[i][1] if i < len(filtered_similarities) else 0.0
                    recommendations.append({
                        'file_path': midi_path,
                        'track_id': track_id,
                        'emotion': self.emotion_mapping[self.training_labels[index]],
                        'similarity': similarity
                    })
        
        return recommendations
    
    # Load all dataset features for recommendations using already processed EMOPIADataset data 
    def _load_dataset_features(self, dataset: 'EMOPIADataset'):
        print("Loading dataset features for recommendations...")
        print("Using pre-processed features from EMOPIADataset...")
        
        self.training_data = np.array(dataset.data)
        self.training_labels = np.array(dataset.labels)
        
        # Store file paths for track ID retrieval
        self._dataset_file_paths = dataset.file_paths if hasattr(dataset, 'file_paths') else []
        
        print(f"Loaded {len(self.training_data)} tracks for recommendations from EMOPIADataset")
    
    # Calculate cosine similarity between two feature vectors
    def _calculate_similarity(self, features1: np.ndarray, features2: np.ndarray) -> float:
        dot_product = np.dot(features1, features2)
        norm1 = np.linalg.norm(features1)
        norm2 = np.linalg.norm(features2)
        
        if norm1 == 0 or norm2 == 0:
            return 0
        
        return dot_product / (norm1 * norm2)

# run exploratory data analysis function
def run_eda(dataset: EMOPIADataset, gan_mer: GANMER, true_labels: List[int], pred_labels: List[int], save_dir: str = "eda"):
    os.makedirs(save_dir, exist_ok=True)
    label_names = [gan_mer.emotion_mapping[i] for i in range(6)]

    # Class balance
    plt.figure(figsize=(8, 6))
    sns.countplot(x=[label_names[l] for l in dataset.labels])
    plt.title('Class Balance (6 emotions)')
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'class_balance.png'))
    plt.close()

    # Confusion matrix
    if len(true_labels) and len(true_labels) == len(pred_labels):
        cm = confusion_matrix(true_labels, pred_labels, labels=list(range(6)))
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_names, yticklabels=label_names)
        plt.ylabel('True')
        plt.xlabel('Predicted')
        plt.title('Confusion Matrix (Test)')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))
        plt.close()

    # Training curves
    g_loss = gan_mer.gan_history.get('g_loss', [])
    d_loss = gan_mer.gan_history.get('d_loss', [])
    if len(g_loss) and len(d_loss):
        plt.figure(figsize=(8, 6))
        plt.plot(g_loss, label='G loss')
        plt.plot(d_loss, label='D loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('GAN Training Losses')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'gan_losses.png'))
        plt.close()

    # Classifier curves
    tr_loss = gan_mer.classifier_history.get('train_loss', [])
    tr_acc = gan_mer.classifier_history.get('train_acc', [])
    va_loss = gan_mer.classifier_history.get('val_loss', [])
    va_acc = gan_mer.classifier_history.get('val_acc', [])
    
    if len(tr_loss):
        plt.figure(figsize=(8, 6))
        plt.plot(tr_loss, label='Train')
        if len(va_loss) and not all(np.isnan(va_loss)):
            plt.plot(va_loss, label='Val')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Classifier Loss')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'classifier_loss.png'))
        plt.close()
    
    if len(tr_acc):
        plt.figure(figsize=(8, 6))
        plt.plot(tr_acc, label='Train')
        if len(va_acc) and not all(np.isnan(va_acc)):
            plt.plot(va_acc, label='Val')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.title('Classifier Accuracy')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'classifier_accuracy.png'))
        plt.close()

# EDA for processed dataset
def run_processed_data_eda(dataset: EMOPIADataset, save_dir: str = "eda"):
    print("\n=== PROCESSED DATA EDA ANALYSIS ===")
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Dataset Overview
    print(f"\n1. Dataset Overview:")
    print(f"   Total samples: {len(dataset.data)}")
    print(f"   Feature dimension: {len(dataset.data[0]) if dataset.data else 0}")
    print(f"   Number of emotions: {len(set(dataset.labels))}")
    print(f"   File paths tracked: {len(dataset.file_paths)}")
    
    # 2. Emotion Distribution Analysis
    print(f"\n2. Emotion Distribution:")
    emotion_counts = {}
    for label in dataset.labels:
        emotion_name = dataset.emotion_mapping[label]
        emotion_counts[emotion_name] = emotion_counts.get(emotion_name, 0) + 1
    
    for emotion, count in emotion_counts.items():
        percentage = (count / len(dataset.labels)) * 100
        print(f"   {emotion}: {count} samples ({percentage:.1f}%)")
    
    # 3. Feature Statistics
    print(f"\n3. Feature Statistics:")
    if dataset.data:
        data_array = np.array(dataset.data)
        print(f"   Feature matrix shape: {data_array.shape}")
        print(f"   Mean feature value: {np.mean(data_array):.4f}")
        print(f"   Std feature value: {np.std(data_array):.4f}")
        print(f"   Min feature value: {np.min(data_array):.4f}")
        print(f"   Max feature value: {np.max(data_array):.4f}")
        
        # Feature variance analysis
        feature_vars = np.var(data_array, axis=0)
        print(f"   Features with zero variance: {np.sum(feature_vars == 0)}")
        print(f"   Most variable feature index: {np.argmax(feature_vars)} (var: {np.max(feature_vars):.4f})")
        print(f"   Least variable feature index: {np.argmin(feature_vars)} (var: {np.min(feature_vars):.4f})")
    
    # 4. Quadrant Analysis
    print(f"\n4. Quadrant Analysis:")
    quadrant_counts = {}
    for i, file_path in enumerate(dataset.file_paths):
        if i < len(dataset.labels):
            filename = os.path.basename(file_path)
            if filename.startswith('Q'):
                quadrant = filename[1]  # Extract Q1, Q2, Q3, Q4
                quadrant_counts[quadrant] = quadrant_counts.get(quadrant, 0) + 1
    
    for quadrant, count in sorted(quadrant_counts.items()):
        print(f"   Q{quadrant}: {count} files")
    
    # 5. Arousal and Tension Analysis
    print(f"\n5. Arousal and Tension Analysis:")
    if dataset.sample_metrics:
        arousal_values = [m.get('arousal', 0) for m in dataset.sample_metrics if 'arousal' in m]
        tension_values = [m.get('tension', 0) for m in dataset.sample_metrics if 'tension' in m]
        
        if arousal_values:
            print(f"   Arousal - Mean: {np.mean(arousal_values):.4f}, Std: {np.std(arousal_values):.4f}")
        if tension_values:
            print(f"   Tension - Mean: {np.mean(tension_values):.4f}, Std: {np.std(tension_values):.4f}")
    
    # 6. Create Visualizations
    print(f"\n6. Creating visualizations...")
    
    # Emotion distribution plot
    plt.figure(figsize=(10, 6))
    emotion_names = list(emotion_counts.keys())
    emotion_counts_list = list(emotion_counts.values())
    colors = plt.cm.Set3(np.linspace(0, 1, len(emotion_names)))
    
    bars = plt.bar(emotion_names, emotion_counts_list, color=colors)
    plt.title('Processed Dataset - Emotion Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Emotion Categories', fontsize=12)
    plt.ylabel('Number of Samples', fontsize=12)
    plt.xticks(rotation=45)
    
    # Add value labels on bars
    for bar, count in zip(bars, emotion_counts_list):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                str(count), ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'processed_emotion_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Feature correlation heatmap (sample of features)
    if dataset.data and len(dataset.data) > 0:
        data_array = np.array(dataset.data)
        if data_array.shape[1] > 20:  # If we have many features, sample them
            feature_indices = np.random.choice(data_array.shape[1], 20, replace=False)
            sample_data = data_array[:, feature_indices]
        else:
            sample_data = data_array
        
        plt.figure(figsize=(12, 10))
        correlation_matrix = np.corrcoef(sample_data.T)
        sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', center=0,
                   xticklabels=False, yticklabels=False)
        plt.title('Processed Dataset - Feature Correlation Matrix (Sample)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'processed_feature_correlation.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # Quadrant distribution
    if quadrant_counts:
        plt.figure(figsize=(8, 6))
        quadrants = sorted(quadrant_counts.keys())
        counts = [quadrant_counts[q] for q in quadrants]
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        
        bars = plt.bar([f'Q{q}' for q in quadrants], counts, color=colors)
        plt.title('Processed Dataset - Quadrant Distribution', fontsize=14, fontweight='bold')
        plt.xlabel('Quadrants', fontsize=12)
        plt.ylabel('Number of Files', fontsize=12)
        
        # Add value labels
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    str(count), ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'processed_quadrant_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # Feature distribution by emotion
    if dataset.data and len(dataset.data) > 0:
        data_array = np.array(dataset.data)
        labels_array = np.array(dataset.labels)
        
        # Select a few key features for visualization
        feature_indices = [0, 10, 20, 30, 40] if data_array.shape[1] > 40 else [0, 5, 10, 15, 20]
        feature_indices = [i for i in feature_indices if i < data_array.shape[1]]
        
        if len(feature_indices) > 0:
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes = axes.flatten()
            
            for i, feat_index in enumerate(feature_indices[:6]):
                if i < len(axes):
                    for emotion_id in range(6):
                        emotion_name = dataset.emotion_mapping[emotion_id]
                        emotion_data = data_array[labels_array == emotion_id, feat_index]
                        if len(emotion_data) > 0:
                            axes[i].hist(emotion_data, alpha=0.6, label=emotion_name, bins=20)
                    
                    axes[i].set_title(f'Feature {feat_index} Distribution by Emotion')
                    axes[i].set_xlabel(f'Feature {feat_index} Value')
                    axes[i].set_ylabel('Frequency')
                    axes[i].legend()
            
            # Hide unused subplots
            for i in range(len(feature_indices), len(axes)):
                axes[i].set_visible(False)
            
            plt.suptitle('Processed Dataset - Feature Distributions by Emotion', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'processed_feature_distributions.png'), dpi=300, bbox_inches='tight')
            plt.close()
    
    # 7. Data Quality Assessment
    print(f"\n7. Data Quality Assessment:")
    if dataset.data:
        data_array = np.array(dataset.data)
        
        # Check for missing values
        nan_count = np.isnan(data_array).sum()
        inf_count = np.isinf(data_array).sum()
        print(f"   Missing values (NaN): {nan_count}")
        print(f"   Infinite values: {inf_count}")
        
        # Check for constant features
        constant_features = np.sum(np.var(data_array, axis=0) == 0)
        print(f"   Constant features: {constant_features}")
        
        # Check for outliers (using IQR method)
        Q1 = np.percentile(data_array, 25, axis=0)
        Q3 = np.percentile(data_array, 75, axis=0)
        IQR = Q3 - Q1
        outlier_threshold = 1.5 * IQR
        outliers = np.sum((data_array < Q1 - outlier_threshold) | (data_array > Q3 + outlier_threshold))
        print(f"   Potential outliers: {outliers} ({outliers/data_array.size*100:.2f}%)")
    
    # 8. Processing Validation
    print(f"\n8. Processing Validation:")
    print(f"   Data-Labels alignment: {len(dataset.data) == len(dataset.labels)}")
    print(f"   Data-FilePaths alignment: {len(dataset.data) == len(dataset.file_paths)}")
    print(f"   Labels-FilePaths alignment: {len(dataset.labels) == len(dataset.file_paths)}")
    
    # Check emotion mapping consistency
    unique_labels = set(dataset.labels)
    expected_labels = set(range(6))
    print(f"   Expected emotions (0-5): {expected_labels}")
    print(f"   Found emotions: {sorted(unique_labels)}")
    print(f"   Emotion mapping complete: {unique_labels == expected_labels}")
    
    print(f"\n EDA saved to: {save_dir}")
    print(f"   - processed_emotion_distribution.png")
    print(f"   - processed_feature_correlation.png") 
    print(f"   - processed_quadrant_distribution.png")
    print(f"   - processed_feature_distributions.png")

def main():
    # Configuration
    DATASET_DIR = "EMOPIA_1.0"
    MIDI_DIR = os.path.join(DATASET_DIR, "midis")
    LABEL_FILE = os.path.join(DATASET_DIR, "label.csv")
    SAVE_DIR = "models"
    
    # RECOMMENDATION TEST CONFIGURATION
    # Change this to test different tracks for recommendations
    TEST_TRACK_FOR_RECOMMENDATIONS = "Q1__kJtgm1OUNA_0.mid"  # Your chosen track
    
    # Hyperparameters
    FEATURE_DIM = 165
    NUM_CLASSES = 6
    BATCH_SIZE = 32
    GAN_EPOCHS = 30
    CLASSIFIER_EPOCHS = 80
    
    print("=== GAN-based Music Emotion Recognition System ===")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    
    # Load dataset
    print("\n1. Loading EMOPIA dataset...")
    dataset = EMOPIADataset(MIDI_DIR, LABEL_FILE, sequence_length=FEATURE_DIM)
    print(f"Loaded {len(dataset)} samples with {NUM_CLASSES} emotion classes")
    
    # Calculate emotion distribution and synthetic samples needed
    print("\n1.1. Analyzing emotion distribution...")
    emotion_counts = {}
    for label in dataset.labels:
        emotion_name = dataset.emotion_mapping[label]
        emotion_counts[emotion_name] = emotion_counts.get(emotion_name, 0) + 1
    
    print("Original dataset distribution:")
    for emotion, count in emotion_counts.items():
        percentage = (count / len(dataset.labels)) * 100
        print(f"  {emotion}: {count} samples ({percentage:.1f}%)")
    
    # Find the maximum count to determine target for balanced dataset
    max_count = max(emotion_counts.values())
    print(f"\nMaximum samples in any category: {max_count}")
    print(f"Target samples per category for balanced dataset: {max_count}")
    
    # Calculate how many synthetic samples needed for each emotion
    synthetic_samples_needed = {}
    for emotion_id in range(NUM_CLASSES):
        emotion_name = dataset.emotion_mapping[emotion_id]
        current_count = emotion_counts[emotion_name]
        needed = max_count - current_count
        synthetic_samples_needed[emotion_id] = needed
        print(f"  {emotion_name}: {current_count} existing, need {needed} synthetic samples")
    
    # Generate balanced synthetic data with improved strategy
    print("\n1.2. Generating balanced synthetic data...")
    all_synthetic_features = []
    all_synthetic_labels = []
    
    # First, train a proper GAN on the original data for better synthetic generation
    print("Training GAN for synthetic data generation...")
    temp_gan = GANMER(FEATURE_DIM, NUM_CLASSES)
    
    # Create temporary data loaders for GAN training
    temp_dataset = [(torch.FloatTensor(features), label) for features, label in zip(dataset.data, dataset.labels)]
    temp_loader = DataLoader(temp_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Train GAN for better synthetic data quality
    temp_gan.train_gan(temp_loader, epochs=80)
    
    for emotion_id in range(NUM_CLASSES):
        emotion_name = dataset.emotion_mapping[emotion_id]
        num_needed = synthetic_samples_needed[emotion_id]
        
        if num_needed > 0:
            print(f"Generating {num_needed} synthetic samples for {emotion_name}...")
            synthetic_data = temp_gan.generate_synthetic_data(num_needed, emotion_id)
            all_synthetic_features.extend(synthetic_data)
            all_synthetic_labels.extend([emotion_id] * num_needed)
            print(f"  Generated {len(synthetic_data)} samples for {emotion_name}")
        else:
            print(f"  {emotion_name} already has enough samples ({emotion_counts[emotion_name]})")
    
    print(f"\nTotal synthetic samples generated: {len(all_synthetic_features)}")
    
    # Combine original and synthetic data for balanced dataset with quality control
    print("\n1.3. Creating balanced dataset...")
    
    # Quality control: Filter out synthetic samples that are too different from original
    if all_synthetic_features:
        print("Applying quality control to synthetic data...")
        filtered_synthetic_features = []
        filtered_synthetic_labels = []
        
        for emotion_id in range(NUM_CLASSES):
            emotion_name = dataset.emotion_mapping[emotion_id]
            synthetic_mask = np.array(all_synthetic_labels) == emotion_id
            synthetic_features_emotion = np.array(all_synthetic_features)[synthetic_mask]
            
            if len(synthetic_features_emotion) > 0:
                # Get original features for this emotion
                original_mask = np.array(dataset.labels) == emotion_id
                original_features_emotion = np.array(dataset.data)[original_mask]
                
                # Calculate similarity and filter
                similarities = []
                for synth_feat in synthetic_features_emotion:
                    max_sim = 0
                    for orig_feat in original_features_emotion:
                        sim = np.corrcoef(synth_feat, orig_feat)[0, 1]
                        if not np.isnan(sim):
                            max_sim = max(max_sim, sim)
                    similarities.append(max_sim)
                
                # Keep only synthetic samples with reasonable similarity
                threshold = 0.1  # Lower threshold for more diversity
                good_indices = [i for i, sim in enumerate(similarities) if sim > threshold]
                
                if len(good_indices) > 0:
                    filtered_synthetic_features.extend(synthetic_features_emotion[good_indices])
                    filtered_synthetic_labels.extend([emotion_id] * len(good_indices))
                    print(f"  {emotion_name}: Kept {len(good_indices)}/{len(synthetic_features_emotion)} synthetic samples")
                else:
                    # If no good synthetic samples, use original data
                    filtered_synthetic_features.extend(original_features_emotion)
                    filtered_synthetic_labels.extend([emotion_id] * len(original_features_emotion))
                    print(f"  {emotion_name}: Using original data (no good synthetic samples)")
        
        balanced_data = dataset.data + filtered_synthetic_features
        balanced_labels = dataset.labels + filtered_synthetic_labels
    else:
        balanced_data = dataset.data
        balanced_labels = dataset.labels
    
    print(f"Balanced dataset: {len(balanced_data)} samples")
    print("Balanced dataset distribution:")
    balanced_emotion_counts = {}
    for label in balanced_labels:
        emotion_name = dataset.emotion_mapping[label]
        balanced_emotion_counts[emotion_name] = balanced_emotion_counts.get(emotion_name, 0) + 1
    
    for emotion, count in balanced_emotion_counts.items():
        percentage = (count / len(balanced_labels)) * 100
        print(f"  {emotion}: {count} samples ({percentage:.1f}%)")
    
    # Split BALANCED dataset
    print("\n1.4. Splitting balanced dataset...")
    train_data, temp_data, train_labels, temp_labels = train_test_split(
        balanced_data, balanced_labels, test_size=0.3, random_state=42, stratify=balanced_labels
    )
    val_data, test_data, val_labels, test_labels = train_test_split(
        temp_data, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
    )
    
    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    print(f"Test samples: {len(test_data)}")
    
    # Create data loaders
    train_dataset = [(torch.FloatTensor(features), label) for features, label in zip(train_data, train_labels)]
    val_dataset = [(torch.FloatTensor(features), label) for features, label in zip(val_data, val_labels)]
    test_dataset = [(torch.FloatTensor(features), label) for features, label in zip(test_data, test_labels)]
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Calculate class weights for handling imbalance with hybrid approach
    print("\n2. Calculating class weights for hybrid approach...")
    from sklearn.utils.class_weight import compute_class_weight
    
    # Use original dataset for class weights to maintain some imbalance awareness
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(dataset.labels),
        y=dataset.labels
    )
    
    print("Class weights (inverse frequency):")
    emotion_names = ["Happy", "Excited", "Tense", "Fearful", "Sad", "Hopeful"]
    for i, (emotion, weight) in enumerate(zip(emotion_names, class_weights)):
        print(f"  {emotion}: {weight:.3f}")
    
    # Normalize class weights to prevent extreme values
    class_weights_normalized = class_weights / class_weights.max()
    print("\nNormalized class weights:")
    for i, (emotion, weight) in enumerate(zip(emotion_names, class_weights_normalized)):
        print(f"  {emotion}: {weight:.3f}")
    
    class_weights_tensor = torch.FloatTensor(class_weights_normalized).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    
    # Initialize GAN MER system with class weights
    print("\n3. Initializing GAN MER system...")
    gan_mer = GANMER(FEATURE_DIM, NUM_CLASSES, class_weights_tensor)
    
    # Train models with improved strategy
    print("\n4. Training models...")
    
    # First train GAN on balanced data
    gan_mer.train_gan(train_loader, epochs=GAN_EPOCHS)
    
    # Train classifier with progressive learning
    print("Training classifier with progressive learning...")
    gan_mer.train_classifier(train_loader, val_loader, epochs=CLASSIFIER_EPOCHS)
    
    # Additional fine-tuning on original data to maintain quality
    print("Fine-tuning on original data...")
    original_train_data = [(torch.FloatTensor(features), label) for features, label in zip(dataset.data, dataset.labels)]
    original_train_loader = DataLoader(original_train_data, batch_size=BATCH_SIZE, shuffle=True)
    
    # Fine-tune for a few epochs on original data
    gan_mer.train_classifier(original_train_loader, val_loader, epochs=20)
    
    # Evaluate classifier
    print("\n5. Evaluating classifier...")
    gan_mer.classifier.eval()
    
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for features, labels in test_loader:
            features = features.to(gan_mer.device)
            labels = labels.to(gan_mer.device)
            
            outputs = gan_mer.classifier(features)
            _, predicted = torch.max(outputs, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = 100 * correct / total
    print(f"CLASSIFIER ACCURACY: {accuracy:.2f}%")
    
    # Print classification report
    target_names = [gan_mer.emotion_mapping[i] for i in range(NUM_CLASSES)]
    print("\nClassification Report:")
    print(classification_report(all_labels, all_predictions, target_names=target_names))
    
    # Show per-class accuracy
    print("\nPer-class Accuracy:")
    
    cm = confusion_matrix(all_labels, all_predictions)
    for i, emotion in enumerate(target_names):
        class_correct = cm[i, i]
        class_total = cm[i, :].sum()
        class_acc = class_correct / class_total if class_total > 0 else 0
        print(f"  {emotion}: {class_acc:.3f} ({class_correct}/{class_total})")
    
    # Run EDA
    print("\n6. Running EDA...")
    run_eda(dataset, gan_mer, all_labels, all_predictions)
    
    # Run additional EDA for processed data
    run_processed_data_eda(dataset, "eda")
    
    # Note: Synthetic data was already generated and used for training
    print("\n8. Using pre-generated balanced dataset for analysis...")
    
    # Create separate dataframes for each emotion with validation
    print("\n9. Creating separate dataframes for each emotion...")
    
    # Create feature column names
    feature_columns = []
    # MIDI features (8)
    feature_columns.extend(['pitch_mean', 'pitch_std', 'pitch_min', 'pitch_max', 
                           'velocity_mean', 'velocity_std', 'duration_mean', 'duration_std'])
    # Tempo and density (2)
    feature_columns.extend(['tempo', 'note_density'])
    # Librosa features (16)
    feature_columns.extend(['spectral_centroid_mean', 'spectral_centroid_std',
                           'spectral_bandwidth_mean', 'spectral_bandwidth_std',
                           'mfcc_0_mean', 'mfcc_0_std', 'mfcc_1_mean', 'mfcc_1_std',
                           'mfcc_2_mean', 'mfcc_2_std', 'mfcc_3_mean', 'mfcc_3_std',
                           'chroma_mean', 'chroma_std', 'librosa_tempo', 'zcr_mean', 'zcr_std',
                           'rms_mean', 'rms_std'])
    # Pad remaining columns
    remaining_features = FEATURE_DIM - len(feature_columns)
    feature_columns.extend([f'feature_{i}' for i in range(len(feature_columns), FEATURE_DIM)])
    
    # Create separate dataframes for each emotion
    emotion_dataframes = {}
    emotion_names = ["Happy", "Excited", "Tense", "Fearful", "Sad", "Hopeful"]
    
    for emotion_id in range(NUM_CLASSES):
        emotion_name = emotion_names[emotion_id]
        
        # Get existing data for this emotion
        existing_mask = np.array(dataset.labels) == emotion_id
        existing_features_emotion = np.array(dataset.data)[existing_mask]
        existing_count = len(existing_features_emotion)
        
        # Get synthetic data for this emotion (from pre-generated balanced dataset)
        synthetic_mask = np.array(all_synthetic_labels) == emotion_id
        synthetic_features_emotion = np.array(all_synthetic_features)[synthetic_mask] if all_synthetic_features else np.array([])
        synthetic_count = len(synthetic_features_emotion)
        
        print(f"\n{emotion_name} emotion:")
        print(f"  Original samples: {existing_count}")
        print(f"  Synthetic samples: {synthetic_count}")
        print(f"  Total samples: {existing_count + synthetic_count}")
        
        # Combine original and synthetic for this emotion
        if existing_count > 0 and synthetic_count > 0:
            emotion_features = np.vstack([existing_features_emotion, synthetic_features_emotion])
            emotion_data_types = ['Original'] * existing_count + ['Synthetic'] * synthetic_count
        elif existing_count > 0:
            emotion_features = existing_features_emotion
            emotion_data_types = ['Original'] * existing_count
        elif synthetic_count > 0:
            emotion_features = synthetic_features_emotion
            emotion_data_types = ['Synthetic'] * synthetic_count
        else:
            print(f"  Warning: No data found for {emotion_name}")
            continue
        
        # Create dataframe for this emotion
        df_data = {}
        for i, col_name in enumerate(feature_columns):
            if i < emotion_features.shape[1]:
                df_data[col_name] = emotion_features[:, i]
            else:
                df_data[col_name] = [0.0] * len(emotion_features)
        
        df_data['emotion_id'] = [emotion_id] * len(emotion_features)
        df_data['emotion_name'] = [emotion_name] * len(emotion_features)
        df_data['data_type'] = emotion_data_types
        
        emotion_df = pd.DataFrame(df_data)
        emotion_dataframes[emotion_name] = emotion_df
        
        # Display sample from this emotion
        print(f"  Sample data:")
        sample_columns = ['emotion_name', 'data_type', 'pitch_mean', 'tempo', 'note_density', 'spectral_centroid_mean']
        print(emotion_df[sample_columns].head(3).to_string(index=True))
    
    # Create overall combined dataframe
    print(f"\nCreating overall combined dataframe...")
    all_emotion_dfs = list(emotion_dataframes.values())
    combined_df = pd.concat(all_emotion_dfs, ignore_index=True) if all_emotion_dfs else pd.DataFrame()
    
    # Display dataframe info
    print(f"Combined dataset shape: {combined_df.shape}")
    print(f"Features: {len(feature_columns)}, Samples: {len(combined_df)}")
    
    # Show final distribution
    print("\nFinal balanced dataset distribution:")
    if not combined_df.empty:
        final_distribution = combined_df.groupby(['emotion_name', 'data_type']).size().unstack(fill_value=0)
        print(final_distribution)
    
    # Show summary statistics
    print(f"\nSummary:")
    print(f"Total samples: {len(combined_df)}")
    print(f"Total original samples: {sum(len(df[df['data_type'] == 'Original']) for df in emotion_dataframes.values())}")
    print(f"Total synthetic samples: {sum(len(df[df['data_type'] == 'Synthetic']) for df in emotion_dataframes.values())}")
    
    # Display first few rows of the combined dataframe
    print(f"\nFirst 10 rows of combined dataframe:")
    if not combined_df.empty:
        display_columns = ['emotion_name', 'data_type', 'pitch_mean', 'tempo', 'note_density', 'spectral_centroid_mean']
        print(combined_df[display_columns].head(10).to_string(index=True))
    
    # Display more detailed dataframe info
    print(f"\nDataframe info:")
    if not combined_df.empty:
        print(f"Columns: {list(combined_df.columns)}")
        print(f"\nData types:")
        print(combined_df.dtypes.value_counts())
    
    # Display individual emotion dataframes
    print(f"\nIndividual emotion dataframes:")
    for emotion_name, emotion_df in emotion_dataframes.items():
        print(f"\n{emotion_name} dataframe:")
        print(f"  Shape: {emotion_df.shape}")
        print(f"  Original samples: {len(emotion_df[emotion_df['data_type'] == 'Original'])}")
        print(f"  Synthetic samples: {len(emotion_df[emotion_df['data_type'] == 'Synthetic'])}")
        if not emotion_df.empty:
            sample_columns = ['emotion_name', 'data_type', 'pitch_mean', 'tempo', 'note_density']
            print(f"  Sample data:")
            print(emotion_df[sample_columns].head(2).to_string(index=True))
    
    # Test recommendation system
    print("\n10. Testing recommendation system...")
    
    # Use the configured test track
    sample_file = os.path.join(MIDI_DIR, TEST_TRACK_FOR_RECOMMENDATIONS)
    
    if os.path.exists(sample_file):
        print(f"Testing with MANUALLY SELECTED track: {TEST_TRACK_FOR_RECOMMENDATIONS}")
        print(f"File path: {sample_file}")
        # Create a balanced dataset object for recommendations
        balanced_dataset = type('BalancedDataset', (), {
            'data': balanced_data,
            'labels': balanced_labels,
            'file_paths': dataset.file_paths + [f"synthetic_{i}" for i in range(len(all_synthetic_features))]
        })()
        
        recommendations = gan_mer.recommend_similar_tracks(sample_file, DATASET_DIR, num_recommendations=5, dataset=balanced_dataset)
        print(f"Found {len(recommendations)} recommendations")
        for i, rec in enumerate(recommendations):
            print(f"  {i+1}. {rec['track_id']} - {rec['emotion']} (similarity: {rec['similarity']:.3f})")
    else:
        print("No Q1 sample files found, skipping recommendation test")
    
    # Show some test predictions with details
    print("\n11. Sample predictions on test set:")
    gan_mer.classifier.eval()
    with torch.no_grad():
        for i, (features, true_label) in enumerate(test_loader):
            if i >= 5:  # Show only first 5 samples
                break
            features = features.to(gan_mer.device)
            outputs = gan_mer.classifier(features)
            _, predicted = torch.max(outputs, 1)
            
            for j in range(features.size(0)):
                true_emotion = gan_mer.emotion_mapping[true_label[j].item()]
                pred_emotion = gan_mer.emotion_mapping[predicted[j].item()]
                confidence = torch.softmax(outputs[j], dim=0)[predicted[j]].item()
                print(f"  Sample {i*BATCH_SIZE + j + 1}: True={true_emotion}, Predicted={pred_emotion} (confidence: {confidence:.3f})")
    
    print("\n=== Training Complete ===")

if __name__ == "__main__":
    main()
