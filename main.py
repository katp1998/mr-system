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
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler
from typing import List, Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

# Enhanced GAN-based Music Emotion Recognition System
# Now includes both MIDI features (13) and librosa audio features (20) for improved emotion recognition
# Total feature dimension: 33 features (13 MIDI + 20 librosa) padded/truncated to FEATURE_DIM

class EMOPIADataset(Dataset):
    """Dataset class for EMOPIA MIDI files with emotion labels"""
    
    def __init__(self, midi_dir: str, label_file: str, sequence_length: int = 128):
        self.midi_dir = midi_dir
        self.sequence_length = sequence_length
        self.data = []
        self.labels = []
        
        # Load labels
        labels_df = pd.read_csv(label_file)
        
        # Emotion mapping based on EMOPIA quadrants -- Add, Hopeful, fearful, excited
        self.emotion_mapping = {
            1: "Happy",      # Q1: High Valence, High Arousal
            2: "Tense",      # Q2: Low Valence, High Arousal  
            3: "Sad",        # Q3: Low Valence, Low Arousal
            4: "Relaxed"     # Q4: High Valence, Low Arousal
        }
        
        # Process each MIDI file
        for _, row in labels_df.iterrows():
            midi_file = f"{row['ID']}.mid"
            midi_path = os.path.join(midi_dir, midi_file)
            
            if os.path.exists(midi_path):
                try:
                    features = self.extract_midi_features(midi_path)
                    if features is not None:
                        self.data.append(features)
                        self.labels.append(row['4Q'] - 1)
                except Exception as e:
                    print(f"Error processing {midi_file}: {e}")
        
        self.data = np.array(self.data)
        self.labels = np.array(self.labels)
        
        scaler = RobustScaler()
        self.data = scaler.fit_transform(self.data)
        
        print(f"Loaded {len(self.data)} samples with {len(np.unique(self.labels))} emotion classes")
    
    def extract_midi_features(self, midi_path: str) -> Optional[np.ndarray]:
        """Extract musical features from MIDI file using both pretty_midi and librosa"""

        # Question: What are the key features that will say "this is a sad song" vs "this is a happy song"?
        try:
            midi_data = pretty_midi.PrettyMIDI(midi_path)
            
            # Extract basic MIDI features
            features = []
            
            # Tempo features
            tempo = midi_data.estimate_tempo()
            tempo_changes = midi_data.get_tempo_changes()[1]
            features.extend([
                tempo, 
                tempo_changes.mean(),
                tempo_changes.std(),
                len(tempo_changes)  # Number of tempo changes
            ])
            
            # Pitch features
            all_notes = []
            for instrument in midi_data.instruments:
                for note in instrument.notes:
                    all_notes.append(note.pitch)
            
            if not all_notes:
                return None
                
            all_notes = np.array(all_notes)
            features.extend([
                all_notes.mean(),  # Average pitch
                all_notes.std(),   # Pitch standard deviation
                all_notes.max() - all_notes.min(),  # Pitch range
                len(np.unique(all_notes)),  # Unique pitches
                np.percentile(all_notes, 25),  # 25th percentile
                np.percentile(all_notes, 75),  # 75th percentile
                np.median(all_notes)  # Median pitch
            ])
            
            # Duration features
            durations = [note.end - note.start for instrument in midi_data.instruments for note in instrument.notes]
            if durations:
                features.extend([
                    np.mean(durations),
                    np.std(durations),
                    np.max(durations),
                    np.min(durations),
                    np.median(durations),
                    np.percentile(durations, 25),
                    np.percentile(durations, 75)
                ])
            else:
                features.extend([0, 0, 0, 0, 0, 0, 0])
            
            # Velocity features
            velocities = [note.velocity for instrument in midi_data.instruments for note in instrument.notes]
            if velocities:
                features.extend([
                    np.mean(velocities),
                    np.std(velocities),
                    np.max(velocities),
                    np.min(velocities),
                    np.median(velocities)
                ])
            else:
                features.extend([0, 0, 0, 0, 0])
            
            # Time signature features
            if midi_data.time_signature_changes:
                features.extend([
                    midi_data.time_signature_changes[0].numerator,
                    midi_data.time_signature_changes[0].denominator,
                    len(midi_data.time_signature_changes)  # Number of time signature changes
                ])
            else:
                features.extend([4, 4, 1])  # Default 4/4
            
            # Advanced features
            # Note density (notes per second)
            total_duration = midi_data.get_end_time()
            note_density = len(all_notes) / total_duration if total_duration > 0 else 0
            features.append(note_density)
            
            # Polyphony (average notes playing simultaneously)
            time_points = np.linspace(0, total_duration, 100)
            polyphony = []
            for t in time_points:
                active_notes = sum(1 for instrument in midi_data.instruments 
                                 for note in instrument.notes 
                                 if note.start <= t <= note.end)
                polyphony.append(active_notes)
            features.extend([np.mean(polyphony), np.std(polyphony)])
            
            # Key signature features
            if midi_data.key_signature_changes:
                features.append(midi_data.key_signature_changes[0].key_number)
            else:
                features.append(0)  # C major
            
            # Convert MIDI to audio for librosa analysis
            try:
                # Synthesize audio from MIDI
                audio_data = midi_data.synthesize(fs=22050)
                
                # Extract librosa features
                librosa_features = self._extract_librosa_features(audio_data, sr=22050)
                features.extend(librosa_features)
                
            except Exception as e:
                print(f"Warning: Could not extract librosa features from {midi_path}: {e}")
                # Add default librosa features if extraction fails
                default_librosa_features = [0.0] * 20  # Default values for librosa features
                features.extend(default_librosa_features)
            
            # Pad or truncate to fixed length
            features = np.array(features)
            if len(features) < self.sequence_length:
                features = np.pad(features, (0, self.sequence_length - len(features)))
            else:
                features = features[:self.sequence_length]
            
            return features
            
        except Exception as e:
            print(f"Error extracting features from {midi_path}: {e}")
            return None
    
    def _extract_librosa_features(self, audio_data: np.ndarray, sr: int = 22050) -> List[float]:
        """Extract advanced audio features using librosa (optimized for speed)"""
        features = []
        
        try:
            # Ensure audio_data is 1D and limit length for faster processing
            if len(audio_data.shape) > 1:
                audio_data = audio_data.flatten()
            
            # Limit audio length to 10 seconds for faster processing
            max_length = sr * 10  # 10 seconds
            if len(audio_data) > max_length:
                audio_data = audio_data[:max_length]
            
            # Spectral features (faster alternatives)
            spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=sr, hop_length=512)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sr, hop_length=512)[0]
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio_data, sr=sr, hop_length=512)[0]
            
            features.extend([
                float(np.mean(spectral_centroids)), float(np.std(spectral_centroids)),
                float(np.mean(spectral_rolloff)), float(np.std(spectral_rolloff)),
                float(np.mean(spectral_bandwidth)), float(np.std(spectral_bandwidth))
            ])
            
            # MFCC features (first 4 coefficients)
            mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13, hop_length=512)
            for i in range(4):  # Use first 4 MFCC coefficients
                features.extend([float(np.mean(mfccs[i])), float(np.std(mfccs[i]))])
            
            # Simplified chroma features (faster)
            try:
                chroma = librosa.feature.chroma_stft(y=audio_data, sr=sr, hop_length=512, n_fft=2048)
                chroma_mean = np.mean(chroma, axis=1)
                features.extend([float(np.mean(chroma_mean)), float(np.std(chroma_mean))])
            except:
                # Fallback if chroma extraction fails
                features.extend([0.5, 0.1])  # Default values
            
            # Rhythm features (simplified)
            try:
                tempo, _ = librosa.beat.beat_track(y=audio_data, sr=sr, hop_length=512)
                features.append(float(tempo))
            except:
                features.append(120.0)  # Default tempo
            
            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(audio_data, hop_length=512)[0]
            features.extend([float(np.mean(zcr)), float(np.std(zcr))])
            
            # Root mean square energy
            rms = librosa.feature.rms(y=audio_data, hop_length=512)[0]
            features.extend([float(np.mean(rms)), float(np.std(rms))])
            
            # Simplified harmonic/percussive (faster)
            try:
                harmonic, percussive = librosa.effects.hpss(audio_data, margin=1.0)
                harmonic_energy = float(np.mean(librosa.feature.rms(y=harmonic, hop_length=512)[0]))
                percussive_energy = float(np.mean(librosa.feature.rms(y=percussive, hop_length=512)[0]))
                features.extend([harmonic_energy, percussive_energy])
            except:
                # Fallback values
                features.extend([0.1, 0.1])
            
        except Exception as e:
            print(f"Error in librosa feature extraction: {e}")
            # Return default values if extraction fails
            features = [0.0] * 20
        
        # Ensure we have exactly 20 features
        if len(features) != 20:
            if len(features) < 20:
                features.extend([0.0] * (20 - len(features)))
            else:
                features = features[:20]
        
        return features
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.data[idx]), self.labels[idx] - 1  # Convert to 0-based indexing

class Generator(nn.Module):
    """Generator network for GAN"""
    
    def __init__(self, latent_dim: int = 100, feature_dim: int = 165, num_classes: int = 4):
        super(Generator, self).__init__()
        
        self.latent_dim = latent_dim
        self.feature_dim = feature_dim
        
        self.model = nn.Sequential(
            # Input: latent_dim + num_classes (conditioning)
            nn.Linear(latent_dim + num_classes, 256),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(256),
            
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(512),
            
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(256),
            
            nn.Linear(256, feature_dim),
            nn.Tanh()  # Output in [-1, 1] range
        )
    
    def forward(self, z, labels):
        # Concatenate noise and labels
        x = torch.cat([z, labels], dim=1)
        return self.model(x)

class Discriminator(nn.Module):
    """Discriminator network for GAN"""
    
    def __init__(self, feature_dim: int = 165, num_classes: int = 4):
        super(Discriminator, self).__init__()
        
        self.feature_dim = feature_dim
        
        # Main discriminator
        self.discriminator = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Auxiliary classifier for emotion labels
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.Linear(64, num_classes),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        validity = self.discriminator(x)
        emotion_labels = self.classifier(x)
        return validity, emotion_labels

class EmotionClassifier(nn.Module):
    """Advanced emotion classifier with attention mechanism"""
    
    def __init__(self, feature_dim: int = 165, num_classes: int = 4):
        super(EmotionClassifier, self).__init__()
        
        # Feature attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, feature_dim),
            nn.Sigmoid()
        )
        
        # Main classifier
        self.feature_extractor = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.4),
            
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.4),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3)
        )
        
        # Multiple heads for different aspects
        self.head1 = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
        
        self.head2 = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
        
        # Final combination
        self.combiner = nn.Linear(num_classes * 2, num_classes)
    
    def forward(self, x):
        # Apply attention
        attention_weights = self.attention(x)
        attended_features = x * attention_weights
        
        # Extract features
        features = self.feature_extractor(attended_features)
        
        # Multiple heads
        out1 = self.head1(features)
        out2 = self.head2(features)
        
        # Combine outputs
        combined = torch.cat([out1, out2], dim=1)
        final_output = self.combiner(combined)
        
        return final_output

class EnsembleClassifier(nn.Module):
    """Ensemble of multiple classifiers for better accuracy"""
    
    def __init__(self, feature_dim: int = 165, num_classes: int = 4, num_models: int = 3):
        super(EnsembleClassifier, self).__init__()
        self.num_models = num_models
        self.classifiers = nn.ModuleList([
            EmotionClassifier(feature_dim, num_classes) for _ in range(num_models)
        ])
    
    def forward(self, x):
        outputs = []
        for classifier in self.classifiers:
            outputs.append(classifier(x))
        # Average the outputs
        return torch.stack(outputs).mean(dim=0)

class GANMER:
    """GAN-based Music Emotion Recognition System"""
    
    def __init__(self, feature_dim: int = 165, latent_dim: int = 100, num_classes: int = 4):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.feature_dim = feature_dim
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        
        # Initialize models
        self.generator = Generator(latent_dim, feature_dim, num_classes).to(self.device)
        self.discriminator = Discriminator(feature_dim, num_classes).to(self.device)
        self.classifier = EmotionClassifier(feature_dim, num_classes).to(self.device)
        
        # Optimizers
        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=0.0001, betas=(0.5, 0.999))
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999))
        self.c_optimizer = optim.Adam(self.classifier.parameters(), lr=0.001, weight_decay=1e-5)
        
        # Loss functions
        self.adversarial_loss = nn.BCELoss()
        self.classification_loss = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        # Focal loss for handling class imbalance
        self.focal_loss = self._focal_loss
        
        # Emotion mapping
        self.emotion_mapping = {
            0: "Happy",
            1: "Tense", 
            2: "Sad",
            3: "Relaxed"
        }
        
        # Store training data for recommendations
        self.training_data = None
        self.training_labels = None
    
    def _focal_loss(self, inputs, targets, alpha=1, gamma=2):
        """Focal loss to handle class imbalance"""
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = alpha * (1-pt)**gamma * ce_loss
        return focal_loss.mean()
    
    def train_gan(self, dataloader: DataLoader, epochs: int = 100):
        """Train the GAN model"""
        print("Training GAN model...")
        
        for epoch in range(epochs):
            g_losses = []
            d_losses = []
            
            for batch_idx, (real_features, real_labels) in enumerate(dataloader):
                real_features = real_features.to(self.device)
                real_labels = real_labels.to(self.device)
                batch_size = real_features.size(0)
                
                # Create one-hot encoded labels
                real_labels_onehot = torch.zeros(batch_size, self.num_classes).to(self.device)
                real_labels_onehot.scatter_(1, real_labels.unsqueeze(1), 1)
                
                # Ground truths
                valid = torch.ones(batch_size, 1).to(self.device)
                fake = torch.zeros(batch_size, 1).to(self.device)
                
                # ---------------------
                #  Train Discriminator
                # ---------------------
                self.d_optimizer.zero_grad()
                
                # Real samples
                real_validity, real_emotion = self.discriminator(real_features)
                d_real_loss = self.adversarial_loss(real_validity, valid)
                d_real_class_loss = self.classification_loss(real_emotion, real_labels)
                
                # Fake samples
                z = torch.randn(batch_size, self.latent_dim).to(self.device)
                fake_features = self.generator(z, real_labels_onehot)
                fake_validity, fake_emotion = self.discriminator(fake_features.detach())
                d_fake_loss = self.adversarial_loss(fake_validity, fake)
                d_fake_class_loss = self.classification_loss(fake_emotion, real_labels)
                
                # Total discriminator loss
                d_loss = (d_real_loss + d_fake_loss) / 2 + (d_real_class_loss + d_fake_class_loss) / 2
                d_loss.backward()
                self.d_optimizer.step()
                
                # -----------------
                #  Train Generator
                # -----------------
                self.g_optimizer.zero_grad()
                
                # Generate fake samples
                fake_validity, fake_emotion = self.discriminator(fake_features)
                g_loss = self.adversarial_loss(fake_validity, valid)
                g_class_loss = self.classification_loss(fake_emotion, real_labels)
                
                # Total generator loss
                total_g_loss = g_loss + g_class_loss
                total_g_loss.backward()
                self.g_optimizer.step()
                
                g_losses.append(total_g_loss.item())
                d_losses.append(d_loss.item())
            
            if epoch % 10 == 0:
                print(f"Epoch [{epoch}/{epochs}] - G Loss: {np.mean(g_losses):.4f}, D Loss: {np.mean(d_losses):.4f}")
    
    def train_classifier(self, train_loader: DataLoader, val_loader: DataLoader = None, epochs: int = 100):
        """Train the emotion classifier on real data"""
        print("Training emotion classifier...")
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.c_optimizer, mode='min', factor=0.5, patience=10)
        
        best_accuracy = 0
        patience_counter = 0
        patience = 20  # Early stopping patience
        
        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            total = 0
            
            # Training mode
            self.classifier.train()
            
            for batch_idx, (features, labels) in enumerate(train_loader):
                features = features.to(self.device)
                labels = labels.to(self.device)
                
                self.c_optimizer.zero_grad()
                
                outputs = self.classifier(features)
                # Use focal loss for better handling of class imbalance
                loss = self.focal_loss(outputs, labels)
                
                loss.backward()
                self.c_optimizer.step()
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            # Calculate training accuracy
            train_accuracy = 100 * correct / total
            train_loss = total_loss / len(train_loader)
            
            # Validation phase
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
                
                val_accuracy = 100 * val_correct / val_total
                val_loss = val_loss / len(val_loader)
                
                # Learning rate scheduling based on validation loss
                scheduler.step(val_loss)
                
                # Early stopping check based on validation accuracy
                if val_accuracy > best_accuracy:
                    best_accuracy = val_accuracy
                    patience_counter = 0
                    # Save best model
                    torch.save(self.classifier.state_dict(), 'best_classifier.pth')
                else:
                    patience_counter += 1
                
                if epoch % 10 == 0:
                    print(f"Epoch [{epoch}/{epochs}] - Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%, Best: {best_accuracy:.2f}%")
            else:
                # Learning rate scheduling based on training loss
                scheduler.step(train_loss)
                
                # Early stopping check based on training accuracy
                if train_accuracy > best_accuracy:
                    best_accuracy = train_accuracy
                    patience_counter = 0
                    # Save best model
                    torch.save(self.classifier.state_dict(), 'best_classifier.pth')
                else:
                    patience_counter += 1
                
                if epoch % 10 == 0:
                    print(f"Epoch [{epoch}/{epochs}] - Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.2f}%, Best: {best_accuracy:.2f}%")
            
            # Early stopping
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch} with best accuracy: {best_accuracy:.2f}%")
                # Load best model
                self.classifier.load_state_dict(torch.load('best_classifier.pth'))
                break
    
    def generate_synthetic_data(self, num_samples: int, emotion_label: int) -> np.ndarray:
        """Generate synthetic data for a specific emotion"""
        self.generator.eval()
        
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim).to(self.device)
            labels = torch.zeros(num_samples, self.num_classes).to(self.device)
            labels[:, emotion_label] = 1
            
            synthetic_features = self.generator(z, labels)
            return synthetic_features.cpu().numpy()
    
    def classify_emotion(self, features: np.ndarray) -> Tuple[int, str]:
        """Classify emotion from features"""
        self.classifier.eval()
        
        with torch.no_grad():
            features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
            outputs = self.classifier(features_tensor)
            _, predicted = torch.max(outputs, 1)
            
            emotion_id = predicted.item()
            emotion_name = self.emotion_mapping[emotion_id]
            
            return emotion_id, emotion_name
    
    def extract_features_from_midi(self, midi_path: str) -> Optional[np.ndarray]:
        """Extract features from a MIDI file using both pretty_midi and librosa"""
        try:
            midi_data = pretty_midi.PrettyMIDI(midi_path)
            
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
                return None
                
            all_notes = np.array(all_notes)
            features.extend([
                all_notes.mean(),
                all_notes.std(),
                all_notes.max() - all_notes.min(),
                len(np.unique(all_notes))
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
                features.extend([4, 4])
            
            # Convert MIDI to audio for librosa analysis
            try:
                # Synthesize audio from MIDI
                audio_data = midi_data.synthesize(fs=22050)
                
                # Extract librosa features
                librosa_features = self._extract_librosa_features(audio_data, sr=22050)
                features.extend(librosa_features)
                
            except Exception as e:
                print(f"Warning: Could not extract librosa features from {midi_path}: {e}")
                # Add default librosa features if extraction fails
                default_librosa_features = [0.0] * 20  # Default values for librosa features
                features.extend(default_librosa_features)
            
            # Pad or truncate to fixed length
            features = np.array(features)
            if len(features) < self.feature_dim:
                features = np.pad(features, (0, self.feature_dim - len(features)))
            else:
                features = features[:self.feature_dim]
            
            return features
            
        except Exception as e:
            print(f"Error extracting features from {midi_path}: {e}")
            return None
    
    def _extract_librosa_features(self, audio_data: np.ndarray, sr: int = 22050) -> List[float]:
        """Extract advanced audio features using librosa (optimized for speed)"""
        features = []
        
        try:
            # Ensure audio_data is 1D and limit length for faster processing
            if len(audio_data.shape) > 1:
                audio_data = audio_data.flatten()
            
            # Limit audio length to 10 seconds for faster processing
            max_length = sr * 10  # 10 seconds
            if len(audio_data) > max_length:
                audio_data = audio_data[:max_length]
            
            # Spectral features (faster alternatives)
            spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=sr, hop_length=512)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sr, hop_length=512)[0]
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio_data, sr=sr, hop_length=512)[0]
            
            features.extend([
                float(np.mean(spectral_centroids)), float(np.std(spectral_centroids)),
                float(np.mean(spectral_rolloff)), float(np.std(spectral_rolloff)),
                float(np.mean(spectral_bandwidth)), float(np.std(spectral_bandwidth))
            ])
            
            # MFCC features (first 4 coefficients)
            mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13, hop_length=512)
            for i in range(4):  # Use first 4 MFCC coefficients
                features.extend([float(np.mean(mfccs[i])), float(np.std(mfccs[i]))])
            
            # Simplified chroma features (faster)
            try:
                chroma = librosa.feature.chroma_stft(y=audio_data, sr=sr, hop_length=512, n_fft=2048)
                chroma_mean = np.mean(chroma, axis=1)
                features.extend([float(np.mean(chroma_mean)), float(np.std(chroma_mean))])
            except:
                # Fallback if chroma extraction fails
                features.extend([0.5, 0.1])  # Default values
            
            # Rhythm features (simplified)
            try:
                tempo, _ = librosa.beat.beat_track(y=audio_data, sr=sr, hop_length=512)
                features.append(float(tempo))
            except:
                features.append(120.0)  # Default tempo
            
            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(audio_data, hop_length=512)[0]
            features.extend([float(np.mean(zcr)), float(np.std(zcr))])
            
            # Root mean square energy
            rms = librosa.feature.rms(y=audio_data, hop_length=512)[0]
            features.extend([float(np.mean(rms)), float(np.std(rms))])
            
            # Simplified harmonic/percussive (faster)
            try:
                harmonic, percussive = librosa.effects.hpss(audio_data, margin=1.0)
                harmonic_energy = float(np.mean(librosa.feature.rms(y=harmonic, hop_length=512)[0]))
                percussive_energy = float(np.mean(librosa.feature.rms(y=percussive, hop_length=512)[0]))
                features.extend([harmonic_energy, percussive_energy])
            except:
                # Fallback values
                features.extend([0.1, 0.1])
            
        except Exception as e:
            print(f"Error in librosa feature extraction: {e}")
            # Return default values if extraction fails
            features = [0.0] * 20
        
        # Ensure we have exactly 20 features
        if len(features) != 20:
            if len(features) < 20:
                features.extend([0.0] * (20 - len(features)))
            else:
                features = features[:20]
        
        return features
    
    def recommend_similar_tracks(self, input_midi_path: str, dataset_dir: str, num_recommendations: int = 5) -> List[Dict]:
        """Recommend similar tracks based on emotion"""
        
        # Extract features from input track
        input_features = self.extract_features_from_midi(input_midi_path)
        if input_features is None:
            print("Could not extract features from input track")
            return []
        
        # Classify emotion of input track
        input_emotion_id, input_emotion_name = self.classify_emotion(input_features)
        print(f"Input track emotion: {input_emotion_name}")
        
        # Load all dataset features and labels
        if self.training_data is None:
            self._load_dataset_features(dataset_dir)
        
        # Find tracks with similar emotion
        similar_indices = np.where(self.training_labels == input_emotion_id)[0]
        
        if len(similar_indices) == 0:
            print(f"No tracks found with emotion: {input_emotion_name}")
            return []
        
        # Calculate similarity scores
        similarities = []
        for idx in similar_indices:
            similarity = self._calculate_similarity(input_features, self.training_data[idx])
            similarities.append((idx, similarity))
        
        # Sort by similarity and get top recommendations
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_indices = [idx for idx, _ in similarities[:num_recommendations]]
        
        # Get file paths for recommendations
        recommendations = []
        labels_df = pd.read_csv(os.path.join(dataset_dir, 'label.csv'))
        
        for idx in top_indices:
            if idx < len(labels_df):
                track_id = labels_df.iloc[idx]['ID']
                midi_file = f"{track_id}.mid"
                midi_path = os.path.join(dataset_dir, 'midis', midi_file)
                
                if os.path.exists(midi_path):
                    recommendations.append({
                        'file_path': midi_path,
                        'track_id': track_id,
                        'emotion': self.emotion_mapping[self.training_labels[idx]],
                        'similarity_score': similarities[top_indices.index(idx)][1]
                    })
        
        return recommendations
    
    def _load_dataset_features(self, dataset_dir: str):
        """Load all dataset features for recommendations"""
        print("Loading dataset features for recommendations...")
        
        labels_df = pd.read_csv(os.path.join(dataset_dir, 'label.csv'))
        midi_dir = os.path.join(dataset_dir, 'midis')
        
        features_list = []
        labels_list = []
        
        for _, row in labels_df.iterrows():
            midi_file = f"{row['ID']}.mid"
            midi_path = os.path.join(midi_dir, midi_file)
            
            if os.path.exists(midi_path):
                features = self.extract_features_from_midi(midi_path)
                if features is not None:
                    features_list.append(features)
                    labels_list.append(row['4Q'] - 1)  # Convert to 0-based
        
        self.training_data = np.array(features_list)
        self.training_labels = np.array(labels_list)
        
        print(f"Loaded {len(self.training_data)} tracks for recommendations")
    
    def _calculate_similarity(self, features1: np.ndarray, features2: np.ndarray) -> float:
        """Calculate cosine similarity between two feature vectors"""
        dot_product = np.dot(features1, features2)
        norm1 = np.linalg.norm(features1)
        norm2 = np.linalg.norm(features2)
        
        if norm1 == 0 or norm2 == 0:
            return 0
        
        return dot_product / (norm1 * norm2)
    
    def save_models(self, save_dir: str):
        """Save trained models"""
        os.makedirs(save_dir, exist_ok=True)
        
        torch.save(self.generator.state_dict(), os.path.join(save_dir, 'generator.pth'))
        torch.save(self.discriminator.state_dict(), os.path.join(save_dir, 'discriminator.pth'))
        torch.save(self.classifier.state_dict(), os.path.join(save_dir, 'classifier.pth'))
        
        print(f"Models saved to {save_dir}")
    
    def load_models(self, save_dir: str):
        """Load trained models"""
        self.generator.load_state_dict(torch.load(os.path.join(save_dir, 'generator.pth'), map_location=self.device))
        self.discriminator.load_state_dict(torch.load(os.path.join(save_dir, 'discriminator.pth'), map_location=self.device))
        self.classifier.load_state_dict(torch.load(os.path.join(save_dir, 'classifier.pth'), map_location=self.device))
        
        print(f"Models loaded from {save_dir}")

def main():
    """Main function to run the GAN-based MER system"""
    
    # Configuration
    DATASET_DIR = "EMOPIA_1.0"
    MIDI_DIR = os.path.join(DATASET_DIR, "midis")
    LABEL_FILE = os.path.join(DATASET_DIR, "label.csv")
    SAVE_DIR = "models"
    
    # Hyperparameters
    FEATURE_DIM = 165  # Increased for additional MIDI features
    LATENT_DIM = 100
    NUM_CLASSES = 4
    BATCH_SIZE = 8  # Even smaller batch size for better generalization
    GAN_EPOCHS = 20  # Reduced GAN epochs since it's not the main focus
    CLASSIFIER_EPOCHS = 120  # More epochs for better classifier training
    
    print("=== GAN-based Music Emotion Recognition System ===")
    print(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    
    # Load dataset
    print("\n1. Loading EMOPIA dataset...")
    dataset = EMOPIADataset(MIDI_DIR, LABEL_FILE, FEATURE_DIM)
    
    if len(dataset) == 0:
        print("No valid samples found in dataset!")
        return
    
    # Split dataset into train, validation, and test
    train_data, temp_data, train_labels, temp_labels = train_test_split(
        dataset.data, dataset.labels, test_size=0.3, random_state=42, stratify=dataset.labels
    )
    
    val_data, test_data, val_labels, test_labels = train_test_split(
        temp_data, temp_labels, test_size=0.67, random_state=42, stratify=temp_labels
    )
    
    # Create data loaders
    train_dataset = [(torch.FloatTensor(features), label) for features, label in zip(train_data, train_labels)]
    val_dataset = [(torch.FloatTensor(features), label) for features, label in zip(val_data, val_labels)]
    test_dataset = [(torch.FloatTensor(features), label) for features, label in zip(test_data, test_labels)]
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    print(f"Test samples: {len(test_data)}")
    
    # Initialize GAN MER system
    print("\n2. Initializing GAN MER system...")
    gan_mer = GANMER(FEATURE_DIM, LATENT_DIM, NUM_CLASSES)
    
    # Train models
    print("\n3. Training models...")
    gan_mer.train_gan(train_loader, epochs=GAN_EPOCHS)
    
    # Generate synthetic data for data augmentation
    print("\n3.1. Generating synthetic data for augmentation...")
    synthetic_features = []
    synthetic_labels = []
    
    # Generate more data for underrepresented classes (Sad has lowest recall)
    class_samples = {
        0: 60,  # Happy - more samples
        1: 50,  # Tense - standard
        2: 80,  # Sad - most samples (lowest recall)
        3: 50   # Relaxed - standard
    }
    
    for emotion_id in range(NUM_CLASSES):
        num_samples = class_samples[emotion_id]
        synthetic_data = gan_mer.generate_synthetic_data(num_samples, emotion_id)
        synthetic_features.extend(synthetic_data)
        synthetic_labels.extend([emotion_id] * num_samples)
    
    # Combine real and synthetic data
    combined_features = np.vstack([train_data, np.array(synthetic_features)])
    combined_labels = np.concatenate([train_labels, np.array(synthetic_labels)])
    
    # Create augmented data loader
    augmented_dataset = [(torch.FloatTensor(features), label) for features, label in zip(combined_features, combined_labels)]
    augmented_loader = DataLoader(augmented_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    print(f"Training with {len(train_data)} real + {len(synthetic_features)} synthetic samples")
    
    # Train classifier with augmented data and validation
    gan_mer.train_classifier(augmented_loader, val_loader, epochs=CLASSIFIER_EPOCHS)
    
    # Evaluate classifier
    print("\n4. Evaluating classifier...")
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
    print(f"Test Accuracy: {accuracy:.2f}%")
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_predictions, 
                              target_names=['Happy', 'Tense', 'Sad', 'Relaxed']))
    
    # Save models
    print("\n5. Saving models...")
    gan_mer.save_models(SAVE_DIR)
    
    # Generate synthetic data for each emotion
    print("\n6. Generating synthetic data...")
    for emotion_id, emotion_name in gan_mer.emotion_mapping.items():
        synthetic_data = gan_mer.generate_synthetic_data(10, emotion_id)
        print(f"Generated {len(synthetic_data)} samples for {emotion_name}")
    
    # # Example recommendation
    # print("\n7. Testing recommendation system...")
    
    # # Find a sample MIDI file for testing
    # sample_files = [f for f in os.listdir(MIDI_DIR) if f.endswith('.mid')]
    # if sample_files:
    #     sample_file = os.path.join(MIDI_DIR, sample_files[0])
    #     print(f"Testing with sample file: {sample_files[0]}")
        
    #     recommendations = gan_mer.recommend_similar_tracks(sample_file, DATASET_DIR, num_recommendations=5)
        
    #     if recommendations:
    #         print("\nTop 5 recommendations:")
    #         for i, rec in enumerate(recommendations, 1):
    #             print(f"{i}. {rec['track_id']} - {rec['emotion']} (similarity: {rec['similarity_score']:.3f})")
    #     else:
    #         print("No recommendations found")
    
    print("\n=== System ready for use! ===")
    print("You can now:")
    print("1. Classify emotions from MIDI files")
    print("2. Generate synthetic music with specific emotions")
    print("3. Get emotion-based music recommendations")

if __name__ == "__main__":
    main()
