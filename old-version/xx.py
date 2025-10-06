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
warnings.filterwarnings('ignore')

# GAN-based Music Emotion Recognition System with Adversarial Augmentation

class EMOPIADataset(Dataset):
    """Dataset class for EMOPIA MIDI files with 6-class granular emotion labels"""
    
    def __init__(self, midi_dir: str, label_file: str, sequence_length: int = 128, q1_arousal_scale: float = 1.0, q2_tension_scale: float = 1.0):
        self.midi_dir = midi_dir
        self.sequence_length = sequence_length
        self.data = []
        self.labels = []  # 6-class labels: 0..5
        self.sample_metrics: List[Dict[str, float]] = []  # per-sample auxiliary metrics aligned with data
        self.q1_arousal_median: float = 0.0
        self.q2_tension_median: float = 0.0
        self.label_names = ['Happy', 'Excited', 'Tense', 'Fearful', 'Sad', 'Hopeful']
        self.q1_arousal_scale = q1_arousal_scale
        self.q2_tension_scale = q2_tension_scale
        
        # Load labels
        labels_df = pd.read_csv(label_file)
        
        # Internal containers to derive thresholds per quadrant
        temp_samples = []  # list of dicts with features, quadrant, metrics
        
        # Process each MIDI file
        for _, row in labels_df.iterrows():
            midi_file = f"{row['ID']}.mid"
            midi_path = os.path.join(midi_dir, midi_file)
            base_quadrant = int(row['4Q'])  # 1..4
            
            if os.path.exists(midi_path):
                try:
                    features, metrics = self.extract_midi_features_with_metrics(midi_path)
                    if features is not None and metrics is not None:
                        temp_samples.append({
                            'features': features,
                            'quadrant': base_quadrant,
                            'metrics': metrics
                        })
                except Exception as e:
                    print(f"Error processing {midi_file}: {e}")
        
        if len(temp_samples) == 0:
            self.data = np.array([])
            self.labels = np.array([])
            print("Loaded 0 samples")
            return
        
        # Compute simple arousal/tension thresholds per quadrant from collected metrics
        # Arousal proxy: tempo + rms_mean + note_density + spectral_centroid_mean
        # Tension proxy (Q2 split): spectral_bandwidth_mean + zcr_mean + chroma_std
        q_to_arousal_values = {1: [], 2: [], 3: [], 4: []}
        q2_tension_values = []
        for sample in temp_samples:
            m = sample['metrics']
            arousal = (m['tempo'] + 120 * m['rms_mean'] + 10 * m['note_density'] + m['spectral_centroid_mean'] / 1000.0)
            q_to_arousal_values[sample['quadrant']].append(arousal)
            if sample['quadrant'] == 2:
                tension = (m['spectral_bandwidth_mean'] / 1000.0 + 10 * m['zcr_mean'] + 5 * m['chroma_std'])
                q2_tension_values.append(tension)
        
        q1_arousal_median = np.median(q_to_arousal_values[1]) if len(q_to_arousal_values[1]) else 0.0
        q2_tension_median = np.median(q2_tension_values) if len(q2_tension_values) else 0.0
        self.q1_arousal_median = float(q1_arousal_median) * float(self.q1_arousal_scale)
        self.q2_tension_median = float(q2_tension_median) * float(self.q2_tension_scale)
        
        # Finalize features and 6-class labels
        finalized_features = []
        finalized_labels = []
        for sample in temp_samples:
            features = sample['features']
            q = sample['quadrant']
            m = sample['metrics']
            # Compute proxies again for decision
            arousal = (m['tempo'] + 120 * m['rms_mean'] + 10 * m['note_density'] + m['spectral_centroid_mean'] / 1000.0)
            tension = (m['spectral_bandwidth_mean'] / 1000.0 + 10 * m['zcr_mean'] + 5 * m['chroma_std'])
            
            # 6-class mapping with better Q1 handling
            # 0: Happy (Q1 lower arousal)
            # 1: Excited (Q1 higher arousal)
            # 2: Tense (Q2 lower tension)
            # 3: Fearful (Q2 higher tension)
            # 4: Sad (Q3)
            # 5: Hopeful (Q4)
            if q == 1:
                # More conservative Q1 split - favor Happy over Excited to avoid Q1->Sad confusion
                label6 = 1 if arousal > q1_arousal_median else 0
            elif q == 2:
                label6 = 3 if tension > q2_tension_median else 2
            elif q == 3:
                label6 = 4
            else:  # q == 4
                label6 = 5
            finalized_features.append(features)
            finalized_labels.append(label6)
            # store metrics with computed proxies
            stored_m = dict(m)
            stored_m['arousal_proxy'] = float(arousal)
            stored_m['tension_proxy'] = float(tension)
            self.sample_metrics.append(stored_m)
        
        self.data = np.array(finalized_features)
        self.labels = np.array(finalized_labels)
        
        scaler = RobustScaler()
        self.data = scaler.fit_transform(self.data)
        
        print(f"Loaded {len(self.data)} samples with 6 emotion classes")
    
    def extract_midi_features_with_metrics(self, midi_path: str) -> Tuple[Optional[np.ndarray], Optional[Dict[str, float]]]:
        """Extract features and auxiliary metrics from a MIDI file using pretty_midi and librosa"""

        # Question: What are the key features that will say "this is a sad song" vs "this is a happy song"?
        try:
            midi_data = pretty_midi.PrettyMIDI(midi_path)
            
            # Extract basic MIDI features
            features = []
            metrics: Dict[str, float] = {}
            
            # Tempo features
            tempo = midi_data.estimate_tempo()
            tempo_changes = midi_data.get_tempo_changes()[1]
            metrics['tempo'] = float(tempo)
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
            pitch_range = all_notes.max() - all_notes.min()
            features.extend([
                all_notes.mean(),  # Average pitch
                all_notes.std(),   # Pitch standard deviation
                pitch_range,  # Pitch range
                len(np.unique(all_notes)),  # Unique pitches
                np.percentile(all_notes, 25),  # 25th percentile
                np.percentile(all_notes, 75),  # 75th percentile
                np.median(all_notes)  # Median pitch
            ])
            metrics['pitch_range'] = float(pitch_range)
            # Pitch-class histogram entropy (dissonance/ambiguity proxy)
            pitch_classes = (all_notes % 12).astype(int)
            pch_counts = np.bincount(pitch_classes, minlength=12).astype(float)
            pch_probs = pch_counts / np.maximum(pch_counts.sum(), 1e-6)
            pch_entropy = float(-(pch_probs * np.log(pch_probs + 1e-12)).sum())
            features.append(pch_entropy)
            metrics['pch_entropy'] = pch_entropy
            
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
            metrics['note_density'] = float(note_density)
            
            # Polyphony (simplified calculation)
            if total_duration > 0:
                time_points = np.linspace(0, total_duration, min(50, int(total_duration * 10)))
                polyphony = [sum(1 for instrument in midi_data.instruments 
                               for note in instrument.notes 
                               if note.start <= t <= note.end) for t in time_points]
                features.extend([np.mean(polyphony), np.std(polyphony)])
            else:
                features.extend([0, 0])
            
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
                librosa_features, librosa_metrics = self._extract_librosa_features_with_metrics(audio_data, sr=22050)
                features.extend(librosa_features)
                
                # Additional audio features: spectral flatness and tonnetz statistics
                try:
                    flatness = librosa.feature.spectral_flatness(y=audio_data, hop_length=512)[0]
                    flat_mean, flat_std = float(np.mean(flatness)), float(np.std(flatness))
                except Exception:
                    flat_mean, flat_std = 0.0, 0.0
                features.extend([flat_mean, flat_std])
                librosa_metrics['spectral_flatness_mean'] = flat_mean
                
                try:
                    chroma_cqt = librosa.feature.chroma_cqt(y=audio_data, sr=22050)
                    tonnetz = librosa.feature.tonnetz(chroma=chroma_cqt)
                    tonnetz_mean = np.mean(tonnetz, axis=1)
                    tonnetz_std = np.std(tonnetz, axis=1)
                    # summarize as means and stds
                    features.extend(list(tonnetz_mean))
                    features.extend(list(tonnetz_std))
                    librosa_metrics['tonnetz_var'] = float(np.var(tonnetz))
                except Exception:
                    # fallback 12 zeros (6 mean + 6 std)
                    features.extend([0.0]*12)
                
                # Additional advanced features for better emotion discrimination
                try:
                    # Spectral contrast
                    contrast = librosa.feature.spectral_contrast(y=audio_data, sr=22050, hop_length=512)
                    features.extend([float(np.mean(contrast)), float(np.std(contrast))])
                    
                    # Spectral rolloff at different percentiles
                    rolloff_95 = librosa.feature.spectral_rolloff(y=audio_data, sr=22050, roll_percent=0.95, hop_length=512)[0]
                    rolloff_85 = librosa.feature.spectral_rolloff(y=audio_data, sr=22050, roll_percent=0.85, hop_length=512)[0]
                    features.extend([float(np.mean(rolloff_95)), float(np.mean(rolloff_85))])
                    
                    # Tempo and rhythm features
                    tempo, beats = librosa.beat.beat_track(y=audio_data, sr=22050, hop_length=512)
                    beat_frames = librosa.frames_to_time(beats, sr=22050, hop_length=512)
                    if len(beat_frames) > 1:
                        beat_intervals = np.diff(beat_frames)
                        features.extend([float(np.mean(beat_intervals)), float(np.std(beat_intervals))])
                    else:
                        features.extend([0.0, 0.0])
                    
                    # Polyphonic features
                    onset_frames = librosa.onset.onset_detect(y=audio_data, sr=22050, hop_length=512)
                    onset_times = librosa.frames_to_time(onset_frames, sr=22050, hop_length=512)
                    if len(onset_times) > 1:
                        onset_intervals = np.diff(onset_times)
                        features.extend([float(np.mean(onset_intervals)), float(np.std(onset_intervals))])
                    else:
                        features.extend([0.0, 0.0])
                        
                except Exception as e:
                    # Add default values for additional features
                    features.extend([0.0] * 6)  # 6 additional features
                    print(f"Warning: Could not extract advanced features from {midi_path}: {e}")
                    
                # Persist selected metrics for downstream splitting
                metrics.update(librosa_metrics)
                
            except Exception as e:
                print(f"Warning: Could not extract librosa features from {midi_path}: {e}")
                # Add default librosa features if extraction fails
                default_librosa_features = [0.0] * 20  # Default values for librosa features
                features.extend(default_librosa_features)
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
    
    def _extract_librosa_features_with_metrics(self, audio_data: np.ndarray, sr: int = 22050) -> Tuple[List[float], Dict[str, float]]:
        """Extract advanced audio features using librosa (optimized for speed), plus summary metrics.
        This mirrors the dataset extractor so GANMER can also use it at inference time."""
        features: List[float] = []
        metrics: Dict[str, float] = {}
        try:
            # Ensure audio_data is 1D and limit length for faster processing
            if len(audio_data.shape) > 1:
                audio_data = audio_data.flatten()
            max_length = sr * 10
            if len(audio_data) > max_length:
                audio_data = audio_data[:max_length]
            # Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=sr, hop_length=512)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sr, hop_length=512)[0]
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio_data, sr=sr, hop_length=512)[0]
            features.extend([
                float(np.mean(spectral_centroids)), float(np.std(spectral_centroids)),
                float(np.mean(spectral_rolloff)), float(np.std(spectral_rolloff)),
                float(np.mean(spectral_bandwidth)), float(np.std(spectral_bandwidth))
            ])
            metrics['spectral_centroid_mean'] = float(np.mean(spectral_centroids))
            metrics['spectral_bandwidth_mean'] = float(np.mean(spectral_bandwidth))
            # MFCC (first 4 coefficients only)
            mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=4, hop_length=512)
            for i in range(4):
                features.extend([float(np.mean(mfccs[i])), float(np.std(mfccs[i]))])
            # Chroma
            try:
                chroma = librosa.feature.chroma_stft(y=audio_data, sr=sr, hop_length=512, n_fft=2048)
                chroma_mean = np.mean(chroma, axis=1)
                chroma_std = float(np.std(chroma_mean))
                features.extend([float(np.mean(chroma_mean)), chroma_std])
                metrics['chroma_std'] = chroma_std
            except:
                features.extend([0.5, 0.1])
                metrics['chroma_std'] = 0.1
            # Tempo
            try:
                tempo, _ = librosa.beat.beat_track(y=audio_data, sr=sr, hop_length=512)
                features.append(float(tempo))
            except:
                features.append(120.0)
            # ZCR
            zcr = librosa.feature.zero_crossing_rate(audio_data, hop_length=512)[0]
            zcr_mean = float(np.mean(zcr))
            features.extend([zcr_mean, float(np.std(zcr))])
            metrics['zcr_mean'] = zcr_mean
            # RMS
            rms = librosa.feature.rms(y=audio_data, hop_length=512)[0]
            rms_mean = float(np.mean(rms))
            features.extend([rms_mean, float(np.std(rms))])
            metrics['rms_mean'] = rms_mean
            # Skip HPSS for efficiency - not essential for emotion recognition
        except Exception as e:
            print(f"Error in librosa feature extraction (GANMER): {e}")
            features = [0.0] * 20
            metrics.update({
                'spectral_centroid_mean': 0.0,
                'spectral_bandwidth_mean': 0.0,
                'zcr_mean': 0.0,
                'rms_mean': 0.0,
                'chroma_std': 0.0
            })
        # Ensure 18 features (reduced from 20 for efficiency)
        if len(features) != 18:
            if len(features) < 18:
                features.extend([0.0] * (18 - len(features)))
            else:
                features = features[:18]
        return features, metrics
        
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
            metrics['spectral_centroid_mean'] = float(np.mean(spectral_centroids))
            metrics['spectral_bandwidth_mean'] = float(np.mean(spectral_bandwidth))
            
            # MFCC features (first 4 coefficients)
            mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13, hop_length=512)
            for i in range(4):  # Use first 4 MFCC coefficients
                features.extend([float(np.mean(mfccs[i])), float(np.std(mfccs[i]))])
            
            # Simplified chroma features (faster)
            try:
                chroma = librosa.feature.chroma_stft(y=audio_data, sr=sr, hop_length=512, n_fft=2048)
                chroma_mean = np.mean(chroma, axis=1)
                chroma_std = float(np.std(chroma_mean))
                features.extend([float(np.mean(chroma_mean)), chroma_std])
                metrics['chroma_std'] = chroma_std
            except:
                # Fallback if chroma extraction fails
                features.extend([0.5, 0.1])  # Default values
                metrics['chroma_std'] = 0.1
            
            # Rhythm features (simplified)
            try:
                tempo, _ = librosa.beat.beat_track(y=audio_data, sr=sr, hop_length=512)
                features.append(float(tempo))
            except:
                features.append(120.0)  # Default tempo
            
            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(audio_data, hop_length=512)[0]
            zcr_mean = float(np.mean(zcr))
            features.extend([zcr_mean, float(np.std(zcr))])
            metrics['zcr_mean'] = zcr_mean
            
            # Root mean square energy
            rms = librosa.feature.rms(y=audio_data, hop_length=512)[0]
            rms_mean = float(np.mean(rms))
            features.extend([rms_mean, float(np.std(rms))])
            metrics['rms_mean'] = rms_mean
            
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
            metrics.update({
                'spectral_centroid_mean': 0.0,
                'spectral_bandwidth_mean': 0.0,
                'zcr_mean': 0.0,
                'rms_mean': 0.0,
                'chroma_std': 0.0
            })
        
        # Ensure we have exactly 20 features
        if len(features) != 20:
            if len(features) < 20:
                features.extend([0.0] * (20 - len(features)))
            else:
                features = features[:20]
        
        return features, metrics
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.data[idx]), int(self.labels[idx])  # already 0-based 6-class

class Generator(nn.Module):
    """Generator network for GAN"""
    
    def __init__(self, latent_dim: int = 100, feature_dim: int = 165, num_classes: int = 6):
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
    
    def __init__(self, feature_dim: int = 165, num_classes: int = 6):
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
            
            nn.Linear(64, 1)  # Hinge loss: no sigmoid
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
    
    def __init__(self, feature_dim: int = 165, num_classes: int = 6):
        super(EmotionClassifier, self).__init__()
        
        # Feature attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, feature_dim),
            nn.Sigmoid()
        )
        
        # Enhanced classifier with residual connections
        self.feature_extractor = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.1)
        )
        
        # Residual connection
        self.residual = nn.Linear(feature_dim, 256)
        
        # Enhanced heads with more capacity
        self.head1 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, num_classes)
        )
        
        self.head2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, num_classes)
        )
        
        # Final combination
        self.combiner = nn.Linear(num_classes * 2, num_classes)
    
    def forward(self, x):
        # Apply attention
        attention_weights = self.attention(x)
        attended_features = x * attention_weights
        
        # Extract features with residual connection
        features = self.feature_extractor(attended_features)
        residual_features = self.residual(x)
        features = features + residual_features  # Residual connection
        
        # Multiple heads
        out1 = self.head1(features)
        out2 = self.head2(features)
        
        # Combine outputs
        combined = torch.cat([out1, out2], dim=1)
        final_output = self.combiner(combined)
        
        return final_output

class AdversarialAugmentation(nn.Module):
    """Adversarial augmentation network for better data generation"""
    
    def __init__(self, feature_dim: int, num_classes: int = 6):
        super(AdversarialAugmentation, self).__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        
        # Generator: learns to augment features
        self.generator = nn.Sequential(
            nn.Linear(feature_dim + num_classes, 512),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.3),
            
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            
            nn.Linear(512, feature_dim),
            nn.Tanh()
        )
        
        # Discriminator: distinguishes real from augmented features
        self.discriminator = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def generate_augmented(self, features, labels):
        """Generate augmented features"""
        labels_onehot = torch.zeros(features.size(0), self.num_classes).to(features.device)
        labels_onehot.scatter_(1, labels.unsqueeze(1), 1)
        
        input_features = torch.cat([features, labels_onehot], dim=1)
        augmented = self.generator(input_features)
        return features + 0.1 * augmented  # Add small augmentation
    
    def discriminate(self, features):
        """Discriminate real vs augmented features"""
        return self.discriminator(features)

class GANMER:
    """Improved GAN-based Music Emotion Recognition System with Adversarial Augmentation"""
    
    def __init__(self, feature_dim: int = 210, latent_dim: int = 100, num_classes: int = 6):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.feature_dim = feature_dim
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        
        # Initialize improved models
        self.augmentation_gan = AdversarialAugmentation(feature_dim, num_classes).to(self.device)
        self.classifier = EmotionClassifier(feature_dim, num_classes).to(self.device)
        
        # Optimizers with different learning rates
        self.g_optimizer = optim.Adam(self.augmentation_gan.generator.parameters(), lr=0.0001, betas=(0.5, 0.999))
        self.d_optimizer = optim.Adam(self.augmentation_gan.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.c_optimizer = optim.Adam(self.classifier.parameters(), lr=0.0003, weight_decay=1e-4)
        
        # Loss functions
        self.adversarial_loss = nn.BCELoss()
        self.classification_loss = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.focal_loss = self._focal_loss
        
        # Emotion mapping
        self.emotion_mapping = {
            0: "Happy", 1: "Excited", 2: "Tense", 
            3: "Fearful", 4: "Sad", 5: "Hopeful"
        }
        
        # Training histories
        self.gan_history = {'g_loss': [], 'd_loss': []}
        self.classifier_history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        
        # Class weights for focal loss
        self.class_weights: Optional[torch.Tensor] = None
        
        # For recommendations
        self.training_data: Optional[np.ndarray] = None
        self.training_labels: Optional[np.ndarray] = None
    
    def _focal_loss(self, inputs, targets, alpha=1, gamma=2):
        """Focal loss to handle class imbalance"""
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        # Use per-class alpha if provided
        if self.class_weights is not None:
            alpha_vec = self.class_weights.to(inputs.device)
            alpha_factor = alpha_vec[targets]
        else:
            alpha_factor = alpha
        focal_loss = alpha_factor * (1-pt)**gamma * ce_loss
        return focal_loss.mean()
    
    def train_gan(self, dataloader: DataLoader, epochs: int = 100):
        """Train the adversarial augmentation GAN"""
        print("Training adversarial augmentation GAN...")
        
        for epoch in range(epochs):
            g_losses = []
            d_losses = []
            
            for batch_idx, (features, labels) in enumerate(dataloader):
                features = features.to(self.device)
                labels = labels.to(self.device)
                batch_size = features.size(0)
                
                # Ground truths
                valid = torch.ones(batch_size, 1).to(self.device)
                fake = torch.zeros(batch_size, 1).to(self.device)
                
                # ---------------------
                #  Train Discriminator
                # ---------------------
                self.d_optimizer.zero_grad()
                
                # Real features
                real_validity = self.augmentation_gan.discriminate(features)
                d_real_loss = self.adversarial_loss(real_validity, valid)
                
                # Augmented features
                augmented_features = self.augmentation_gan.generate_augmented(features, labels)
                fake_validity = self.augmentation_gan.discriminate(augmented_features.detach())
                d_fake_loss = self.adversarial_loss(fake_validity, fake)
                
                d_loss = (d_real_loss + d_fake_loss) / 2
                d_loss.backward()
                self.d_optimizer.step()
                
                # -----------------
                #  Train Generator
                # -----------------
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
    
    def train_classifier(self, train_loader: DataLoader, val_loader: DataLoader = None, epochs: int = 100):
        """Train the emotion classifier with adversarial augmentation"""
        print("Training emotion classifier with adversarial augmentation...")
        
        # Better learning rate scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(self.c_optimizer, T_0=20, T_mult=2, eta_min=1e-6)
        
        best_accuracy = 0
        patience_counter = 0
        patience = 25  # Early stopping patience
        
        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            total = 0
            
            # Training mode
            self.classifier.train()
            
            for batch_idx, (features, labels) in enumerate(train_loader):
                features = features.to(self.device)
                labels = labels.to(self.device)
                
                # Adversarial augmentation
                if np.random.random() < 0.5:  # 50% chance of adversarial augmentation
                    with torch.no_grad():
                        augmented_features = self.augmentation_gan.generate_augmented(features, labels)
                        features = 0.7 * features + 0.3 * augmented_features
                
                # Standard data augmentation
                if np.random.random() < 0.3 and features.size(0) > 1:  # 30% chance of MixUp
                    lam = np.random.beta(0.2, 0.2)
                    index = torch.randperm(features.size(0)).to(self.device)
                    mixed_x = lam * features + (1 - lam) * features[index, :]
                    mixed_labels = lam * labels.float() + (1 - lam) * labels[index].float()
                else:
                    mixed_x = features
                    mixed_labels = labels.float()
                
                self.c_optimizer.zero_grad()
                
                outputs = self.classifier(mixed_x)
                # Use focal loss with mixed targets
                loss = self.focal_loss(outputs, mixed_labels.long())
                
                loss.backward()
                self.c_optimizer.step()
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            # Calculate training accuracy
            train_accuracy = 100 * correct / total if total else 0.0
            train_loss = total_loss / len(train_loader) if len(train_loader) else 0.0
            self.classifier_history['train_loss'].append(float(train_loss))
            self.classifier_history['train_acc'].append(float(train_accuracy))
            
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
                
                val_accuracy = 100 * val_correct / val_total if val_total else 0.0
                val_loss = val_loss / len(val_loader) if len(val_loader) else 0.0
                self.classifier_history['val_loss'].append(float(val_loss))
                self.classifier_history['val_acc'].append(float(val_accuracy))
                
                # Learning rate scheduling
                scheduler.step()
                
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
                # Learning rate scheduling
                scheduler.step()
                
                # Early stopping check based on training accuracy
                if train_accuracy > best_accuracy:
                    best_accuracy = train_accuracy
                    patience_counter = 0
                    # Save best model
                    torch.save(self.classifier.state_dict(), 'best_classifier.pth')
                else:
                    patience_counter += 1
                # When no val_loader, still log placeholders for val history
                self.classifier_history['val_loss'].append(float('nan'))
                self.classifier_history['val_acc'].append(float('nan'))
                
                if epoch % 10 == 0:
                    print(f"Epoch [{epoch}/{epochs}] - Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.2f}%, Best: {best_accuracy:.2f}%")
            
            # Early stopping
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch} with best accuracy: {best_accuracy:.2f}%")
                # Load best model
                self.classifier.load_state_dict(torch.load('best_classifier.pth'))
                break
    
    def generate_synthetic_data(self, num_samples: int, emotion_label: int) -> np.ndarray:
        """Generate synthetic data for a specific emotion using adversarial augmentation"""
        self.augmentation_gan.eval()
        
        with torch.no_grad():
            # Generate random features as base
            base_features = torch.randn(num_samples, self.feature_dim).to(self.device)
            labels = torch.full((num_samples,), emotion_label, dtype=torch.long).to(self.device)
            
            # Apply adversarial augmentation
            synthetic_features = self.augmentation_gan.generate_augmented(base_features, labels)
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
                
                # Extract librosa features (with robust fallback if helper not present)
                if hasattr(self, '_extract_librosa_features_with_metrics'):
                    librosa_features, _ = self._extract_librosa_features_with_metrics(audio_data, sr=22050)
                else:
                    # Fallback to legacy extractor without metrics
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
                    # Use current classifier to assign one of 6 emotions
                    try:
                        pred_id, _ = self.classify_emotion(features)
                        labels_list.append(pred_id)
                    except Exception:
                        labels_list.append(0)
        
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
        
        torch.save(self.augmentation_gan.state_dict(), os.path.join(save_dir, 'augmentation_gan.pth'))
        torch.save(self.classifier.state_dict(), os.path.join(save_dir, 'classifier.pth'))
        
        print(f"Models saved to {save_dir}")
    
    def load_models(self, save_dir: str):
        """Load trained models"""
        self.augmentation_gan.load_state_dict(torch.load(os.path.join(save_dir, 'augmentation_gan.pth'), map_location=self.device))
        self.classifier.load_state_dict(torch.load(os.path.join(save_dir, 'classifier.pth'), map_location=self.device))
        
        print(f"Models loaded from {save_dir}")

def run_eda(dataset: EMOPIADataset, gan_mer: GANMER, true_labels: List[int], pred_labels: List[int], save_dir: str = "eda"):
    os.makedirs(save_dir, exist_ok=True)
    label_names = dataset.label_names if hasattr(dataset, 'label_names') else [gan_mer.emotion_mapping[i] for i in range(6)]
    
    # 1) Class balance
    plt.figure(figsize=(6,4))
    sns.countplot(x=[label_names[l] for l in dataset.labels])
    plt.title('Class Balance (6 emotions)')
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'class_balance.png'))
    plt.close()
    
    # 2) Arousal and Tension distributions with medians
    if hasattr(dataset, 'sample_metrics') and len(dataset.sample_metrics) == len(dataset.labels):
        arousal_vals = [m.get('arousal_proxy', 0.0) for m in dataset.sample_metrics]
        tension_vals = [m.get('tension_proxy', 0.0) for m in dataset.sample_metrics]
        
        plt.figure(figsize=(6,4))
        sns.histplot(arousal_vals, bins=30, kde=True)
        if hasattr(dataset, 'q1_arousal_median'):
            plt.axvline(dataset.q1_arousal_median, color='r', linestyle='--', label='Q1 arousal median')
            plt.legend()
        plt.title('Arousal proxy distribution')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'arousal_distribution.png'))
        plt.close()
        
        plt.figure(figsize=(6,4))
        sns.histplot(tension_vals, bins=30, kde=True)
        if hasattr(dataset, 'q2_tension_median'):
            plt.axvline(dataset.q2_tension_median, color='r', linestyle='--', label='Q2 tension median')
            plt.legend()
        plt.title('Tension proxy distribution')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'tension_distribution.png'))
        plt.close()
    
    # 3) Confusion matrix on test set
    if len(true_labels) and len(true_labels) == len(pred_labels):
        cm = confusion_matrix(true_labels, pred_labels, labels=list(range(6)))
        plt.figure(figsize=(6,5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_names, yticklabels=label_names)
        plt.ylabel('True')
        plt.xlabel('Predicted')
        plt.title('Confusion Matrix (Test)')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))
        plt.close()
    
    # 4) GAN training curves (G/D losses)
    g_loss = gan_mer.gan_history.get('g_loss', [])
    d_loss = gan_mer.gan_history.get('d_loss', [])
    if len(g_loss) and len(d_loss):
        plt.figure(figsize=(6,4))
        plt.plot(g_loss, label='G loss')
        plt.plot(d_loss, label='D loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('GAN Training Losses')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'gan_losses.png'))
        plt.close()
    
    # 5) Classifier curves (train/val)
    tr_loss = gan_mer.classifier_history.get('train_loss', [])
    tr_acc = gan_mer.classifier_history.get('train_acc', [])
    va_loss = gan_mer.classifier_history.get('val_loss', [])
    va_acc = gan_mer.classifier_history.get('val_acc', [])
    if len(tr_loss):
        plt.figure(figsize=(6,4))
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
        plt.figure(figsize=(6,4))
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


def main():
    """Main function to run the GAN-based MER system"""
    
    # Configuration
    DATASET_DIR = "EMOPIA_1.0"
    MIDI_DIR = os.path.join(DATASET_DIR, "midis")
    LABEL_FILE = os.path.join(DATASET_DIR, "label.csv")
    SAVE_DIR = "models"
    
    # Hyperparameters
    FEATURE_DIM = 210  # Increased for additional advanced features (spectral contrast, rhythm, onset features)
    LATENT_DIM = 100
    NUM_CLASSES = 6
    BATCH_SIZE = 8  # Even smaller batch size for better generalization
    GAN_EPOCHS = 40  # Increased GAN epochs for stronger generator prior to augmentation
    CLASSIFIER_EPOCHS = 120  # More epochs for better classifier training
    
    print("=== GAN-based Music Emotion Recognition System ===")
    print(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    
    # Load dataset
    print("\n1. Loading EMOPIA dataset...")
    # Adjust thresholds: significantly reduce Q1 split to keep more Q1 as Happy/Excited, not Sad
    dataset = EMOPIADataset(MIDI_DIR, LABEL_FILE, FEATURE_DIM, q1_arousal_scale=0.85, q2_tension_scale=1.05)
    
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
    
    # Initialize improved GAN MER system
    print("\n2. Initializing improved GAN MER system...")
    gan_mer = GANMER(FEATURE_DIM, LATENT_DIM, NUM_CLASSES)
    
    # Train models
    print("\n3. Training models...")
    gan_mer.train_gan(train_loader, epochs=GAN_EPOCHS)
    
    # Generate synthetic data for data augmentation
    print("\n3.1. Generating synthetic data for augmentation...")
    synthetic_features = []
    synthetic_labels = []
    
    # Generate more data for underrepresented classes (heuristic for 6 classes)
    class_samples = {
        0: 60,   # Happy
        1: 150,  # Excited (boost)
        2: 50,   # Tense
        3: 50,   # Fearful
        4: 180,  # Sad (boost)
        5: 60    # Hopeful
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
    
    # Set class-balanced weights for focal loss (inverse frequency from combined_labels)
    class_counts = np.bincount(combined_labels, minlength=NUM_CLASSES).astype(np.float32)
    class_freq = class_counts / class_counts.sum()
    # Avoid division by zero
    inv_freq = 1.0 / np.maximum(class_freq, 1e-6)
    inv_freq = inv_freq / inv_freq.sum() * NUM_CLASSES  # normalize around 1
    gan_mer.class_weights = torch.tensor(inv_freq, dtype=torch.float32)
    
    # Train classifier with adversarial augmentation
    gan_mer.train_classifier(train_loader, val_loader, epochs=CLASSIFIER_EPOCHS)
    
    # Evaluate improved classifier
    print("\n4. Evaluating improved classifier...")
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
                              target_names=['Happy', 'Excited', 'Tense', 'Fearful', 'Sad', 'Hopeful']))
    
    # Run EDA
    print("\n5. Running EDA...")
    run_eda(dataset, gan_mer, all_labels, all_predictions, save_dir="eda")
    
    # Save models
    print("\n6. Saving models...")
    gan_mer.save_models(SAVE_DIR)
    
    # Generate synthetic data for each emotion
    print("\n7. Generating synthetic data...")
    for emotion_id, emotion_name in gan_mer.emotion_mapping.items():
        synthetic_data = gan_mer.generate_synthetic_data(10, emotion_id)
        print(f"Generated {len(synthetic_data)} samples for {emotion_name}")
    
    # # Example recommendation
    print("\n7. Testing recommendation system...")
    
    # Find a sample MIDI file for testing
    sample_files = [f for f in os.listdir(MIDI_DIR) if f.endswith('.mid')]
    if sample_files:
        sample_file = os.path.join(MIDI_DIR, sample_files[0])
        print(f"Testing with sample file: {sample_files[0]}")
        
        recommendations = gan_mer.recommend_similar_tracks(sample_file, DATASET_DIR, num_recommendations=5)
        
        if recommendations:
            print("\nTop 5 recommendations:")
            for i, rec in enumerate(recommendations, 1):
                print(f"{i}. {rec['track_id']} - {rec['emotion']} (similarity: {rec['similarity_score']:.3f})")
        else:
            print("No recommendations found")
    
    print("\n=== System ready for use! ===")
    print("You can now:")
    print("1. Classify emotions from MIDI files")
    print("2. Generate synthetic music with specific emotions")
    print("3. Get emotion-based music recommendations")

if __name__ == "__main__":
    main()
    
