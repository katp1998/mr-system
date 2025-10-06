"""
Simple MIDI feature extraction for emotion classification
"""
import os
import pandas as pd
import numpy as np
import pretty_midi
from music21 import stream, pitch, key, meter, tempo

class MIDIFeatureExtractor:
    def __init__(self):
        self.features = []
    
    def extract_features(self, midi_path):
        """Extract all features from a MIDI file"""
        try:
            midi_data = pretty_midi.PrettyMIDI(midi_path)
            features = {}
            
            # Basic features
            features['file_path'] = midi_path
            features['duration'] = midi_data.get_end_time()
            
            # Pitch features
            features.update(self._extract_pitch_features(midi_data))
            
            # Velocity features  
            features.update(self._extract_velocity_features(midi_data))
            
            # Tempo features
            features.update(self._extract_tempo_features(midi_data))
            
            # Harmony features
            features.update(self._extract_harmony_features(midi_data))
            
            # Rhythm features
            features.update(self._extract_rhythm_features(midi_data))
            
            # Articulation features
            features.update(self._extract_articulation_features(midi_data))
            
            # Temporal features
            features.update(self._extract_temporal_features(midi_data))
            
            # Dynamic features
            features.update(self._extract_dynamic_features(midi_data))
            
            return features
            
        except Exception as e:
            print(f"Error processing {midi_path}: {e}")
            return None
    
    def _extract_pitch_features(self, midi_data):
        """Extract pitch-related features"""
        all_notes = []
        for instrument in midi_data.instruments:
            for note in instrument.notes:
                all_notes.append(note.pitch)
        
        if not all_notes:
            return {'pitch_min': 0, 'pitch_max': 0, 'pitch_mean': 0, 'pitch_std': 0}
        
        return {
            'pitch_min': min(all_notes),
            'pitch_max': max(all_notes),
            'pitch_mean': np.mean(all_notes),
            'pitch_std': np.std(all_notes)
        }
    
    def _extract_velocity_features(self, midi_data):
        """Extract velocity-related features"""
        all_velocities = []
        for instrument in midi_data.instruments:
            for note in instrument.notes:
                all_velocities.append(note.velocity)
        
        if not all_velocities:
            return {'velocity_min': 0, 'velocity_max': 0, 'velocity_mean': 0, 'velocity_std': 0}
        
        return {
            'velocity_min': min(all_velocities),
            'velocity_max': max(all_velocities),
            'velocity_mean': np.mean(all_velocities),
            'velocity_std': np.std(all_velocities)
        }
    
    def _extract_tempo_features(self, midi_data):
        """Extract tempo-related features"""
        tempo_changes = midi_data.get_tempo_changes()
        if len(tempo_changes[1]) == 0:
            return {'tempo_mean': 120, 'tempo_std': 0}
        
        return {
            'tempo_mean': np.mean(tempo_changes[1]),
            'tempo_std': np.std(tempo_changes[1])
        }
    
    def _extract_harmony_features(self, midi_data):
        """Extract harmony-related features"""
        # Simple chord detection
        chord_changes = 0
        for instrument in midi_data.instruments:
            if instrument.is_drum:
                continue
            notes = sorted(instrument.notes, key=lambda x: x.start)
            for i in range(1, len(notes)):
                if abs(notes[i].start - notes[i-1].start) < 0.1:  # Simultaneous notes
                    chord_changes += 1
        
        return {
            'chord_changes': chord_changes,
            'harmony_complexity': chord_changes / max(1, len(midi_data.instruments))
        }
    
    def _extract_rhythm_features(self, midi_data):
        """Extract rhythm-related features"""
        all_durations = []
        for instrument in midi_data.instruments:
            for note in instrument.notes:
                all_durations.append(note.end - note.start)
        
        if not all_durations:
            return {'rhythm_mean': 0, 'rhythm_std': 0, 'syncopation': 0}
        
        # Simple syncopation measure
        syncopation = np.std(all_durations) / np.mean(all_durations) if np.mean(all_durations) > 0 else 0
        
        return {
            'rhythm_mean': np.mean(all_durations),
            'rhythm_std': np.std(all_durations),
            'syncopation': syncopation
        }
    
    def _extract_articulation_features(self, midi_data):
        """Extract articulation features"""
        note_overlaps = 0
        total_notes = 0
        
        for instrument in midi_data.instruments:
            if instrument.is_drum:
                continue
            notes = sorted(instrument.notes, key=lambda x: x.start)
            total_notes += len(notes)
            for i in range(1, len(notes)):
                if notes[i].start < notes[i-1].end:  # Overlapping notes
                    note_overlaps += 1
        
        overlap_ratio = note_overlaps / max(1, total_notes)
        
        return {
            'note_overlap_ratio': overlap_ratio,
            'total_notes': total_notes
        }
    
    def _extract_temporal_features(self, midi_data):
        """Extract temporal features"""
        tempo_changes = midi_data.get_tempo_changes()
        tempo_variation = np.std(tempo_changes[1]) if len(tempo_changes[1]) > 1 else 0
        
        return {
            'tempo_variation': tempo_variation,
            'duration': midi_data.get_end_time()
        }
    
    def _extract_dynamic_features(self, midi_data):
        """Extract dynamic features"""
        all_velocities = []
        for instrument in midi_data.instruments:
            for note in instrument.notes:
                all_velocities.append(note.velocity)
        
        if not all_velocities:
            return {'dynamic_range': 0, 'dynamic_contrast': 0}
        
        dynamic_range = max(all_velocities) - min(all_velocities)
        dynamic_contrast = np.std(all_velocities)
        
        return {
            'dynamic_range': dynamic_range,
            'dynamic_contrast': dynamic_contrast
        }
