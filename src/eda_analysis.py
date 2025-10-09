import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

class EDAAnalyzer:
    def __init__(self, output_dir="results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Set a style for better plots
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Training history storage
        self.training_history = {
            'discriminator_loss': [],
            'generator_loss': [],
            'epochs': []
        }
    
    def analyze_dataset(self, df, title="Music Emotion Dataset"):
        """Do EDA on the dataset"""
        print(f"\n{'='*60}")
        print(f"EXPLORATORY DATA ANALYSIS: {title}")
        print(f"{'='*60}")
        
        # Basic dataset information
        self._basic_info(df)
        
        print(f"\nEDA completed! Results saved in {self.output_dir}/")
    
    def _basic_info(self, df):
        """Basic dataset information"""
        print(f"\n1. BASIC DATASET INFORMATION")
        print(f"{'='*40}")
        print(f"Total samples: {len(df)}")
        print(f"Total features: {len(df.columns)}")
        print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        print(f"Missing values: {df.isnull().sum().sum()}")
        print(f"Duplicate rows: {df.duplicated().sum()}")
        
        # Data types
        print(f"\nData types:")
        print(df.dtypes.value_counts())
    
    def compare_datasets(self, original_df, balanced_df):
        """Compare original and balanced datasets"""
        print(f"\n6. DATASET COMPARISON (Original vs Balanced)")
        print(f"{'='*50}")
        
        # Emotion distribution comparison
        original_dist = original_df['emotion'].value_counts()
        balanced_dist = balanced_df['emotion'].value_counts()
        
        comparison_df = pd.DataFrame({
            'Original': original_dist,
            'Balanced': balanced_dist
        }).fillna(0)
        
        print(f"\nEmotion distribution comparison:")
        print(comparison_df)
        
        # Create comparison plot
        plt.figure(figsize=(14, 6))
        
        # Before and after comparison
        plt.subplot(1, 2, 1)
        comparison_df.plot(kind='bar', ax=plt.gca())
        plt.title('Dataset Balance: Before vs After', fontsize=14, fontweight='bold')
        plt.xlabel('Emotion Category', fontsize=12)
        plt.ylabel('Number of Samples', fontsize=12)
        plt.xticks(rotation=45)
        plt.legend(['Original', 'Balanced'])
        plt.grid(axis='y', alpha=0.3)
        
        # Balance ratio comparison
        plt.subplot(1, 2, 2)
        original_balance = original_dist.min() / original_dist.max()
        balanced_balance = balanced_dist.min() / balanced_dist.max()
        
        balance_data = pd.DataFrame({
            'Original': [original_balance],
            'Balanced': [balanced_balance]
        })
        
        balance_data.plot(kind='bar', ax=plt.gca(), color=['red', 'green'])
        plt.title('Balance Ratio Comparison', fontsize=14, fontweight='bold')
        plt.xlabel('Dataset', fontsize=12)
        plt.ylabel('Balance Ratio', fontsize=12)
        plt.xticks([0], ['Balance Ratio'])
        plt.legend(['Original', 'Balanced'])
        plt.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/dataset_comparison.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"\nBalance improvement:")
        print(f"  Original balance ratio: {original_balance:.3f}")
        print(f"  Balanced balance ratio: {balanced_balance:.3f}")
        print(f"  Improvement: {((balanced_balance - original_balance) / original_balance * 100):.1f}%")
    
    def record_training_step(self, epoch, d_loss, g_loss):
        """Record training step for visualization"""
        self.training_history['epochs'].append(epoch)
        self.training_history['discriminator_loss'].append(d_loss)
        self.training_history['generator_loss'].append(g_loss)
    
    def visualize_training(self):
        """Create comprehensive training visualizations"""
        print(f"\n7. MODEL TRAINING VISUALIZATION")
        print(f"{'='*50}")
        
        if not self.training_history['epochs']:
            print("No training history available for visualization")
            return
        
        # Create comprehensive training plots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('GAN Training Analysis', fontsize=16, fontweight='bold')
        
        # 1. Loss curves
        axes[0, 0].plot(self.training_history['epochs'], self.training_history['discriminator_loss'], 
                       'b-', label='Discriminator Loss', linewidth=2)
        axes[0, 0].plot(self.training_history['epochs'], self.training_history['generator_loss'], 
                       'r-', label='Generator Loss', linewidth=2)
        axes[0, 0].set_title('GAN Loss Curves', fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Loss ratio analysis
        loss_ratio = [d/g if g > 0 else 0 for d, g in zip(self.training_history['discriminator_loss'], 
                                                          self.training_history['generator_loss'])]
        axes[0, 1].plot(self.training_history['epochs'], loss_ratio, 'g-', linewidth=2)
        axes[0, 1].axhline(y=1, color='k', linestyle='--', alpha=0.5, label='Balanced (D/G=1)')
        axes[0, 1].set_title('Discriminator/Generator Loss Ratio', fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('D/G Loss Ratio')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Loss stability analysis
        d_loss_diff = np.diff(self.training_history['discriminator_loss'])
        g_loss_diff = np.diff(self.training_history['generator_loss'])
        axes[0, 2].plot(self.training_history['epochs'][1:], d_loss_diff, 'b-', label='D Loss Change', alpha=0.7)
        axes[0, 2].plot(self.training_history['epochs'][1:], g_loss_diff, 'r-', label='G Loss Change', alpha=0.7)
        axes[0, 2].axhline(y=0, color='k', linestyle='-', alpha=0.3)
        axes[0, 2].set_title('Loss Change Rate', fontweight='bold')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Loss Change')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Training convergence analysis
        window_size = max(1, len(self.training_history['epochs']) // 10)
        d_loss_smooth = np.convolve(self.training_history['discriminator_loss'], 
                                   np.ones(window_size)/window_size, mode='valid')
        g_loss_smooth = np.convolve(self.training_history['generator_loss'], 
                                   np.ones(window_size)/window_size, mode='valid')
        
        axes[1, 0].plot(self.training_history['epochs'][window_size-1:], d_loss_smooth, 
                       'b-', label='D Loss (Smoothed)', linewidth=2)
        axes[1, 0].plot(self.training_history['epochs'][window_size-1:], g_loss_smooth, 
                       'r-', label='G Loss (Smoothed)', linewidth=2)
        axes[1, 0].set_title('Smoothed Loss Curves', fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Smoothed Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Training stability metrics
        d_loss_std = np.std(self.training_history['discriminator_loss'])
        g_loss_std = np.std(self.training_history['generator_loss'])
        d_loss_mean = np.mean(self.training_history['discriminator_loss'])
        g_loss_mean = np.mean(self.training_history['generator_loss'])
        
        stability_metrics = ['D Loss Mean', 'G Loss Mean', 'D Loss Std', 'G Loss Std']
        stability_values = [d_loss_mean, g_loss_mean, d_loss_std, g_loss_std]
        
        bars = axes[1, 1].bar(stability_metrics, stability_values, 
                             color=['skyblue', 'lightcoral', 'lightblue', 'lightpink'])
        axes[1, 1].set_title('Training Stability Metrics', fontweight='bold')
        axes[1, 1].set_ylabel('Value')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, stability_values):
            axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom', fontsize=9)
        
        # 6. Training progress analysis
        progress_epochs = np.array(self.training_history['epochs'])
        progress_d_loss = np.array(self.training_history['discriminator_loss'])
        progress_g_loss = np.array(self.training_history['generator_loss'])
        
        # Calculate training phases
        total_epochs = len(progress_epochs)
        # early_phase = progress_epochs[:total_epochs//3]
        # mid_phase = progress_epochs[total_epochs//3:2*total_epochs//3]
        # late_phase = progress_epochs[2*total_epochs//3:]
        
        phases = ['Early', 'Mid', 'Late']
        d_loss_phases = [
            np.mean(progress_d_loss[:total_epochs//3]),
            np.mean(progress_d_loss[total_epochs//3:2*total_epochs//3]),
            np.mean(progress_d_loss[2*total_epochs//3:])
        ]
        g_loss_phases = [
            np.mean(progress_g_loss[:total_epochs//3]),
            np.mean(progress_g_loss[total_epochs//3:2*total_epochs//3]),
            np.mean(progress_g_loss[2*total_epochs//3:])
        ]
        
        x = np.arange(len(phases))
        width = 0.35
        
        axes[1, 2].bar(x - width/2, d_loss_phases, width, label='Discriminator', color='skyblue')
        axes[1, 2].bar(x + width/2, g_loss_phases, width, label='Generator', color='lightcoral')
        axes[1, 2].set_title('Training Phases Analysis', fontweight='bold')
        axes[1, 2].set_xlabel('Training Phase')
        axes[1, 2].set_ylabel('Average Loss')
        axes[1, 2].set_xticks(x)
        axes[1, 2].set_xticklabels(phases)
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/gan_training_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Print training analysis summary
        print(f"\nTraining Analysis Summary:")
        print(f"  Total epochs: {len(self.training_history['epochs'])}")
        print(f"  Final D loss: {self.training_history['discriminator_loss'][-1]:.4f}")
        print(f"  Final G loss: {self.training_history['generator_loss'][-1]:.4f}")
        print(f"  D loss range: {min(self.training_history['discriminator_loss']):.4f} - {max(self.training_history['discriminator_loss']):.4f}")
        print(f"  G loss range: {min(self.training_history['generator_loss']):.4f} - {max(self.training_history['generator_loss']):.4f}")
        print(f"  Training stability: D_std={d_loss_std:.4f}, G_std={g_loss_std:.4f}")
        
        # Determine training quality
        if d_loss_std < 0.1 and g_loss_std < 0.1:
            print(f"  Training quality: STABLE")
        elif d_loss_std < 0.2 and g_loss_std < 0.2:
            print(f"  Training quality: MODERATE")
        else:
            print(f"  Training quality: UNSTABLE")
        
        print(f"  Training visualization saved to: {self.output_dir}/gan_training_analysis.png")
    