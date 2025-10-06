"""
Exploratory Data Analysis (EDA) for Music Emotion Analysis
"""
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
        
        # Set style for better plots
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Training history storage
        self.training_history = {
            'discriminator_loss': [],
            'generator_loss': [],
            'epochs': []
        }
    
    def analyze_dataset(self, df, title="Music Emotion Dataset"):
        """Perform comprehensive EDA on the dataset"""
        print(f"\n{'='*60}")
        print(f"EXPLORATORY DATA ANALYSIS: {title}")
        print(f"{'='*60}")
        
        # Basic dataset information
        self._basic_info(df)
        
        # Emotion distribution analysis
        self._emotion_distribution(df)
        
        # Feature analysis
        self._feature_analysis(df)
        
        # Correlation analysis
        self._correlation_analysis(df)
        
        # Statistical tests
        self._statistical_tests(df)
        
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
    
    def _emotion_distribution(self, df):
        """Analyze emotion distribution"""
        print(f"\n2. EMOTION DISTRIBUTION ANALYSIS")
        print(f"{'='*40}")
        
        emotion_counts = df['emotion'].value_counts()
        print(f"Emotion distribution:")
        for emotion, count in emotion_counts.items():
            percentage = (count / len(df)) * 100
            print(f"  {emotion}: {count} samples ({percentage:.1f}%)")
        
        # Create emotion distribution plot
        try:
            plt.figure(figsize=(12, 6))
            
            # Bar plot
            plt.subplot(1, 2, 1)
            emotion_counts.plot(kind='bar', color='skyblue', edgecolor='black')
            plt.title('Emotion Distribution', fontsize=14, fontweight='bold')
            plt.xlabel('Emotion Category', fontsize=12)
            plt.ylabel('Number of Samples', fontsize=12)
            plt.xticks(rotation=45)
            plt.grid(axis='y', alpha=0.3)
            
            # Pie chart
            plt.subplot(1, 2, 2)
            colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#ff99cc', '#c2c2f0']
            plt.pie(emotion_counts.values, labels=emotion_counts.index, autopct='%1.1f%%', 
                    colors=colors[:len(emotion_counts)], startangle=90)
            plt.title('Emotion Distribution (Percentage)', fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/emotion_distribution.png", dpi=300, bbox_inches='tight')
            plt.close()  # Close instead of show to avoid display issues
        except Exception as e:
            print(f"  Note: Could not create emotion distribution plot: {e}")
        
        # Balance analysis
        print(f"\nDataset balance analysis:")
        max_count = emotion_counts.max()
        min_count = emotion_counts.min()
        balance_ratio = min_count / max_count
        print(f"  Most frequent emotion: {emotion_counts.idxmax()} ({max_count} samples)")
        print(f"  Least frequent emotion: {emotion_counts.idxmin()} ({min_count} samples)")
        print(f"  Balance ratio: {balance_ratio:.3f} (1.0 = perfectly balanced)")
        
        if balance_ratio < 0.1:
            print(f"  WARNING: Dataset is highly imbalanced (ratio < 0.1)")
        elif balance_ratio < 0.5:
            print(f"  WARNING: Dataset is moderately imbalanced (ratio < 0.5)")
        else:
            print(f"  OK: Dataset is reasonably balanced (ratio >= 0.5)")
    
    def _feature_analysis(self, df):
        """Analyze musical features"""
        print(f"\n3. MUSICAL FEATURE ANALYSIS")
        print(f"{'='*40}")
        
        # Select numeric features
        numeric_features = df.select_dtypes(include=[np.number]).drop(['emotion'], axis=1, errors='ignore')
        
        print(f"Number of numeric features: {len(numeric_features.columns)}")
        
        # Feature statistics
        print(f"\nFeature statistics:")
        stats_df = numeric_features.describe()
        print(stats_df.round(3))
        
        # Create feature distribution plots
        try:
            n_features = len(numeric_features.columns)
            n_cols = 4
            n_rows = (n_features + n_cols - 1) // n_cols
            
            plt.figure(figsize=(16, 4 * n_rows))
            for i, feature in enumerate(numeric_features.columns):
                plt.subplot(n_rows, n_cols, i + 1)
                
                # Histogram
                plt.hist(numeric_features[feature], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
                plt.title(f'{feature}', fontsize=10, fontweight='bold')
                plt.xlabel('Value', fontsize=8)
                plt.ylabel('Frequency', fontsize=8)
                plt.grid(alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/feature_distributions.png", dpi=300, bbox_inches='tight')
            plt.close()  # Close instead of show
        except Exception as e:
            print(f"  Note: Could not create feature distribution plots: {e}")
        
        # Feature correlation with emotions
        print(f"\nFeature correlation with emotions:")
        emotion_corr = {}
        for emotion in df['emotion'].unique():
            if emotion != 'unknown':
                emotion_data = df[df['emotion'] == emotion]
                correlations = {}
                for feature in numeric_features.columns:
                    if feature in emotion_data.columns:
                        # Calculate mean feature value for this emotion
                        mean_value = emotion_data[feature].mean()
                        correlations[feature] = mean_value
                emotion_corr[emotion] = correlations
        
        # Show top correlated features for each emotion
        for emotion, corrs in emotion_corr.items():
            if corrs:
                top_features = sorted(corrs.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
                print(f"  {emotion}: {[f'{feat}({corr:.3f})' for feat, corr in top_features]}")
    
    def _correlation_analysis(self, df):
        """Analyze feature correlations"""
        print(f"\n4. FEATURE CORRELATION ANALYSIS")
        print(f"{'='*40}")
        
        # Select numeric features
        numeric_features = df.select_dtypes(include=[np.number]).drop(['emotion'], axis=1, errors='ignore')
        
        # Calculate correlation matrix
        correlation_matrix = numeric_features.corr()
        
        # Create correlation heatmap
        plt.figure(figsize=(14, 12))
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
        plt.title('Feature Correlation Matrix', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/correlation_heatmap.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        # Find highly correlated features
        high_corr_pairs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_val = correlation_matrix.iloc[i, j]
                if abs(corr_val) > 0.7:  # High correlation threshold
                    high_corr_pairs.append((
                        correlation_matrix.columns[i],
                        correlation_matrix.columns[j],
                        corr_val
                    ))
        
        if high_corr_pairs:
            print(f"\nHighly correlated feature pairs (|r| > 0.7):")
            for feat1, feat2, corr in sorted(high_corr_pairs, key=lambda x: abs(x[2]), reverse=True):
                print(f"  {feat1} <-> {feat2}: {corr:.3f}")
        else:
            print(f"\nNo highly correlated features found (|r| > 0.7)")
    
    def _statistical_tests(self, df):
        """Perform statistical tests"""
        print(f"\n5. STATISTICAL TESTS")
        print(f"{'='*40}")
        
        # ANOVA test for feature differences across emotions
        numeric_features = df.select_dtypes(include=[np.number]).drop(['emotion'], axis=1, errors='ignore')
        
        print(f"ANOVA tests for feature differences across emotions:")
        anova_results = {}
        
        for feature in numeric_features.columns:
            if feature in df.columns:
                # Prepare data for ANOVA
                emotion_groups = []
                for emotion in df['emotion'].unique():
                    if emotion != 'unknown':
                        group_data = df[df['emotion'] == emotion][feature].dropna()
                        if len(group_data) > 0:
                            emotion_groups.append(group_data)
                
                if len(emotion_groups) >= 2:
                    # Perform ANOVA
                    f_stat, p_value = stats.f_oneway(*emotion_groups)
                    anova_results[feature] = {'f_stat': f_stat, 'p_value': p_value}
        
        # Sort by p-value (most significant first)
        sorted_results = sorted(anova_results.items(), key=lambda x: x[1]['p_value'])
        
        print(f"\nTop 10 most significant features (ANOVA):")
        for i, (feature, results) in enumerate(sorted_results[:10]):
            significance = "***" if results['p_value'] < 0.001 else "**" if results['p_value'] < 0.01 else "*" if results['p_value'] < 0.05 else ""
            print(f"  {i+1:2d}. {feature}: F={results['f_stat']:.3f}, p={results['p_value']:.6f} {significance}")
        
        # Save detailed results
        anova_df = pd.DataFrame([
            {'feature': feat, 'f_statistic': res['f_stat'], 'p_value': res['p_value']}
            for feat, res in anova_results.items()
        ]).sort_values('p_value')
        
        anova_df.to_csv(f"{self.output_dir}/anova_results.csv", index=False)
        print(f"\nDetailed ANOVA results saved to: {self.output_dir}/anova_results.csv")
    
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
        early_phase = progress_epochs[:total_epochs//3]
        mid_phase = progress_epochs[total_epochs//3:2*total_epochs//3]
        late_phase = progress_epochs[2*total_epochs//3:]
        
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
    
    def visualize_model_architecture(self, gan_model):
        """Visualize GAN model architecture"""
        print(f"\n8. MODEL ARCHITECTURE VISUALIZATION")
        print(f"{'='*50}")
        
        try:
            # Create model architecture visualization
            fig, axes = plt.subplots(1, 2, figsize=(16, 8))
            fig.suptitle('GAN Model Architecture', fontsize=16, fontweight='bold')
            
            # Generator architecture
            generator_layers = []
            generator_params = []
            
            for layer in gan_model.generator.layers:
                layer_type = type(layer).__name__
                generator_layers.append(layer_type)
                if hasattr(layer, 'units'):
                    generator_params.append(layer.units)
                else:
                    generator_params.append('N/A')
            
            # Discriminator architecture
            discriminator_layers = []
            discriminator_params = []
            
            for layer in gan_model.discriminator.layers:
                layer_type = type(layer).__name__
                discriminator_layers.append(layer_type)
                if hasattr(layer, 'units'):
                    discriminator_params.append(layer.units)
                else:
                    discriminator_params.append('N/A')
            
            # Plot generator
            y_pos = np.arange(len(generator_layers))
            axes[0].barh(y_pos, [1]*len(generator_layers), color='lightblue', alpha=0.7)
            axes[0].set_yticks(y_pos)
            axes[0].set_yticklabels([f'{layer}\n({params})' for layer, params in zip(generator_layers, generator_params)])
            axes[0].set_xlabel('Layer')
            axes[0].set_title('Generator Architecture', fontweight='bold')
            axes[0].invert_yaxis()
            
            # Plot discriminator
            y_pos = np.arange(len(discriminator_layers))
            axes[1].barh(y_pos, [1]*len(discriminator_layers), color='lightcoral', alpha=0.7)
            axes[1].set_yticks(y_pos)
            axes[1].set_yticklabels([f'{layer}\n({params})' for layer, params in zip(discriminator_layers, discriminator_params)])
            axes[1].set_xlabel('Layer')
            axes[1].set_title('Discriminator Architecture', fontweight='bold')
            axes[1].invert_yaxis()
            
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/model_architecture.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Model architecture visualization saved to: {self.output_dir}/model_architecture.png")
            
        except Exception as e:
            print(f"Could not create model architecture visualization: {e}")
    
    def visualize_feature_importance(self, df):
        """Visualize feature importance for emotion classification"""
        print(f"\n9. FEATURE IMPORTANCE ANALYSIS")
        print(f"{'='*50}")
        
        try:
            # Calculate feature importance using variance analysis
            numeric_features = df.select_dtypes(include=[np.number]).drop(['emotion'], axis=1, errors='ignore')
            
            # Calculate feature variance across emotions
            feature_importance = {}
            for feature in numeric_features.columns:
                if feature in df.columns:
                    # Calculate variance within each emotion
                    within_variance = 0
                    between_variance = 0
                    
                    emotion_means = []
                    emotion_vars = []
                    
                    for emotion in df['emotion'].unique():
                        if emotion != 'unknown':
                            emotion_data = df[df['emotion'] == emotion][feature]
                            if len(emotion_data) > 0:
                                emotion_means.append(emotion_data.mean())
                                emotion_vars.append(emotion_data.var())
                                within_variance += emotion_data.var() * len(emotion_data)
                    
                    if len(emotion_means) > 1:
                        # Calculate between-group variance
                        overall_mean = df[feature].mean()
                        between_variance = sum([(mean - overall_mean)**2 for mean in emotion_means])
                        
                        # F-ratio as importance measure
                        if within_variance > 0:
                            f_ratio = between_variance / (within_variance / len(df))
                            feature_importance[feature] = f_ratio
                        else:
                            feature_importance[feature] = 0
                    else:
                        feature_importance[feature] = 0
            
            # Sort features by importance
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            
            # Create feature importance visualization
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('Feature Importance Analysis', fontsize=16, fontweight='bold')
            
            # 1. Top 15 most important features
            top_features = sorted_features[:15]
            feature_names = [f[0] for f in top_features]
            importance_values = [f[1] for f in top_features]
            
            bars = axes[0, 0].barh(range(len(feature_names)), importance_values, color='skyblue')
            axes[0, 0].set_yticks(range(len(feature_names)))
            axes[0, 0].set_yticklabels(feature_names)
            axes[0, 0].set_xlabel('F-Ratio (Importance)')
            axes[0, 0].set_title('Top 15 Most Important Features', fontweight='bold')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Add value labels
            for i, (bar, value) in enumerate(zip(bars, importance_values)):
                axes[0, 0].text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                               f'{value:.2f}', ha='left', va='center', fontsize=8)
            
            # 2. Feature importance distribution
            all_importance = [f[1] for f in sorted_features]
            axes[0, 1].hist(all_importance, bins=20, color='lightcoral', alpha=0.7, edgecolor='black')
            axes[0, 1].set_xlabel('F-Ratio (Importance)')
            axes[0, 1].set_ylabel('Number of Features')
            axes[0, 1].set_title('Feature Importance Distribution', fontweight='bold')
            axes[0, 1].grid(True, alpha=0.3)
            
            # 3. Feature categories analysis
            feature_categories = {
                'Pitch': [f for f in feature_names if 'pitch' in f.lower()],
                'Velocity': [f for f in feature_names if 'velocity' in f.lower()],
                'Tempo': [f for f in feature_names if 'tempo' in f.lower()],
                'Rhythm': [f for f in feature_names if 'rhythm' in f.lower()],
                'Harmony': [f for f in feature_names if 'harmony' in f.lower() or 'chord' in f.lower()],
                'Dynamic': [f for f in feature_names if 'dynamic' in f.lower()],
                'Other': [f for f in feature_names if not any(cat in f.lower() for cat in ['pitch', 'velocity', 'tempo', 'rhythm', 'harmony', 'chord', 'dynamic'])]
            }
            
            category_counts = {cat: len(features) for cat, features in feature_categories.items() if features}
            category_importance = {}
            for cat, features in feature_categories.items():
                if features:
                    cat_importance = sum([feature_importance[f] for f in features if f in feature_importance])
                    category_importance[cat] = cat_importance
            
            # Plot category counts
            axes[1, 0].pie(category_counts.values(), labels=category_counts.keys(), autopct='%1.1f%%',
                          colors=plt.cm.Set3.colors[:len(category_counts)])
            axes[1, 0].set_title('Feature Categories Distribution', fontweight='bold')
            
            # Plot category importance
            if category_importance:
                cat_names = list(category_importance.keys())
                cat_values = list(category_importance.values())
                bars = axes[1, 1].bar(cat_names, cat_values, color='lightgreen', alpha=0.7)
                axes[1, 1].set_ylabel('Total Importance (F-Ratio)')
                axes[1, 1].set_title('Feature Categories Importance', fontweight='bold')
                axes[1, 1].tick_params(axis='x', rotation=45)
                axes[1, 1].grid(True, alpha=0.3)
                
                # Add value labels
                for bar, value in zip(bars, cat_values):
                    axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                                   f'{value:.1f}', ha='center', va='bottom', fontsize=9)
            
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/feature_importance.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Feature importance analysis saved to: {self.output_dir}/feature_importance.png")
            print(f"Top 5 most important features:")
            for i, (feature, importance) in enumerate(sorted_features[:5]):
                print(f"  {i+1}. {feature}: {importance:.3f}")
                
        except Exception as e:
            print(f"Could not create feature importance visualization: {e}")
