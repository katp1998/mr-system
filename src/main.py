import os
import pandas as pd
import numpy as np
from feature_extractor import MIDIFeatureExtractor
from emotion_classifier import EmotionClassifier
from simple_gan import SimpleGAN
from recommendation_system import RecommendationSystem
from eda_analysis import EDAAnalyzer

def main():
    print("Music Emotion Analysis and Recommendation System")
    print("=" * 50)
    
    # Setup paths
    midi_dir = "../EMOPIA_1.0/midis"
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1: Extract features from MIDI files
    print("\nExtracting features from MIDI files...")
    extractor = MIDIFeatureExtractor()
    
    features_list = []
    midi_files = [f for f in os.listdir(midi_dir) if f.endswith('.mid')]
    
    print(f"Found {len(midi_files)} MIDI files")
    
    for i, midi_file in enumerate(midi_files):  # Iterate over all files
        if i % 100 == 0:
            print(f"Processing file {i+1}/{len(midi_files)}")
        
        midi_path = os.path.join(midi_dir, midi_file)
        features = extractor.extract_features(midi_path)
        
        if features:
            features_list.append(features)
    
    # Convert to DataFrame
    df = pd.DataFrame(features_list)
    print(f"Extracted features from {len(df)} files")
    
    # Classify emotions
    print("\n2. Classifying emotions...")
    classifier = EmotionClassifier()
    df = classifier.classify_dataset(df)
    
    eda = EDAAnalyzer(output_dir)
    
    # Show emotion distribution
    emotion_dist = classifier.get_emotion_distribution(df)
    print("Emotion distribution:")
    print(emotion_dist)
    
    # Save initial results
    df.to_csv(f"{output_dir}/initial_features.csv", index=False)
    
    # 3: Balance dataset with GAN
    print("\nBalancing dataset with GAN...")
    
    # Prepare features for GAN
    numeric_features = df.select_dtypes(include=[np.number]).drop(['emotion'], axis=1, errors='ignore')
    feature_dim = len(numeric_features.columns)
    
    # Initialize and train GAN
    gan = SimpleGAN(feature_dim=feature_dim)
    gan.train(df, epochs=50, eda_analyzer=eda)  # Pass EDA analyzer for loss tracking
    
    # Balance dataset to match the largest category (happy = 553)
    balanced_df = gan.balance_dataset(df, target_samples_per_emotion=553)
    
    # Save balanced dataset
    # balanced_df.to_csv(f"{output_dir}/balanced_dataset.csv", index=False)
    
    # Step 3.5: EDA on balanced dataset and comparison
    print("\nExploratory Data Analysis (Balanced Dataset)...")
    eda.analyze_dataset(balanced_df, "Balanced Dataset (After GAN)")
    eda.compare_datasets(df, balanced_df)
    
    # Step 3.6: Model training visualizations
    print("\nModel Training Visualizations...")
    eda.visualize_training()
    
    # 4: Train recommendation system
    print("\nraining recommendation system...")
    rec_system = RecommendationSystem()
    rec_system.train(balanced_df)
    
    # Show system stats
    stats = rec_system.get_emotion_stats()
    print("Recommendation system stats:")
    print(stats)
    
    # 5: Test recommendation system
    print("\nTesting recommendation system...")
    
    # Find a query file
    if len(df) > 0:
 
        target_file = "Q1_0vLPYiPN7qY_0.mid" # CHANGABLE VARIABLE!!
        query_row = df[df['file_path'].str.contains(target_file)]
        
        if len(query_row) > 0:
            query_file = query_row.iloc[0]['file_path']
            query_emotion = query_row.iloc[0]['emotion']
        else:
            # Fallback to first file if target not found
            query_file = df.iloc[0]['file_path']
            query_emotion = df.iloc[0]['emotion']
        print(f"Query file: {query_file}")
        print(f"Query emotion category: {query_emotion}")
        
        # Get recommendations
        recommendations = rec_system.find_similar(query_file, n_recommendations=5)
        
        if recommendations:
            print("\nTop 5 similar tracks:")
            correct_predictions = 0
            total_predictions = len(recommendations)
            
            for i, (file_path, similarity) in enumerate(recommendations, 1):
                # Find the emotion of the recommended file
                rec_file_row = df[df['file_path'] == file_path]
                if len(rec_file_row) > 0:
                    rec_emotion = rec_file_row.iloc[0]['emotion']
                    if rec_emotion == query_emotion:
                        correct_predictions += 1
                    print(f"{i}. {os.path.basename(str(file_path))} (similarity: {similarity:.3f}, emotion: {rec_emotion})")
                else:
                    print(f"{i}. {os.path.basename(str(file_path))} (similarity: {similarity:.3f}, emotion: unknown)")
            
            # Calculate and print test accuracy
            test_accuracy = (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0
            print(f"\nTest Accuracy: {test_accuracy:.1f}% ({correct_predictions}/{total_predictions} correct predictions)")
            print(f"Query emotion category: {query_emotion}")
        else:
            print("No recommendations found")
    
    print("\nWhoo Hoo! System completed successfully!")
    
if __name__ == "__main__":
    main()
