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
    
    # Step 1: Extract features from MIDI files
    print("\n1. Extracting features from MIDI files...")
    extractor = MIDIFeatureExtractor()
    
    features_list = []
    midi_files = [f for f in os.listdir(midi_dir) if f.endswith('.mid')]
    
    print(f"Found {len(midi_files)} MIDI files")
    
    for i, midi_file in enumerate(midi_files):  # Process all files
        if i % 100 == 0:
            print(f"Processing file {i+1}/{len(midi_files)}")
        
        midi_path = os.path.join(midi_dir, midi_file)
        features = extractor.extract_features(midi_path)
        
        if features:
            features_list.append(features)
    
    # Convert to DataFrame
    df = pd.DataFrame(features_list)
    print(f"Extracted features from {len(df)} files")
    
    # Step 2: Classify emotions
    print("\n2. Classifying emotions...")
    classifier = EmotionClassifier()
    df = classifier.classify_dataset(df)
    
    # Step 2.5: Initial EDA on classified dataset
    print("\n2.5. Exploratory Data Analysis (Classified Dataset)...")
    eda = EDAAnalyzer(output_dir)
    eda.analyze_dataset(df, "Classified EMOPIA Dataset")
    
    # Show emotion distribution
    emotion_dist = classifier.get_emotion_distribution(df)
    print("Emotion distribution:")
    print(emotion_dist)
    
    # Save initial results
    df.to_csv(f"{output_dir}/initial_features.csv", index=False)
    
    # Step 3: Balance dataset with GAN
    print("\n3. Balancing dataset with GAN...")
    
    # Prepare features for GAN
    numeric_features = df.select_dtypes(include=[np.number]).drop(['emotion'], axis=1, errors='ignore')
    feature_dim = len(numeric_features.columns)
    
    # Initialize and train GAN
    gan = SimpleGAN(feature_dim=feature_dim)
    gan.train(df, epochs=50, eda_analyzer=eda)  # Pass EDA analyzer for loss tracking
    
    # Balance dataset to match the largest category (happy = 553)
    balanced_df = gan.balance_dataset(df, target_samples_per_emotion=553)
    
    print("Balanced dataset emotion distribution:")
    print(balanced_df['emotion'].value_counts())
    
    # Save balanced dataset
    balanced_df.to_csv(f"{output_dir}/balanced_dataset.csv", index=False)
    
    # Step 3.5: EDA on balanced dataset and comparison
    print("\n3.5. Exploratory Data Analysis (Balanced Dataset)...")
    eda.analyze_dataset(balanced_df, "Balanced Dataset (After GAN)")
    eda.compare_datasets(df, balanced_df)
    
    # Step 3.6: Model training visualizations
    print("\n3.6. Model Training Visualizations...")
    eda.visualize_training()
    eda.visualize_model_architecture(gan)
    eda.visualize_feature_importance(balanced_df)
    
    # Step 4: Train recommendation system
    print("\n4. Training recommendation system...")
    rec_system = RecommendationSystem()
    rec_system.train(balanced_df)
    
    # Show system stats
    stats = rec_system.get_emotion_stats()
    print("Recommendation system stats:")
    print(stats)
    
    # Step 5: Test recommendation system
    print("\n5. Testing recommendation system...")
    
    # Find a query file (specify the file you want to test)
    if len(df) > 0:
        # Option 1: Use a specific file by name
        target_file = "Q3_veG92Oi-DlU_0.mid"  # Change this to your desired file
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
    
    print("\nSystem completed successfully!")
    print(f"Results saved in {output_dir}/ directory")

if __name__ == "__main__":
    main()
