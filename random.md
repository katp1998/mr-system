Track-1-Q1__kJtgm1OUNA_0
1. Survey Results
Happy: 23.3%
Sad:0%
Hopeful: 33.3%
Fearful: 10%
Tense: 13.3%
Excited: 20%

2. System Results:
Music Emotion Analysis and Recommendation System
==================================================

1. Extracting features from MIDI files...
Found 1078 MIDI files
Processing file 1/1078
Processing file 101/1078
Processing file 201/1078
Processing file 301/1078
Processing file 401/1078
Processing file 501/1078
Processing file 601/1078
Processing file 701/1078
Processing file 801/1078
Processing file 901/1078
Processing file 1001/1078
Extracted features from 1078 files

2. Classifying emotions...

2.5. Exploratory Data Analysis (Classified Dataset)...

============================================================
EXPLORATORY DATA ANALYSIS: Classified EMOPIA Dataset
============================================================

1. BASIC DATASET INFORMATION
========================================
Total samples: 1078
Total features: 23
Memory usage: 0.42 MB
Missing values: 0
Duplicate rows: 0

Data types:
float64    13
int64       8
object      2
Name: count, dtype: int64

2. EMOTION DISTRIBUTION ANALYSIS
========================================
Emotion distribution:
  happy: 553 samples (51.3%)
  sad: 338 samples (31.4%)
  hopeful: 174 samples (16.1%)
  tense: 13 samples (1.2%)

Dataset balance analysis:
  Most frequent emotion: happy (553 samples)
  Least frequent emotion: tense (13 samples)
  Balance ratio: 0.024 (1.0 = perfectly balanced)
  WARNING: Dataset is highly imbalanced (ratio < 0.1)

3. MUSICAL FEATURE ANALYSIS
========================================
Number of numeric features: 21

Feature statistics:
       duration  pitch_min  pitch_max  pitch_mean  ...  total_notes  tempo_variation  dynamic_range  dynamic_contrast
count  1078.000   1078.000   1078.000    1078.000  ...     1078.000           1078.0       1078.000          1078.000
mean     39.924     34.680     85.633      61.822  ...      264.970              0.0         61.499            11.939
std      10.475      8.154      8.248       5.755  ...      146.962              0.0         11.879             2.714
min      18.009     22.000     63.000      41.353  ...       41.000              0.0         28.000             4.896
25%      29.990     28.000     80.000      58.047  ...      162.000              0.0         53.000            10.017
50%      39.584     34.500     85.000      61.610  ...      229.000              0.0         62.000            11.762
75%      46.725     40.000     91.000      65.203  ...      339.000              0.0         69.000            13.607
max     109.990     69.000    105.000      85.422  ...     1228.000              0.0         99.000            24.903

[8 rows x 21 columns]

Feature correlation with emotions:
  happy: ['total_notes(266.995)', 'chord_changes(145.190)', 'harmony_complexity(145.190)']
  sad: ['total_notes(235.935)', 'tempo_mean(120.000)', 'chord_changes(115.003)']
  hopeful: ['total_notes(305.592)', 'chord_changes(170.138)', 'harmony_complexity(170.138)']
  tense: ['total_notes(390.077)', 'chord_changes(247.077)', 'harmony_complexity(247.077)']

4. FEATURE CORRELATION ANALYSIS
========================================

Highly correlated feature pairs (|r| > 0.7):
  velocity_std <-> dynamic_contrast: 1.000
  chord_changes <-> harmony_complexity: 1.000
  chord_changes <-> total_notes: 0.952
  harmony_complexity <-> total_notes: 0.952
  velocity_max <-> velocity_mean: 0.892
  rhythm_mean <-> rhythm_std: 0.821
  velocity_min <-> velocity_mean: 0.781

5. STATISTICAL TESTS
========================================
ANOVA tests for feature differences across emotions:
C:\Users\perer\AppData\Local\Programs\Python\Python311\Lib\site-packages\scipy\stats\_stats_py.py:4167: ConstantInputWarning: Each of the input arrays is constant;the F statistic is not defined or infinite
  warnings.warn(stats.ConstantInputWarning(msg))

Top 10 most significant features (ANOVA):
   1. pitch_mean: F=133.385, p=0.000000 ***
   2. pitch_max: F=100.888, p=0.000000 ***
   3. velocity_mean: F=73.824, p=0.000000 ***
   4. velocity_max: F=61.414, p=0.000000 ***
   5. note_overlap_ratio: F=42.039, p=0.000000 ***
   6. velocity_min: F=36.637, p=0.000000 ***
   7. pitch_std: F=28.778, p=0.000000 ***
   8. syncopation: F=24.819, p=0.000000 ***
   9. chord_changes: F=14.426, p=0.000000 ***
  10. harmony_complexity: F=14.426, p=0.000000 ***

Detailed ANOVA results saved to: results/anova_results.csv

EDA completed! Results saved in results/
Emotion distribution:
emotion
happy      553
sad        338
hopeful    174
tense       13
Name: count, dtype: int64

3. Balancing dataset with GAN...
Epoch 0, D Loss: 0.7047, G Loss: 0.7003
Epoch 20, D Loss: 0.8286, G Loss: 0.4918
Epoch 40, D Loss: 0.8374, G Loss: 0.4866
Generating 215 samples for sad
Generating 379 samples for hopeful
Generating 540 samples for tense
Balanced dataset emotion distribution:
emotion
happy      553
sad        553
hopeful    553
tense      553
Name: count, dtype: int64

3.5. Exploratory Data Analysis (Balanced Dataset)...

============================================================
EXPLORATORY DATA ANALYSIS: Balanced Dataset (After GAN)
============================================================

1. BASIC DATASET INFORMATION
========================================
Total samples: 2212
Total features: 23
Memory usage: 0.70 MB
Missing values: 1134
Duplicate rows: 0

Data types:
float64    21
object      2
Name: count, dtype: int64

2. EMOTION DISTRIBUTION ANALYSIS
========================================
Emotion distribution:
  happy: 553 samples (25.0%)
  sad: 553 samples (25.0%)
  hopeful: 553 samples (25.0%)
  tense: 553 samples (25.0%)

Dataset balance analysis:
  Most frequent emotion: happy (553 samples)
  Least frequent emotion: happy (553 samples)
  Balance ratio: 1.000 (1.0 = perfectly balanced)
  OK: Dataset is reasonably balanced (ratio >= 0.5)

3. MUSICAL FEATURE ANALYSIS
========================================
Number of numeric features: 21

Feature statistics:
       duration  pitch_min  pitch_max  pitch_mean  ...  total_notes  tempo_variation  dynamic_range  dynamic_contrast
count  2212.000   2212.000   2212.000    2212.000  ...     2212.000         2212.000       2212.000          2212.000
mean     35.526     31.945     80.998      61.400  ...      229.054            0.495         70.206            12.942
std       8.853      6.960      7.607       5.305  ...      121.697            0.483         12.654             2.166
min      18.009     22.000     63.000      41.353  ...       41.000            0.000         28.000             4.896
25%      29.990     25.448     74.845      56.787  ...      129.197            0.000         62.000            11.872
50%      34.537     31.956     80.026      60.801  ...      223.500            0.894         72.000            13.369
75%      39.419     35.391     84.000      66.271  ...      275.625            0.974         84.665            14.386
max     109.990     69.000    105.000      85.422  ...     1228.000            0.998         99.000            24.903

[8 rows x 21 columns]

Feature correlation with emotions:
  happy: ['total_notes(266.995)', 'chord_changes(145.190)', 'harmony_complexity(145.190)']
  sad: ['total_notes(186.148)', 'harmony_complexity(144.005)', 'tempo_mean(119.612)']
  hopeful: ['harmony_complexity(248.666)', 'total_notes(184.802)', 'tempo_mean(119.316)']
  tense: ['harmony_complexity(320.462)', 'total_notes(278.269)', 'chord_changes(166.885)']

4. FEATURE CORRELATION ANALYSIS
========================================

Highly correlated feature pairs (|r| > 0.7):
  tempo_mean <-> tempo_std: 1.000
  velocity_std <-> dynamic_contrast: 0.999
  tempo_std <-> tempo_variation: -0.999
  tempo_mean <-> tempo_variation: -0.999
  chord_changes <-> total_notes: 0.962
  rhythm_mean <-> rhythm_std: 0.897
  velocity_max <-> velocity_mean: 0.809
  pitch_std <-> velocity_mean: 0.798
  harmony_complexity <-> rhythm_mean: -0.746
  harmony_complexity <-> syncopation: 0.743
  rhythm_mean <-> syncopation: -0.742
  velocity_mean <-> harmony_complexity: 0.733
  pitch_std <-> harmony_complexity: 0.722
  tempo_mean <-> syncopation: -0.720
  tempo_std <-> syncopation: -0.720
  syncopation <-> tempo_variation: 0.719
  pitch_min <-> pitch_std: -0.707
  harmony_complexity <-> rhythm_std: -0.705

5. STATISTICAL TESTS
========================================
ANOVA tests for feature differences across emotions:

Top 10 most significant features (ANOVA):
   1. pitch_mean: F=821.289, p=0.000000 ***
   2. velocity_max: F=983.372, p=0.000000 ***
   3. velocity_mean: F=1168.145, p=0.000000 ***
   4. tempo_mean: F=808.326, p=0.000000 ***
   5. tempo_std: F=808.467, p=0.000000 ***
   6. syncopation: F=769.390, p=0.000000 ***
   7. tempo_variation: F=808.279, p=0.000000 ***
   8. dynamic_range: F=652.720, p=0.000000 ***
   9. pitch_max: F=629.578, p=0.000000 ***
  10. harmony_complexity: F=576.785, p=0.000000 ***

Detailed ANOVA results saved to: results/anova_results.csv

EDA completed! Results saved in results/

6. DATASET COMPARISON (Original vs Balanced)
==================================================

Emotion distribution comparison:
         Original  Balanced
emotion
happy         553       553
sad           338       553
hopeful       174       553
tense          13       553

Balance improvement:
  Original balance ratio: 0.024
  Balanced balance ratio: 1.000
  Improvement: 4153.8%

3.6. Model Training Visualizations...

7. MODEL TRAINING VISUALIZATION
==================================================

Training Analysis Summary:
  Total epochs: 50
  Final D loss: 0.8358
  Final G loss: 0.4859
  D loss range: 0.7047 - 0.8407
  G loss range: 0.4857 - 0.7003
  Training stability: D_std=0.0328, G_std=0.0503
  Training quality: STABLE
  Training visualization saved to: results/gan_training_analysis.png

8. MODEL ARCHITECTURE VISUALIZATION
==================================================
Model architecture visualization saved to: results/model_architecture.png

9. FEATURE IMPORTANCE ANALYSIS
==================================================
Feature importance analysis saved to: results/feature_importance.png
Top 5 most important features:
  1. velocity_mean: 6.337
  2. velocity_max: 5.335
  3. pitch_mean: 4.455
  4. tempo_std: 4.386
  5. tempo_mean: 4.385

4. Training recommendation system...
Training recommendation system...
Trained on 2212 samples
Feature dimensions: 21
Recommendation system stats:
{'total_samples': 2212, 'emotion_distribution': {'happy': 553, 'sad': 553, 'hopeful': 553, 'tense': 553}, 'feature_dimensions': 21}  

5. Testing recommendation system...
Query file: ../EMOPIA_1.0/midis\Q1__kJtgm1OUNA_0.mid
Query emotion category: hopeful

Top 5 similar tracks:
1. Q3_FUAK5TBaNY8_2.mid (similarity: 0.891, emotion: hopeful)
2. Q3_wfXSdMsd4q8_1.mid (similarity: 0.859, emotion: hopeful)
3. Q3_wfXSdMsd4q8_4.mid (similarity: 0.833, emotion: hopeful)
4. Q4_ldCQ6N9G6Mk_0.mid (similarity: 0.801, emotion: hopeful)
5. Q3_FUAK5TBaNY8_3.mid (similarity: 0.796, emotion: hopeful)

Test Accuracy: 100.0% (5/5 correct predictions)
Query emotion category: hopeful

Whoo Hoo System completed successfully!
Results saved in results/ directory
--------------------------------------------------------------

Track-2-Q4_biROWEwkDQQ_1
1. Survey Results
Happy: 26.7%
Sad: 50%
Hopeful: 23.3%
Fearful:0%
Tense:0%
Excited:0%

2. System Results:
Music Emotion Analysis and Recommendation System
==================================================

1. Extracting features from MIDI files...
Found 1078 MIDI files
Processing file 1/1078
Processing file 101/1078
Processing file 201/1078
Processing file 301/1078
Processing file 401/1078
Processing file 501/1078
Processing file 601/1078
Processing file 701/1078
Processing file 801/1078
Processing file 901/1078
Processing file 1001/1078
Extracted features from 1078 files

2. Classifying emotions...

2.5. Exploratory Data Analysis (Classified Dataset)...

============================================================
EXPLORATORY DATA ANALYSIS: Classified EMOPIA Dataset
============================================================

1. BASIC DATASET INFORMATION
========================================
Total samples: 1078
Total features: 23
Memory usage: 0.42 MB
Missing values: 0
Duplicate rows: 0

Data types:
float64    13
int64       8
object      2
Name: count, dtype: int64

2. EMOTION DISTRIBUTION ANALYSIS
========================================
Emotion distribution:
  happy: 553 samples (51.3%)
  sad: 338 samples (31.4%)
  hopeful: 174 samples (16.1%)
  tense: 13 samples (1.2%)

Dataset balance analysis:
  Most frequent emotion: happy (553 samples)
  Least frequent emotion: tense (13 samples)
  Balance ratio: 0.024 (1.0 = perfectly balanced)
  WARNING: Dataset is highly imbalanced (ratio < 0.1)

3. MUSICAL FEATURE ANALYSIS
========================================
Number of numeric features: 21

Feature statistics:
       duration  pitch_min  ...  dynamic_range  dynamic_contrast
count  1078.000   1078.000  ...       1078.000          1078.000
mean     39.924     34.680  ...         61.499            11.939
std      10.475      8.154  ...         11.879             2.714
min      18.009     22.000  ...         28.000             4.896
25%      29.990     28.000  ...         53.000            10.017
50%      39.584     34.500  ...         62.000            11.762
75%      46.725     40.000  ...         69.000            13.607
max     109.990     69.000  ...         99.000            24.903

[8 rows x 21 columns]

Feature correlation with emotions:
  happy: ['total_notes(266.995)', 'chord_changes(145.190)', 'harmony_complexity(145.190)']
  sad: ['total_notes(235.935)', 'tempo_mean(120.000)', 'chord_changes(115.003)']
  hopeful: ['total_notes(305.592)', 'chord_changes(170.138)', 'harmony_complexity(170.138)']
  tense: ['total_notes(390.077)', 'chord_changes(247.077)', 'harmony_complexity(247.077)']

4. FEATURE CORRELATION ANALYSIS
========================================

Highly correlated feature pairs (|r| > 0.7):
  velocity_std <-> dynamic_contrast: 1.000
  chord_changes <-> harmony_complexity: 1.000
  chord_changes <-> total_notes: 0.952
  harmony_complexity <-> total_notes: 0.952
  velocity_max <-> velocity_mean: 0.892
  rhythm_mean <-> rhythm_std: 0.821
  velocity_min <-> velocity_mean: 0.781

5. STATISTICAL TESTS
========================================
ANOVA tests for feature differences across emotions:
C:\Users\perer\AppData\Local\Programs\Python\Python311\Lib\site-packages\scipy\stats\_stats_py.py:4167: ConstantInputWarning: Each of the input arrays is constant;the F statistic is not defined or infinite
  warnings.warn(stats.ConstantInputWarning(msg))

Top 10 most significant features (ANOVA):
   1. pitch_mean: F=133.385, p=0.000000 ***
   2. pitch_max: F=100.888, p=0.000000 ***
   3. velocity_mean: F=73.824, p=0.000000 ***
   4. velocity_max: F=61.414, p=0.000000 ***
   5. note_overlap_ratio: F=42.039, p=0.000000 ***
   6. velocity_min: F=36.637, p=0.000000 ***
   7. pitch_std: F=28.778, p=0.000000 ***
   8. syncopation: F=24.819, p=0.000000 ***
   9. chord_changes: F=14.426, p=0.000000 ***
  10. harmony_complexity: F=14.426, p=0.000000 ***

Detailed ANOVA results saved to: results/anova_results.csv

EDA completed! Results saved in results/
Emotion distribution:
emotion
happy      553
sad        338
hopeful    174
tense       13
Name: count, dtype: int64

3. Balancing dataset with GAN...
Epoch 0, D Loss: 0.7124, G Loss: 0.6784
Epoch 20, D Loss: 0.9111, G Loss: 0.3967
Epoch 40, D Loss: 0.9116, G Loss: 0.3903
Generating 215 samples for sad
Generating 379 samples for hopeful
Generating 540 samples for tense
Balanced dataset emotion distribution:
emotion
happy      553
sad        553
hopeful    553
tense      553
Name: count, dtype: int64

3.5. Exploratory Data Analysis (Balanced Dataset)...

============================================================
EXPLORATORY DATA ANALYSIS: Balanced Dataset (After GAN)
============================================================

1. BASIC DATASET INFORMATION
========================================
Total samples: 2212
Total features: 23
Memory usage: 0.70 MB
Missing values: 1134
Duplicate rows: 0

Data types:
float64    21
object      2
Name: count, dtype: int64

2. EMOTION DISTRIBUTION ANALYSIS
========================================
Emotion distribution:
  happy: 553 samples (25.0%)
  sad: 553 samples (25.0%)
  hopeful: 553 samples (25.0%)
  tense: 553 samples (25.0%)

Dataset balance analysis:
  Most frequent emotion: happy (553 samples)
  Least frequent emotion: happy (553 samples)
  Balance ratio: 1.000 (1.0 = perfectly balanced)
  OK: Dataset is reasonably balanced (ratio >= 0.5)

3. MUSICAL FEATURE ANALYSIS
========================================
Number of numeric features: 21

Feature statistics:
       duration  pitch_min  ...  dynamic_range  dynamic_contrast
count  2212.000   2212.000  ...       2212.000          2212.000
mean     44.800     34.601  ...         70.196            10.429
std       8.914      7.440  ...         12.647             2.414
min      18.009     22.000  ...         28.000             4.896
25%      39.752     27.708  ...         62.000             9.229
50%      46.762     35.000  ...         72.000             9.273
75%      51.699     40.814  ...         84.625            11.698
max     109.990     69.000  ...         99.000            24.903

[8 rows x 21 columns]

Feature correlation with emotions:
  happy: ['total_notes(266.995)', 'chord_changes(145.190)', 'harmony_complexity(145.190)']
  sad: ['total_notes(186.097)', 'tempo_mean(119.614)', 'pitch_max(82.841)']    
  hopeful: ['total_notes(184.679)', 'tempo_mean(119.319)', 'pitch_max(92.286)']
  tense: ['total_notes(278.187)', 'chord_changes(161.417)', 'harmony_complexity(161.266)']

4. FEATURE CORRELATION ANALYSIS
========================================

Highly correlated feature pairs (|r| > 0.7):
  chord_changes <-> harmony_complexity: 1.000
  tempo_mean <-> tempo_variation: 1.000
  harmony_complexity <-> total_notes: 0.965
  chord_changes <-> total_notes: 0.965
  velocity_max <-> velocity_mean: 0.953
  rhythm_mean <-> rhythm_std: 0.841
  velocity_min <-> velocity_mean: 0.737
  rhythm_mean <-> total_notes: -0.718

5. STATISTICAL TESTS
========================================
ANOVA tests for feature differences across emotions:

Top 10 most significant features (ANOVA):
   1. pitch_mean: F=744.761, p=0.000000 ***
   2. velocity_min: F=1055.838, p=0.000000 ***
   3. velocity_max: F=973.459, p=0.000000 ***
   4. velocity_mean: F=1270.981, p=0.000000 ***
   5. tempo_mean: F=807.969, p=0.000000 ***
   6. tempo_variation: F=807.721, p=0.000000 ***
   7. dynamic_range: F=651.969, p=0.000000 ***
   8. pitch_std: F=431.772, p=0.000000 ***
   9. pitch_min: F=398.221, p=0.000000 ***
  10. pitch_max: F=276.946, p=0.000000 ***

Detailed ANOVA results saved to: results/anova_results.csv

EDA completed! Results saved in results/

6. DATASET COMPARISON (Original vs Balanced)
==================================================

Emotion distribution comparison:
         Original  Balanced
emotion
happy         553       553
sad           338       553
hopeful       174       553
tense          13       553

Balance improvement:
  Original balance ratio: 0.024
  Balanced balance ratio: 1.000
  Improvement: 4153.8%

3.6. Model Training Visualizations...

7. MODEL TRAINING VISUALIZATION
==================================================

Training Analysis Summary:
  Total epochs: 50
  Final D loss: 0.9214
  Final G loss: 0.3892
  D loss range: 0.7124 - 0.9490
  G loss range: 0.3892 - 0.6784
  Training stability: D_std=0.0495, G_std=0.0601
  Training quality: STABLE
  Training visualization saved to: results/gan_training_analysis.png

8. MODEL ARCHITECTURE VISUALIZATION
==================================================
Model architecture visualization saved to: results/model_architecture.png

9. FEATURE IMPORTANCE ANALYSIS
==================================================
Feature importance analysis saved to: results/feature_importance.png
Top 5 most important features:
  1. velocity_mean: 6.895
  2. velocity_min: 5.728
  3. velocity_max: 5.281
  4. tempo_mean: 4.383
  5. tempo_variation: 4.382

4. Training recommendation system...
Training recommendation system...
Trained on 2212 samples
Feature dimensions: 21
Recommendation system stats:
{'total_samples': 2212, 'emotion_distribution': {'happy': 553, 'sad': 553, 'hopeful': 553, 'tense': 553}, 'feature_dimensions': 21}

5. Testing recommendation system...
Query file: ../EMOPIA_1.0/midis\Q4_biROWEwkDQQ_1.mid
Query emotion category: happy

Top 5 similar tracks:
1. Q4_HYVmgq5Y93g_0.mid (similarity: 0.860, emotion: happy)
2. Q4_V3Y9L4UOcpk_1.mid (similarity: 0.841, emotion: happy)
3. Q3_02t1bj7ZABY_2.mid (similarity: 0.835, emotion: happy)
4. Q3_Ie5koh4qvJc_14.mid (similarity: 0.812, emotion: happy)
5. Q3_Ie5koh4qvJc_15.mid (similarity: 0.810, emotion: happy)

Test Accuracy: 100.0% (5/5 correct predictions)
Query emotion category: happy

System completed successfully!
Results saved in results/ directory
--------------------------------------------------------------

Track-3-Q1_0vLPYiPN7qY_0
1. Survey Results
Happy: 30%
Sad:3.3%
Hopeful:30%
Fearful:0%
Tense:16.7%
Excited:20%

2. System Results:
Music Emotion Analysis and Recommendation System
==================================================

1. Extracting features from MIDI files...
Found 1078 MIDI files
Processing file 1/1078
Processing file 101/1078
Processing file 201/1078
Processing file 301/1078
Processing file 401/1078
Processing file 501/1078
Processing file 601/1078
Processing file 701/1078
Processing file 801/1078
Processing file 901/1078
Processing file 1001/1078
Extracted features from 1078 files

2. Classifying emotions...

2.5. Exploratory Data Analysis (Classified Dataset)...

============================================================
EXPLORATORY DATA ANALYSIS: Classified EMOPIA Dataset
============================================================

1. BASIC DATASET INFORMATION
========================================
Total samples: 1078
Total features: 23
Memory usage: 0.42 MB
Missing values: 0
Duplicate rows: 0

Data types:
float64    13
int64       8
object      2
Name: count, dtype: int64

2. EMOTION DISTRIBUTION ANALYSIS
========================================
Emotion distribution:
  happy: 553 samples (51.3%)
  sad: 338 samples (31.4%)
  hopeful: 174 samples (16.1%)
  tense: 13 samples (1.2%)

Dataset balance analysis:
  Most frequent emotion: happy (553 samples)
  Least frequent emotion: tense (13 samples)
  Balance ratio: 0.024 (1.0 = perfectly balanced)
  WARNING: Dataset is highly imbalanced (ratio < 0.1)

3. MUSICAL FEATURE ANALYSIS
========================================
Number of numeric features: 21

Feature statistics:
       duration  pitch_min  pitch_max  ...  tempo_variation  dynamic_range  dynamic_contrast
count  1078.000   1078.000   1078.000  ...           1078.0       1078.000          1078.000  
mean     39.924     34.680     85.633  ...              0.0         61.499            11.939  
std      10.475      8.154      8.248  ...              0.0         11.879             2.714  
min      18.009     22.000     63.000  ...              0.0         28.000             4.896  
25%      29.990     28.000     80.000  ...              0.0         53.000            10.017  
50%      39.584     34.500     85.000  ...              0.0         62.000            11.762  
75%      46.725     40.000     91.000  ...              0.0         69.000            13.607  
max     109.990     69.000    105.000  ...              0.0         99.000            24.903  

[8 rows x 21 columns]

Feature correlation with emotions:
  happy: ['total_notes(266.995)', 'chord_changes(145.190)', 'harmony_complexity(145.190)']    
  sad: ['total_notes(235.935)', 'tempo_mean(120.000)', 'chord_changes(115.003)']
  hopeful: ['total_notes(305.592)', 'chord_changes(170.138)', 'harmony_complexity(170.138)']  
  tense: ['total_notes(390.077)', 'chord_changes(247.077)', 'harmony_complexity(247.077)']    

4. FEATURE CORRELATION ANALYSIS
========================================

Highly correlated feature pairs (|r| > 0.7):
  velocity_std <-> dynamic_contrast: 1.000
  chord_changes <-> harmony_complexity: 1.000
  chord_changes <-> total_notes: 0.952
  harmony_complexity <-> total_notes: 0.952
  velocity_max <-> velocity_mean: 0.892
  rhythm_mean <-> rhythm_std: 0.821
  velocity_min <-> velocity_mean: 0.781

5. STATISTICAL TESTS
========================================
ANOVA tests for feature differences across emotions:
C:\Users\perer\AppData\Local\Programs\Python\Python311\Lib\site-packages\scipy\stats\_stats_py.py:4167: ConstantInputWarning: Each of the input arrays is constant;the F statistic is not defined or infinite
  warnings.warn(stats.ConstantInputWarning(msg))

Top 10 most significant features (ANOVA):
   1. pitch_mean: F=133.385, p=0.000000 ***
   2. pitch_max: F=100.888, p=0.000000 ***
   3. velocity_mean: F=73.824, p=0.000000 ***
   4. velocity_max: F=61.414, p=0.000000 ***
   5. note_overlap_ratio: F=42.039, p=0.000000 ***
   6. velocity_min: F=36.637, p=0.000000 ***
   7. pitch_std: F=28.778, p=0.000000 ***
   8. syncopation: F=24.819, p=0.000000 ***
   9. chord_changes: F=14.426, p=0.000000 ***
  10. harmony_complexity: F=14.426, p=0.000000 ***

Detailed ANOVA results saved to: results/anova_results.csv

EDA completed! Results saved in results/
Emotion distribution:
emotion
happy      553
sad        338
hopeful    174
tense       13
Name: count, dtype: int64

3. Balancing dataset with GAN...
Epoch 0, D Loss: 0.7184, G Loss: 0.6714
Epoch 20, D Loss: 0.9533, G Loss: 0.3667
Epoch 40, D Loss: 0.9596, G Loss: 0.3622
Generating 215 samples for sad
Generating 379 samples for hopeful
Generating 540 samples for tense
Balanced dataset emotion distribution:
emotion
happy      553
sad        553
hopeful    553
tense      553
Name: count, dtype: int64

3.5. Exploratory Data Analysis (Balanced Dataset)...

============================================================
EXPLORATORY DATA ANALYSIS: Balanced Dataset (After GAN)
============================================================

1. BASIC DATASET INFORMATION
========================================
Total samples: 2212
Total features: 23
Memory usage: 0.70 MB
Missing values: 1134
Duplicate rows: 0

Data types:
float64    21
object      2
Name: count, dtype: int64

2. EMOTION DISTRIBUTION ANALYSIS
========================================
Emotion distribution:
  happy: 553 samples (25.0%)
  sad: 553 samples (25.0%)
  hopeful: 553 samples (25.0%)
  tense: 553 samples (25.0%)

Dataset balance analysis:
  Most frequent emotion: happy (553 samples)
  Least frequent emotion: happy (553 samples)
  Balance ratio: 1.000 (1.0 = perfectly balanced)
  OK: Dataset is reasonably balanced (ratio >= 0.5)

3. MUSICAL FEATURE ANALYSIS
========================================
Number of numeric features: 21

Feature statistics:
       duration  pitch_min  pitch_max  ...  tempo_variation  dynamic_range  dynamic_contrast
count  2212.000   2212.000   2212.000  ...         2212.000       2212.000          2212.000  
mean     44.832     35.520     88.465  ...           -0.510         56.506            12.871  
std       8.934      7.885      6.676  ...            0.498          9.709             2.137  
min      18.009     22.000     63.000  ...           -1.000         28.000             4.896  
25%      39.752     28.482     85.000  ...           -0.997         50.316            11.872  
50%      46.794     35.500     90.218  ...           -0.980         53.019            13.273  
75%      51.796     42.000     94.539  ...            0.000         61.000            14.266  
max     109.990     69.000    105.000  ...            0.000         99.000            24.903  

[8 rows x 21 columns]

Feature correlation with emotions:
  happy: ['total_notes(266.995)', 'chord_changes(145.190)', 'harmony_complexity(145.190)']
  sad: ['total_notes(188.891)', 'chord_changes(148.476)', 'tempo_mean(119.614)']
  hopeful: ['chord_changes(261.281)', 'total_notes(191.832)', 'tempo_mean(119.320)']
  tense: ['chord_changes(332.025)', 'total_notes(285.203)', 'harmony_complexity(161.852)']    

4. FEATURE CORRELATION ANALYSIS
========================================

Highly correlated feature pairs (|r| > 0.7):
  tempo_std <-> tempo_variation: 1.000
  tempo_mean <-> tempo_std: 1.000
  tempo_mean <-> tempo_variation: 1.000
  harmony_complexity <-> total_notes: 0.964
  velocity_max <-> velocity_mean: 0.817
  velocity_std <-> dynamic_range: 0.773
  chord_changes <-> syncopation: 0.760
  chord_changes <-> rhythm_mean: -0.757
  rhythm_mean <-> syncopation: -0.742
  velocity_mean <-> chord_changes: 0.723
  tempo_mean <-> syncopation: -0.720
  syncopation <-> tempo_variation: -0.720
  tempo_std <-> syncopation: -0.720

5. STATISTICAL TESTS
========================================
ANOVA tests for feature differences across emotions:

Top 10 most significant features (ANOVA):
   1. pitch_mean: F=825.493, p=0.000000 ***
   2. velocity_max: F=965.116, p=0.000000 ***
   3. velocity_mean: F=1230.495, p=0.000000 ***
   4. tempo_mean: F=808.555, p=0.000000 ***
   5. tempo_std: F=808.264, p=0.000000 ***
   6. syncopation: F=770.576, p=0.000000 ***
   7. tempo_variation: F=808.419, p=0.000000 ***
   8. chord_changes: F=619.804, p=0.000000 ***
   9. pitch_min: F=419.297, p=0.000000 ***
  10. rhythm_mean: F=330.126, p=0.000000 ***

Detailed ANOVA results saved to: results/anova_results.csv

EDA completed! Results saved in results/

6. DATASET COMPARISON (Original vs Balanced)
==================================================

Emotion distribution comparison:
         Original  Balanced
emotion
happy         553       553
sad           338       553
hopeful       174       553
tense          13       553

Balance improvement:
  Original balance ratio: 0.024
  Balanced balance ratio: 1.000
  Improvement: 4153.8%

3.6. Model Training Visualizations...

7. MODEL TRAINING VISUALIZATION
==================================================

Training Analysis Summary:
  Total epochs: 50
  Final D loss: 0.9622
  Final G loss: 0.3616
  D loss range: 0.7184 - 0.9721
  G loss range: 0.3616 - 0.6714
  Training stability: D_std=0.0523, G_std=0.0628
  Training quality: STABLE
  Training visualization saved to: results/gan_training_analysis.png

8. MODEL ARCHITECTURE VISUALIZATION
==================================================
Model architecture visualization saved to: results/model_architecture.png

9. FEATURE IMPORTANCE ANALYSIS
==================================================
Feature importance analysis saved to: results/feature_importance.png
Top 5 most important features:
  1. velocity_mean: 6.675
  2. velocity_max: 5.236
  3. pitch_mean: 4.478
  4. tempo_mean: 4.386
  5. tempo_variation: 4.386

4. Training recommendation system...
Training recommendation system...
Trained on 2212 samples
Feature dimensions: 21
Recommendation system stats:
{'total_samples': 2212, 'emotion_distribution': {'happy': 553, 'sad': 553, 'hopeful': 553, 'tense': 553}, 'feature_dimensions': 21}

5. Testing recommendation system...
Query file: ../EMOPIA_1.0/midis\Q1_0vLPYiPN7qY_0.mid
Query emotion category: happy

Top 5 similar tracks:
1. Q4_6kRPHamGDSo_2.mid (similarity: 0.884, emotion: happy)
2. Q1_KxlqB3j0zys_1.mid (similarity: 0.864, emotion: happy)
3. Q3_REq37pDAm3A_1.mid (similarity: 0.848, emotion: happy)
4. Q1_dfNdpy8TUzA_0.mid (similarity: 0.847, emotion: happy)
5. Q4_9eMxCs-LXCE_1.mid (similarity: 0.843, emotion: happy)

Test Accuracy: 100.0% (5/5 correct predictions)
Query emotion category: happy

System completed successfully!
Results saved in results/ directory
--------------------------------------------------------------

Track-4-Q4_FyK_c-TIcCA_1
1. Survey Results
Happy:40%
Sad:10%
Hopeful:23.3%
Fearful:3.3%
Tense:13.3%
Excited:10%

2. System Results:
Music Emotion Analysis and Recommendation System
==================================================

1. Extracting features from MIDI files...
Found 1078 MIDI files
Processing file 1/1078
Processing file 101/1078
Processing file 201/1078
Processing file 301/1078
Processing file 401/1078
Processing file 501/1078
Processing file 601/1078
Processing file 701/1078
Processing file 801/1078
Processing file 901/1078
Processing file 1001/1078
Extracted features from 1078 files

2. Classifying emotions...

2.5. Exploratory Data Analysis (Classified Dataset)...

============================================================
EXPLORATORY DATA ANALYSIS: Classified EMOPIA Dataset
============================================================

1. BASIC DATASET INFORMATION
========================================
Total samples: 1078
Total features: 23
Memory usage: 0.42 MB
Missing values: 0
Duplicate rows: 0

Data types:
float64    13
int64       8
object      2
Name: count, dtype: int64

2. EMOTION DISTRIBUTION ANALYSIS
========================================
Emotion distribution:
  happy: 553 samples (51.3%)
  sad: 338 samples (31.4%)
  hopeful: 174 samples (16.1%)
  tense: 13 samples (1.2%)

Dataset balance analysis:
  Most frequent emotion: happy (553 samples)
  Least frequent emotion: tense (13 samples)
  Balance ratio: 0.024 (1.0 = perfectly balanced)
  WARNING: Dataset is highly imbalanced (ratio < 0.1)

3. MUSICAL FEATURE ANALYSIS
========================================
Number of numeric features: 21

Feature statistics:
       duration  pitch_min  pitch_max  pitch_mean  ...  total_notes  tempo_variation  dynamic_range  dynamic_contrast
count  1078.000   1078.000   1078.000    1078.000  ...     1078.000           1078.0       1078.000          1078.000
mean     39.924     34.680     85.633      61.822  ...      264.970              0.0         61.499            11.939
std      10.475      8.154      8.248       5.755  ...      146.962              0.0         11.879             2.714
min      18.009     22.000     63.000      41.353  ...       41.000              0.0         28.000             4.896
25%      29.990     28.000     80.000      58.047  ...      162.000              0.0         53.000            10.017
50%      39.584     34.500     85.000      61.610  ...      229.000              0.0         62.000            11.762
75%      46.725     40.000     91.000      65.203  ...      339.000              0.0         69.000            13.607
max     109.990     69.000    105.000      85.422  ...     1228.000              0.0         99.000            24.903

[8 rows x 21 columns]

Feature correlation with emotions:
  happy: ['total_notes(266.995)', 'chord_changes(145.190)', 'harmony_complexity(145.190)']
  sad: ['total_notes(235.935)', 'tempo_mean(120.000)', 'chord_changes(115.003)']
  hopeful: ['total_notes(305.592)', 'chord_changes(170.138)', 'harmony_complexity(170.138)']
  tense: ['total_notes(390.077)', 'chord_changes(247.077)', 'harmony_complexity(247.077)']

4. FEATURE CORRELATION ANALYSIS
========================================

Highly correlated feature pairs (|r| > 0.7):
  velocity_std <-> dynamic_contrast: 1.000
  chord_changes <-> harmony_complexity: 1.000
  chord_changes <-> total_notes: 0.952
  harmony_complexity <-> total_notes: 0.952
  velocity_max <-> velocity_mean: 0.892
  rhythm_mean <-> rhythm_std: 0.821
  velocity_min <-> velocity_mean: 0.781

5. STATISTICAL TESTS
========================================
ANOVA tests for feature differences across emotions:
C:\Users\perer\AppData\Local\Programs\Python\Python311\Lib\site-packages\scipy\stats\_stats_py.py:4167: ConstantInputWarning: Each of the input arrays is constant;the F statistic is not defined or infinite
  warnings.warn(stats.ConstantInputWarning(msg))

Top 10 most significant features (ANOVA):
   1. pitch_mean: F=133.385, p=0.000000 ***
   2. pitch_max: F=100.888, p=0.000000 ***
   3. velocity_mean: F=73.824, p=0.000000 ***
   4. velocity_max: F=61.414, p=0.000000 ***
   5. note_overlap_ratio: F=42.039, p=0.000000 ***
   6. velocity_min: F=36.637, p=0.000000 ***
   7. pitch_std: F=28.778, p=0.000000 ***
   8. syncopation: F=24.819, p=0.000000 ***
   9. chord_changes: F=14.426, p=0.000000 ***
  10. harmony_complexity: F=14.426, p=0.000000 ***

Detailed ANOVA results saved to: results/anova_results.csv

EDA completed! Results saved in results/
Emotion distribution:
emotion
happy      553
sad        338
hopeful    174
tense       13
Name: count, dtype: int64

3. Balancing dataset with GAN...
Epoch 0, D Loss: 0.7137, G Loss: 0.6835
Epoch 20, D Loss: 0.8866, G Loss: 0.4225
Epoch 40, D Loss: 0.9059, G Loss: 0.4164
Generating 215 samples for sad
Generating 379 samples for hopeful
Generating 540 samples for tense
Balanced dataset emotion distribution:
emotion
happy      553
sad        553
hopeful    553
tense      553
Name: count, dtype: int64

3.5. Exploratory Data Analysis (Balanced Dataset)...

============================================================
EXPLORATORY DATA ANALYSIS: Balanced Dataset (After GAN)
============================================================

1. BASIC DATASET INFORMATION
========================================
Total samples: 2212
Total features: 23
Memory usage: 0.70 MB
Missing values: 1134
Duplicate rows: 0

Data types:
float64    21
object      2
Name: count, dtype: int64

2. EMOTION DISTRIBUTION ANALYSIS
========================================
Emotion distribution:
  happy: 553 samples (25.0%)
  sad: 553 samples (25.0%)
  hopeful: 553 samples (25.0%)
  tense: 553 samples (25.0%)

Dataset balance analysis:
  Most frequent emotion: happy (553 samples)
  Least frequent emotion: happy (553 samples)
  Balance ratio: 1.000 (1.0 = perfectly balanced)
  OK: Dataset is reasonably balanced (ratio >= 0.5)

3. MUSICAL FEATURE ANALYSIS
========================================
Number of numeric features: 21

Feature statistics:
       duration  pitch_min  pitch_max  pitch_mean  ...  total_notes  tempo_variation  dynamic_range  dynamic_contrast
count  2212.000   2212.000   2212.000    2212.000  ...     2212.000         2212.000       2212.000          2212.000
mean     44.749     34.301     80.561      62.253  ...      230.831           -0.510         70.219            10.426
std       8.884      7.345      7.858       5.556  ...      121.097            0.498         12.663             2.416
min      18.009     22.000     63.000      41.353  ...       41.000           -1.000         28.000             4.896
25%      39.752     27.483     74.090      57.691  ...      132.285           -0.997         62.000             9.229
50%      46.714     35.000     79.659      62.006  ...      223.500           -0.982         72.000             9.271
75%      51.583     40.062     84.000      67.046  ...      278.942            0.000         84.765            11.698
max     109.990     69.000    105.000      85.422  ...     1228.000            0.000         99.000            24.903

[8 rows x 21 columns]

Feature correlation with emotions:
  happy: ['total_notes(266.995)', 'chord_changes(145.190)', 'harmony_complexity(145.190)']
  sad: ['total_notes(187.457)', 'tempo_mean(119.613)', 'harmony_complexity(104.970)']
  hopeful: ['total_notes(187.677)', 'harmony_complexity(142.634)', 'tempo_mean(119.317)']
  tense: ['total_notes(281.195)', 'harmony_complexity(221.996)', 'chord_changes(164.479)']

4. FEATURE CORRELATION ANALYSIS
========================================

Highly correlated feature pairs (|r| > 0.7):
  tempo_mean <-> tempo_std: 1.000
  tempo_std <-> tempo_variation: 1.000
  tempo_mean <-> tempo_variation: 1.000
  chord_changes <-> total_notes: 0.964
  chord_changes <-> harmony_complexity: 0.912
  harmony_complexity <-> total_notes: 0.864
  velocity_min <-> velocity_max: 0.834
  velocity_max <-> velocity_mean: 0.786
  pitch_std <-> velocity_max: 0.780
  syncopation <-> note_overlap_ratio: -0.766
  rhythm_mean <-> rhythm_std: 0.745
  rhythm_mean <-> syncopation: -0.742
  velocity_min <-> velocity_mean: 0.737
  pitch_std <-> velocity_min: 0.728
  velocity_max <-> dynamic_range: 0.727
  velocity_std <-> dynamic_contrast: 0.724
  syncopation <-> tempo_variation: -0.720
  tempo_mean <-> syncopation: -0.720
  tempo_std <-> syncopation: -0.720

5. STATISTICAL TESTS
========================================
ANOVA tests for feature differences across emotions:

Top 10 most significant features (ANOVA):
   1. pitch_mean: F=747.053, p=0.000000 ***
   2. velocity_min: F=1056.093, p=0.000000 ***
   3. velocity_max: F=867.716, p=0.000000 ***
   4. velocity_mean: F=1270.537, p=0.000000 ***
   5. tempo_mean: F=807.821, p=0.000000 ***
   6. tempo_std: F=807.564, p=0.000000 ***
   7. syncopation: F=770.138, p=0.000000 ***
   8. tempo_variation: F=808.151, p=0.000000 ***
   9. note_overlap_ratio: F=670.409, p=0.000000 ***
  10. pitch_max: F=669.945, p=0.000000 ***

Detailed ANOVA results saved to: results/anova_results.csv

EDA completed! Results saved in results/

6. DATASET COMPARISON (Original vs Balanced)
==================================================

Emotion distribution comparison:
         Original  Balanced
emotion
happy         553       553
sad           338       553
hopeful       174       553
tense          13       553

Balance improvement:
  Original balance ratio: 0.024
  Balanced balance ratio: 1.000
  Improvement: 4153.8%

3.6. Model Training Visualizations...

7. MODEL TRAINING VISUALIZATION
==================================================

Training Analysis Summary:
  Total epochs: 50
  Final D loss: 0.8811
  Final G loss: 0.4153
  D loss range: 0.7137 - 0.9059
  G loss range: 0.4151 - 0.6835
  Training stability: D_std=0.0450, G_std=0.0588
  Training quality: STABLE
  Training visualization saved to: results/gan_training_analysis.png

8. MODEL ARCHITECTURE VISUALIZATION
==================================================
Model architecture visualization saved to: results/model_architecture.png

9. FEATURE IMPORTANCE ANALYSIS
==================================================
Feature importance analysis saved to: results/feature_importance.png
Top 5 most important features:
  1. velocity_mean: 6.893
  2. velocity_min: 5.729
  3. velocity_max: 4.707
  4. tempo_variation: 4.384
  5. tempo_mean: 4.382

4. Training recommendation system...
Training recommendation system...
Trained on 2212 samples
Feature dimensions: 21
Recommendation system stats:
{'total_samples': 2212, 'emotion_distribution': {'happy': 553, 'sad': 553, 'hopeful': 553, 'tense': 553}, 'feature_dimensions': 21}  

5. Testing recommendation system...
Query file: ../EMOPIA_1.0/midis\Q4_FyK_c-TIcCA_1.mid
Query emotion category: happy

Top 5 similar tracks:
1. Q4_UpHutdJvZMI_2.mid (similarity: 0.850, emotion: happy)
2. Q3_-vAz_HTFEXs_1.mid (similarity: 0.847, emotion: happy)
3. Q4_Ie5koh4qvJc_27.mid (similarity: 0.838, emotion: happy)
4. Q3_c6CwY8Gbw0c_4.mid (similarity: 0.838, emotion: happy)
5. Q3_PLfFWFZflQU_1.mid (similarity: 0.835, emotion: happy)

Test Accuracy: 100.0% (5/5 correct predictions)
Query emotion category: happy

System completed successfully!
Results saved in results/ directory
--------------------------------------------------------------

Track-5-Q4_ADJ_PDauy-g_1
1. Survey Results
Happy:10%
Sad:10%
Hopeful:3.3%
Fearful:10%
Tense:66.7%
Excited:0%

2. System Results:
Music Emotion Analysis and Recommendation System
==================================================

1. Extracting features from MIDI files...
Found 1078 MIDI files
Processing file 1/1078
Processing file 101/1078
Processing file 201/1078
Processing file 301/1078
Processing file 401/1078
Processing file 501/1078
Processing file 601/1078
Processing file 701/1078
Processing file 801/1078
Processing file 901/1078
Processing file 1001/1078
Extracted features from 1078 files

2. Classifying emotions...

2.5. Exploratory Data Analysis (Classified Dataset)...

============================================================
EXPLORATORY DATA ANALYSIS: Classified EMOPIA Dataset
============================================================

1. BASIC DATASET INFORMATION
========================================
Total samples: 1078
Total features: 23
Memory usage: 0.42 MB
Missing values: 0
Duplicate rows: 0

Data types:
float64    13
int64       8
object      2
Name: count, dtype: int64

2. EMOTION DISTRIBUTION ANALYSIS
========================================
Emotion distribution:
  happy: 553 samples (51.3%)
  sad: 338 samples (31.4%)
  hopeful: 174 samples (16.1%)
  tense: 13 samples (1.2%)

Dataset balance analysis:
  Most frequent emotion: happy (553 samples)
  Least frequent emotion: tense (13 samples)
  Balance ratio: 0.024 (1.0 = perfectly balanced)
  WARNING: Dataset is highly imbalanced (ratio < 0.1)

3. MUSICAL FEATURE ANALYSIS
========================================
Number of numeric features: 21

Feature statistics:
       duration  pitch_min  pitch_max  ...  tempo_variation  dynamic_range  dynamic_contrast
count  1078.000   1078.000   1078.000  ...           1078.0       1078.000          1078.000
mean     39.924     34.680     85.633  ...              0.0         61.499            11.939
std      10.475      8.154      8.248  ...              0.0         11.879             2.714
min      18.009     22.000     63.000  ...              0.0         28.000             4.896
25%      29.990     28.000     80.000  ...              0.0         53.000            10.017
50%      39.584     34.500     85.000  ...              0.0         62.000            11.762
75%      46.725     40.000     91.000  ...              0.0         69.000            13.607
max     109.990     69.000    105.000  ...              0.0         99.000            24.903

[8 rows x 21 columns]

Feature correlation with emotions:
  happy: ['total_notes(266.995)', 'chord_changes(145.190)', 'harmony_complexity(145.190)']
  sad: ['total_notes(235.935)', 'tempo_mean(120.000)', 'chord_changes(115.003)']
  hopeful: ['total_notes(305.592)', 'chord_changes(170.138)', 'harmony_complexity(170.138)']
  tense: ['total_notes(390.077)', 'chord_changes(247.077)', 'harmony_complexity(247.077)']

4. FEATURE CORRELATION ANALYSIS
========================================

Highly correlated feature pairs (|r| > 0.7):
  velocity_std <-> dynamic_contrast: 1.000
  chord_changes <-> harmony_complexity: 1.000
  chord_changes <-> total_notes: 0.952
  harmony_complexity <-> total_notes: 0.952
  velocity_max <-> velocity_mean: 0.892
  rhythm_mean <-> rhythm_std: 0.821
  velocity_min <-> velocity_mean: 0.781

5. STATISTICAL TESTS
========================================
ANOVA tests for feature differences across emotions:
C:\Users\perer\AppData\Local\Programs\Python\Python311\Lib\site-packages\scipy\stats\_stats_py.py:4167: ConstantInputWarning: Each of the input arrays is constant;the F statistic is not defined or infinite
  warnings.warn(stats.ConstantInputWarning(msg))

Top 10 most significant features (ANOVA):
   1. pitch_mean: F=133.385, p=0.000000 ***
   2. pitch_max: F=100.888, p=0.000000 ***
   3. velocity_mean: F=73.824, p=0.000000 ***
   4. velocity_max: F=61.414, p=0.000000 ***
   5. note_overlap_ratio: F=42.039, p=0.000000 ***
   6. velocity_min: F=36.637, p=0.000000 ***
   7. pitch_std: F=28.778, p=0.000000 ***
   8. syncopation: F=24.819, p=0.000000 ***
   9. chord_changes: F=14.426, p=0.000000 ***
  10. harmony_complexity: F=14.426, p=0.000000 ***

Detailed ANOVA results saved to: results/anova_results.csv

EDA completed! Results saved in results/
Emotion distribution:
emotion
happy      553
sad        338
hopeful    174
tense       13
Name: count, dtype: int64

3. Balancing dataset with GAN...
Epoch 0, D Loss: 0.7601, G Loss: 0.6974
Epoch 20, D Loss: 0.8445, G Loss: 0.5044
Epoch 40, D Loss: 0.8650, G Loss: 0.4726
Generating 215 samples for sad
Generating 379 samples for hopeful
Generating 540 samples for tense
Balanced dataset emotion distribution:
emotion
happy      553
sad        553
hopeful    553
tense      553
Name: count, dtype: int64

3.5. Exploratory Data Analysis (Balanced Dataset)...

============================================================
EXPLORATORY DATA ANALYSIS: Balanced Dataset (After GAN)
============================================================

1. BASIC DATASET INFORMATION
========================================
Total samples: 2212
Total features: 23
Memory usage: 0.70 MB
Missing values: 1134
Duplicate rows: 0

Data types:
float64    21
object      2
Name: count, dtype: int64

2. EMOTION DISTRIBUTION ANALYSIS
========================================
Emotion distribution:
  happy: 553 samples (25.0%)
  sad: 553 samples (25.0%)
  hopeful: 553 samples (25.0%)
  tense: 553 samples (25.0%)

Dataset balance analysis:
  Most frequent emotion: happy (553 samples)
  Least frequent emotion: happy (553 samples)
  Balance ratio: 1.000 (1.0 = perfectly balanced)
  OK: Dataset is reasonably balanced (ratio >= 0.5)

3. MUSICAL FEATURE ANALYSIS
========================================
Number of numeric features: 21

Feature statistics:
       duration  pitch_min  pitch_max  ...  tempo_variation  dynamic_range  dynamic_contrast
count  2212.000   2212.000   2212.000  ...         2212.000       2212.000          2212.000
mean     44.416     29.607     80.562  ...           -0.068         70.180            12.992
std       8.698      7.650      7.858  ...            0.155         12.634             2.190
min      18.009     22.000     63.000  ...           -0.681         28.000             4.896
25%      39.752     23.000     74.086  ...           -0.154         62.000            11.872
50%      46.344     26.528     79.664  ...            0.000         72.000            13.421
75%      50.547     34.000     84.000  ...            0.000         84.501            14.553
max     109.990     69.000    105.000  ...            0.545         99.000            24.903

[8 rows x 21 columns]

Feature correlation with emotions:
  happy: ['total_notes(266.995)', 'chord_changes(145.190)', 'harmony_complexity(145.190)']
  sad: ['total_notes(285.655)', 'harmony_complexity(148.783)', 'chord_changes(148.525)']
  hopeful: ['total_notes(426.251)', 'harmony_complexity(262.231)', 'chord_changes(261.590)']
  tense: ['total_notes(501.719)', 'harmony_complexity(332.913)', 'chord_changes(332.286)']

4. FEATURE CORRELATION ANALYSIS
========================================

Highly correlated feature pairs (|r| > 0.7):
  velocity_std <-> dynamic_contrast: 1.000
  chord_changes <-> harmony_complexity: 1.000
  tempo_mean <-> tempo_std: -1.000
  harmony_complexity <-> total_notes: 0.976
  chord_changes <-> total_notes: 0.976
  velocity_max <-> velocity_mean: 0.951
  rhythm_mean <-> rhythm_std: 0.843
  syncopation <-> total_notes: 0.763
  pitch_min <-> pitch_std: -0.763
  harmony_complexity <-> syncopation: 0.760
  chord_changes <-> syncopation: 0.759
  pitch_std <-> harmony_complexity: 0.726
  pitch_std <-> chord_changes: 0.726
  pitch_max <-> pitch_mean: 0.717
  tempo_std <-> syncopation: 0.716
  tempo_mean <-> syncopation: -0.716
  syncopation <-> note_overlap_ratio: -0.715

5. STATISTICAL TESTS
========================================
ANOVA tests for feature differences across emotions:

Top 10 most significant features (ANOVA):
   1. pitch_mean: F=986.799, p=0.000000 ***
   2. velocity_max: F=983.849, p=0.000000 ***
   3. velocity_mean: F=1299.794, p=0.000000 ***
   4. tempo_mean: F=808.092, p=0.000000 ***
   5. tempo_std: F=808.221, p=0.000000 ***
   6. syncopation: F=760.234, p=0.000000 ***
   7. pitch_max: F=670.154, p=0.000000 ***
   8. dynamic_range: F=650.246, p=0.000000 ***
   9. harmony_complexity: F=623.044, p=0.000000 ***
  10. chord_changes: F=621.010, p=0.000000 ***

Detailed ANOVA results saved to: results/anova_results.csv

EDA completed! Results saved in results/

6. DATASET COMPARISON (Original vs Balanced)
==================================================

Emotion distribution comparison:
         Original  Balanced
emotion
happy         553       553
sad           338       553
hopeful       174       553
tense          13       553

Balance improvement:
  Original balance ratio: 0.024
  Balanced balance ratio: 1.000
  Improvement: 4153.8%

3.6. Model Training Visualizations...

7. MODEL TRAINING VISUALIZATION
==================================================

Training Analysis Summary:
  Total epochs: 50
  Final D loss: 0.8983
  Final G loss: 0.4718
  D loss range: 0.7395 - 0.8987
  G loss range: 0.4712 - 0.6974
  Training stability: D_std=0.0377, G_std=0.0504
  Training quality: STABLE
  Training visualization saved to: results/gan_training_analysis.png

8. MODEL ARCHITECTURE VISUALIZATION
==================================================
Model architecture visualization saved to: results/model_architecture.png

9. FEATURE IMPORTANCE ANALYSIS
==================================================
Feature importance analysis saved to: results/feature_importance.png
Top 5 most important features:
  1. velocity_mean: 7.051
  2. pitch_mean: 5.353
  3. velocity_max: 5.337
  4. tempo_std: 4.385
  5. tempo_mean: 4.384

4. Training recommendation system...
Training recommendation system...
Trained on 2212 samples
Feature dimensions: 21
Recommendation system stats:
{'total_samples': 2212, 'emotion_distribution': {'happy': 553, 'sad': 553, 'hopeful': 553, 'tense': 553}, 'feature_dimensions': 21}

5. Testing recommendation system...
Query file: ../EMOPIA_1.0/midis\Q4_ADJ_PDauy-g_1.mid
Query emotion category: sad

Top 5 similar tracks:
1. Q4_DvIH5yUrboc_2.mid (similarity: 0.922, emotion: sad)
2. Q4_I2MrA-o5H8I_1.mid (similarity: 0.901, emotion: sad)
3. Q4_UaYYMRkHMYc_2.mid (similarity: 0.880, emotion: sad)
4. Q4_heBmBQDWj-M_1.mid (similarity: 0.872, emotion: sad)
5. Q4_I2MrA-o5H8I_2.mid (similarity: 0.868, emotion: sad)

Test Accuracy: 100.0% (5/5 correct predictions)
Query emotion category: sad

System completed successfully!
Results saved in results/ directory
--------------------------------------------------------------

Track-6-Q2_jIKX3ShvLxo_2
1. Survey Results
Happy:10%
Sad:3.3%
Hopeful:13.3%
Fearful:3.3%
Tense:43.3%
Excited:26.7%

2. System Results:
Music Emotion Analysis and Recommendation System
==================================================

1. Extracting features from MIDI files...
Found 1078 MIDI files
Processing file 1/1078
Processing file 101/1078
Processing file 201/1078
Processing file 301/1078
Processing file 401/1078
Processing file 501/1078
Processing file 601/1078
Processing file 701/1078
Processing file 801/1078
Processing file 901/1078
Processing file 1001/1078
Extracted features from 1078 files

2. Classifying emotions...

2.5. Exploratory Data Analysis (Classified Dataset)...

============================================================
EXPLORATORY DATA ANALYSIS: Classified EMOPIA Dataset
============================================================

1. BASIC DATASET INFORMATION
========================================
Total samples: 1078
Total features: 23
Memory usage: 0.42 MB
Missing values: 0
Duplicate rows: 0

Data types:
float64    13
int64       8
object      2
Name: count, dtype: int64

2. EMOTION DISTRIBUTION ANALYSIS
========================================
Emotion distribution:
  happy: 553 samples (51.3%)
  sad: 338 samples (31.4%)
  hopeful: 174 samples (16.1%)
  tense: 13 samples (1.2%)

Dataset balance analysis:
  Most frequent emotion: happy (553 samples)
  Least frequent emotion: tense (13 samples)
  Balance ratio: 0.024 (1.0 = perfectly balanced)
  WARNING: Dataset is highly imbalanced (ratio < 0.1)

3. MUSICAL FEATURE ANALYSIS
========================================
Number of numeric features: 21

Feature statistics:
       duration  pitch_min  pitch_max  ...  tempo_variation  dynamic_range  dynamic_contrast
count  1078.000   1078.000   1078.000  ...           1078.0       1078.000          1078.000
mean     39.924     34.680     85.633  ...              0.0         61.499            11.939
std      10.475      8.154      8.248  ...              0.0         11.879             2.714
min      18.009     22.000     63.000  ...              0.0         28.000             4.896
25%      29.990     28.000     80.000  ...              0.0         53.000            10.017
50%      39.584     34.500     85.000  ...              0.0         62.000            11.762
75%      46.725     40.000     91.000  ...              0.0         69.000            13.607
max     109.990     69.000    105.000  ...              0.0         99.000            24.903

[8 rows x 21 columns]

Feature correlation with emotions:
  happy: ['total_notes(266.995)', 'chord_changes(145.190)', 'harmony_complexity(145.190)']
  sad: ['total_notes(235.935)', 'tempo_mean(120.000)', 'chord_changes(115.003)']
  hopeful: ['total_notes(305.592)', 'chord_changes(170.138)', 'harmony_complexity(170.138)']
  tense: ['total_notes(390.077)', 'chord_changes(247.077)', 'harmony_complexity(247.077)']

4. FEATURE CORRELATION ANALYSIS
========================================

Highly correlated feature pairs (|r| > 0.7):
  velocity_std <-> dynamic_contrast: 1.000
  chord_changes <-> harmony_complexity: 1.000
  chord_changes <-> total_notes: 0.952
  harmony_complexity <-> total_notes: 0.952
  velocity_max <-> velocity_mean: 0.892
  rhythm_mean <-> rhythm_std: 0.821
  velocity_min <-> velocity_mean: 0.781

5. STATISTICAL TESTS
========================================
ANOVA tests for feature differences across emotions:
C:\Users\perer\AppData\Local\Programs\Python\Python311\Lib\site-packages\scipy\stats\_stats_py.py:4167: ConstantInputWarning: Each of the input arrays is constant;the F statistic is not defined or infinite
  warnings.warn(stats.ConstantInputWarning(msg))

Top 10 most significant features (ANOVA):
   1. pitch_mean: F=133.385, p=0.000000 ***
   2. pitch_max: F=100.888, p=0.000000 ***
   3. velocity_mean: F=73.824, p=0.000000 ***
   4. velocity_max: F=61.414, p=0.000000 ***
   5. note_overlap_ratio: F=42.039, p=0.000000 ***
   6. velocity_min: F=36.637, p=0.000000 ***
   7. pitch_std: F=28.778, p=0.000000 ***
   8. syncopation: F=24.819, p=0.000000 ***
   9. chord_changes: F=14.426, p=0.000000 ***
  10. harmony_complexity: F=14.426, p=0.000000 ***

Detailed ANOVA results saved to: results/anova_results.csv

EDA completed! Results saved in results/
Emotion distribution:
emotion
happy      553
sad        338
hopeful    174
tense       13
Name: count, dtype: int64

3. Balancing dataset with GAN...
Epoch 0, D Loss: 0.7016, G Loss: 0.6692
Epoch 20, D Loss: 0.8759, G Loss: 0.4313
Epoch 40, D Loss: 0.8729, G Loss: 0.4221
Generating 215 samples for sad
Generating 379 samples for hopeful
Generating 540 samples for tense
Balanced dataset emotion distribution:
emotion
happy      553
sad        553
hopeful    553
tense      553
Name: count, dtype: int64

3.5. Exploratory Data Analysis (Balanced Dataset)...

============================================================
EXPLORATORY DATA ANALYSIS: Balanced Dataset (After GAN)
============================================================

1. BASIC DATASET INFORMATION
========================================
Total samples: 2212
Total features: 23
Memory usage: 0.70 MB
Missing values: 1134
Duplicate rows: 0

Data types:
float64    21
object      2
Name: count, dtype: int64

2. EMOTION DISTRIBUTION ANALYSIS
========================================
Emotion distribution:
  happy: 553 samples (25.0%)
  sad: 553 samples (25.0%)
  hopeful: 553 samples (25.0%)
  tense: 553 samples (25.0%)

Dataset balance analysis:
  Most frequent emotion: happy (553 samples)
  Least frequent emotion: happy (553 samples)
  Balance ratio: 1.000 (1.0 = perfectly balanced)
  OK: Dataset is reasonably balanced (ratio >= 0.5)

3. MUSICAL FEATURE ANALYSIS
========================================
Number of numeric features: 21

Feature statistics:
       duration  pitch_min  pitch_max  ...  tempo_variation  dynamic_range  dynamic_contrast
count  2212.000   2212.000   2212.000  ...         2212.000       2212.000          2212.000
mean     35.519     35.542     88.458  ...           -0.156         69.994            12.982
std       8.858      7.897      6.674  ...            0.202         12.500             2.185
min      18.009     22.000     63.000  ...           -0.791         28.000             4.896
25%      29.990     28.499     85.000  ...           -0.315         62.000            11.872
50%      34.534     35.500     90.199  ...            0.000         72.000            13.411
75%      39.419     42.000     94.437  ...            0.000         83.458            14.493
max     109.990     69.000    105.000  ...            0.350         99.000            24.903

[8 rows x 21 columns]

Feature correlation with emotions:
  happy: ['total_notes(266.995)', 'chord_changes(145.190)', 'harmony_complexity(145.190)']
  sad: ['total_notes(229.883)', 'harmony_complexity(148.784)', 'tempo_mean(120.385)']
  hopeful: ['total_notes(292.068)', 'harmony_complexity(262.263)', 'tempo_mean(120.680)']
  tense: ['total_notes(376.321)', 'harmony_complexity(332.938)', 'chord_changes(161.217)']

4. FEATURE CORRELATION ANALYSIS
========================================

Highly correlated feature pairs (|r| > 0.7):
  velocity_std <-> dynamic_contrast: 1.000
  tempo_mean <-> tempo_std: -1.000
  velocity_max <-> velocity_mean: 0.944
  chord_changes <-> total_notes: 0.819
  harmony_complexity <-> total_notes: 0.817
  tempo_std <-> tempo_variation: 0.757
  tempo_mean <-> tempo_variation: -0.757
  harmony_complexity <-> rhythm_std: -0.717
  pitch_min <-> pitch_mean: 0.704

5. STATISTICAL TESTS
========================================
ANOVA tests for feature differences across emotions:

Top 10 most significant features (ANOVA):
   1. pitch_mean: F=745.386, p=0.000000 ***
   2. velocity_max: F=965.177, p=0.000000 ***
   3. velocity_mean: F=1316.745, p=0.000000 ***
   4. tempo_mean: F=808.532, p=0.000000 ***
   5. tempo_std: F=808.113, p=0.000000 ***
   6. note_overlap_ratio: F=668.086, p=0.000000 ***
   7. dynamic_range: F=633.309, p=0.000000 ***
   8. harmony_complexity: F=623.179, p=0.000000 ***
   9. pitch_min: F=419.772, p=0.000000 ***
  10. tempo_variation: F=318.358, p=0.000000 ***

Detailed ANOVA results saved to: results/anova_results.csv

EDA completed! Results saved in results/

6. DATASET COMPARISON (Original vs Balanced)
==================================================

Emotion distribution comparison:
         Original  Balanced
emotion
happy         553       553
sad           338       553
hopeful       174       553
tense          13       553

Balance improvement:
  Original balance ratio: 0.024
  Balanced balance ratio: 1.000
  Improvement: 4153.8%

3.6. Model Training Visualizations...

7. MODEL TRAINING VISUALIZATION
==================================================

Training Analysis Summary:
  Total epochs: 50
  Final D loss: 0.8854
  Final G loss: 0.4217
  D loss range: 0.7016 - 0.8898
  G loss range: 0.4212 - 0.6692
  Training stability: D_std=0.0466, G_std=0.0624
  Training quality: STABLE
  Training visualization saved to: results/gan_training_analysis.png

8. MODEL ARCHITECTURE VISUALIZATION
==================================================
Model architecture visualization saved to: results/model_architecture.png

9. FEATURE IMPORTANCE ANALYSIS
==================================================
Feature importance analysis saved to: results/feature_importance.png
Top 5 most important features:
  1. velocity_mean: 7.143
  2. velocity_max: 5.236
  3. tempo_mean: 4.386
  4. tempo_std: 4.384
  5. pitch_mean: 4.044

4. Training recommendation system...
Training recommendation system...
Trained on 2212 samples
Feature dimensions: 21
Recommendation system stats:
{'total_samples': 2212, 'emotion_distribution': {'happy': 553, 'sad': 553, 'hopeful': 553, 'tense': 553}, 'feature_dimensions': 21}

5. Testing recommendation system...
Query file: ../EMOPIA_1.0/midis\Q2_jIKX3ShvLxo_2.mid
Query emotion category: sad

Top 5 similar tracks:
1. Q2_jIKX3ShvLxo_1.mid (similarity: 0.915, emotion: sad)
2. Q1_gSwv8hZGM-s_0.mid (similarity: 0.911, emotion: sad)
3. Q2_lalnGhxT3PQ_1.mid (similarity: 0.898, emotion: sad)
4. Q1_qlbazHayULg_8.mid (similarity: 0.890, emotion: sad)
5. Q2_ItGNJM6skM4_1.mid (similarity: 0.866, emotion: sad)

Test Accuracy: 100.0% (5/5 correct predictions)
Query emotion category: sad

System completed successfully!
Results saved in results/ directory
-----------------------------------------------------------------------------------

Track-7-Q4_B3aRl8iTEKw_1
1. Survey Results
Happy:10%
Sad:43.3%
Hopeful:36.7%
Fearful:3.3%
Tense:3.3%
Excited:3.3%

2. System Results:
Music Emotion Analysis and Recommendation System
==================================================

1. Extracting features from MIDI files...
Found 1078 MIDI files
Processing file 1/1078
Processing file 101/1078
Processing file 201/1078
Processing file 301/1078
Processing file 401/1078
Processing file 501/1078
Processing file 601/1078
Processing file 701/1078
Processing file 801/1078
Processing file 901/1078
Processing file 1001/1078
Extracted features from 1078 files

2. Classifying emotions...

2.5. Exploratory Data Analysis (Classified Dataset)...

============================================================
EXPLORATORY DATA ANALYSIS: Classified EMOPIA Dataset
============================================================

1. BASIC DATASET INFORMATION
========================================
Total samples: 1078
Total features: 23
Memory usage: 0.42 MB
Missing values: 0
Duplicate rows: 0

Data types:
float64    13
int64       8
object      2
Name: count, dtype: int64

2. EMOTION DISTRIBUTION ANALYSIS
========================================
Emotion distribution:
  happy: 553 samples (51.3%)
  sad: 338 samples (31.4%)
  hopeful: 174 samples (16.1%)
  tense: 13 samples (1.2%)

Dataset balance analysis:
  Most frequent emotion: happy (553 samples)
  Least frequent emotion: tense (13 samples)
  Balance ratio: 0.024 (1.0 = perfectly balanced)
  WARNING: Dataset is highly imbalanced (ratio < 0.1)

3. MUSICAL FEATURE ANALYSIS
========================================
Number of numeric features: 21

Feature statistics:
       duration  pitch_min  pitch_max  ...  tempo_variation  dynamic_range  dynamic_contrast
count  1078.000   1078.000   1078.000  ...           1078.0       1078.000          1078.000
mean     39.924     34.680     85.633  ...              0.0         61.499            11.939
std      10.475      8.154      8.248  ...              0.0         11.879             2.714
min      18.009     22.000     63.000  ...              0.0         28.000             4.896
25%      29.990     28.000     80.000  ...              0.0         53.000            10.017
50%      39.584     34.500     85.000  ...              0.0         62.000            11.762
75%      46.725     40.000     91.000  ...              0.0         69.000            13.607
max     109.990     69.000    105.000  ...              0.0         99.000            24.903

[8 rows x 21 columns]

Feature correlation with emotions:
  happy: ['total_notes(266.995)', 'chord_changes(145.190)', 'harmony_complexity(145.190)']
  sad: ['total_notes(235.935)', 'tempo_mean(120.000)', 'chord_changes(115.003)']
  hopeful: ['total_notes(305.592)', 'chord_changes(170.138)', 'harmony_complexity(170.138)']
  tense: ['total_notes(390.077)', 'chord_changes(247.077)', 'harmony_complexity(247.077)']

4. FEATURE CORRELATION ANALYSIS
========================================

Highly correlated feature pairs (|r| > 0.7):
  velocity_std <-> dynamic_contrast: 1.000
  chord_changes <-> harmony_complexity: 1.000
  chord_changes <-> total_notes: 0.952
  harmony_complexity <-> total_notes: 0.952
  velocity_max <-> velocity_mean: 0.892
  rhythm_mean <-> rhythm_std: 0.821
  velocity_min <-> velocity_mean: 0.781

5. STATISTICAL TESTS
========================================
ANOVA tests for feature differences across emotions:
C:\Users\perer\AppData\Local\Programs\Python\Python311\Lib\site-packages\scipy\stats\_stats_py.py:4167: ConstantInputWarning: Each of the input arrays is constant;the F statistic is not defined or infinite
  warnings.warn(stats.ConstantInputWarning(msg))

Top 10 most significant features (ANOVA):
   1. pitch_mean: F=133.385, p=0.000000 ***
   2. pitch_max: F=100.888, p=0.000000 ***
   3. velocity_mean: F=73.824, p=0.000000 ***
   4. velocity_max: F=61.414, p=0.000000 ***
   5. note_overlap_ratio: F=42.039, p=0.000000 ***
   6. velocity_min: F=36.637, p=0.000000 ***
   7. pitch_std: F=28.778, p=0.000000 ***
   8. syncopation: F=24.819, p=0.000000 ***
   9. chord_changes: F=14.426, p=0.000000 ***
  10. harmony_complexity: F=14.426, p=0.000000 ***

Detailed ANOVA results saved to: results/anova_results.csv

EDA completed! Results saved in results/
Emotion distribution:
emotion
happy      553
sad        338
hopeful    174
tense       13
Name: count, dtype: int64

3. Balancing dataset with GAN...
Epoch 0, D Loss: 0.7277, G Loss: 0.6701
Epoch 20, D Loss: 0.9175, G Loss: 0.3923
Epoch 40, D Loss: 0.9279, G Loss: 0.3869
Generating 215 samples for sad
Generating 379 samples for hopeful
Generating 540 samples for tense
Balanced dataset emotion distribution:
emotion
happy      553
sad        553
hopeful    553
tense      553
Name: count, dtype: int64

3.5. Exploratory Data Analysis (Balanced Dataset)...

============================================================
EXPLORATORY DATA ANALYSIS: Balanced Dataset (After GAN)
============================================================

1. BASIC DATASET INFORMATION
========================================
Total samples: 2212
Total features: 23
Memory usage: 0.70 MB
Missing values: 1134
Duplicate rows: 0

Data types:
float64    21
object      2
Name: count, dtype: int64

2. EMOTION DISTRIBUTION ANALYSIS
========================================
Emotion distribution:
  happy: 553 samples (25.0%)
  sad: 553 samples (25.0%)
  hopeful: 553 samples (25.0%)
  tense: 553 samples (25.0%)

Dataset balance analysis:
  Most frequent emotion: happy (553 samples)
  Least frequent emotion: happy (553 samples)
  Balance ratio: 1.000 (1.0 = perfectly balanced)
  OK: Dataset is reasonably balanced (ratio >= 0.5)

3. MUSICAL FEATURE ANALYSIS
========================================
Number of numeric features: 21

Feature statistics:
       duration  pitch_min  pitch_max  ...  tempo_variation  dynamic_range  dynamic_contrast
count  2212.000   2212.000   2212.000  ...         2212.000       2212.000          2212.000
mean     44.833     29.611     80.592  ...           -0.511         55.796            12.969
std       8.934      7.648      7.839  ...            0.498         10.030             2.179
min      18.009     22.000     63.000  ...           -1.000         28.000             4.896
25%      39.752     23.000     74.139  ...           -0.998         49.182            11.872
50%      46.794     26.558     79.692  ...           -0.987         51.767            13.395
75%      51.801     34.000     84.000  ...            0.000         61.000            14.444
max     109.990     69.000    105.000  ...            0.000         99.000            24.903

[8 rows x 21 columns]

Feature correlation with emotions:
  happy: ['total_notes(266.995)', 'chord_changes(145.190)', 'harmony_complexity(145.190)']
  sad: ['total_notes(247.992)', 'harmony_complexity(148.758)', 'chord_changes(138.456)']
  hopeful: ['total_notes(334.764)', 'harmony_complexity(262.212)', 'chord_changes(234.708)']
  tense: ['total_notes(415.695)', 'harmony_complexity(332.908)', 'chord_changes(306.654)']

4. FEATURE CORRELATION ANALYSIS
========================================

Highly correlated feature pairs (|r| > 0.7):
  tempo_mean <-> tempo_std: 1.000
  tempo_mean <-> tempo_variation: -1.000
  tempo_std <-> tempo_variation: -1.000
  chord_changes <-> harmony_complexity: 0.989
  chord_changes <-> total_notes: 0.926
  harmony_complexity <-> total_notes: 0.904
  velocity_max <-> velocity_mean: 0.771
  velocity_max <-> harmony_complexity: 0.738
  velocity_std <-> dynamic_range: 0.729
  harmony_complexity <-> rhythm_std: -0.718
  velocity_max <-> chord_changes: 0.714

5. STATISTICAL TESTS
========================================
ANOVA tests for feature differences across emotions:

Top 10 most significant features (ANOVA):
   1. pitch_mean: F=754.938, p=0.000000 ***
   2. velocity_max: F=850.244, p=0.000000 ***
   3. velocity_mean: F=1271.113, p=0.000000 ***
   4. tempo_mean: F=808.354, p=0.000000 ***
   5. tempo_std: F=808.240, p=0.000000 ***
   6. tempo_variation: F=808.586, p=0.000000 ***
   7. pitch_max: F=668.073, p=0.000000 ***
   8. harmony_complexity: F=623.152, p=0.000000 ***
   9. chord_changes: F=520.822, p=0.000000 ***
  10. pitch_min: F=350.206, p=0.000000 ***

Detailed ANOVA results saved to: results/anova_results.csv

EDA completed! Results saved in results/

6. DATASET COMPARISON (Original vs Balanced)
==================================================

Emotion distribution comparison:
         Original  Balanced
emotion
happy         553       553
sad           338       553
hopeful       174       553
tense          13       553

Balance improvement:
  Original balance ratio: 0.024
  Balanced balance ratio: 1.000
  Improvement: 4153.8%

3.6. Model Training Visualizations...

7. MODEL TRAINING VISUALIZATION
==================================================

Training Analysis Summary:
  Total epochs: 50
  Final D loss: 0.9148
  Final G loss: 0.3865
  D loss range: 0.7277 - 0.9439
  G loss range: 0.3863 - 0.6701
  Training stability: D_std=0.0474, G_std=0.0586
  Training quality: STABLE
  Training visualization saved to: results/gan_training_analysis.png

8. MODEL ARCHITECTURE VISUALIZATION
==================================================
Model architecture visualization saved to: results/model_architecture.png

9. FEATURE IMPORTANCE ANALYSIS
==================================================
Feature importance analysis saved to: results/feature_importance.png
Top 5 most important features:
  1. velocity_mean: 6.896
  2. velocity_max: 4.613
  3. tempo_variation: 4.387
  4. tempo_mean: 4.385
  5. tempo_std: 4.385

4. Training recommendation system...
Training recommendation system...
Trained on 2212 samples
Feature dimensions: 21
Recommendation system stats:
{'total_samples': 2212, 'emotion_distribution': {'happy': 553, 'sad': 553, 'hopeful': 553, 'tense': 553}, 'feature_dimensions': 21}

5. Testing recommendation system...
Query file: ../EMOPIA_1.0/midis\Q4_B3aRl8iTEKw_1.mid
Query emotion category: happy

Top 5 similar tracks:
1. Q4_egYSmNuIFGk_0.mid (similarity: 0.893, emotion: happy)
2. Q4_pRuwHN-lI44_0.mid (similarity: 0.880, emotion: happy)
3. Q4_Ie5koh4qvJc_32.mid (similarity: 0.865, emotion: happy)
4. Q3_fnqb8zqHqdY_0.mid (similarity: 0.863, emotion: happy)
5. Q4_FoTXpYZXxJs_0.mid (similarity: 0.858, emotion: happy)

Test Accuracy: 100.0% (5/5 correct predictions)
Query emotion category: happy

System completed successfully!
Results saved in results/ directory
------------------------------------------------------------------------------

Track-8-Q1_6wFJhmhNeeg_0
1. Survey Results
Happy:50%
Sad:6.7%
Hopeful:26.7%
Fearful:0%
Tense:0%
Excited:16.7%

2. System Results:
Music Emotion Analysis and Recommendation System
==================================================

1. Extracting features from MIDI files...
Found 1078 MIDI files
Processing file 1/1078
Processing file 101/1078
Processing file 201/1078
Processing file 301/1078
Processing file 401/1078
Processing file 501/1078
Processing file 601/1078
Processing file 701/1078
Processing file 801/1078
Processing file 901/1078
Processing file 1001/1078
Extracted features from 1078 files

2. Classifying emotions...

2.5. Exploratory Data Analysis (Classified Dataset)...

============================================================
EXPLORATORY DATA ANALYSIS: Classified EMOPIA Dataset
============================================================

1. BASIC DATASET INFORMATION
========================================
Total samples: 1078
Total features: 23
Memory usage: 0.42 MB
Missing values: 0
Duplicate rows: 0

Data types:
float64    13
int64       8
object      2
Name: count, dtype: int64

2. EMOTION DISTRIBUTION ANALYSIS
========================================
Emotion distribution:
  happy: 553 samples (51.3%)
  sad: 338 samples (31.4%)
  hopeful: 174 samples (16.1%)
  tense: 13 samples (1.2%)

Dataset balance analysis:
  Most frequent emotion: happy (553 samples)
  Least frequent emotion: tense (13 samples)
  Balance ratio: 0.024 (1.0 = perfectly balanced)
  WARNING: Dataset is highly imbalanced (ratio < 0.1)

3. MUSICAL FEATURE ANALYSIS
========================================
Number of numeric features: 21

Feature statistics:
       duration  pitch_min  pitch_max  ...  tempo_variation  dynamic_range  dynamic_contrast
count  1078.000   1078.000   1078.000  ...           1078.0       1078.000          1078.000
mean     39.924     34.680     85.633  ...              0.0         61.499            11.939
std      10.475      8.154      8.248  ...              0.0         11.879             2.714
min      18.009     22.000     63.000  ...              0.0         28.000             4.896
25%      29.990     28.000     80.000  ...              0.0         53.000            10.017
50%      39.584     34.500     85.000  ...              0.0         62.000            11.762
75%      46.725     40.000     91.000  ...              0.0         69.000            13.607
max     109.990     69.000    105.000  ...              0.0         99.000            24.903

[8 rows x 21 columns]

Feature correlation with emotions:
  happy: ['total_notes(266.995)', 'chord_changes(145.190)', 'harmony_complexity(145.190)']
  sad: ['total_notes(235.935)', 'tempo_mean(120.000)', 'chord_changes(115.003)']
  hopeful: ['total_notes(305.592)', 'chord_changes(170.138)', 'harmony_complexity(170.138)']
  tense: ['total_notes(390.077)', 'chord_changes(247.077)', 'harmony_complexity(247.077)']

4. FEATURE CORRELATION ANALYSIS
========================================

Highly correlated feature pairs (|r| > 0.7):
  velocity_std <-> dynamic_contrast: 1.000
  chord_changes <-> harmony_complexity: 1.000
  chord_changes <-> total_notes: 0.952
  harmony_complexity <-> total_notes: 0.952
  velocity_max <-> velocity_mean: 0.892
  rhythm_mean <-> rhythm_std: 0.821
  velocity_min <-> velocity_mean: 0.781

5. STATISTICAL TESTS
========================================
ANOVA tests for feature differences across emotions:
C:\Users\perer\AppData\Local\Programs\Python\Python311\Lib\site-packages\scipy\stats\_stats_py.py:4167: ConstantInputWarning: Each of the input arrays is constant;the F statistic is not defined or infinite
  warnings.warn(stats.ConstantInputWarning(msg))

Top 10 most significant features (ANOVA):
   1. pitch_mean: F=133.385, p=0.000000 ***
   2. pitch_max: F=100.888, p=0.000000 ***
   3. velocity_mean: F=73.824, p=0.000000 ***
   4. velocity_max: F=61.414, p=0.000000 ***
   5. note_overlap_ratio: F=42.039, p=0.000000 ***
   6. velocity_min: F=36.637, p=0.000000 ***
   7. pitch_std: F=28.778, p=0.000000 ***
   8. syncopation: F=24.819, p=0.000000 ***
   9. chord_changes: F=14.426, p=0.000000 ***
  10. harmony_complexity: F=14.426, p=0.000000 ***

Detailed ANOVA results saved to: results/anova_results.csv

EDA completed! Results saved in results/
Emotion distribution:
emotion
happy      553
sad        338
hopeful    174
tense       13
Name: count, dtype: int64

3. Balancing dataset with GAN...
Epoch 0, D Loss: 0.6889, G Loss: 0.6607
Epoch 20, D Loss: 0.8841, G Loss: 0.4015
Epoch 40, D Loss: 0.8962, G Loss: 0.3961
Generating 215 samples for sad
Generating 379 samples for hopeful
Generating 540 samples for tense
Balanced dataset emotion distribution:
emotion
happy      553
sad        553
hopeful    553
tense      553
Name: count, dtype: int64

3.5. Exploratory Data Analysis (Balanced Dataset)...

============================================================
EXPLORATORY DATA ANALYSIS: Balanced Dataset (After GAN)
============================================================

1. BASIC DATASET INFORMATION
========================================
Total samples: 2212
Total features: 23
Memory usage: 0.70 MB
Missing values: 1134
Duplicate rows: 0

Data types:
float64    21
object      2
Name: count, dtype: int64

2. EMOTION DISTRIBUTION ANALYSIS
========================================
Emotion distribution:
  happy: 553 samples (25.0%)
  sad: 553 samples (25.0%)
  hopeful: 553 samples (25.0%)
  tense: 553 samples (25.0%)

Dataset balance analysis:
  Most frequent emotion: happy (553 samples)
  Least frequent emotion: happy (553 samples)
  Balance ratio: 1.000 (1.0 = perfectly balanced)
  OK: Dataset is reasonably balanced (ratio >= 0.5)

3. MUSICAL FEATURE ANALYSIS
========================================
Number of numeric features: 21

Feature statistics:
       duration  pitch_min  pitch_max  ...  tempo_variation  dynamic_range  dynamic_contrast
count  2212.000   2212.000   2212.000  ...         2212.000       2212.000          2212.000
mean     44.835     29.598     80.584  ...           -0.405         70.222            10.644
std       8.936      7.655      7.844  ...            0.405         12.665             2.293
min      18.009     22.000     63.000  ...           -0.973         28.000             4.896
25%      39.752     23.000     74.116  ...           -0.824         62.000             9.356
50%      46.795     26.455     79.678  ...           -0.490         72.000             9.681
75%      51.801     34.000     84.000  ...            0.000         84.774            11.698
max     109.990     69.000    105.000  ...            0.000         99.000            24.903

[8 rows x 21 columns]

Feature correlation with emotions:
  happy: ['total_notes(266.995)', 'chord_changes(145.190)', 'harmony_complexity(145.190)']
  sad: ['total_notes(285.496)', 'tempo_mean(120.017)', 'harmony_complexity(81.534)']
  hopeful: ['total_notes(425.799)', 'tempo_mean(120.045)', 'velocity_max(90.889)']
  tense: ['total_notes(501.356)', 'harmony_complexity(161.969)', 'chord_changes(161.701)']

4. FEATURE CORRELATION ANALYSIS
========================================

Highly correlated feature pairs (|r| > 0.7):
  chord_changes <-> harmony_complexity: 1.000
  tempo_std <-> tempo_variation: 0.977
  velocity_min <-> velocity_mean: 0.913
  rhythm_mean <-> rhythm_std: 0.844
  rhythm_mean <-> total_notes: -0.785
  velocity_max <-> velocity_mean: 0.781
  velocity_mean <-> total_notes: 0.717

5. STATISTICAL TESTS
========================================
ANOVA tests for feature differences across emotions:

Top 10 most significant features (ANOVA):
   1. pitch_mean: F=894.149, p=0.000000 ***
   2. velocity_min: F=1057.831, p=0.000000 ***
   3. velocity_max: F=965.808, p=0.000000 ***
   4. velocity_mean: F=1167.677, p=0.000000 ***
   5. tempo_std: F=808.343, p=0.000000 ***
   6. tempo_variation: F=709.336, p=0.000000 ***
   7. pitch_max: F=667.921, p=0.000000 ***
   8. note_overlap_ratio: F=666.637, p=0.000000 ***
   9. dynamic_range: F=653.797, p=0.000000 ***
  10. total_notes: F=545.276, p=0.000000 ***

Detailed ANOVA results saved to: results/anova_results.csv

EDA completed! Results saved in results/

6. DATASET COMPARISON (Original vs Balanced)
==================================================

Emotion distribution comparison:
         Original  Balanced
emotion
happy         553       553
sad           338       553
hopeful       174       553
tense          13       553

Balance improvement:
  Original balance ratio: 0.024
  Balanced balance ratio: 1.000
  Improvement: 4153.8%

3.6. Model Training Visualizations...

7. MODEL TRAINING VISUALIZATION
==================================================

Training Analysis Summary:
  Total epochs: 50
  Final D loss: 0.8888
  Final G loss: 0.3950
  D loss range: 0.6889 - 0.9027
  G loss range: 0.3950 - 0.6607
  Training stability: D_std=0.0498, G_std=0.0622
  Training quality: STABLE
  Training visualization saved to: results/gan_training_analysis.png

8. MODEL ARCHITECTURE VISUALIZATION
==================================================
Model architecture visualization saved to: results/model_architecture.png

9. FEATURE IMPORTANCE ANALYSIS
==================================================
Feature importance analysis saved to: results/feature_importance.png
Top 5 most important features:
  1. velocity_mean: 6.335
  2. velocity_min: 5.739
  3. velocity_max: 5.239
  4. pitch_mean: 4.851
  5. tempo_std: 4.385

4. Training recommendation system...
Training recommendation system...
Trained on 2212 samples
Feature dimensions: 21
Recommendation system stats:
{'total_samples': 2212, 'emotion_distribution': {'happy': 553, 'sad': 553, 'hopeful': 553, 'tense': 553}, 'feature_dimensions': 21}

5. Testing recommendation system...
Query file: ../EMOPIA_1.0/midis\Q1_6wFJhmhNeeg_0.mid
Query emotion category: tense

Top 5 similar tracks:
1. Q2_cm6E860vDjY_0.mid (similarity: 0.825, emotion: tense)
2. Q2_epX33OVpkmA_1.mid (similarity: 0.779, emotion: tense)
3. Q2_dtS02mrDMsM_2.mid (similarity: 0.719, emotion: tense)
4. Q2_UE0y8MHqT-g_2.mid (similarity: 0.689, emotion: tense)
5. Q2_JMt4m608s3k_2.mid (similarity: 0.534, emotion: tense)

Test Accuracy: 100.0% (5/5 correct predictions)
Query emotion category: tense

System completed successfully!
Results saved in results/ directory
--------------------------------------------------------------

Track-9-Q2_Dg935IbDggo_0
1. Survey Results
Happy:0%
Sad:6.7%
Hopeful:20%
Fearful:40%
Tense:23.3%
Excited:6.7%

2. System Results:
Music Emotion Analysis and Recommendation System
==================================================

1. Extracting features from MIDI files...
Found 1078 MIDI files
Processing file 1/1078
Processing file 101/1078
Processing file 201/1078
Processing file 301/1078
Processing file 401/1078
Processing file 501/1078
Processing file 601/1078
Processing file 701/1078
Processing file 801/1078
Processing file 901/1078
Processing file 1001/1078
Extracted features from 1078 files

2. Classifying emotions...

2.5. Exploratory Data Analysis (Classified Dataset)...

============================================================
EXPLORATORY DATA ANALYSIS: Classified EMOPIA Dataset
============================================================

1. BASIC DATASET INFORMATION
========================================
Total samples: 1078
Total features: 23
Memory usage: 0.42 MB
Missing values: 0
Duplicate rows: 0

Data types:
float64    13
int64       8
object      2
Name: count, dtype: int64

2. EMOTION DISTRIBUTION ANALYSIS
========================================
Emotion distribution:
  happy: 553 samples (51.3%)
  sad: 338 samples (31.4%)
  hopeful: 174 samples (16.1%)
  tense: 13 samples (1.2%)

Dataset balance analysis:
  Most frequent emotion: happy (553 samples)
  Least frequent emotion: tense (13 samples)
  Balance ratio: 0.024 (1.0 = perfectly balanced)
  WARNING: Dataset is highly imbalanced (ratio < 0.1)

3. MUSICAL FEATURE ANALYSIS
========================================
Number of numeric features: 21

Feature statistics:
       duration  pitch_min  pitch_max  ...  tempo_variation  dynamic_range  dynamic_contrast
count  1078.000   1078.000   1078.000  ...           1078.0       1078.000          1078.000
mean     39.924     34.680     85.633  ...              0.0         61.499            11.939
std      10.475      8.154      8.248  ...              0.0         11.879             2.714
min      18.009     22.000     63.000  ...              0.0         28.000             4.896
25%      29.990     28.000     80.000  ...              0.0         53.000            10.017
50%      39.584     34.500     85.000  ...              0.0         62.000            11.762
75%      46.725     40.000     91.000  ...              0.0         69.000            13.607
max     109.990     69.000    105.000  ...              0.0         99.000            24.903

[8 rows x 21 columns]

Feature correlation with emotions:
  happy: ['total_notes(266.995)', 'chord_changes(145.190)', 'harmony_complexity(145.190)']
  sad: ['total_notes(235.935)', 'tempo_mean(120.000)', 'chord_changes(115.003)']
  hopeful: ['total_notes(305.592)', 'chord_changes(170.138)', 'harmony_complexity(170.138)']
  tense: ['total_notes(390.077)', 'chord_changes(247.077)', 'harmony_complexity(247.077)']

4. FEATURE CORRELATION ANALYSIS
========================================

Highly correlated feature pairs (|r| > 0.7):
  velocity_std <-> dynamic_contrast: 1.000
  chord_changes <-> harmony_complexity: 1.000
  chord_changes <-> total_notes: 0.952
  harmony_complexity <-> total_notes: 0.952
  velocity_max <-> velocity_mean: 0.892
  rhythm_mean <-> rhythm_std: 0.821
  velocity_min <-> velocity_mean: 0.781

5. STATISTICAL TESTS
========================================
ANOVA tests for feature differences across emotions:
C:\Users\perer\AppData\Local\Programs\Python\Python311\Lib\site-packages\scipy\stats\_stats_py.py:4167: ConstantInputWarning: Each of the input arrays is constant;the F statistic is not defined or infinite
  warnings.warn(stats.ConstantInputWarning(msg))

Top 10 most significant features (ANOVA):
   1. pitch_mean: F=133.385, p=0.000000 ***
   2. pitch_max: F=100.888, p=0.000000 ***
   3. velocity_mean: F=73.824, p=0.000000 ***
   4. velocity_max: F=61.414, p=0.000000 ***
   5. note_overlap_ratio: F=42.039, p=0.000000 ***
   6. velocity_min: F=36.637, p=0.000000 ***
   7. pitch_std: F=28.778, p=0.000000 ***
   8. syncopation: F=24.819, p=0.000000 ***
   9. chord_changes: F=14.426, p=0.000000 ***
  10. harmony_complexity: F=14.426, p=0.000000 ***

Detailed ANOVA results saved to: results/anova_results.csv

EDA completed! Results saved in results/
Emotion distribution:
emotion
happy      553
sad        338
hopeful    174
tense       13
Name: count, dtype: int64

3. Balancing dataset with GAN...
Epoch 0, D Loss: 0.7300, G Loss: 0.6776
Epoch 20, D Loss: 0.8992, G Loss: 0.4072
Epoch 40, D Loss: 0.9207, G Loss: 0.4012
Generating 215 samples for sad
Generating 379 samples for hopeful
Generating 540 samples for tense
Balanced dataset emotion distribution:
emotion
happy      553
sad        553
hopeful    553
tense      553
Name: count, dtype: int64

3.5. Exploratory Data Analysis (Balanced Dataset)...

============================================================
EXPLORATORY DATA ANALYSIS: Balanced Dataset (After GAN)
============================================================

1. BASIC DATASET INFORMATION
========================================
Total samples: 2212
Total features: 23
Memory usage: 0.70 MB
Missing values: 1134
Duplicate rows: 0

Data types:
float64    21
object      2
Name: count, dtype: int64

2. EMOTION DISTRIBUTION ANALYSIS
========================================
Emotion distribution:
  happy: 553 samples (25.0%)
  sad: 553 samples (25.0%)
  hopeful: 553 samples (25.0%)
  tense: 553 samples (25.0%)

Dataset balance analysis:
  Most frequent emotion: happy (553 samples)
  Least frequent emotion: happy (553 samples)
  Balance ratio: 1.000 (1.0 = perfectly balanced)
  OK: Dataset is reasonably balanced (ratio >= 0.5)

3. MUSICAL FEATURE ANALYSIS
========================================
Number of numeric features: 21

Feature statistics:
       duration  pitch_min  pitch_max  ...  tempo_variation  dynamic_range  dynamic_contrast
count  2212.000   2212.000   2212.000  ...         2212.000       2212.000          2212.000
mean     36.116     35.491     88.467  ...            0.511         59.382            10.425
std       8.535      7.869      6.677  ...            0.498          8.888             2.416
min      18.009     22.000     63.000  ...            0.000         28.000             4.896
25%      29.990     28.457     85.000  ...            0.000         54.391             9.229
50%      35.082     35.500     90.225  ...            0.988         58.130             9.271
75%      39.419     42.000     94.569  ...            0.998         63.000            11.698
max     109.990     69.000    105.000  ...            1.000         99.000            24.903

[8 rows x 21 columns]

Feature correlation with emotions:
  happy: ['total_notes(266.995)', 'chord_changes(145.190)', 'harmony_complexity(145.190)']
  sad: ['total_notes(285.749)', 'chord_changes(147.974)', 'tempo_mean(119.981)']
  hopeful: ['total_notes(426.463)', 'chord_changes(260.338)', 'tempo_mean(119.991)']
  tense: ['total_notes(501.964)', 'chord_changes(331.199)', 'harmony_complexity(161.229)']

4. FEATURE CORRELATION ANALYSIS
========================================

Highly correlated feature pairs (|r| > 0.7):
  velocity_std <-> dynamic_contrast: 1.000
  chord_changes <-> total_notes: 0.976
  tempo_std <-> tempo_variation: 0.931
  velocity_max <-> velocity_mean: 0.921
  velocity_min <-> velocity_mean: 0.913
  rhythm_mean <-> rhythm_std: 0.886
  velocity_min <-> velocity_max: 0.790
  rhythm_mean <-> total_notes: -0.768
  velocity_mean <-> chord_changes: 0.740
  chord_changes <-> rhythm_mean: -0.740
  rhythm_std <-> total_notes: -0.731
  velocity_mean <-> total_notes: 0.717
  chord_changes <-> rhythm_std: -0.716
  pitch_min <-> pitch_mean: 0.700

5. STATISTICAL TESTS
========================================
ANOVA tests for feature differences across emotions:

Top 10 most significant features (ANOVA):
   1. pitch_mean: F=756.459, p=0.000000 ***
   2. velocity_min: F=1051.613, p=0.000000 ***
   3. velocity_max: F=977.095, p=0.000000 ***
   4. velocity_mean: F=1167.997, p=0.000000 ***
   5. tempo_variation: F=808.703, p=0.000000 ***
   6. note_overlap_ratio: F=670.299, p=0.000000 ***
   7. chord_changes: F=617.945, p=0.000000 ***
   8. tempo_std: F=605.611, p=0.000000 ***
   9. total_notes: F=546.799, p=0.000000 ***
  10. pitch_min: F=418.890, p=0.000000 ***

Detailed ANOVA results saved to: results/anova_results.csv

EDA completed! Results saved in results/

6. DATASET COMPARISON (Original vs Balanced)
==================================================

Emotion distribution comparison:
         Original  Balanced
emotion
happy         553       553
sad           338       553
hopeful       174       553
tense          13       553

Balance improvement:
  Original balance ratio: 0.024
  Balanced balance ratio: 1.000
  Improvement: 4153.8%

3.6. Model Training Visualizations...

7. MODEL TRAINING VISUALIZATION
==================================================

Training Analysis Summary:
  Total epochs: 50
  Final D loss: 0.9049
  Final G loss: 0.4004
  D loss range: 0.7300 - 0.9406
  G loss range: 0.4000 - 0.6776
  Training stability: D_std=0.0498, G_std=0.0651
  Training quality: STABLE
  Training visualization saved to: results/gan_training_analysis.png

8. MODEL ARCHITECTURE VISUALIZATION
==================================================
Model architecture visualization saved to: results/model_architecture.png

9. FEATURE IMPORTANCE ANALYSIS
==================================================
Feature importance analysis saved to: results/feature_importance.png
Top 5 most important features:
  1. velocity_mean: 6.336
  2. velocity_min: 5.705
  3. velocity_max: 5.301
  4. tempo_variation: 4.387
  5. pitch_mean: 4.104

4. Training recommendation system...
Training recommendation system...
Trained on 2212 samples
Feature dimensions: 21
Recommendation system stats:
{'total_samples': 2212, 'emotion_distribution': {'happy': 553, 'sad': 553, 'hopeful': 553, 'tense': 553}, 'feature_dimensions': 21}

5. Testing recommendation system...
Query file: ../EMOPIA_1.0/midis\Q2_Dg935IbDggo_0.mid
Query emotion category: hopeful

Top 5 similar tracks:
1. Q1_jziI_cuN_H8_0.mid (similarity: 0.753, emotion: hopeful)
2. Q1_UOSlDydo94E_0.mid (similarity: 0.724, emotion: hopeful)
3. Q3_G4rL_OtfFAU_0.mid (similarity: 0.724, emotion: hopeful)
4. Q2_ANZf1QXsNrY_6.mid (similarity: 0.700, emotion: hopeful)
5. Q3_TYkTOwBfFB8_0.mid (similarity: 0.679, emotion: hopeful)

Test Accuracy: 100.0% (5/5 correct predictions)
Query emotion category: hopeful

System completed successfully!
Results saved in results/ directory
--------------------------------------------------------------

Track-10-Q2_9v2WSpn4FCw_4
1. Survey Results
Happy:6.7%
Sad:3.3%
Hopeful:6.7%
Fearful:10%
Tense:26.7%
Excited:46.7%

2. System Results:
Music Emotion Analysis and Recommendation System
==================================================

1. Extracting features from MIDI files...
Found 1078 MIDI files
Processing file 1/1078
Processing file 101/1078
Processing file 201/1078
Processing file 301/1078
Processing file 401/1078
Processing file 501/1078
Processing file 601/1078
Processing file 701/1078
Processing file 801/1078
Processing file 901/1078
Processing file 1001/1078
Extracted features from 1078 files

2. Classifying emotions...

2.5. Exploratory Data Analysis (Classified Dataset)...

============================================================
EXPLORATORY DATA ANALYSIS: Classified EMOPIA Dataset
============================================================

1. BASIC DATASET INFORMATION
========================================
Total samples: 1078
Total features: 23
Memory usage: 0.42 MB
Missing values: 0
Duplicate rows: 0

Data types:
float64    13
int64       8
object      2
Name: count, dtype: int64

2. EMOTION DISTRIBUTION ANALYSIS
========================================
Emotion distribution:
  happy: 553 samples (51.3%)
  sad: 338 samples (31.4%)
  hopeful: 174 samples (16.1%)
  tense: 13 samples (1.2%)

Dataset balance analysis:
  Most frequent emotion: happy (553 samples)
  Least frequent emotion: tense (13 samples)
  Balance ratio: 0.024 (1.0 = perfectly balanced)
  WARNING: Dataset is highly imbalanced (ratio < 0.1)

3. MUSICAL FEATURE ANALYSIS
========================================
Number of numeric features: 21

Feature statistics:
       duration  pitch_min  pitch_max  ...  tempo_variation  dynamic_range  dynamic_contrast
count  1078.000   1078.000   1078.000  ...           1078.0       1078.000          1078.000
mean     39.924     34.680     85.633  ...              0.0         61.499            11.939
std      10.475      8.154      8.248  ...              0.0         11.879             2.714
min      18.009     22.000     63.000  ...              0.0         28.000             4.896
25%      29.990     28.000     80.000  ...              0.0         53.000            10.017
50%      39.584     34.500     85.000  ...              0.0         62.000            11.762
75%      46.725     40.000     91.000  ...              0.0         69.000            13.607
max     109.990     69.000    105.000  ...              0.0         99.000            24.903

[8 rows x 21 columns]

Feature correlation with emotions:
  happy: ['total_notes(266.995)', 'chord_changes(145.190)', 'harmony_complexity(145.190)']
  sad: ['total_notes(235.935)', 'tempo_mean(120.000)', 'chord_changes(115.003)']
  hopeful: ['total_notes(305.592)', 'chord_changes(170.138)', 'harmony_complexity(170.138)']
  tense: ['total_notes(390.077)', 'chord_changes(247.077)', 'harmony_complexity(247.077)']

4. FEATURE CORRELATION ANALYSIS
========================================

Highly correlated feature pairs (|r| > 0.7):
  velocity_std <-> dynamic_contrast: 1.000
  chord_changes <-> harmony_complexity: 1.000
  chord_changes <-> total_notes: 0.952
  harmony_complexity <-> total_notes: 0.952
  velocity_max <-> velocity_mean: 0.892
  rhythm_mean <-> rhythm_std: 0.821
  velocity_min <-> velocity_mean: 0.781

5. STATISTICAL TESTS
========================================
ANOVA tests for feature differences across emotions:
C:\Users\perer\AppData\Local\Programs\Python\Python311\Lib\site-packages\scipy\stats\_stats_py.py:4167: ConstantInputWarning: Each of the input arrays is constant;the F statistic is not defined or infinite
  warnings.warn(stats.ConstantInputWarning(msg))

Top 10 most significant features (ANOVA):
   1. pitch_mean: F=133.385, p=0.000000 ***
   2. pitch_max: F=100.888, p=0.000000 ***
   3. velocity_mean: F=73.824, p=0.000000 ***
   4. velocity_max: F=61.414, p=0.000000 ***
   5. note_overlap_ratio: F=42.039, p=0.000000 ***
   6. velocity_min: F=36.637, p=0.000000 ***
   7. pitch_std: F=28.778, p=0.000000 ***
   8. syncopation: F=24.819, p=0.000000 ***
   9. chord_changes: F=14.426, p=0.000000 ***
  10. harmony_complexity: F=14.426, p=0.000000 ***

Detailed ANOVA results saved to: results/anova_results.csv

EDA completed! Results saved in results/
Emotion distribution:
emotion
happy      553
sad        338
hopeful    174
tense       13
Name: count, dtype: int64

3. Balancing dataset with GAN...
Epoch 0, D Loss: 0.7296, G Loss: 0.7096
Epoch 20, D Loss: 0.8576, G Loss: 0.4941
Epoch 40, D Loss: 0.8699, G Loss: 0.4889
Generating 215 samples for sad
Generating 379 samples for hopeful
Generating 540 samples for tense
Balanced dataset emotion distribution:
emotion
happy      553
sad        553
hopeful    553
tense      553
Name: count, dtype: int64

3.5. Exploratory Data Analysis (Balanced Dataset)...

============================================================
EXPLORATORY DATA ANALYSIS: Balanced Dataset (After GAN)
============================================================

1. BASIC DATASET INFORMATION
========================================
Total samples: 2212
Total features: 23
Memory usage: 0.70 MB
Missing values: 1134
Duplicate rows: 0

Data types:
float64    21
object      2
Name: count, dtype: int64

2. EMOTION DISTRIBUTION ANALYSIS
========================================
Emotion distribution:
  happy: 553 samples (25.0%)
  sad: 553 samples (25.0%)
  hopeful: 553 samples (25.0%)
  tense: 553 samples (25.0%)

Dataset balance analysis:
  Most frequent emotion: happy (553 samples)
  Least frequent emotion: happy (553 samples)
  Balance ratio: 1.000 (1.0 = perfectly balanced)
  OK: Dataset is reasonably balanced (ratio >= 0.5)

3. MUSICAL FEATURE ANALYSIS
========================================
Number of numeric features: 21

Feature statistics:
       duration  pitch_min  pitch_max  ...  tempo_variation  dynamic_range  dynamic_contrast
count  2212.000   2212.000   2212.000  ...         2212.000       2212.000          2212.000
mean     44.838     29.604     80.534  ...           -0.333         57.118            12.986
std       8.937      7.652      7.875  ...            0.337          9.458             2.187
min      18.009     22.000     63.000  ...           -0.907         28.000             4.896
25%      39.752     23.000     74.048  ...           -0.673         51.286            11.872
50%      46.797     26.503     79.642  ...           -0.354         54.199            13.416
75%      51.809     34.000     84.000  ...            0.000         61.000            14.513
max     109.990     69.000    105.000  ...            0.000         99.000            24.903

[8 rows x 21 columns]

Feature correlation with emotions:
  happy: ['total_notes(266.995)', 'chord_changes(145.190)', 'harmony_complexity(145.190)']
  sad: ['total_notes(186.159)', 'tempo_mean(119.832)', 'chord_changes(106.271)']
  hopeful: ['total_notes(184.767)', 'chord_changes(143.509)', 'tempo_mean(119.694)']
  tense: ['total_notes(278.248)', 'chord_changes(221.949)', 'harmony_complexity(175.172)']

4. FEATURE CORRELATION ANALYSIS
========================================

Highly correlated feature pairs (|r| > 0.7):
  velocity_std <-> dynamic_contrast: 1.000
  harmony_complexity <-> total_notes: 0.956
  velocity_max <-> velocity_mean: 0.945
  chord_changes <-> harmony_complexity: 0.934
  tempo_std <-> tempo_variation: -0.867
  tempo_mean <-> tempo_variation: 0.861
  chord_changes <-> total_notes: 0.859
  velocity_min <-> velocity_mean: 0.780
  tempo_mean <-> tempo_std: -0.775
  pitch_max <-> pitch_mean: 0.766
  pitch_min <-> pitch_std: -0.765
  pitch_std <-> velocity_min: 0.726
  pitch_min <-> pitch_mean: 0.713
  rhythm_mean <-> total_notes: -0.710

5. STATISTICAL TESTS
========================================
ANOVA tests for feature differences across emotions:

Top 10 most significant features (ANOVA):
   1. pitch_mean: F=970.677, p=0.000000 ***
   2. velocity_min: F=1034.125, p=0.000000 ***
   3. velocity_max: F=965.225, p=0.000000 ***
   4. velocity_mean: F=1314.375, p=0.000000 ***
   5. tempo_variation: F=711.415, p=0.000000 ***
   6. pitch_max: F=672.534, p=0.000000 ***
   7. pitch_std: F=555.232, p=0.000000 ***
   8. tempo_mean: F=532.205, p=0.000000 ***
   9. tempo_std: F=520.474, p=0.000000 ***
  10. note_overlap_ratio: F=364.259, p=0.000000 ***

Detailed ANOVA results saved to: results/anova_results.csv

EDA completed! Results saved in results/

6. DATASET COMPARISON (Original vs Balanced)
==================================================

Emotion distribution comparison:
         Original  Balanced
emotion
happy         553       553
sad           338       553
hopeful       174       553
tense          13       553

Balance improvement:
  Original balance ratio: 0.024
  Balanced balance ratio: 1.000
  Improvement: 4153.8%

3.6. Model Training Visualizations...

7. MODEL TRAINING VISUALIZATION
==================================================

Training Analysis Summary:
  Total epochs: 50
  Final D loss: 0.8711
  Final G loss: 0.4873
  D loss range: 0.7296 - 0.8822
  G loss range: 0.4873 - 0.7096
  Training stability: D_std=0.0319, G_std=0.0486
  Training quality: STABLE
  Training visualization saved to: results/gan_training_analysis.png

8. MODEL ARCHITECTURE VISUALIZATION
==================================================
Model architecture visualization saved to: results/model_architecture.png

9. FEATURE IMPORTANCE ANALYSIS
==================================================
Feature importance analysis saved to: results/feature_importance.png
Top 5 most important features:
  1. velocity_mean: 7.130
  2. velocity_min: 5.610
  3. pitch_mean: 5.266
  4. velocity_max: 5.236
  5. tempo_variation: 3.859

4. Training recommendation system...
Training recommendation system...
Trained on 2212 samples
Feature dimensions: 21
Recommendation system stats:
{'total_samples': 2212, 'emotion_distribution': {'happy': 553, 'sad': 553, 'hopeful': 553, 'tense': 553}, 'feature_dimensions': 21}

5. Testing recommendation system...
Query file: ../EMOPIA_1.0/midis\Q2_9v2WSpn4FCw_4.mid
Query emotion category: hopeful

Top 5 similar tracks:
1. Q2_9v2WSpn4FCw_6.mid (similarity: 0.960, emotion: hopeful)
2. Q2_0v2N1ROvEI0_0.mid (similarity: 0.946, emotion: hopeful)
3. Q2_qlbazHayULg_0.mid (similarity: 0.920, emotion: hopeful)
4. Q3_uj3Gif77SYM_8.mid (similarity: 0.870, emotion: hopeful)
5. Q2_k-FNDbK6Qhg_0.mid (similarity: 0.843, emotion: hopeful)

Test Accuracy: 100.0% (5/5 correct predictions)
Query emotion category: hopeful

System completed successfully!
Results saved in results/ directory
--------------------------------------------------------------

Track-11-Q1__8v0MFBZoco_1
1. Survey Results
Happy:53.3%
Sad:0%
Hopeful:13.3%
Fearful:0%
Tense:10%
Excited:23.3%

2. System Results:
Music Emotion Analysis and Recommendation System
==================================================

1. Extracting features from MIDI files...
Found 1078 MIDI files
Processing file 1/1078
Processing file 101/1078
Processing file 201/1078
Processing file 301/1078
Processing file 401/1078
Processing file 501/1078
Processing file 601/1078
Processing file 701/1078
Processing file 801/1078
Processing file 901/1078
Processing file 1001/1078
Extracted features from 1078 files

2. Classifying emotions...

2.5. Exploratory Data Analysis (Classified Dataset)...

============================================================
EXPLORATORY DATA ANALYSIS: Classified EMOPIA Dataset
============================================================

1. BASIC DATASET INFORMATION
========================================
Total samples: 1078
Total features: 23
Memory usage: 0.42 MB
Missing values: 0
Duplicate rows: 0

Data types:
float64    13
int64       8
object      2
Name: count, dtype: int64

2. EMOTION DISTRIBUTION ANALYSIS
========================================
Emotion distribution:
  happy: 553 samples (51.3%)
  sad: 338 samples (31.4%)
  hopeful: 174 samples (16.1%)
  tense: 13 samples (1.2%)

Dataset balance analysis:
  Most frequent emotion: happy (553 samples)
  Least frequent emotion: tense (13 samples)
  Balance ratio: 0.024 (1.0 = perfectly balanced)
  WARNING: Dataset is highly imbalanced (ratio < 0.1)

3. MUSICAL FEATURE ANALYSIS
========================================
Number of numeric features: 21

Feature statistics:
       duration  pitch_min  pitch_max  ...  tempo_variation  dynamic_range  dynamic_contrast
count  1078.000   1078.000   1078.000  ...           1078.0       1078.000          1078.000
mean     39.924     34.680     85.633  ...              0.0         61.499            11.939
std      10.475      8.154      8.248  ...              0.0         11.879             2.714
min      18.009     22.000     63.000  ...              0.0         28.000             4.896
25%      29.990     28.000     80.000  ...              0.0         53.000            10.017
50%      39.584     34.500     85.000  ...              0.0         62.000            11.762
75%      46.725     40.000     91.000  ...              0.0         69.000            13.607
max     109.990     69.000    105.000  ...              0.0         99.000            24.903

[8 rows x 21 columns]

Feature correlation with emotions:
  happy: ['total_notes(266.995)', 'chord_changes(145.190)', 'harmony_complexity(145.190)']
  sad: ['total_notes(235.935)', 'tempo_mean(120.000)', 'chord_changes(115.003)']
  hopeful: ['total_notes(305.592)', 'chord_changes(170.138)', 'harmony_complexity(170.138)']
  tense: ['total_notes(390.077)', 'chord_changes(247.077)', 'harmony_complexity(247.077)']

4. FEATURE CORRELATION ANALYSIS
========================================

Highly correlated feature pairs (|r| > 0.7):
  velocity_std <-> dynamic_contrast: 1.000
  chord_changes <-> harmony_complexity: 1.000
  chord_changes <-> total_notes: 0.952
  harmony_complexity <-> total_notes: 0.952
  velocity_max <-> velocity_mean: 0.892
  rhythm_mean <-> rhythm_std: 0.821
  velocity_min <-> velocity_mean: 0.781

5. STATISTICAL TESTS
========================================
ANOVA tests for feature differences across emotions:
C:\Users\perer\AppData\Local\Programs\Python\Python311\Lib\site-packages\scipy\stats\_stats_py.py:4167: ConstantInputWarning: Each of the input arrays is constant;the F statistic is not defined or infinite
  warnings.warn(stats.ConstantInputWarning(msg))

Top 10 most significant features (ANOVA):
   1. pitch_mean: F=133.385, p=0.000000 ***
   2. pitch_max: F=100.888, p=0.000000 ***
   3. velocity_mean: F=73.824, p=0.000000 ***
   4. velocity_max: F=61.414, p=0.000000 ***
   5. note_overlap_ratio: F=42.039, p=0.000000 ***
   6. velocity_min: F=36.637, p=0.000000 ***
   7. pitch_std: F=28.778, p=0.000000 ***
   8. syncopation: F=24.819, p=0.000000 ***
   9. chord_changes: F=14.426, p=0.000000 ***
  10. harmony_complexity: F=14.426, p=0.000000 ***

Detailed ANOVA results saved to: results/anova_results.csv

EDA completed! Results saved in results/
Emotion distribution:
emotion
happy      553
sad        338
hopeful    174
tense       13
Name: count, dtype: int64

3. Balancing dataset with GAN...
Epoch 0, D Loss: 0.7324, G Loss: 0.6788
Epoch 20, D Loss: 0.8669, G Loss: 0.4684
Epoch 40, D Loss: 0.8701, G Loss: 0.4626
Generating 215 samples for sad
Generating 379 samples for hopeful
Generating 540 samples for tense
Balanced dataset emotion distribution:
emotion
happy      553
sad        553
hopeful    553
tense      553
Name: count, dtype: int64

3.5. Exploratory Data Analysis (Balanced Dataset)...

============================================================
EXPLORATORY DATA ANALYSIS: Balanced Dataset (After GAN)
============================================================

1. BASIC DATASET INFORMATION
========================================
Total samples: 2212
Total features: 23
Memory usage: 0.70 MB
Missing values: 1134
Duplicate rows: 0

Data types:
float64    21
object      2
Name: count, dtype: int64

2. EMOTION DISTRIBUTION ANALYSIS
========================================
Emotion distribution:
  happy: 553 samples (25.0%)
  sad: 553 samples (25.0%)
  hopeful: 553 samples (25.0%)
  tense: 553 samples (25.0%)

Dataset balance analysis:
  Most frequent emotion: happy (553 samples)
  Least frequent emotion: happy (553 samples)
  Balance ratio: 1.000 (1.0 = perfectly balanced)
  OK: Dataset is reasonably balanced (ratio >= 0.5)

3. MUSICAL FEATURE ANALYSIS
========================================
Number of numeric features: 21

Feature statistics:
       duration  pitch_min  pitch_max  ...  tempo_variation  dynamic_range  dynamic_contrast
count  2212.000   2212.000   2212.000  ...         2212.000       2212.000          2212.000
mean     35.534     35.540     87.779  ...            0.511         55.791            12.984
std       8.848      7.896      6.454  ...            0.499         10.033             2.186
min      18.009     22.000     63.000  ...            0.000         28.000             4.896
25%      29.990     28.498     84.845  ...            0.000         49.181            11.872
50%      34.542     35.500     88.725  ...            0.991         51.765            13.414
75%      39.419     42.000     92.905  ...            0.998         61.000            14.501
max     109.990     69.000    105.000  ...            1.000         99.000            24.903

[8 rows x 21 columns]

Feature correlation with emotions:
  happy: ['total_notes(266.995)', 'chord_changes(145.190)', 'harmony_complexity(145.190)']
  sad: ['total_notes(186.199)', 'chord_changes(148.666)', 'tempo_mean(120.041)']
  hopeful: ['chord_changes(261.909)', 'total_notes(184.923)', 'tempo_mean(120.074)']
  tense: ['chord_changes(332.672)', 'total_notes(278.333)', 'harmony_complexity(161.529)']

4. FEATURE CORRELATION ANALYSIS
========================================

Highly correlated feature pairs (|r| > 0.7):
  tempo_std <-> tempo_variation: 1.000
  harmony_complexity <-> total_notes: 0.965
  velocity_max <-> velocity_mean: 0.935
  rhythm_mean <-> rhythm_std: 0.897
  velocity_std <-> dynamic_contrast: 0.826
  pitch_std <-> velocity_max: 0.783
  pitch_std <-> velocity_mean: 0.782
  chord_changes <-> rhythm_mean: -0.757
  velocity_max <-> chord_changes: 0.738
  pitch_std <-> chord_changes: 0.729
  chord_changes <-> rhythm_std: -0.718
  velocity_mean <-> chord_changes: 0.712

5. STATISTICAL TESTS
========================================
ANOVA tests for feature differences across emotions:

Top 10 most significant features (ANOVA):
   1. pitch_mean: F=956.376, p=0.000000 ***
   2. velocity_max: F=850.584, p=0.000000 ***
   3. velocity_mean: F=1260.455, p=0.000000 ***
   4. tempo_std: F=809.286, p=0.000000 ***
   5. tempo_variation: F=809.113, p=0.000000 ***
   6. chord_changes: F=622.350, p=0.000000 ***
   7. pitch_std: F=554.846, p=0.000000 ***
   8. pitch_min: F=419.671, p=0.000000 ***
   9. rhythm_mean: F=329.014, p=0.000000 ***
  10. rhythm_std: F=299.902, p=0.000000 ***

Detailed ANOVA results saved to: results/anova_results.csv

EDA completed! Results saved in results/

6. DATASET COMPARISON (Original vs Balanced)
==================================================

Emotion distribution comparison:
         Original  Balanced
emotion
happy         553       553
sad           338       553
hopeful       174       553
tense          13       553

Balance improvement:
  Original balance ratio: 0.024
  Balanced balance ratio: 1.000
  Improvement: 4153.8%

3.6. Model Training Visualizations...

7. MODEL TRAINING VISUALIZATION
==================================================

Training Analysis Summary:
  Total epochs: 50
  Final D loss: 0.8670
  Final G loss: 0.4610
  D loss range: 0.7324 - 0.8985
  G loss range: 0.4610 - 0.6788
  Training stability: D_std=0.0329, G_std=0.0486
  Training quality: STABLE
  Training visualization saved to: results/gan_training_analysis.png

8. MODEL ARCHITECTURE VISUALIZATION
==================================================
Model architecture visualization saved to: results/model_architecture.png

9. FEATURE IMPORTANCE ANALYSIS
==================================================
Feature importance analysis saved to: results/feature_importance.png
Top 5 most important features:
  1. velocity_mean: 6.838
  2. pitch_mean: 5.188
  3. velocity_max: 4.614
  4. tempo_std: 4.390
  5. tempo_variation: 4.389

4. Training recommendation system...
Training recommendation system...
Trained on 2212 samples
Feature dimensions: 21
Recommendation system stats:
{'total_samples': 2212, 'emotion_distribution': {'happy': 553, 'sad': 553, 'hopeful': 553, 'tense': 553}, 'feature_dimensions': 21}

5. Testing recommendation system...
Query file: ../EMOPIA_1.0/midis\Q1__8v0MFBZoco_1.mid
Query emotion category: happy

Top 5 similar tracks:
1. Q2_Q5b5unyP8BM_0.mid (similarity: 0.906, emotion: happy)
2. Q2_7n32BtYb4d4_2.mid (similarity: 0.904, emotion: happy)
3. Q2_7n32BtYb4d4_0.mid (similarity: 0.896, emotion: happy)
4. Q1__8v0MFBZoco_0.mid (similarity: 0.896, emotion: happy)
5. Q1_e838BE_MH6s_4.mid (similarity: 0.894, emotion: happy)

Test Accuracy: 100.0% (5/5 correct predictions)
Query emotion category: happy

System completed successfully!
Results saved in results/ directory
--------------------------------------------------------------

Track-12-Q4_7yW9c7t8Hq0_0
1. Survey Results
Happy:16.7%
Sad:46.7%
Hopeful:33.3%
Fearful:0%
Tense:0%
Excited:3.3%

2. System Results:
Music Emotion Analysis and Recommendation System
==================================================

1. Extracting features from MIDI files...
Found 1078 MIDI files
Processing file 1/1078
Processing file 101/1078
Processing file 201/1078
Processing file 301/1078
Processing file 401/1078
Processing file 501/1078
Processing file 601/1078
Processing file 701/1078
Processing file 801/1078
Processing file 901/1078
Processing file 1001/1078
Extracted features from 1078 files

2. Classifying emotions...

2.5. Exploratory Data Analysis (Classified Dataset)...

============================================================
EXPLORATORY DATA ANALYSIS: Classified EMOPIA Dataset
============================================================

1. BASIC DATASET INFORMATION
========================================
Total samples: 1078
Total features: 23
Memory usage: 0.42 MB
Missing values: 0
Duplicate rows: 0

Data types:
float64    13
int64       8
object      2
Name: count, dtype: int64

2. EMOTION DISTRIBUTION ANALYSIS
========================================
Emotion distribution:
  happy: 553 samples (51.3%)
  sad: 338 samples (31.4%)
  hopeful: 174 samples (16.1%)
  tense: 13 samples (1.2%)

Dataset balance analysis:
  Most frequent emotion: happy (553 samples)
  Least frequent emotion: tense (13 samples)
  Balance ratio: 0.024 (1.0 = perfectly balanced)
  WARNING: Dataset is highly imbalanced (ratio < 0.1)

3. MUSICAL FEATURE ANALYSIS
========================================
Number of numeric features: 21

Feature statistics:
       duration  pitch_min  pitch_max  ...  tempo_variation  dynamic_range  dynamic_contrast
count  1078.000   1078.000   1078.000  ...           1078.0       1078.000          1078.000
mean     39.924     34.680     85.633  ...              0.0         61.499            11.939
std      10.475      8.154      8.248  ...              0.0         11.879             2.714
min      18.009     22.000     63.000  ...              0.0         28.000             4.896
25%      29.990     28.000     80.000  ...              0.0         53.000            10.017
50%      39.584     34.500     85.000  ...              0.0         62.000            11.762
75%      46.725     40.000     91.000  ...              0.0         69.000            13.607
max     109.990     69.000    105.000  ...              0.0         99.000            24.903

[8 rows x 21 columns]

Feature correlation with emotions:
  happy: ['total_notes(266.995)', 'chord_changes(145.190)', 'harmony_complexity(145.190)']
  sad: ['total_notes(235.935)', 'tempo_mean(120.000)', 'chord_changes(115.003)']
  hopeful: ['total_notes(305.592)', 'chord_changes(170.138)', 'harmony_complexity(170.138)']
  tense: ['total_notes(390.077)', 'chord_changes(247.077)', 'harmony_complexity(247.077)']

4. FEATURE CORRELATION ANALYSIS
========================================

Highly correlated feature pairs (|r| > 0.7):
  velocity_std <-> dynamic_contrast: 1.000
  chord_changes <-> harmony_complexity: 1.000
  chord_changes <-> total_notes: 0.952
  harmony_complexity <-> total_notes: 0.952
  velocity_max <-> velocity_mean: 0.892
  rhythm_mean <-> rhythm_std: 0.821
  velocity_min <-> velocity_mean: 0.781

5. STATISTICAL TESTS
========================================
ANOVA tests for feature differences across emotions:
C:\Users\perer\AppData\Local\Programs\Python\Python311\Lib\site-packages\scipy\stats\_stats_py.py:4167: ConstantInputWarning: Each of the input arrays is constant;the F statistic is not defined or infinite
  warnings.warn(stats.ConstantInputWarning(msg))

Top 10 most significant features (ANOVA):
   1. pitch_mean: F=133.385, p=0.000000 ***
   2. pitch_max: F=100.888, p=0.000000 ***
   3. velocity_mean: F=73.824, p=0.000000 ***
   4. velocity_max: F=61.414, p=0.000000 ***
   5. note_overlap_ratio: F=42.039, p=0.000000 ***
   6. velocity_min: F=36.637, p=0.000000 ***
   7. pitch_std: F=28.778, p=0.000000 ***
   8. syncopation: F=24.819, p=0.000000 ***
   9. chord_changes: F=14.426, p=0.000000 ***
  10. harmony_complexity: F=14.426, p=0.000000 ***

Detailed ANOVA results saved to: results/anova_results.csv

EDA completed! Results saved in results/
Emotion distribution:
emotion
happy      553
sad        338
hopeful    174
tense       13
Name: count, dtype: int64

3. Balancing dataset with GAN...
Epoch 0, D Loss: 0.7206, G Loss: 0.6467
Epoch 20, D Loss: 0.8975, G Loss: 0.3850
Epoch 40, D Loss: 0.9130, G Loss: 0.3807
Generating 215 samples for sad
Generating 379 samples for hopeful
Generating 540 samples for tense
Balanced dataset emotion distribution:
emotion
happy      553
sad        553
hopeful    553
tense      553
Name: count, dtype: int64

3.5. Exploratory Data Analysis (Balanced Dataset)...

============================================================
EXPLORATORY DATA ANALYSIS: Balanced Dataset (After GAN)
============================================================

1. BASIC DATASET INFORMATION
========================================
Total samples: 2212
Total features: 23
Memory usage: 0.70 MB
Missing values: 1134
Duplicate rows: 0

Data types:
float64    21
object      2
Name: count, dtype: int64

2. EMOTION DISTRIBUTION ANALYSIS
========================================
Emotion distribution:
  happy: 553 samples (25.0%)
  sad: 553 samples (25.0%)
  hopeful: 553 samples (25.0%)
  tense: 553 samples (25.0%)

Dataset balance analysis:
  Most frequent emotion: happy (553 samples)
  Least frequent emotion: happy (553 samples)
  Balance ratio: 1.000 (1.0 = perfectly balanced)
  OK: Dataset is reasonably balanced (ratio >= 0.5)

3. MUSICAL FEATURE ANALYSIS
========================================
Number of numeric features: 21

Feature statistics:
       duration  pitch_min  pitch_max  ...  tempo_variation  dynamic_range  dynamic_contrast
count  2212.000   2212.000   2212.000  ...         2212.000       2212.000          2212.000
mean     35.532     29.597     88.472  ...            0.059         70.202            10.425
std       8.850      7.655      6.680  ...            0.178         12.651             2.416
min      18.009     22.000     63.000  ...           -0.669         28.000             4.896
25%      29.990     23.000     85.000  ...            0.000         62.000             9.228
50%      34.543     26.460     90.240  ...            0.000         72.000             9.270
75%      39.419     34.000     94.618  ...            0.143         84.646            11.698
max     109.990     69.000    105.000  ...            0.727         99.000            24.903

[8 rows x 21 columns]

Feature correlation with emotions:
  happy: ['total_notes(266.995)', 'chord_changes(145.190)', 'harmony_complexity(145.190)']
  sad: ['total_notes(285.290)', 'harmony_complexity(148.770)', 'chord_changes(148.717)']
  hopeful: ['total_notes(425.454)', 'harmony_complexity(262.249)', 'chord_changes(262.072)']
  tense: ['total_notes(501.016)', 'harmony_complexity(332.912)', 'chord_changes(332.770)']

4. FEATURE CORRELATION ANALYSIS
========================================

Highly correlated feature pairs (|r| > 0.7):
  chord_changes <-> harmony_complexity: 1.000
  harmony_complexity <-> total_notes: 0.976
  chord_changes <-> total_notes: 0.976
  tempo_mean <-> tempo_std: 0.966
  velocity_max <-> velocity_mean: 0.909
  syncopation <-> note_overlap_ratio: -0.765
  syncopation <-> total_notes: 0.763
  harmony_complexity <-> syncopation: 0.761
  chord_changes <-> syncopation: 0.761
  velocity_mean <-> harmony_complexity: 0.736
  velocity_mean <-> chord_changes: 0.736
  rhythm_std <-> total_notes: -0.730
  harmony_complexity <-> rhythm_std: -0.718
  chord_changes <-> rhythm_std: -0.718
  tempo_mean <-> syncopation: -0.717
  pitch_min <-> pitch_mean: 0.713
  velocity_mean <-> total_notes: 0.710

5. STATISTICAL TESTS
========================================
ANOVA tests for feature differences across emotions:

Top 10 most significant features (ANOVA):
   1. pitch_mean: F=971.073, p=0.000000 ***
   2. velocity_max: F=997.827, p=0.000000 ***
   3. velocity_mean: F=1192.470, p=0.000000 ***
   4. tempo_mean: F=806.851, p=0.000000 ***
   5. syncopation: F=764.414, p=0.000000 ***
   6. tempo_std: F=703.397, p=0.000000 ***
   7. note_overlap_ratio: F=668.757, p=0.000000 ***
   8. dynamic_range: F=652.253, p=0.000000 ***
   9. harmony_complexity: F=623.119, p=0.000000 ***
  10. chord_changes: F=622.624, p=0.000000 ***

Detailed ANOVA results saved to: results/anova_results.csv

EDA completed! Results saved in results/

6. DATASET COMPARISON (Original vs Balanced)
==================================================

Emotion distribution comparison:
         Original  Balanced
emotion
happy         553       553
sad           338       553
hopeful       174       553
tense          13       553

Balance improvement:
  Original balance ratio: 0.024
  Balanced balance ratio: 1.000
  Improvement: 4153.8%

3.6. Model Training Visualizations...

7. MODEL TRAINING VISUALIZATION
==================================================

Training Analysis Summary:
  Total epochs: 50
  Final D loss: 0.8988
  Final G loss: 0.3795
  D loss range: 0.7206 - 0.9253
  G loss range: 0.3795 - 0.6467
  Training stability: D_std=0.0488, G_std=0.0593
  Training quality: STABLE
  Training visualization saved to: results/gan_training_analysis.png

8. MODEL ARCHITECTURE VISUALIZATION
==================================================
Model architecture visualization saved to: results/model_architecture.png

9. FEATURE IMPORTANCE ANALYSIS
==================================================
Feature importance analysis saved to: results/feature_importance.png
Top 5 most important features:
  1. velocity_mean: 6.469
  2. velocity_max: 5.413
  3. pitch_mean: 5.268
  4. tempo_mean: 4.377
  5. syncopation: 4.147

Training recommendation system...
Training recommendation system...
Trained on 2212 samples
Feature dimensions: 21
Recommendation system stats:
{'total_samples': 2212, 'emotion_distribution': {'happy': 553, 'sad': 553, 'hopeful': 553, 'tense': 553}, 'feature_dimensions': 21}

5. Testing recommendation system...
Query file: ../EMOPIA_1.0/midis\Q4_7yW9c7t8Hq0_0.mid
Query emotion category: happy

Top 5 similar tracks:
1. Q4_vBHoSz8cJIE_4.mid (similarity: 0.928, emotion: happy)
2. Q4_Tc684nD8CEA_1.mid (similarity: 0.927, emotion: happy)
3. Q3_MbvPxCUMSek_2.mid (similarity: 0.917, emotion: happy)
4. Q3_QJlnTN7HRwE_1.mid (similarity: 0.904, emotion: happy)
5. Q4_NP0lwB-_-og_0.mid (similarity: 0.903, emotion: happy)

Test Accuracy: 100.0% (5/5 correct predictions)
Query emotion category: happy

System completed successfully!
Results saved in results/ directory 
--------------------------------------------------------------

Track-13-Q2_08t73Qgjkt4_0
1. Survey Results
Happy:10%
Sad:66.7%
Hopeful:10%
Fearful:6.7%
Tense:3.3%
Excited:3.3%

2. System Results:
Music Emotion Analysis and Recommendation System
==================================================

1. Extracting features from MIDI files...
Found 1078 MIDI files
Processing file 1/1078
Processing file 101/1078
Processing file 201/1078
Processing file 301/1078
Processing file 401/1078
Processing file 501/1078
Processing file 601/1078
Processing file 701/1078
Processing file 801/1078
Processing file 901/1078
Processing file 1001/1078
Extracted features from 1078 files

2. Classifying emotions...

2.5. Exploratory Data Analysis (Classified Dataset)...

============================================================
EXPLORATORY DATA ANALYSIS: Classified EMOPIA Dataset
============================================================

1. BASIC DATASET INFORMATION
========================================
Total samples: 1078
Total features: 23
Memory usage: 0.42 MB
Missing values: 0
Duplicate rows: 0

Data types:
float64    13
int64       8
object      2
Name: count, dtype: int64

2. EMOTION DISTRIBUTION ANALYSIS
========================================
Emotion distribution:
  happy: 553 samples (51.3%)
  sad: 338 samples (31.4%)
  hopeful: 174 samples (16.1%)
  tense: 13 samples (1.2%)

Dataset balance analysis:
  Most frequent emotion: happy (553 samples)
  Least frequent emotion: tense (13 samples)
  Balance ratio: 0.024 (1.0 = perfectly balanced)
  WARNING: Dataset is highly imbalanced (ratio < 0.1)

3. MUSICAL FEATURE ANALYSIS
========================================
Number of numeric features: 21

Feature statistics:
       duration  pitch_min  ...  dynamic_range  dynamic_contrast
count  1078.000   1078.000  ...       1078.000          1078.000
mean     39.924     34.680  ...         61.499            11.939
std      10.475      8.154  ...         11.879             2.714
min      18.009     22.000  ...         28.000             4.896
25%      29.990     28.000  ...         53.000            10.017
50%      39.584     34.500  ...         62.000            11.762
75%      46.725     40.000  ...         69.000            13.607
max     109.990     69.000  ...         99.000            24.903

[8 rows x 21 columns]

Feature correlation with emotions:
  happy: ['total_notes(266.995)', 'chord_changes(145.190)', 'harmony_complexity(145.190)']
  sad: ['total_notes(235.935)', 'tempo_mean(120.000)', 'chord_changes(115.003)']    
  hopeful: ['total_notes(305.592)', 'chord_changes(170.138)', 'harmony_complexity(170.138)']
  tense: ['total_notes(390.077)', 'chord_changes(247.077)', 'harmony_complexity(247.077)']

4. FEATURE CORRELATION ANALYSIS
========================================

Highly correlated feature pairs (|r| > 0.7):
  velocity_std <-> dynamic_contrast: 1.000
  chord_changes <-> harmony_complexity: 1.000
  chord_changes <-> total_notes: 0.952
  harmony_complexity <-> total_notes: 0.952
  velocity_max <-> velocity_mean: 0.892
  rhythm_mean <-> rhythm_std: 0.821
  velocity_min <-> velocity_mean: 0.781

5. STATISTICAL TESTS
========================================
ANOVA tests for feature differences across emotions:
C:\Users\perer\AppData\Local\Programs\Python\Python311\Lib\site-packages\scipy\stats\_stats_py.py:4167: ConstantInputWarning: Each of the input arrays is constant;the F statistic is not defined or infinite
  warnings.warn(stats.ConstantInputWarning(msg))

Top 10 most significant features (ANOVA):
   1. pitch_mean: F=133.385, p=0.000000 ***
   2. pitch_max: F=100.888, p=0.000000 ***
   3. velocity_mean: F=73.824, p=0.000000 ***
   4. velocity_max: F=61.414, p=0.000000 ***
   5. note_overlap_ratio: F=42.039, p=0.000000 ***
   6. velocity_min: F=36.637, p=0.000000 ***
   7. pitch_std: F=28.778, p=0.000000 ***
   8. syncopation: F=24.819, p=0.000000 ***
   9. chord_changes: F=14.426, p=0.000000 ***
  10. harmony_complexity: F=14.426, p=0.000000 ***

Detailed ANOVA results saved to: results/anova_results.csv

EDA completed! Results saved in results/
Emotion distribution:
emotion
happy      553
sad        338
hopeful    174
tense       13
Name: count, dtype: int64

3. Balancing dataset with GAN...
Epoch 0, D Loss: 0.6872, G Loss: 0.6300
Epoch 20, D Loss: 0.8786, G Loss: 0.3879
Epoch 40, D Loss: 0.8744, G Loss: 0.3819
Generating 215 samples for sad
Generating 379 samples for hopeful
Generating 540 samples for tense
Balanced dataset emotion distribution:
emotion
happy      553
sad        553
hopeful    553
tense      553
Name: count, dtype: int64

3.5. Exploratory Data Analysis (Balanced Dataset)...

============================================================
EXPLORATORY DATA ANALYSIS: Balanced Dataset (After GAN)
============================================================

1. BASIC DATASET INFORMATION
========================================
Total samples: 2212
Total features: 23
Memory usage: 0.70 MB
Missing values: 1134
Duplicate rows: 0

Data types:
float64    21
object      2
Name: count, dtype: int64

2. EMOTION DISTRIBUTION ANALYSIS
========================================
Emotion distribution:
  happy: 553 samples (25.0%)
  sad: 553 samples (25.0%)
  hopeful: 553 samples (25.0%)
  tense: 553 samples (25.0%)

Dataset balance analysis:
  Most frequent emotion: happy (553 samples)
  Least frequent emotion: happy (553 samples)
  Balance ratio: 1.000 (1.0 = perfectly balanced)
  OK: Dataset is reasonably balanced (ratio >= 0.5)

3. MUSICAL FEATURE ANALYSIS
========================================
Number of numeric features: 21

Feature statistics:
       duration  pitch_min  ...  dynamic_range  dynamic_contrast
count  2212.000   2212.000  ...       2212.000          2212.000
mean     44.839     32.177  ...         70.079            10.432
std       8.938      6.971  ...         12.565             2.412
min      18.009     22.000  ...         28.000             4.896
25%      39.752     25.761  ...         62.000             9.230
50%      46.798     32.015  ...         72.000             9.276
75%      51.810     36.000  ...         83.995            11.698
max     109.990     69.000  ...         99.000            24.903

[8 rows x 21 columns]

Feature correlation with emotions:
  happy: ['total_notes(266.995)', 'chord_changes(145.190)', 'harmony_complexity(145.190)']
  sad: ['total_notes(186.238)', 'harmony_complexity(148.464)', 'chord_changes(122.539)']
  hopeful: ['harmony_complexity(261.328)', 'chord_changes(191.456)', 'total_notes(185.034)']
  tense: ['harmony_complexity(332.076)', 'total_notes(278.458)', 'chord_changes(267.608)']

4. FEATURE CORRELATION ANALYSIS
========================================

Highly correlated feature pairs (|r| > 0.7):
  tempo_std <-> tempo_variation: 1.000
  tempo_mean <-> tempo_variation: -1.000
  tempo_mean <-> tempo_std: -1.000
  velocity_std <-> dynamic_contrast: 1.000
  velocity_max <-> velocity_mean: 0.953
  chord_changes <-> harmony_complexity: 0.935
  rhythm_mean <-> rhythm_std: 0.897
  pitch_max <-> pitch_mean: 0.765
  harmony_complexity <-> rhythm_mean: -0.757
  harmony_complexity <-> rhythm_std: -0.717

5. STATISTICAL TESTS
========================================
ANOVA tests for feature differences across emotions:

Top 10 most significant features (ANOVA):
   1. pitch_mean: F=970.074, p=0.000000 ***
   2. velocity_max: F=966.229, p=0.000000 ***
   3. velocity_mean: F=1271.898, p=0.000000 ***
   4. tempo_mean: F=808.859, p=0.000000 ***
   5. tempo_std: F=808.639, p=0.000000 ***
   6. tempo_variation: F=808.637, p=0.000000 ***
   7. note_overlap_ratio: F=670.253, p=0.000000 ***
   8. pitch_max: F=670.102, p=0.000000 ***
   9. dynamic_range: F=642.845, p=0.000000 ***
  10. harmony_complexity: F=620.156, p=0.000000 ***

Detailed ANOVA results saved to: results/anova_results.csv

EDA completed! Results saved in results/

6. DATASET COMPARISON (Original vs Balanced)
==================================================

Emotion distribution comparison:
         Original  Balanced
emotion
happy         553       553
sad           338       553
hopeful       174       553
tense          13       553

Balance improvement:
  Original balance ratio: 0.024
  Balanced balance ratio: 1.000
  Improvement: 4153.8%

3.6. Model Training Visualizations...

7. MODEL TRAINING VISUALIZATION
==================================================

Training Analysis Summary:
  Total epochs: 50
  Final D loss: 0.8840
  Final G loss: 0.3814
  D loss range: 0.6872 - 0.8921
  G loss range: 0.3814 - 0.6300
  Training stability: D_std=0.0439, G_std=0.0522
  Training quality: STABLE
  Training visualization saved to: results/gan_training_analysis.png

8. MODEL ARCHITECTURE VISUALIZATION
==================================================
Model architecture visualization saved to: results/model_architecture.png

9. FEATURE IMPORTANCE ANALYSIS
==================================================
Feature importance analysis saved to: results/feature_importance.png
Top 5 most important features:
  1. velocity_mean: 6.900
  2. pitch_mean: 5.263
  3. velocity_max: 5.242
  4. tempo_mean: 4.388
  5. tempo_std: 4.387

4. Training recommendation system...
Training recommendation system...
Trained on 2212 samples
Feature dimensions: 21
Recommendation system stats:
{'total_samples': 2212, 'emotion_distribution': {'happy': 553, 'sad': 553, 'hopeful': 553, 'tense': 553}, 'feature_dimensions': 21}

5. Testing recommendation system...
Query file: ../EMOPIA_1.0/midis\Q2_08t73Qgjkt4_0.mid
Query emotion category: sad

Top 5 similar tracks:
1. Q4_1vjy9oMFa8c_0.mid (similarity: 0.870, emotion: sad)
2. Q3_2v6gJi03LlA_0.mid (similarity: 0.850, emotion: sad)
3. Q2_1kny88W533Q_0.mid (similarity: 0.849, emotion: sad)
4. Q3_E5qEloUO3SM_1.mid (similarity: 0.842, emotion: sad)
5. Q3_PXOWy7NiZhk_0.mid (similarity: 0.834, emotion: sad)

Test Accuracy: 100.0% (5/5 correct predictions)
Query emotion category: sad

System completed successfully!
Results saved in results/ director
--------------------------------------------------------------

Track-14-Q3_veG92Oi-DlU_0
1. Survey Results
Happy:6.7%
Sad:50%
Hopeful:23.3%
Fearful:3.3%
Tense:6.7%
Excited:10%

2. System Results:
Music Emotion Analysis and Recommendation System
==================================================

1. Extracting features from MIDI files...
Found 1078 MIDI files
Processing file 1/1078
Processing file 101/1078
Processing file 201/1078
Processing file 301/1078
Processing file 401/1078
Processing file 501/1078
Processing file 601/1078
Processing file 701/1078
Processing file 801/1078
Processing file 901/1078
Processing file 1001/1078
Extracted features from 1078 files

2. Classifying emotions...

2.5. Exploratory Data Analysis (Classified Dataset)...

============================================================
EXPLORATORY DATA ANALYSIS: Classified EMOPIA Dataset
============================================================

1. BASIC DATASET INFORMATION
========================================
Total samples: 1078
Total features: 23
Memory usage: 0.42 MB
Missing values: 0
Duplicate rows: 0

Data types:
float64    13
int64       8
object      2
Name: count, dtype: int64

2. EMOTION DISTRIBUTION ANALYSIS
========================================
Emotion distribution:
  happy: 553 samples (51.3%)
  sad: 338 samples (31.4%)
  hopeful: 174 samples (16.1%)
  tense: 13 samples (1.2%)

Dataset balance analysis:
  Most frequent emotion: happy (553 samples)
  Least frequent emotion: tense (13 samples)
  Balance ratio: 0.024 (1.0 = perfectly balanced)
  WARNING: Dataset is highly imbalanced (ratio < 0.1)

3. MUSICAL FEATURE ANALYSIS
========================================
Number of numeric features: 21

Feature statistics:
       duration  pitch_min  ...  dynamic_range  dynamic_contrast
count  1078.000   1078.000  ...       1078.000          1078.000
mean     39.924     34.680  ...         61.499            11.939
std      10.475      8.154  ...         11.879             2.714
min      18.009     22.000  ...         28.000             4.896
25%      29.990     28.000  ...         53.000            10.017
50%      39.584     34.500  ...         62.000            11.762
75%      46.725     40.000  ...         69.000            13.607
max     109.990     69.000  ...         99.000            24.903

[8 rows x 21 columns]

Feature correlation with emotions:
  happy: ['total_notes(266.995)', 'chord_changes(145.190)', 'harmony_complexity(145.190)']
  sad: ['total_notes(235.935)', 'tempo_mean(120.000)', 'chord_changes(115.003)']    
  hopeful: ['total_notes(305.592)', 'chord_changes(170.138)', 'harmony_complexity(170.138)']
  tense: ['total_notes(390.077)', 'chord_changes(247.077)', 'harmony_complexity(247.077)']

4. FEATURE CORRELATION ANALYSIS
========================================

Highly correlated feature pairs (|r| > 0.7):
  velocity_std <-> dynamic_contrast: 1.000
  chord_changes <-> harmony_complexity: 1.000
  chord_changes <-> total_notes: 0.952
  harmony_complexity <-> total_notes: 0.952
  velocity_max <-> velocity_mean: 0.892
  rhythm_mean <-> rhythm_std: 0.821
  velocity_min <-> velocity_mean: 0.781

5. STATISTICAL TESTS
========================================
ANOVA tests for feature differences across emotions:
C:\Users\perer\AppData\Local\Programs\Python\Python311\Lib\site-packages\scipy\stats\_stats_py.py:4167: ConstantInputWarning: Each of the input arrays is constant;the F statistic is not defined or infinite
  warnings.warn(stats.ConstantInputWarning(msg))

Top 10 most significant features (ANOVA):
   1. pitch_mean: F=133.385, p=0.000000 ***
   2. pitch_max: F=100.888, p=0.000000 ***
   3. velocity_mean: F=73.824, p=0.000000 ***
   4. velocity_max: F=61.414, p=0.000000 ***
   5. note_overlap_ratio: F=42.039, p=0.000000 ***
   6. velocity_min: F=36.637, p=0.000000 ***
   7. pitch_std: F=28.778, p=0.000000 ***
   8. syncopation: F=24.819, p=0.000000 ***
   9. chord_changes: F=14.426, p=0.000000 ***
  10. harmony_complexity: F=14.426, p=0.000000 ***

Detailed ANOVA results saved to: results/anova_results.csv

EDA completed! Results saved in results/
Emotion distribution:
emotion
happy      553
sad        338
hopeful    174
tense       13
Name: count, dtype: int64

3. Balancing dataset with GAN...
Epoch 0, D Loss: 0.6804, G Loss: 0.6632
Epoch 20, D Loss: 0.9377, G Loss: 0.3417
Epoch 40, D Loss: 0.9459, G Loss: 0.3365
Generating 215 samples for sad
Generating 379 samples for hopeful
Generating 540 samples for tense
Balanced dataset emotion distribution:
emotion
happy      553
sad        553
hopeful    553
tense      553
Name: count, dtype: int64

3.5. Exploratory Data Analysis (Balanced Dataset)...

============================================================
EXPLORATORY DATA ANALYSIS: Balanced Dataset (After GAN)
============================================================

1. BASIC DATASET INFORMATION
========================================
Total samples: 2212
Total features: 23
Memory usage: 0.70 MB
Missing values: 1134
Duplicate rows: 0

Data types:
float64    21
object      2
Name: count, dtype: int64

2. EMOTION DISTRIBUTION ANALYSIS
========================================
Emotion distribution:
  happy: 553 samples (25.0%)
  sad: 553 samples (25.0%)
  hopeful: 553 samples (25.0%)
  tense: 553 samples (25.0%)

Dataset balance analysis:
  Most frequent emotion: happy (553 samples)
  Least frequent emotion: happy (553 samples)
  Balance ratio: 1.000 (1.0 = perfectly balanced)
  OK: Dataset is reasonably balanced (ratio >= 0.5)

3. MUSICAL FEATURE ANALYSIS
========================================
Number of numeric features: 21

Feature statistics:
       duration  pitch_min  ...  dynamic_range  dynamic_contrast
count  2212.000   2212.000  ...       2212.000          2212.000
mean     35.530     29.611  ...         56.034            10.426
std       8.851      7.648  ...          9.914             2.415
min      18.009     22.000  ...         28.000             4.896
25%      29.990     23.000  ...         49.500             9.229
50%      34.543     26.566  ...         52.191             9.271
75%      39.419     34.000  ...         61.000            11.698
max     109.990     69.000  ...         99.000            24.903

[8 rows x 21 columns]

Feature correlation with emotions:
  happy: ['total_notes(266.995)', 'chord_changes(145.190)', 'harmony_complexity(145.190)']
  sad: ['total_notes(186.155)', 'chord_changes(145.820)', 'tempo_mean(120.387)']
  hopeful: ['chord_changes(253.815)', 'total_notes(184.809)', 'tempo_mean(120.682)']
  tense: ['chord_changes(325.119)', 'total_notes(278.292)', 'harmony_complexity(161.193)']

4. FEATURE CORRELATION ANALYSIS
========================================

Highly correlated feature pairs (|r| > 0.7):
  tempo_mean <-> tempo_variation: 1.000
  tempo_std <-> tempo_variation: -1.000
  tempo_mean <-> tempo_std: -1.000
  harmony_complexity <-> total_notes: 0.965
  velocity_max <-> velocity_mean: 0.953
  rhythm_mean <-> rhythm_std: 0.897
  dynamic_range <-> dynamic_contrast: 0.784
  chord_changes <-> rhythm_mean: -0.750
  velocity_min <-> velocity_mean: 0.742
  chord_changes <-> rhythm_std: -0.710

5. STATISTICAL TESTS
========================================
ANOVA tests for feature differences across emotions:

Top 10 most significant features (ANOVA):
   1. pitch_mean: F=745.026, p=0.000000 ***
   2. velocity_min: F=1057.114, p=0.000000 ***
   3. velocity_max: F=964.877, p=0.000000 ***
   4. velocity_mean: F=1277.826, p=0.000000 ***
   5. tempo_mean: F=808.480, p=0.000000 ***
   6. tempo_std: F=808.666, p=0.000000 ***
   7. tempo_variation: F=808.261, p=0.000000 ***
   8. chord_changes: F=594.053, p=0.000000 ***
   9. pitch_min: F=350.247, p=0.000000 ***
  10. rhythm_mean: F=330.384, p=0.000000 ***

Detailed ANOVA results saved to: results/anova_results.csv

EDA completed! Results saved in results/

6. DATASET COMPARISON (Original vs Balanced)
==================================================

Emotion distribution comparison:
         Original  Balanced
emotion
happy         553       553
sad           338       553
hopeful       174       553
tense          13       553

Balance improvement:
  Original balance ratio: 0.024
  Balanced balance ratio: 1.000
  Improvement: 4153.8%

3.6. Model Training Visualizations...

7. MODEL TRAINING VISUALIZATION
==================================================

Training Analysis Summary:
  Total epochs: 50
  Final D loss: 0.9401
  Final G loss: 0.3360
  D loss range: 0.6804 - 0.9534
  G loss range: 0.3360 - 0.6632
  Training stability: D_std=0.0621, G_std=0.0712
  Training quality: STABLE
  Training visualization saved to: results/gan_training_analysis.png

8. MODEL ARCHITECTURE VISUALIZATION
==================================================
Model architecture visualization saved to: results/model_architecture.png

9. FEATURE IMPORTANCE ANALYSIS
==================================================
Feature importance analysis saved to: results/feature_importance.png
Top 5 most important features:
  1. velocity_mean: 6.932
  2. velocity_min: 5.735
  3. velocity_max: 5.234
  4. tempo_std: 4.387
  5. tempo_mean: 4.386

4. Training recommendation system...
Training recommendation system...
Trained on 2212 samples
Feature dimensions: 21
Recommendation system stats:
{'total_samples': 2212, 'emotion_distribution': {'happy': 553, 'sad': 553, 'hopeful': 553, 'tense': 553}, 'feature_dimensions': 21}

5. Testing recommendation system...
Query file: ../EMOPIA_1.0/midis\Q3_veG92Oi-DlU_0.mid
Query emotion category: sad

Top 5 similar tracks:
1. Q3_g-yM0Lsp4lc_0.mid (similarity: 0.965, emotion: sad)
2. Q4_mGYKWCVnMwQ_0.mid (similarity: 0.940, emotion: sad)
3. Q4_xgtwQGeB6_0_0.mid (similarity: 0.923, emotion: sad)
4. Q3_TonQX8XbvX8_2.mid (similarity: 0.892, emotion: sad)
5. Q4_mGYKWCVnMwQ_1.mid (similarity: 0.889, emotion: sad)

Test Accuracy: 100.0% (5/5 correct predictions)
Query emotion category: sad

System completed successfully!
Results saved in results/ directory
--------------------------------------------------------------

Track-15-Q1_5Ju9q1N2x0E_2
1. Survey Results
Happy:16.7%
Sad:10%
Hopeful:60%
Fearful:0%
Tense:0%
Excited:13.3%

2. System Results:
Music Emotion Analysis and Recommendation System
==================================================

1. Extracting features from MIDI files...
Found 1078 MIDI files
Processing file 1/1078
Processing file 101/1078
Processing file 201/1078
Processing file 301/1078
Processing file 401/1078
Processing file 501/1078
Processing file 601/1078
Processing file 701/1078
Processing file 801/1078
Processing file 901/1078
Processing file 1001/1078
Extracted features from 1078 files

2. Classifying emotions...

2.5. Exploratory Data Analysis (Classified Dataset)...

============================================================
EXPLORATORY DATA ANALYSIS: Classified EMOPIA Dataset
============================================================

1. BASIC DATASET INFORMATION
========================================
Total samples: 1078
Total features: 23
Memory usage: 0.42 MB
Missing values: 0
Duplicate rows: 0

Data types:
float64    13
int64       8
object      2
Name: count, dtype: int64

2. EMOTION DISTRIBUTION ANALYSIS
========================================
Emotion distribution:
  happy: 553 samples (51.3%)
  sad: 338 samples (31.4%)
  hopeful: 174 samples (16.1%)
  tense: 13 samples (1.2%)

Dataset balance analysis:
  Most frequent emotion: happy (553 samples)
  Least frequent emotion: tense (13 samples)
  Balance ratio: 0.024 (1.0 = perfectly balanced)
  WARNING: Dataset is highly imbalanced (ratio < 0.1)

3. MUSICAL FEATURE ANALYSIS
========================================
Number of numeric features: 21

Feature statistics:
       duration  pitch_min  pitch_max  ...  tempo_variation  dynamic_range  dynamic_contrast
count  1078.000   1078.000   1078.000  ...           1078.0       1078.000          1078.000  
mean     39.924     34.680     85.633  ...              0.0         61.499            11.939  
std      10.475      8.154      8.248  ...              0.0         11.879             2.714  
min      18.009     22.000     63.000  ...              0.0         28.000             4.896  
25%      29.990     28.000     80.000  ...              0.0         53.000            10.017  
50%      39.584     34.500     85.000  ...              0.0         62.000            11.762  
75%      46.725     40.000     91.000  ...              0.0         69.000            13.607  
max     109.990     69.000    105.000  ...              0.0         99.000            24.903  

[8 rows x 21 columns]

Feature correlation with emotions:
  happy: ['total_notes(266.995)', 'chord_changes(145.190)', 'harmony_complexity(145.190)']
  sad: ['total_notes(235.935)', 'tempo_mean(120.000)', 'chord_changes(115.003)']
  hopeful: ['total_notes(305.592)', 'chord_changes(170.138)', 'harmony_complexity(170.138)']  
  tense: ['total_notes(390.077)', 'chord_changes(247.077)', 'harmony_complexity(247.077)']    

4. FEATURE CORRELATION ANALYSIS
========================================

Highly correlated feature pairs (|r| > 0.7):
  velocity_std <-> dynamic_contrast: 1.000
  chord_changes <-> harmony_complexity: 1.000
  chord_changes <-> total_notes: 0.952
  harmony_complexity <-> total_notes: 0.952
  velocity_max <-> velocity_mean: 0.892
  rhythm_mean <-> rhythm_std: 0.821
  velocity_min <-> velocity_mean: 0.781

5. STATISTICAL TESTS
========================================
ANOVA tests for feature differences across emotions:
C:\Users\perer\AppData\Local\Programs\Python\Python311\Lib\site-packages\scipy\stats\_stats_py.py:4167: ConstantInputWarning: Each of the input arrays is constant;the F statistic is not defined or infinite
  warnings.warn(stats.ConstantInputWarning(msg))

Top 10 most significant features (ANOVA):
   1. pitch_mean: F=133.385, p=0.000000 ***
   2. pitch_max: F=100.888, p=0.000000 ***
   3. velocity_mean: F=73.824, p=0.000000 ***
   4. velocity_max: F=61.414, p=0.000000 ***
   5. note_overlap_ratio: F=42.039, p=0.000000 ***
   6. velocity_min: F=36.637, p=0.000000 ***
   7. pitch_std: F=28.778, p=0.000000 ***
   8. syncopation: F=24.819, p=0.000000 ***
   9. chord_changes: F=14.426, p=0.000000 ***
  10. harmony_complexity: F=14.426, p=0.000000 ***

Detailed ANOVA results saved to: results/anova_results.csv

EDA completed! Results saved in results/
Emotion distribution:
emotion
happy      553
sad        338
hopeful    174
tense       13
Name: count, dtype: int64

3. Balancing dataset with GAN...
Epoch 0, D Loss: 0.7071, G Loss: 0.6161
Epoch 20, D Loss: 0.8554, G Loss: 0.4181
Epoch 40, D Loss: 0.8706, G Loss: 0.4148
Generating 215 samples for sad
Generating 379 samples for hopeful
Generating 540 samples for tense
Balanced dataset emotion distribution:
emotion
happy      553
sad        553
hopeful    553
tense      553
Name: count, dtype: int64

3.5. Exploratory Data Analysis (Balanced Dataset)...

============================================================
EXPLORATORY DATA ANALYSIS: Balanced Dataset (After GAN)
============================================================

1. BASIC DATASET INFORMATION
========================================
Total samples: 2212
Total features: 23
Memory usage: 0.70 MB
Missing values: 1134
Duplicate rows: 0

Data types:
float64    21
object      2
Name: count, dtype: int64

2. EMOTION DISTRIBUTION ANALYSIS
========================================
Emotion distribution:
  happy: 553 samples (25.0%)
  sad: 553 samples (25.0%)
  hopeful: 553 samples (25.0%)
  tense: 553 samples (25.0%)

Dataset balance analysis:
  Most frequent emotion: happy (553 samples)
  Least frequent emotion: happy (553 samples)
  Balance ratio: 1.000 (1.0 = perfectly balanced)
  OK: Dataset is reasonably balanced (ratio >= 0.5)

3. MUSICAL FEATURE ANALYSIS
========================================
Number of numeric features: 21

Feature statistics:
       duration  pitch_min  pitch_max  ...  tempo_variation  dynamic_range  dynamic_contrast
count  2212.000   2212.000   2212.000  ...         2212.000       2212.000          2212.000  
mean     44.809     29.956     88.473  ...            0.511         55.815            10.995  
std       8.920      7.489      6.680  ...            0.499         10.020             2.132  
min      18.009     22.000     63.000  ...            0.000         28.000             4.896  
25%      39.752     24.000     85.000  ...            0.000         49.194             9.840  
50%      46.778     27.401     90.240  ...            0.992         51.805            10.329  
75%      51.743     34.000     94.626  ...            0.998         61.000            11.699  
max     109.990     69.000    105.000  ...            1.000         99.000            24.903  

[8 rows x 21 columns]

Feature correlation with emotions:
  happy: ['total_notes(266.995)', 'chord_changes(145.190)', 'harmony_complexity(145.190)']    
  sad: ['total_notes(186.266)', 'harmony_complexity(148.781)', 'chord_changes(148.757)']      
  hopeful: ['harmony_complexity(262.214)', 'chord_changes(262.134)', 'total_notes(185.042)']  
  tense: ['harmony_complexity(332.922)', 'chord_changes(332.892)', 'total_notes(278.471)']    

4. FEATURE CORRELATION ANALYSIS
========================================

Highly correlated feature pairs (|r| > 0.7):
  chord_changes <-> harmony_complexity: 1.000
  tempo_mean <-> tempo_variation: -1.000
  tempo_mean <-> tempo_std: 0.947
  tempo_std <-> tempo_variation: -0.947
  velocity_std <-> dynamic_contrast: 0.919
  rhythm_mean <-> rhythm_std: 0.896
  velocity_min <-> velocity_max: 0.836
  pitch_std <-> velocity_max: 0.783
  velocity_max <-> velocity_mean: 0.771
  pitch_min <-> pitch_std: -0.763
  harmony_complexity <-> rhythm_mean: -0.757
  chord_changes <-> rhythm_mean: -0.757
  dynamic_range <-> dynamic_contrast: 0.744
  velocity_max <-> harmony_complexity: 0.738
  velocity_max <-> chord_changes: 0.738
  velocity_min <-> velocity_mean: 0.737
  pitch_std <-> chord_changes: 0.730
  pitch_std <-> harmony_complexity: 0.730
  pitch_std <-> velocity_min: 0.729
  harmony_complexity <-> rhythm_std: -0.717
  chord_changes <-> rhythm_std: -0.717

5. STATISTICAL TESTS
========================================
ANOVA tests for feature differences across emotions:

Top 10 most significant features (ANOVA):
   1. pitch_mean: F=747.325, p=0.000000 ***
   2. velocity_min: F=1056.918, p=0.000000 ***
   3. velocity_max: F=850.869, p=0.000000 ***
   4. velocity_mean: F=1270.391, p=0.000000 ***
   5. tempo_mean: F=808.294, p=0.000000 ***
   6. tempo_variation: F=808.376, p=0.000000 ***
   7. tempo_std: F=635.823, p=0.000000 ***
   8. harmony_complexity: F=623.099, p=0.000000 ***
   9. chord_changes: F=623.061, p=0.000000 ***
  10. pitch_std: F=556.442, p=0.000000 ***

Detailed ANOVA results saved to: results/anova_results.csv

EDA completed! Results saved in results/

6. DATASET COMPARISON (Original vs Balanced)
==================================================

Emotion distribution comparison:
         Original  Balanced
emotion
happy         553       553
sad           338       553
hopeful       174       553
tense          13       553

Balance improvement:
  Original balance ratio: 0.024
  Balanced balance ratio: 1.000
  Improvement: 4153.8%

3.6. Model Training Visualizations...

7. MODEL TRAINING VISUALIZATION
==================================================

Training Analysis Summary:
  Total epochs: 50
  Final D loss: 0.8571
  Final G loss: 0.4109
  D loss range: 0.7071 - 0.8706
  G loss range: 0.4109 - 0.6161
  Training stability: D_std=0.0337, G_std=0.0411
  Training quality: STABLE
  Training visualization saved to: results/gan_training_analysis.png

8. MODEL ARCHITECTURE VISUALIZATION
==================================================
Model architecture visualization saved to: results/model_architecture.png

9. FEATURE IMPORTANCE ANALYSIS
==================================================
Feature importance analysis saved to: results/feature_importance.png
Top 5 most important features:
  1. velocity_mean: 6.892
  2. velocity_min: 5.734
  3. velocity_max: 4.616
  5. tempo_mean: 4.385
  5. tempo_mean: 4.385

4. Training recommendation system...
Training recommendation system...
Trained on 2212 samples
Feature dimensions: 21
Recommendation system stats:
{'total_samples': 2212, 'emotion_distribution': {'happy': 553, 'sad': 553, 'hopeful': 553, 'tense': 553}, 'feature_dimensions': 21}

5. Testing recommendation system...
Query file: ../EMOPIA_1.0/midis\Q1_5Ju9q1N2x0E_2.mid
Query emotion category: happy

Top 5 similar tracks:
1. Q1_Ie5koh4qvJc_6.mid (similarity: 0.852, emotion: happy)
2. Q4_ldCQ6N9G6Mk_5.mid (similarity: 0.812, emotion: happy)
3. Q1_yFw_kO7DF-Y_2.mid (similarity: 0.768, emotion: happy)
4. Q3_bbU31JLtlug_1.mid (similarity: 0.765, emotion: happy)
5. Q2_CxFXrYgdSSI_2.mid (similarity: 0.758, emotion: happy)

Test Accuracy: 100.0% (5/5 correct predictions)
Query emotion category: happy

System completed successfully!
Results saved in results/ directory
--------------------------------------------------------------

Track-16-Q3_vVTP0DOL_2Q_0
1. Survey Results
Happy:10%
Sad:66.7%
Hopeful:6.7%
Fearful:3.3%
Tense:10%
Excited:3.3%

2. System Results:
Music Emotion Analysis and Recommendation System
==================================================

1. Extracting features from MIDI files...
Found 1078 MIDI files
Processing file 1/1078
Processing file 101/1078
Processing file 201/1078
Processing file 301/1078
Processing file 401/1078
Processing file 501/1078
Processing file 601/1078
Processing file 701/1078
Processing file 801/1078
Processing file 901/1078
Processing file 1001/1078
Extracted features from 1078 files

2. Classifying emotions...

2.5. Exploratory Data Analysis (Classified Dataset)...

============================================================
EXPLORATORY DATA ANALYSIS: Classified EMOPIA Dataset
============================================================

1. BASIC DATASET INFORMATION
========================================
Total samples: 1078
Total features: 23
Memory usage: 0.42 MB
Missing values: 0
Duplicate rows: 0

Data types:
float64    13
int64       8
object      2
Name: count, dtype: int64

2. EMOTION DISTRIBUTION ANALYSIS
========================================
Emotion distribution:
  happy: 553 samples (51.3%)
  sad: 338 samples (31.4%)
  hopeful: 174 samples (16.1%)
  tense: 13 samples (1.2%)

Dataset balance analysis:
  Most frequent emotion: happy (553 samples)
  Least frequent emotion: tense (13 samples)
  Balance ratio: 0.024 (1.0 = perfectly balanced)
  WARNING: Dataset is highly imbalanced (ratio < 0.1)

3. MUSICAL FEATURE ANALYSIS
========================================
Number of numeric features: 21

Feature statistics:
       duration  pitch_min  pitch_max  ...  tempo_variation  dynamic_range  dynamic_contrast
count  1078.000   1078.000   1078.000  ...           1078.0       1078.000          1078.000  
mean     39.924     34.680     85.633  ...              0.0         61.499            11.939  
std      10.475      8.154      8.248  ...              0.0         11.879             2.714  
min      18.009     22.000     63.000  ...              0.0         28.000             4.896  
25%      29.990     28.000     80.000  ...              0.0         53.000            10.017  
50%      39.584     34.500     85.000  ...              0.0         62.000            11.762  
75%      46.725     40.000     91.000  ...              0.0         69.000            13.607  
max     109.990     69.000    105.000  ...              0.0         99.000            24.903  

[8 rows x 21 columns]

Feature correlation with emotions:
  happy: ['total_notes(266.995)', 'chord_changes(145.190)', 'harmony_complexity(145.190)']    
  sad: ['total_notes(235.935)', 'tempo_mean(120.000)', 'chord_changes(115.003)']
  hopeful: ['total_notes(305.592)', 'chord_changes(170.138)', 'harmony_complexity(170.138)']  
  tense: ['total_notes(390.077)', 'chord_changes(247.077)', 'harmony_complexity(247.077)']    

4. FEATURE CORRELATION ANALYSIS
========================================

Highly correlated feature pairs (|r| > 0.7):
  velocity_std <-> dynamic_contrast: 1.000
  chord_changes <-> harmony_complexity: 1.000
  chord_changes <-> total_notes: 0.952
  harmony_complexity <-> total_notes: 0.952
  velocity_max <-> velocity_mean: 0.892
  rhythm_mean <-> rhythm_std: 0.821
  velocity_min <-> velocity_mean: 0.781

5. STATISTICAL TESTS
========================================
ANOVA tests for feature differences across emotions:
C:\Users\perer\AppData\Local\Programs\Python\Python311\Lib\site-packages\scipy\stats\_stats_py.py:4167: ConstantInputWarning: Each of the input arrays is constant;the F statistic is not defined or infinite
  warnings.warn(stats.ConstantInputWarning(msg))

Top 10 most significant features (ANOVA):
   1. pitch_mean: F=133.385, p=0.000000 ***
   2. pitch_max: F=100.888, p=0.000000 ***
   3. velocity_mean: F=73.824, p=0.000000 ***
   4. velocity_max: F=61.414, p=0.000000 ***
   5. note_overlap_ratio: F=42.039, p=0.000000 ***
   6. velocity_min: F=36.637, p=0.000000 ***
   7. pitch_std: F=28.778, p=0.000000 ***
   8. syncopation: F=24.819, p=0.000000 ***
   9. chord_changes: F=14.426, p=0.000000 ***
  10. harmony_complexity: F=14.426, p=0.000000 ***

Detailed ANOVA results saved to: results/anova_results.csv

EDA completed! Results saved in results/
Emotion distribution:
emotion
happy      553
sad        338
hopeful    174
tense       13
Name: count, dtype: int64

3. Balancing dataset with GAN...
Epoch 0, D Loss: 0.7194, G Loss: 0.6634
Epoch 20, D Loss: 0.8682, G Loss: 0.4540
Epoch 40, D Loss: 0.8849, G Loss: 0.4496
Generating 215 samples for sad
Generating 379 samples for hopeful
Generating 540 samples for tense
Balanced dataset emotion distribution:
emotion
happy      553
sad        553
hopeful    553
tense      553
Name: count, dtype: int64

3.5. Exploratory Data Analysis (Balanced Dataset)...

============================================================
EXPLORATORY DATA ANALYSIS: Balanced Dataset (After GAN)
============================================================

1. BASIC DATASET INFORMATION
========================================
Total samples: 2212
Total features: 23
Memory usage: 0.70 MB
Missing values: 1134
Duplicate rows: 0

Data types:
float64    21
object      2
Name: count, dtype: int64

2. EMOTION DISTRIBUTION ANALYSIS
========================================
Emotion distribution:
  happy: 553 samples (25.0%)
  sad: 553 samples (25.0%)
  hopeful: 553 samples (25.0%)
  tense: 553 samples (25.0%)

Dataset balance analysis:
  Most frequent emotion: happy (553 samples)
  Least frequent emotion: happy (553 samples)
  Balance ratio: 1.000 (1.0 = perfectly balanced)
  OK: Dataset is reasonably balanced (ratio >= 0.5)

3. MUSICAL FEATURE ANALYSIS
========================================
Number of numeric features: 21

Feature statistics:
       duration  pitch_min  pitch_max  ...  tempo_variation  dynamic_range  dynamic_contrast
count  2212.000   2212.000   2212.000  ...         2212.000       2212.000          2212.000  
mean     44.836     34.115     80.624  ...            0.511         55.796            10.817  
std       8.936      7.243      7.820  ...            0.499         10.030             2.207  
min      18.009     22.000     63.000  ...            0.000         28.000             4.896  
25%      39.752     27.285     74.175  ...            0.000         49.185             9.583  
50%      46.796     35.000     79.716  ...            0.990         51.771             9.975  
75%      51.806     39.754     84.000  ...            0.999         61.000            11.698  
max     109.990     69.000    105.000  ...            1.000         99.000            24.903  

[8 rows x 21 columns]

Feature correlation with emotions:
  happy: ['total_notes(266.995)', 'chord_changes(145.190)', 'harmony_complexity(145.190)']
  sad: ['total_notes(285.641)', 'tempo_mean(119.613)', 'harmony_complexity(102.519)']
  hopeful: ['total_notes(426.256)', 'harmony_complexity(136.526)', 'tempo_mean(119.318)']     
  tense: ['total_notes(501.766)', 'harmony_complexity(215.764)', 'chord_changes(189.406)']    

4. FEATURE CORRELATION ANALYSIS
========================================

Highly correlated feature pairs (|r| > 0.7):
  tempo_mean <-> tempo_variation: -1.000
  tempo_std <-> tempo_variation: 1.000
  tempo_mean <-> tempo_std: -1.000
  chord_changes <-> harmony_complexity: 0.965
  velocity_max <-> velocity_mean: 0.953
  dynamic_range <-> dynamic_contrast: 0.764
  syncopation <-> total_notes: 0.754
  syncopation <-> note_overlap_ratio: -0.753
  velocity_min <-> velocity_mean: 0.738
  harmony_complexity <-> total_notes: 0.729
  rhythm_std <-> total_notes: -0.706

5. STATISTICAL TESTS
========================================
ANOVA tests for feature differences across emotions:

Top 10 most significant features (ANOVA):
   1. pitch_mean: F=746.779, p=0.000000 ***
   2. velocity_min: F=1057.979, p=0.000000 ***
   3. velocity_max: F=965.182, p=0.000000 ***
   4. velocity_mean: F=1271.802, p=0.000000 ***
   5. tempo_mean: F=808.784, p=0.000000 ***
   6. tempo_std: F=808.652, p=0.000000 ***
   7. tempo_variation: F=808.406, p=0.000000 ***
   8. syncopation: F=705.684, p=0.000000 ***
   9. note_overlap_ratio: F=670.083, p=0.000000 ***
  10. pitch_max: F=665.177, p=0.000000 ***

Detailed ANOVA results saved to: results/anova_results.csv

EDA completed! Results saved in results/

6. DATASET COMPARISON (Original vs Balanced)
==================================================

Emotion distribution comparison:
         Original  Balanced
emotion
happy         553       553
sad           338       553
hopeful       174       553
tense          13       553

Balance improvement:
  Original balance ratio: 0.024
  Balanced balance ratio: 1.000
  Improvement: 4153.8%

3.6. Model Training Visualizations...

7. MODEL TRAINING VISUALIZATION
==================================================

Training Analysis Summary:
  Total epochs: 50
  Final D loss: 0.8629
  Final G loss: 0.4491
  D loss range: 0.7194 - 0.8928
  G loss range: 0.4487 - 0.6634
  Training stability: D_std=0.0335, G_std=0.0458
  Training quality: STABLE
  Training visualization saved to: results/gan_training_analysis.png

8. MODEL ARCHITECTURE VISUALIZATION
==================================================
Model architecture visualization saved to: results/model_architecture.png

9. FEATURE IMPORTANCE ANALYSIS
==================================================
Feature importance analysis saved to: results/feature_importance.png
Top 5 most important features:
  1. velocity_mean: 6.899
  2. velocity_min: 5.739
  3. velocity_max: 5.236
  4. tempo_mean: 4.388
  5. tempo_std: 4.387

4. Training recommendation system...
Training recommendation system...
Trained on 2212 samples
Feature dimensions: 21
Recommendation system stats:
{'total_samples': 2212, 'emotion_distribution': {'happy': 553, 'sad': 553, 'hopeful': 553, 'tense': 553}, 'feature_dimensions': 21}

5. Testing recommendation system...
Query file: ../EMOPIA_1.0/midis\Q3_vVTP0DOL_2Q_0.mid
Query emotion category: sad

Top 5 similar tracks:
1. Q3_vVTP0DOL_2Q_3.mid (similarity: 0.985, emotion: sad)
2. Q3_vVTP0DOL_2Q_1.mid (similarity: 0.969, emotion: sad)
3. Q3_wfXSdMsd4q8_0.mid (similarity: 0.951, emotion: sad)
4. Q3_K84TcgjCRt4_0.mid (similarity: 0.920, emotion: sad)
5. Q3_gvWDOIiocuE_2.mid (similarity: 0.899, emotion: sad)

Test Accuracy: 100.0% (5/5 correct predictions)
Query emotion category: sad

System completed successfully!
Results saved in results/ directory
--------------------------------------------------------------

Track-17-Q3_wlOa1jkTn_U_2
1. Survey Results
Happy:0%
Sad:36.7%
Hopeful:16.7%
Fearful:23.3%
Tense:16.7%
Excited:6.7%

2. System Results:
Music Emotion Analysis and Recommendation System
==================================================

1. Extracting features from MIDI files...
Found 1078 MIDI files
Processing file 1/1078
Processing file 101/1078
Processing file 201/1078
Processing file 301/1078
Processing file 401/1078
Processing file 501/1078
Processing file 601/1078
Processing file 701/1078
Processing file 801/1078
Processing file 901/1078
Processing file 1001/1078
Extracted features from 1078 files

2. Classifying emotions...

2.5. Exploratory Data Analysis (Classified Dataset)...

============================================================
EXPLORATORY DATA ANALYSIS: Classified EMOPIA Dataset
============================================================

1. BASIC DATASET INFORMATION
========================================
Total samples: 1078
Total features: 23
Memory usage: 0.42 MB
Missing values: 0
Duplicate rows: 0

Data types:
float64    13
int64       8
object      2
Name: count, dtype: int64

2. EMOTION DISTRIBUTION ANALYSIS
========================================
Emotion distribution:
  happy: 553 samples (51.3%)
  sad: 338 samples (31.4%)
  hopeful: 174 samples (16.1%)
  tense: 13 samples (1.2%)

Dataset balance analysis:
  Most frequent emotion: happy (553 samples)
  Least frequent emotion: tense (13 samples)
  Balance ratio: 0.024 (1.0 = perfectly balanced)
  WARNING: Dataset is highly imbalanced (ratio < 0.1)

3. MUSICAL FEATURE ANALYSIS
========================================
Number of numeric features: 21

Feature statistics:
       duration  pitch_min  pitch_max  ...  tempo_variation  dynamic_range  dynamic_contrast
count  1078.000   1078.000   1078.000  ...           1078.0       1078.000          1078.000  
mean     39.924     34.680     85.633  ...              0.0         61.499            11.939  
std      10.475      8.154      8.248  ...              0.0         11.879             2.714  
min      18.009     22.000     63.000  ...              0.0         28.000             4.896  
25%      29.990     28.000     80.000  ...              0.0         53.000            10.017  
50%      39.584     34.500     85.000  ...              0.0         62.000            11.762  
75%      46.725     40.000     91.000  ...              0.0         69.000            13.607  
max     109.990     69.000    105.000  ...              0.0         99.000            24.903  

[8 rows x 21 columns]

Feature correlation with emotions:
  happy: ['total_notes(266.995)', 'chord_changes(145.190)', 'harmony_complexity(145.190)']
  sad: ['total_notes(235.935)', 'tempo_mean(120.000)', 'chord_changes(115.003)']
  hopeful: ['total_notes(305.592)', 'chord_changes(170.138)', 'harmony_complexity(170.138)']  
  tense: ['total_notes(390.077)', 'chord_changes(247.077)', 'harmony_complexity(247.077)']    

4. FEATURE CORRELATION ANALYSIS
========================================

Highly correlated feature pairs (|r| > 0.7):
  velocity_std <-> dynamic_contrast: 1.000
  chord_changes <-> harmony_complexity: 1.000
  chord_changes <-> total_notes: 0.952
  harmony_complexity <-> total_notes: 0.952
  velocity_max <-> velocity_mean: 0.892
  rhythm_mean <-> rhythm_std: 0.821
  velocity_min <-> velocity_mean: 0.781

5. STATISTICAL TESTS
========================================
ANOVA tests for feature differences across emotions:
C:\Users\perer\AppData\Local\Programs\Python\Python311\Lib\site-packages\scipy\stats\_stats_py.py:4167: ConstantInputWarning: Each of the input arrays is constant;the F statistic is not defined or infinite
  warnings.warn(stats.ConstantInputWarning(msg))

Top 10 most significant features (ANOVA):
   1. pitch_mean: F=133.385, p=0.000000 ***
   2. pitch_max: F=100.888, p=0.000000 ***
   3. velocity_mean: F=73.824, p=0.000000 ***
   4. velocity_max: F=61.414, p=0.000000 ***
   5. note_overlap_ratio: F=42.039, p=0.000000 ***
   6. velocity_min: F=36.637, p=0.000000 ***
   7. pitch_std: F=28.778, p=0.000000 ***
   8. syncopation: F=24.819, p=0.000000 ***
   9. chord_changes: F=14.426, p=0.000000 ***
  10. harmony_complexity: F=14.426, p=0.000000 ***

Detailed ANOVA results saved to: results/anova_results.csv

EDA completed! Results saved in results/
Emotion distribution:
emotion
happy      553
sad        338
hopeful    174
tense       13
Name: count, dtype: int64

3. Balancing dataset with GAN...
Epoch 0, D Loss: 0.7074, G Loss: 0.6296
Epoch 20, D Loss: 0.8753, G Loss: 0.3923
Epoch 40, D Loss: 0.8924, G Loss: 0.3879
Generating 215 samples for sad
Generating 379 samples for hopeful
Generating 540 samples for tense
Balanced dataset emotion distribution:
emotion
happy      553
sad        553
hopeful    553
tense      553
Name: count, dtype: int64

3.5. Exploratory Data Analysis (Balanced Dataset)...

============================================================
EXPLORATORY DATA ANALYSIS: Balanced Dataset (After GAN)
============================================================

1. BASIC DATASET INFORMATION
========================================
Total samples: 2212
Total features: 23
Memory usage: 0.70 MB
Missing values: 1134
Duplicate rows: 0

Data types:
float64    21
object      2
Name: count, dtype: int64

2. EMOTION DISTRIBUTION ANALYSIS
========================================
Emotion distribution:
  happy: 553 samples (25.0%)
  sad: 553 samples (25.0%)
  hopeful: 553 samples (25.0%)
  tense: 553 samples (25.0%)

Dataset balance analysis:
  Most frequent emotion: happy (553 samples)
  Least frequent emotion: happy (553 samples)
  Balance ratio: 1.000 (1.0 = perfectly balanced)
  OK: Dataset is reasonably balanced (ratio >= 0.5)

3. MUSICAL FEATURE ANALYSIS
========================================
Number of numeric features: 21

Feature statistics:
       duration  pitch_min  pitch_max  ...  tempo_variation  dynamic_range  dynamic_contrast
count  2212.000   2212.000   2212.000  ...         2212.000       2212.000          2212.000  
mean     35.534     29.595     88.275  ...           -0.245         70.138            12.892  
std       8.849      7.656      6.609  ...            0.264         12.606             2.145  
min      18.009     22.000     63.000  ...           -0.836         28.000             4.896  
25%      29.990     23.000     85.000  ...           -0.500         62.000            11.872  
50%      34.544     26.446     89.715  ...           -0.135         72.000            13.312  
75%      39.419     34.000     93.946  ...            0.000         84.138            14.305  
max     109.990     69.000    105.000  ...            0.106         99.000            24.903  

[8 rows x 21 columns]

Feature correlation with emotions:
  happy: ['total_notes(266.995)', 'chord_changes(145.190)', 'harmony_complexity(145.190)']    
  sad: ['total_notes(186.122)', 'tempo_mean(120.387)', 'chord_changes(118.519)']
  hopeful: ['total_notes(184.718)', 'chord_changes(179.169)', 'tempo_mean(120.682)']
  tense: ['total_notes(278.183)', 'chord_changes(255.257)', 'harmony_complexity(179.931)']    

4. FEATURE CORRELATION ANALYSIS
========================================

Highly correlated feature pairs (|r| > 0.7):
  velocity_std <-> dynamic_contrast: 0.998
  harmony_complexity <-> total_notes: 0.950
  velocity_max <-> velocity_mean: 0.949
  velocity_min <-> velocity_mean: 0.913
  tempo_mean <-> tempo_variation: -0.906
  chord_changes <-> harmony_complexity: 0.852
  velocity_min <-> velocity_max: 0.835
  syncopation <-> note_overlap_ratio: -0.766
  rhythm_mean <-> rhythm_std: 0.753
  chord_changes <-> total_notes: 0.743
  velocity_max <-> dynamic_range: 0.730
  tempo_mean <-> syncopation: 0.720
  rhythm_mean <-> total_notes: -0.717
  pitch_min <-> pitch_mean: 0.714

5. STATISTICAL TESTS
========================================
ANOVA tests for feature differences across emotions:

Top 10 most significant features (ANOVA):
   1. pitch_mean: F=969.955, p=0.000000 ***
   2. velocity_min: F=1054.001, p=0.000000 ***
   3. velocity_max: F=850.568, p=0.000000 ***
   4. velocity_mean: F=1168.183, p=0.000000 ***
   5. tempo_mean: F=809.027, p=0.000000 ***
   6. syncopation: F=769.802, p=0.000000 ***
   7. note_overlap_ratio: F=670.632, p=0.000000 ***
   8. dynamic_range: F=647.039, p=0.000000 ***
   9. tempo_variation: F=556.675, p=0.000000 ***
  10. pitch_min: F=350.305, p=0.000000 ***

Detailed ANOVA results saved to: results/anova_results.csv

EDA completed! Results saved in results/

6. DATASET COMPARISON (Original vs Balanced)
==================================================

Emotion distribution comparison:
         Original  Balanced
emotion
happy         553       553
sad           338       553
hopeful       174       553
tense          13       553

Balance improvement:
  Original balance ratio: 0.024
  Balanced balance ratio: 1.000
  Improvement: 4153.8%

3.6. Model Training Visualizations...

7. MODEL TRAINING VISUALIZATION
==================================================

Training Analysis Summary:
  Total epochs: 50
  Final D loss: 0.8929
  Final G loss: 0.3871
  D loss range: 0.7074 - 0.9006
  G loss range: 0.3868 - 0.6296
  Training stability: D_std=0.0400, G_std=0.0489
  Training quality: STABLE
  Training visualization saved to: results/gan_training_analysis.png

8. MODEL ARCHITECTURE VISUALIZATION
==================================================
Model architecture visualization saved to: results/model_architecture.png

9. FEATURE IMPORTANCE ANALYSIS
==================================================
Feature importance analysis saved to: results/feature_importance.png
Top 5 most important features:
  1. velocity_mean: 6.337
  2. velocity_min: 5.718
  3. pitch_mean: 5.262
  4. velocity_max: 4.614
  5. tempo_mean: 4.389

4. Training recommendation system...
Training recommendation system...
Trained on 2212 samples
Feature dimensions: 21
Recommendation system stats:
{'total_samples': 2212, 'emotion_distribution': {'happy': 553, 'sad': 553, 'hopeful': 553, 'tense': 553}, 'feature_dimensions': 21}

5. Testing recommendation system...
Query file: ../EMOPIA_1.0/midis\Q3_wlOa1jkTn_U_2.mid
Query emotion category: happy

Top 5 similar tracks:
1. Q2_S1T3UF1vhSk_1.mid (similarity: 0.906, emotion: happy)
2. Q1_60LLKmpgzRM_0.mid (similarity: 0.894, emotion: happy)
3. Q3_ilCcZpD39HU_2.mid (similarity: 0.894, emotion: happy)
4. Q4_NP0lwB-_-og_0.mid (similarity: 0.887, emotion: happy)
5. Q4_d_49EtXDMFE_2.mid (similarity: 0.878, emotion: happy)

Test Accuracy: 100.0% (5/5 correct predictions)
Query emotion category: happy

System completed successfully!
Results saved in results/ directory
--------------------------------------------------------------

Track-18-Q1__SJQaaRzD-A_1
1. Survey Results
Happy:43.3%
Sad:6.7%
Hopeful:33.3%
Fearful:0%
Tense:0%
Excited:16.7%

2. System Results:
Music Emotion Analysis and Recommendation System
==================================================

1. Extracting features from MIDI files...
Found 1078 MIDI files
Processing file 1/1078
Processing file 101/1078
Processing file 201/1078
Processing file 301/1078
Processing file 401/1078
Processing file 501/1078
Processing file 601/1078
Processing file 701/1078
Processing file 801/1078
Processing file 901/1078
Processing file 1001/1078
Extracted features from 1078 files

2. Classifying emotions...

2.5. Exploratory Data Analysis (Classified Dataset)...

============================================================
EXPLORATORY DATA ANALYSIS: Classified EMOPIA Dataset
============================================================

1. BASIC DATASET INFORMATION
========================================
Total samples: 1078
Total features: 23
Memory usage: 0.42 MB
Missing values: 0
Duplicate rows: 0

Data types:
float64    13
int64       8
object      2
Name: count, dtype: int64

2. EMOTION DISTRIBUTION ANALYSIS
========================================
Emotion distribution:
  happy: 553 samples (51.3%)
  sad: 338 samples (31.4%)
  hopeful: 174 samples (16.1%)
  tense: 13 samples (1.2%)

Dataset balance analysis:
  Most frequent emotion: happy (553 samples)
  Least frequent emotion: tense (13 samples)
  Balance ratio: 0.024 (1.0 = perfectly balanced)
  WARNING: Dataset is highly imbalanced (ratio < 0.1)

3. MUSICAL FEATURE ANALYSIS
========================================
Number of numeric features: 21

Feature statistics:
       duration  pitch_min  pitch_max  ...  tempo_variation  dynamic_range  dynamic_contrast
count  1078.000   1078.000   1078.000  ...           1078.0       1078.000          1078.000  
mean     39.924     34.680     85.633  ...              0.0         61.499            11.939  
std      10.475      8.154      8.248  ...              0.0         11.879             2.714  
min      18.009     22.000     63.000  ...              0.0         28.000             4.896  
25%      29.990     28.000     80.000  ...              0.0         53.000            10.017  
50%      39.584     34.500     85.000  ...              0.0         62.000            11.762  
75%      46.725     40.000     91.000  ...              0.0         69.000            13.607  
max     109.990     69.000    105.000  ...              0.0         99.000            24.903  

[8 rows x 21 columns]

Feature correlation with emotions:
  happy: ['total_notes(266.995)', 'chord_changes(145.190)', 'harmony_complexity(145.190)']    
  sad: ['total_notes(235.935)', 'tempo_mean(120.000)', 'chord_changes(115.003)']
  hopeful: ['total_notes(305.592)', 'chord_changes(170.138)', 'harmony_complexity(170.138)']  
  tense: ['total_notes(390.077)', 'chord_changes(247.077)', 'harmony_complexity(247.077)']    

4. FEATURE CORRELATION ANALYSIS
========================================

Highly correlated feature pairs (|r| > 0.7):
  velocity_std <-> dynamic_contrast: 1.000
  chord_changes <-> harmony_complexity: 1.000
  chord_changes <-> total_notes: 0.952
  harmony_complexity <-> total_notes: 0.952
  velocity_max <-> velocity_mean: 0.892
  rhythm_mean <-> rhythm_std: 0.821
  velocity_min <-> velocity_mean: 0.781

5. STATISTICAL TESTS
========================================
ANOVA tests for feature differences across emotions:
C:\Users\perer\AppData\Local\Programs\Python\Python311\Lib\site-packages\scipy\stats\_stats_py.py:4167: ConstantInputWarning: Each of the input arrays is constant;the F statistic is not defined or infinite
  warnings.warn(stats.ConstantInputWarning(msg))

Top 10 most significant features (ANOVA):
   1. pitch_mean: F=133.385, p=0.000000 ***
   2. pitch_max: F=100.888, p=0.000000 ***
   3. velocity_mean: F=73.824, p=0.000000 ***
   4. velocity_max: F=61.414, p=0.000000 ***
   5. note_overlap_ratio: F=42.039, p=0.000000 ***
   6. velocity_min: F=36.637, p=0.000000 ***
   7. pitch_std: F=28.778, p=0.000000 ***
   8. syncopation: F=24.819, p=0.000000 ***
   9. chord_changes: F=14.426, p=0.000000 ***
  10. harmony_complexity: F=14.426, p=0.000000 ***

Detailed ANOVA results saved to: results/anova_results.csv

EDA completed! Results saved in results/
Emotion distribution:
emotion
happy      553
sad        338
hopeful    174
tense       13
Name: count, dtype: int64

3. Balancing dataset with GAN...
Epoch 0, D Loss: 0.6831, G Loss: 0.6352
Epoch 20, D Loss: 0.9041, G Loss: 0.3658
Epoch 40, D Loss: 0.9048, G Loss: 0.3615
Generating 215 samples for sad
Generating 379 samples for hopeful
Generating 540 samples for tense
Balanced dataset emotion distribution:
emotion
happy      553
sad        553
hopeful    553
tense      553
Name: count, dtype: int64

3.5. Exploratory Data Analysis (Balanced Dataset)...

============================================================
EXPLORATORY DATA ANALYSIS: Balanced Dataset (After GAN)
============================================================

1. BASIC DATASET INFORMATION
========================================
Total samples: 2212
Total features: 23
Memory usage: 0.70 MB
Missing values: 1134
Duplicate rows: 0

Data types:
float64    21
object      2
Name: count, dtype: int64

2. EMOTION DISTRIBUTION ANALYSIS
========================================
Emotion distribution:
  happy: 553 samples (25.0%)
  sad: 553 samples (25.0%)
  hopeful: 553 samples (25.0%)
  tense: 553 samples (25.0%)

Dataset balance analysis:
  Most frequent emotion: happy (553 samples)
  Least frequent emotion: happy (553 samples)
  Balance ratio: 1.000 (1.0 = perfectly balanced)
  OK: Dataset is reasonably balanced (ratio >= 0.5)

3. MUSICAL FEATURE ANALYSIS
========================================
Number of numeric features: 21

Feature statistics:
       duration  pitch_min  pitch_max  ...  tempo_variation  dynamic_range  dynamic_contrast
count  2212.000   2212.000   2212.000  ...         2212.000       2212.000          2212.000  
mean     35.563     29.593     84.883  ...           -0.169         55.774            12.994  
std       8.832      7.657      6.282  ...            0.234         10.042             2.191  
min      18.009     22.000     63.000  ...           -0.836         28.000             4.896  
25%      29.990     23.000     81.000  ...           -0.353         49.175            11.872  
50%      34.566     26.431     84.000  ...            0.000         51.731            13.423  
75%      39.419     34.000     88.002  ...            0.000         61.000            14.562  
max     109.990     69.000    105.000  ...            0.523         99.000            24.903  

[8 rows x 21 columns]

Feature correlation with emotions:
  happy: ['total_notes(266.995)', 'chord_changes(145.190)', 'harmony_complexity(145.190)']    
  sad: ['total_notes(186.257)', 'tempo_mean(120.383)', 'velocity_max(90.247)']
  hopeful: ['total_notes(185.091)', 'tempo_mean(120.675)', 'velocity_max(105.107)']
  tense: ['total_notes(278.558)', 'chord_changes(161.304)', 'harmony_complexity(161.188)']    

4. FEATURE CORRELATION ANALYSIS
========================================

Highly correlated feature pairs (|r| > 0.7):
  chord_changes <-> harmony_complexity: 1.000
  tempo_mean <-> tempo_std: -1.000
  harmony_complexity <-> total_notes: 0.965
  chord_changes <-> total_notes: 0.965
  velocity_min <-> velocity_max: 0.836
  velocity_std <-> dynamic_range: 0.790
  pitch_std <-> velocity_max: 0.783
  velocity_max <-> velocity_mean: 0.771
  pitch_min <-> pitch_std: -0.765
  rhythm_mean <-> syncopation: -0.742
  velocity_min <-> velocity_mean: 0.738
  pitch_std <-> velocity_min: 0.729
  tempo_mean <-> syncopation: 0.718
  tempo_std <-> syncopation: -0.718
  tempo_mean <-> tempo_variation: -0.702
  tempo_std <-> tempo_variation: 0.700

5. STATISTICAL TESTS
========================================
ANOVA tests for feature differences across emotions:

Top 10 most significant features (ANOVA):
   1. pitch_mean: F=744.922, p=0.000000 ***
   2. velocity_min: F=1057.558, p=0.000000 ***
   3. velocity_max: F=850.352, p=0.000000 ***
   4. velocity_mean: F=1271.443, p=0.000000 ***
   5. tempo_mean: F=806.026, p=0.000000 ***
   6. tempo_std: F=805.111, p=0.000000 ***
   7. syncopation: F=766.487, p=0.000000 ***
   8. pitch_std: F=554.744, p=0.000000 ***
   9. pitch_min: F=350.288, p=0.000000 ***
  10. rhythm_mean: F=330.399, p=0.000000 ***

Detailed ANOVA results saved to: results/anova_results.csv

EDA completed! Results saved in results/

6. DATASET COMPARISON (Original vs Balanced)
==================================================

Emotion distribution comparison:
         Original  Balanced
emotion
happy         553       553
sad           338       553
hopeful       174       553
tense          13       553

Balance improvement:
  Original balance ratio: 0.024
  Balanced balance ratio: 1.000
  Improvement: 4153.8%

3.6. Model Training Visualizations...

7. MODEL TRAINING VISUALIZATION
==================================================

Training Analysis Summary:
  Total epochs: 50
  Final D loss: 0.9251
  Final G loss: 0.3582
  D loss range: 0.6831 - 0.9251
  G loss range: 0.3582 - 0.6352
  Training stability: D_std=0.0494, G_std=0.0574
  Training quality: STABLE
  Training visualization saved to: results/gan_training_analysis.png

8. MODEL ARCHITECTURE VISUALIZATION
==================================================
Model architecture visualization saved to: results/model_architecture.png

9. FEATURE IMPORTANCE ANALYSIS
==================================================
Feature importance analysis saved to: results/feature_importance.png
Top 5 most important features:
  1. velocity_mean: 6.898
  2. velocity_min: 5.737
  3. velocity_max: 4.613
  4. tempo_mean: 4.373
  5. tempo_std: 4.368

4. Training recommendation system...
Training recommendation system...
Trained on 2212 samples
Feature dimensions: 21
Recommendation system stats:
{'total_samples': 2212, 'emotion_distribution': {'happy': 553, 'sad': 553, 'hopeful': 553, 'tense': 553}, 'feature_dimensions': 21}

5. Testing recommendation system...
Query file: ../EMOPIA_1.0/midis\Q1__SJQaaRzD-A_1.mid
Query emotion category: hopeful

Top 5 similar tracks:
1. Q1_pNwQ9Tu_bCs_2.mid (similarity: 0.886, emotion: hopeful)
2. Q1_pNwQ9Tu_bCs_0.mid (similarity: 0.859, emotion: hopeful)
3. Q1_zmZHNy9T8Pg_2.mid (similarity: 0.847, emotion: hopeful)
4. Q2_XC_SiJszQx0_1.mid (similarity: 0.840, emotion: hopeful)
5. Q1_8rupdevqfuI_0.mid (similarity: 0.832, emotion: hopeful)

Test Accuracy: 100.0% (5/5 correct predictions)
Query emotion category: hopeful

System completed successfully!
Results saved in results/ directory
--------------------------------------------------------------

Track-19-Q2_b8HVQtIoBYU_1
1. Survey Results
Happy:10%
Sad:13.3%
Hopeful:13.3%
Fearful:16.7%
Tense:23.3%
Excited:23.3%

2. System Results:
Music Emotion Analysis and Recommendation System
==================================================

1. Extracting features from MIDI files...
Found 1078 MIDI files
Processing file 1/1078
Processing file 101/1078
Processing file 201/1078
Processing file 301/1078
Processing file 401/1078
Processing file 501/1078
Processing file 601/1078
Processing file 701/1078
Processing file 801/1078
Processing file 901/1078
Processing file 1001/1078
Extracted features from 1078 files

2. Classifying emotions...

2.5. Exploratory Data Analysis (Classified Dataset)...

============================================================
EXPLORATORY DATA ANALYSIS: Classified EMOPIA Dataset
============================================================

1. BASIC DATASET INFORMATION
========================================
Total samples: 1078
Total features: 23
Memory usage: 0.42 MB
Missing values: 0
Duplicate rows: 0

Data types:
float64    13
int64       8
object      2
Name: count, dtype: int64

2. EMOTION DISTRIBUTION ANALYSIS
========================================
Emotion distribution:
  happy: 553 samples (51.3%)
  sad: 338 samples (31.4%)
  hopeful: 174 samples (16.1%)
  tense: 13 samples (1.2%)

Dataset balance analysis:
  Most frequent emotion: happy (553 samples)
  Least frequent emotion: tense (13 samples)
  Balance ratio: 0.024 (1.0 = perfectly balanced)
  WARNING: Dataset is highly imbalanced (ratio < 0.1)

3. MUSICAL FEATURE ANALYSIS
========================================
Number of numeric features: 21

Feature statistics:
       duration  pitch_min  pitch_max  ...  tempo_variation  dynamic_range  dynamic_contrast
count  1078.000   1078.000   1078.000  ...           1078.0       1078.000          1078.000  
mean     39.924     34.680     85.633  ...              0.0         61.499            11.939  
std      10.475      8.154      8.248  ...              0.0         11.879             2.714  
min      18.009     22.000     63.000  ...              0.0         28.000             4.896  
25%      29.990     28.000     80.000  ...              0.0         53.000            10.017  
50%      39.584     34.500     85.000  ...              0.0         62.000            11.762  
75%      46.725     40.000     91.000  ...              0.0         69.000            13.607  
max     109.990     69.000    105.000  ...              0.0         99.000            24.903  

[8 rows x 21 columns]

Feature correlation with emotions:
  happy: ['total_notes(266.995)', 'chord_changes(145.190)', 'harmony_complexity(145.190)']
  sad: ['total_notes(235.935)', 'tempo_mean(120.000)', 'chord_changes(115.003)']
  hopeful: ['total_notes(305.592)', 'chord_changes(170.138)', 'harmony_complexity(170.138)']  
  tense: ['total_notes(390.077)', 'chord_changes(247.077)', 'harmony_complexity(247.077)']    

4. FEATURE CORRELATION ANALYSIS
========================================

Highly correlated feature pairs (|r| > 0.7):
  velocity_std <-> dynamic_contrast: 1.000
  chord_changes <-> harmony_complexity: 1.000
  chord_changes <-> total_notes: 0.952
  harmony_complexity <-> total_notes: 0.952
  velocity_max <-> velocity_mean: 0.892
  rhythm_mean <-> rhythm_std: 0.821
  velocity_min <-> velocity_mean: 0.781

5. STATISTICAL TESTS
========================================
ANOVA tests for feature differences across emotions:
C:\Users\perer\AppData\Local\Programs\Python\Python311\Lib\site-packages\scipy\stats\_stats_py.py:4167: ConstantInputWarning: Each of the input arrays is constant;the F statistic is not defined or infinite
  warnings.warn(stats.ConstantInputWarning(msg))

Top 10 most significant features (ANOVA):
   1. pitch_mean: F=133.385, p=0.000000 ***
   2. pitch_max: F=100.888, p=0.000000 ***
   3. velocity_mean: F=73.824, p=0.000000 ***
   4. velocity_max: F=61.414, p=0.000000 ***
   5. note_overlap_ratio: F=42.039, p=0.000000 ***
   6. velocity_min: F=36.637, p=0.000000 ***
   7. pitch_std: F=28.778, p=0.000000 ***
   8. syncopation: F=24.819, p=0.000000 ***
   9. chord_changes: F=14.426, p=0.000000 ***
  10. harmony_complexity: F=14.426, p=0.000000 ***

Detailed ANOVA results saved to: results/anova_results.csv

EDA completed! Results saved in results/
Emotion distribution:
emotion
happy      553
sad        338
hopeful    174
tense       13
Name: count, dtype: int64

3. Balancing dataset with GAN...
Epoch 0, D Loss: 0.7279, G Loss: 0.6771
Epoch 20, D Loss: 0.9625, G Loss: 0.3589
Epoch 40, D Loss: 0.9710, G Loss: 0.3527
Generating 215 samples for sad
Generating 379 samples for hopeful
Generating 540 samples for tense
Balanced dataset emotion distribution:
emotion
happy      553
sad        553
hopeful    553
tense      553
Name: count, dtype: int64

3.5. Exploratory Data Analysis (Balanced Dataset)...

============================================================
EXPLORATORY DATA ANALYSIS: Balanced Dataset (After GAN)
============================================================

1. BASIC DATASET INFORMATION
========================================
Total samples: 2212
Total features: 23
Memory usage: 0.70 MB
Missing values: 1134
Duplicate rows: 0

Data types:
float64    21
object      2
Name: count, dtype: int64

2. EMOTION DISTRIBUTION ANALYSIS
========================================
Emotion distribution:
  happy: 553 samples (25.0%)
  sad: 553 samples (25.0%)
  hopeful: 553 samples (25.0%)
  tense: 553 samples (25.0%)

Dataset balance analysis:
  Most frequent emotion: happy (553 samples)
  Least frequent emotion: happy (553 samples)
  Balance ratio: 1.000 (1.0 = perfectly balanced)
  OK: Dataset is reasonably balanced (ratio >= 0.5)

3. MUSICAL FEATURE ANALYSIS
========================================
Number of numeric features: 21

Feature statistics:
       duration  pitch_min  pitch_max  ...  tempo_variation  dynamic_range  dynamic_contrast
count  2212.000   2212.000   2212.000  ...         2212.000       2212.000          2212.000  
mean     44.834     29.599     80.551  ...            0.122         70.028            12.990  
std       8.935      7.654      7.865  ...            0.195         12.529             2.189  
min      18.009     22.000     63.000  ...           -0.406         28.000             4.896  
25%      39.752     23.000     74.067  ...            0.000         62.000            11.872  
50%      46.795     26.486     79.654  ...            0.000         72.000            13.420  
75%      51.800     34.000     84.000  ...            0.256         83.774            14.541  
max     109.990     69.000    105.000  ...            0.814         99.000            24.903  

[8 rows x 21 columns]

Feature correlation with emotions:
  happy: ['total_notes(266.995)', 'chord_changes(145.190)', 'harmony_complexity(145.190)']    
  sad: ['total_notes(285.737)', 'harmony_complexity(148.160)', 'tempo_mean(119.857)']
  hopeful: ['total_notes(426.485)', 'harmony_complexity(260.602)', 'tempo_mean(119.743)']     
  tense: ['total_notes(501.961)', 'harmony_complexity(331.383)', 'chord_changes(164.830)']    

4. FEATURE CORRELATION ANALYSIS
========================================

Highly correlated feature pairs (|r| > 0.7):
  velocity_std <-> dynamic_contrast: 1.000
  harmony_complexity <-> total_notes: 0.976
  rhythm_mean <-> rhythm_std: 0.897
  tempo_mean <-> tempo_std: -0.829
  pitch_std <-> velocity_mean: 0.797
  rhythm_mean <-> total_notes: -0.786
  velocity_max <-> velocity_mean: 0.781
  pitch_max <-> pitch_mean: 0.766
  syncopation <-> total_notes: 0.765
  pitch_min <-> pitch_std: -0.764
  harmony_complexity <-> syncopation: 0.760
  harmony_complexity <-> rhythm_mean: -0.756
  rhythm_mean <-> syncopation: -0.742
  velocity_mean <-> harmony_complexity: 0.740
  rhythm_std <-> total_notes: -0.731
  pitch_std <-> harmony_complexity: 0.728
  tempo_std <-> syncopation: 0.720
  velocity_mean <-> total_notes: 0.717
  syncopation <-> note_overlap_ratio: -0.716
  harmony_complexity <-> rhythm_std: -0.716
  pitch_min <-> pitch_mean: 0.714

5. STATISTICAL TESTS
========================================
ANOVA tests for feature differences across emotions:

Top 10 most significant features (ANOVA):
   1. pitch_mean: F=970.208, p=0.000000 ***
   2. velocity_max: F=965.628, p=0.000000 ***
   3. velocity_mean: F=1168.797, p=0.000000 ***
   4. tempo_std: F=808.947, p=0.000000 ***
   5. syncopation: F=770.630, p=0.000000 ***
   6. pitch_max: F=671.555, p=0.000000 ***
   7. dynamic_range: F=638.644, p=0.000000 ***
   8. harmony_complexity: F=617.920, p=0.000000 ***
   9. pitch_std: F=553.653, p=0.000000 ***
  10. total_notes: F=546.846, p=0.000000 ***

Detailed ANOVA results saved to: results/anova_results.csv

EDA completed! Results saved in results/

6. DATASET COMPARISON (Original vs Balanced)
==================================================

Emotion distribution comparison:
         Original  Balanced
emotion
happy         553       553
sad           338       553
hopeful       174       553
tense          13       553

Balance improvement:
  Original balance ratio: 0.024
  Balanced balance ratio: 1.000
  Improvement: 4153.8%

3.6. Model Training Visualizations...

7. MODEL TRAINING VISUALIZATION
==================================================

Training Analysis Summary:
  Total epochs: 50
  Final D loss: 0.9567
  Final G loss: 0.3518
  D loss range: 0.7279 - 0.9831
  G loss range: 0.3518 - 0.6771
  Training stability: D_std=0.0604, G_std=0.0721
  Training quality: STABLE
  Training visualization saved to: results/gan_training_analysis.png

8. MODEL ARCHITECTURE VISUALIZATION
==================================================
Model architecture visualization saved to: results/model_architecture.png

9. FEATURE IMPORTANCE ANALYSIS
==================================================
Feature importance analysis saved to: results/feature_importance.png
Top 5 most important features:
  1. velocity_mean: 6.341
  2. pitch_mean: 5.263
  3. velocity_max: 5.238
  4. tempo_std: 4.388
  5. syncopation: 4.181

4. Training recommendation system...
Training recommendation system...
Trained on 2212 samples
Feature dimensions: 21
Recommendation system stats:
{'total_samples': 2212, 'emotion_distribution': {'happy': 553, 'sad': 553, 'hopeful': 553, 'tense': 553}, 'feature_dimensions': 21}

5. Testing recommendation system...
Query file: ../EMOPIA_1.0/midis\Q2_b8HVQtIoBYU_1.mid
Query emotion category: hopeful

Top 5 similar tracks:
1. Q2_P-We4q3goKU_5.mid (similarity: 0.785, emotion: hopeful)
2. Q2_x5ibvz38jOs_1.mid (similarity: 0.782, emotion: hopeful)
3. Q2_cQp1BYDGcRo_0.mid (similarity: 0.674, emotion: hopeful)
4. Q1_dfNdpy8TUzA_1.mid (similarity: 0.649, emotion: hopeful)
5. Q1_8rupdevqfuI_0.mid (similarity: 0.629, emotion: hopeful)

Test Accuracy: 100.0% (5/5 correct predictions)
Query emotion category: hopeful

System completed successfully!
Results saved in results/ directory
--------------------------------------------------------------

Track-20-Q4_9v2WSpn4FCw_9
1. Survey Results
Happy:10%
Sad:40%
Hopeful:26.7%
Fearful:13.3%
Tense:6.7%
Excited:3.3%

2. System Results:
Music Emotion Analysis and Recommendation System
==================================================

1. Extracting features from MIDI files...
Found 1078 MIDI files
Processing file 1/1078
Processing file 101/1078
Processing file 201/1078
Processing file 301/1078
Processing file 401/1078
Processing file 501/1078
Processing file 601/1078
Processing file 701/1078
Processing file 801/1078
Processing file 901/1078
Processing file 1001/1078
Extracted features from 1078 files

2. Classifying emotions...

2.5. Exploratory Data Analysis (Classified Dataset)...

============================================================
EXPLORATORY DATA ANALYSIS: Classified EMOPIA Dataset
============================================================

1. BASIC DATASET INFORMATION
========================================
Total samples: 1078
Total features: 23
Memory usage: 0.42 MB
Missing values: 0
Duplicate rows: 0

Data types:
float64    13
int64       8
object      2
Name: count, dtype: int64

2. EMOTION DISTRIBUTION ANALYSIS
========================================
Emotion distribution:
  happy: 553 samples (51.3%)
  sad: 338 samples (31.4%)
  hopeful: 174 samples (16.1%)
  tense: 13 samples (1.2%)

Dataset balance analysis:
  Most frequent emotion: happy (553 samples)
  Least frequent emotion: tense (13 samples)
  Balance ratio: 0.024 (1.0 = perfectly balanced)
  WARNING: Dataset is highly imbalanced (ratio < 0.1)

3. MUSICAL FEATURE ANALYSIS
========================================
Number of numeric features: 21

Feature statistics:
       duration  pitch_min  pitch_max  ...  tempo_variation  dynamic_range  dynamic_contrast
count  1078.000   1078.000   1078.000  ...           1078.0       1078.000          1078.000  
mean     39.924     34.680     85.633  ...              0.0         61.499            11.939  
std      10.475      8.154      8.248  ...              0.0         11.879             2.714  
min      18.009     22.000     63.000  ...              0.0         28.000             4.896  
25%      29.990     28.000     80.000  ...              0.0         53.000            10.017  
50%      39.584     34.500     85.000  ...              0.0         62.000            11.762  
75%      46.725     40.000     91.000  ...              0.0         69.000            13.607  
max     109.990     69.000    105.000  ...              0.0         99.000            24.903  

[8 rows x 21 columns]

Feature correlation with emotions:
  happy: ['total_notes(266.995)', 'chord_changes(145.190)', 'harmony_complexity(145.190)']
  sad: ['total_notes(235.935)', 'tempo_mean(120.000)', 'chord_changes(115.003)']
  hopeful: ['total_notes(305.592)', 'chord_changes(170.138)', 'harmony_complexity(170.138)']  
  tense: ['total_notes(390.077)', 'chord_changes(247.077)', 'harmony_complexity(247.077)']    

4. FEATURE CORRELATION ANALYSIS
========================================

Highly correlated feature pairs (|r| > 0.7):
  velocity_std <-> dynamic_contrast: 1.000
  chord_changes <-> harmony_complexity: 1.000
  chord_changes <-> total_notes: 0.952
  harmony_complexity <-> total_notes: 0.952
  velocity_max <-> velocity_mean: 0.892
  rhythm_mean <-> rhythm_std: 0.821
  velocity_min <-> velocity_mean: 0.781

5. STATISTICAL TESTS
========================================
ANOVA tests for feature differences across emotions:
C:\Users\perer\AppData\Local\Programs\Python\Python311\Lib\site-packages\scipy\stats\_stats_py.py:4167: ConstantInputWarning: Each of the input arrays is constant;the F statistic is not defined or infinite
  warnings.warn(stats.ConstantInputWarning(msg))

Top 10 most significant features (ANOVA):
   1. pitch_mean: F=133.385, p=0.000000 ***
   2. pitch_max: F=100.888, p=0.000000 ***
   3. velocity_mean: F=73.824, p=0.000000 ***
   4. velocity_max: F=61.414, p=0.000000 ***
   5. note_overlap_ratio: F=42.039, p=0.000000 ***
   6. velocity_min: F=36.637, p=0.000000 ***
   7. pitch_std: F=28.778, p=0.000000 ***
   8. syncopation: F=24.819, p=0.000000 ***
   9. chord_changes: F=14.426, p=0.000000 ***
  10. harmony_complexity: F=14.426, p=0.000000 ***

Detailed ANOVA results saved to: results/anova_results.csv

EDA completed! Results saved in results/
Emotion distribution:
emotion
happy      553
sad        338
hopeful    174
tense       13
Name: count, dtype: int64

3. Balancing dataset with GAN...
Epoch 0, D Loss: 0.6850, G Loss: 0.6201
Epoch 20, D Loss: 0.8923, G Loss: 0.3559
Epoch 40, D Loss: 0.8978, G Loss: 0.3518
Generating 215 samples for sad
Generating 379 samples for hopeful
Generating 540 samples for tense
Balanced dataset emotion distribution:
emotion
happy      553
sad        553
hopeful    553
tense      553
Name: count, dtype: int64

3.5. Exploratory Data Analysis (Balanced Dataset)...

============================================================
EXPLORATORY DATA ANALYSIS: Balanced Dataset (After GAN)
============================================================

1. BASIC DATASET INFORMATION
========================================
Total samples: 2212
Total features: 23
Memory usage: 0.70 MB
Missing values: 1134
Duplicate rows: 0

Data types:
float64    21
object      2
Name: count, dtype: int64

2. EMOTION DISTRIBUTION ANALYSIS
========================================
Emotion distribution:
  happy: 553 samples (25.0%)
  sad: 553 samples (25.0%)
  hopeful: 553 samples (25.0%)
  tense: 553 samples (25.0%)

Dataset balance analysis:
  Most frequent emotion: happy (553 samples)
  Least frequent emotion: happy (553 samples)
  Balance ratio: 1.000 (1.0 = perfectly balanced)
  OK: Dataset is reasonably balanced (ratio >= 0.5)

3. MUSICAL FEATURE ANALYSIS
========================================
Number of numeric features: 21

Feature statistics:
       duration  pitch_min  pitch_max  ...  tempo_variation  dynamic_range  dynamic_contrast
count  2212.000   2212.000   2212.000  ...         2212.000       2212.000          2212.000  
mean     44.822     35.508     80.596  ...            0.512         70.212            12.994  
std       8.928      7.877      7.837  ...            0.499         12.658             2.191  
min      18.009     22.000     63.000  ...            0.000         28.000             4.896  
25%      39.752     28.470     74.137  ...            0.000         62.000            11.872  
50%      46.784     35.500     79.687  ...            0.992         72.000            13.423  
75%      51.771     42.000     84.000  ...            0.999         84.717            14.559  
max     109.990     69.000    105.000  ...            1.000         99.000            24.903  

[8 rows x 21 columns]

Feature correlation with emotions:
  happy: ['total_notes(266.995)', 'chord_changes(145.190)', 'harmony_complexity(145.190)']    
  sad: ['total_notes(186.139)', 'tempo_mean(119.619)', 'harmony_complexity(87.332)']
  hopeful: ['total_notes(184.777)', 'tempo_mean(119.329)', 'harmony_complexity(94.064)']      
  tense: ['total_notes(278.224)', 'harmony_complexity(176.077)', 'chord_changes(161.272)']    

4. FEATURE CORRELATION ANALYSIS
========================================

Highly correlated feature pairs (|r| > 0.7):
  tempo_mean <-> tempo_variation: -1.000
  chord_changes <-> harmony_complexity: 0.992
  tempo_mean <-> tempo_std: 0.983
  tempo_std <-> tempo_variation: -0.982
  chord_changes <-> total_notes: 0.965
  harmony_complexity <-> total_notes: 0.954
  pitch_std <-> velocity_mean: 0.798
  velocity_max <-> velocity_mean: 0.780
  rhythm_mean <-> total_notes: -0.714

5. STATISTICAL TESTS
========================================
ANOVA tests for feature differences across emotions:

Top 10 most significant features (ANOVA):
   1. pitch_mean: F=872.696, p=0.000000 ***
   2. velocity_max: F=964.863, p=0.000000 ***
   3. velocity_mean: F=1168.244, p=0.000000 ***
   4. tempo_mean: F=807.008, p=0.000000 ***
   5. tempo_std: F=762.266, p=0.000000 ***
   6. tempo_variation: F=808.480, p=0.000000 ***
   7. pitch_max: F=667.498, p=0.000000 ***
   8. dynamic_range: F=653.199, p=0.000000 ***
   9. note_overlap_ratio: F=613.696, p=0.000000 ***
  10. pitch_std: F=555.993, p=0.000000 ***

Detailed ANOVA results saved to: results/anova_results.csv

EDA completed! Results saved in results/

6. DATASET COMPARISON (Original vs Balanced)
==================================================

Emotion distribution comparison:
         Original  Balanced
emotion
happy         553       553
sad           338       553
hopeful       174       553
tense          13       553

Balance improvement:
  Original balance ratio: 0.024
  Balanced balance ratio: 1.000
  Improvement: 4153.8%

3.6. Model Training Visualizations...

7. MODEL TRAINING VISUALIZATION
==================================================

Training Analysis Summary:
  Total epochs: 50
  Final D loss: 0.9031
  Final G loss: 0.3509
  D loss range: 0.6850 - 0.9151
  G loss range: 0.3509 - 0.6201
  Training stability: D_std=0.0504, G_std=0.0579
  Training quality: STABLE
  Training visualization saved to: results/gan_training_analysis.png

8. MODEL ARCHITECTURE VISUALIZATION
==================================================
Model architecture visualization saved to: results/model_architecture.png

9. FEATURE IMPORTANCE ANALYSIS
==================================================
Feature importance analysis saved to: results/feature_importance.png
Top 5 most important features:
  1. velocity_mean: 6.338
  2. velocity_max: 5.234
  3. pitch_mean: 4.734
  4. tempo_variation: 4.386
  5. tempo_mean: 4.378

4. Training recommendation system...
Training recommendation system...
Trained on 2212 samples
Feature dimensions: 21
Recommendation system stats:
{'total_samples': 2212, 'emotion_distribution': {'happy': 553, 'sad': 553, 'hopeful': 553, 'tense': 553}, 'feature_dimensions': 21}

5. Testing recommendation system...
Query file: ../EMOPIA_1.0/midis\Q4_9v2WSpn4FCw_9.mid
Query emotion category: happy

Top 5 similar tracks:
1. Q1_T92R7xjce34_1.mid (similarity: 0.924, emotion: happy)
2. Q3_ltxNPJda7zE_2.mid (similarity: 0.846, emotion: happy)
3. Q1_9v2WSpn4FCw_5.mid (similarity: 0.840, emotion: happy)
4. Q1_9v2WSpn4FCw_0.mid (similarity: 0.789, emotion: happy)
5. Q2_epX33OVpkmA_2.mid (similarity: 0.739, emotion: happy)

Test Accuracy: 100.0% (5/5 correct predictions)
Query emotion category: happy

System completed successfully!
Results saved in results/ directory
--------------------------------------------------------------

Track-21-Q2_ANZf1QXsNrY_7
1. Survey Results
Happy:6.7%
Sad:16.7%
Hopeful:23.3%
Fearful:3.3%
Tense:23.3%
Excited:26.7%

2. System Results:
Music Emotion Analysis and Recommendation System
==================================================

1. Extracting features from MIDI files...
Found 1078 MIDI files
Processing file 1/1078
Processing file 101/1078
Processing file 201/1078
Processing file 301/1078
Processing file 401/1078
Processing file 501/1078
Processing file 601/1078
Processing file 701/1078
Processing file 801/1078
Processing file 901/1078
Processing file 1001/1078
Extracted features from 1078 files

2. Classifying emotions...

2.5. Exploratory Data Analysis (Classified Dataset)...

============================================================
EXPLORATORY DATA ANALYSIS: Classified EMOPIA Dataset
============================================================

1. BASIC DATASET INFORMATION
========================================
Total samples: 1078
Total features: 23
Memory usage: 0.42 MB
Missing values: 0
Duplicate rows: 0

Data types:
float64    13
int64       8
object      2
Name: count, dtype: int64

2. EMOTION DISTRIBUTION ANALYSIS
========================================
Emotion distribution:
  happy: 553 samples (51.3%)
  sad: 338 samples (31.4%)
  hopeful: 174 samples (16.1%)
  tense: 13 samples (1.2%)

Dataset balance analysis:
  Most frequent emotion: happy (553 samples)
  Least frequent emotion: tense (13 samples)
  Balance ratio: 0.024 (1.0 = perfectly balanced)
  WARNING: Dataset is highly imbalanced (ratio < 0.1)

3. MUSICAL FEATURE ANALYSIS
========================================
Number of numeric features: 21

Feature statistics:
       duration  pitch_min  pitch_max  ...  tempo_variation  dynamic_range  dynamic_contrast
count  1078.000   1078.000   1078.000  ...           1078.0       1078.000          1078.000  
mean     39.924     34.680     85.633  ...              0.0         61.499            11.939  
std      10.475      8.154      8.248  ...              0.0         11.879             2.714  
min      18.009     22.000     63.000  ...              0.0         28.000             4.896  
25%      29.990     28.000     80.000  ...              0.0         53.000            10.017  
50%      39.584     34.500     85.000  ...              0.0         62.000            11.762  
75%      46.725     40.000     91.000  ...              0.0         69.000            13.607  
max     109.990     69.000    105.000  ...              0.0         99.000            24.903  

[8 rows x 21 columns]

Feature correlation with emotions:
  happy: ['total_notes(266.995)', 'chord_changes(145.190)', 'harmony_complexity(145.190)']    
  sad: ['total_notes(235.935)', 'tempo_mean(120.000)', 'chord_changes(115.003)']
  hopeful: ['total_notes(305.592)', 'chord_changes(170.138)', 'harmony_complexity(170.138)']  
  tense: ['total_notes(390.077)', 'chord_changes(247.077)', 'harmony_complexity(247.077)']    

4. FEATURE CORRELATION ANALYSIS
========================================

Highly correlated feature pairs (|r| > 0.7):
  velocity_std <-> dynamic_contrast: 1.000
  chord_changes <-> harmony_complexity: 1.000
  chord_changes <-> total_notes: 0.952
  harmony_complexity <-> total_notes: 0.952
  velocity_max <-> velocity_mean: 0.892
  rhythm_mean <-> rhythm_std: 0.821
  velocity_min <-> velocity_mean: 0.781

5. STATISTICAL TESTS
========================================
ANOVA tests for feature differences across emotions:
C:\Users\perer\AppData\Local\Programs\Python\Python311\Lib\site-packages\scipy\stats\_stats_py.py:4167: ConstantInputWarning: Each of the input arrays is constant;the F statistic is not defined or infinite
  warnings.warn(stats.ConstantInputWarning(msg))

Top 10 most significant features (ANOVA):
   1. pitch_mean: F=133.385, p=0.000000 ***
   2. pitch_max: F=100.888, p=0.000000 ***
   3. velocity_mean: F=73.824, p=0.000000 ***
   4. velocity_max: F=61.414, p=0.000000 ***
   5. note_overlap_ratio: F=42.039, p=0.000000 ***
   6. velocity_min: F=36.637, p=0.000000 ***
   7. pitch_std: F=28.778, p=0.000000 ***
   8. syncopation: F=24.819, p=0.000000 ***
   9. chord_changes: F=14.426, p=0.000000 ***
  10. harmony_complexity: F=14.426, p=0.000000 ***

Detailed ANOVA results saved to: results/anova_results.csv

EDA completed! Results saved in results/
Emotion distribution:
emotion
happy      553
sad        338
hopeful    174
tense       13
Name: count, dtype: int64

3. Balancing dataset with GAN...
Epoch 0, D Loss: 0.7458, G Loss: 0.7298
Epoch 20, D Loss: 0.8972, G Loss: 0.4798
Epoch 40, D Loss: 0.8947, G Loss: 0.4719
Generating 215 samples for sad
Generating 379 samples for hopeful
Generating 540 samples for tense
Balanced dataset emotion distribution:
emotion
happy      553
sad        553
hopeful    553
tense      553
Name: count, dtype: int64

3.5. Exploratory Data Analysis (Balanced Dataset)...

============================================================
EXPLORATORY DATA ANALYSIS: Balanced Dataset (After GAN)
============================================================

1. BASIC DATASET INFORMATION
========================================
Total samples: 2212
Total features: 23
Memory usage: 0.70 MB
Missing values: 1134
Duplicate rows: 0

Data types:
float64    21
object      2
Name: count, dtype: int64

2. EMOTION DISTRIBUTION ANALYSIS
========================================
Emotion distribution:
  happy: 553 samples (25.0%)
  sad: 553 samples (25.0%)
  hopeful: 553 samples (25.0%)
  tense: 553 samples (25.0%)

Dataset balance analysis:
  Most frequent emotion: happy (553 samples)
  Least frequent emotion: happy (553 samples)
  Balance ratio: 1.000 (1.0 = perfectly balanced)
  OK: Dataset is reasonably balanced (ratio >= 0.5)

3. MUSICAL FEATURE ANALYSIS
========================================
Number of numeric features: 21

Feature statistics:
       duration  pitch_min  pitch_max  ...  tempo_variation  dynamic_range  dynamic_contrast
count  2212.000   2212.000   2212.000  ...         2212.000       2212.000          2212.000  
mean     35.518     29.598     88.464  ...            0.464         68.032            10.425  
std       8.858      7.655      6.676  ...            0.455         11.249             2.416  
min      18.009     22.000     63.000  ...            0.000         28.000             4.896  
25%      29.990     23.000     85.000  ...            0.000         62.000             9.228  
50%      34.534     26.448     90.219  ...            0.752         69.000             9.271  
75%      39.419     34.000     94.542  ...            0.919         77.460            11.698  
max     109.990     69.000    105.000  ...            0.994         99.000            24.903  

[8 rows x 21 columns]

Feature correlation with emotions:
  happy: ['total_notes(266.995)', 'chord_changes(145.190)', 'harmony_complexity(145.190)']
  sad: ['total_notes(195.584)', 'chord_changes(148.663)', 'tempo_mean(119.612)']
  hopeful: ['chord_changes(261.948)', 'total_notes(206.428)', 'tempo_mean(119.316)']
  tense: ['chord_changes(332.621)', 'total_notes(298.716)', 'harmony_complexity(161.383)']    

4. FEATURE CORRELATION ANALYSIS
========================================

Highly correlated feature pairs (|r| > 0.7):
  tempo_mean <-> tempo_std: 1.000
  tempo_std <-> tempo_variation: -0.996
  tempo_mean <-> tempo_variation: -0.996
  harmony_complexity <-> total_notes: 0.956
  rhythm_mean <-> rhythm_std: 0.840
  velocity_max <-> velocity_mean: 0.817
  chord_changes <-> syncopation: 0.761
  velocity_max <-> chord_changes: 0.737
  tempo_std <-> syncopation: -0.719
  tempo_mean <-> syncopation: -0.719
  syncopation <-> tempo_variation: 0.716
  pitch_min <-> pitch_mean: 0.714
  rhythm_mean <-> total_notes: -0.707
  pitch_std <-> velocity_max: 0.703

5. STATISTICAL TESTS
========================================
ANOVA tests for feature differences across emotions:

Top 10 most significant features (ANOVA):
   1. pitch_mean: F=969.950, p=0.000000 ***
   2. velocity_max: F=852.221, p=0.000000 ***
   3. velocity_mean: F=1318.669, p=0.000000 ***
   4. tempo_mean: F=808.209, p=0.000000 ***
   5. tempo_std: F=808.682, p=0.000000 ***
   6. syncopation: F=768.034, p=0.000000 ***
   7. tempo_variation: F=792.968, p=0.000000 ***
   8. chord_changes: F=622.103, p=0.000000 ***
   9. dynamic_range: F=456.350, p=0.000000 ***
  10. pitch_std: F=363.617, p=0.000000 ***

Detailed ANOVA results saved to: results/anova_results.csv

EDA completed! Results saved in results/

6. DATASET COMPARISON (Original vs Balanced)
==================================================

Emotion distribution comparison:
         Original  Balanced
emotion
happy         553       553
sad           338       553
hopeful       174       553
tense          13       553

Balance improvement:
  Original balance ratio: 0.024
  Balanced balance ratio: 1.000
  Improvement: 4153.8%

3.6. Model Training Visualizations...

7. MODEL TRAINING VISUALIZATION
==================================================

Training Analysis Summary:
  Total epochs: 50
  Final D loss: 0.9079
  Final G loss: 0.4709
  D loss range: 0.7458 - 0.9079
  G loss range: 0.4709 - 0.7298
  Training stability: D_std=0.0374, G_std=0.0567
  Training quality: STABLE
  Training visualization saved to: results/gan_training_analysis.png

8. MODEL ARCHITECTURE VISUALIZATION
==================================================
Model architecture visualization saved to: results/model_architecture.png

9. FEATURE IMPORTANCE ANALYSIS
==================================================
Feature importance analysis saved to: results/feature_importance.png
Top 5 most important features:
  1. velocity_mean: 7.154
  2. pitch_mean: 5.262
  3. velocity_max: 4.623
  4. tempo_std: 4.387
  5. tempo_mean: 4.384

4. Training recommendation system...
Training recommendation system...
Trained on 2212 samples
Feature dimensions: 21
Recommendation system stats:
{'total_samples': 2212, 'emotion_distribution': {'happy': 553, 'sad': 553, 'hopeful': 553, 'tense': 553}, 'feature_dimensions': 21}

5. Testing recommendation system...
Query file: ../EMOPIA_1.0/midis\Q2_ANZf1QXsNrY_7.mid
Query emotion category: happy

Top 5 similar tracks:
1. Q1_NGE9ynTJABg_0.mid (similarity: 0.852, emotion: happy)
2. Q2_cm6E860vDjY_1.mid (similarity: 0.844, emotion: happy)
3. Q1_7yW9c7t8Hq0_1.mid (similarity: 0.834, emotion: happy)
4. Q1_jziI_cuN_H8_2.mid (similarity: 0.831, emotion: happy)
5. Q2_qlbazHayULg_6.mid (similarity: 0.813, emotion: happy)

Test Accuracy: 100.0% (5/5 correct predictions)
Query emotion category: happy

System completed successfully!
Results saved in results/ directory
--------------------------------------------------------------

Track-22-Q1_2Z9SjI131jA_0
1. Survey Results
Happy:40%
Sad:6.7%
Hopeful:20%
Fearful:0%
Tense:6.7%
Excited:26.7%

2. System Results:
Music Emotion Analysis and Recommendation System
==================================================

1. Extracting features from MIDI files...
Found 1078 MIDI files
Processing file 1/1078
Processing file 101/1078
Processing file 201/1078
Processing file 301/1078
Processing file 401/1078
Processing file 501/1078
Processing file 601/1078
Processing file 701/1078
Processing file 801/1078
Processing file 901/1078
Processing file 1001/1078
Extracted features from 1078 files

2. Classifying emotions...

2.5. Exploratory Data Analysis (Classified Dataset)...

============================================================
EXPLORATORY DATA ANALYSIS: Classified EMOPIA Dataset
============================================================

1. BASIC DATASET INFORMATION
========================================
Total samples: 1078
Total features: 23
Memory usage: 0.42 MB
Missing values: 0
Duplicate rows: 0

Data types:
float64    13
int64       8
object      2
Name: count, dtype: int64

2. EMOTION DISTRIBUTION ANALYSIS
========================================
Emotion distribution:
  happy: 553 samples (51.3%)
  sad: 338 samples (31.4%)
  hopeful: 174 samples (16.1%)
  tense: 13 samples (1.2%)

Dataset balance analysis:
  Most frequent emotion: happy (553 samples)
  Least frequent emotion: tense (13 samples)
  Balance ratio: 0.024 (1.0 = perfectly balanced)
  WARNING: Dataset is highly imbalanced (ratio < 0.1)

Feature statistics:
       duration  pitch_min  ...  dynamic_range  dynamic_contrast
count  1078.000   1078.000  ...       1078.000          1078.000
mean     39.924     34.680  ...         61.499            11.939
std      10.475      8.154  ...         11.879             2.714
min      18.009     22.000  ...         28.000             4.896
25%      29.990     28.000  ...         53.000            10.017
50%      39.584     34.500  ...         62.000            11.762
75%      46.725     40.000  ...         69.000            13.607
max     109.990     69.000  ...         99.000            24.903

[8 rows x 21 columns]

Feature correlation with emotions:
  happy: ['total_notes(266.995)', 'chord_changes(145.190)', 'harmony_complexity(145.190)']    
  sad: ['total_notes(235.935)', 'tempo_mean(120.000)', 'chord_changes(115.003)']
  hopeful: ['total_notes(305.592)', 'chord_changes(170.138)', 'harmony_complexity(170.138)']  
  tense: ['total_notes(390.077)', 'chord_changes(247.077)', 'harmony_complexity(247.077)']    

4. FEATURE CORRELATION ANALYSIS
========================================

Highly correlated feature pairs (|r| > 0.7):
  velocity_std <-> dynamic_contrast: 1.000
  chord_changes <-> harmony_complexity: 1.000
  chord_changes <-> total_notes: 0.952
  harmony_complexity <-> total_notes: 0.952
  velocity_max <-> velocity_mean: 0.892
  rhythm_mean <-> rhythm_std: 0.821
  velocity_min <-> velocity_mean: 0.781

5. STATISTICAL TESTS
========================================
ANOVA tests for feature differences across emotions:
C:\Users\perer\AppData\Local\Programs\Python\Python311\Lib\site-packages\scipy\stats\_stats_py.py:4167: ConstantInputWarning: Each of the input arrays is constant;the F statistic is not defined or infinite
  warnings.warn(stats.ConstantInputWarning(msg))

Top 10 most significant features (ANOVA):
   1. pitch_mean: F=133.385, p=0.000000 ***
   2. pitch_max: F=100.888, p=0.000000 ***
   3. velocity_mean: F=73.824, p=0.000000 ***
   4. velocity_max: F=61.414, p=0.000000 ***
   5. note_overlap_ratio: F=42.039, p=0.000000 ***
   6. velocity_min: F=36.637, p=0.000000 ***
   7. pitch_std: F=28.778, p=0.000000 ***
   8. syncopation: F=24.819, p=0.000000 ***
   9. chord_changes: F=14.426, p=0.000000 ***
  10. harmony_complexity: F=14.426, p=0.000000 ***

Detailed ANOVA results saved to: results/anova_results.csv

EDA completed! Results saved in results/
Emotion distribution:
emotion
happy      553
sad        338
hopeful    174
tense       13
Name: count, dtype: int64

3. Balancing dataset with GAN...
Epoch 0, D Loss: 0.6887, G Loss: 0.7039
Epoch 20, D Loss: 0.8933, G Loss: 0.4309
Epoch 40, D Loss: 0.8834, G Loss: 0.4259
Generating 215 samples for sad
Generating 379 samples for hopeful
Generating 540 samples for tense
Balanced dataset emotion distribution:
emotion
happy      553
sad        553
hopeful    553
tense      553
Name: count, dtype: int64

3.5. Exploratory Data Analysis (Balanced Dataset)...

============================================================
EXPLORATORY DATA ANALYSIS: Balanced Dataset (After GAN)
============================================================

1. BASIC DATASET INFORMATION
========================================
Total samples: 2212
Total features: 23
Memory usage: 0.70 MB
Missing values: 1134
Duplicate rows: 0

Data types:
float64    21
object      2
Name: count, dtype: int64

2. EMOTION DISTRIBUTION ANALYSIS
========================================
Emotion distribution:
  happy: 553 samples (25.0%)
  sad: 553 samples (25.0%)
  hopeful: 553 samples (25.0%)
  tense: 553 samples (25.0%)

Dataset balance analysis:
  Most frequent emotion: happy (553 samples)
  Least frequent emotion: happy (553 samples)
  Balance ratio: 1.000 (1.0 = perfectly balanced)
  OK: Dataset is reasonably balanced (ratio >= 0.5)

3. MUSICAL FEATURE ANALYSIS
========================================
Number of numeric features: 21

Feature statistics:
       duration  pitch_min  pitch_max  ...  tempo_variation  dynamic_range  dynamic_contrast
count  2212.000   2212.000   2212.000  ...         2212.000       2212.000          2212.000  
mean     35.526     29.597     80.538  ...            0.506         70.212            12.391  
std       8.854      7.655      7.872  ...            0.494         12.658             1.994  
min      18.009     22.000     63.000  ...            0.000         28.000             4.896  
25%      29.990     23.000     74.053  ...            0.000         62.000            11.677  
50%      34.537     26.453     79.645  ...            0.957         72.000            12.615  
75%      39.419     34.000     84.000  ...            0.991         84.719            13.369  
max     109.990     69.000    105.000  ...            0.999         99.000            24.903  

[8 rows x 21 columns]

Feature correlation with emotions:
  happy: ['total_notes(266.995)', 'chord_changes(145.190)', 'harmony_complexity(145.190)']
  sad: ['total_notes(212.496)', 'tempo_mean(120.387)', 'harmony_complexity(98.427)']
  hopeful: ['total_notes(252.078)', 'harmony_complexity(125.702)', 'tempo_mean(120.683)']     
  tense: ['total_notes(339.932)', 'harmony_complexity(204.623)', 'chord_changes(161.634)']    

4. FEATURE CORRELATION ANALYSIS
========================================

Highly correlated feature pairs (|r| > 0.7):
  tempo_mean <-> tempo_std: 1.000
  tempo_std <-> tempo_variation: 1.000
  tempo_mean <-> tempo_variation: 1.000
  velocity_std <-> dynamic_contrast: 0.949
  chord_changes <-> harmony_complexity: 0.947
  harmony_complexity <-> total_notes: 0.944
  velocity_max <-> velocity_mean: 0.934
  chord_changes <-> total_notes: 0.906
  rhythm_mean <-> rhythm_std: 0.877
  pitch_min <-> pitch_std: -0.764

5. STATISTICAL TESTS
========================================
ANOVA tests for feature differences across emotions:

Top 10 most significant features (ANOVA):
   1. pitch_mean: F=749.494, p=0.000000 ***
   2. velocity_max: F=1007.945, p=0.000000 ***
   3. velocity_mean: F=1273.006, p=0.000000 ***
   4. tempo_mean: F=808.368, p=0.000000 ***
   5. tempo_std: F=808.468, p=0.000000 ***
   6. tempo_variation: F=807.418, p=0.000000 ***
   7. pitch_max: F=672.139, p=0.000000 ***
   8. dynamic_range: F=653.115, p=0.000000 ***
   9. note_overlap_ratio: F=624.096, p=0.000000 ***
  10. pitch_std: F=553.131, p=0.000000 ***

Detailed ANOVA results saved to: results/anova_results.csv

EDA completed! Results saved in results/

6. DATASET COMPARISON (Original vs Balanced)
==================================================

Emotion distribution comparison:
         Original  Balanced
emotion
happy         553       553
sad           338       553
hopeful       174       553
tense          13       553

Balance improvement:
  Original balance ratio: 0.024
  Balanced balance ratio: 1.000
  Improvement: 4153.8%

3.6. Model Training Visualizations...

7. MODEL TRAINING VISUALIZATION
==================================================

Training Analysis Summary:
  Total epochs: 50
  Final D loss: 0.8908
  Final G loss: 0.4246
  D loss range: 0.6887 - 0.9011
  G loss range: 0.4246 - 0.7039
  Training stability: D_std=0.0453, G_std=0.0619
  Training quality: STABLE
  Training visualization saved to: results/gan_training_analysis.png

8. MODEL ARCHITECTURE VISUALIZATION
==================================================
Model architecture visualization saved to: results/model_architecture.png

9. FEATURE IMPORTANCE ANALYSIS
==================================================
Feature importance analysis saved to: results/feature_importance.png
Top 5 most important features:
  1. velocity_mean: 6.906
  2. velocity_max: 5.468
  3. tempo_std: 4.386
  4. tempo_mean: 4.385
  5. tempo_variation: 4.380

4. Training recommendation system...
Training recommendation system...
Trained on 2212 samples
Feature dimensions: 21
Recommendation system stats:
{'total_samples': 2212, 'emotion_distribution': {'happy': 553, 'sad': 553, 'hopeful': 553, 'tense': 553}, 'feature_dimensions': 21}

5. Testing recommendation system...
Query file: ../EMOPIA_1.0/midis\Q1_2Z9SjI131jA_0.mid
Query emotion category: happy

Top 5 similar tracks:
1. Q4_gxsdWKc1QaM_0.mid (similarity: 0.880, emotion: happy)
2. Q4_iHPKusssXzk_0.mid (similarity: 0.870, emotion: happy)
3. Q1_2Z9SjI131jA_12.mid (similarity: 0.868, emotion: happy)
4. Q2_5CEAeMiXKaA_0.mid (similarity: 0.865, emotion: happy)
5. Q4_k6aLLVX1D7Y_1.mid (similarity: 0.860, emotion: happy)

Test Accuracy: 100.0% (5/5 correct predictions)
Query emotion category: happy

System completed successfully!
Results saved in results/ directory
--------------------------------------------------------------

Track-23-Q3_7JIdJLkJ0S4_1
1. Survey Results
Happy:3.3%
Sad:20%
Hopeful:0%
Fearful:36.7%
Tense:33.3%
Excited:6.7%

2. System Results:
Music Emotion Analysis and Recommendation System
==================================================

1. Extracting features from MIDI files...
Found 1078 MIDI files
Processing file 1/1078
Processing file 101/1078
Processing file 201/1078
Processing file 301/1078
Processing file 401/1078
Processing file 501/1078
Processing file 601/1078
Processing file 701/1078
Processing file 801/1078
Processing file 901/1078
Processing file 1001/1078
Extracted features from 1078 files

2. Classifying emotions...

2.5. Exploratory Data Analysis (Classified Dataset)...

============================================================
EXPLORATORY DATA ANALYSIS: Classified EMOPIA Dataset
============================================================

1. BASIC DATASET INFORMATION
========================================
Total samples: 1078
Total features: 23
Memory usage: 0.42 MB
Missing values: 0
Duplicate rows: 0

Data types:
float64    13
int64       8
object      2
Name: count, dtype: int64

2. EMOTION DISTRIBUTION ANALYSIS
========================================
Emotion distribution:
  happy: 553 samples (51.3%)
  sad: 338 samples (31.4%)
  hopeful: 174 samples (16.1%)
  tense: 13 samples (1.2%)

Dataset balance analysis:
  Most frequent emotion: happy (553 samples)
  Least frequent emotion: tense (13 samples)
  Balance ratio: 0.024 (1.0 = perfectly balanced)
  WARNING: Dataset is highly imbalanced (ratio < 0.1)

3. MUSICAL FEATURE ANALYSIS
========================================
Number of numeric features: 21

Feature statistics:
       duration  pitch_min  pitch_max  ...  tempo_variation  dynamic_range  dynamic_contrast
count  1078.000   1078.000   1078.000  ...           1078.0       1078.000          1078.000  
mean     39.924     34.680     85.633  ...              0.0         61.499            11.939  
std      10.475      8.154      8.248  ...              0.0         11.879             2.714  
min      18.009     22.000     63.000  ...              0.0         28.000             4.896  
25%      29.990     28.000     80.000  ...              0.0         53.000            10.017  
50%      39.584     34.500     85.000  ...              0.0         62.000            11.762  
75%      46.725     40.000     91.000  ...              0.0         69.000            13.607  
max     109.990     69.000    105.000  ...              0.0         99.000            24.903  

[8 rows x 21 columns]

Feature correlation with emotions:
  happy: ['total_notes(266.995)', 'chord_changes(145.190)', 'harmony_complexity(145.190)']
  sad: ['total_notes(235.935)', 'tempo_mean(120.000)', 'chord_changes(115.003)']
  hopeful: ['total_notes(305.592)', 'chord_changes(170.138)', 'harmony_complexity(170.138)']  
  tense: ['total_notes(390.077)', 'chord_changes(247.077)', 'harmony_complexity(247.077)']    

4. FEATURE CORRELATION ANALYSIS
========================================

Highly correlated feature pairs (|r| > 0.7):
  velocity_std <-> dynamic_contrast: 1.000
  chord_changes <-> harmony_complexity: 1.000
  chord_changes <-> total_notes: 0.952
  harmony_complexity <-> total_notes: 0.952
  velocity_max <-> velocity_mean: 0.892
  rhythm_mean <-> rhythm_std: 0.821
  velocity_min <-> velocity_mean: 0.781

5. STATISTICAL TESTS
========================================
ANOVA tests for feature differences across emotions:
C:\Users\perer\AppData\Local\Programs\Python\Python311\Lib\site-packages\scipy\stats\_stats_py.py:4167: ConstantInputWarning: Each of the input arrays is constant;the F statistic is not defined or infinite
  warnings.warn(stats.ConstantInputWarning(msg))

Top 10 most significant features (ANOVA):
   1. pitch_mean: F=133.385, p=0.000000 ***
   2. pitch_max: F=100.888, p=0.000000 ***
   3. velocity_mean: F=73.824, p=0.000000 ***
   4. velocity_max: F=61.414, p=0.000000 ***
   5. note_overlap_ratio: F=42.039, p=0.000000 ***
   6. velocity_min: F=36.637, p=0.000000 ***
   7. pitch_std: F=28.778, p=0.000000 ***
   8. syncopation: F=24.819, p=0.000000 ***
   9. chord_changes: F=14.426, p=0.000000 ***
  10. harmony_complexity: F=14.426, p=0.000000 ***

Detailed ANOVA results saved to: results/anova_results.csv

EDA completed! Results saved in results/
Emotion distribution:
emotion
happy      553
sad        338
hopeful    174
tense       13
Name: count, dtype: int64

3. Balancing dataset with GAN...
Epoch 0, D Loss: 0.7063, G Loss: 0.6742
Epoch 20, D Loss: 0.9354, G Loss: 0.3677
Epoch 40, D Loss: 0.9380, G Loss: 0.3614
Generating 215 samples for sad
Generating 379 samples for hopeful
Generating 540 samples for tense
Balanced dataset emotion distribution:
emotion
happy      553
sad        553
hopeful    553
tense      553
Name: count, dtype: int64

3.5. Exploratory Data Analysis (Balanced Dataset)...

============================================================
EXPLORATORY DATA ANALYSIS: Balanced Dataset (After GAN)
============================================================

1. BASIC DATASET INFORMATION
========================================
Total samples: 2212
Total features: 23
Memory usage: 0.70 MB
Missing values: 1134
Duplicate rows: 0

Data types:
float64    21
object      2
Name: count, dtype: int64

2. EMOTION DISTRIBUTION ANALYSIS
========================================
Emotion distribution:
  happy: 553 samples (25.0%)
  sad: 553 samples (25.0%)
  hopeful: 553 samples (25.0%)
  tense: 553 samples (25.0%)

Dataset balance analysis:
  Most frequent emotion: happy (553 samples)
  Least frequent emotion: happy (553 samples)
  Balance ratio: 1.000 (1.0 = perfectly balanced)
  OK: Dataset is reasonably balanced (ratio >= 0.5)

3. MUSICAL FEATURE ANALYSIS
========================================
Number of numeric features: 21

Feature statistics:
       duration  pitch_min  pitch_max  ...  tempo_variation  dynamic_range  dynamic_contrast
count  2212.000   2212.000   2212.000  ...         2212.000       2212.000          2212.000  
mean     44.835     29.605     88.468  ...           -0.484         55.883            10.426  
std       8.935      7.651      6.678  ...            0.473          9.986             2.415  
min      18.009     22.000     63.000  ...           -0.997         28.000             4.896  
25%      39.752     23.000     85.000  ...           -0.955         49.276             9.229  
50%      46.796     26.526     90.228  ...           -0.829         51.931             9.271  
75%      51.802     34.000     94.559  ...            0.000         61.000            11.698  
max     109.990     69.000    105.000  ...            0.000         99.000            24.903  

[8 rows x 21 columns]

Feature correlation with emotions:
  happy: ['total_notes(266.995)', 'chord_changes(145.190)', 'harmony_complexity(145.190)']
  sad: ['total_notes(285.681)', 'harmony_complexity(148.765)', 'tempo_mean(119.622)']
  hopeful: ['total_notes(426.301)', 'harmony_complexity(262.186)', 'tempo_mean(119.334)']     
  tense: ['total_notes(501.766)', 'harmony_complexity(332.879)', 'chord_changes(161.499)']    

4. FEATURE CORRELATION ANALYSIS
========================================

Highly correlated feature pairs (|r| > 0.7):
  velocity_std <-> dynamic_contrast: 1.000
  tempo_mean <-> tempo_std: -0.999
  tempo_mean <-> tempo_variation: 0.998
  tempo_std <-> tempo_variation: -0.998
  harmony_complexity <-> total_notes: 0.976
  velocity_max <-> velocity_mean: 0.948
  velocity_min <-> velocity_mean: 0.913
  rhythm_mean <-> rhythm_std: 0.841
  velocity_min <-> velocity_max: 0.836
  dynamic_range <-> dynamic_contrast: 0.788
  velocity_std <-> dynamic_range: 0.787
  velocity_mean <-> harmony_complexity: 0.740
  velocity_max <-> harmony_complexity: 0.738
  velocity_max <-> total_notes: 0.723
  velocity_mean <-> total_notes: 0.715
  pitch_min <-> pitch_mean: 0.713

5. STATISTICAL TESTS
========================================
ANOVA tests for feature differences across emotions:

Top 10 most significant features (ANOVA):
   1. pitch_mean: F=970.767, p=0.000000 ***
   2. velocity_min: F=1056.477, p=0.000000 ***
   3. velocity_max: F=850.591, p=0.000000 ***
   4. velocity_mean: F=1175.436, p=0.000000 ***
   5. tempo_mean: F=805.883, p=0.000000 ***
   6. tempo_std: F=805.155, p=0.000000 ***
   7. tempo_variation: F=799.566, p=0.000000 ***
   8. note_overlap_ratio: F=669.921, p=0.000000 ***
   9. harmony_complexity: F=622.956, p=0.000000 ***
  10. total_notes: F=546.273, p=0.000000 ***

Detailed ANOVA results saved to: results/anova_results.csv

EDA completed! Results saved in results/

6. DATASET COMPARISON (Original vs Balanced)
==================================================

Emotion distribution comparison:
         Original  Balanced
emotion
happy         553       553
sad           338       553
hopeful       174       553
tense          13       553

Balance improvement:
  Original balance ratio: 0.024
  Balanced balance ratio: 1.000
  Improvement: 4153.8%

3.6. Model Training Visualizations...

7. MODEL TRAINING VISUALIZATION
==================================================

Training Analysis Summary:
  Total epochs: 50
  Final D loss: 0.9420
  Final G loss: 0.3609
  D loss range: 0.7063 - 0.9511
  G loss range: 0.3606 - 0.6742
  Training stability: D_std=0.0573, G_std=0.0696
  Training quality: STABLE
  Training visualization saved to: results/gan_training_analysis.png

8. MODEL ARCHITECTURE VISUALIZATION
==================================================
Model architecture visualization saved to: results/model_architecture.png

9. FEATURE IMPORTANCE ANALYSIS
==================================================
Feature importance analysis saved to: results/feature_importance.png
Top 5 most important features:
  1. velocity_mean: 6.377
  2. velocity_min: 5.731
  3. pitch_mean: 5.266
  4. velocity_max: 4.614
  5. tempo_mean: 4.372

4. Training recommendation system...
Training recommendation system...
Trained on 2212 samples
Feature dimensions: 21
Recommendation system stats:
{'total_samples': 2212, 'emotion_distribution': {'happy': 553, 'sad': 553, 'hopeful': 553, 'tense': 553}, 'feature_dimensions': 21}

5. Testing recommendation system...
Query file: ../EMOPIA_1.0/midis\Q3_7JIdJLkJ0S4_1.mid
Query emotion category: sad

Top 5 similar tracks:
1. Q1_im4Qxn3GQvo_0.mid (similarity: 0.858, emotion: sad)
2. Q2_Ie5koh4qvJc_10.mid (similarity: 0.842, emotion: sad)
3. Q3_wlOa1jkTn_U_0.mid (similarity: 0.809, emotion: sad)
4. Q1_GbUV3TXUzeQ_2.mid (similarity: 0.801, emotion: sad)
5. Q1_1Qc15G0ZHIg_1.mid (similarity: 0.797, emotion: sad)

Test Accuracy: 100.0% (5/5 correct predictions)
Query emotion category: sad

System completed successfully!
Results saved in results/ directory
--------------------------------------------------------------

Track-24-Q3_Adil-zpJdEs_0
1. Survey Results
Happy:3.3%
Sad:53.3%
Hopeful:16.7%
Fearful:3.3%
Tense:13.3%
Excited:10%

2. System Results:
Music Emotion Analysis and Recommendation System
==================================================

1. Extracting features from MIDI files...
Found 1078 MIDI files
Processing file 1/1078
Processing file 101/1078
Processing file 201/1078
Processing file 301/1078
Processing file 401/1078
Processing file 501/1078
Processing file 601/1078
Processing file 701/1078
Processing file 801/1078
Processing file 901/1078
Processing file 1001/1078
Extracted features from 1078 files

2. Classifying emotions...

2.5. Exploratory Data Analysis (Classified Dataset)...

============================================================
EXPLORATORY DATA ANALYSIS: Classified EMOPIA Dataset
============================================================

1. BASIC DATASET INFORMATION
========================================
Total samples: 1078
Total features: 23
Memory usage: 0.42 MB
Missing values: 0
Duplicate rows: 0

Data types:
float64    13
int64       8
object      2
Name: count, dtype: int64

2. EMOTION DISTRIBUTION ANALYSIS
========================================
Emotion distribution:
  happy: 553 samples (51.3%)
  sad: 338 samples (31.4%)
  hopeful: 174 samples (16.1%)
  tense: 13 samples (1.2%)

Dataset balance analysis:
  Most frequent emotion: happy (553 samples)
  Least frequent emotion: tense (13 samples)
  Balance ratio: 0.024 (1.0 = perfectly balanced)
  WARNING: Dataset is highly imbalanced (ratio < 0.1)

3. MUSICAL FEATURE ANALYSIS
========================================
Number of numeric features: 21

Feature statistics:
       duration  pitch_min  pitch_max  ...  tempo_variation  dynamic_range  dynamic_contrast
count  1078.000   1078.000   1078.000  ...           1078.0       1078.000          1078.000  
mean     39.924     34.680     85.633  ...              0.0         61.499            11.939  
std      10.475      8.154      8.248  ...              0.0         11.879             2.714  
min      18.009     22.000     63.000  ...              0.0         28.000             4.896  
25%      29.990     28.000     80.000  ...              0.0         53.000            10.017  
50%      39.584     34.500     85.000  ...              0.0         62.000            11.762  
75%      46.725     40.000     91.000  ...              0.0         69.000            13.607  
max     109.990     69.000    105.000  ...              0.0         99.000            24.903  

[8 rows x 21 columns]

Feature correlation with emotions:
  happy: ['total_notes(266.995)', 'chord_changes(145.190)', 'harmony_complexity(145.190)']
  sad: ['total_notes(235.935)', 'tempo_mean(120.000)', 'chord_changes(115.003)']
  hopeful: ['total_notes(305.592)', 'chord_changes(170.138)', 'harmony_complexity(170.138)']  
  tense: ['total_notes(390.077)', 'chord_changes(247.077)', 'harmony_complexity(247.077)']    

4. FEATURE CORRELATION ANALYSIS
========================================

Highly correlated feature pairs (|r| > 0.7):
  velocity_std <-> dynamic_contrast: 1.000
  chord_changes <-> harmony_complexity: 1.000
  chord_changes <-> total_notes: 0.952
  harmony_complexity <-> total_notes: 0.952
  velocity_max <-> velocity_mean: 0.892
  rhythm_mean <-> rhythm_std: 0.821
  velocity_min <-> velocity_mean: 0.781

5. STATISTICAL TESTS
========================================
ANOVA tests for feature differences across emotions:
C:\Users\perer\AppData\Local\Programs\Python\Python311\Lib\site-packages\scipy\stats\_stats_py.py:4167: ConstantInputWarning: Each of the input arrays is constant;the F statistic is not defined or infinite
  warnings.warn(stats.ConstantInputWarning(msg))

Top 10 most significant features (ANOVA):
   1. pitch_mean: F=133.385, p=0.000000 ***
   2. pitch_max: F=100.888, p=0.000000 ***
   3. velocity_mean: F=73.824, p=0.000000 ***
   4. velocity_max: F=61.414, p=0.000000 ***
   5. note_overlap_ratio: F=42.039, p=0.000000 ***
   6. velocity_min: F=36.637, p=0.000000 ***
   7. pitch_std: F=28.778, p=0.000000 ***
   8. syncopation: F=24.819, p=0.000000 ***
   9. chord_changes: F=14.426, p=0.000000 ***
  10. harmony_complexity: F=14.426, p=0.000000 ***

Detailed ANOVA results saved to: results/anova_results.csv

EDA completed! Results saved in results/
Emotion distribution:
emotion
happy      553
sad        338
hopeful    174
tense       13
Name: count, dtype: int64

3. Balancing dataset with GAN...
Epoch 0, D Loss: 0.7065, G Loss: 0.5934
Epoch 20, D Loss: 0.9046, G Loss: 0.3647
Epoch 40, D Loss: 0.8898, G Loss: 0.3593
Generating 215 samples for sad
Generating 379 samples for hopeful
Generating 540 samples for tense
Balanced dataset emotion distribution:
emotion
happy      553
sad        553
hopeful    553
tense      553
Name: count, dtype: int64

3.5. Exploratory Data Analysis (Balanced Dataset)...

============================================================
EXPLORATORY DATA ANALYSIS: Balanced Dataset (After GAN)
============================================================

1. BASIC DATASET INFORMATION
========================================
Total samples: 2212
Total features: 23
Memory usage: 0.70 MB
Missing values: 1134
Duplicate rows: 0

Data types:
float64    21
object      2
Name: count, dtype: int64

2. EMOTION DISTRIBUTION ANALYSIS
========================================
Emotion distribution:
  happy: 553 samples (25.0%)
  sad: 553 samples (25.0%)
  hopeful: 553 samples (25.0%)
  tense: 553 samples (25.0%)

Dataset balance analysis:
  Most frequent emotion: happy (553 samples)
  Least frequent emotion: happy (553 samples)
  Balance ratio: 1.000 (1.0 = perfectly balanced)
  OK: Dataset is reasonably balanced (ratio >= 0.5)

3. MUSICAL FEATURE ANALYSIS
========================================
Number of numeric features: 21

Feature statistics:
       duration  pitch_min  pitch_max  ...  tempo_variation  dynamic_range  dynamic_contrast
count  2212.000   2212.000   2212.000  ...         2212.000       2212.000          2212.000  
mean     35.519     29.713     88.470  ...            0.511         55.789            10.476  
std       8.858      7.598      6.679  ...            0.498         10.034             2.386  
min      18.009     22.000     63.000  ...            0.000         28.000             4.896  
25%      29.990     23.365     85.000  ...            0.000         49.180             9.240  
50%      34.534     26.797     90.229  ...            0.988         51.760             9.359  
75%      39.419     34.000     94.595  ...            0.998         61.000            11.698  
max     109.990     69.000    105.000  ...            1.000         99.000            24.903  

[8 rows x 21 columns]

Feature correlation with emotions:
  happy: ['total_notes(266.995)', 'chord_changes(145.190)', 'harmony_complexity(145.190)']
  sad: ['total_notes(285.678)', 'tempo_mean(119.613)', 'velocity_max(90.229)']
  hopeful: ['total_notes(426.340)', 'tempo_mean(119.318)', 'velocity_max(105.083)']
  tense: ['total_notes(501.817)', 'harmony_complexity(161.287)', 'chord_changes(161.168)']    

4. FEATURE CORRELATION ANALYSIS
========================================

Highly correlated feature pairs (|r| > 0.7):
  chord_changes <-> harmony_complexity: 1.000
  tempo_mean <-> tempo_variation: -1.000
  tempo_std <-> tempo_variation: -1.000
  tempo_mean <-> tempo_std: 1.000
  velocity_std <-> dynamic_contrast: 1.000
  rhythm_mean <-> rhythm_std: 0.867
  velocity_min <-> velocity_max: 0.836
  velocity_std <-> dynamic_range: 0.789
  dynamic_range <-> dynamic_contrast: 0.787
  velocity_max <-> velocity_mean: 0.772
  syncopation <-> note_overlap_ratio: -0.766
  pitch_std <-> velocity_max: 0.765
  syncopation <-> total_notes: 0.764
  pitch_min <-> pitch_std: -0.746
  rhythm_mean <-> total_notes: -0.744
  velocity_min <-> velocity_mean: 0.737
  rhythm_std <-> total_notes: -0.730
  velocity_max <-> total_notes: 0.723
  tempo_std <-> syncopation: -0.720
  syncopation <-> tempo_variation: 0.720
  tempo_mean <-> syncopation: -0.720
  pitch_min <-> pitch_mean: 0.712
  pitch_std <-> velocity_min: 0.705

5. STATISTICAL TESTS
========================================
ANOVA tests for feature differences across emotions:

Top 10 most significant features (ANOVA):
   1. pitch_mean: F=970.391, p=0.000000 ***
   2. velocity_min: F=1057.879, p=0.000000 ***
   3. velocity_max: F=851.252, p=0.000000 ***
   4. velocity_mean: F=1271.347, p=0.000000 ***
   5. tempo_mean: F=808.395, p=0.000000 ***
   6. tempo_std: F=809.016, p=0.000000 ***
   7. syncopation: F=770.031, p=0.000000 ***
   8. tempo_variation: F=808.640, p=0.000000 ***
   9. note_overlap_ratio: F=670.234, p=0.000000 ***
  10. total_notes: F=546.489, p=0.000000 ***

Detailed ANOVA results saved to: results/anova_results.csv

EDA completed! Results saved in results/

6. DATASET COMPARISON (Original vs Balanced)
==================================================

Emotion distribution comparison:
         Original  Balanced
emotion
happy         553       553
sad           338       553
hopeful       174       553
tense          13       553

Balance improvement:
  Original balance ratio: 0.024
  Balanced balance ratio: 1.000
  Improvement: 4153.8%

3.6. Model Training Visualizations...

7. MODEL TRAINING VISUALIZATION
==================================================

Training Analysis Summary:
  Total epochs: 50
  Final D loss: 0.8963
  Final G loss: 0.3588
  D loss range: 0.7065 - 0.9063
  G loss range: 0.3586 - 0.5934
  Training stability: D_std=0.0429, G_std=0.0470
  Training quality: STABLE
  Training visualization saved to: results/gan_training_analysis.png

8. MODEL ARCHITECTURE VISUALIZATION
==================================================
Model architecture visualization saved to: results/model_architecture.png

9. FEATURE IMPORTANCE ANALYSIS
==================================================
Feature importance analysis saved to: results/feature_importance.png
Top 5 most important features:
  1. velocity_mean: 6.897
  2. velocity_min: 5.739
  3. pitch_mean: 5.264
  4. velocity_max: 4.618
  5. tempo_std: 4.389

4. Training recommendation system...
Training recommendation system...
Trained on 2212 samples
Feature dimensions: 21
Recommendation system stats:
{'total_samples': 2212, 'emotion_distribution': {'happy': 553, 'sad': 553, 'hopeful': 553, 'tense': 553}, 'feature_dimensions': 21}

5. Testing recommendation system...
Query file: ../EMOPIA_1.0/midis\Q3_Adil-zpJdEs_0.mid
Query emotion category: happy

Top 5 similar tracks:
1. Q3__vZOEQCYSaY_2.mid (similarity: 0.904, emotion: happy)
2. Q3_K84TcgjCRt4_2.mid (similarity: 0.892, emotion: happy)
3. Q4_HYVmgq5Y93g_0.mid (similarity: 0.890, emotion: happy)
4. Q3_2v6gJi03LlA_1.mid (similarity: 0.874, emotion: happy)
5. Q3_TYkTOwBfFB8_1.mid (similarity: 0.869, emotion: happy)

Test Accuracy: 100.0% (5/5 correct predictions)
Query emotion category: happy

System completed successfully!
Results saved in results/ directory
-------------------------------------------------------------------------------

Track-25-Q3_2eIsQtm4YNs_1
1. Survey Results
Happy:10%
Sad:36.7%
Hopeful:26.7%
Fearful:3.3%
Tense:3.3%
Excited:20%

2. System Results:
Music Emotion Analysis and Recommendation System
==================================================

1. Extracting features from MIDI files...
Found 1078 MIDI files
Processing file 1/1078
Processing file 101/1078
Processing file 201/1078
Processing file 301/1078
Processing file 401/1078
Processing file 501/1078
Processing file 601/1078
Processing file 701/1078
Processing file 801/1078
Processing file 901/1078
Processing file 1001/1078
Extracted features from 1078 files

2. Classifying emotions...

2.5. Exploratory Data Analysis (Classified Dataset)...

============================================================
EXPLORATORY DATA ANALYSIS: Classified EMOPIA Dataset
============================================================

1. BASIC DATASET INFORMATION
========================================
Total samples: 1078
Total features: 23
Memory usage: 0.42 MB
Missing values: 0
Duplicate rows: 0

Data types:
float64    13
int64       8
object      2
Name: count, dtype: int64

2. EMOTION DISTRIBUTION ANALYSIS
========================================
Emotion distribution:
  happy: 553 samples (51.3%)
  sad: 338 samples (31.4%)
  hopeful: 174 samples (16.1%)
  tense: 13 samples (1.2%)

Dataset balance analysis:
  Most frequent emotion: happy (553 samples)
  Least frequent emotion: tense (13 samples)
  Balance ratio: 0.024 (1.0 = perfectly balanced)
  WARNING: Dataset is highly imbalanced (ratio < 0.1)

3. MUSICAL FEATURE ANALYSIS
========================================
Number of numeric features: 21

Feature statistics:
       duration  pitch_min  pitch_max  ...  tempo_variation  dynamic_range  dynamic_contrast
count  1078.000   1078.000   1078.000  ...           1078.0       1078.000          1078.000  
mean     39.924     34.680     85.633  ...              0.0         61.499            11.939  
std      10.475      8.154      8.248  ...              0.0         11.879             2.714  
min      18.009     22.000     63.000  ...              0.0         28.000             4.896  
25%      29.990     28.000     80.000  ...              0.0         53.000            10.017  
50%      39.584     34.500     85.000  ...              0.0         62.000            11.762  
75%      46.725     40.000     91.000  ...              0.0         69.000            13.607  
max     109.990     69.000    105.000  ...              0.0         99.000            24.903  

[8 rows x 21 columns]

Feature correlation with emotions:
  happy: ['total_notes(266.995)', 'chord_changes(145.190)', 'harmony_complexity(145.190)']
  sad: ['total_notes(235.935)', 'tempo_mean(120.000)', 'chord_changes(115.003)']
  hopeful: ['total_notes(305.592)', 'chord_changes(170.138)', 'harmony_complexity(170.138)']  
  tense: ['total_notes(390.077)', 'chord_changes(247.077)', 'harmony_complexity(247.077)']    

4. FEATURE CORRELATION ANALYSIS
========================================

Highly correlated feature pairs (|r| > 0.7):
  velocity_std <-> dynamic_contrast: 1.000
  chord_changes <-> harmony_complexity: 1.000
  chord_changes <-> total_notes: 0.952
  harmony_complexity <-> total_notes: 0.952
  velocity_max <-> velocity_mean: 0.892
  rhythm_mean <-> rhythm_std: 0.821
  velocity_min <-> velocity_mean: 0.781

5. STATISTICAL TESTS
========================================
ANOVA tests for feature differences across emotions:
C:\Users\perer\AppData\Local\Programs\Python\Python311\Lib\site-packages\scipy\stats\_stats_py.py:4167: ConstantInputWarning: Each of the input arrays is constant;the F statistic is not defined or infinite
  warnings.warn(stats.ConstantInputWarning(msg))

Top 10 most significant features (ANOVA):
   1. pitch_mean: F=133.385, p=0.000000 ***
   2. pitch_max: F=100.888, p=0.000000 ***
   3. velocity_mean: F=73.824, p=0.000000 ***
   4. velocity_max: F=61.414, p=0.000000 ***
   5. note_overlap_ratio: F=42.039, p=0.000000 ***
   6. velocity_min: F=36.637, p=0.000000 ***
   7. pitch_std: F=28.778, p=0.000000 ***
   8. syncopation: F=24.819, p=0.000000 ***
   9. chord_changes: F=14.426, p=0.000000 ***
  10. harmony_complexity: F=14.426, p=0.000000 ***

Detailed ANOVA results saved to: results/anova_results.csv

EDA completed! Results saved in results/
Emotion distribution:
emotion
happy      553
sad        338
hopeful    174
tense       13
Name: count, dtype: int64

3. Balancing dataset with GAN...
Epoch 0, D Loss: 0.7202, G Loss: 0.7281
Epoch 20, D Loss: 0.8765, G Loss: 0.4846
Epoch 40, D Loss: 0.8958, G Loss: 0.4795
Generating 215 samples for sad
Generating 379 samples for hopeful
Generating 540 samples for tense
Balanced dataset emotion distribution:
emotion
happy      553
sad        553
hopeful    553
tense      553
Name: count, dtype: int64

3.5. Exploratory Data Analysis (Balanced Dataset)...

============================================================
EXPLORATORY DATA ANALYSIS: Balanced Dataset (After GAN)
============================================================

1. BASIC DATASET INFORMATION
========================================
Total samples: 2212
Total features: 23
Memory usage: 0.70 MB
Missing values: 1134
Duplicate rows: 0

Data types:
float64    21
object      2
Name: count, dtype: int64

2. EMOTION DISTRIBUTION ANALYSIS
========================================
Emotion distribution:
  happy: 553 samples (25.0%)
  sad: 553 samples (25.0%)
  hopeful: 553 samples (25.0%)
  tense: 553 samples (25.0%)

Dataset balance analysis:
  Most frequent emotion: happy (553 samples)
  Least frequent emotion: happy (553 samples)
  Balance ratio: 1.000 (1.0 = perfectly balanced)
  OK: Dataset is reasonably balanced (ratio >= 0.5)

3. MUSICAL FEATURE ANALYSIS
========================================
Number of numeric features: 21

Feature statistics:
       duration  pitch_min  pitch_max  ...  tempo_variation  dynamic_range  dynamic_contrast
count  2212.000   2212.000   2212.000  ...         2212.000       2212.000          2212.000  
mean     35.526     29.598     88.429  ...           -0.393         55.789            12.992  
std       8.854      7.655      6.661  ...            0.392         10.034             2.190  
min      18.009     22.000     63.000  ...           -0.966         28.000             4.896  
25%      29.990     23.000     85.000  ...           -0.795         49.181            11.872  
50%      34.537     26.464     90.127  ...           -0.498         51.757            13.421  
75%      39.419     34.000     94.226  ...            0.000         61.000            14.540  
max     109.990     69.000    105.000  ...            0.000         99.000            24.903  

[8 rows x 21 columns]

Feature correlation with emotions:
  happy: ['total_notes(266.995)', 'chord_changes(145.190)', 'harmony_complexity(145.190)']
  sad: ['total_notes(186.156)', 'harmony_complexity(148.716)', 'chord_changes(126.230)']      
  hopeful: ['harmony_complexity(261.996)', 'chord_changes(200.917)', 'total_notes(184.864)']  
  tense: ['harmony_complexity(332.727)', 'total_notes(278.302)', 'chord_changes(276.157)']    

4. FEATURE CORRELATION ANALYSIS
========================================

Highly correlated feature pairs (|r| > 0.7):
  velocity_std <-> dynamic_contrast: 1.000
  tempo_mean <-> tempo_variation: -0.978
  chord_changes <-> harmony_complexity: 0.952
  velocity_max <-> velocity_mean: 0.944
  velocity_min <-> velocity_mean: 0.912
  velocity_min <-> velocity_max: 0.821
  harmony_complexity <-> syncopation: 0.759
  harmony_complexity <-> rhythm_mean: -0.741
  velocity_mean <-> harmony_complexity: 0.740
  rhythm_mean <-> syncopation: -0.714
  tempo_mean <-> syncopation: 0.711
  velocity_max <-> harmony_complexity: 0.706

5. STATISTICAL TESTS
========================================
ANOVA tests for feature differences across emotions:

Top 10 most significant features (ANOVA):
   1. pitch_mean: F=746.980, p=0.000000 ***
   2. velocity_min: F=1029.060, p=0.000000 ***
   3. velocity_max: F=921.293, p=0.000000 ***
   4. velocity_mean: F=1171.547, p=0.000000 ***
   5. tempo_mean: F=808.882, p=0.000000 ***
   6. syncopation: F=750.976, p=0.000000 ***
   7. tempo_variation: F=740.213, p=0.000000 ***
   8. harmony_complexity: F=622.376, p=0.000000 ***
   9. note_overlap_ratio: F=416.682, p=0.000000 ***
  10. chord_changes: F=396.066, p=0.000000 ***

Detailed ANOVA results saved to: results/anova_results.csv

EDA completed! Results saved in results/

6. DATASET COMPARISON (Original vs Balanced)
==================================================

Emotion distribution comparison:
         Original  Balanced
emotion
happy         553       553
sad           338       553
hopeful       174       553
tense          13       553

Balance improvement:
  Original balance ratio: 0.024
  Balanced balance ratio: 1.000
  Improvement: 4153.8%

3.6. Model Training Visualizations...

7. MODEL TRAINING VISUALIZATION
==================================================

Training Analysis Summary:
  Total epochs: 50
  Final D loss: 0.8730
  Final G loss: 0.4783
  D loss range: 0.7202 - 0.8958
  G loss range: 0.4783 - 0.7281
  Training stability: D_std=0.0356, G_std=0.0544
  Training quality: STABLE
  Training visualization saved to: results/gan_training_analysis.png

8. MODEL ARCHITECTURE VISUALIZATION
==================================================
Model architecture visualization saved to: results/model_architecture.png

9. FEATURE IMPORTANCE ANALYSIS
==================================================
Feature importance analysis saved to: results/feature_importance.png
Top 5 most important features:
  1. velocity_mean: 6.356
  2. velocity_min: 5.583
  3. velocity_max: 4.998
  4. tempo_mean: 4.388
  5. syncopation: 4.074

4. Training recommendation system...
Training recommendation system...
Trained on 2212 samples
Feature dimensions: 21
Recommendation system stats:
{'total_samples': 2212, 'emotion_distribution': {'happy': 553, 'sad': 553, 'hopeful': 553, 'tense': 553}, 'feature_dimensions': 21}

5. Testing recommendation system...
Query file: ../EMOPIA_1.0/midis\Q3_2eIsQtm4YNs_1.mid
Query emotion category: sad

Top 5 similar tracks:
1. Q4_bfopzItCYrE_2.mid (similarity: 0.858, emotion: sad)
2. Q1_K6OFDxBU370_0.mid (similarity: 0.825, emotion: sad)
3. Q4_fOYX0uH8mSQ_0.mid (similarity: 0.821, emotion: sad)
4. Q4_a5QCcwEjxAk_0.mid (similarity: 0.820, emotion: sad)
5. Q4_JxSU49jFKwM_1.mid (similarity: 0.803, emotion: sad)

Test Accuracy: 100.0% (5/5 correct predictions)
Query emotion category: sad

System completed successfully!
Results saved in results/ directory
--------------------------------------------------------------
