"""
Advanced Customer Voice & Churn Analysis System
Portfolio-Grade ML Pipeline with Topic Modeling & Power BI Integration

Author: Senior Data Scientist
Description: Predict customer churn, identify root causes, and export actionable insights
"""

import pandas as pd
import numpy as np
import re
import warnings
warnings.filterwarnings('ignore')

# NLP and ML libraries
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.decomposition import LatentDirichletAllocation

# Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Download required NLTK data
print("Downloading NLTK data...")
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
except:
    pass
print("✓ NLTK data ready\n")

# Set style for professional plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")

# ============================================================================
# 1. DATA LOADING & STRATEGIC LABELING
# ============================================================================

print("=" * 80)
print("STEP 1: DATA LOADING & STRATEGIC LABELING")
print("=" * 80)

# Load the dataset
df = pd.read_csv('bank_reviews3.csv')
print(f"✓ Dataset loaded successfully!")
print(f"  Initial shape: {df.shape[0]} rows × {df.shape[1]} columns\n")

# Show rating distribution before filtering
print("Original Rating Distribution:")
print(df['rating'].value_counts().sort_index())
print()

# Filter out neutral ratings (3.0) - keep only strong opinions
print("🔍 Filtering out neutral ratings (3.0)...")
df_filtered = df[df['rating'] != 3.0].copy()
print(f"✓ Filtered dataset: {len(df_filtered)} rows (removed {len(df) - len(df_filtered)} neutral reviews)\n")

# Create binary Churn_Label
# Churn (1): Ratings 1.0 and 2.0 (Negative/High Risk)
# Retained (0): Ratings 4.0 and 5.0 (Positive/Safe)
df_filtered['Churn_Label'] = df_filtered['rating'].apply(
    lambda x: 1 if x in [1.0, 2.0] else 0
)

# Display class distribution
print("Binary Class Distribution:")
print(df_filtered['Churn_Label'].value_counts())
print("\nPercentage Distribution:")
churn_pct = df_filtered['Churn_Label'].value_counts(normalize=True) * 100
print(f"  Retained (0): {churn_pct[0]:.2f}%")
print(f"  Churn (1):    {churn_pct[1]:.2f}%")
print()

# ============================================================================
# 2. ADVANCED NLP PREPROCESSING (LEMMATIZATION)
# ============================================================================

print("=" * 80)
print("STEP 2: ADVANCED NLP PREPROCESSING (LEMMATIZATION)")
print("=" * 80)

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    """
    Advanced text preprocessing with lemmatization
    
    Steps:
    1. Convert to lowercase
    2. Remove punctuation and numbers
    3. Remove stopwords
    4. Apply lemmatization (better than stemming - preserves word meaning)
    
    Args:
        text (str): Raw text input
        
    Returns:
        str: Cleaned and lemmatized text
    """
    if pd.isna(text):
        return ""
    
    # Convert to lowercase
    text = str(text).lower()
    
    # Remove punctuation and numbers (keep only letters and spaces)
    text = re.sub(r'[^a-z\s]', ' ', text)
    
    # Tokenize, remove stopwords, and filter short words
    words = text.split()
    words = [word for word in words if word not in stop_words and len(word) > 2]
    
    # Apply lemmatization (better than stemming - "running" → "run", "better" → "good")
    words = [lemmatizer.lemmatize(word) for word in words]
    
    # Join back into string
    return ' '.join(words)

# Apply cleaning function
print("Cleaning reviews with lemmatization... This may take a moment.")
df_filtered['Cleaned_Review'] = df_filtered['review'].apply(clean_text)
print(f"✓ Cleaned {len(df_filtered)} reviews successfully!\n")

# Display sample
print("Sample Cleaned Reviews (Lemmatization):")
print("-" * 80)
for i in range(2):
    print(f"Original: {df_filtered['review'].iloc[i][:100]}...")
    print(f"Cleaned:  {df_filtered['Cleaned_Review'].iloc[i][:100]}...")
    print()

# Remove rows with empty cleaned reviews
df_filtered = df_filtered[df_filtered['Cleaned_Review'].str.strip() != ''].copy()
print(f"✓ Final dataset: {len(df_filtered)} rows\n")

# ============================================================================
# 3. FEATURE ENGINEERING WITH N-GRAMS
# ============================================================================

print("=" * 80)
print("STEP 3: FEATURE ENGINEERING (TF-IDF + BIGRAMS)")
print("=" * 80)

# TF-IDF Vectorization with bigrams to capture phrases
print("Creating TF-IDF features with unigrams + bigrams...")
tfidf = TfidfVectorizer(
    max_features=2000,
    ngram_range=(1, 2),  # Captures phrases like "bad service", "hidden charges"
    min_df=2  # Must appear in at least 2 documents
)
X = tfidf.fit_transform(df_filtered['Cleaned_Review'])
y = df_filtered['Churn_Label']

# Get feature names for later use
feature_names = tfidf.get_feature_names_out()

print(f"✓ TF-IDF vectorization complete!")
print(f"  Feature matrix shape: {X.shape}")
print(f"  Number of features: {X.shape[1]}")
print(f"  Target variable shape: {y.shape}")
print(f"\nSample features (including bigrams):")
print(f"  {list(feature_names[:10])}")
print()

# ============================================================================
# 4. RANDOM FOREST CLASSIFICATION
# ============================================================================

print("=" * 80)
print("STEP 4: RANDOM FOREST CLASSIFICATION")
print("=" * 80)

# Train-test split (80-20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train set: {X_train.shape[0]} samples")
print(f"Test set:  {X_test.shape[0]} samples\n")

# Train Random Forest with balanced class weights
print("Training Random Forest Classifier (class_weight='balanced')...")
rf_model = RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced',  # Handles imbalanced data
    random_state=42,
    max_depth=15,
    min_samples_split=5,
    n_jobs=-1  # Use all CPU cores
)
rf_model.fit(X_train, y_train)
print("✓ Model training complete!\n")

# ============================================================================
# 5. MODEL EVALUATION
# ============================================================================

print("=" * 80)
print("STEP 5: MODEL EVALUATION")
print("=" * 80)

# Make predictions
y_pred = rf_model.predict(X_test)
y_pred_proba = rf_model.predict_proba(X_test)

# Accuracy Score
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy Score: {accuracy:.4f} ({accuracy*100:.2f}%)\n")

# Classification Report
print("Classification Report:")
print("=" * 80)
print(classification_report(y_test, y_pred, 
                           target_names=['Retained (0)', 'Churn (1)'],
                           digits=3))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

# ============================================================================
# 6. TOPIC MODELING - ROOT CAUSE ANALYSIS
# ============================================================================

print("\n" + "=" * 80)
print("STEP 6: TOPIC MODELING - WHY DO CUSTOMERS CHURN?")
print("=" * 80)

# Isolate churned customer reviews
churned_reviews = df_filtered[df_filtered['Churn_Label'] == 1]['Cleaned_Review']
print(f"Analyzing {len(churned_reviews)} churned customer reviews...\n")

# Create TF-IDF for LDA (LDA works better with term frequency, but TF-IDF also works)
tfidf_lda = TfidfVectorizer(max_features=1000, ngram_range=(1, 1))
X_churned = tfidf_lda.fit_transform(churned_reviews)
feature_names_lda = tfidf_lda.get_feature_names_out()

# Apply LDA to find 3 hidden topics
print("Running Latent Dirichlet Allocation (LDA) to discover 3 churn topics...")
lda_model = LatentDirichletAllocation(
    n_components=3,  # 3 topics
    random_state=42,
    max_iter=20,
    learning_method='online'
)
lda_model.fit(X_churned)
print("✓ Topic modeling complete!\n")

# Display top 10 words for each topic
print("🔍 TOP 3 HIDDEN TOPICS CAUSING CHURN:")
print("=" * 80)
for topic_idx, topic in enumerate(lda_model.components_):
    top_indices = topic.argsort()[-10:][::-1]
    top_words = [feature_names_lda[i] for i in top_indices]
    print(f"\n📌 TOPIC {topic_idx + 1}: {', '.join(top_words[:10])}")

print("\n")

# ============================================================================
# 7. VISUALIZATION 1: FEATURE IMPORTANCE
# ============================================================================

print("=" * 80)
print("STEP 7: VISUALIZATION - FEATURE IMPORTANCE")
print("=" * 80)

# Get feature importances from Random Forest
importances = rf_model.feature_importances_
indices = np.argsort(importances)[-20:][::-1]  # Top 20

# Create DataFrame for plotting
top_features_df = pd.DataFrame({
    'Feature': [feature_names[i] for i in indices],
    'Importance': importances[indices]
})

# Plot feature importance
plt.figure(figsize=(12, 8))
colors = plt.cm.viridis(np.linspace(0.3, 0.9, 20))
plt.barh(range(20), top_features_df['Importance'], color=colors)
plt.yticks(range(20), top_features_df['Feature'], fontsize=10)
plt.xlabel('Feature Importance Score', fontsize=12, fontweight='bold')
plt.ylabel('Words/Phrases', fontsize=12, fontweight='bold')
plt.title('Top 20 Most Important Features Predicting Customer Churn', 
          fontsize=14, fontweight='bold', pad=20)
plt.gca().invert_yaxis()
plt.tight_layout()
print("✓ Feature Importance plot generated!")

plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
print("✓ Saved as 'feature_importance.png'")
plt.close()

# ============================================================================
# 8. VISUALIZATION 2: CONFUSION MATRIX
# ============================================================================

print("\n" + "=" * 80)
print("STEP 8: VISUALIZATION - CONFUSION MATRIX")
print("=" * 80)

plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlGn_r', cbar=True,
            xticklabels=['Retained', 'Churn'],
            yticklabels=['Retained', 'Churn'],
            annot_kws={"size": 16, "weight": "bold"},
            linewidths=2, linecolor='white')
plt.title('Confusion Matrix - Random Forest Churn Prediction', 
          fontsize=16, fontweight='bold', pad=20)
plt.ylabel('Actual', fontsize=14, fontweight='bold')
plt.xlabel('Predicted', fontsize=14, fontweight='bold')
plt.tight_layout()
print("✓ Confusion Matrix generated!")

plt.savefig('confusion_matrix_rf.png', dpi=300, bbox_inches='tight')
print("✓ Saved as 'confusion_matrix_rf.png'")
plt.close()

# ============================================================================
# 9. DASHBOARD EXPORT FOR POWER BI
# ============================================================================

print("\n" + "=" * 80)
print("STEP 9: POWER BI DASHBOARD EXPORT")
print("=" * 80)

# Get predictions for entire dataset
X_full = tfidf.transform(df_filtered['Cleaned_Review'])
full_predictions = rf_model.predict(X_full)
full_probabilities = rf_model.predict_proba(X_full)[:, 1]  # Probability of churn

# Create dashboard DataFrame
dashboard_df = pd.DataFrame({
    'Original_Review': df_filtered['review'].values,
    'Actual_Rating': df_filtered['rating'].values,
    'Predicted_Label': full_predictions,
    'Churn_Probability': full_probabilities,
    'Cleaned_Review': df_filtered['Cleaned_Review'].values
})

# Save to CSV
dashboard_df.to_csv('dashboard_data.csv', index=False)
print(f"✓ Dashboard data exported!")
print(f"  File: dashboard_data.csv")
print(f"  Rows: {len(dashboard_df)}")
print(f"  Columns: {len(dashboard_df.columns)}")
print(f"\nColumns included:")
for col in dashboard_df.columns:
    print(f"  • {col}")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("✅ ADVANCED CUSTOMER VOICE & CHURN ANALYSIS COMPLETE!")
print("=" * 80)

print("\n📊 Project Summary:")
print(f"  • Dataset: {len(df_filtered)} customer reviews (after filtering neutral)")
print(f"  • Features: {X.shape[1]} TF-IDF features (unigrams + bigrams)")
print(f"  • Model: Random Forest (balanced class weights)")
print(f"  • Accuracy: {accuracy*100:.2f}%")
print(f"  • Topics Discovered: 3 churn themes via LDA")

print("\n📁 Files Generated:")
print("  1. feature_importance.png - Top 20 predictive words/phrases")
print("  2. confusion_matrix_rf.png - Model performance heatmap")
print("  3. dashboard_data.csv - Power BI ready dataset")

print("\n💼 Business Insights:")
print("  ✓ Identified key words/phrases driving churn predictions")
print("  ✓ Discovered 3 hidden topics explaining why customers leave")
print("  ✓ Created actionable dashboard for stakeholder analysis")

print("\n🚀 Next Steps:")
print("  → Import dashboard_data.csv into Power BI")
print("  → Create visualizations using Churn_Probability scores")
print("  → Monitor high-risk customers (Predicted_Label = 1)")
print("  → Address root causes identified in LDA topics")

print("\n" + "=" * 80)
print("Portfolio-grade deliverable ready for presentation! 🎯")
print("=" * 80)
