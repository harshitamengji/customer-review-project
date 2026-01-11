"""
Customer Churn & Sentiment Prediction System
Author: Expert Data Scientist
Description: End-to-end ML pipeline for predicting customer churn from bank reviews
"""

import pandas as pd
import numpy as np
import re
import warnings
warnings.filterwarnings('ignore')

# NLP and ML libraries
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Download required NLTK data
print("Downloading NLTK data...")
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
except:
    pass
print("✓ NLTK data ready\n")

# Set style for professional plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ============================================================================
# 1. DATA LOADING & PREPARATION
# ============================================================================

print("=" * 70)
print("STEP 1: DATA LOADING & PREPARATION")
print("=" * 70)

# Load the dataset
df = pd.read_csv('bank_reviews3.csv')
print(f"✓ Dataset loaded successfully!")
print(f"  Shape: {df.shape[0]} rows × {df.shape[1]} columns\n")

# Create Churn_Label column
# Rating 1.0 or 2.0 → Churn (1)
# Rating 3.0, 4.0, or 5.0 → Not Churn (0)
df['Churn_Label'] = df['rating'].apply(lambda x: 1 if x in [1.0, 2.0] else 0)

# Display class distribution
print("Class Distribution:")
print(df['Churn_Label'].value_counts())
print("\nPercentage Distribution:")
print(df['Churn_Label'].value_counts(normalize=True) * 100)
print()

# ============================================================================
# 2. NLP PREPROCESSING
# ============================================================================

print("=" * 70)
print("STEP 2: NLP PREPROCESSING")
print("=" * 70)

# Initialize stemmer
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    """
    Clean and preprocess text data
    
    Steps:
    1. Convert to lowercase
    2. Remove special characters and punctuation
    3. Remove stopwords
    4. Apply stemming
    
    Args:
        text (str): Raw text input
        
    Returns:
        str: Cleaned and processed text
    """
    if pd.isna(text):
        return ""
    
    # Convert to lowercase
    text = str(text).lower()
    
    # Remove special characters and punctuation (keep only letters and spaces)
    text = re.sub(r'[^a-z\s]', ' ', text)
    
    # Tokenize and remove stopwords
    words = text.split()
    words = [word for word in words if word not in stop_words and len(word) > 2]
    
    # Apply stemming
    words = [stemmer.stem(word) for word in words]
    
    # Join back into string
    return ' '.join(words)

# Apply cleaning function
print("Cleaning reviews... This may take a moment.")
df['Cleaned_Review'] = df['review'].apply(clean_text)
print(f"✓ Cleaned {len(df)} reviews successfully!\n")

# Display sample
print("Sample Cleaned Reviews:")
print("-" * 70)
for i in range(2):
    print(f"Original: {df['review'].iloc[i][:100]}...")
    print(f"Cleaned:  {df['Cleaned_Review'].iloc[i][:100]}...")
    print()

# ============================================================================
# 3. FEATURE ENGINEERING (TF-IDF)
# ============================================================================

print("=" * 70)
print("STEP 3: FEATURE ENGINEERING (TF-IDF)")
print("=" * 70)

# Remove rows with empty cleaned reviews
df = df[df['Cleaned_Review'].str.strip() != '']
print(f"✓ Removed empty reviews. Remaining: {len(df)} rows")

# TF-IDF Vectorization
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X = tfidf.fit_transform(df['Cleaned_Review'])
y = df['Churn_Label']

print(f"✓ TF-IDF vectorization complete!")
print(f"  Feature matrix shape: {X.shape}")
print(f"  Number of features: {X.shape[1]}")
print(f"  Target variable shape: {y.shape}\n")

# ============================================================================
# 4. MODEL TRAINING
# ============================================================================

print("=" * 70)
print("STEP 4: MODEL TRAINING")
print("=" * 70)

# Train-test split (80-20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train set: {X_train.shape[0]} samples")
print(f"Test set:  {X_test.shape[0]} samples\n")

# Train Logistic Regression model
print("Training Logistic Regression model...")
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)
print("✓ Model training complete!\n")

# ============================================================================
# 5. EVALUATION & VISUALIZATION
# ============================================================================

print("=" * 70)
print("STEP 5: MODEL EVALUATION")
print("=" * 70)

# Make predictions
y_pred = model.predict(X_test)

# Accuracy Score
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy Score: {accuracy:.4f} ({accuracy*100:.2f}%)\n")

# Classification Report
print("Classification Report:")
print("=" * 70)
print(classification_report(y_test, y_pred, target_names=['Not Churn (0)', 'Churn (1)']))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

# ============================================================================
# VISUALIZATION 1: CONFUSION MATRIX
# ============================================================================

print("\n" + "=" * 70)
print("VISUALIZATION 1: CONFUSION MATRIX")
print("=" * 70)

plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
            xticklabels=['Not Churn', 'Churn'],
            yticklabels=['Not Churn', 'Churn'],
            annot_kws={"size": 16})
plt.title('Confusion Matrix - Customer Churn Prediction', fontsize=16, fontweight='bold', pad=20)
plt.ylabel('Actual', fontsize=14, fontweight='bold')
plt.xlabel('Predicted', fontsize=14, fontweight='bold')
plt.tight_layout()
print("✓ Confusion Matrix generated!")

# Save the figure
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
print("✓ Saved as 'confusion_matrix.png'")
plt.close()

# ============================================================================
# VISUALIZATION 2: WORD CLOUD FOR CHURNED CUSTOMERS
# ============================================================================

print("\n" + "=" * 70)
print("VISUALIZATION 2: WORD CLOUD FOR CHURNED CUSTOMERS")
print("=" * 70)

# Get all reviews from churned customers
churned_reviews = df[df['Churn_Label'] == 1]['Cleaned_Review']
churned_text = ' '.join(churned_reviews)

print(f"Analyzing {len(churned_reviews)} churned customer reviews...")

# Generate word cloud
wordcloud = WordCloud(
    width=1200, 
    height=600,
    background_color='white',
    colormap='Reds',
    max_words=100,
    relative_scaling=0.5,
    min_font_size=10
).generate(churned_text)

# Plot word cloud
plt.figure(figsize=(16, 8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud: What Angry Customers Are Talking About (Churn = 1)', 
          fontsize=18, fontweight='bold', pad=20)
plt.tight_layout()
print("✓ Word Cloud generated!")

# Save the figure
plt.savefig('wordcloud_churned.png', dpi=300, bbox_inches='tight')
print("✓ Saved as 'wordcloud_churned.png'")
plt.close()

# ============================================================================
# 6. PREDICTION SYSTEM
# ============================================================================

print("\n" + "=" * 70)
print("STEP 6: PREDICTION SYSTEM")
print("=" * 70)

def predict_churn(text):
    """
    Predict if a customer is likely to churn based on their review
    
    Args:
        text (str): Raw customer review text
        
    Returns:
        str: Prediction result with probability
    """
    # Clean the text
    cleaned = clean_text(text)
    
    # Vectorize
    vectorized = tfidf.transform([cleaned])
    
    # Predict
    prediction = model.predict(vectorized)[0]
    probability = model.predict_proba(vectorized)[0]
    
    # Format result
    if prediction == 1:
        result = f"🔴 HIGH RISK (Churn) - Probability: {probability[1]:.2%}"
    else:
        result = f"🟢 LOW RISK (Retained) - Probability: {probability[0]:.2%}"
    
    return result

# ============================================================================
# TEST PREDICTIONS
# ============================================================================

print("\nTesting Prediction System:")
print("=" * 70)

test_cases = [
    "The app is terrible and the fees are too high.",
    "I love using this bank, great service."
]

for i, test_text in enumerate(test_cases, 1):
    print(f"\nTest Case {i}:")
    print(f"Input: \"{test_text}\"")
    print(f"Prediction: {predict_churn(test_text)}")

print("\n" + "=" * 70)
print("✅ CUSTOMER CHURN PREDICTION PROJECT COMPLETE!")
print("=" * 70)
print("\nProject Summary:")
print(f"  • Dataset: {len(df)} customer reviews")
print(f"  • Features: {X.shape[1]} TF-IDF features")
print(f"  • Model Accuracy: {accuracy*100:.2f}%")
print(f"  • Visualizations: 2 (Confusion Matrix + Word Cloud)")
print("\nFiles Generated:")
print("  1. confusion_matrix.png")
print("  2. wordcloud_churned.png")
print("=" * 70)
