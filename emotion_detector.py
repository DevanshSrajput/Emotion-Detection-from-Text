"""
Emotion Detection Model
A comprehensive text classification system for detecting emotions in user messages.
Author: DevanshSrajput
Date: 2025-06-26
"""

import pandas as pd
import numpy as np
import re
import string
from collections import Counter
import joblib
import os

# ML Libraries
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix, 
                           accuracy_score, precision_recall_fscore_support)

# NLP Libraries
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objects as go

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
except Exception as e:
    print(f"Warning: Could not download NLTK data: {e}")

class EmotionDetector:
    """
    A comprehensive emotion detection system that classifies text into emotional categories.
    """
    
    def __init__(self):
        """Initialize the EmotionDetector with preprocessing tools and models."""
        try:
            self.stop_words = set(stopwords.words('english'))
        except:
            # Fallback stopwords if NLTK data is not available
            self.stop_words = {'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'through', 'during', 'before', 'after', 'above', 'below', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once'}
        
        try:
            self.lemmatizer = WordNetLemmatizer()
        except:
            self.lemmatizer = None
            
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words='english'
        )
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.is_trained = False
        self.emotion_labels = ['joy', 'sadness', 'anger', 'fear', 'surprise', 'love']
        
    def create_sample_dataset(self):
        """
        Create a sample dataset for demonstration purposes.
        In a real project, you would load your own dataset.
        """
        sample_data = {
            'text': [
                # Joy/Happy
                "I'm so excited about my vacation!", "This is the best day ever!",
                "I love spending time with my family", "Congratulations on your success!",
                "I'm feeling fantastic today", "What a beautiful sunny day!",
                "I'm thrilled about the new opportunity", "This makes me so happy!",
                "Amazing news! I got the job!", "I feel wonderful and grateful!",
                
                # Sadness
                "I'm feeling really down today", "This is so disappointing",
                "I miss my old friends", "I feel lonely and sad",
                "This news broke my heart", "I'm going through a tough time",
                "Everything seems hopeless", "I feel empty inside",
                "I'm crying because of this", "This makes me feel depressed",
                
                # Anger
                "This is absolutely frustrating!", "I'm so angry right now",
                "This is completely unacceptable", "I hate when this happens",
                "This makes my blood boil", "I'm furious about this situation",
                "This is driving me crazy", "I can't stand this anymore",
                "I'm outraged by this behavior", "This is making me mad",
                
                # Fear
                "I'm really scared about tomorrow", "This worries me a lot",
                "I'm anxious about the results", "This situation terrifies me",
                "I'm afraid of what might happen", "This gives me nightmares",
                "I feel nervous and uneasy", "I'm dreading this meeting",
                "This is frightening me", "I'm panicking about this",
                
                # Surprise
                "I can't believe this happened!", "This is so unexpected!",
                "What a shocking revelation", "I'm amazed by this news",
                "This caught me completely off guard", "I never saw this coming",
                "This is absolutely mind-blowing", "What a pleasant surprise!",
                "I'm stunned by this information", "This is incredible!",
                
                # Love
                "I adore my partner so much", "This person means everything to me",
                "I'm deeply in love", "My heart is full of love",
                "I cherish every moment with you", "You complete me perfectly",
                "I love you more than words can say", "My heart belongs to you",
                "I'm passionate about this relationship", "You are my everything"
            ],
            'emotion': [
                'joy', 'joy', 'joy', 'joy', 'joy', 'joy', 'joy', 'joy', 'joy', 'joy',
                'sadness', 'sadness', 'sadness', 'sadness', 'sadness', 'sadness', 'sadness', 'sadness', 'sadness', 'sadness',
                'anger', 'anger', 'anger', 'anger', 'anger', 'anger', 'anger', 'anger', 'anger', 'anger',
                'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear', 'fear',
                'surprise', 'surprise', 'surprise', 'surprise', 'surprise', 'surprise', 'surprise', 'surprise', 'surprise', 'surprise',
                'love', 'love', 'love', 'love', 'love', 'love', 'love', 'love', 'love', 'love'
            ]
        }
        return pd.DataFrame(sample_data)
    
    def preprocess_text(self, text):
        """
        Clean and preprocess text data.
        
        Args:
            text (str): Raw text to preprocess
            
        Returns:
            str: Cleaned and preprocessed text
        """
        if not isinstance(text, str):
            return ""
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove user mentions and hashtags
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Tokenize
        try:
            tokens = word_tokenize(text)
        except:
            # Fallback tokenization
            tokens = text.split()
        
        # Remove stopwords and lemmatize
        processed_tokens = []
        for token in tokens:
            if token not in self.stop_words and len(token) > 2:
                if self.lemmatizer:
                    try:
                        token = self.lemmatizer.lemmatize(token)
                    except:
                        pass  # Keep original token if lemmatization fails
                processed_tokens.append(token)
        
        return ' '.join(processed_tokens)
    
    def prepare_data(self, df):
        """
        Prepare data for training by preprocessing text and encoding labels.
        
        Args:
            df (pd.DataFrame): DataFrame with 'text' and 'emotion' columns
            
        Returns:
            tuple: Preprocessed features and labels
        """
        if df is None or df.empty:
            raise ValueError("DataFrame cannot be None or empty")
        if 'text' not in df.columns or 'emotion' not in df.columns:
            raise ValueError("DataFrame must contain 'text' and 'emotion' columns")
            
        # Preprocess text
        df = df.copy()
        df['cleaned_text'] = df['text'].apply(self.preprocess_text)
        
        # Remove empty texts
        df = df[df['cleaned_text'].str.len() > 0]
        
        if df.empty:
            raise ValueError("No valid text data found after preprocessing")
        
        return df['cleaned_text'], df['emotion']
    
    def train_models(self, X, y):
        """
        Train multiple classification models and compare their performance.
        
        Args:
            X: Feature vectors
            y: Target labels
        """
        if len(X) == 0 or len(y) == 0:
            raise ValueError("Training data cannot be empty")
            
        print(f"Training with {len(X)} samples...")
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Vectorize the text
        print("Vectorizing text data...")
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)
        
        # Define models to train
        models_to_train = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Naive Bayes': MultinomialNB(),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
        }
        
        # Train and evaluate models
        results = {}
        best_f1 = 0
        
        for name, model in models_to_train.items():
            print(f"\nTraining {name}...")
            
            try:
                # Train the model
                model.fit(X_train_vec, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test_vec)
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
                
                # Store results
                results[name] = {
                    'model': model,
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'y_test': y_test,
                    'y_pred': y_pred
                }
                
                # Store model
                self.models[name] = model
                
                # Check if this is the best model
                if f1 > best_f1:
                    best_f1 = f1
                    self.best_model = model
                    self.best_model_name = name
                
                print(f"Accuracy: {accuracy:.4f}")
                print(f"Precision: {precision:.4f}")
                print(f"Recall: {recall:.4f}")
                print(f"F1-Score: {f1:.4f}")
                
            except Exception as e:
                print(f"Error training {name}: {e}")
                continue
        
        if self.best_model is None:
            raise RuntimeError("No models were successfully trained")
        
        # Mark as trained
        self.is_trained = True
        
        print(f"\nBest model: {self.best_model_name} with F1-Score: {best_f1:.4f}")
        
        return results, X_test, y_test
    
    def predict_emotion(self, text):
        """
        Predict emotion for a given text.
        
        Args:
            text (str): Input text
            
        Returns:
            tuple: Predicted emotion and confidence scores
        """
        if not self.is_trained or self.best_model is None:
            raise ValueError("Model not trained yet. Please train the model first.")
        
        if not isinstance(text, str) or len(text.strip()) == 0:
            raise ValueError("Input text must be a non-empty string")
        
        # Preprocess text
        cleaned_text = self.preprocess_text(text)
        
        if len(cleaned_text.strip()) == 0:
            # If preprocessing results in empty text, use simple fallback
            cleaned_text = text.lower()
        
        # Vectorize
        text_vec = self.vectorizer.transform([cleaned_text])
        
        # Predict
        prediction = self.best_model.predict(text_vec)[0]
        
        # Get probabilities
        if hasattr(self.best_model, 'predict_proba'):
            probabilities = self.best_model.predict_proba(text_vec)[0]
            emotion_scores = dict(zip(self.best_model.classes_, probabilities))
        else:
            # Fallback for models without predict_proba
            emotion_scores = {emotion: 0.0 for emotion in self.emotion_labels}
            emotion_scores[prediction] = 1.0
        
        return prediction, emotion_scores
    
    def save_model(self, filepath='emotion_model.pkl'):
        """Save the trained model and vectorizer."""
        if not self.is_trained or self.best_model is None:
            raise ValueError("No trained model to save")
            
        model_data = {
            'vectorizer': self.vectorizer,
            'best_model': self.best_model,
            'best_model_name': self.best_model_name,
            'models': self.models,
            'is_trained': self.is_trained,
            'emotion_labels': self.emotion_labels
        }
        
        try:
            joblib.dump(model_data, filepath)
            print(f"Model saved to {filepath}")
        except Exception as e:
            print(f"Error saving model: {e}")
            raise
    
    def load_model(self, filepath='emotion_model.pkl'):
        """Load a trained model and vectorizer."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file {filepath} not found")
            
        try:
            model_data = joblib.load(filepath)
            
            # Load all components
            self.vectorizer = model_data['vectorizer']
            self.best_model = model_data['best_model']
            self.best_model_name = model_data.get('best_model_name', 'Unknown')
            self.models = model_data.get('models', {})
            self.is_trained = model_data.get('is_trained', True)
            self.emotion_labels = model_data.get('emotion_labels', self.emotion_labels)
            
            print(f"Model loaded from {filepath}")
            print(f"Best model: {self.best_model_name}")
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

def main():
    """Main function to demonstrate the emotion detection system."""
    try:
        # Initialize detector
        detector = EmotionDetector()
        
        # Create sample dataset
        print("Creating sample dataset...")
        df = detector.create_sample_dataset()
        print(f"Dataset created with {len(df)} samples")
        
        # Prepare data
        print("\nPreparing data...")
        X, y = detector.prepare_data(df)
        
        # Train models
        print("\nTraining models...")
        results, X_test, y_test = detector.train_models(X, y)
        
        # Save model
        print("\nSaving model...")
        detector.save_model()
        
        # Test predictions
        print("\n" + "="*50)
        print("TESTING PREDICTIONS")
        print("="*50)
        
        test_texts = [
            "I'm absolutely thrilled about this opportunity!",
            "This makes me feel so sad and lonely",
            "I'm really angry about what happened",
            "I'm terrified of what might happen next",
            "This is such a shocking surprise!",
            "I love you with all my heart"
        ]
        
        for text in test_texts:
            try:
                emotion, scores = detector.predict_emotion(text)
                print(f"\nText: '{text}'")
                print(f"Predicted Emotion: {emotion}")
                print(f"Confidence: {scores[emotion]:.3f}")
                
                # Show top 3 emotions
                top_emotions = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]
                print("Top 3 predictions:")
                for emo, score in top_emotions:
                    print(f"  {emo}: {score:.3f}")
                    
            except Exception as e:
                print(f"Error predicting emotion for '{text}': {e}")
        
        print("\n" + "="*50)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("="*50)
        
    except Exception as e:
        print(f"Error in main function: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()