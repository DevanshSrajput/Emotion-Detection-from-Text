"""
Streamlit Web Application for Emotion Detection
Author: DevanshSrajput
Date: 2025-06-26
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
import sys

# Add the current directory to the path to import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from emotion_detector import EmotionDetector
from visualizations import EmotionVisualizer

# Page configuration
st.set_page_config(
    page_title="Emotion Detection App",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .emotion-card {
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    .prediction-text {
        font-size: 1.5rem;
        font-weight: bold;
        text-align: center;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize detector in session state for persistence
def init_detector():
    """Initialize detector in session state if not exists."""
    if 'detector' not in st.session_state:
        st.session_state.detector = EmotionDetector()
    return st.session_state.detector

# Safe rerun function that works across Streamlit versions
def safe_rerun():
    """Safely trigger a rerun based on Streamlit version."""
    try:
        # Try the new method first (Streamlit >= 1.27.0)
        st.rerun()
    except AttributeError:
        # If rerun doesn't exist, just show a message
        st.info("Please refresh the page to see updated status.")

class EmotionApp:
    """Main application class for the Emotion Detection Streamlit app."""
    
    def __init__(self):
        """Initialize the application."""
        # Use session state detector for persistence
        self.detector = init_detector()
        self.visualizer = EmotionVisualizer()
        self.emotion_emojis = {
            'joy': '😊',
            'sadness': '😢', 
            'anger': '😠',
            'fear': '😨',
            'surprise': '😲',
            'love': '❤️'
        }
        
        # Initialize session state
        if 'model_trained' not in st.session_state:
            st.session_state.model_trained = False
        if 'training_in_progress' not in st.session_state:
            st.session_state.training_in_progress = False
        if 'model_ready' not in st.session_state:
            st.session_state.model_ready = False
    
    def check_model_status(self):
        """Check if the detector has a trained model."""
        try:
            has_trained_model = (
                hasattr(self.detector, 'is_trained') and 
                self.detector.is_trained and 
                self.detector.best_model is not None and
                hasattr(self.detector, 'vectorizer') and
                self.detector.vectorizer is not None
            )
            
            # Update session state
            st.session_state.model_ready = has_trained_model
            
            return has_trained_model
        except Exception as e:
            st.session_state.model_ready = False
            return False
    
    def load_existing_model(self):
        """Try to load existing model from file."""
        model_path = 'emotion_model.pkl'
        
        if os.path.exists(model_path):
            try:
                success = self.detector.load_model(model_path)
                if success and self.check_model_status():
                    st.session_state.model_trained = True
                    st.session_state.model_ready = True
                    return True
                else:
                    return False
            except Exception as e:
                st.error(f"Error loading existing model: {e}")
                return False
        return False
    
    def train_new_model(self):
        """Train a new model with sample data."""
        st.session_state.training_in_progress = True
        
        try:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Step 1: Create dataset
            status_text.text("Creating sample dataset...")
            progress_bar.progress(10)
            df = self.detector.create_sample_dataset()
            
            # Step 2: Prepare data
            status_text.text("Preparing data...")
            progress_bar.progress(30)
            X, y = self.detector.prepare_data(df)
            
            # Step 3: Train models
            status_text.text("Training models... This may take a moment.")
            progress_bar.progress(50)
            results, X_test, y_test = self.detector.train_models(X, y)
            
            # Step 4: Save model
            status_text.text("Saving model...")
            progress_bar.progress(80)
            self.detector.save_model()
            
            # Step 5: Complete
            progress_bar.progress(100)
            status_text.text("Training completed!")
            
            # Update session state
            st.session_state.model_trained = True
            st.session_state.model_ready = True
            st.session_state.training_in_progress = False
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
            
            # Show success message
            st.success("✅ Model trained successfully!")
            st.balloons()
            
            # Display training results
            st.subheader("Training Results")
            st.info(f"🏆 Best Model: {self.detector.best_model_name}")
            
            # Show performance metrics
            if self.detector.best_model_name in results:
                best_results = results[self.detector.best_model_name]
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Accuracy", f"{best_results['accuracy']:.3f}")
                with col2:
                    st.metric("Precision", f"{best_results['precision']:.3f}")
                with col3:
                    st.metric("F1-Score", f"{best_results['f1']:.3f}")
            
            # Test the model immediately
            st.subheader("Quick Test")
            test_text = "I'm so happy and excited today!"
            try:
                emotion, scores = self.detector.predict_emotion(test_text)
                st.success(f"Test prediction for '{test_text}': **{emotion}** (confidence: {scores[emotion]:.3f})")
            except Exception as e:
                st.error(f"Test prediction failed: {e}")
            
            return True
                
        except Exception as e:
            st.session_state.training_in_progress = False
            st.session_state.model_trained = False
            st.session_state.model_ready = False
            st.error(f"❌ Error training model: {str(e)}")
            return False
    
    def predict_emotion_with_confidence(self, text):
        """Predict emotion and return formatted results."""
        try:
            if not self.check_model_status():
                return None, "Model not ready. Please train the model first."
            
            emotion, scores = self.detector.predict_emotion(text)
            return emotion, scores
            
        except Exception as e:
            return None, f"Error during prediction: {str(e)}"
    
    def create_confidence_chart(self, emotion_scores):
        """Create a confidence score chart."""
        if not emotion_scores:
            return None
            
        emotions = list(emotion_scores.keys())
        scores = list(emotion_scores.values())
        
        # Sort by score
        sorted_data = sorted(zip(emotions, scores), key=lambda x: x[1], reverse=True)
        emotions, scores = zip(*sorted_data)
        
        # Create color map
        colors = [self.visualizer.emotion_colors.get(emotion, '#333333') for emotion in emotions]
        
        fig = go.Figure(data=[
            go.Bar(
                x=list(emotions),
                y=list(scores),
                text=[f'{score:.3f}' for score in scores],
                textposition='auto',
                marker_color=colors,
                hovertemplate='<b>%{x}</b><br>Confidence: %{y:.3f}<extra></extra>'
            )
        ])
        
        fig.update_layout(
            title="Emotion Confidence Scores",
            xaxis_title="Emotions",
            yaxis_title="Confidence Score",
            yaxis=dict(range=[0, 1]),
            template="plotly_white",
            height=400
        )
        
        return fig
    
    def main_interface(self):
        """Create the main application interface."""
        # Header
        st.markdown('<h1 class="main-header">🎭 Emotion Detection App</h1>', unsafe_allow_html=True)
        st.markdown("### Analyze emotions in text using AI-powered sentiment analysis")
        
        # Sidebar
        st.sidebar.header("🔧 Controls")
        st.sidebar.subheader("Model Management")
        
        # Check if model exists and load it
        if not st.session_state.model_trained and not st.session_state.training_in_progress:
            if self.load_existing_model():
                st.sidebar.success("✅ Existing model loaded!")
            else:
                st.sidebar.warning("⚠️ No trained model found.")
        
        # Model status display
        model_status = self.check_model_status()
        
        if st.session_state.training_in_progress:
            st.sidebar.info("🔄 Training in progress...")
        elif model_status:
            st.sidebar.success("✅ Model ready for predictions!")
            if hasattr(self.detector, 'best_model_name') and self.detector.best_model_name:
                st.sidebar.info(f"Current model: {self.detector.best_model_name}")
        else:
            st.sidebar.warning("⚠️ Model not ready. Please train a new model.")
        
        # Train button
        if st.sidebar.button("🚀 Train New Model", disabled=st.session_state.training_in_progress):
            st.session_state.model_trained = False
            st.session_state.model_ready = False
            # Train the model
            if self.train_new_model():
                # Instead of rerun, just update the interface
                st.success("Model training completed! You can now make predictions.")
        
        # Debug info (expandable)
        with st.sidebar.expander("🔍 Debug Info"):
            st.write(f"Model trained: {st.session_state.model_trained}")
            st.write(f"Model ready: {st.session_state.model_ready}")
            st.write(f"Training in progress: {st.session_state.training_in_progress}")
            st.write(f"Detector is_trained: {getattr(self.detector, 'is_trained', False)}")
            st.write(f"Best model exists: {self.detector.best_model is not None}")
            st.write(f"Vectorizer exists: {hasattr(self.detector, 'vectorizer') and self.detector.vectorizer is not None}")
            if hasattr(self.detector, 'best_model_name'):
                st.write(f"Best model name: {self.detector.best_model_name}")
        
        # Main content area
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📝 Text Input")
            
            # Input method selection
            input_method = st.radio(
                "Choose input method:",
                ["Type your text", "Select from examples"]
            )
            
            if input_method == "Type your text":
                user_text = st.text_area(
                    "Enter your text here:",
                    height=100,
                    placeholder="Type something to analyze its emotion..."
                )
            else:
                example_texts = [
                    "I love you with all my heart",
                    "I'm so excited about this opportunity!",
                    "This is really disappointing and sad",
                    "I'm absolutely furious about this situation",
                    "I'm terrified of what might happen",
                    "What a shocking surprise this is!"
                ]
                
                selected_example = st.selectbox(
                    "Select an example:",
                    example_texts
                )
                user_text = selected_example
            
            # Analyze button
            if st.button("🔍 Analyze Emotion", type="primary"):
                if user_text and user_text.strip():
                    # Re-check model status before prediction
                    if self.check_model_status():
                        with st.spinner("Analyzing emotion..."):
                            emotion, scores = self.predict_emotion_with_confidence(user_text)
                            
                            if emotion and isinstance(scores, dict):
                                # Display prediction
                                emoji = self.emotion_emojis.get(emotion, "🎭")
                                st.markdown(f"""
                                <div class="prediction-text" style="background-color: #e8f4fd; color: #1f77b4;">
                                    Predicted Emotion: {emoji} <strong>{emotion.upper()}</strong>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Display confidence chart
                                fig = self.create_confidence_chart(scores)
                                if fig:
                                    st.plotly_chart(fig, use_container_width=True)
                                
                                # Show detailed scores
                                st.subheader("Detailed Confidence Scores")
                                for emo, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
                                    emoji = self.emotion_emojis.get(emo, "🎭")
                                    st.write(f"{emoji} **{emo.capitalize()}**")
                                    st.progress(score, text=f"{score:.3f}")
                            else:
                                st.error(f"Prediction error: {scores}")
                    else:
                        st.error("⚠️ Model not ready. Please train the model first by clicking 'Train New Model' in the sidebar.")
                else:
                    st.warning("⚠️ Please enter some text to analyze.")
        
        with col2:
            st.subheader("ℹ️ About")
            
            st.markdown("""
            **Emotion Detection Model**
            
            This AI model can detect 6 different emotions:
            
            • 😊 Joy/Happiness
            • 😢 Sadness  
            • 😠 Anger
            • 😨 Fear
            • 😲 Surprise
            • ❤️ Love
            
            The model uses advanced NLP techniques including:
            
            • Text preprocessing
            • TF-IDF vectorization
            • Machine learning classification
            """)
            
            # Show training instructions if model not ready
            if not model_status:
                st.warning("""
                **Getting Started:**
                
                1. Click 'Train New Model' in the sidebar
                2. Wait for training to complete
                3. Start analyzing emotions!
                """)
    
    def analytics_page(self):
        """Create analytics and visualization page."""
        st.header("📊 Analytics Dashboard")
        
        if not self.check_model_status():
            st.warning("⚠️ Model not ready. Please train a model first from the Home page.")
            
            if st.button("🚀 Train Model Now"):
                if self.train_new_model():
                    st.success("Training completed! Model is now ready.")
            return
        
        # Sample dataset analysis
        st.subheader("Dataset Analysis")
        
        try:
            # Create sample dataset for analysis
            df = self.detector.create_sample_dataset()
            
            # Display dataset info
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Samples", len(df))
            with col2:
                st.metric("Unique Emotions", df['emotion'].nunique())
            with col3:
                avg_length = df['text'].str.len().mean()
                st.metric("Avg Text Length", f"{avg_length:.1f}")
            
            # Emotion distribution
            st.subheader("Emotion Distribution")
            
            emotion_counts = df['emotion'].value_counts()
            
            # Create pie chart
            fig = px.pie(
                values=emotion_counts.values,
                names=emotion_counts.index,
                title="Distribution of Emotions in Dataset",
                color_discrete_map=self.visualizer.emotion_colors
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Show sample texts for each emotion
            st.subheader("Sample Texts by Emotion")
            
            selected_emotion = st.selectbox(
                "Select emotion to view sample texts:",
                df['emotion'].unique()
            )
            
            sample_texts = df[df['emotion'] == selected_emotion]['text'].head(3)
            
            for i, text in enumerate(sample_texts, 1):
                st.write(f"**Example {i}:** {text}")
                
        except Exception as e:
            st.error(f"Error loading analytics: {e}")

def main():
    """Main function to run the Streamlit app."""
    app = EmotionApp()
    
    # Navigation
    st.sidebar.title("🧭 Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["🏠 Home", "📊 Analytics"]
    )
    
    if page == "🏠 Home":
        app.main_interface()
    elif page == "📊 Analytics":
        app.analytics_page()
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Created by:** Devansh")

if __name__ == "__main__":
    main()