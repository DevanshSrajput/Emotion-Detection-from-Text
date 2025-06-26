"""
Visualization utilities for the Emotion Detection project.
Author: DevanshSrajput
Date: 2025-06-26
"""

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from wordcloud import WordCloud
from sklearn.metrics import confusion_matrix
from collections import Counter

class EmotionVisualizer:
    """Class for creating various visualizations for emotion detection analysis."""
    
    def __init__(self):
        """Initialize the visualizer with styling preferences."""
        # Set style - fix deprecated seaborn style
        plt.style.use('default')
        sns.set_theme()  # Updated seaborn styling
        sns.set_palette("husl")
        
        # Color mapping for emotions
        self.emotion_colors = {
            'joy': '#FFD700',
            'sadness': '#4169E1', 
            'anger': '#DC143C',
            'fear': '#8A2BE2',
            'surprise': '#FF8C00',
            'love': '#FF1493'
        }
        
        # Emoji mapping
        self.emotion_emojis = {
            'joy': '😊',
            'sadness': '😢',
            'anger': '😠',
            'fear': '😨',
            'surprise': '😲',
            'love': '❤️'
        }
    
    def plot_emotion_distribution(self, df, save_path=None):
        """
        Plot the distribution of emotions in the dataset.
        
        Args:
            df (pd.DataFrame): DataFrame with emotion labels
            save_path (str): Optional path to save the plot
        """
        # Validate input
        if df is None or df.empty:
            raise ValueError("DataFrame cannot be None or empty")
        if 'emotion' not in df.columns:
            raise ValueError("DataFrame must contain 'emotion' column")
            
        # Count emotions
        emotion_counts = df['emotion'].value_counts()
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Bar plot
        bars = ax1.bar(emotion_counts.index, emotion_counts.values, 
                      color=[self.emotion_colors.get(emotion, '#333333') 
                            for emotion in emotion_counts.index])
        ax1.set_title('Emotion Distribution in Dataset', fontsize=16, fontweight='bold')
        ax1.set_xlabel('Emotions', fontsize=12)
        ax1.set_ylabel('Count', fontsize=12)
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom')
        
        # Pie chart
        colors = [self.emotion_colors.get(emotion, '#333333') for emotion in emotion_counts.index]
        wedges, texts, autotexts = ax2.pie(emotion_counts.values, labels=emotion_counts.index, 
                                          autopct='%1.1f%%', colors=colors, startangle=90)
        ax2.set_title('Emotion Distribution (Percentage)', fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_confusion_matrix(self, y_true, y_pred, model_name="Model", save_path=None):
        """
        Plot confusion matrix for model evaluation.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            model_name (str): Name of the model
            save_path (str): Optional path to save the plot
        """
        # Validate inputs
        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have the same length")
        if len(y_true) == 0:
            raise ValueError("Input arrays cannot be empty")
            
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Get unique labels
        labels = sorted(list(set(y_true) | set(y_pred)))
        
        # Create heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=labels, yticklabels=labels,
                   square=True, cbar_kws={'label': 'Count'})
        
        plt.title(f'Confusion Matrix - {model_name}', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Labels', fontsize=12)
        plt.ylabel('True Labels', fontsize=12)
        plt.xticks(rotation=45)
        plt.yticks(rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_model_comparison(self, results, save_path=None):
        """
        Compare performance of different models.
        
        Args:
            results (dict): Dictionary containing model results
            save_path (str): Optional path to save the plot
        """
        # Validate input
        if not results:
            raise ValueError("Results dictionary cannot be empty")
            
        # Prepare data
        models = list(results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        
        # Validate that all models have required metrics
        for model in models:
            for metric in metrics:
                if metric not in results[model]:
                    raise KeyError(f"Metric '{metric}' not found for model '{model}'")
        
        # Create DataFrame for plotting
        data = []
        for model in models:
            for metric in metrics:
                data.append({
                    'Model': model,
                    'Metric': metric.capitalize(),
                    'Score': results[model][metric]
                })
        
        df_results = pd.DataFrame(data)
        
        # Create grouped bar plot
        plt.figure(figsize=(12, 8))
        sns.barplot(data=df_results, x='Model', y='Score', hue='Metric')
        
        plt.title('Model Performance Comparison', fontsize=16, fontweight='bold')
        plt.xlabel('Models', fontsize=12)
        plt.ylabel('Score', fontsize=12)
        plt.xticks(rotation=45)
        plt.legend(title='Metrics', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.ylim(0, 1)
        
        # Add value labels
        ax = plt.gca()
        for p in ax.patches:
            height = p.get_height()
            if not np.isnan(height):  # Check for NaN values
                ax.annotate(f'{height:.3f}', 
                           (p.get_x() + p.get_width()/2., height),
                           ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_emotion_wordclouds(self, df, save_path=None):
        """
        Create word clouds for each emotion category.
        
        Args:
            df (pd.DataFrame): DataFrame with text and emotion columns
            save_path (str): Optional path to save the plot
        """
        # Validate input
        if df is None or df.empty:
            raise ValueError("DataFrame cannot be None or empty")
        if 'emotion' not in df.columns or 'text' not in df.columns:
            raise ValueError("DataFrame must contain 'emotion' and 'text' columns")
            
        emotions = df['emotion'].unique()
        n_emotions = len(emotions)
        
        if n_emotions == 0:
            raise ValueError("No emotions found in dataset")
        
        # Calculate grid dimensions
        n_cols = min(3, n_emotions)
        n_rows = (n_emotions + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        
        # Handle single subplot case
        if n_emotions == 1:
            axes = np.array([axes])
        elif n_rows == 1 and n_cols > 1:
            axes = axes.reshape(1, -1)
        elif n_rows > 1 and n_cols == 1:
            axes = axes.reshape(-1, 1)
        
        for idx, emotion in enumerate(emotions):
            row = idx // n_cols
            col = idx % n_cols
            
            if n_rows == 1 and n_cols == 1:
                ax = axes[0]
            elif n_rows == 1:
                ax = axes[0, col]
            elif n_cols == 1:
                ax = axes[row, 0]
            else:
                ax = axes[row, col]
            
            # Get text for this emotion
            emotion_texts = df[df['emotion'] == emotion]['text'].dropna().values
            if len(emotion_texts) == 0:
                ax.text(0.5, 0.5, f'No text data\nfor {emotion}', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{emotion.capitalize()} {self.emotion_emojis.get(emotion, "")}', 
                            fontsize=14, fontweight='bold')
                ax.axis('off')
                continue
                
            emotion_text = ' '.join(emotion_texts)
            
            # Skip if text is too short
            if len(emotion_text.strip()) < 10:
                ax.text(0.5, 0.5, f'Insufficient text\nfor {emotion}', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{emotion.capitalize()} {self.emotion_emojis.get(emotion, "")}', 
                            fontsize=14, fontweight='bold')
                ax.axis('off')
                continue
            
            try:
                # Create word cloud
                wordcloud = WordCloud(
                    width=400, height=300,
                    background_color='white',
                    colormap='viridis',
                    max_words=100,
                    random_state=42
                ).generate(emotion_text)
                
                # Plot
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.set_title(f'{emotion.capitalize()} {self.emotion_emojis.get(emotion, "")}', 
                            fontsize=14, fontweight='bold')
                ax.axis('off')
            except Exception as e:
                ax.text(0.5, 0.5, f'Error creating\nwordcloud for {emotion}', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{emotion.capitalize()} {self.emotion_emojis.get(emotion, "")}', 
                            fontsize=14, fontweight='bold')
                ax.axis('off')
        
        # Hide unused subplots
        for idx in range(n_emotions, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            if n_rows == 1 and n_cols > 1:
                axes[0, col].axis('off')
            elif n_cols == 1 and n_rows > 1:
                axes[row, 0].axis('off')
            else:
                axes[row, col].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_interactive_emotion_scores(self, emotion_scores, title="Emotion Prediction Scores"):
        """
        Create an interactive plot of emotion scores using Plotly.
        
        Args:
            emotion_scores (dict): Dictionary of emotion scores
            title (str): Title for the plot
        """
        if not emotion_scores:
            raise ValueError("Emotion scores dictionary cannot be empty")
            
        emotions = list(emotion_scores.keys())
        scores = list(emotion_scores.values())
        colors = [self.emotion_colors.get(emotion, '#333333') for emotion in emotions]
        
        # Create bar plot
        fig = go.Figure(data=[
            go.Bar(
                x=emotions,
                y=scores,
                marker_color=colors,
                text=[f'{score:.3f}' for score in scores],
                textposition='auto',
            )
        ])
        
        fig.update_layout(
            title=title,
            xaxis_title="Emotions",
            yaxis_title="Confidence Score",
            yaxis=dict(range=[0, 1]),
            template="plotly_white",
            font=dict(size=12)
        )
        
        return fig
    
    def create_comprehensive_dashboard(self, df, results, sample_text="I'm feeling great today!",
                                     detector=None, save_path=None):
        """
        Create a comprehensive dashboard with multiple visualizations.
        
        Args:
            df (pd.DataFrame): Dataset
            results (dict): Model results
            sample_text (str): Sample text for prediction
            detector: Trained emotion detector
            save_path (str): Optional path to save the plot
        """
        # Validate inputs
        if df is None or df.empty:
            raise ValueError("DataFrame cannot be None or empty")
        if not results:
            raise ValueError("Results dictionary cannot be empty")
            
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Emotion Distribution', 'Model Performance Comparison',
                'Sample Prediction', 'Emotion Confidence Scores',
                'Data Statistics', 'Key Insights'
            ),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "bar"}],
                   [{"type": "table"}, {"type": "scatter"}]]
        )
        
        # 1. Emotion Distribution
        emotion_counts = df['emotion'].value_counts()
        fig.add_trace(
            go.Bar(x=emotion_counts.index, y=emotion_counts.values,
                  name="Emotion Count", showlegend=False,
                  marker_color=[self.emotion_colors.get(e, '#333333') for e in emotion_counts.index]),
            row=1, col=1
        )
        
        # 2. Model Performance
        models = list(results.keys())
        f1_scores = [results[model].get('f1', 0) for model in models]  # Safe get with default
        fig.add_trace(
            go.Bar(x=models, y=f1_scores, name="F1 Score", showlegend=False,
                  marker_color='lightblue'),
            row=1, col=2
        )
        
        # 3. Sample Prediction (if detector available)
        if detector:
            try:
                pred_emotion, emotion_scores = detector.predict_emotion(sample_text)
                emotions = list(emotion_scores.keys())
                scores = list(emotion_scores.values())
                
                fig.add_trace(
                    go.Bar(x=emotions, y=scores, name="Prediction", showlegend=False,
                          marker_color=[self.emotion_colors.get(e, '#333333') for e in emotions]),
                    row=2, col=1
                )
                
                # 4. Top emotion scores
                top_emotions = sorted(emotion_scores.items(), key=lambda x: x[1], reverse=True)[:3]
                fig.add_trace(
                    go.Bar(x=[e[0] for e in top_emotions], y=[e[1] for e in top_emotions],
                          name="Top Predictions", showlegend=False,
                          marker_color='orange'),
                    row=2, col=2
                )
            except Exception as e:
                print(f"Warning: Could not generate predictions: {e}")
        
        # 5. Data Statistics Table
        try:
            avg_length = df['text'].str.len().mean() if 'text' in df.columns else 0
            most_common = df['emotion'].mode()[0] if not df['emotion'].mode().empty else 'Unknown'
            
            stats_data = [
                ['Total Samples', len(df)],
                ['Unique Emotions', df['emotion'].nunique()],
                ['Avg Text Length', f"{avg_length:.1f} chars"],
                ['Most Common Emotion', most_common]
            ]
        except Exception as e:
            stats_data = [
                ['Total Samples', len(df)],
                ['Error', str(e)]
            ]
        
        fig.add_trace(
            go.Table(
                header=dict(values=['Statistic', 'Value']),
                cells=dict(values=[[row[0] for row in stats_data],
                                 [row[1] for row in stats_data]])
            ),
            row=3, col=1
        )
        
        # Update layout
        fig.update_layout(
            height=1200,
            title_text="Emotion Detection Analysis Dashboard",
            showlegend=False
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig

# Example usage
def create_sample_visualizations():
    """Create sample visualizations for demonstration."""
    try:
        from emotion_detector import EmotionDetector
        
        # Initialize detector and create sample data
        detector = EmotionDetector()
        df = detector.create_sample_dataset()
        
        # Prepare and train
        X, y = detector.prepare_data(df)
        results, X_test, y_test = detector.train_models(X, y)
        
        # Initialize visualizer
        visualizer = EmotionVisualizer()
        
        # Create visualizations
        print("Creating emotion distribution plot...")
        visualizer.plot_emotion_distribution(df)
        
        print("Creating model comparison plot...")
        visualizer.plot_model_comparison(results)
        
        print("Creating word clouds...")
        visualizer.create_emotion_wordclouds(df)
        
        # Get predictions for confusion matrix
        if results:
            best_model_name = max(results, key=lambda x: results[x].get('f1', 0))
            if 'y_pred' in results[best_model_name] and 'y_test' in results[best_model_name]:
                y_pred = results[best_model_name]['y_pred']
                y_test = results[best_model_name]['y_test']
                
                print("Creating confusion matrix...")
                visualizer.plot_confusion_matrix(y_test, y_pred, best_model_name)
        
        print("All visualizations created successfully!")
        
    except ImportError as e:
        print(f"Error importing required modules: {e}")
        print("Please ensure all dependencies are installed and emotion_detector.py exists")
    except Exception as e:
        print(f"Error creating visualizations: {e}")

if __name__ == "__main__":
    create_sample_visualizations()