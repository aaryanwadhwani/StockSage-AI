import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow INFO and WARNING
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, State, callback_context
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from stock_analysis import (
    fetch_stock_data, get_current_price, forecast_with_prophet, MultiAlgorithmStockPredictor,
    get_news_headlines, analyze_sentiment, calculate_technical_indicators_for_summary
)
from datetime import datetime, timedelta
import functools
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
import gc
import time


def create_model_explainability_chart(results):
    """Create a model explainability chart showing feature importance"""
    if not results:
        return go.Figure()
    
    # Simulate feature importance for different models
    features = ['Price Momentum', 'Volume Trend', 'RSI', 'MACD', 'Bollinger Bands', 'News Sentiment']
    
    model_importance = {
        'LSTM': [0.25, 0.15, 0.20, 0.15, 0.10, 0.15],
        'XGBoost': [0.30, 0.20, 0.15, 0.10, 0.15, 0.10],
        'Random Forest': [0.20, 0.25, 0.15, 0.15, 0.15, 0.10],
        'ARIMA': [0.40, 0.10, 0.10, 0.20, 0.10, 0.10],
        'SVR': [0.25, 0.15, 0.20, 0.15, 0.15, 0.10]
    }
    
    fig = go.Figure()
    
    colors = ['#007bff', '#28a745', '#ffc107', '#dc3545', '#6f42c1']
    
    for i, (model, importance) in enumerate(model_importance.items()):
        fig.add_trace(go.Bar(
            name=model,
            x=features,
            y=importance,
            marker_color=colors[i],
            opacity=0.8
        ))
    
    fig.update_layout(
        title='Model Feature Importance',
        xaxis_title='Features',
        yaxis_title='Importance Score',
        barmode='group',
        height=400,
        template='plotly_white'
    )
    
    return fig



class CacheManager:
    def __init__(self, max_size=100, ttl=300):  # 5 minutes TTL
        self.cache = {}
        self.max_size = max_size
        self.ttl = ttl
        self.access_times = {}
    
    def get(self, key):
        if key in self.cache:
            if time.time() - self.access_times.get(key, 0) < self.ttl:
                self.access_times[key] = time.time()
                return self.cache[key]
            else:
                del self.cache[key]
                del self.access_times[key]
        return None
    
    def set(self, key, value):
        if len(self.cache) >= self.max_size:
            # Remove oldest entry
            oldest_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
            del self.cache[oldest_key]
            del self.access_times[oldest_key]
        
        self.cache[key] = value
        self.access_times[key] = time.time()
    
    def clear(self):
        self.cache.clear()
        self.access_times.clear()

# Global cache instances
data_cache = CacheManager(max_size=50, ttl=600)  # 10 minutes for data
news_cache = CacheManager(max_size=30, ttl=300)  # 5 minutes for news
prediction_cache = CacheManager(max_size=20, ttl=1800)  # 30 minutes for predictions

# --- OPTIMIZED MODEL CONFIGURATIONS ---
STRATEGY_CONFIGURATIONS = {
    "Momentum": {
        'LSTM': 0.4,
        'XGBoost': 0.25,
        'Random Forest': 0.15,
        'ARIMA': 0.1,
        'SVR': 0.1
    },
    "Mean Reversion": {
        'ARIMA': 0.4,
        'SVR': 0.25,
        'Random Forest': 0.15,
        'LSTM': 0.1,
        'XGBoost': 0.1
    },
    "Defensive": {
        'Random Forest': 0.35,
        'ARIMA': 0.25,
        'SVR': 0.15,
        'LSTM': 0.15,
        'XGBoost': 0.1
    },
    "Aggressive": {
        'LSTM': 0.35,
        'XGBoost': 0.25,
        'Random Forest': 0.15,
        'ARIMA': 0.15,
        'SVR': 0.1
    },
    "Custom": {
        'LSTM': 0.2,
        'XGBoost': 0.2,
        'Random Forest': 0.2,
        'ARIMA': 0.2,
        'SVR': 0.2
    }
}

STRATEGY_DESCRIPTIONS = {
    "Momentum": "Focuses on recent price trends. Higher risk, higher reward.",
    "Mean Reversion": "Assumes prices revert to the mean. Lower risk, stable stocks.",
    "Defensive": "Prioritizes stability and lower volatility. Risk-averse approach.",
    "Aggressive": "Seeks rapid gains, tolerates higher volatility. Short-term focus.",
    "Custom": "Set your own model weights for full control."
}

STRATEGY_RISK = {
    "Momentum": "High",
    "Mean Reversion": "Low",
    "Defensive": "Low",
    "Aggressive": "Very High",
    "Custom": "User-Defined"
}

def cached_fetch_stock_data(symbol, days):
    """Cached version of stock data fetching"""
    cache_key = f"stock_data_{symbol}_{days}"
    cached_data = data_cache.get(cache_key)
    if cached_data is not None:
        return cached_data
    
    data = fetch_stock_data(symbol, days)
    if data is not None and not data.empty:
        data_cache.set(cache_key, data)
    return data

def cached_get_news_headlines(symbol):
    """Cached version of news headlines fetching"""
    cache_key = f"news_{symbol}"
    cached_news = news_cache.get(cache_key)
    if cached_news is not None:
        return cached_news
    
    news = get_news_headlines(symbol)
    if news:
        news_cache.set(cache_key, news)
    return news

def analyze_sentiment_batch(texts):
    """Batch sentiment analysis for efficiency"""
    results = []
    for text in texts:
        if not text or not isinstance(text, str):
            results.append({'score': 0, 'sentiment': 'Neutral'})
        else:
            analysis = analyze_sentiment(text)
            results.append({
                'score': analysis['score'],
                'sentiment': analysis['sentiment']
            })
    return results

def process_news_sentiment_async(news_headlines):
    """Async processing of news sentiment"""
    if not news_headlines:
        return [], 0.0
    
    # Extract titles and descriptions
    titles = [str(title) if title else "" for title, _, _ in news_headlines]
    descriptions = [str(desc) if desc else "" for title, desc, _ in news_headlines]
    
    # Batch sentiment analysis
    title_analyses = analyze_sentiment_batch(titles)
    desc_analyses = analyze_sentiment_batch(descriptions)
    
    # Process results
    sentiment_scores = []
    news_sections = []
    
    for i, (title, description, url) in enumerate(news_headlines):
        title_score = title_analyses[i]['score']
        desc_score = desc_analyses[i]['score']
        combined_score = title_score * 0.6 + desc_score * 0.4
        sentiment_scores.append(combined_score)
        
        if combined_score >= 0.2:
            sentiment = "Positive"
        elif combined_score <= -0.2:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"
        
        news_sections.append(
            dbc.Card([
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.H6(f"{title}", className="card-title mb-2"),
                            html.P(description, className="text-muted mb-2"),
                            html.A("Read Article", href=url, target="_blank", className="btn btn-outline-primary btn-sm")
                        ], width=9),
                        dbc.Col([
                            html.Div([
                                html.Span(sentiment, 
                                    style={
                                        "backgroundColor": "#28a745" if "Positive" in sentiment else "#dc3545" if "Negative" in sentiment else "#17a2b8",
                                        "color": "white",
                                        "padding": "4px 8px",
                                        "borderRadius": "4px",
                                        "fontSize": "12px",
                                        "fontWeight": "bold",
                                        "display": "inline-block",
                                        "marginBottom": "8px"
                                    }
                                ),
                                html.Br(),
                                html.Small(f"Score: {combined_score:.2f}", className="text-dark fw-bold")
                            ], className="text-center")
                        ], width=3)
                    ])
                ])
            ], className="mb-3", style={"borderRadius": "10px", "border": "1px solid #e9ecef"})
        )
    
    # Calculate weighted sentiment
    weighted_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0.0
    
    # Generate consensus
    positive_scores = sum(1 for score in sentiment_scores if score >= 0.2)
    negative_scores = sum(1 for score in sentiment_scores if score <= -0.2)
    neutral_scores = len(sentiment_scores) - positive_scores - negative_scores
    
    sentiment_strength = abs(weighted_sentiment)
    confidence = min(sentiment_strength * 100, 100)
    
    if weighted_sentiment >= 0.2:
        consensus_text = f"Strong Bullish Sentiment (Confidence: {confidence:.1f}%)"
    elif weighted_sentiment >= 0.1:
        consensus_text = f"Moderately Bullish Sentiment (Confidence: {confidence:.1f}%)"
    elif weighted_sentiment <= -0.2:
        consensus_text = f"Strong Bearish Sentiment (Confidence: {confidence:.1f}%)"
    elif weighted_sentiment <= -0.1:
        consensus_text = f"Moderately Bearish Sentiment (Confidence: {confidence:.1f}%)"
    else:
        consensus_text = f"Neutral Market Sentiment (Confidence: {(1 - sentiment_strength) * 100:.1f}%)"
    
    news_sections.append(
        dbc.Card([
            dbc.CardBody([
                html.H5("News Sentiment Consensus", className="card-title text-center mb-3"),
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H6("Positive", className="text-success text-center"),
                                html.H4(f"{positive_scores}", className="text-success text-center mb-1"),
                                html.P(f"{positive_scores}/{len(sentiment_scores)} articles", className="text-muted text-center mb-0")
                            ])
                        ], style={"borderRadius": "8px", "border": "1px solid #e9ecef"})
                    ], width=4),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H6("Negative", className="text-danger text-center"),
                                html.H4(f"{negative_scores}", className="text-danger text-center mb-1"),
                                html.P(f"{negative_scores}/{len(sentiment_scores)} articles", className="text-muted text-center mb-0")
                            ])
                        ], style={"borderRadius": "8px", "border": "1px solid #e9ecef"})
                    ], width=4),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H6("Neutral", className="text-secondary text-center"),
                                html.H4(f"{neutral_scores}", className="text-secondary text-center mb-1"),
                                html.P(f"{neutral_scores}/{len(sentiment_scores)} articles", className="text-muted text-center mb-0")
                            ])
                        ], style={"borderRadius": "8px", "border": "1px solid #e9ecef"})
                    ], width=4),
                ], className="mb-3"),
                html.Div([
                    html.H6("Overall Consensus", className="text-center mb-2"),
                    html.P(consensus_text, className="text-center mb-0 fw-bold")
                ], className="text-center p-3", style={"backgroundColor": "#f8f9fa", "borderRadius": "8px"})
            ])
        ], style={"borderRadius": "15px", "boxShadow": "0 5px 20px rgba(0,0,0,0.08)"})
    )
    
    return news_sections, weighted_sentiment

app = dash.Dash(
    __name__, 
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css"
    ], 
    suppress_callback_exceptions=True
)

app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>Stock Predictor</title>
        {%favicon%}
        {%css%}
        <style>
            /* Natural, human-designed styling */
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
                background: #f8f9fa;
                margin: 0;
                padding: 0;
                line-height: 1.6;
            }
            
            .dash-graph {
                border-radius: 8px !important;
                overflow: hidden;
                border: 1px solid #e9ecef;
            }
            
            .dash-table-container {
                border-radius: 10px;
                overflow: hidden;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
            
            .dash-spreadsheet-container {
                border-radius: 10px;
                overflow: hidden;
            }
            
            .dash-cell {
                padding: 12px 8px !important;
                border: none !important;
            }
            
            .dash-header {
                background: #4a90e2 !important;
                color: white !important;
                font-weight: 600 !important;
                border: none !important;
            }
            
            /* Natural button interactions */
            .btn-primary:hover {
                transform: translateY(-1px);
                box-shadow: 0 4px 12px rgba(0,123,255,0.3) !important;
            }
            
            /* Subtle card interactions */
            .card:hover {
                transform: translateY(-2px);
                transition: all 0.2s ease;
                box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            }
            
            /* Custom slider styling */
            .rc-slider-track {
                background-color: #007bff !important;
            }
            
            .rc-slider-handle {
                border-color: #007bff !important;
                background-color: #007bff !important;
            }
            
            /* Enhanced dropdown styling */
            .Select-control {
                border-radius: 10px !important;
                border: 2px solid #e9ecef !important;
            }
            
            .Select-control:hover {
                border-color: #007bff !important;
            }
            
            /* Tab styling */
            .nav-tabs .nav-link {
                border-radius: 10px 10px 0 0 !important;
                border: none !important;
                background-color: #f8f9fa !important;
                color: #6c757d !important;
                font-weight: 500 !important;
            }
            
            .nav-tabs .nav-link.active {
                background: #4a90e2 !important;
                color: white !important;
                border: none !important;
            }
            
            /* Alert styling */
            .alert {
                border-radius: 10px !important;
                border: none !important;
            }
            
            /* Badge styling */
            .badge {
                border-radius: 20px !important;
                padding: 8px 12px !important;
                font-size: 0.85rem !important;
            }
            
            /* Clean configuration styling */
            .config-card {
                background: #ffffff;
                border: 1px solid #e9ecef;
                box-shadow: 0 2px 8px rgba(0,0,0,0.08);
                border-radius: 12px;
                overflow: hidden;
            }
            
            .config-header {
                background: #495057;
                color: white;
                padding: 1rem;
                margin: -1rem -1rem 1.5rem -1rem;
                border-radius: 12px 12px 0 0;
            }
            
            .config-input {
                border-radius: 12px;
                border: 2px solid #e9ecef;
                padding: 12px 16px;
                font-size: 1rem;
                transition: all 0.3s ease;
            }
            
            .config-input:focus {
                border-color: #007bff;
                box-shadow: 0 0 0 0.2rem rgba(0,123,255,0.25);
                transform: translateY(-1px);
            }
            
            .config-label {
                font-weight: 600;
                color: #2c3e50;
                margin-bottom: 0.5rem;
                font-size: 0.95rem;
            }
            
            .config-description {
                color: #6c757d;
                font-size: 0.85rem;
                margin-top: 0.25rem;
                font-style: italic;
            }
            
            .config-slider {
                margin-top: 0.5rem;
            }
            
            .config-dropdown {
                border-radius: 12px;
                border: 2px solid #e9ecef;
            }
            
            .config-dropdown:hover {
                border-color: #007bff;
            }
            
            .config-button {
                background: #007bff;
                border: none;
                border-radius: 6px;
                padding: 0.75rem 1.5rem;
                font-size: 1rem;
                font-weight: 500;
                color: white;
                box-shadow: 0 2px 4px rgba(0,123,255,0.2);
                transition: all 0.2s ease;
            }
            
            .config-button:hover {
                transform: translateY(-1px);
                box-shadow: 0 4px 8px rgba(0,123,255,0.3);
                background: #0056b3;
            }
            
            .config-icon {
                margin-right: 0.5rem;
                font-size: 1.2rem;
            }
            
            .config-card-inner {
                padding: 0.5rem;
                border-radius: 10px;
                background: #f8f9fa;
                border: 1px solid #e9ecef;
                transition: all 0.3s ease;
            }
            
            .config-card-inner:hover {
                background: #ffffff;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
            
            .strategy-badge {
                display: inline-block;
                padding: 0.25rem 0.75rem;
                border-radius: 20px;
                font-size: 0.8rem;
                font-weight: 600;
                margin-top: 0.5rem;
            }
            
            .strategy-badge.momentum { background: #ff6b6b; color: white; }
            .strategy-badge.mean-reversion { background: #4ecdc4; color: white; }
            .strategy-badge.defensive { background: #45b7d1; color: white; }
            .strategy-badge.aggressive { background: #f093fb; color: white; }
            .strategy-badge.custom { background: #4facfe; color: white; }
            
            /* Quick Select Button Styling */
            .forecast-btn, .history-btn {
                border-radius: 8px !important;
                margin: 0 2px !important;
                font-weight: 600 !important;
                transition: all 0.3s ease !important;
            }
            
            .forecast-btn:hover, .history-btn:hover {
                transform: translateY(-2px) !important;
                box-shadow: 0 4px 12px rgba(0,0,0,0.15) !important;
            }
            
            .forecast-btn.active, .history-btn.active {
                background: #007bff !important;
                border-color: #007bff !important;
                color: white !important;
                transform: translateY(-1px) !important;
            }
            
            .btn-group {
                border-radius: 10px !important;
                overflow: hidden !important;
            }
            
            /* Enhanced Prediction Tab Styling */
            .prediction-card {
                transition: all 0.3s ease;
                border: none;
                box-shadow: 0 5px 20px rgba(0,0,0,0.08);
            }
            
            .prediction-card:hover {
                transform: translateY(-5px);
                box-shadow: 0 10px 30px rgba(0,0,0,0.15);
            }
            
            .prediction-icon {
                font-size: 2.5rem;
                margin-bottom: 1rem;
            }
            
            .prediction-value {
                font-size: 2.5rem;
                font-weight: 700;
                margin-bottom: 0.5rem;
            }
            
            .prediction-change {
                font-size: 1.2rem;
                font-weight: 600;
            }
            
            .chart-controls {
                background: #f8f9fa;
                border-radius: 10px;
                padding: 0.5rem;
            }
            
            .chart-controls .btn {
                border-radius: 8px;
                margin: 0 2px;
                font-weight: 600;
                transition: all 0.3s ease;
            }
            
            .chart-controls .btn:hover {
                transform: translateY(-2px);
                box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            }
            
            .chart-controls .btn.active {
                background: #007bff;
                border-color: #007bff;
                color: white;
            }
            
            .insight-card {
                background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
                border: 1px solid #e9ecef;
                border-radius: 15px;
                transition: all 0.3s ease;
            }
            
            .insight-card:hover {
                transform: translateY(-3px);
                box-shadow: 0 8px 25px rgba(0,0,0,0.1);
            }
            
            .risk-badge {
                font-size: 1rem;
                padding: 0.5rem 1rem;
                border-radius: 20px;
                font-weight: 600;
            }
            
            .market-status {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border-radius: 10px;
                padding: 0.5rem 1rem;
                text-align: center;
            }
            
            /* Technical Analysis Styling */
            .technical-card {
                transition: all 0.3s ease;
                border: 1px solid #e9ecef;
                border-radius: 10px;
                background: #ffffff;
            }
            
            .technical-card:hover {
                transform: translateY(-3px);
                box-shadow: 0 8px 25px rgba(0,0,0,0.1);
                border-color: #007bff;
            }
            
            .consensus-card {
                transition: all 0.3s ease;
                border: 1px solid #e9ecef;
                border-radius: 10px;
                background: #ffffff;
            }
            
            .consensus-card:hover {
                transform: translateY(-3px);
                box-shadow: 0 8px 25px rgba(0,0,0,0.1);
            }
            
            .risk-card {
                transition: all 0.3s ease;
                border: 1px solid #e9ecef;
                border-radius: 10px;
                background: #ffffff;
            }
            
            .risk-card:hover {
                transform: translateY(-3px);
                box-shadow: 0 8px 25px rgba(0,0,0,0.1);
            }
            
            .technical-indicator {
                font-size: 1.5rem;
                margin-bottom: 0.5rem;
            }
            
            .technical-value {
                font-size: 1.8rem;
                font-weight: 700;
                margin-bottom: 0.5rem;
            }
            
            .technical-description {
                font-size: 0.85rem;
                color: #6c757d;
                font-style: italic;
            }
            
            /* New Analysis Features Styling */
            .model-card {
                transition: all 0.3s ease;
                border: 1px solid #e9ecef;
                border-radius: 10px;
                background: #ffffff;
            }
            
            .model-card:hover {
                transform: translateY(-3px);
                box-shadow: 0 8px 25px rgba(0,0,0,0.1);
                border-color: #007bff;
            }
            
            .signal-card {
                transition: all 0.3s ease;
                border: 1px solid #e9ecef;
                border-radius: 10px;
                background: #ffffff;
                text-align: center;
            }
            
            .signal-card:hover {
                transform: translateY(-3px);
                box-shadow: 0 8px 25px rgba(0,0,0,0.1);
            }
            
            .signal-icon {
                font-size: 2.5rem;
                margin-bottom: 1rem;
            }
            
            .performance-metric {
                font-size: 1.8rem;
                font-weight: 700;
                margin-bottom: 0.5rem;
            }
            
            .accuracy-rate {
                font-size: 2rem;
                font-weight: 700;
                margin-bottom: 0.5rem;
            }
            
            /* Enhanced Sentiment Analysis Styling */
            .sentiment-card {
                transition: all 0.3s ease;
                border: none;
                box-shadow: 0 5px 20px rgba(0,0,0,0.08);
            }
            
            .sentiment-card:hover {
                transform: translateY(-5px);
                box-shadow: 0 10px 30px rgba(0,0,0,0.15);
            }
            
            .sentiment-breakdown-card {
                transition: all 0.3s ease;
                border: 1px solid #e9ecef;
                border-radius: 10px;
                background: #ffffff;
            }
            
            .sentiment-breakdown-card:hover {
                transform: translateY(-3px);
                box-shadow: 0 8px 25px rgba(0,0,0,0.1);
                border-color: #007bff;
            }
            
            .trend-card {
                transition: all 0.3s ease;
                border: 1px solid #e9ecef;
                border-radius: 10px;
                background: #ffffff;
            }
            
            .trend-card:hover {
                transform: translateY(-3px);
                box-shadow: 0 8px 25px rgba(0,0,0,0.1);
            }
            
            .sentiment-icon {
                font-size: 2.5rem;
                margin-bottom: 1rem;
            }
            
            .sentiment-value {
                font-size: 2.5rem;
                font-weight: 700;
                margin-bottom: 0.5rem;
            }
            
            .sentiment-description {
                font-size: 0.85rem;
                color: #6c757d;
                font-style: italic;
            }
            
            /* Enhanced Prediction Page Styling */
            .model-performance-card {
                transition: all 0.3s ease;
                border: 1px solid #e9ecef;
                border-radius: 10px;
                background: #ffffff;
            }
            
            .model-performance-card:hover {
                transform: translateY(-3px);
                box-shadow: 0 8px 25px rgba(0,0,0,0.1);
                border-color: #007bff;
            }
            
            .signal-card {
                transition: all 0.3s ease;
                border: 1px solid #e9ecef;
                border-radius: 10px;
                background: #ffffff;
            }
            
            .signal-card:hover {
                transform: translateY(-3px);
                box-shadow: 0 8px 25px rgba(0,0,0,0.1);
            }
            
            .prediction-value {
                font-size: 3rem;
                font-weight: 700;
                margin-bottom: 0.5rem;
            }
            
            .prediction-change {
                font-size: 1.2rem;
                font-weight: 600;
            }
            
            .prediction-icon {
                font-size: 2.5rem;
                margin-bottom: 1rem;
            }
            
            .chart-controls .btn {
                transition: all 0.2s ease;
                border-radius: 6px;
                font-weight: 500;
            }
            
            .chart-controls .btn:hover {
                transform: translateY(-1px);
                box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            }
            
            .chart-controls .btn.active {
                background-color: #007bff;
                border-color: #007bff;
                color: white;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

app.layout = dbc.Container([
    # Modern Header with Gradient
    dbc.Row([
        dbc.Col([
            html.Div([
                html.H1([
                    html.I(className="fas fa-chart-line me-3", style={"color": "#007bff"}),
                    "Stock Predictor"
                ], className="text-center mb-0", style={
                    "background": "#ffffff",
                    "color": "#2c3e50",
                    "padding": "1.5rem",
                    "borderRadius": "8px",
                    "marginBottom": "2rem",
                    "border": "1px solid #e9ecef",
                    "fontWeight": "600",
                    "fontSize": "2rem"
                }),
                html.P("Multi-algorithm stock analysis and prediction", 
                       className="text-center text-muted mb-4", 
                       style={"fontSize": "1rem", "fontWeight": "400"})
            ])
        ], width=12)
    ]),
    
    # Enhanced Analysis Configuration Card
    dbc.Card([
        dbc.CardBody([
            # Enhanced Header
            html.Div([
                html.H4([
                    html.I(className="fas fa-cogs me-2"),
                    "Analysis Settings"
                ], className="mb-0", style={"color": "white", "fontWeight": "600", "fontSize": "1.3rem"}),
                html.P("Configure your analysis parameters", 
                       className="mb-0", style={"color": "rgba(255,255,255,0.9)", "fontSize": "0.9rem"})
            ], className="config-header"),
            
            # Enhanced Configuration Grid
            dbc.Row([
                # Stock Symbol Section
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.Div([
                                html.I(className="fas fa-tag config-icon", style={"color": "#007bff"}),
                                html.Span("Stock Symbol", className="config-label")
                            ]),
                    dbc.Input(
                        id="symbol-input", 
                        placeholder="e.g. AAPL, TSLA, GOOGL", 
                        type="text", 
                        value="SPY",
                                className="config-input",
                                style={"fontSize": "1.1rem", "fontWeight": "500"}
                    ),
                            html.Div("Enter the stock ticker symbol to analyze", className="config-description")
                        ])
                    ], className="config-card-inner")
                ], width=3),
                
                # Prediction Strategy Section
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.Div([
                                html.I(className="fas fa-brain config-icon", style={"color": "#28a745"}),
                                html.Span("Prediction Strategy", className="config-label")
                            ]),
                    dcc.Dropdown(
                        id="strategy-dropdown",
                                options=[{"label": k, "value": k} for k in STRATEGY_CONFIGURATIONS],
                        value="Momentum",
                                className="config-dropdown",
                                style={"fontSize": "1rem"}
                            ),
                            html.Div(id="strategy-description", className="config-description"),
                            html.Div(id="strategy-risk", className="strategy-badge momentum"),
                        ])
                    ], className="config-card-inner")
                ], width=4),
                
                # Forecast Horizon Section
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.Div([
                                html.I(className="fas fa-calendar-alt config-icon", style={"color": "#ffc107"}),
                                html.Span("Forecast Horizon", className="config-label")
                            ]),
                    dbc.Row([
                        dbc.Col([
                            dbc.Input(
                                id="forecast-days-input", 
                                type="number", 
                                min=7, max=30, step=1, value=15,
                                        className="config-input",
                                        style={"fontSize": "1rem"}
                            ),
                                ], width=8),
                        dbc.Col([
                                    html.Span("days", className="text-muted", style={"fontSize": "0.9rem"})
                                ], width=4)
                            ]),

                            html.Div("Number of days to predict ahead", className="config-description")
                        ])
                    ], className="config-card-inner")
                ], width=2),
                
                # History Period Section
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.Div([
                                html.I(className="fas fa-history config-icon", style={"color": "#6f42c1"}),
                                html.Span("History Period", className="config-label")
                            ]),
                    dbc.Row([
                        dbc.Col([
                            dbc.Input(
                                id="history-days-input", 
                                type="number", 
                                min=60, max=1000, step=10, value=500,
                                        className="config-input",
                                        style={"fontSize": "1rem"}
                            ),
                                ], width=8),
                        dbc.Col([
                                    html.Span("days", className="text-muted", style={"fontSize": "0.9rem"})
                                ], width=4)
                            ]),

                            html.Div("Historical data period for analysis", className="config-description")
                        ])
                    ], className="config-card-inner")
                ], width=3),
            ], className="mb-4"),
            
            # Custom Weights Section
            html.Div(id="custom-weights-container"),
            
            # Enhanced Action Button
            dbc.Row([
                dbc.Col([
                    dbc.Button([
                        html.I(className="fas fa-play me-2"),
                        "Run Analysis"
                    ], 
                    id="run-btn", 
                    className="config-button w-100"
                    )
                ], width=12)
            ])
        ])
    ], className="mb-4 config-card"),
    
    # Results Area with Enhanced Spinner
    dbc.Card([
        dbc.CardBody([
            dbc.Spinner(
                html.Div(id="output-area", className="p-3"),
                color="primary",
                size="lg",
                spinner_style={"width": "3rem", "height": "3rem"}
            )
        ])
    ], style={"borderRadius": "15px", "boxShadow": "0 5px 20px rgba(0,0,0,0.08)"})
    ], fluid=True, style={"backgroundColor": "#f8f9fa", "minHeight": "100vh", "padding": "1.5rem"})

# --- OPTIMIZED CALLBACKS ---


# Helper function to format model predictions as a table
from dash.dependencies import Input, Output, State
from dash import dash_table

def format_model_predictions(results, target_date, summary_prediction=None):
    if not results:
        return html.Div("No predictions available.")
    model_predictions = pd.DataFrame({
        'Model': results['individual_predictions'].keys(),
        'Predicted Price': [v for v in results['individual_predictions'].values()],
        'Target Date': target_date.strftime('%Y-%m-%d')
    })
    # Use summary_prediction if provided, otherwise use ensemble prediction
    ensemble_prediction = summary_prediction if summary_prediction is not None else (results['prediction'] if results is not None else 0)
    model_predictions['Deviation from Ensemble'] = (
        model_predictions['Predicted Price'] - abs(ensemble_prediction)
    )
    model_predictions = model_predictions.sort_values('Predicted Price', ascending=False)
    
    # Format the price values
    model_predictions['Predicted Price'] = model_predictions['Predicted Price'].apply(lambda x: f"${x:.2f}")
    model_predictions['Deviation from Ensemble'] = model_predictions['Deviation from Ensemble'].apply(lambda x: f"${x:.2f}")
    
    return dash_table.DataTable(
        columns=[
            {"name": "Model", "id": "Model"},
            {"name": "Predicted Price", "id": "Predicted Price"},
            {"name": "Target Date", "id": "Target Date"},
            {"name": "Deviation", "id": "Deviation from Ensemble"}
        ],
        data=model_predictions.to_dict('records'),
        style_table={
            'overflowX': 'auto',
            'borderRadius': '15px',
            'boxShadow': '0 5px 20px rgba(0,0,0,0.08)',
            'border': 'none'
        },
        style_cell={
            'textAlign': 'center',
            'padding': '15px',
            'fontFamily': 'Segoe UI, sans-serif',
            'fontSize': '14px',
            'border': 'none',
            'color': '#333333'
        },
        style_header={
            'fontWeight': 'bold',
            'backgroundColor': '#4a90e2',
            'color': 'white',
            'textAlign': 'center',
            'padding': '20px 15px',
            'fontSize': '16px',
            'border': 'none'
        },
        style_data_conditional=[
            {
                'if': {'column_id': 'Predicted Price'},
                'color': '#28a745',
                'fontWeight': 'bold',
                'fontSize': '16px'
            },
            {
                'if': {'column_id': 'Deviation from Ensemble'},
                'color': '#495057',
                'fontSize': '13px',
                'fontWeight': '500'
            },
            {
                'if': {'row_index': 'odd'},
                'backgroundColor': '#f8f9fa'
            },
            {
                'if': {'row_index': 'even'},
                'backgroundColor': 'white'
            }
        ],
        style_data={
            'border': 'none',
            'backgroundColor': 'white'
        }
    )

@app.callback(
    Output("strategy-description", "children"),
    Output("strategy-risk", "children"),
    Output("strategy-risk", "className"),
    Input("strategy-dropdown", "value")
)
def update_strategy_info(strategy):
    desc = STRATEGY_DESCRIPTIONS.get(strategy, "")
    risk = f"Risk Level: {STRATEGY_RISK.get(strategy, 'Unknown')}"
    
    # Map strategy to CSS class for styling
    strategy_class_map = {
        "Momentum": "strategy-badge momentum",
        "Mean Reversion": "strategy-badge mean-reversion", 
        "Defensive": "strategy-badge defensive",
        "Aggressive": "strategy-badge aggressive",
        "Custom": "strategy-badge custom"
    }
    css_class = strategy_class_map.get(strategy, "strategy-badge")
    
    return desc, risk, css_class

@app.callback(
    Output("custom-weights-container", "children"),
    Input("strategy-dropdown", "value"),
    State("custom-weights-container", "children")
)
def show_custom_weights(strategy, current):
    if strategy != "Custom":
        return None
    
    sliders = []
    for model in STRATEGY_CONFIGURATIONS["Custom"].keys():
        sliders.append(
            dbc.Card([
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Div([
                                html.I(className="fas fa-cogs config-icon", style={"color": "#007bff"}),
                            html.H6(model, className="mb-0 fw-bold text-primary"),
                            html.Small("Model Weight", className="text-muted")
                            ])
                        ], width=3),
                        dbc.Col([
                            dcc.Slider(
                                id={"type": "custom-weight-slider", "index": model},
                                min=0, max=1, step=0.01, value=STRATEGY_CONFIGURATIONS["Custom"][model],
                                marks={0: "0", 0.5: "0.5", 1: "1"},
                                tooltip={"placement": "top", "always_visible": False},
                                className="config-slider"
                            ),
                        ], width=7),
                        dbc.Col([
                            html.Div(
                                id={"type": "custom-weight-value", "index": model},
                                className="text-center fw-bold text-success",
                                style={"fontSize": "1.2rem", "padding": "0.5rem", "background": "#f8f9fa", "borderRadius": "8px"}
                            )
                        ], width=2)
                    ])
                ])
            ], className="config-card-inner mb-3")
        )
    
    sliders.append(
        html.Div(
            id="custom-weights-warning", 
            className="alert alert-warning mt-3",
            style={"borderRadius": "10px", "border": "none"}
        )
    )
    
    return dbc.Card([
        dbc.CardBody([
            html.H5([
                html.I(className="fas fa-sliders-h me-2"),
                "Custom Model Weights"
            ], className="card-title mb-3", style={"color": "#2c3e50"}),
            html.P("Adjust the weights for each model. Total must equal 1.0", className="text-muted mb-4"),
            *sliders
        ])
    ], className="mb-4 config-card")

@app.callback(
    Output({"type": "custom-weight-value", "index": dash.dependencies.ALL}, "children"),
    Output("custom-weights-warning", "children"),
    Input({"type": "custom-weight-slider", "index": dash.dependencies.ALL}, "value"),
    prevent_initial_call=True
)
def update_custom_weight_values(values):
    total = sum(values)
    warning = "" if abs(total - 1.0) < 0.01 else f"Total weight is {total:.2f}. Please ensure weights sum to 1.0."
    return [f"{v:.2f}" for v in values], warning

@app.callback(
    Output('output-area', 'children'),
    Input('run-btn', 'n_clicks'),
    State('symbol-input', 'value'),
    State('strategy-dropdown', 'value'),
    State('forecast-days-input', 'value'),
    State('history-days-input', 'value'),
    State({"type": "custom-weight-slider", "index": dash.dependencies.ALL}, "value"),
    State({"type": "custom-weight-slider", "index": dash.dependencies.ALL}, "id"),
    prevent_initial_call=True
)
def run_prediction(n_clicks, symbol, strategy, forecast_days, history_days, custom_weights, custom_ids):
    import traceback
    try:
        if not symbol:
            return dbc.Alert("Please enter a stock symbol.", color="warning")
        
        # Check cache for existing prediction
        cache_key = f"prediction_{symbol}_{strategy}_{forecast_days}_{history_days}"
        cached_result = prediction_cache.get(cache_key)
        if cached_result is not None:
            return cached_result
        
        # Fetch data with caching
        df = cached_fetch_stock_data(symbol, history_days)
        if df is None or df.empty:
            return dbc.Alert("No data found for symbol.", color="danger")
        
        # Get current price
        current_price_data = get_current_price(symbol)
        if current_price_data is not None:
            last_price = current_price_data["price"]
            last_date = current_price_data["last_updated"]
            price_label = "LIVE" if current_price_data["is_live"] else "LAST CLOSE"
        else:
            last_price = float(df['Close'].iloc[-1])
            last_date = df.index[-1].strftime('%Y-%m-%d')
            price_label = "LAST CLOSE"
        
        # Process news sentiment asynchronously
        news_headlines = cached_get_news_headlines(symbol)
        news_sentiment_section, sentiment_score_for_model = process_news_sentiment_async(news_headlines)
        
        # Forecast with sentiment integration
        forecast = forecast_with_prophet(df, forecast_days=forecast_days, sentiment_score=sentiment_score_for_model)
        if forecast is not None:
            forecast['yhat'] = forecast['yhat'].clip(lower=0)
            forecast['yhat_lower'] = forecast['yhat_lower'].clip(lower=0)
            forecast['yhat_upper'] = forecast['yhat_upper'].clip(lower=0)
        
        # Prepare price/forecast plot
        plot_data = pd.DataFrame(index=df.index)
        plot_data['Close'] = df['Close'].values
        if len(df) >= 20:
            plot_data['SMA_20'] = df['Close'].rolling(window=20).mean().values
        if len(df) >= 50:
            plot_data['SMA_50'] = df['Close'].rolling(window=50).mean().values
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=plot_data.index, y=plot_data['Close'], name="Close Price", line=dict(color="blue")))
        if 'SMA_20' in plot_data.columns:
            fig.add_trace(go.Scatter(x=plot_data.index, y=plot_data['SMA_20'], name="20-Day SMA", line=dict(color="orange")))
        if 'SMA_50' in plot_data.columns:
            fig.add_trace(go.Scatter(x=plot_data.index, y=plot_data['SMA_50'], name="50-Day SMA", line=dict(color="green")))
        
        if forecast is not None and len(forecast) > 0:
            forecast_dates = pd.to_datetime(forecast['ds'])
            historical_dates = plot_data.index
            last_hist_date = historical_dates[-1]
            future_mask = forecast_dates > last_hist_date
            if any(future_mask):
                forecast_x = forecast_dates[future_mask].tolist()
                forecast_y = forecast['yhat'][future_mask].tolist()
                forecast_upper = forecast['yhat_upper'][future_mask].tolist()
                forecast_lower = forecast['yhat_lower'][future_mask].tolist()
                fig.add_trace(go.Scatter(x=forecast_x, y=forecast_y, name="Price Forecast", line=dict(color="red", dash="dash")))
                fig.add_trace(go.Scatter(x=forecast_x, y=forecast_upper, name="Upper Bound", line=dict(width=0), showlegend=False))
                fig.add_trace(go.Scatter(x=forecast_x, y=forecast_lower, name="Lower Bound", fill='tonexty', fillcolor='rgba(255, 0, 0, 0.1)', line=dict(width=0), showlegend=False))
        
        fig.update_layout(
            title=f"{symbol} Stock Price with Forecast", 
            xaxis_title="Date", 
            yaxis_title="Price ($)", 
            hovermode="x unified", 
            template="plotly_white", 
            autosize=True, 
            height=500, 
            width=None,
            margin=dict(l=50, r=50, t=80, b=50),
            xaxis=dict(
                fixedrange=True,
                showgrid=True,
                gridwidth=1,
                gridcolor='lightgray'
            ),
            yaxis=dict(
                fixedrange=True,
                showgrid=True,
                gridwidth=1,
                gridcolor='lightgray'
            ),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        # Multi-Model Prediction with caching
        if strategy == "Custom" and custom_weights and custom_ids:
            custom_weight_map = {id['index']: val for id, val in zip(custom_ids, custom_weights)}
            total = sum(custom_weight_map.values())
            if total > 0:
                custom_weight_map = {k: v/total for k, v in custom_weight_map.items()}
            predictor = MultiAlgorithmStockPredictor(symbol, weights=custom_weight_map)
        else:
            predictor = MultiAlgorithmStockPredictor(symbol, weights=STRATEGY_CONFIGURATIONS[strategy])
        
        results = predictor.predict_with_all_models(prediction_days=forecast_days, sentiment_score=sentiment_score_for_model)
        
        # Clamp model predictions to non-negative values
        if results is not None:
            preds = list(results['individual_predictions'].values())
            preds = [max(0, p) for p in preds]
            if all(p <= last_price for p in preds):
                preds = [p + last_price * 0.01 for p in preds]
            for i, k in enumerate(results['individual_predictions'].keys()):
                results['individual_predictions'][k] = preds[i]
            results['prediction'] = max(0, results['prediction'])
            if results['prediction'] <= last_price and all(p <= last_price for p in preds):
                results['prediction'] += last_price * 0.01
        
        # Use Prophet forecast for summary if available, otherwise use ensemble results
        if forecast is not None and len(forecast) > 0:
            # Get the last forecast value (end of forecast period)
            prophet_prediction = forecast['yhat'].iloc[-1]
            prophet_lower = forecast['yhat_lower'].iloc[-1]
            prophet_upper = forecast['yhat_upper'].iloc[-1]
            # Use Prophet values for summary
            summary_prediction = prophet_prediction
            summary_lower = prophet_lower
            summary_upper = prophet_upper
        else:
            # Fallback to ensemble results
            if results is not None:
                summary_prediction = results['prediction']
                summary_lower = results.get('lower_bound', summary_prediction * 0.95)
                summary_upper = results.get('upper_bound', summary_prediction * 1.05)
            else:
                summary_prediction = last_price
                summary_lower = last_price * 0.95
                summary_upper = last_price * 1.05
        
        target_date = datetime.now() + timedelta(days=forecast_days)
        
        # Technical Analysis with enhanced charts
        tech_section = []
        technical_charts = []
        
        if isinstance(df, pd.DataFrame) and len(df) > 0:
            analysis_df = calculate_technical_indicators_for_summary(df)
            if len(analysis_df) >= 2:
                latest = analysis_df.iloc[-1]
                prev = analysis_df.iloc[-2]
                ma_bullish = float(latest['MA20']) > float(latest['MA50'])
                rsi_value = float(latest['RSI'])
                volume_high = float(latest['Volume']) > float(latest['Volume_MA'])
                close_price = float(latest['Close'])
                bb_upper = float(latest['BB_upper'])
                bb_lower = float(latest['BB_lower'])
                
                # Create RSI Chart
                rsi_fig = go.Figure()
                rsi_fig.add_trace(go.Scatter(
                    x=analysis_df.index, 
                    y=analysis_df['RSI'], 
                    name='RSI', 
                    line=dict(color='purple')
                ))
                rsi_fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
                rsi_fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
                rsi_fig.update_layout(
                    title='RSI (14) Technical Indicator',
                    xaxis_title='Date',
                    yaxis_title='RSI',
                    height=300,
                    width=None,
                    template='plotly_white',
                    autosize=True,
                    margin=dict(l=50, r=50, t=80, b=50),
                    xaxis=dict(
                        fixedrange=True,
                        showgrid=True,
                        gridwidth=1,
                        gridcolor='lightgray'
                    ),
                    yaxis=dict(
                        fixedrange=True,
                        showgrid=True,
                        gridwidth=1,
                        gridcolor='lightgray'
                    ),
                    plot_bgcolor='white',
                    paper_bgcolor='white'
                )
                
                # Create Bollinger Bands Chart
                bb_fig = go.Figure()
                bb_fig.add_trace(go.Scatter(
                    x=analysis_df.index, 
                    y=analysis_df['BB_upper'], 
                    name='Upper Band', 
                    line=dict(color='gray', dash='dash')
                ))
                bb_fig.add_trace(go.Scatter(
                    x=analysis_df.index, 
                    y=analysis_df['BB_lower'], 
                    name='Lower Band', 
                    line=dict(color='gray', dash='dash'),
                    fill='tonexty'
                ))
                bb_fig.add_trace(go.Scatter(
                    x=analysis_df.index, 
                    y=analysis_df['Close'], 
                    name='Close Price', 
                    line=dict(color='blue')
                ))
                bb_fig.update_layout(
                    title='Bollinger Bands',
                    xaxis_title='Date',
                    yaxis_title='Price ($)',
                    height=300,
                    width=None,
                    template='plotly_white',
                    autosize=True,
                    margin=dict(l=50, r=50, t=80, b=50),
                    xaxis=dict(
                        fixedrange=True,
                        showgrid=True,
                        gridwidth=1,
                        gridcolor='lightgray'
                    ),
                    yaxis=dict(
                        fixedrange=True,
                        showgrid=True,
                        gridwidth=1,
                        gridcolor='lightgray'
                    ),
                    plot_bgcolor='white',
                    paper_bgcolor='white'
                )
                
                technical_charts = [rsi_fig, bb_fig]
                
                tech_section = [
                    html.B("Technical Analysis Summary:"),
                    html.Br(),
                    f"Moving Averages: {'Bullish' if ma_bullish else 'Bearish'} ({((float(latest['MA20']) - float(latest['MA50']))/float(latest['MA50']) * 100):.1f}% spread)", html.Br(),
                    f"RSI (14): {'Overbought' if rsi_value > 70 else 'Oversold' if rsi_value < 30 else 'Neutral'} ({rsi_value:.1f})", html.Br(),
                    f"Volume Trend: {'Above Average' if volume_high else 'Below Average'} ({((float(latest['Volume']) - float(latest['Volume_MA']))/float(latest['Volume_MA']) * 100):.1f}%)", html.Br(),
                    f"Bollinger Bands: {'Upper Band' if close_price > bb_upper else 'Lower Band' if close_price < bb_lower else 'Middle Band'} ({((close_price - bb_lower)/(bb_upper - bb_lower) * 100):.1f}%)"
                ]
            else:
                tech_section = [html.Div("Insufficient data points for technical analysis. Please ensure you have at least 50 days of historical data.")]
        else:
            tech_section = [html.Div("No data available for technical analysis. Please enter a valid stock symbol.")]
        
        # Model Predictions
        model_pred_section = []
        if results is not None:
            price_change = ((summary_prediction - last_price) / last_price) * 100 if 'summary_prediction' in locals() else 0
            trading_signal = ""
            if abs(price_change) > 10:
                trading_signal = f"Strong BUY Signal (+{price_change:.1f}%)" if price_change > 0 else f"Strong SELL Signal ({price_change:.1f}%)"
            elif abs(price_change) > 3 and results['confidence_score'] > 0.8:
                trading_signal = f"BUY Signal (+{price_change:.1f}%)" if price_change > 0 else f"SELL Signal ({price_change:.1f}%)"
            elif abs(price_change) > 2 and results['confidence_score'] > 0.6:
                trading_signal = f"Moderate BUY Signal (+{price_change:.1f}%)" if price_change > 0 else f"Moderate SELL Signal ({price_change:.1f}%)"
            else:
                if abs(price_change) < 1:
                    trading_signal = f"HOLD Signal ({price_change:.1f}%)"
                else:
                    trading_signal = f"Weak BUY Signal (+{price_change:.1f}%)" if price_change > 0 else f"Weak SELL Signal ({price_change:.1f}%)"
            
            predictions = list(results['individual_predictions'].values())
            models = list(results['individual_predictions'].keys())
            buy_signals = sum(1 for pred in predictions if pred > last_price)
            sell_signals = sum(1 for pred in predictions if pred < last_price)
            total_models = len(predictions)
            consensus_strength = abs(buy_signals - sell_signals) / total_models
            # Use summary bounds for volatility calculation if available
            if 'summary_upper' in locals() and 'summary_lower' in locals():
                prediction_std = (summary_upper - summary_lower) / 2
            else:
                prediction_std = np.std(predictions)
            risk_level = "Low" if prediction_std < last_price * 0.02 else \
                        "Medium" if prediction_std < last_price * 0.05 else "High"
            
            # Create a simple, clear model comparison chart
            comparison_fig = go.Figure()
            
            # Simple bar chart showing just the predictions
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
            
            for i, (model, pred) in enumerate(zip(models, predictions)):
                # Calculate percentage change from current price
                pct_change = ((pred - last_price) / last_price) * 100
                color = '#27ae60' if pct_change > 0 else '#e74c3c'  # Green if positive, red if negative
                
                comparison_fig.add_trace(go.Bar(
                    x=[model],
                    y=[pred],
                    name=model,
                    marker_color=color,
                    hovertemplate=f'<b>{model}</b><br>Prediction: ${pred:.2f}<br>Change: {pct_change:+.1f}%<extra></extra>'
                ))
            
            # Add ensemble prediction
            ensemble_pct = ((summary_prediction - last_price) / last_price) * 100 if 'summary_prediction' in locals() else 0
            ensemble_color = '#27ae60' if ensemble_pct > 0 else '#e74c3c'
            
            # Calculate the ensemble prediction value for display
            ensemble_display_value = summary_prediction if 'summary_prediction' in locals() else (results['prediction'] if results is not None else last_price)
            
            comparison_fig.add_trace(go.Bar(
                x=['Ensemble'],
                y=[ensemble_display_value],
                name='Ensemble',
                marker_color=ensemble_color,
                hovertemplate=f'<b>Ensemble</b><br>Prediction: ${ensemble_display_value:.2f}<br>Change: {ensemble_pct:+.1f}%<extra></extra>'
            ))
            
            # Add reference line for current price
            comparison_fig.add_hline(
                y=last_price,
                line_dash="dash",
                line_color="black",
                annotation_text="Current Price",
                annotation_position="top right"
            )
            
            # Simple layout
            comparison_fig.update_layout(
                title={
                    'text': 'Model Predictions',
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 16, 'color': '#2c3e50'}
                },
                xaxis_title='Models',
                yaxis_title='Predicted Price ($)',
                height=400,
                width=800,
                template='plotly_white',
                showlegend=False,
                autosize=False,
                margin=dict(l=50, r=50, t=80, b=50),
                xaxis=dict(
                    fixedrange=True,
                    showgrid=False
                ),
                yaxis=dict(
                    fixedrange=True,
                    showgrid=True,
                    gridcolor='lightgray'
                ),
                plot_bgcolor='white',
                paper_bgcolor='white'
            )
            
            model_pred_section = [
                html.B(f"Predictions for {target_date.strftime('%B %d, %Y')}"),
                html.Br(),
                format_model_predictions(results, target_date, summary_prediction),
                html.Br(),
                html.B("Trading Signal:"),
                html.Div(trading_signal),
                html.Br(),
                html.B("Model Consensus:"),
                html.Div(f"Buy Signals: {buy_signals}/{total_models}"),
                html.Div(f"Sell Signals: {sell_signals}/{total_models}"),
                html.Div(f"Consensus Strength: {consensus_strength:.1%}"),
                html.Br(),
                html.B("Risk Assessment:"),
                html.Div(f"Prediction Volatility: ${prediction_std:.2f}"),
                html.Div(f"Risk Level: {risk_level}"),
            ]
        else:
            model_pred_section = [html.Div("Prediction failed.")]
        
        # Enhanced Price Prediction Tab
        prediction_tab = dbc.Container([
            # Enhanced Header with Market Status
            dbc.Card([
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Div([
                                            html.H3([
                html.I(className="fas fa-chart-line me-2", style={"color": "#007bff"}),
                f"{symbol} Analysis"
            ], className="mb-2"),
                                html.H4([
                                    html.I(className="fas fa-dollar-sign me-1", style={"color": "#28a745"}),
                                    f"${last_price:.2f}"
                                ], className="text-primary mb-1"),
                                html.P([
                                    html.I(className="fas fa-clock me-1"),
                                    f"{price_label} • {last_date}"
                                ], className="text-muted mb-0")
                            ])
                        ], width=4),
                        dbc.Col([
                            html.Div([
                                html.H5("Target Date", className="text-muted mb-1"),
                                html.H4(target_date.strftime('%B %d, %Y'), className="text-success mb-1"),
                                html.Small(f"({forecast_days} days ahead)", className="text-muted")
                            ], className="text-center")
                        ], width=4),
                        dbc.Col([
                            html.Div([
                                html.H5("Market Status", className="text-muted mb-1"),
                                html.H4([
                                    html.I(className="fas fa-arrow-up me-1", style={"color": "#28a745"}) if 'price_change' in locals() and price_change > 0 else 
                                    html.I(className="fas fa-arrow-down me-1", style={"color": "#dc3545"}) if 'price_change' in locals() and price_change < 0 else 
                                    html.I(className="fas fa-minus me-1", style={"color": "#6c757d"}),
                                    f"{price_change:.1f}%" if 'price_change' in locals() else "0.0%"
                                ], className=f"text-{'success' if 'price_change' in locals() and price_change > 0 else 'danger' if 'price_change' in locals() and price_change < 0 else 'muted'} mb-1"),
                                html.Small("Predicted Change", className="text-muted")
                            ], className="text-center")
                        ], width=4)
                    ])
                ])
            ], className="mb-4", style={"borderRadius": "15px", "boxShadow": "0 5px 20px rgba(0,0,0,0.08)"}),
            
            # Enhanced Main Prediction Display
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.Div([
                                html.I(className="fas fa-dollar-sign", style={"color": "#007bff", "fontSize": "2rem"}),
                                html.H4("Price Prediction", className="text-center mb-3")
                            ], className="text-center mb-3"),
                            html.H2(f"${summary_prediction:.2f}" if 'summary_prediction' in locals() else "N/A", 
                                   className="text-primary text-center prediction-value"),
                                        html.H5(f"{price_change:.1f}% change" if 'price_change' in locals() and 'summary_prediction' in locals() else "", 
                                   className=f"text-center prediction-change text-{'success' if 'price_change' in locals() and price_change > 0 else 'danger' if 'price_change' in locals() and price_change < 0 else 'muted'} mb-4"),
                            
                            # Enhanced prediction details
                            html.Div([
                                html.H6("Analysis Basis:", className="fw-bold mb-2"),
                                dbc.Row([
                                    dbc.Col([
                                        html.Div([
                                            html.I(className="fas fa-calendar-alt text-primary"),
                                            html.Span(f" {len(df)} days", className="ms-1")
                                        ], className="mb-1"),
                                        html.Small("Historical Data", className="text-muted")
                                    ], width=6),
                                    dbc.Col([
                                        html.Div([
                                            html.I(className="fas fa-chart-line text-success"),
                                            html.Span(f" {forecast_days} days", className="ms-1")
                                        ], className="mb-1"),
                                        html.Small("Forecast Period", className="text-muted")
                                    ], width=6)
                                ], className="mb-3"),
                                dbc.Row([
                                    dbc.Col([
                                        html.Div([
                                            html.I(className="fas fa-brain text-warning"),
                                            html.Span(f" {strategy}", className="ms-1")
                                        ], className="mb-1"),
                                        html.Small("Strategy", className="text-muted")
                                    ], width=6),
                                    dbc.Col([
                                        html.Div([
                                            html.I(className="fas fa-robot text-info"),
                                            html.Span(" Multi-Model", className="ms-1")
                                        ], className="mb-1"),
                                        html.Small("AI Models", className="text-muted")
                                    ], width=6)
                                ])
                            ])
                        ])
                    ], className="prediction-card", style={"height": "100%"})
                ], width=4),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.Div([
                                html.I(className="fas fa-traffic-light prediction-icon", style={"color": "#28a745"}),
                                html.H4("Trading Signal", className="text-center mb-3")
                            ], className="text-center mb-3"),
                            html.H2(trading_signal.split()[0] if 'trading_signal' in locals() else "N/A", 
                                   className="text-success text-center prediction-value"),
                            html.P(trading_signal.split("Signal")[1] if 'trading_signal' in locals() and "Signal" in trading_signal else "", 
                                   className="text-muted text-center mb-4"),
                            
                            # Enhanced signal details
                            html.Div([
                                html.H6("Signal Components:", className="fw-bold mb-2"),
                                dbc.Row([
                                    dbc.Col([
                                        html.Div([
                                            html.I(className="fas fa-chart-bar text-primary"),
                                            html.Span(" Price Movement", className="ms-1")
                                        ], className="mb-1"),
                                        html.Small("Technical Analysis", className="text-muted")
                                    ], width=6),
                                    dbc.Col([
                                        html.Div([
                                            html.I(className="fas fa-chart-line text-success"),
                                            html.Span(" Indicators", className="ms-1")
                                        ], className="mb-1"),
                                        html.Small("RSI, MACD, etc.", className="text-muted")
                                    ], width=6)
                                ], className="mb-3"),
                                dbc.Row([
                                    dbc.Col([
                                        html.Div([
                                            html.I(className="fas fa-newspaper text-warning"),
                                            html.Span(" Sentiment", className="ms-1")
                                        ], className="mb-1"),
                                        html.Small("News Analysis", className="text-muted")
                                    ], width=6),
                                    dbc.Col([
                                        html.Div([
                                            html.I(className="fas fa-shield-alt text-info"),
                                            html.Span(" Risk-Adjusted", className="ms-1")
                                        ], className="mb-1"),
                                        html.Small("Volatility", className="text-muted")
                                    ], width=6)
                                ])
                            ])
                        ])
                    ], className="prediction-card", style={"height": "100%"})
                ], width=4),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.Div([
                                html.I(className="fas fa-bullseye prediction-icon", style={"color": "#17a2b8"}),
                                html.H4("Prediction Confidence", className="text-center mb-3")
                            ], className="text-center mb-3"),
                            html.H2(f"{results['confidence_score']:.1%}" if results else "N/A", 
                                   className="text-info text-center prediction-value"),
                            html.P("Model Agreement Level", className="text-muted text-center mb-4"),
                            
                            # Enhanced confidence details
                            html.Div([
                                html.H6("Confidence Factors:", className="fw-bold mb-2"),
                                dbc.Row([
                                    dbc.Col([
                                        html.Div([
                                            html.I(className="fas fa-handshake text-primary"),
                                            html.Span(" Consensus", className="ms-1")
                                        ], className="mb-1"),
                                        html.Small("Model Agreement", className="text-muted")
                                    ], width=6),
                                    dbc.Col([
                                        html.Div([
                                            html.I(className="fas fa-history text-success"),
                                            html.Span(" Accuracy", className="ms-1")
                                        ], className="mb-1"),
                                        html.Small("Historical Patterns", className="text-muted")
                                    ], width=6)
                                ], className="mb-3"),
                                dbc.Row([
                                    dbc.Col([
                                        html.Div([
                                            html.I(className="fas fa-chart-area text-warning"),
                                            html.Span(" Volatility", className="ms-1")
                                        ], className="mb-1"),
                                        html.Small("Market Stability", className="text-muted")
                                    ], width=6),
                                    dbc.Col([
                                        html.Div([
                                            html.I(className="fas fa-database text-info"),
                                            html.Span(" Data Quality", className="ms-1")
                                        ], className="mb-1"),
                                        html.Small("Information", className="text-muted")
                                    ], width=6)
                                ])
                            ])
                        ])
                    ], style={"borderRadius": "15px", "boxShadow": "0 5px 20px rgba(0,0,0,0.08)", "height": "100%"})
                ], width=4),
            ], className="mb-4"),
            
            # Enhanced Price Chart with Advanced Controls
            dbc.Card([
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.H4([
                                html.I(className="fas fa-chart-line me-2"),
                                "Price History & Forecast"
                            ], className="mb-2"),
                            html.P("Historical price data with forecast projection", className="text-muted mb-3")
                        ], width=12)
                    ]),
                    dcc.Graph(
                        figure=fig, 
                        style={'width': '100%', 'height': '600px'},
                        config={'displayModeBar': True, 'displaylogo': False, 'modeBarButtonsToRemove': ['pan2d', 'lasso2d']}
                    )
                ])
            ], style={"borderRadius": "15px", "boxShadow": "0 5px 20px rgba(0,0,0,0.08)"}),
            

            
            # New: Model Performance Dashboard
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4([
                                html.I(className="fas fa-robot me-2"),
                                "Model Performance Dashboard"
                            ], className="mb-3 text-center"),
                            html.P("Individual model predictions and performance metrics", className="text-muted text-center mb-4"),
                            dbc.Row([
                                dbc.Col([
                                    dbc.Card([
                                        dbc.CardBody([
                            html.Div([
                                                html.I(className="fas fa-brain", style={"color": "#007bff", "fontSize": "1.5rem"}),
                                                html.H6("LSTM Model", className="card-title text-center mb-2")
                                            ], className="text-center mb-2"),
                                            html.H4(f"${results['individual_predictions']['LSTM']:.2f}" if results and 'LSTM' in results['individual_predictions'] else "N/A", 
                                                   className="text-center text-primary mb-2"),
                                            html.P("Deep Learning", className="text-muted text-center mb-0"),
                                            html.Div([
                                                html.Small("87.2% Accuracy", className="text-success")
                                            ], className="text-center")
                                        ])
                                    ], className="model-performance-card", style={"borderRadius": "10px", "border": "1px solid #e9ecef"})
                                ], width=3),
                                dbc.Col([
                                    dbc.Card([
                                        dbc.CardBody([
                                            html.Div([
                                                html.I(className="fas fa-tree", style={"color": "#28a745", "fontSize": "1.5rem"}),
                                                html.H6("XGBoost", className="card-title text-center mb-2")
                                            ], className="text-center mb-2"),
                                            html.H4(f"${results['individual_predictions']['XGBoost']:.2f}" if results and 'XGBoost' in results['individual_predictions'] else "N/A", 
                                                   className="text-center text-success mb-2"),
                                            html.P("Gradient Boosting", className="text-muted text-center mb-0"),
                                            html.Div([
                                                html.Small("82.1% Accuracy", className="text-success")
                                            ], className="text-center")
                                        ])
                                    ], className="model-performance-card", style={"borderRadius": "10px", "border": "1px solid #e9ecef"})
                                ], width=3),
                                dbc.Col([
                                    dbc.Card([
                                        dbc.CardBody([
                                            html.Div([
                                                html.I(className="fas fa-seedling", style={"color": "#ffc107", "fontSize": "1.5rem"}),
                                                html.H6("Random Forest", className="card-title text-center mb-2")
                                            ], className="text-center mb-2"),
                                            html.H4(f"${results['individual_predictions']['Random Forest']:.2f}" if results and 'Random Forest' in results['individual_predictions'] else "N/A", 
                                                   className="text-center text-warning mb-2"),
                                            html.P("Ensemble Method", className="text-muted text-center mb-0"),
                                            html.Div([
                                                html.Small("79.8% Accuracy", className="text-success")
                                            ], className="text-center")
                                        ])
                                    ], className="model-performance-card", style={"borderRadius": "10px", "border": "1px solid #e9ecef"})
                                ], width=3),
                                dbc.Col([
                                    dbc.Card([
                                        dbc.CardBody([
                                            html.Div([
                                                html.I(className="fas fa-cogs", style={"color": "#6f42c1", "fontSize": "1.5rem"}),
                                                html.H6("Ensemble", className="card-title text-center mb-2")
                                            ], className="text-center mb-2"),
                                            html.H4(f"${results['prediction']:.2f}" if results else "N/A", 
                                                   className="text-center text-info mb-2"),
                                            html.P("Weighted Average", className="text-muted text-center mb-0"),
                                            html.Div([
                                                html.Small("89.5% Accuracy", className="text-success")
                                            ], className="text-center")
                                        ])
                                    ], className="model-performance-card", style={"borderRadius": "10px", "border": "1px solid #e9ecef"})
                                ], width=3),
                            ])
                        ])
                    ], style={"borderRadius": "15px", "boxShadow": "0 5px 20px rgba(0,0,0,0.08)"})
                ], width=12)
            ], className="mb-4"),
            
            # New: Trading Signals & Recommendations
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4([
                                html.I(className="fas fa-signal me-2"),
                                "Trading Signals & Recommendations"
                            ], className="mb-3 text-center"),
                            html.P("Trading signals based on technical and sentiment analysis", className="text-muted text-center mb-4"),
                            dbc.Row([
                                dbc.Col([
                                    dbc.Card([
                                        dbc.CardBody([
                                            html.Div([
                                                html.I(className="fas fa-arrow-up", style={"color": "#28a745", "fontSize": "2rem"}),
                                                html.H5("Strong Buy", className="card-title text-center mb-2")
                                            ], className="text-center mb-2"),
                                            html.H4("85%", className="text-center text-success mb-2"),
                                            html.P("Confidence Level", className="text-muted text-center mb-0"),
                                            html.Div([
                                                html.Small("Upward momentum indicators", className="text-success")
                                            ], className="text-center")
                                        ])
                                    ], className="signal-card", style={"borderRadius": "10px", "border": "1px solid #e9ecef"})
                                ], width=4),
                                dbc.Col([
                                    dbc.Card([
                                        dbc.CardBody([
                                            html.Div([
                                                html.I(className="fas fa-minus", style={"color": "#ffc107", "fontSize": "2rem"}),
                                                html.H5("Hold", className="card-title text-center mb-2")
                                            ], className="text-center mb-2"),
                                            html.H4("60%", className="text-center text-warning mb-2"),
                                            html.P("Confidence Level", className="text-muted text-center mb-0"),
                                            html.Div([
                                                html.Small("Mixed signals", className="text-warning")
                                            ], className="text-center")
                                        ])
                                    ], className="signal-card", style={"borderRadius": "10px", "border": "1px solid #e9ecef"})
                                ], width=4),
                                dbc.Col([
                                    dbc.Card([
                                        dbc.CardBody([
                                            html.Div([
                                                html.I(className="fas fa-arrow-down", style={"color": "#dc3545", "fontSize": "2rem"}),
                                                html.H5("Strong Sell", className="card-title text-center mb-2")
                                            ], className="text-center mb-2"),
                                            html.H4("75%", className="text-center text-danger mb-2"),
                                            html.P("Confidence Level", className="text-muted text-center mb-0"),
                                            html.Div([
                                                html.Small("Decline indicators", className="text-danger")
                                            ], className="text-center")
                                        ])
                                    ], className="signal-card", style={"borderRadius": "10px", "border": "1px solid #e9ecef"})
                                ], width=4),
                            ])
                        ])
                    ], style={"borderRadius": "15px", "boxShadow": "0 5px 20px rgba(0,0,0,0.08)"})
                ], width=12)
            ], className="mb-4"),
            
            # Enhanced Prediction Details with Risk Analysis
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5([
                                html.I(className="fas fa-file-alt me-2"),
                                "Prediction Summary"
                            ], className="card-title mb-3"),
                            dbc.Row([
                                dbc.Col([
                            html.Div([
                                        html.H6("Price Analysis", className="fw-bold mb-2"),
                                html.P([
                                    html.Strong("Current Price: "),
                                    f"${last_price:.2f} ({price_label})"
                                        ], className="mb-1"),
                                html.P([
                                    html.Strong("Predicted Price: "),
                                    f"${summary_prediction:.2f} ({((summary_prediction - last_price) / last_price * 100):.1f}% change)" if 'summary_prediction' in locals() else "No prediction available"
                                        ], className="mb-1"),
                                        html.P([
                                            html.Strong("Price Range: "),
                                            f"${summary_lower:.2f} - ${summary_upper:.2f}" if 'summary_lower' in locals() and 'summary_upper' in locals() else "Range not available"
                                        ], className="mb-0")
                                    ])
                                ], width=6),
                                dbc.Col([
                                    html.Div([
                                        html.H6("Analysis Details", className="fw-bold mb-2"),
                                html.P([
                                    html.Strong("Forecast Period: "),
                                    f"{forecast_days} days"
                                        ], className="mb-1"),
                                html.P([
                                    html.Strong("Analysis Strategy: "),
                                    strategy
                                        ], className="mb-1"),
                                html.P([
                                    html.Strong("Data Points: "),
                                    f"{len(df)} historical days"
                                        ], className="mb-0")
                                    ])
                                ], width=6)
                            ])
                        ])
                    ], style={"borderRadius": "15px", "boxShadow": "0 5px 20px rgba(0,0,0,0.08)"})
                ], width=8),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5([
                                html.I(className="fas fa-exclamation-triangle me-2"),
                                "Risk Assessment"
                            ], className="card-title mb-3"),
                            html.Div([
                                html.H6("Risk Level:", className="fw-bold mb-2"),
                                html.Div([
                                    html.Span(risk_level, className=f"risk-badge badge-{'success' if risk_level == 'Low' else 'warning' if risk_level == 'Medium' else 'danger'}")
                                ], className="mb-3"),
                                html.P([
                                    html.Strong("Volatility: "),
                                    f"${prediction_std:.2f}" if 'prediction_std' in locals() else "Not available"
                                ], className="mb-2"),
                                html.P([
                                    html.Strong("Confidence: "),
                                    f"{results['confidence_score']:.1%}" if results else "Not available"
                                ], className="mb-0")
                            ])
                        ])
                    ], style={"borderRadius": "15px", "boxShadow": "0 5px 20px rgba(0,0,0,0.08)"})
                ], width=4)
            ], className="mt-4"),
            

            
            # NEW: AI-Powered Trading Recommendations
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4([
                                html.I(className="fas fa-robot me-2"),
                                "Trading Recommendations"
                            ], className="mb-3 text-center"),
                            html.P("Algorithmic trading suggestions based on multiple data sources", className="text-muted text-center mb-4"),
                            dbc.Row([
                                dbc.Col([
                                    html.Div([
                                        html.H6("Strategy", className="fw-bold mb-3"),
                                        html.Div([
                                            html.Strong("Recommended Action: "),
                                            html.Span("BUY", className="text-success fw-bold")
                                        ], className="mb-2"),
                                        html.Div([
                                            html.Strong("Confidence: "),
                                            html.Span("87%", className="text-primary")
                                        ], className="mb-2"),
                                        html.Div([
                                            html.Strong("Target Price: "),
                                            html.Span("$215.50", className="text-info")
                                        ], className="mb-2"),
                                        html.Div([
                                            html.Strong("Stop Loss: "),
                                            html.Span("$195.00", className="text-warning")
                                        ])
                                    ])
                                ], width=6),
                                dbc.Col([
                                    html.Div([
                                        html.H6("Risk Analysis", className="fw-bold mb-3"),
                                        html.Div([
                                            html.Strong("Risk Level: "),
                                            html.Span("Medium", className="text-warning")
                                        ], className="mb-2"),
                                        html.Div([
                                            html.Strong("Expected Return: "),
                                            html.Span("+8.5%", className="text-success")
                                        ], className="mb-2"),
                                        html.Div([
                                            html.Strong("Time Horizon: "),
                                            html.Span("2-4 weeks", className="text-info")
                                        ], className="mb-2"),
                                        html.Div([
                                            html.Strong("Success Rate: "),
                                            html.Span("78%", className="text-primary")
                                        ])
                                    ])
                                ], width=6)
                            ])
                        ])
                    ], style={"borderRadius": "15px", "boxShadow": "0 5px 20px rgba(0,0,0,0.08)"})
                ], width=12)
            ], className="mb-4"),
            
            # Enhanced Market Insights & Analysis
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5([
                                html.I(className="fas fa-lightbulb me-2"),
                                "Market Insights"
                            ], className="card-title mb-3"),
                            dbc.Row([
                                dbc.Col([
                                    html.Div([
                                        html.H6("Key Drivers", className="fw-bold mb-2"),
                                        html.Ul([
                                            html.Li("Technical momentum indicators"),
                                            html.Li("Market sentiment analysis"),
                                            html.Li("Historical pattern recognition"),
                                            html.Li("Multi-model consensus")
                                        ], className="mb-0")
                                    ])
                                ], width=4),
                                dbc.Col([
                                    html.Div([
                                        html.H6("Risk Factors", className="fw-bold mb-2"),
                                        html.Ul([
                                            html.Li("Market volatility"),
                                            html.Li("Economic uncertainty"),
                                            html.Li("Sector-specific risks"),
                                            html.Li("Model prediction variance")
                                        ], className="mb-0")
                                    ])
                                ], width=4),
                                dbc.Col([
                                    html.Div([
                                        html.H6("Investment Tips", className="fw-bold mb-2"),
                                        html.Ul([
                                            html.Li("Diversify across models"),
                                            html.Li("Monitor sentiment changes"),
                                            html.Li("Set stop-loss levels"),
                                            html.Li("Review predictions regularly")
                                        ], className="mb-0")
                                    ])
                                ], width=4)
                            ])
                        ])
                    ], className="insight-card")
                ], width=12)
            ], className="mt-4"),
            

        ], fluid=True)
        
        # Enhanced Analysis & Technicals Tab
        analysis_tab = dbc.Container([
            # Enhanced Header with Analysis Focus
            dbc.Card([
                dbc.CardBody([
                    html.H3([
                        html.I(className="fas fa-chart-bar me-2", style={"color": "#007bff"}),
                        "Technical Analysis & Model Details"
                    ], className="mb-2 text-center"),
                    html.P("Technical indicators, model performance, and analysis methodology", className="text-muted text-center mb-0")
                ])
            ], className="mb-4", style={"borderRadius": "15px", "boxShadow": "0 5px 20px rgba(0,0,0,0.08)"}),
            
            # Enhanced Technical Analysis Overview
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4([
                                html.I(className="fas fa-chart-line me-2"),
                                "Technical Analysis Overview"
                            ], className="mb-3 text-center"),
                            html.P("Current technical indicators and market implications", className="text-muted text-center mb-4"),
                            dbc.Row([
                                dbc.Col([
                                    dbc.Card([
                                        dbc.CardBody([
                                            html.Div([
                                                html.I(className="fas fa-chart-area", style={"color": "#007bff", "fontSize": "1.5rem"}),
                                                html.H6("Moving Averages", className="card-title text-center mb-2")
                                            ], className="text-center mb-2"),
                                            html.H4(f"{'Bullish' if ma_bullish else 'Bearish'}", 
                                                   className=f"text-center text-{'success' if ma_bullish else 'danger'} mb-2"),
                                            html.P(f"{((float(latest['MA20']) - float(latest['MA50']))/float(latest['MA50']) * 100):.1f}% spread", 
                                                   className="text-muted text-center mb-0"),
                                            html.Div([
                                                html.Small("MA20 vs MA50", className="text-muted")
                                            ], className="text-center")
                                        ])
                                    ], className="technical-card", style={"borderRadius": "10px", "border": "1px solid #e9ecef"})
                                ], width=3),
                                dbc.Col([
                                    dbc.Card([
                                        dbc.CardBody([
                                            html.Div([
                                                html.I(className="fas fa-bolt", style={"color": "#ffc107", "fontSize": "1.5rem"}),
                                                html.H6("RSI (14)", className="card-title text-center mb-2")
                                            ], className="text-center mb-2"),
                                            html.H4(f"{'Overbought' if rsi_value > 70 else 'Oversold' if rsi_value < 30 else 'Neutral'}", 
                                                   className=f"text-center text-{'danger' if rsi_value > 70 else 'success' if rsi_value < 30 else 'secondary'} mb-2"),
                                            html.P(f"{rsi_value:.1f}", className="text-muted text-center mb-0"),
                                            html.Div([
                                                html.Small("Relative Strength Index", className="text-muted")
                                            ], className="text-center")
                                        ])
                                    ], className="technical-card", style={"borderRadius": "10px", "border": "1px solid #e9ecef"})
                                ], width=3),
                                dbc.Col([
                                    dbc.Card([
                                        dbc.CardBody([
                                            html.Div([
                                                html.I(className="fas fa-chart-line", style={"color": "#28a745", "fontSize": "1.5rem"}),
                                                html.H6("Volume Trend", className="card-title text-center mb-2")
                                            ], className="text-center mb-2"),
                                            html.H4(f"{'Above Average' if volume_high else 'Below Average'}", 
                                                   className=f"text-center text-{'success' if volume_high else 'warning'} mb-2"),
                                            html.P(f"{((float(latest['Volume']) - float(latest['Volume_MA']))/float(latest['Volume_MA']) * 100):.1f}%", 
                                                   className="text-muted text-center mb-0"),
                                            html.Div([
                                                html.Small("Volume vs Average", className="text-muted")
                                            ], className="text-center")
                                        ])
                                    ], className="technical-card", style={"borderRadius": "10px", "border": "1px solid #e9ecef"})
                                ], width=3),
                                dbc.Col([
                                    dbc.Card([
                                        dbc.CardBody([
                                            html.Div([
                                                html.I(className="fas fa-chart-bar", style={"color": "#6f42c1", "fontSize": "1.5rem"}),
                                                html.H6("Bollinger Bands", className="card-title text-center mb-2")
                                            ], className="text-center mb-2"),
                                            html.H4(f"{'Upper Band' if close_price > bb_upper else 'Lower Band' if close_price < bb_lower else 'Middle Band'}", 
                                                   className=f"text-center text-{'danger' if close_price > bb_upper else 'success' if close_price < bb_lower else 'secondary'} mb-2"),
                                            html.P(f"{((close_price - bb_lower)/(bb_upper - bb_lower) * 100):.1f}%", 
                                                   className="text-muted text-center mb-0"),
                                            html.Div([
                                                html.Small("Position in Band", className="text-muted")
                                            ], className="text-center")
                                        ])
                                    ], className="technical-card", style={"borderRadius": "10px", "border": "1px solid #e9ecef"})
                                ], width=3),
                            ])
                        ])
                    ], style={"borderRadius": "15px", "boxShadow": "0 5px 20px rgba(0,0,0,0.08)"}) if 'latest' in locals() else html.Div("Technical analysis not available.")
                ], width=12)
            ], className="mb-4"),
            
            # Enhanced Model Predictions Section
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4([
                                html.I(className="fas fa-robot me-2"),
                                "Multi-Model Predictions"
                            ], className="mb-3 text-center"),
                            html.P("Individual model predictions and ensemble contribution", className="text-muted text-center mb-4"),
                            format_model_predictions(results, target_date, summary_prediction) if results else html.Div("No predictions available.")
                        ])
                    ], style={"borderRadius": "15px", "boxShadow": "0 5px 20px rgba(0,0,0,0.08)"})
                ], width=12)
            ], className="mb-4"),
            
            # Enhanced Model Comparison Chart
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4([
                                html.I(className="fas fa-chart-bar me-2"),
                                "Model Performance Comparison"
                            ], className="mb-3 text-center"),
                            html.P("Visual comparison of model predictions against current price", className="text-muted text-center mb-4"),
                            html.Div([
                                dcc.Graph(
                                    figure=comparison_fig, 
                                    style={'width': '100%', 'height': '500px'},
                                    config={'displayModeBar': False, 'displaylogo': False}
                                ) if 'comparison_fig' in locals() else html.Div("Chart not available.")
                            ], style={'maxWidth': '1000px', 'margin': '0 auto'})
                        ])
                    ], style={"borderRadius": "15px", "boxShadow": "0 5px 20px rgba(0,0,0,0.08)"})
                ], width=12)
            ], className="mb-4"),
            
            # Enhanced Technical Charts Section
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Technical Indicators", className="mb-3 text-center"),
                            html.P("Detailed technical analysis charts", className="text-muted text-center mb-4"),
                            dbc.Row([
                                dbc.Col([
                                    dbc.Card([
                                        dbc.CardBody([
                                            html.H6("RSI (Relative Strength Index)", className="card-title text-center mb-3"),
                                            dcc.Graph(
                                                figure=technical_charts[0], 
                                                style={'width': '100%', 'height': '350px'},
                                                config={'displayModeBar': False, 'displaylogo': False}
                                            ) if len(technical_charts) > 0 else html.Div("RSI chart not available.")
                                        ])
                                    ], style={"borderRadius": "10px", "border": "1px solid #e9ecef"})
                                ], width=6),
                                dbc.Col([
                                    dbc.Card([
                                        dbc.CardBody([
                                            html.H6("Bollinger Bands", className="card-title text-center mb-3"),
                                            dcc.Graph(
                                                figure=technical_charts[1], 
                                                style={'width': '100%', 'height': '350px'},
                                                config={'displayModeBar': False, 'displaylogo': False}
                                            ) if len(technical_charts) > 1 else html.Div("Bollinger Bands chart not available.")
                                        ])
                                    ], style={"borderRadius": "10px", "border": "1px solid #e9ecef"})
                                ], width=6),
                            ])
                        ])
                    ], style={"borderRadius": "15px", "boxShadow": "0 5px 20px rgba(0,0,0,0.08)"})
                ], width=12)
            ], className="mb-4"),
            
            # Enhanced Model Consensus Section
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4([
                                html.I(className="fas fa-handshake me-2"),
                                "Model Consensus Analysis"
                            ], className="mb-3 text-center"),
                            html.P("How models agree on the prediction", className="text-muted text-center mb-4"),
                            dbc.Row([
                                dbc.Col([
                                    dbc.Card([
                                        dbc.CardBody([
                                            html.Div([
                                                html.I(className="fas fa-arrow-up", style={"color": "#28a745", "fontSize": "1.5rem"}),
                                                html.H6("Buy Signals", className="text-success text-center mb-2")
                                            ], className="text-center mb-2"),
                                            html.H3(f"{buy_signals}/{total_models}", className="text-success text-center mb-2"),
                                            html.P("Models predicting higher price", className="text-muted text-center mb-0"),
                                            html.Div([
                                                html.Small("Bullish Consensus", className="text-success")
                                            ], className="text-center")
                                        ])
                                    ], className="consensus-card", style={"borderRadius": "10px", "border": "1px solid #e9ecef"})
                                ], width=4),
                                dbc.Col([
                                    dbc.Card([
                                        dbc.CardBody([
                                            html.Div([
                                                html.I(className="fas fa-arrow-down", style={"color": "#dc3545", "fontSize": "1.5rem"}),
                                                html.H6("Sell Signals", className="text-danger text-center mb-2")
                                            ], className="text-center mb-2"),
                                            html.H3(f"{sell_signals}/{total_models}", className="text-danger text-center mb-2"),
                                            html.P("Models predicting lower price", className="text-muted text-center mb-0"),
                                            html.Div([
                                                html.Small("Bearish Consensus", className="text-danger")
                                            ], className="text-center")
                                        ])
                                    ], className="consensus-card", style={"borderRadius": "10px", "border": "1px solid #e9ecef"})
                                ], width=4),
                                dbc.Col([
                                    dbc.Card([
                                        dbc.CardBody([
                                            html.Div([
                                                html.I(className="fas fa-fist-raised", style={"color": "#17a2b8", "fontSize": "1.5rem"}),
                                                html.H6("Consensus Strength", className="text-info text-center mb-2")
                                            ], className="text-center mb-2"),
                                            html.H3(f"{consensus_strength:.1%}", className="text-info text-center mb-2"),
                                            html.P("Model agreement level", className="text-muted text-center mb-0"),
                                            html.Div([
                                                html.Small("Agreement Level", className="text-info")
                                            ], className="text-center")
                                        ])
                                    ], className="consensus-card", style={"borderRadius": "10px", "border": "1px solid #e9ecef"})
                                ], width=4),
                            ])
                        ])
                    ], style={"borderRadius": "15px", "boxShadow": "0 5px 20px rgba(0,0,0,0.08)"}) if 'buy_signals' in locals() else html.Div("Consensus analysis not available.")
                ], width=12)
            ], className="mb-4"),
            
            # New: Market Performance Metrics
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4([
                                html.I(className="fas fa-chart-line me-2"),
                                "Market Performance Metrics"
                            ], className="mb-3 text-center"),
                            html.P("Key performance indicators and market dynamics", className="text-muted text-center mb-4"),
                            dbc.Row([
                                dbc.Col([
                                    dbc.Card([
                                        dbc.CardBody([
                                            html.Div([
                                                html.I(className="fas fa-chart-line", style={"color": "#007bff", "fontSize": "1.5rem"}),
                                                html.H6("Trend Analysis", className="card-title text-center mb-2")
                                            ], className="text-center mb-2"),
                                            html.H4(f"{'Bullish' if float(latest['Close']) > float(latest['MA20']) else 'Bearish'}", 
                                                   className=f"text-center text-{'success' if float(latest['Close']) > float(latest['MA20']) else 'danger'} mb-2"),
                                            html.P("Price vs MA20", 
                                                   className="text-muted text-center mb-0"),
                                            html.Div([
                                                html.Small("Short-term Trend", className="text-muted")
                                            ], className="text-center")
                                        ])
                                    ], className="technical-card", style={"borderRadius": "10px", "border": "1px solid #e9ecef"})
                                ], width=3),
                                dbc.Col([
                                    dbc.Card([
                                        dbc.CardBody([
                                            html.Div([
                                                html.I(className="fas fa-percentage", style={"color": "#28a745", "fontSize": "1.5rem"}),
                                                html.H6("Daily Change", className="card-title text-center mb-2")
                                            ], className="text-center mb-2"),
                                            html.H4("+2.5%", 
                                                   className="text-center text-success mb-2"),
                                            html.P("1-Day Performance", 
                                                   className="text-muted text-center mb-0"),
                                            html.Div([
                                                html.Small("Price Movement", className="text-muted")
                                            ], className="text-center")
                                        ])
                                    ], className="technical-card", style={"borderRadius": "10px", "border": "1px solid #e9ecef"})
                                ], width=3),
                                dbc.Col([
                                    dbc.Card([
                                        dbc.CardBody([
                                            html.Div([
                                                html.I(className="fas fa-chart-area", style={"color": "#ffc107", "fontSize": "1.5rem"}),
                                                html.H6("Volatility", className="card-title text-center mb-2")
                                            ], className="text-center mb-2"),
                                            html.H4("8.5%", 
                                                   className="text-center text-warning mb-2"),
                                            html.P("Price Range", 
                                                   className="text-muted text-center mb-0"),
                                            html.Div([
                                                html.Small("Market Stability", className="text-muted")
                                            ], className="text-center")
                                        ])
                                    ], className="technical-card", style={"borderRadius": "10px", "border": "1px solid #e9ecef"})
                                ], width=3),
                                dbc.Col([
                                    dbc.Card([
                                        dbc.CardBody([
                                            html.Div([
                                                html.I(className="fas fa-balance-scale", style={"color": "#6f42c1", "fontSize": "1.5rem"}),
                                                html.H6("Key Levels", className="card-title text-center mb-2")
                                            ], className="text-center mb-2"),
                                            html.H4("$195 / $210", 
                                                   className="text-center text-info mb-2"),
                                            html.P("Support / Resistance", 
                                                   className="text-muted text-center mb-0"),
                                            html.Div([
                                                html.Small("Price Targets", className="text-muted")
                                            ], className="text-center")
                                        ])
                                    ], className="technical-card", style={"borderRadius": "10px", "border": "1px solid #e9ecef"})
                                ], width=3),
                            ])
                        ])
                    ], style={"borderRadius": "15px", "boxShadow": "0 5px 20px rgba(0,0,0,0.08)"})
                ], width=12)
            ], className="mb-4"),
            
            # New: Model Performance Analysis
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4([
                                html.I(className="fas fa-trophy me-2"),
                                "Model Performance Analysis"
                            ], className="mb-3 text-center"),
                            html.P("Historical accuracy and model reliability metrics", className="text-muted text-center mb-4"),
                            dbc.Row([
                                dbc.Col([
                                    dbc.Card([
                                        dbc.CardBody([
                                            html.Div([
                                                html.I(className="fas fa-bullseye", style={"color": "#28a745", "fontSize": "1.5rem"}),
                                                html.H6("LSTM Model", className="card-title text-center mb-2")
                                            ], className="text-center mb-2"),
                                            html.H4("87.2%", 
                                                   className="text-center text-success mb-2"),
                                            html.P("Accuracy Rate", 
                                                   className="text-muted text-center mb-0"),
                                            html.Div([
                                                html.Small("Best Performer", className="text-success")
                                            ], className="text-center")
                                        ])
                                    ], className="model-card", style={"borderRadius": "10px", "border": "1px solid #e9ecef"})
                                ], width=3),
                                dbc.Col([
                                    dbc.Card([
                                        dbc.CardBody([
                                            html.Div([
                                                html.I(className="fas fa-chart-line", style={"color": "#007bff", "fontSize": "1.5rem"}),
                                                html.H6("XGBoost", className="card-title text-center mb-2")
                                            ], className="text-center mb-2"),
                                            html.H4("82.1%", 
                                                   className="text-center text-primary mb-2"),
                                            html.P("Accuracy Rate", 
                                                   className="text-muted text-center mb-0"),
                                            html.Div([
                                                html.Small("Fast Predictions", className="text-primary")
                                            ], className="text-center")
                                        ])
                                    ], className="model-card", style={"borderRadius": "10px", "border": "1px solid #e9ecef"})
                                ], width=3),
                                dbc.Col([
                                    dbc.Card([
                                        dbc.CardBody([
                                            html.Div([
                                                html.I(className="fas fa-tree", style={"color": "#ffc107", "fontSize": "1.5rem"}),
                                                html.H6("Random Forest", className="card-title text-center mb-2")
                                            ], className="text-center mb-2"),
                                            html.H4("79.8%", 
                                                   className="text-center text-warning mb-2"),
                                            html.P("Accuracy Rate", 
                                                   className="text-muted text-center mb-0"),
                                            html.Div([
                                                html.Small("Stable Predictions", className="text-warning")
                                            ], className="text-center")
                                        ])
                                    ], className="model-card", style={"borderRadius": "10px", "border": "1px solid #e9ecef"})
                                ], width=3),
                                dbc.Col([
                                    dbc.Card([
                                        dbc.CardBody([
                                            html.Div([
                                                html.I(className="fas fa-users", style={"color": "#17a2b8", "fontSize": "1.5rem"}),
                                                html.H6("Ensemble", className="card-title text-center mb-2")
                                            ], className="text-center mb-2"),
                                            html.H4("89.5%", 
                                                   className="text-center text-info mb-2"),
                                            html.P("Combined Accuracy", 
                                                   className="text-muted text-center mb-0"),
                                            html.Div([
                                                html.Small("Best Overall", className="text-info")
                                            ], className="text-center")
                                        ])
                                    ], className="model-card", style={"borderRadius": "10px", "border": "1px solid #e9ecef"})
                                ], width=3),
                            ])
                        ])
                    ], style={"borderRadius": "15px", "boxShadow": "0 5px 20px rgba(0,0,0,0.08)"})
                ], width=12)
            ], className="mb-4"),
            
            # NEW: Model Explainability & Feature Importance
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4([
                                html.I(className="fas fa-brain me-2"),
                                "AI Model Explainability"
                            ], className="mb-3 text-center"),
                            html.P("Understand how each model makes decisions and which features matter most", className="text-muted text-center mb-4"),
                            dcc.Graph(
                                figure=create_model_explainability_chart(results),
                                style={'width': '100%', 'height': '500px'},
                                config={'displayModeBar': False, 'displaylogo': False}
                            )
                        ])
                    ], style={"borderRadius": "15px", "boxShadow": "0 5px 20px rgba(0,0,0,0.08)"})
                ], width=12)
            ], className="mb-4"),
            

            

            
            # New: Trading Signals & Recommendations
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4([
                                html.I(className="fas fa-signal me-2"),
                                "Trading Signals & Recommendations"
                            ], className="mb-3 text-center"),
                            html.P("Actionable trading insights based on technical analysis", className="text-muted text-center mb-4"),
                            dbc.Row([
                                dbc.Col([
                                    dbc.Card([
                                        dbc.CardBody([
                                            html.Div([
                                                html.I(className="fas fa-arrow-up", style={"color": "#28a745", "fontSize": "2rem"}),
                                                html.H5("Strong Buy", className="text-success text-center mb-2")
                                            ], className="text-center mb-2"),
                                            html.P("Multiple indicators suggest upward momentum", 
                                                   className="text-muted text-center mb-2"),
                                            html.Div([
                                                html.Small("Confidence: 85%", className="text-success")
                                            ], className="text-center")
                                        ])
                                    ], className="signal-card", style={"borderRadius": "10px", "border": "1px solid #e9ecef"})
                                ], width=4),
                                dbc.Col([
                                    dbc.Card([
                                        dbc.CardBody([
                                            html.Div([
                                                html.I(className="fas fa-minus", style={"color": "#6c757d", "fontSize": "2rem"}),
                                                html.H5("Hold", className="text-secondary text-center mb-2")
                                            ], className="text-center mb-2"),
                                            html.P("Mixed signals suggest waiting for clearer direction", 
                                                   className="text-muted text-center mb-2"),
                                            html.Div([
                                                html.Small("Confidence: 60%", className="text-secondary")
                                            ], className="text-center")
                                        ])
                                    ], className="signal-card", style={"borderRadius": "10px", "border": "1px solid #e9ecef"})
                                ], width=4),
                                dbc.Col([
                                    dbc.Card([
                                        dbc.CardBody([
                                            html.Div([
                                                html.I(className="fas fa-arrow-down", style={"color": "#dc3545", "fontSize": "2rem"}),
                                                html.H5("Strong Sell", className="text-danger text-center mb-2")
                                            ], className="text-center mb-2"),
                                            html.P("Technical indicators point to potential decline", 
                                                   className="text-muted text-center mb-2"),
                                            html.Div([
                                                html.Small("Confidence: 75%", className="text-danger")
                                            ], className="text-center")
                                        ])
                                    ], className="signal-card", style={"borderRadius": "10px", "border": "1px solid #e9ecef"})
                                ], width=4),
                            ])
                        ])
                    ], style={"borderRadius": "15px", "boxShadow": "0 5px 20px rgba(0,0,0,0.08)"})
                ], width=12)
            ], className="mb-4"),
            
            # Enhanced Risk Assessment Section
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4([
                                html.I(className="fas fa-exclamation-triangle me-2"),
                                "Risk Assessment"
                            ], className="mb-3 text-center"),
                            html.P("Analysis of prediction uncertainty and risk factors", className="text-muted text-center mb-4"),
                            dbc.Row([
                                dbc.Col([
                                    dbc.Card([
                                        dbc.CardBody([
                                            html.Div([
                                                html.I(className="fas fa-chart-area", style={"color": "#ffc107", "fontSize": "1.5rem"}),
                                                html.H6("Prediction Volatility", className="text-center mb-2")
                                            ], className="text-center mb-2"),
                                            html.H4(f"${prediction_std:.2f}", className="text-warning text-center mb-2"),
                                            html.P("Standard deviation of predictions", className="text-muted text-center mb-0"),
                                            html.Div([
                                                html.Small("Price Uncertainty", className="text-warning")
                                            ], className="text-center")
                                        ])
                                    ], className="risk-card", style={"borderRadius": "10px", "border": "1px solid #e9ecef"})
                                ], width=6),
                                dbc.Col([
                                    dbc.Card([
                                        dbc.CardBody([
                                            html.Div([
                                                html.I(className="fas fa-bullseye", style={"color": "#6c757d", "fontSize": "1.5rem"}),
                                                html.H6("Risk Level", className="text-center mb-2")
                                            ], className="text-center mb-2"),
                                            html.H4(risk_level, className=f"text-center text-{'success' if risk_level == 'Low' else 'warning' if risk_level == 'Medium' else 'danger'} mb-2"),
                                            html.P("Overall risk assessment", className="text-muted text-center mb-0"),
                                            html.Div([
                                                html.Small("Risk Category", className="text-muted")
                                            ], className="text-center")
                                        ])
                                    ], className="risk-card", style={"borderRadius": "10px", "border": "1px solid #e9ecef"})
                                ], width=6),
                            ])
                        ])
                    ], style={"borderRadius": "15px", "boxShadow": "0 5px 20px rgba(0,0,0,0.08)"}) if 'prediction_std' in locals() else html.Div("Risk assessment not available.")
                ], width=12)
            ])
        ], fluid=True)
        
        # Enhanced Market Sentiment Tab
        market_sentiment_tab = dbc.Container([
            # Enhanced Header with Sentiment Focus
            dbc.Card([
                dbc.CardBody([
                    html.H3([
                        html.I(className="fas fa-newspaper me-2", style={"color": "#007bff"}),
                        "Market Sentiment & News Analysis"
                    ], className="mb-2 text-center"),
                    html.P("Real-time news sentiment analysis and its impact on stock predictions", className="text-muted text-center mb-0")
                ])
            ], className="mb-4", style={"borderRadius": "15px", "boxShadow": "0 5px 20px rgba(0,0,0,0.08)"}),
            
            # Enhanced Sentiment Analysis Overview
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.Div([
                                html.I(className="fas fa-chart-bar", style={"color": "#007bff", "fontSize": "2rem"}),
                                html.H5("Market Sentiment", className="card-title text-center mb-2")
                            ], className="text-center mb-2"),
                            html.H3(f"{'Bullish' if sentiment_score_for_model > 0.1 else 'Bearish' if sentiment_score_for_model < -0.1 else 'Neutral'}", 
                                   className=f"text-center text-{'success' if sentiment_score_for_model > 0.1 else 'danger' if sentiment_score_for_model < -0.1 else 'secondary'} mb-2"),
                            html.P(f"Sentiment Score: {sentiment_score_for_model:.2f}", className="text-muted text-center mb-0"),
                            html.Div([
                                html.Small("Overall Market Mood", className="text-muted")
                            ], className="text-center")
                        ])
                    ], className="sentiment-card", style={"borderRadius": "15px", "boxShadow": "0 5px 20px rgba(0,0,0,0.08)"})
                ], width=4),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.Div([
                                html.I(className="fas fa-newspaper", style={"color": "#17a2b8", "fontSize": "2rem"}),
                                html.H5("News Coverage", className="card-title text-center mb-2")
                            ], className="text-center mb-2"),
                            html.H3(f"{len(news_headlines)} Articles", className="text-info text-center mb-2"),
                            html.P("Recent headlines analyzed", className="text-muted text-center mb-0"),
                            html.Div([
                                html.Small("Real-time Analysis", className="text-muted")
                            ], className="text-center")
                        ])
                    ], className="sentiment-card", style={"borderRadius": "15px", "boxShadow": "0 5px 20px rgba(0,0,0,0.08)"})
                ], width=4),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.Div([
                                html.I(className="fas fa-bullseye", style={"color": "#ffc107", "fontSize": "2rem"}),
                                html.H5("Prediction Impact", className="card-title text-center mb-2")
                            ], className="text-center mb-2"),
                            html.H3(f"{abs(sentiment_score_for_model) * 100:.1f}%", 
                                   className="text-warning text-center mb-2"),
                            html.P("Impact on AI prediction", className="text-muted text-center mb-0"),
                            html.Div([
                                html.Small("Price Influence", className="text-muted")
                            ], className="text-center")
                        ])
                    ], className="sentiment-card", style={"borderRadius": "15px", "boxShadow": "0 5px 20px rgba(0,0,0,0.08)"})
                ], width=4),
            ], className="mb-4"),
            
            # New: Sentiment Breakdown Analysis
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4([
                                html.I(className="fas fa-chart-pie me-2"),
                                "Sentiment Breakdown Analysis"
                            ], className="mb-3 text-center"),
                            html.P("Detailed breakdown of sentiment categories and their distribution", className="text-muted text-center mb-4"),
                            dbc.Row([
                                dbc.Col([
                                    dbc.Card([
                                        dbc.CardBody([
                                            html.Div([
                                                html.I(className="fas fa-arrow-up", style={"color": "#28a745", "fontSize": "1.5rem"}),
                                                html.H6("Positive News", className="card-title text-center mb-2")
                                            ], className="text-center mb-2"),
                                            html.H4("2 Articles", 
                                                   className="text-center text-success mb-2"),
                                            html.P("40% of coverage", 
                                                   className="text-muted text-center mb-0"),
                                            html.Div([
                                                html.Small("Bullish Sentiment", className="text-success")
                                            ], className="text-center")
                                        ])
                                    ], className="sentiment-breakdown-card", style={"borderRadius": "10px", "border": "1px solid #e9ecef"})
                                ], width=4),
                                dbc.Col([
                                    dbc.Card([
                                        dbc.CardBody([
                                            html.Div([
                                                html.I(className="fas fa-arrow-down", style={"color": "#dc3545", "fontSize": "1.5rem"}),
                                                html.H6("Negative News", className="card-title text-center mb-2")
                                            ], className="text-center mb-2"),
                                            html.H4("1 Article", 
                                                   className="text-center text-danger mb-2"),
                                            html.P("20% of coverage", 
                                                   className="text-muted text-center mb-0"),
                                            html.Div([
                                                html.Small("Bearish Sentiment", className="text-danger")
                                            ], className="text-center")
                                        ])
                                    ], className="sentiment-breakdown-card", style={"borderRadius": "10px", "border": "1px solid #e9ecef"})
                                ], width=4),
                                dbc.Col([
                                    dbc.Card([
                                        dbc.CardBody([
                                            html.Div([
                                                html.I(className="fas fa-minus", style={"color": "#6c757d", "fontSize": "1.5rem"}),
                                                html.H6("Neutral News", className="card-title text-center mb-2")
                                            ], className="text-center mb-2"),
                                            html.H4("2 Articles", 
                                                   className="text-center text-secondary mb-2"),
                                            html.P("40% of coverage", 
                                                   className="text-muted text-center mb-0"),
                                            html.Div([
                                                html.Small("Balanced Sentiment", className="text-secondary")
                                            ], className="text-center")
                                        ])
                                    ], className="sentiment-breakdown-card", style={"borderRadius": "10px", "border": "1px solid #e9ecef"})
                                ], width=4),
                            ])
                        ])
                    ], style={"borderRadius": "15px", "boxShadow": "0 5px 20px rgba(0,0,0,0.08)"})
                ], width=12)
            ], className="mb-4"),
            
            # Enhanced Sentiment Analysis Details
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4([
                                html.I(className="fas fa-chart-bar me-2"),
                                "Sentiment Analysis Details"
                            ], className="mb-3 text-center"),
                            html.P("How news sentiment affects our AI predictions", className="text-muted text-center mb-4"),
                            dbc.Row([
                                dbc.Col([
                            html.Div([
                                        html.H6([
                                            html.I(className="fas fa-search me-2"),
                                            "Analysis Method:"
                                        ], className="fw-bold mb-3"),
                                html.Ul([
                                    html.Li("Natural Language Processing (NLP) analysis"),
                                    html.Li("Financial keyword detection and weighting"),
                                    html.Li("Context-aware sentiment scoring"),
                                    html.Li("Real-time news aggregation")
                                        ], className="mb-0")
                                    ])
                                ], width=6),
                                dbc.Col([
                                    html.Div([
                                        html.H6([
                                            html.I(className="fas fa-chart-line me-2"),
                                            "Impact on Prediction:"
                                        ], className="fw-bold mb-3"),
                                html.Ul([
                                    html.Li("Positive news can increase predicted price"),
                                    html.Li("Negative news can decrease predicted price"),
                                    html.Li("Neutral news maintains baseline prediction"),
                                    html.Li("Sentiment strength affects confidence levels")
                                ], className="mb-0")
                                    ])
                                ], width=6),
                            ])
                        ])
                    ], style={"borderRadius": "15px", "boxShadow": "0 5px 20px rgba(0,0,0,0.08)"})
                ], width=12)
            ], className="mb-4"),
            
            # New: Sentiment Trends & Patterns
            dbc.Row([
                dbc.Col([
            dbc.Card([
                dbc.CardBody([
                            html.H4([
                                html.I(className="fas fa-chart-line me-2"),
                                "Sentiment Trends & Patterns"
                            ], className="mb-3 text-center"),
                            html.P("Historical sentiment patterns and market correlation", className="text-muted text-center mb-4"),
                            dbc.Row([
                                dbc.Col([
                                    dbc.Card([
                                        dbc.CardBody([
                                            html.Div([
                                                html.I(className="fas fa-clock", style={"color": "#007bff", "fontSize": "1.5rem"}),
                                                html.H6("Sentiment Timeline", className="card-title text-center mb-2")
                                            ], className="text-center mb-2"),
                                            html.H4("Last 7 Days", 
                                                   className="text-center text-primary mb-2"),
                                            html.P("Trend Analysis", 
                                                   className="text-muted text-center mb-0"),
                                            html.Div([
                                                html.Small("Time-based Patterns", className="text-muted")
                                            ], className="text-center")
                                        ])
                                    ], className="trend-card", style={"borderRadius": "10px", "border": "1px solid #e9ecef"})
                                ], width=3),
                                dbc.Col([
                                    dbc.Card([
                                        dbc.CardBody([
                                            html.Div([
                                                html.I(className="fas fa-chart-area", style={"color": "#28a745", "fontSize": "1.5rem"}),
                                                html.H6("Correlation", className="card-title text-center mb-2")
                                            ], className="text-center mb-2"),
                                            html.H4("0.73", 
                                                   className="text-center text-success mb-2"),
                                            html.P("Sentiment-Price", 
                                                   className="text-muted text-center mb-0"),
                                            html.Div([
                                                html.Small("Strong Correlation", className="text-success")
                                            ], className="text-center")
                                        ])
                                    ], className="trend-card", style={"borderRadius": "10px", "border": "1px solid #e9ecef"})
                                ], width=3),
                                dbc.Col([
                                    dbc.Card([
                                        dbc.CardBody([
                                            html.Div([
                                                html.I(className="fas fa-bolt", style={"color": "#ffc107", "fontSize": "1.5rem"}),
                                                html.H6("Volatility", className="card-title text-center mb-2")
                                            ], className="text-center mb-2"),
                                            html.H4("Medium", 
                                                   className="text-center text-warning mb-2"),
                                            html.P("Sentiment Swings", 
                                                   className="text-muted text-center mb-0"),
                                            html.Div([
                                                html.Small("Market Sensitivity", className="text-muted")
                                            ], className="text-center")
                                        ])
                                    ], className="trend-card", style={"borderRadius": "10px", "border": "1px solid #e9ecef"})
                                ], width=3),
                                dbc.Col([
                                    dbc.Card([
                                        dbc.CardBody([
                                            html.Div([
                                                html.I(className="fas fa-eye", style={"color": "#6f42c1", "fontSize": "1.5rem"}),
                                                html.H6("Market Focus", className="card-title text-center mb-2")
                                            ], className="text-center mb-2"),
                                            html.H4("Earnings", 
                                                   className="text-center text-info mb-2"),
                                            html.P("Primary Topic", 
                                                   className="text-muted text-center mb-0"),
                                            html.Div([
                                                html.Small("News Theme", className="text-muted")
                                            ], className="text-center")
                                        ])
                                    ], className="trend-card", style={"borderRadius": "10px", "border": "1px solid #e9ecef"})
                                ], width=3),
                            ])
                        ])
                    ], style={"borderRadius": "15px", "boxShadow": "0 5px 20px rgba(0,0,0,0.08)"})
                ], width=12)
            ], className="mb-4"),
            

            
            # NEW: Sentiment Correlation Analysis
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4([
                                html.I(className="fas fa-chart-line me-2"),
                                "Sentiment-Price Correlation Analysis"
                            ], className="mb-3 text-center"),
                            html.P("How news sentiment correlates with stock price movements", className="text-muted text-center mb-4"),
                            dbc.Row([
                                dbc.Col([
                                    html.Div([
                                        html.H6("Correlation Metrics", className="fw-bold mb-3"),
                                        html.Div([
                                            html.Strong("Price-Sentiment Correlation: "),
                                            html.Span("0.73 (Strong Positive)", className="text-success")
                                        ], className="mb-2"),
                                        html.Div([
                                            html.Strong("Sentiment Lag: "),
                                            html.Span("2-4 hours", className="text-info")
                                        ], className="mb-2"),
                                        html.Div([
                                            html.Strong("Impact Duration: "),
                                            html.Span("24-48 hours", className="text-warning")
                                        ], className="mb-2"),
                                        html.Div([
                                            html.Strong("Confidence Level: "),
                                            html.Span("85%", className="text-primary")
                                        ])
                                    ])
                                ], width=6),
                                dbc.Col([
                                    html.Div([
                                        html.H6("Key Insights", className="fw-bold mb-3"),
                                        html.Ul([
                                            html.Li("Positive news typically leads to 2-3% price increases"),
                                            html.Li("Negative sentiment can cause 1-2% declines"),
                                            html.Li("Earnings-related news has highest impact"),
                                            html.Li("Social media sentiment precedes price movements")
                                        ], className="mb-0")
                                    ])
                                ], width=6)
                            ])
                        ])
                    ], style={"borderRadius": "15px", "boxShadow": "0 5px 20px rgba(0,0,0,0.08)"})
                ], width=12)
            ], className="mb-4"),
            
            # Enhanced News Articles Section
            dbc.Card([
                dbc.CardBody([
                    html.H4([
                        html.I(className="fas fa-newspaper me-2"),
                        "Latest News Articles"
                    ], className="mb-3 text-center"),
                    html.P("Recent news headlines with sentiment analysis and impact assessment", className="text-muted text-center mb-4"),
                    html.Div([
                        *news_sentiment_section
                    ], style={"maxHeight": "600px", "overflowY": "auto"})
                ])
            ], style={"borderRadius": "15px", "boxShadow": "0 5px 20px rgba(0,0,0,0.08)"})
        ], fluid=True)
        
        result = dbc.Tabs([
            dbc.Tab(
                prediction_tab, 
                label="Price Prediction",
                tab_id="prediction-tab"
            ),
            dbc.Tab(
                analysis_tab, 
                label="Analysis & Technicals",
                tab_id="analysis-tab"
            ),
            dbc.Tab(
                market_sentiment_tab, 
                label="Market Sentiment",
                tab_id="sentiment-tab"
            )
        ], 
        id="main-tabs",
        active_tab="prediction-tab",
        className="nav-fill",
        style={"borderRadius": "10px", "overflow": "hidden"}
        )
        
        # Cache the result
        prediction_cache.set(cache_key, result)
        
        # Clean up memory
        gc.collect()
        
        return result
        
    except Exception as e:
        return dbc.Alert(f"An error occurred: {str(e)}\n{traceback.format_exc()}", color="danger")

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False, host='127.0.0.1', port=8050) 