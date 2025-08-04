# --- WARNING SUPPRESSION (see below for details) ---
import os
import warnings
from contextlib import contextmanager
from typing import Dict, List, Tuple, Optional, Union, Any
import logging

# Environment and warning configuration
# Suppress TensorFlow warnings more aggressively
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU to prevent some warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Core data science imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

# Machine learning imports
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA

# Deep learning imports
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping

# External APIs and data sources
from newsapi import NewsApiClient
import yfinance as yf
from prophet import Prophet

# Visualization and analysis
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Natural language processing
from textblob import TextBlob
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import re

# Configure TensorFlow logging
tf.get_logger().setLevel('ERROR')

# Download required NLTK data with error handling
@contextmanager
def nltk_data_downloader():
    """Context manager for downloading NLTK data with proper error handling"""
    try:
        # Check if data already exists
        nltk.data.find('vader_lexicon')
        nltk.data.find('punkt')
        # If we get here, data exists, so just yield
        yield
    except LookupError:
        print("Downloading required NLTK data...")
        try:
            nltk.download('vader_lexicon', quiet=True)
            nltk.download('punkt', quiet=True)
            print("NLTK data download completed.")
        except Exception as e:
            print(f"Warning: Could not download NLTK data: {str(e)}")
        yield

# Initialize NLTK data once at module level
_nltk_initialized = False
if not _nltk_initialized:
    try:
        # Check if data already exists without downloading
        nltk.data.find('vader_lexicon')
        nltk.data.find('punkt')
        _nltk_initialized = True
    except LookupError:
        # Only download if not already present
        try:
            print("Downloading required NLTK data...")
            nltk.download('vader_lexicon', quiet=True)
            nltk.download('punkt', quiet=True)
            print("NLTK data download completed.")
            _nltk_initialized = True
        except Exception as e:
            print(f"Warning: Could not download NLTK data: {str(e)}")
            _nltk_initialized = True  # Mark as initialized to prevent repeated attempts
    except Exception as e:
        print(f"Warning: NLTK initialization failed: {str(e)}")
        _nltk_initialized = True  # Mark as initialized to prevent repeated attempts

# API configuration with environment variable support
NEWS_API_KEY = os.getenv('NEWS_API_KEY', '11f0c440d3bd415e811ca9a9b2b2c987')
newsapi = NewsApiClient(api_key=NEWS_API_KEY)

# Configuration classes
class ModelWeights(Enum):
    """Predefined weight configurations for different market conditions"""
    CONSERVATIVE = {
        'LSTM': 0.25,
        'SVR': 0.20,
        'Random Forest': 0.25,
        'XGBoost': 0.20,
        'GBM': 0.10
    }
    AGGRESSIVE = {
        'LSTM': 0.35,
        'SVR': 0.15,
        'Random Forest': 0.20,
        'XGBoost': 0.25,
        'GBM': 0.05
    }
    BALANCED = {
        'LSTM': 0.30,
        'SVR': 0.20,
        'Random Forest': 0.20,
        'XGBoost': 0.20,
        'GBM': 0.10
    }

@dataclass
class PredictionResult:
    """Structured result for prediction outputs"""
    prediction: float
    lower_bound: float
    upper_bound: float
    confidence_score: float
    individual_predictions: Dict[str, float]
    model_weights: Dict[str, float]
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

@dataclass
class SentimentAnalysis:
    """Structured sentiment analysis result"""
    sentiment: str
    confidence: float
    color: str
    score: float
    keywords_found: List[str] = None
    
    def __post_init__(self):
        if self.keywords_found is None:
            self.keywords_found = []

# Default weight configuration
WEIGHT_CONFIGURATIONS = {
    "Default": ModelWeights.BALANCED.value,
    "Conservative": ModelWeights.CONSERVATIVE.value,
    "Aggressive": ModelWeights.AGGRESSIVE.value
}

class DataFetcher:
    """Modern data fetching utility with caching and error handling"""
    
    def __init__(self, cache_duration: int = 300):
        self.cache_duration = cache_duration
        self._cache = {}
    
    def fetch_stock_data(self, symbol: str, days: int) -> pd.DataFrame:
        """Fetch stock data with intelligent caching and error handling"""
        cache_key = f"{symbol}_{days}"
        current_time = datetime.now()
        
        # Check cache first
        if cache_key in self._cache:
            cached_data, timestamp = self._cache[cache_key]
            if (current_time - timestamp).seconds < self.cache_duration:
                return cached_data
        
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            df = yf.download(symbol, start=start_date, end=end_date, progress=False)
            
            if df.empty:
                print(f"No data found for {symbol} in the last {days} days. Fetching maximum available data.")
                df = yf.download(symbol, period="max", progress=False)
            
            # Cache the result
            self._cache[cache_key] = (df, current_time)
            return df

        except Exception as e:
            print(f"Error fetching data for {symbol}: {str(e)}")
            return pd.DataFrame()

class NewsAnalyzer:
    """Enhanced news analysis with sentiment tracking"""
    
    def __init__(self, api_key: str):
        self.newsapi = NewsApiClient(api_key=api_key)
        self._sentiment_cache = {}
    
    def get_news_headlines(self, symbol: str, max_articles: int = 5) -> List[Tuple[str, str, str]]:
        """Fetch and process news headlines with intelligent filtering"""
        try:
            # Try to get company-specific news first
            news = self.newsapi.get_everything(
                q=f'"{symbol}" OR "{symbol} stock"',
                language='en',
                sort_by='relevancy',
                page_size=max_articles,
                from_param=(datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
            )
            
            if not news.get('articles'):
                # Fallback to broader search
                news = self.newsapi.get_everything(
                    q=symbol,
                    language='en',
                    sort_by='relevancy',
                    page_size=max_articles
                )
            
            articles = []
            for article in news.get('articles', []):
                title = article.get('title', '')
                description = article.get('description', '')
                url = article.get('url', '')
                
                # Filter out irrelevant articles
                if self._is_relevant_article(title, description, symbol):
                    articles.append((title, description, url))
            
            return articles[:max_articles]
            
        except Exception as e:
            print(f"News API error: {str(e)}")
            return []

    def _is_relevant_article(self, title: str, description: str, symbol: str) -> bool:
        """Filter articles for relevance to the stock"""
        text = f"{title} {description}".lower()
        
        # Keywords that indicate financial relevance
        financial_keywords = [
            'stock', 'price', 'market', 'trading', 'earnings', 'revenue',
            'profit', 'loss', 'quarterly', 'annual', 'dividend', 'shares',
            'investor', 'analyst', 'upgrade', 'downgrade', 'target'
        ]
        
        return any(keyword in text for keyword in financial_keywords)

class PriceMonitor:
    """Real-time price monitoring with market status detection"""
    
    def __init__(self):
        self._market_hours = {
            'start': '09:30',
            'end': '16:00',
            'timezone': 'America/New_York'
        }
    
    def get_current_price(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Fetch current live price with market status detection"""
        try:
            ticker = yf.Ticker(symbol)
            todays_data = ticker.history(period='1d')
            
            if todays_data.empty:
                return None
            
            # Get market status
            market_status = self._get_market_status()
            
            # Determine current price
            if market_status['is_open'] and 'regularMarketPrice' in ticker.info:
                current_price = ticker.info['regularMarketPrice']
                is_live = True
                price_type = "Live Market Price"
            else:
                current_price = float(todays_data['Close'].iloc[-1])
                is_live = False
                price_type = "Previous Close"
            
            return {
                "price": current_price,
                "is_live": is_live,
                "price_type": price_type,
                "market_status": market_status['status'],
                "last_updated": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "volume": ticker.info.get('volume', 0),
                "market_cap": ticker.info.get('marketCap', 0)
            }
            
        except Exception as e:
            print(f"Error fetching current price: {str(e)}")
            return None

    def _get_market_status(self) -> Dict[str, Any]:
        """Determine if US stock market is currently open"""
        now = datetime.now()
        current_time = now.strftime('%H:%M')
        
        # Simple market hours check (weekdays 9:30 AM - 4:00 PM EST)
        is_weekday = now.weekday() < 5  # Monday = 0, Friday = 4
        is_market_hours = '09:30' <= current_time <= '16:00'
        
        if is_weekday and is_market_hours:
            return {'is_open': True, 'status': 'Market Open'}
        elif is_weekday:
            return {'is_open': False, 'status': 'Market Closed'}
        else:
            return {'is_open': False, 'status': 'Weekend'}

# Initialize global instances
data_fetcher = DataFetcher()
news_analyzer = NewsAnalyzer(NEWS_API_KEY)
price_monitor = PriceMonitor()

# Legacy function wrappers for backward compatibility
def fetch_stock_data(symbol, days):
    return data_fetcher.fetch_stock_data(symbol, days)

def get_news_headlines(symbol):
    return news_analyzer.get_news_headlines(symbol)

def get_current_price(symbol):
    return price_monitor.get_current_price(symbol)

class SentimentAnalyzer:
    """Advanced sentiment analysis with financial context awareness"""
    
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()
        self._financial_keywords = self._initialize_financial_keywords()
        self._market_indicators = self._initialize_market_indicators()
    
    def _initialize_financial_keywords(self) -> Dict[str, Dict[str, float]]:
        """Initialize comprehensive financial keyword dictionary"""
        return {
            'positive': {
                'strong': 1.2, 'climbed': 1.3, 'up': 1.1, 'higher': 1.1,
                'beat': 1.2, 'exceeded': 1.2, 'growth': 1.1, 'profit': 1.1,
                'gain': 1.1, 'positive': 1.1, 'bullish': 1.3, 'outperform': 1.2,
                'buy': 1.1, 'upgrade': 1.2, 'recovers': 1.3, 'rose': 1.3,
                'closed higher': 1.4, 'rally': 1.3, 'surge': 1.4, 'jump': 1.3,
                'soar': 1.4, 'leap': 1.3, 'spike': 1.3, 'bounce': 1.2,
                'recovery': 1.3, 'rebound': 1.3, 'turnaround': 1.4, 'breakout': 1.4,
                'bull run': 1.5, 'green': 1.1, 'positive momentum': 1.4
            },
            'negative': {
                'weak': 1.2, 'fell': 1.3, 'down': 1.1, 'lower': 1.1,
                'miss': 1.2, 'missed': 1.2, 'decline': 1.1, 'loss': 1.1,
                'negative': 1.1, 'bearish': 1.3, 'underperform': 1.2,
                'sell': 1.1, 'downgrade': 1.2, 'sell-off': 1.4, 'rattled': 1.3,
                'correction': 1.3, 'crossed below': 1.4, 'pain': 1.3,
                'plunge': 1.4, 'crash': 1.5, 'tumble': 1.3, 'drop': 1.2,
                'slump': 1.3, 'dive': 1.4, 'sink': 1.3, 'collapse': 1.5,
                'bear market': 1.5, 'red': 1.1, 'negative momentum': 1.4,
                'breakdown': 1.4, 'support broken': 1.4
            }
        }
    
    def _initialize_market_indicators(self) -> Dict[str, Dict[str, float]]:
        """Initialize market-specific indicators"""
        return {
            'technical': {
                'moving average': 1.2, 'support': 1.1, 'resistance': 1.1,
                'rsi': 1.1, 'macd': 1.1, 'bollinger': 1.1, 'volume': 1.1,
                'momentum': 1.2, 'trend': 1.1, 'breakout': 1.3, 'breakdown': 1.3
            },
            'fundamental': {
                'earnings': 1.2, 'revenue': 1.2, 'profit': 1.2, 'loss': 1.2,
                'dividend': 1.1, 'p/e ratio': 1.1, 'market cap': 1.1,
                'quarterly': 1.2, 'annual': 1.2, 'guidance': 1.3
            }
        }
    
    def analyze_sentiment(self, text: str) -> SentimentAnalysis:
        """Enhanced sentiment analysis with financial context awareness"""
        # Handle empty or invalid text
        if not text or not isinstance(text, str):
            return SentimentAnalysis(
                sentiment="Neutral",
                confidence=0,
                color="gray",
                score=0
            )
        
        # Preprocess text
        cleaned_text = self._preprocess_text(text)
        
        # Get base sentiment scores
        vader_scores = self.sia.polarity_scores(cleaned_text)
        blob = TextBlob(cleaned_text)
        textblob_polarity = blob.sentiment.polarity
    
        # Calculate financial context score
        financial_score = self._calculate_financial_context(cleaned_text)
        
        # Calculate percentage impact
        percentage_score = self._calculate_percentage_impact(cleaned_text)
        
        # Calculate technical indicator impact
        technical_score = self._calculate_technical_impact(cleaned_text)
        
        # Combine all scores with weighted approach
        combined_score = self._combine_scores(
            vader_scores['compound'],
            textblob_polarity,
            financial_score,
            percentage_score,
            technical_score
        )
        
        # Determine sentiment category and confidence
        sentiment_result = self._determine_sentiment(combined_score)
        
        # Extract relevant keywords
        keywords = self._extract_relevant_keywords(cleaned_text)
        
        return SentimentAnalysis(
            sentiment=sentiment_result['sentiment'],
            confidence=sentiment_result['confidence'],
            color=sentiment_result['color'],
            score=combined_score,
            keywords_found=keywords
        )
    
    def _preprocess_text(self, text: str) -> str:
        """Clean and normalize text for analysis"""
        # Remove special characters but keep important punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\%\+\-]', '', text)
        # Normalize whitespace
        text = ' '.join(text.split())
        return text.lower()
    
    def _calculate_financial_context(self, text: str) -> float:
        """Calculate financial context score"""
        words = text.split()
        pos_score = sum(self._financial_keywords['positive'].get(word, 0) for word in words)
        neg_score = sum(self._financial_keywords['negative'].get(word, 0) for word in words)
        
        if pos_score + neg_score == 0:
            return 0
        
        return (pos_score - neg_score) / (pos_score + neg_score)
    
    def _calculate_percentage_impact(self, text: str) -> float:
        """Calculate impact of percentage changes mentioned"""
        percent_pattern = r'(\d+(?:\.\d+)?)\s*%'
        percentages = re.findall(percent_pattern, text)
        
        if not percentages:
            return 0
        
        total_impact = 0
        for pct in percentages:
            pct_value = float(pct)
            
            # Determine if percentage is positive or negative based on context
            if any(term in text for term in ["rose", "up", "climb", "gain", "higher", "increase"]):
                total_impact += pct_value * 0.15
            elif any(term in text for term in ["down", "fall", "drop", "lower", "decline", "decrease"]):
                total_impact -= pct_value * 0.15
            else:
                # Neutral context - smaller impact
                total_impact += pct_value * 0.05
        
        return total_impact
    
    def _calculate_technical_impact(self, text: str) -> float:
        """Calculate impact of technical analysis terms"""
        technical_score = 0
        
        # Moving average crossovers
        if "moving average" in text:
            if "crossed below" in text or "below" in text:
                technical_score -= 1.2
            elif "crossed above" in text or "above" in text:
                technical_score += 1.2
        
        # Market action terms
        if "sell-off" in text or "selloff" in text:
            technical_score -= 1.3
        if "recovery" in text or "recovers" in text:
            technical_score += 1.3
        
        # Support and resistance
        if "support broken" in text or "broke support" in text:
            technical_score -= 1.2
        if "resistance broken" in text or "broke resistance" in text:
            technical_score += 1.2
        
        return technical_score
    
    def _combine_scores(self, vader_score: float, textblob_score: float, 
                       financial_score: float, percentage_score: float, 
                       technical_score: float) -> float:
        """Combine all sentiment scores with appropriate weights"""
        return (
            vader_score * 0.25 +           # VADER sentiment
            textblob_score * 0.15 +         # TextBlob sentiment
            financial_score * 0.35 +        # Financial context (highest weight)
            percentage_score * 0.15 +       # Percentage changes
            technical_score * 0.10          # Technical indicators
        )
    
    def _determine_sentiment(self, combined_score: float) -> Dict[str, Any]:
        """Determine final sentiment category and confidence"""
        if combined_score >= 0.15:
            sentiment = "Positive"
            confidence = min(abs(combined_score) * 150, 100)
            color = "green"
        elif combined_score <= -0.15:
            sentiment = "Negative"
            confidence = min(abs(combined_score) * 150, 100)
            color = "red"
        else:
            sentiment = "Neutral"
            confidence = (1 - abs(combined_score)) * 100
            color = "gray"
        
        return {
            'sentiment': sentiment,
            'confidence': confidence,
            'color': color
        }
    
    def _extract_relevant_keywords(self, text: str) -> List[str]:
        """Extract relevant financial keywords from text"""
        words = text.split()
        relevant_keywords = []
        
        # Check for financial keywords
        for category in self._financial_keywords.values():
            for keyword in category.keys():
                if keyword in words:
                    relevant_keywords.append(keyword)
        
        # Check for technical indicators
        for category in self._market_indicators.values():
            for keyword in category.keys():
                if keyword in text:
                    relevant_keywords.append(keyword)
        
        return list(set(relevant_keywords))  # Remove duplicates

# Initialize global sentiment analyzer
sentiment_analyzer = SentimentAnalyzer()

# Legacy function wrapper
def analyze_sentiment(text):
    """Legacy wrapper that returns a dictionary for backward compatibility"""
    sentiment_result = sentiment_analyzer.analyze_sentiment(text)
    
    # Convert SentimentAnalysis object to dictionary for backward compatibility
    return {
        'sentiment': sentiment_result.sentiment,
        'confidence': sentiment_result.confidence,
        'color': sentiment_result.color,
        'score': sentiment_result.score,
        'keywords_found': sentiment_result.keywords_found
    }

class ProphetForecaster:
    """Advanced Prophet-based forecasting with enhanced features"""
    
    def __init__(self):
        self._model_cache = {}
        self._holiday_cache = {}
    
    def forecast_with_prophet(self, df: pd.DataFrame, forecast_days: int = 30, 
                            sentiment_score: float = 0.0) -> Optional[pd.DataFrame]:
        """Enhanced Prophet forecasting with comprehensive feature engineering"""
        try:
            # Validate input data
            if len(df) < 30:
                print("Not enough historical data for reliable forecasting (< 30 data points)")
                return self._simple_forecast_fallback(df, forecast_days)
            
            # Prepare data for Prophet
            prophet_df = self._prepare_prophet_data(df, sentiment_score)
            
            if prophet_df is None or len(prophet_df) < 20:
                print("Insufficient valid data after preparation")
                return self._simple_forecast_fallback(df, forecast_days)
            
            # Build and configure Prophet model
            model = self._build_prophet_model(prophet_df)
            
            # Add custom regressors and seasonality
            self._add_custom_regressors(model, prophet_df)
            self._add_custom_seasonality(model, prophet_df)
            
            # Fit the model
            model.fit(prophet_df)
            
            # Create future dataframe
            future = self._create_future_dataframe(prophet_df, forecast_days)
            
            # Make predictions
            forecast = model.predict(future)
            
            # Post-process predictions
            forecast = self._post_process_forecast(forecast, df)
            
            return forecast
            
        except Exception as e:
            print(f"Prophet model failed: {str(e)}. Using simple forecast instead.")
            return self._simple_forecast_fallback(df, forecast_days)
    
    def _prepare_prophet_data(self, df: pd.DataFrame, sentiment_score: float) -> Optional[pd.DataFrame]:
        """Prepare data for Prophet with comprehensive feature engineering"""
        try:
            df_copy = df.copy()
            df_copy = df_copy.reset_index()
            
            # Find date and close columns
            date_col = self._find_date_column(df_copy)
            close_col = self._find_close_column(df_copy)
            
            if date_col is None or close_col is None:
                return None
            
            # Create base Prophet dataframe
            prophet_df = pd.DataFrame({
                'ds': pd.to_datetime(df_copy[date_col]),
                'y': df_copy[close_col].astype(float),
                'sentiment': sentiment_score
            })
            
            # Add volume features if available
            if 'Volume' in df_copy.columns:
                prophet_df = self._add_volume_features(prophet_df, df_copy)
            
            # Add technical indicators
            prophet_df = self._add_technical_indicators(prophet_df)
            
            # Add market regime features
            prophet_df = self._add_market_regime_features(prophet_df)
            
            # Handle outliers
            prophet_df = self._handle_outliers(prophet_df)
            
            # Drop NaN values
            prophet_df = prophet_df.dropna()
            
            return prophet_df
            
        except Exception as e:
            print(f"Error preparing Prophet data: {str(e)}")
            return None
    
    def _find_date_column(self, df: pd.DataFrame) -> Optional[str]:
        """Find the date column in the dataframe"""
        for col in df.columns:
            col_str = col if isinstance(col, str) else col[0]
            if isinstance(col_str, str) and col_str.lower() in ['date', 'datetime', 'time', 'index']:
                return col
        return None
    
    def _find_close_column(self, df: pd.DataFrame) -> Optional[str]:
        """Find the close price column in the dataframe"""
        if isinstance(df.columns, pd.MultiIndex):
            for col in df.columns:
                if isinstance(col, tuple) and col[0] == 'Close':
                    return col
        else:
            if 'Close' in df.columns:
                return 'Close'
        return None
    
    def _add_volume_features(self, prophet_df: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based features"""
        prophet_df['volume'] = df['Volume'].astype(float)
        prophet_df['log_volume'] = np.log1p(prophet_df['volume'])
        prophet_df['volume_roc'] = prophet_df['volume'].pct_change(periods=5).fillna(0)
        prophet_df['volume_ma'] = prophet_df['volume'].rolling(window=20).mean().fillna(method='bfill')
        return prophet_df
        
    def _add_technical_indicators(self, prophet_df: pd.DataFrame) -> pd.DataFrame:
        """Add comprehensive technical indicators"""
        # Volatility measures
        prophet_df['volatility_5d'] = prophet_df['y'].rolling(window=5).std().fillna(0)
        prophet_df['volatility_10d'] = prophet_df['y'].rolling(window=10).std().fillna(0)
        prophet_df['volatility_20d'] = prophet_df['y'].rolling(window=20).std().fillna(0)
        
        # RSI
        delta = prophet_df['y'].diff()
        gain = delta.mask(delta < 0, 0).rolling(window=14).mean()
        loss = -delta.mask(delta > 0, 0).rolling(window=14).mean()
        rs = gain / loss
        prophet_df['rsi'] = 100 - (100 / (1 + rs)).fillna(50)
        
        # Momentum indicators
        prophet_df['momentum_5d'] = prophet_df['y'].pct_change(periods=5).fillna(0)
        prophet_df['momentum_10d'] = prophet_df['y'].pct_change(periods=10).fillna(0)
        
        # Moving averages
        prophet_df['ma10'] = prophet_df['y'].rolling(window=10).mean().fillna(method='bfill')
        prophet_df['ma20'] = prophet_df['y'].rolling(window=20).mean().fillna(method='bfill')
        prophet_df['ma10_dist'] = (prophet_df['y'] / prophet_df['ma10'] - 1)
        prophet_df['ma20_dist'] = (prophet_df['y'] / prophet_df['ma20'] - 1)
        
        # Bollinger bands
        bb_std = prophet_df['y'].rolling(window=20).std().fillna(0)
        prophet_df['bb_position'] = (prophet_df['y'] - prophet_df['ma20']) / (2 * bb_std + 1e-10)
        
        return prophet_df
    
    def _add_market_regime_features(self, prophet_df: pd.DataFrame) -> pd.DataFrame:
        """Add market regime detection features"""
        # Day of week effects
        prophet_df['day_of_week'] = prophet_df['ds'].dt.dayofweek
        for i in range(5):
            prophet_df[f'day_{i}'] = (prophet_df['day_of_week'] == i).astype(int)
        
        # Month start/end effects
        prophet_df['month_start'] = (prophet_df['ds'].dt.day <= 3).astype(int)
        prophet_df['month_end'] = (prophet_df['ds'].dt.day >= 28).astype(int)
        
        # Quarterly effects (earnings season)
        prophet_df['earnings_season'] = (
            (prophet_df['ds'].dt.month % 3 == 0) & 
            (prophet_df['ds'].dt.day >= 15) & 
            (prophet_df['ds'].dt.day <= 30)
        ).astype(int)
        
        return prophet_df
    
    def _handle_outliers(self, prophet_df: pd.DataFrame) -> pd.DataFrame:
        """Handle outliers using winsorization"""
        for col in prophet_df.columns:
            if col != 'ds' and prophet_df[col].dtype.kind in 'fc':
                q1 = prophet_df[col].quantile(0.01)
                q3 = prophet_df[col].quantile(0.99)
                prophet_df[col] = prophet_df[col].clip(q1, q3)
        return prophet_df
    
    def _build_prophet_model(self, prophet_df: pd.DataFrame) -> Prophet:
        """Build and configure Prophet model with adaptive parameters"""
        # Determine seasonality based on data size
        daily_seasonality = len(prophet_df) > 90
        yearly_seasonality = len(prophet_df) > 365
        
        # Adaptive parameters based on volatility
        recent_volatility = prophet_df['volatility_20d'].mean()
        avg_price = prophet_df['y'].mean()
        rel_volatility = recent_volatility / avg_price
        
        # Adjust changepoint prior scale based on volatility
        cp_prior_scale = min(0.05 + rel_volatility * 0.5, 0.5)  
        
        model = Prophet(
            daily_seasonality=daily_seasonality,
            weekly_seasonality=False,  # Disabled for stocks
            yearly_seasonality=yearly_seasonality,
            changepoint_prior_scale=cp_prior_scale,
            seasonality_prior_scale=10.0,
            seasonality_mode='multiplicative',
            changepoint_range=0.95,
            interval_width=0.9
        )
        
        # Add US holidays
        model.add_country_holidays(country_name='US')
        
        return model
    
    def _add_custom_regressors(self, model: Prophet, prophet_df: pd.DataFrame):
        """Add custom regressors to the model"""
        # Volume regressors
        if 'volume' in prophet_df.columns:
            model.add_regressor('log_volume', mode='multiplicative')
            model.add_regressor('volume_roc', mode='additive')
            
        # Technical indicators
        technical_regressors = [
            'volatility_5d', 'volatility_20d', 'rsi', 'momentum_5d', 
            'momentum_10d', 'ma10_dist', 'ma20_dist', 'bb_position'
        ]
        
        for regressor in technical_regressors:
            if regressor in prophet_df.columns:
                mode = 'multiplicative' if 'volatility' in regressor else 'additive'
                model.add_regressor(regressor, mode=mode)
        
        # Market regime regressors
        regime_regressors = ['month_start', 'month_end', 'earnings_season']
        for regressor in regime_regressors:
            if regressor in prophet_df.columns:
                model.add_regressor(regressor, mode='additive')
        
        # Sentiment regressor
        model.add_regressor('sentiment', mode='additive')
        
    def _add_custom_seasonality(self, model: Prophet, prophet_df: pd.DataFrame):
        """Add custom seasonality patterns"""
        if len(prophet_df) > 60:
            model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
            model.add_seasonality(name='quarterly', period=91.25, fourier_order=5)
            
    def _create_future_dataframe(self, prophet_df: pd.DataFrame, forecast_days: int) -> pd.DataFrame:
        """Create future dataframe with regressor values"""
        last_date = prophet_df['ds'].max()
        
        # Generate business days only
        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1), 
            periods=forecast_days * 1.4,
            freq='B'
        )[:forecast_days]
        
        future = pd.DataFrame({'ds': future_dates})
        
        # Add regressor values to future dataframe
        self._add_future_regressors(future, prophet_df)
        
        return future
    
    def _add_future_regressors(self, future: pd.DataFrame, prophet_df: pd.DataFrame):
        """Add regressor values to future dataframe"""
        # Volume regressors
        if 'volume' in prophet_df.columns:
            median_volume = prophet_df['volume'].tail(30).median()
            future['volume'] = median_volume
            future['log_volume'] = np.log1p(future['volume'])
            future['volume_roc'] = prophet_df['volume_roc'].tail(5).mean()
        
        # Technical indicators (use recent averages)
        technical_cols = ['volatility_5d', 'volatility_20d', 'rsi', 'momentum_5d', 
                         'momentum_10d', 'ma10_dist', 'ma20_dist', 'bb_position']
        
        for col in technical_cols:
            if col in prophet_df.columns:
                future[col] = prophet_df[col].tail(10).mean()
        
        # Market regime features
        if 'month_start' in prophet_df.columns:
            future['month_start'] = (future['ds'].dt.day <= 3).astype(int)
            future['month_end'] = (future['ds'].dt.day >= 28).astype(int)
            
        if 'earnings_season' in prophet_df.columns:
            future['earnings_season'] = (
                (future['ds'].dt.month % 3 == 0) & 
                (future['ds'].dt.day >= 15) & 
                (future['ds'].dt.day <= 30)
            ).astype(int)
        
        # Sentiment (use latest value)
        if 'sentiment' in prophet_df.columns:
            future['sentiment'] = prophet_df['sentiment'].iloc[-1]
    
    def _post_process_forecast(self, forecast: pd.DataFrame, original_df: pd.DataFrame) -> pd.DataFrame:
        """Post-process forecast for improved accuracy"""
        # Ensure non-negative prices
        forecast['yhat'] = np.maximum(forecast['yhat'], 0)
        forecast['yhat_lower'] = np.maximum(forecast['yhat_lower'], 0)
        
        # Apply uncertainty growth for longer forecasts
        if len(forecast) > 7:
            self._apply_uncertainty_growth(forecast, original_df)
        
        # Remove weekend forecasts
        forecast = forecast[forecast['ds'].dt.dayofweek < 5]
        
        return forecast
        
    def _apply_uncertainty_growth(self, forecast: pd.DataFrame, original_df: pd.DataFrame):
        """Apply exponential decay to prediction intervals"""
        last_historical_date = original_df.index[-1]
        future_dates = pd.to_datetime(forecast['ds']) > last_historical_date
        days_out = np.arange(1, sum(future_dates) + 1)
        uncertainty_multiplier = 1 + (np.sqrt(days_out) * 0.01)
        
        future_indices = np.where(future_dates)[0]
        for i, idx in enumerate(future_indices):
            forecast.loc[idx, 'yhat_upper'] = (
                forecast.loc[idx, 'yhat'] + 
                (forecast.loc[idx, 'yhat_upper'] - forecast.loc[idx, 'yhat']) * 
                uncertainty_multiplier[i]
            )
            forecast.loc[idx, 'yhat_lower'] = (
                forecast.loc[idx, 'yhat'] - 
                (forecast.loc[idx, 'yhat'] - forecast.loc[idx, 'yhat_lower']) * 
                uncertainty_multiplier[i]
            )
    
    def _simple_forecast_fallback(self, df: pd.DataFrame, forecast_days: int) -> Optional[pd.DataFrame]:
        """Simple linear regression fallback when Prophet fails"""
        try:
            close_prices = df['Close'].values.flatten()
            x = np.arange(len(close_prices)).reshape(-1, 1)
            y = close_prices
            
            model = LinearRegression()
            model.fit(x, y)
            
            # Generate business days
            last_date = df.index[-1]
            future_dates = pd.date_range(
                start=last_date + pd.Timedelta(days=1), 
                periods=forecast_days * 1.4,
                freq='B'
            )[:forecast_days]
            
            # Predict future values
            future_x = np.arange(len(close_prices), len(close_prices) + len(future_dates)).reshape(-1, 1)
            future_y = model.predict(future_x)
            
            # Calculate confidence interval
            historical_y = model.predict(x)
            mse = np.mean((y - historical_y) ** 2)
            sigma = np.sqrt(mse)
            
            # Create forecast dataframe
            forecast = pd.DataFrame({
                'ds': future_dates,
                'yhat': future_y,
                'yhat_lower': future_y - 1.96 * sigma,
                'yhat_upper': future_y + 1.96 * sigma,
                'trend': future_y,
                'weekly': np.zeros(len(future_y)),
                'yearly': np.zeros(len(future_y))
            })
            
            return forecast
            
        except Exception as e:
            print(f"Simple forecast also failed: {str(e)}")
            return None

# Initialize global forecaster
prophet_forecaster = ProphetForecaster()

# Legacy function wrapper
def forecast_with_prophet(df, forecast_days=30, sentiment_score=0.0):
    return prophet_forecaster.forecast_with_prophet(df, forecast_days, sentiment_score)

def simple_forecast_fallback(df, forecast_days=30):
    return prophet_forecaster._simple_forecast_fallback(df, forecast_days)

def calculate_technical_indicators_for_summary(df):
    analysis_df = df.copy()
    
    # Calculate Moving Averages
    analysis_df['MA20'] = analysis_df['Close'].rolling(window=20).mean().values
    analysis_df['MA50'] = analysis_df['Close'].rolling(window=50).mean().values
    
    # Calculate RSI
    delta = analysis_df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    analysis_df['RSI'] = 100 - (100 / (1 + rs))
    
    # Calculate Volume MA
    analysis_df['Volume_MA'] = analysis_df['Volume'].rolling(window=20).mean()
    
    # Calculate Bollinger Bands
    ma20 = analysis_df['Close'].rolling(window=20).mean()
    std20 = analysis_df['Close'].rolling(window=20).std()
    analysis_df['BB_upper'] = ma20 + (std20 * 2)
    analysis_df['BB_lower'] = ma20 - (std20 * 2)
    analysis_df['BB_middle'] = ma20
    
    return analysis_df

class TechnicalAnalyzer:
    """Advanced technical analysis with comprehensive indicators"""
    
    def __init__(self):
        self._indicator_cache = {}
    
    def calculate_technical_indicators_for_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive technical indicators for analysis"""
        analysis_df = df.copy()
        
        # Moving Averages
        analysis_df['MA20'] = analysis_df['Close'].rolling(window=20).mean()
        analysis_df['MA50'] = analysis_df['Close'].rolling(window=50).mean()
        analysis_df['MA200'] = analysis_df['Close'].rolling(window=200).mean()
        
        # RSI
        analysis_df['RSI'] = self._calculate_rsi(analysis_df['Close'])
        
        # Volume Analysis
        analysis_df['Volume_MA'] = analysis_df['Volume'].rolling(window=20).mean()
        # Fix the Volume_Ratio calculation to avoid DataFrame assignment issue
        volume_ma = analysis_df['Volume'].rolling(window=20).mean()
        analysis_df['Volume_Ratio'] = analysis_df['Volume'].div(volume_ma, fill_value=0)
        
        # Bollinger Bands
        ma20 = analysis_df['Close'].rolling(window=20).mean()
        std20 = analysis_df['Close'].rolling(window=20).std()
        analysis_df['BB_upper'] = ma20 + (std20 * 2)
        analysis_df['BB_lower'] = ma20 - (std20 * 2)
        analysis_df['BB_middle'] = ma20
        # Fix the BB_position calculation to avoid DataFrame assignment issue
        bb_upper = ma20 + (std20 * 2)
        bb_lower = ma20 - (std20 * 2)
        bb_range = bb_upper - bb_lower
        analysis_df['BB_position'] = (analysis_df['Close'] - bb_lower).div(bb_range, fill_value=0)
        
        # MACD
        analysis_df['MACD'] = self._calculate_macd(analysis_df['Close'])
        analysis_df['MACD_signal'] = analysis_df['MACD'].ewm(span=9).mean()
        analysis_df['MACD_histogram'] = analysis_df['MACD'] - analysis_df['MACD_signal']
        
        # Stochastic Oscillator
        analysis_df['STOCH_K'] = self._calculate_stochastic(analysis_df)
        analysis_df['STOCH_D'] = analysis_df['STOCH_K'].rolling(window=3).mean()
        
        # Williams %R
        analysis_df['WILLR'] = self._calculate_williams_r(analysis_df)
        
        # Average True Range (ATR)
        analysis_df['ATR'] = self._calculate_atr(analysis_df)
        
        # Price Channels
        analysis_df['Highest_High'] = analysis_df['High'].rolling(window=20).max()
        analysis_df['Lowest_Low'] = analysis_df['Low'].rolling(window=20).min()
        # Fix the Price_Channel_Position calculation to avoid DataFrame assignment issue
        highest_high = analysis_df['High'].rolling(window=20).max()
        lowest_low = analysis_df['Low'].rolling(window=20).min()
        channel_range = highest_high - lowest_low
        analysis_df['Price_Channel_Position'] = (analysis_df['Close'] - lowest_low).div(channel_range, fill_value=0)
        
        return analysis_df
    
    @staticmethod
    def _calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def _calculate_macd(prices: pd.Series, slow: int = 26, fast: int = 12) -> pd.Series:
        """Calculate MACD"""
        exp1 = prices.ewm(span=fast, adjust=False).mean()
        exp2 = prices.ewm(span=slow, adjust=False).mean()
        return exp1 - exp2
    
    @staticmethod
    def _calculate_stochastic(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Stochastic Oscillator %K"""
        low_min = df['Low'].rolling(window=period).min()
        high_max = df['High'].rolling(window=period).max()
        return 100 * ((df['Close'] - low_min) / (high_max - low_min))
    
    @staticmethod
    def _calculate_williams_r(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Williams %R"""
        high_max = df['High'].rolling(window=period).max()
        low_min = df['Low'].rolling(window=period).min()
        return -100 * ((high_max - df['Close']) / (high_max - low_min))
    
    @staticmethod
    def _calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        return true_range.rolling(period).mean()
    
class MultiAlgorithmStockPredictor:
    """Enhanced multi-algorithm stock predictor with modern architecture"""
    
    def __init__(self, symbol: str, training_years: int = 2, weights: Optional[Dict[str, float]] = None):
        self.symbol = symbol
        self.training_years = training_years
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.weights = weights if weights is not None else WEIGHT_CONFIGURATIONS["Default"]
        self.technical_analyzer = TechnicalAnalyzer()
        self._model_cache = {}
        self._prediction_cache = {}
    
    def fetch_historical_data(self) -> pd.DataFrame:
        """Fetch historical data with intelligent fallback"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365 * self.training_years)
        
        try:
            df = yf.download(self.symbol, start=start_date, end=end_date, progress=False)
            if df.empty:
                print(f"Data for the last {self.training_years} years is unavailable. Fetching maximum available data.")
                df = yf.download(self.symbol, period="max", progress=False)
            return df
        except Exception as e:
            print(f"Error fetching data: {str(e)}")
            return yf.download(self.symbol, period="max", progress=False)
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive technical indicators"""
        # Basic moving averages
        df['MA5'] = df['Close'].rolling(window=5).mean()
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['MA50'] = df['Close'].rolling(window=50).mean()
        df['MA200'] = df['Close'].rolling(window=200).mean()
        
        # Momentum indicators
        df['RSI'] = self._calculate_rsi(df['Close'])
        df['MACD'] = self._calculate_macd(df['Close'])
        df['ROC'] = df['Close'].pct_change(periods=10) * 100
        
        # Volatility indicators
        df['ATR'] = self._calculate_atr(df)
        df['BB_upper'], df['BB_lower'] = self._calculate_bollinger_bands(df['Close'])
        
        # Volume indicators
        df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Rate'] = df['Volume'] / df['Volume'].rolling(window=20).mean()
        
        # Advanced indicators
        df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
        df['MOM'] = df['Close'].diff(10)
        df['STOCH_K'] = self._calculate_stochastic(df)
        df['WILLR'] = self._calculate_williams_r(df)
        
        return df.dropna()
    
    def prepare_data(self, df: pd.DataFrame, seq_length: int = 60, sentiment_score: float = 0.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """Enhanced data preparation with comprehensive feature engineering"""
        # Core feature columns
        feature_columns = [
            'Close', 'MA5', 'MA20', 'MA50', 'MA200', 'RSI', 'MACD', 'ROC', 'ATR',
            'BB_upper', 'BB_lower', 'Volume_Rate', 'EMA12', 'EMA26', 'MOM',
            'STOCH_K', 'WILLR'
        ]
        
        # Add derivative features
        df['Price_Momentum'] = df['Close'].pct_change(5)
        df['MA_Crossover'] = (df['MA5'] > df['MA20']).astype(int)
        df['RSI_Momentum'] = df['RSI'].diff(3)
        df['MACD_Signal'] = df['MACD'] - df['MACD'].ewm(span=9).mean()
        df['Volume_Shock'] = ((df['Volume'] - df['Volume'].shift(1)) / df['Volume'].shift(1)).clip(-1, 1)
        
        # Market regime detection
        df['ADX'] = self._calculate_adx(df)
        df['Is_Trending'] = (df['ADX'] > 25).astype(int)
        
        # Volatility features
        df['Volatility_20d'] = df['Close'].pct_change().rolling(window=20).std() * np.sqrt(252)
        
        # Time-based features
        df['DayOfWeek'] = df.index.dayofweek
        for i in range(5):
            df[f'Day_{i}'] = (df['DayOfWeek'] == i).astype(int)
        
        # Sentiment feature
        df['Sentiment'] = sentiment_score
        
        # Handle outliers
        df = self._handle_outliers(df)
        
        # Select final features
        enhanced_features = feature_columns + [
            'Price_Momentum', 'MA_Crossover', 'RSI_Momentum', 'MACD_Signal',
            'Volume_Shock', 'ADX', 'Is_Trending', 'Volatility_20d',
            'Day_0', 'Day_1', 'Day_2', 'Day_3', 'Day_4', 'Sentiment'
        ]
        
        # Ensure all features exist
        available_features = [col for col in enhanced_features if col in df.columns]
        df_cleaned = df[available_features].copy()
        df_cleaned = df_cleaned.dropna()
        
        # Scale features
        scaled_data = self.scaler.fit_transform(df_cleaned)
        
        # Prepare sequences for LSTM
        X_lstm, y = [], []
        for i in range(seq_length, len(scaled_data)):
            X_lstm.append(scaled_data[i-seq_length:i])
            y.append(scaled_data[i, 0])  # Close price is first column
            
        # Prepare data for other models
        X_other = scaled_data[seq_length:]
        
        return np.array(X_lstm), X_other, np.array(y), df_cleaned.columns.tolist()
    
    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle outliers using winsorization"""
        for col in df.columns:
            if col != 'DayOfWeek' and df[col].dtype in [np.float64, np.int64]:
                q1 = df[col].quantile(0.01)
                q3 = df[col].quantile(0.99)
                df[col] = df[col].clip(q1, q3)
        return df
    
    def build_lstm_model(self, input_shape: Tuple[int, int]) -> Sequential:
        """Build optimized LSTM model"""
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='huber', metrics=['mae'])
        return model

    def train_arima(self, df: pd.DataFrame):
        """Train ARIMA model with auto-optimization"""
        try:
            from pmdarima import auto_arima
            model = auto_arima(
                df['Close'], 
                start_p=1, start_q=1,
                max_p=5, max_q=5,
                d=1, seasonal=False,
                stepwise=True,
                suppress_warnings=True,
                error_action='ignore',
                max_order=5
            )
            return model
        except:
            # Fallback to standard ARIMA
            model = ARIMA(df['Close'], order=(5,1,0))
            return model.fit()

    def predict_with_all_models(self, prediction_days: int = 30, sequence_length: int = 30, 
                               sentiment_score: float = 0.0) -> Optional[Dict[str, Any]]:
        """Enhanced prediction with all models - returns dictionary for backward compatibility"""
        try:
            # Fetch and prepare data
            df = self.fetch_historical_data()
            
            if len(df) < sequence_length + 20:
                print(f"Insufficient historical data. Need at least {sequence_length + 20} days.")
                sequence_length = max(10, len(df) - 20)
                
            # Calculate technical indicators
            df = self.calculate_technical_indicators(df)
            
            if df.isnull().any().any():
                df = df.ffill().bfill()
                
            if len(df.dropna()) < sequence_length:
                print("Insufficient valid data after calculating indicators.")
                return None
                
            # Prepare data
            X_lstm, X_other, y, feature_names = self.prepare_data(df, sequence_length, sentiment_score)
            
            if len(X_lstm) == 0 or len(y) == 0:
                print("Could not create valid sequences for prediction.")
                return None
                
            # Split data
            split_idx = int(len(X_lstm) * 0.8)
            X_lstm_train, X_lstm_test = X_lstm[:split_idx], X_lstm[split_idx:]
            X_other_train, X_other_test = X_other[:split_idx], X_other[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]

            predictions = {}
            
            # Train and predict with LSTM
            lstm_model = self.build_lstm_model((sequence_length, X_lstm.shape[2]))
            early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
            lstm_model.fit(
                X_lstm_train, y_train, 
                epochs=20, batch_size=32,
                validation_data=(X_lstm_test, y_test),
                callbacks=[early_stopping], verbose=0
            )
            lstm_pred = lstm_model.predict(X_lstm_test[-1:], verbose=0)[0][0]
            predictions['LSTM'] = lstm_pred

            # Train and predict with SVR
            svr_model = SVR(kernel='rbf', C=100, epsilon=0.1)
            svr_model.fit(X_other_train, y_train)
            svr_pred = svr_model.predict(X_other_test[-1:])
            predictions['SVR'] = svr_pred[0]

            # Train and predict with Random Forest
            rf_model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
            rf_model.fit(X_other_train, y_train)
            rf_pred = rf_model.predict(X_other_test[-1:])
            predictions['Random Forest'] = rf_pred[0]

            # Train and predict with XGBoost
            xgb_model = XGBRegressor(objective='reg:squarederror', random_state=42, n_estimators=50)
            xgb_model.fit(X_other_train, y_train)
            xgb_pred = xgb_model.predict(X_other_test[-1:])
            predictions['XGBoost'] = xgb_pred[0]

            # Train GBM if we have enough data
            if len(X_other_train) > 100:
                gbm_model = GradientBoostingRegressor(random_state=42, n_estimators=50)
                gbm_model.fit(X_other_train, y_train)
                gbm_pred = gbm_model.predict(X_other_test[-1:])
                predictions['GBM'] = gbm_pred[0]

            # Train ARIMA if we have few other models
            if len(predictions) < 3:
                try:
                    close_prices = df['Close'].values
                    arima_model = ARIMA(close_prices, order=(2,1,0))
                    arima_fit = arima_model.fit()
                    arima_pred = arima_fit.forecast(steps=1)[0]
                    arima_scaled = (arima_pred - df['Close'].mean()) / df['Close'].std()
                    predictions['ARIMA'] = arima_scaled
                except Exception as e:
                    print(f"ARIMA prediction failed: {str(e)}")

            # Calculate ensemble prediction
            weights = self.weights
            available_models = list(predictions.keys())
            total_weight = sum(weights.get(model, 0.1) for model in available_models)
            adjusted_weights = {model: weights.get(model, 0.1)/total_weight for model in available_models}

            ensemble_pred = sum(pred * adjusted_weights[model] for model, pred in predictions.items())
            
            # Inverse transform predictions
            dummy_array = np.zeros((1, X_other.shape[1]))
            dummy_array[0, 0] = ensemble_pred
            final_prediction = self.scaler.inverse_transform(dummy_array)[0, 0]

            # Calculate prediction range
            individual_predictions = {}
            for model, pred in predictions.items():
                dummy = dummy_array.copy()
                dummy[0, 0] = pred
                individual_predictions[model] = self.scaler.inverse_transform(dummy)[0, 0]
            
            std_dev = np.std(list(individual_predictions.values()))
            
            # Return dictionary for backward compatibility
            return {
                'prediction': final_prediction,
                'lower_bound': final_prediction - std_dev,
                'upper_bound': final_prediction + std_dev,
                'confidence_score': 1 / (1 + std_dev / final_prediction),
                'individual_predictions': individual_predictions,
                'model_weights': adjusted_weights,
                'timestamp': datetime.now()
            }

        except Exception as e:
            print(f"Error in prediction: {str(e)}")
            return None

    # Technical indicator calculation methods (static methods for compatibility)
    @staticmethod
    def calculate_stochastic(df, period=14):
        return TechnicalAnalyzer._calculate_stochastic(df, period)
    
    @staticmethod
    def calculate_williams_r(df, period=14):
        return TechnicalAnalyzer._calculate_williams_r(df, period)
    
    @staticmethod
    def calculate_rsi(prices, period=14):
        return TechnicalAnalyzer._calculate_rsi(prices, period)
    
    @staticmethod
    def calculate_macd(prices, slow=26, fast=12, signal=9):
        return TechnicalAnalyzer._calculate_macd(prices, slow, fast)
    
    @staticmethod
    def calculate_atr(df, period=14):
        return TechnicalAnalyzer._calculate_atr(df, period)
    
    @staticmethod
    def calculate_bollinger_bands(prices, period=20, std_dev=2):
        ma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper_band = ma + (std * std_dev)
        lower_band = ma - (std * std_dev)
        return upper_band, lower_band
    
    @staticmethod
    def calculate_adx(df, period=14):
        """Calculate Average Directional Index (ADX)"""
        try:
            high_low = df['High'] - df['Low']
            high_close = abs(df['High'] - df['Close'].shift())
            low_close = abs(df['Low'] - df['Close'].shift())
            
            ranges = pd.DataFrame({'hl': high_low, 'hc': high_close, 'lc': low_close})
            tr = ranges.max(axis=1)
            atr = tr.rolling(period).mean()
            
            plus_dm = df['High'].diff()
            minus_dm = df['Low'].diff()
            
            plus_dm_mask = (plus_dm > 0) & (plus_dm > minus_dm.abs())
            plus_dm = plus_dm.where(plus_dm_mask, 0)
            
            minus_dm_mask = (minus_dm < 0) & (minus_dm.abs() > plus_dm)
            minus_dm = minus_dm.abs().where(minus_dm_mask, 0)
            
            smoothed_plus_dm = plus_dm.rolling(period).sum()
            smoothed_minus_dm = minus_dm.rolling(period).sum()
            
            atr_safe = atr.replace(0, np.nan)
            
            plus_di = 100 * smoothed_plus_dm / atr_safe
            minus_di = 100 * smoothed_minus_dm / atr_safe
            
            di_sum = plus_di + minus_di
            di_sum_safe = di_sum.replace(0, np.nan)
            
            dx = 100 * abs(plus_di - minus_di) / di_sum_safe
            adx = dx.rolling(period).mean()
            
            return adx
        except Exception as e:
            return pd.Series(0, index=df.index)
    
    # Add the missing _calculate_rsi method
    def _calculate_rsi(self, prices, period=14):
        """Calculate RSI for internal use"""
        return self.calculate_rsi(prices, period)
    
    def _calculate_macd(self, prices, slow=26, fast=12):
        """Calculate MACD for internal use"""
        return self.calculate_macd(prices, slow, fast)
    
    def _calculate_stochastic(self, df, period=14):
        """Calculate Stochastic for internal use"""
        return self.calculate_stochastic(df, period)
    
    def _calculate_williams_r(self, df, period=14):
        """Calculate Williams %R for internal use"""
        return self.calculate_williams_r(df, period)
    
    def _calculate_atr(self, df, period=14):
        """Calculate ATR for internal use"""
        return self.calculate_atr(df, period)
    
    def _calculate_bollinger_bands(self, prices, period=20, std_dev=2):
        """Calculate Bollinger Bands for internal use"""
        return self.calculate_bollinger_bands(prices, period, std_dev)
    
    def _calculate_adx(self, df, period=14):
        """Calculate ADX for internal use"""
        return self.calculate_adx(df, period)

# Initialize global technical analyzer
technical_analyzer = TechnicalAnalyzer()

# Legacy function wrapper
def calculate_technical_indicators_for_summary(df):
    return technical_analyzer.calculate_technical_indicators_for_summary(df)
