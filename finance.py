import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning Libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, classification_report, confusion_matrix
from sklearn.decomposition import PCA

# Deep Learning
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

class MLStockAnalyzer:
    def __init__(self, tickers, start_date='2018-01-01', end_date='2024-03-27'):
        """
        Initialize ML-powered stock analysis
        
        :param tickers: List of stock tickers
        :param start_date: Start of historical data
        :param end_date: End of historical data
        """
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.stock_data = {}
        self.ml_models = {}
    
    def fetch_and_prepare_data(self):
        """
        Fetch historical data and prepare features for ML
        """
        for ticker in self.tickers:
            try:
                # Fetch stock data
                stock = yf.Ticker(ticker)
                df = stock.history(start=self.start_date, end=self.end_date)
                
                if len(df) > 0:
                    # Feature Engineering
                    df['Returns'] = df['Close'].pct_change()
                    df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
                    
                    # Technical Indicators
                    df['SMA_20'] = df['Close'].rolling(window=20).mean()
                    df['SMA_50'] = df['Close'].rolling(window=50).mean()
                    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
                    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
                    
                    # Volatility Indicators
                    df['Bollinger_High'] = df['SMA_20'] + (df['Close'].rolling(window=20).std() * 2)
                    df['Bollinger_Low'] = df['SMA_20'] - (df['Close'].rolling(window=20).std() * 2)
                    
                    # Momentum Indicators
                    df['RSI'] = self._calculate_rsi(df['Close'])
                    df['MACD'], df['MACD_Signal'] = self._calculate_macd(df['Close'])
                    
                    # Target Variables
                    df['Future_Returns_5d'] = df['Returns'].shift(-5)
                    df['Return_Direction'] = np.where(df['Future_Returns_5d'] > 0, 1, 0)
                    
                    # Drop NaN values
                    df.dropna(inplace=True)
                    
                    self.stock_data[ticker] = df
                else:
                    print(f"No data available for {ticker}")
            
            except Exception as e:
                print(f"Error processing {ticker}: {e}")
    
    def _calculate_rsi(self, prices, periods=14):
        """
        Calculate Relative Strength Index
        """
        delta = prices.diff()
        
        up = delta.clip(lower=0)
        down = -1 * delta.clip(upper=0)
        
        ma_up = up.ewm(com=periods-1, adjust=True, min_periods=periods).mean()
        ma_down = down.ewm(com=periods-1, adjust=True, min_periods=periods).mean()
        
        rsi = ma_up / ma_down
        rsi = 100.0 - (100.0 / (1.0 + rsi))
        
        return rsi
    
    def _calculate_macd(self, prices, slow=26, fast=12, signal=9):
        """
        Calculate MACD
        """
        exp1 = prices.ewm(span=fast, adjust=False).mean()
        exp2 = prices.ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        
        return macd, signal_line
    
    def prepare_ml_features(self, ticker):
        """
        Prepare features for machine learning
        """
        df = self.stock_data[ticker]
        
        # Select features
        features = [
            'Returns', 'Log_Returns', 
            'SMA_20', 'SMA_50', 
            'EMA_12', 'EMA_26',
            'Bollinger_High', 'Bollinger_Low',
            'RSI', 'MACD', 'MACD_Signal'
        ]
        
        X = df[features]
        y_returns = df['Future_Returns_5d']
        y_direction = df['Return_Direction']
        
        return X, y_returns, y_direction
    
    def train_predictive_models(self):
        """
        Train machine learning models for each stock
        """
        for ticker in self.tickers:
            try:
                # Prepare features
                X, y_returns, y_direction = self.prepare_ml_features(ticker)
                
                # Split data
                X_train, X_test, y_returns_train, y_returns_test, y_direction_train, y_direction_test = train_test_split(
                    X, y_returns, y_direction, test_size=0.2, random_state=42
                )
                
                # Scale features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Regression Model (Returns Prediction)
                rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
                rf_regressor.fit(X_train_scaled, y_returns_train)
                
                # Classification Model (Return Direction)
                rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
                rf_classifier.fit(X_train_scaled, y_direction_train)
                
                # Deep Learning LSTM Model
                X_lstm = X.values.reshape((X.shape[0], 1, X.shape[1]))
                X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm = train_test_split(
                    X_lstm, y_returns, test_size=0.2, random_state=42
                )
                
                lstm_model = Sequential([
                    LSTM(50, activation='relu', input_shape=(1, X.shape[1]), return_sequences=True),
                    Dropout(0.2),
                    LSTM(50, activation='relu'),
                    Dense(25, activation='relu'),
                    Dense(1)
                ])
                
                lstm_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
                lstm_model.fit(X_train_lstm, y_train_lstm, epochs=50, batch_size=32, verbose=0)
                
                # Store models
                self.ml_models[ticker] = {
                    'Regressor': rf_regressor,
                    'Classifier': rf_classifier,
                    'LSTM': lstm_model,
                    'Scaler': scaler
                }
                
                # Evaluate models
                self._evaluate_models(ticker, X_test_scaled, y_returns_test, y_direction_test)
            
            except Exception as e:
                print(f"Error training models for {ticker}: {e}")
    
    def _evaluate_models(self, ticker, X_test, y_returns_test, y_direction_test):
        """
        Evaluate model performance
        """
        models = self.ml_models[ticker]
        
        # Regression Evaluation
        returns_pred = models['Regressor'].predict(X_test)
        mse = mean_squared_error(y_returns_test, returns_pred)
        print(f"\n{ticker} - Returns Prediction MSE: {mse}")
        
        # Classification Evaluation
        direction_pred = models['Classifier'].predict(X_test)
        print(f"\n{ticker} - Return Direction Classification Report:")
        print(classification_report(y_direction_test, direction_pred))
    
    def feature_importance_analysis(self):
        """
        Analyze feature importance across stocks
        """
        feature_importances = {}
        
        for ticker, models in self.ml_models.items():
            # Extract feature importance from Random Forest
            importances = models['Regressor'].feature_importances_
            feature_names = list(self.prepare_ml_features(ticker)[0].columns)
            
            feature_importances[ticker] = pd.Series(importances, index=feature_names)
        
        # Aggregate feature importance
        aggregated_importance = pd.DataFrame(feature_importances).mean(axis=1)
        
        # Visualization
        plt.figure(figsize=(10, 6))
        aggregated_importance.sort_values(ascending=False).plot(kind='bar')
        plt.title('Average Feature Importance Across Stocks')
        plt.xlabel('Features')
        plt.ylabel('Importance Score')
        plt.tight_layout()
        plt.show()
        
        return aggregated_importance
    
    def portfolio_optimization(self):
        """
        Perform portfolio optimization using ML predictions
        """
        portfolio_predictions = {}
        
        for ticker, models in self.ml_models.items():
            # Get latest data
            latest_data = self.stock_data[ticker].iloc[-1]
            X_latest = latest_data[self.prepare_ml_features(ticker)[0].columns].values.reshape(1, -1)
            
            # Scale latest data
            X_latest_scaled = models['Scaler'].transform(X_latest)
            
            # Predict returns and direction
            predicted_return = models['Regressor'].predict(X_latest_scaled)[0]
            predicted_direction = models['Classifier'].predict(X_latest_scaled)[0]
            
            portfolio_predictions[ticker] = {
                'Predicted_Return': predicted_return,
                'Predicted_Direction': 'Positive' if predicted_direction == 1 else 'Negative'
            }
        
        # Convert to DataFrame
        portfolio_df = pd.DataFrame.from_dict(portfolio_predictions, orient='index')
        portfolio_df = portfolio_df.sort_values('Predicted_Return', ascending=False)
        
        print("\n--- Portfolio Optimization Predictions ---")
        print(portfolio_df)
        
        return portfolio_df

# Define SAS stocks (50 across different sectors)
sas_stocks = [
    # Technology
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 
    'CSCO', 'INTC', 'CRM', 'ADBE', 'ORCL',
    
    # Finance
    'JPM', 'BAC', 'WFC', 'GS', 'MS', 
    'V', 'MA', 'AXP', 'PYPL', 'SQ',
    
    # Healthcare
    'JNJ', 'PFE', 'MRK', 'UNH', 'ABT', 
    'BMY', 'ABBV', 'DHR', 'GILD', 'REGN',
    
    # Energy
    'XOM', 'CVX', 'COP', 'PSX', 'EOG', 
    'SLB', 'HES', 'OXY', 'VLO', 'MRO',
    
    # Retail
    'WMT', 'HD', 'TGT', 'COST', 'LOW', 
    'SBUX', 'MCD', 'NKE', 'AMZN', 'BABA'
]

# Execute the machine learning analysis
def run_ml_stock_analysis():
    # Initialize the ML analyzer
    ml_analyzer = MLStockAnalyzer(sas_stocks)
    
    # Fetch and prepare data
    ml_analyzer.fetch_and_prepare_data()
    
    # Train predictive models
    ml_analyzer.train_predictive_models()
    
    # Analyze feature importance
    feature_importance = ml_analyzer.feature_importance_analysis()
    
    # Perform portfolio optimization
    portfolio_predictions = ml_analyzer.portfolio_optimization()
    
    return ml_analyzer

# Run the analysis
ml_analysis = run_ml_stock_analysis()