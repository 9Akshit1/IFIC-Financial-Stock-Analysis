import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Machine Learning and Deep Learning Libraries
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.feature_selection import mutual_info_regression, SelectKBest
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

# Deep Learning
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# Advanced ML Libraries
import xgboost as xgb


class EnhancedStockPredictor:
    def __init__(self, tickers, start_date='2010-01-01', end_date='2024-03-27'):
        """
        Enhanced stock prediction framework with advanced techniques and company-specific financial indicators
        """
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.stock_data = {}
        self.financial_data = {}  # To store company financial data
        self.models = {}
        self.feature_importances = {}
    
    def fetch_stock_data(self):
        """
        Enhanced data fetching with more comprehensive preprocessing
        """
        successful_tickers = []
        for ticker in self.tickers:
            try:
                # Download more historical data
                df = yf.download(ticker, start=self.start_date, end=self.end_date, auto_adjust=False)
                
                if len(df) > 0:
                    # Fetch financial data for the company
                    financial_data = self._fetch_company_financials(ticker)
                    
                    if financial_data is not None:
                        self.financial_data[ticker] = financial_data
                    
                    # Advanced preprocessing
                    processed_df = self._advanced_preprocessing(df, ticker)
                    
                    if not processed_df.empty:
                        self.stock_data[ticker] = processed_df
                        successful_tickers.append(ticker)
                    else:
                        print(f"Preprocessing failed for {ticker}")
                else:
                    print(f"No data available for {ticker}")
            
            except Exception as e:
                print(f"Error fetching data for {ticker}: {e}")
        
        self.tickers = successful_tickers
        return self.tickers
    
    def _fetch_company_financials(self, ticker):
        """
        Fetch company financial data for calculating financial ratios
        """
        try:
            stock = yf.Ticker(ticker)
            
            # Get company information
            info = stock.info
            
            # Get balance sheet data
            balance_sheet = stock.balance_sheet
            
            # Get income statement data
            income_stmt = stock.income_stmt
            
            # Get cash flow statement data
            cash_flow = stock.cashflow
            
            # Get quarterly financial data for more recent metrics
            quarterly_financials = stock.quarterly_financials
            quarterly_balance_sheet = stock.quarterly_balance_sheet
            quarterly_cashflow = stock.quarterly_cashflow
            
            # Return all financial data
            return {
                'info': info,
                'balance_sheet': balance_sheet,
                'income_stmt': income_stmt,
                'cash_flow': cash_flow,
                'quarterly_financials': quarterly_financials,
                'quarterly_balance_sheet': quarterly_balance_sheet,
                'quarterly_cashflow': quarterly_cashflow
            }
        except Exception as e:
            print(f"Error fetching financial data for {ticker}: {e}")
            return None
    
    def _calculate_financial_ratios(self, df, ticker):
        """
        Calculate company-specific financial ratios and add them to the dataframe
        """
        if ticker not in self.financial_data:
            return df
        
        financial_data = self.financial_data[ticker]
        
        # Make a copy to avoid modifying the input
        result_df = df.copy()
        
        # Helper function to safely get values from financial data
        def safe_get(data_dict, key, default=np.nan):
            return data_dict.get(key, default)
        
        # Create date index mapping for the financial data
        # This allows us to merge quarterly/annual data with daily price data
        result_df['Year'] = result_df.index.year
        result_df['Quarter'] = result_df.index.quarter
        
        # Initialize financial ratio columns
        ratio_columns = [
            'EV_Sales', 'PE_Ratio', 'PS_Ratio', 'Rule_of_40', 'Beta', 
            'CAPM_Expected_Return', 'FCF_Yield', 'RD_Revenue_Ratio', 
            'RD_Efficiency', 'LTV_CAC_Ratio',
            # Components of ratios as separate features
            'Enterprise_Value', 'Market_Cap', 'Total_Revenue', 'Net_Income',
            'Free_Cash_Flow', 'RD_Expense', 'Revenue_Growth', 'Profit_Margin',
            'Risk_Free_Rate', 'Market_Return', 'Customer_Acquisition_Cost',
            'Customer_Lifetime_Value'
        ]
        
        for col in ratio_columns:
            result_df[col] = np.nan
        
        try:
            # Get the info dictionary
            info = financial_data.get('info', {})
            
            # Extract key metrics that are available in info
            result_df['Beta'] = safe_get(info, 'beta', 1.0)
            result_df['PE_Ratio'] = safe_get(info, 'trailingPE')
            result_df['PS_Ratio'] = safe_get(info, 'priceToSalesTrailing12Months')
            result_df['Market_Cap'] = safe_get(info, 'marketCap')
            
            # Process financial statements quarterly
            if 'quarterly_financials' in financial_data and isinstance(financial_data['quarterly_financials'], pd.DataFrame) and not financial_data['quarterly_financials'].empty:
                qf = financial_data['quarterly_financials']
                qbs = financial_data.get('quarterly_balance_sheet', pd.DataFrame())
                qcf = financial_data.get('quarterly_cashflow', pd.DataFrame())
                
                # Process each quarter's data
                for date in qf.columns:
                    year = date.year
                    quarter = (date.month - 1) // 3 + 1
                    
                    # Get quarterly data
                    quarterly_revenue = qf.loc['Total Revenue', date] if 'Total Revenue' in qf.index else np.nan
                    quarterly_net_income = qf.loc['Net Income', date] if 'Net Income' in qf.index else np.nan
                    quarterly_rd = qf.loc['Research Development', date] if 'Research Development' in qf.index else np.nan
                    
                    # Previous quarter for growth calculations
                    prev_date = None
                    prev_quarters = [d for d in qf.columns if d < date]
                    if prev_quarters:
                        prev_date = max(prev_quarters)
                    
                    # Calculate Revenue Growth if previous quarter data is available
                    revenue_growth = np.nan
                    if prev_date and 'Total Revenue' in qf.index:
                        prev_revenue = qf.loc['Total Revenue', prev_date]
                        if prev_revenue > 0:
                            revenue_growth = (quarterly_revenue - prev_revenue) / prev_revenue * 100
                    
                    # Calculate Profit Margin
                    profit_margin = np.nan
                    if not np.isnan(quarterly_revenue) and quarterly_revenue > 0:
                        profit_margin = (quarterly_net_income / quarterly_revenue) * 100
                    
                    # Calculate Rule of 40
                    rule_of_40 = np.nan
                    if not np.isnan(revenue_growth) and not np.isnan(profit_margin):
                        rule_of_40 = revenue_growth + profit_margin
                    
                    # Extract Free Cash Flow from cash flow statement
                    fcf = np.nan
                    if isinstance(qcf, pd.DataFrame) and not qcf.empty and 'Free Cash Flow' in qcf.index and date in qcf.columns:
                        fcf = qcf.loc['Free Cash Flow', date]
                    
                    # Calculate R&D to Revenue Ratio
                    rd_revenue_ratio = np.nan
                    if not np.isnan(quarterly_rd) and not np.isnan(quarterly_revenue) and quarterly_revenue > 0:
                        rd_revenue_ratio = quarterly_rd / quarterly_revenue
                    
                    # Calculate R&D Efficiency (Revenue Growth / R&D as % of Revenue)
                    rd_efficiency = np.nan
                    if not np.isnan(revenue_growth) and not np.isnan(rd_revenue_ratio) and rd_revenue_ratio > 0:
                        rd_efficiency = revenue_growth / (rd_revenue_ratio * 100)
                    
                    # Calculate Enterprise Value components
                    total_debt = 0
                    if isinstance(qbs, pd.DataFrame) and not qbs.empty:
                        if 'Total Debt' in qbs.index and date in qbs.columns:
                            total_debt = qbs.loc['Total Debt', date]
                        elif 'Long Term Debt' in qbs.index and date in qbs.columns:
                            total_debt = qbs.loc['Long Term Debt', date]
                    
                    cash_and_equivalents = 0
                    if isinstance(qbs, pd.DataFrame) and not qbs.empty and 'Cash And Cash Equivalents' in qbs.index and date in qbs.columns:
                        cash_and_equivalents = qbs.loc['Cash And Cash Equivalents', date]
                    
                    # Market Cap might vary daily, but we'll use the latest available
                    market_cap = result_df['Market_Cap'].iloc[0] if not pd.isna(result_df['Market_Cap'].iloc[0]) else 0
                    
                    # Calculate Enterprise Value
                    enterprise_value = market_cap + total_debt - cash_and_equivalents
                    
                    # Calculate EV/Sales Ratio
                    ev_sales = np.nan
                    if enterprise_value > 0 and not np.isnan(quarterly_revenue) and quarterly_revenue > 0:
                        ev_sales = enterprise_value / quarterly_revenue
                    
                    # Calculate FCF Yield
                    fcf_yield = np.nan
                    if not np.isnan(fcf) and market_cap > 0:
                        fcf_yield = fcf / market_cap
                    
                    # Apply the financial metrics to the matching dates in the result_df
                    mask = (result_df['Year'] == year) & (result_df['Quarter'] == quarter)
                    if mask.any():
                        result_df.loc[mask, 'Total_Revenue'] = quarterly_revenue
                        result_df.loc[mask, 'Net_Income'] = quarterly_net_income
                        result_df.loc[mask, 'RD_Expense'] = quarterly_rd
                        result_df.loc[mask, 'Revenue_Growth'] = revenue_growth
                        result_df.loc[mask, 'Profit_Margin'] = profit_margin
                        result_df.loc[mask, 'Rule_of_40'] = rule_of_40
                        result_df.loc[mask, 'Free_Cash_Flow'] = fcf
                        result_df.loc[mask, 'FCF_Yield'] = fcf_yield
                        result_df.loc[mask, 'RD_Revenue_Ratio'] = rd_revenue_ratio
                        result_df.loc[mask, 'RD_Efficiency'] = rd_efficiency
                        result_df.loc[mask, 'Enterprise_Value'] = enterprise_value
                        result_df.loc[mask, 'EV_Sales'] = ev_sales
            
            # CAPM Calculation (E(R) = Rf + β × (Rm - Rf))
            risk_free_rate = 0.03  # Approximate risk-free rate (3%)
            market_return = 0.10   # Approximate market return (10%)
            result_df['Risk_Free_Rate'] = risk_free_rate
            result_df['Market_Return'] = market_return
            
            # Calculate CAPM Expected Return
            result_df['CAPM_Expected_Return'] = result_df['Risk_Free_Rate'] + result_df['Beta'] * (result_df['Market_Return'] - result_df['Risk_Free_Rate'])
            
            # LTV and CAC are typically not available directly from financial statements
            # We'll use approximations based on available data
            # For simplicity, we'll use placeholder values that can be updated manually
            # In a real scenario, these would be calculated from customer acquisition costs and retention rates
            result_df['Customer_Acquisition_Cost'] = np.nan  # Placeholder
            result_df['Customer_Lifetime_Value'] = np.nan    # Placeholder
            result_df['LTV_CAC_Ratio'] = np.nan             # Placeholder
            
            # If SaaS company, estimate LTV/CAC from financial data
            if 'quarterly_financials' in financial_data and isinstance(financial_data['quarterly_financials'], pd.DataFrame) and not financial_data['quarterly_financials'].empty:
                # Very rough approximation for SaaS companies
                # For a real model, these would need company-specific data
                if 'Marketing Expense' in financial_data['quarterly_financials'].index:
                    # Assuming 60% of marketing expense goes to customer acquisition
                    marketing_expense = financial_data['quarterly_financials'].loc['Marketing Expense', :].mean()
                    estimated_cac = marketing_expense * 0.6 / (safe_get(info, 'fullTimeEmployees', 1000) / 100)
                    result_df['Customer_Acquisition_Cost'] = estimated_cac
                
                # Estimate LTV using gross margin and churn rate
                if 'Gross Profit' in financial_data['quarterly_financials'].index and 'Total Revenue' in financial_data['quarterly_financials'].index:
                    avg_gross_profit = financial_data['quarterly_financials'].loc['Gross Profit', :].mean()
                    avg_revenue = financial_data['quarterly_financials'].loc['Total Revenue', :].mean()
                    
                    # Gross margin
                    gross_margin = avg_gross_profit / avg_revenue if avg_revenue > 0 else 0
                    
                    # Assumed monthly churn rate for a SaaS company (5%)
                    monthly_churn = 0.05
                    
                    # Estimate customer LTV: (Annual Revenue per Customer * Gross Margin) / Churn Rate
                    # This is a simplified LTV formula
                    if 'fullTimeEmployees' in info and info['fullTimeEmployees'] > 0:
                        arpu = (avg_revenue * 4) / (info['fullTimeEmployees'] * 10)  # Annual revenue per user (rough estimate)
                        ltv = (arpu * gross_margin) / monthly_churn
                        result_df['Customer_Lifetime_Value'] = ltv
                
                # Calculate LTV/CAC ratio
                mask = (~pd.isna(result_df['Customer_Lifetime_Value'])) & (~pd.isna(result_df['Customer_Acquisition_Cost'])) & (result_df['Customer_Acquisition_Cost'] > 0)
                if mask.any():
                    result_df.loc[mask, 'LTV_CAC_Ratio'] = result_df.loc[mask, 'Customer_Lifetime_Value'] / result_df.loc[mask, 'Customer_Acquisition_Cost']
        
        except Exception as e:
            print(f"Error calculating financial ratios for {ticker}: {e}")
        
        # Forward fill financial ratios (since they're reported quarterly/annually)
        ratio_columns = [
            'EV_Sales', 'PE_Ratio', 'PS_Ratio', 'Rule_of_40', 'CAPM_Expected_Return', 
            'FCF_Yield', 'RD_Revenue_Ratio', 'RD_Efficiency', 'LTV_CAC_Ratio',
            'Enterprise_Value', 'Market_Cap', 'Total_Revenue', 'Net_Income',
            'Free_Cash_Flow', 'RD_Expense', 'Revenue_Growth', 'Profit_Margin',
            'Customer_Acquisition_Cost', 'Customer_Lifetime_Value'
        ]
        
        result_df[ratio_columns] = result_df[ratio_columns].fillna(method='ffill')
        
        # Drop temporary columns
        result_df.drop(['Year', 'Quarter'], axis=1, inplace=True)
        
        return result_df
    
    def _advanced_preprocessing(self, df, ticker):
        """
        Comprehensive feature engineering and preprocessing with financial ratios
        """
        # Ensure DataFrame has required columns
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in required_columns):
            return pd.DataFrame()
        
        # Enhanced feature engineering
        df['Returns'] = df['Close'].pct_change()
        df['Log_Returns'] = np.log(1 + df['Returns'])
        
        # Advanced Moving Averages and Trends
        df['SMA_10'] = df['Close'].rolling(window=10).mean()
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
        
        # Volatility Indicators
        df['Volatility_20d'] = df['Returns'].rolling(window=20).std() * np.sqrt(252)
        
        # Momentum Indicators
        df['Momentum_10d'] = df['Close'].pct_change(10)
        
        # Relative Strength Index (RSI)
        df['RSI'] = self._calculate_rsi(df['Close'])
        
        # Bollinger Bands
        rolling_mean = df['Close'].rolling(window=20).mean()
        rolling_std = df['Close'].rolling(window=20).std()
        df['BB_Middle'] = rolling_mean
        df['BB_Upper'] = rolling_mean + (rolling_std * 2)
        df['BB_Lower'] = rolling_mean - (rolling_std * 2)
        
        # Additional Technical Indicators
        
        # MACD (Moving Average Convergence Divergence)
        df['MACD'] = df['Close'].ewm(span=12, adjust=False).mean() - df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
        
        # Stochastic Oscillator
        df['Stoch_K'] = self._calculate_stochastic(df)
        df['Stoch_D'] = df['Stoch_K'].rolling(window=3).mean()
        
        # Average True Range (ATR)
        df['ATR'] = self._calculate_atr(df)
        
        # On-Balance Volume (OBV)
        df['OBV'] = self._calculate_obv(df)
        
        # Price Rate of Change (ROC)
        df['ROC_5'] = df['Close'].pct_change(periods=5) * 100
        df['ROC_10'] = df['Close'].pct_change(periods=10) * 100
        df['ROC_20'] = df['Close'].pct_change(periods=20) * 100
        
        # Volume indicators
        df['Volume_ROC'] = df['Volume'].pct_change() * 100
        df['Volume_SMA_5'] = df['Volume'].rolling(window=5).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA_5']
        
        # Target Variable: Future Returns
        df['Future_Return_5d'] = df['Returns'].shift(-5)
        df['Future_Return_10d'] = df['Returns'].shift(-10)
        
        # Add financial ratios and company-specific metrics
        df = self._calculate_financial_ratios(df, ticker)
        
        # Drop NaN values
        df.dropna(inplace=True)
        
        return df
    
    def _calculate_rsi(self, prices, periods=14):
        """Enhanced RSI Calculation"""
        delta = prices.diff()
        
        gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_stochastic(self, df, k_period=14):
        """Calculate Stochastic Oscillator"""
        low_min = df['Low'].rolling(window=k_period).min()
        high_max = df['High'].rolling(window=k_period).max()
        
        stoch_k = 100 * ((df['Close'] - low_min) / (high_max - low_min))
        return stoch_k
    
    def _calculate_atr(self, df, period=14):
        """Calculate Average True Range (ATR)"""
        high_low = df['High'] - df['Low']
        high_close = (df['High'] - df['Close'].shift()).abs()
        low_close = (df['Low'] - df['Close'].shift()).abs()
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        
        atr = true_range.rolling(window=period).mean()
        return atr
    
    def _calculate_obv(self, df):
        """Calculate On-Balance Volume (OBV)"""
        obv = np.zeros(len(df))
        
        for i in range(1, len(df)):
            if df['Close'].iloc[i] > df['Close'].iloc[i-1]:
                obv[i] = obv[i-1] + df['Volume'].iloc[i]
            elif df['Close'].iloc[i] < df['Close'].iloc[i-1]:
                obv[i] = obv[i-1] - df['Volume'].iloc[i]
            else:
                obv[i] = obv[i-1]
        
        return pd.Series(obv, index=df.index)
    
    def prepare_ml_data(self, ticker, target_column='Future_Return_5d'):
        """
        Advanced data preparation with feature selection including financial ratios
        """
        df = self.stock_data[ticker]
        
        # Select features dynamically (including the new financial indicators)
        features = [
            # Price and Returns Features
            'Returns', 'Log_Returns', 'Open', 'High', 'Low', 'Close', 'Volume',
            
            # Technical Indicators
            'SMA_10', 'SMA_20', 'SMA_50', 'EMA_20', 
            'Volatility_20d', 'Momentum_10d', 'RSI',
            'BB_Middle', 'BB_Upper', 'BB_Lower',
            'MACD', 'MACD_Signal', 'MACD_Hist',
            'Stoch_K', 'Stoch_D', 'ATR', 'OBV',
            'ROC_5', 'ROC_10', 'ROC_20',
            'Volume_ROC', 'Volume_SMA_5', 'Volume_Ratio',
            
            # Financial Ratios
            'EV_Sales', 'PE_Ratio', 'PS_Ratio', 'Rule_of_40', 'Beta',
            'CAPM_Expected_Return', 'FCF_Yield', 'RD_Revenue_Ratio',
            'RD_Efficiency', 'LTV_CAC_Ratio',
            
            # Components of Financial Ratios
            'Enterprise_Value', 'Market_Cap', 'Total_Revenue', 'Net_Income',
            'Free_Cash_Flow', 'RD_Expense', 'Revenue_Growth', 'Profit_Margin',
            'Customer_Acquisition_Cost', 'Customer_Lifetime_Value'
        ]
        
        # Filter to only include columns that exist in the dataframe
        features = [f for f in features if f in df.columns]
        
        X = df[features]
        y = df[target_column]
        
        # Feature selection using mutual information
        # We'll use a higher k value to accommodate more features
        selector = SelectKBest(score_func=mutual_info_regression, k=min(20, len(features)))
        X_selected = selector.fit_transform(X, y)
        
        # Get selected feature names
        selected_feature_names = [features[i] for i in selector.get_support(indices=True)]
        
        # Store feature importances
        feature_scores = selector.scores_
        self.feature_importances[ticker] = dict(zip(features, feature_scores))
        
        # Handle missing values
        imputer = SimpleImputer(strategy='median')
        X_imputed = imputer.fit_transform(X_selected)
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_imputed)
        
        # Use time series split for more robust validation
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Split data while preserving time series structure
        splits = list(tscv.split(X_scaled))
        train_index, test_index = splits[-1]
        X_train, X_test = X_scaled[train_index], X_scaled[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        return X_train, X_test, y_train, y_test, selected_feature_names
    
    def train_xgboost_model(self, X_train, y_train):
        """
        Advanced XGBoost model with hyperparameter tuning
        """
        # Hyperparameter grid
        param_grid = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.1, 0.3],
            'max_depth': [3, 5, 7],
            'subsample': [0.6, 0.8, 1.0]
        }
        
        # Create XGBoost model
        model = xgb.XGBRegressor(random_state=42)
        
        # Grid search with cross-validation
        grid_search = GridSearchCV(
            estimator=model, 
            param_grid=param_grid, 
            cv=TimeSeriesSplit(n_splits=3),
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        
        # Fit grid search
        grid_search.fit(X_train, y_train)
        
        # Best model
        best_model = grid_search.best_estimator_
        
        return best_model
    
    def train_lstm_model(self, X_train, y_train):
        """
        Advanced LSTM model with more sophisticated architecture
        """
        # Reshape input for LSTM [samples, timesteps, features]
        X_train_lstm = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
        
        model = Sequential([
            Bidirectional(LSTM(100, return_sequences=True), input_shape=(1, X_train.shape[1])),
            Dropout(0.3),
            Bidirectional(LSTM(50, return_sequences=True)),
            Dropout(0.3),
            Bidirectional(LSTM(25)),
            Dropout(0.2),
            Dense(10, activation='relu'),
            Dense(1)
        ])
        
        # Compile with adaptive learning rate
        optimizer = Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss='mse')
        
        # Advanced callbacks
        early_stop = EarlyStopping(
            patience=15, 
            restore_best_weights=True, 
            monitor='val_loss'
        )
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss', 
            factor=0.5, 
            patience=5, 
            min_lr=0.00001
        )
        model_checkpoint = ModelCheckpoint(
            f'best_lstm_model.h5', 
            save_best_only=True, 
            monitor='val_loss'
        )
        
        # Train with more sophisticated setup
        history = model.fit(
            X_train_lstm, y_train, 
            epochs=100, 
            batch_size=32, 
            validation_split=0.2,
            callbacks=[early_stop, reduce_lr, model_checkpoint],
            verbose=0
        )
        
        return model
    
    def evaluate_models(self):
        """
        Comprehensive model evaluation with detailed analysis
        """
        results = {}
        
        for ticker in self.tickers:
            try:
                # Prepare data
                X_train, X_test, y_train, y_test, feature_names = self.prepare_ml_data(ticker)
                
                # Train models
                xgb_model = self.train_xgboost_model(X_train, y_train)
                lstm_model = self.train_lstm_model(X_train, y_train)
                
                # Prepare test data for LSTM
                X_test_lstm = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
                
                # Predictions
                xgb_pred = xgb_model.predict(X_test)
                lstm_pred = lstm_model.predict(X_test_lstm).flatten()
                
                # Weighted Ensemble (giving more weight to better performing model)
                xgb_mse = mean_squared_error(y_test, xgb_pred)
                lstm_mse = mean_squared_error(y_test, lstm_pred)
                total_mse = xgb_mse + lstm_mse
                
                # Weighted average ensemble
                ensemble_pred = (
                    (lstm_mse / total_mse) * xgb_pred + 
                    (xgb_mse / total_mse) * lstm_pred
                )
                
                # Compute metrics
                results[ticker] = {
                    'XGBoost': {
                        'MSE': mean_squared_error(y_test, xgb_pred),
                        'MAE': mean_absolute_error(y_test, xgb_pred),
                        'R2': r2_score(y_test, xgb_pred)
                    },
                    'LSTM': {
                        'MSE': mean_squared_error(y_test, lstm_pred),
                        'MAE': mean_absolute_error(y_test, lstm_pred),
                        'R2': r2_score(y_test, lstm_pred)
                    },
                    'Ensemble': {
                        'MSE': mean_squared_error(y_test, ensemble_pred),
                        'MAE': mean_absolute_error(y_test, ensemble_pred),
                        'R2': r2_score(y_test, ensemble_pred)
                    },
                    'Top_Features': self.feature_importances[ticker]
                }
                
                # Store models
                self.models[ticker] = {
                    'XGBoost': xgb_model,
                    'LSTM': lstm_model
                }
            
            except Exception as e:
                print(f"Model training error for {ticker}: {e}")
        
        return results
        
    def visualize_performance(self, results):
        """
        Comprehensive visualization of model performance
        """
        if not results:
            print("No results to visualize. Make sure models were trained successfully.")
            return
            
        # Performance Metrics Boxplot
        plt.figure(figsize=(18, 6))
        metrics = ['MSE', 'MAE', 'R2']
        models = ['XGBoost', 'LSTM', 'Ensemble']
        
        for i, metric in enumerate(metrics, 1):
            plt.subplot(1, 3, i)
            data = []
            
            for model in models:
                model_data = []
                for ticker in results:
                    if model in results[ticker] and metric in results[ticker][model]:
                        model_data.append(results[ticker][model][metric])
                data.append(model_data)
            
            if all(len(d) > 0 for d in data):  # Only plot if we have data
                plt.boxplot(data, labels=models)
                plt.title(f'{metric} Performance Across Stocks')
                plt.ylabel(metric)
            else:
                plt.text(0.5, 0.5, f'No {metric} data available', 
                         horizontalalignment='center', verticalalignment='center')
        
        plt.tight_layout()
        plt.show()
        
        # Feature Importance Visualization
        plt.figure(figsize=(15, 5))
        feature_scores = {}
        
        # Aggregate feature importances
        for ticker in results:
            if 'Top_Features' in results[ticker]:
                for feature, score in results[ticker]['Top_Features'].items():
                    if feature not in feature_scores:
                        feature_scores[feature] = []
                    feature_scores[feature].append(score)
        
        if feature_scores:
            # Average feature scores
            avg_feature_scores = {f: np.mean(scores) for f, scores in feature_scores.items()}
            sorted_features = sorted(avg_feature_scores.items(), key=lambda x: x[1], reverse=True)
            
            # Limit to top 20 features if there are many
            if len(sorted_features) > 20:
                sorted_features = sorted_features[:20]
                
            plt.bar([f[0] for f in sorted_features], [f[1] for f in sorted_features])
            plt.title('Average Feature Importance Across Stocks')
            plt.xlabel('Features')
            plt.ylabel('Importance Score')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
        else:
            plt.text(0.5, 0.5, 'No feature importance data available', 
                     horizontalalignment='center', verticalalignment='center')
            
        plt.show()

# Main execution
tickers = [
    'AAPL', 'GOOGL', 'MSFT', 'AMZN', 'CRM', 
    'ADBE', 'SNOW', 'TEAM', 'NOW', 'DOCU', 
    'WDAY', 'ZM', 'SHOP', 'PYPL', 'ADSK', 
    'OKTA', 'HUBS'
]

# Initialize predictor
predictor = EnhancedStockPredictor(tickers)

# Fetch and process data
successful_tickers = predictor.fetch_stock_data()

# Train and evaluate models
results = predictor.evaluate_models()

# Visualize results
predictor.visualize_performance(results)

print("Successful Tickers:", successful_tickers)