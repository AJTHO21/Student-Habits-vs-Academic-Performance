import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_data(file_path):
    """Load and return the dataset."""
    return pd.read_csv(file_path)

def clean_data(df):
    """Perform basic data cleaning operations."""
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Handle missing values
    for col in df.columns:
        if df[col].dtype in ['int64', 'float64']:
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode()[0])
    
    return df

def analyze_correlations(df, target_column):
    """Analyze correlations with the target variable."""
    correlations = df.corr()[target_column].sort_values(ascending=False)
    return correlations

def plot_correlation_heatmap(df, figsize=(12, 10)):
    """Create a correlation heatmap."""
    plt.figure(figsize=figsize)
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    return plt

def plot_distribution(df, column, bins=30, figsize=(10, 6)):
    """Plot distribution of a column."""
    plt.figure(figsize=figsize)
    sns.histplot(data=df, x=column, bins=bins, kde=True)
    plt.title(f'Distribution of {column}')
    plt.tight_layout()
    return plt

def prepare_data_for_modeling(df, target_column, test_size=0.2, random_state=42):
    """Prepare data for machine learning modeling."""
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def plot_feature_importance(importance, feature_names, top_n=10, figsize=(10, 6)):
    """Plot feature importance."""
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    }).sort_values('Importance', ascending=False).head(top_n)
    
    plt.figure(figsize=figsize)
    sns.barplot(data=importance_df, x='Importance', y='Feature')
    plt.title('Feature Importance')
    plt.tight_layout()
    return plt 