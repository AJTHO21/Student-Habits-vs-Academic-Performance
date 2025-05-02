# %% [markdown]
# # Student Habits vs Academic Performance Analysis
# 
# This analysis explores the relationships between various student lifestyle habits and their academic performance.
# 
# ## Setup and Data Loading

# %%
# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
try:
    import networkx as nx
except ImportError:
    print("Installing networkx...")
    import subprocess
    subprocess.check_call(["pip", "install", "networkx>=3.2.0"])
    import networkx as nx
import os

# %%
# Load the dataset
current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, '..', 'data', 'student_habits_performance.csv')
df = pd.read_csv(data_path)

# Display the first few rows and basic information
print("First few rows of the dataset:")
print(df.head())

print("\nDataset shape:")
print(f"Number of rows: {df.shape[0]}")
print(f"Number of columns: {df.shape[1]}")

print("\nColumn names:")
print(df.columns.tolist())

# %%
# Basic statistics and data quality check
print("\nBasic statistics for numerical columns:")
print(df.describe())

print("\nMissing values per column:")
print(df.isnull().sum())

# %%
# Visualize distributions of key numerical features
plt.figure(figsize=(20, 15))

# Create subplots for key features
features = ['study_hours_per_day', 'sleep_hours', 'social_media_hours', 
           'netflix_hours', 'attendance_percentage', 'exercise_frequency',
           'mental_health_rating', 'exam_score']

for i, feature in enumerate(features, 1):
    plt.subplot(3, 3, i)
    sns.histplot(data=df, x=feature, kde=True)
    plt.title(f'Distribution of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Count')

plt.tight_layout()
plt.show()

# %%
# Analyze categorical variables
categorical_features = ['gender', 'diet_quality', 'parental_education_level', 
                       'internet_quality', 'part_time_job', 'extracurricular_participation']

plt.figure(figsize=(20, 15))
for i, feature in enumerate(categorical_features, 1):
    plt.subplot(2, 3, i)
    sns.countplot(data=df, x=feature)
    plt.title(f'Distribution of {feature}')
    plt.xticks(rotation=45)
    plt.xlabel(feature)
    plt.ylabel('Count')

plt.tight_layout()
plt.show()

# %%
# Analyze relationships with exam score
plt.figure(figsize=(20, 15))

# Numerical features vs exam score
numerical_features = ['study_hours_per_day', 'sleep_hours', 'social_media_hours',
                     'netflix_hours', 'attendance_percentage', 'exercise_frequency',
                     'mental_health_rating']

for i, feature in enumerate(numerical_features, 1):
    plt.subplot(3, 3, i)
    sns.scatterplot(data=df, x=feature, y='exam_score')
    plt.title(f'{feature} vs Exam Score')
    plt.xlabel(feature)
    plt.ylabel('Exam Score')

plt.tight_layout()
plt.show()

# %%
# Categorical features vs exam score
plt.figure(figsize=(20, 15))
for i, feature in enumerate(categorical_features, 1):
    plt.subplot(2, 3, i)
    sns.boxplot(data=df, x=feature, y='exam_score')
    plt.title(f'{feature} vs Exam Score')
    plt.xticks(rotation=45)
    plt.xlabel(feature)
    plt.ylabel('Exam Score')

plt.tight_layout()
plt.show()

# %%
# Correlation analysis for numerical features only
numerical_columns = ['age', 'study_hours_per_day', 'social_media_hours', 
                    'netflix_hours', 'attendance_percentage', 'sleep_hours',
                    'exercise_frequency', 'mental_health_rating', 'exam_score']

print("\nCorrelation with exam score:")
correlations = df[numerical_columns].corr()['exam_score'].sort_values(ascending=False)
print(correlations)

# %%
# Enhanced correlation visualizations

# %%
# 1. Enhanced Correlation Heatmap with Seaborn
plt.figure(figsize=(15, 12))
mask = np.triu(np.ones_like(df[numerical_columns].corr(), dtype=bool))
sns.heatmap(df[numerical_columns].corr(), 
            mask=mask,
            annot=True, 
            fmt='.2f',
            cmap='RdBu_r',
            center=0,
            square=True,
            linewidths=1)
plt.title('Correlation Heatmap of Student Habits', pad=20, size=16)
plt.tight_layout()
plt.show()

# %%
# 2. Network Plot
def create_network_plot(corr_matrix, threshold=0.1):
    plt.figure(figsize=(15, 15))
    G = nx.Graph()
    
    # Add nodes
    for col in corr_matrix.columns:
        G.add_node(col)
    
    # Add edges with correlation values
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr = corr_matrix.iloc[i, j]
            if abs(corr) > threshold:
                G.add_edge(corr_matrix.columns[i], 
                          corr_matrix.columns[j], 
                          weight=abs(corr),
                          correlation=corr)
    
    # Set layout with more iterations for better stability
    pos = nx.spring_layout(G, k=2, iterations=100, seed=42)
    
    # Draw edges with varying thickness and color based on correlation
    edges = G.edges()
    weights = [G[u][v]['weight']*5 for u,v in edges]
    edge_colors = [G[u][v]['correlation'] for u,v in edges]
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, 
                          width=weights,
                          edge_color=edge_colors,
                          edge_cmap=plt.cm.RdBu_r,
                          edge_vmin=-1,
                          edge_vmax=1,
                          alpha=0.7)
    
    # Draw nodes with custom colors and sizes
    node_sizes = [3000 if 'exam_score' in node else 2000 for node in G.nodes()]
    node_colors = ['lightcoral' if 'exam_score' in node else 'skyblue' for node in G.nodes()]
    
    nx.draw_networkx_nodes(G, pos, 
                          node_size=node_sizes,
                          node_color=node_colors,
                          alpha=0.8)
    
    # Add labels with better formatting
    labels = {node: '\n'.join(node.split('_')).title() for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, 
                           labels=labels,
                           font_size=10,
                           font_weight='bold')
    
    # Add a colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.RdBu_r, norm=plt.Normalize(vmin=-1, vmax=1))
    plt.colorbar(sm, label='Correlation Strength')
    
    plt.title("Network Visualization of Student Habits Correlations\n" + 
              f"(Showing correlations with magnitude > {threshold})", 
              pad=20, size=16)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# Create network plot with slightly higher threshold
create_network_plot(df[numerical_columns].corr(), threshold=0.15)

# %%
# 3. Interactive Scatter Matrix
key_vars = ['exam_score', 'study_hours_per_day', 'mental_health_rating', 'sleep_hours']
fig = px.scatter_matrix(df,
                       dimensions=key_vars,
                       color='diet_quality',
                       title='Scatter Matrix of Key Variables',
                       opacity=0.5)
fig.update_layout(title_x=0.5)
fig.show()

# %%
# 4. Correlation Sunburst
def create_correlation_sunburst(corr_matrix, target='exam_score'):
    # Prepare data for sunburst
    correlations = corr_matrix[target].sort_values(ascending=False)
    correlations = correlations[correlations.index != target]
    
    def get_correlation_category(corr):
        if abs(corr) >= 0.7:
            return 'Very Strong'
        elif abs(corr) >= 0.5:
            return 'Strong'
        elif abs(corr) >= 0.3:
            return 'Moderate'
        elif abs(corr) >= 0.1:
            return 'Weak'
        else:
            return 'Very Weak'
    
    # Create hierarchical data
    data = {
        'labels': ['Root', target],
        'parents': ['', 'Root'],
        'values': [1, 1],
        'customdata': ['', '1.0'],
        'text': ['', '1.0']
    }
    
    categories = ['Very Strong', 'Strong', 'Moderate', 'Weak', 'Very Weak']
    for category in categories:
        data['labels'].append(category)
        data['parents'].append(target)
        data['values'].append(1)
        data['customdata'].append('')
        data['text'].append('')
    
    for var, corr in correlations.items():
        category = get_correlation_category(corr)
        data['labels'].append(var)
        data['parents'].append(category)
        data['values'].append(abs(corr))
        data['customdata'].append(f'{corr:.2f}')
        data['text'].append(f'{corr:.2f}')
    
    # Create sunburst plot
    fig = go.Figure(go.Sunburst(
        labels=data['labels'],
        parents=data['parents'],
        values=data['values'],
        branchvalues='total',
        customdata=data['customdata'],
        text=data['text'],
        hovertemplate='<b>%{label}</b><br>Correlation: %{customdata}<extra></extra>',
        maxdepth=2
    ))
    
    fig.update_layout(
        title=f'Hierarchical View of Correlations with {target.replace("_", " ").title()}',
        width=800,
        height=800
    )
    fig.show()

# Create sunburst visualization
create_correlation_sunburst(df[numerical_columns].corr())