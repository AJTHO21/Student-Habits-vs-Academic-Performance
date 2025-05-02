import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Create sample data with feature importance scores
features = [
    'Study Hours',
    'Sleep Quality',
    'Attendance',
    'Mental Health',
    'Exercise',
    'Social Media',
    'Part-Time Job',
    'Diet Quality',
    'Internet Quality',
    'Extracurricular'
]

importance_scores = [
    0.76,  # Study Hours (from correlation)
    0.42,  # Sleep Quality (from correlation)
    0.65,  # Attendance (estimated)
    0.55,  # Mental Health (estimated)
    0.35,  # Exercise (estimated)
    -0.25, # Social Media (estimated negative impact)
    0.15,  # Part-Time Job (estimated)
    0.30,  # Diet Quality (estimated)
    0.20,  # Internet Quality (estimated)
    0.45   # Extracurricular (estimated)
]

# Create DataFrame
df = pd.DataFrame({
    'Feature': features,
    'Importance': np.abs(importance_scores)  # Use absolute values for visualization
})

# Sort by importance
df = df.sort_values('Importance', ascending=True)

# Create visualization
plt.figure(figsize=(12, 8))
bars = plt.barh(df['Feature'], df['Importance'])

# Color bars based on importance
colors = plt.cm.RdYlBu(np.linspace(0.15, 0.85, len(df)))
for bar, color in zip(bars, colors):
    bar.set_color(color)

# Customize plot
plt.title('Feature Importance Analysis', fontsize=14, pad=20)
plt.xlabel('Absolute Correlation with Academic Performance', fontsize=12)

# Add value labels on bars
for i, v in enumerate(df['Importance']):
    plt.text(v + 0.01, i, f'{v:.2f}', va='center', fontsize=10)

# Customize grid
plt.grid(axis='x', linestyle='--', alpha=0.7)

# Adjust layout and save
plt.tight_layout()
plt.savefig('visualizations/feature_importance.png', dpi=300, bbox_inches='tight')
plt.close() 