import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)

# Number of students
n_students = 1000

# Generate basic data
data = {
    'student_id': range(1, n_students + 1),
    'study_hours_per_day': np.random.normal(6, 2, n_students).clip(0, 12),
    'sleep_hours': np.random.normal(7, 1.5, n_students).clip(4, 10),
    'social_media_hours': np.random.normal(3, 1.5, n_students).clip(0, 8),
    'netflix_hours': np.random.normal(2, 1, n_students).clip(0, 6),
    'attendance_percentage': np.random.normal(85, 10, n_students).clip(50, 100),
    'exercise_frequency': np.random.normal(3, 1.5, n_students).clip(0, 7),
    'mental_health_rating': np.random.normal(7, 2, n_students).clip(1, 10)
}

# Add categorical variables
data['gender'] = np.random.choice(['Male', 'Female', 'Other'], n_students)
data['diet_quality'] = np.random.choice(['Poor', 'Average', 'Good', 'Excellent'], n_students)
data['parental_education_level'] = np.random.choice(
    ['High School', "Bachelor's", "Master's", 'PhD'], 
    n_students, 
    p=[0.4, 0.3, 0.2, 0.1]
)
data['internet_quality'] = np.random.choice(['Low', 'Medium', 'High'], n_students)
data['part_time_job'] = np.random.choice(['Yes', 'No'], n_students)
data['extracurricular_participation'] = np.random.choice(['Yes', 'No'], n_students)

# Generate exam scores with dependencies on other variables
base_score = 70 + np.random.normal(0, 5, n_students)

# Study hours impact (strong positive)
study_impact = 10 * (data['study_hours_per_day'] - data['study_hours_per_day'].mean()) / data['study_hours_per_day'].std()

# Sleep impact (moderate positive)
sleep_impact = 5 * (data['sleep_hours'] - data['sleep_hours'].mean()) / data['sleep_hours'].std()

# Mental health impact (moderate positive)
mental_impact = 5 * (data['mental_health_rating'] - data['mental_health_rating'].mean()) / data['mental_health_rating'].std()

# Social media impact (weak negative)
social_impact = -3 * (data['social_media_hours'] - data['social_media_hours'].mean()) / data['social_media_hours'].std()

# Calculate final exam scores
data['exam_score'] = (base_score + study_impact + sleep_impact + mental_impact + social_impact).clip(0, 100)

# Generate timestamps for the last 30 days
start_date = datetime.now() - timedelta(days=30)
data['timestamp'] = [start_date + timedelta(days=np.random.randint(0, 30)) for _ in range(n_students)]

# Create DataFrame
df = pd.DataFrame(data)

# Create data directory if it doesn't exist
import os
os.makedirs('../data', exist_ok=True)

# Save to CSV
df.to_csv('../data/student_habits_performance.csv', index=False)

print("Sample dataset generated successfully!")
print(f"Shape of the dataset: {df.shape}")
print("\nSummary statistics:")
print(df.describe()) 