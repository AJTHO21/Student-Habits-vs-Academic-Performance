# Student Habits vs Academic Performance Analysis

![Analytics Dashboard](images/dashboard.png)

## Project Overview
This project analyzes the relationship between student habits and academic performance using a comprehensive dataset of 1,000 students. The analysis includes various factors such as study habits, sleep patterns, social media usage, and other lifestyle factors that may impact academic success. Through advanced statistical analysis and machine learning techniques, we've uncovered meaningful patterns and actionable insights that can help both students and educators optimize academic performance.

### Distribution of Exam Scores
![Exam Score Distribution](visualizations/distributions/distribution_exam_score.png)
![Age Distribution](visualizations/distributions/distribution_age.png)

## Key Findings

### 1. Study Habits Impact
Our analysis revealed a robust positive correlation (r = 0.76, p < 0.001) between study hours and academic performance, making it the strongest predictor of success among all factors analyzed. Using a combination of linear regression and non-parametric tests, we found that students who maintain 6-7 hours of daily study time achieve optimal results.

![Study Hours Impact](visualizations/relationships/relationship_study_hours_per_day_exam_score_enhanced.png)
![Study Hours Distribution](visualizations/distributions/distribution_study_hours_per_day.png)

### 2. Sleep Patterns
Our comprehensive sleep analysis revealed a moderate but significant positive correlation (r = 0.42, p < 0.001) between sleep quality and academic performance.

![Sleep Patterns Impact](visualizations/relationships/relationship_sleep_hours_exam_score_enhanced.png)
![Sleep Hours Distribution](visualizations/distributions/distribution_sleep_hours.png)

### 3. Social Media Usage
Our analysis of social media impact revealed a complex relationship with academic performance.

![Social Media Impact](visualizations/relationships/relationship_social_media_hours_exam_score_enhanced.png)
![Social Media Usage Distribution](visualizations/distributions/distribution_social_media_hours.png)

### 4. Mental Health
Mental health emerged as a critical factor in academic success.

![Mental Health Impact](visualizations/relationships/relationship_mental_health_rating_exam_score_enhanced.png)
![Mental Health Distribution](visualizations/distributions/distribution_mental_health_rating.png)

### 5. Physical Activity and Exercise
Regular exercise showed a positive correlation with academic performance.

![Exercise Impact](visualizations/relationships/relationship_exercise_frequency_exam_score_enhanced.png)
![Exercise Distribution](visualizations/distributions/distribution_exercise_frequency.png)

### 6. Attendance and Engagement
Attendance proved to be a significant predictor of academic success.

![Attendance Impact](visualizations/relationships/relationship_attendance_percentage_exam_score_enhanced.png)
![Attendance Distribution](visualizations/distributions/distribution_attendance_percentage.png)

### 7. Entertainment and Leisure
Analysis of Netflix usage provided insights into entertainment habits.

![Netflix Usage Impact](visualizations/relationships/relationship_netflix_hours_exam_score_enhanced.png)
![Netflix Hours Distribution](visualizations/distributions/distribution_netflix_hours.png)

### 8. Demographic Factors

#### Age and Performance
![Age Impact](visualizations/relationships/relationship_age_exam_score_enhanced.png)

#### Gender Distribution and Performance
![Gender Impact](visualizations/relationships/relationship_gender_exam_score_enhanced.png)
![Gender Distribution](visualizations/distributions/distribution_gender_enhanced.png)

#### Parental Education Level
![Parental Education Impact](visualizations/relationships/relationship_parental_education_level_exam_score_enhanced.png)
![Parental Education Distribution](visualizations/distributions/distribution_parental_education_level_enhanced.png)

#### Diet Quality
![Diet Quality Impact](visualizations/relationships/relationship_diet_quality_exam_score_enhanced.png)
![Diet Quality Distribution](visualizations/distributions/distribution_diet_quality_enhanced.png)

### 9. Environmental Factors

#### Internet Quality
![Internet Quality Impact](visualizations/relationships/relationship_internet_quality_exam_score_enhanced.png)
![Internet Quality Distribution](visualizations/distributions/distribution_internet_quality_enhanced.png)

#### Part-Time Job
![Part-Time Job Impact](visualizations/relationships/relationship_part_time_job_exam_score_enhanced.png)
![Part-Time Job Distribution](visualizations/distributions/distribution_part_time_job_enhanced.png)

#### Extracurricular Activities
![Extracurricular Impact](visualizations/relationships/relationship_extracurricular_participation_exam_score_enhanced.png)
![Extracurricular Distribution](visualizations/distributions/distribution_extracurricular_participation_enhanced.png)

## Statistical Analysis

### Correlation Analysis
![Correlation Matrix](visualizations/correlation_heatmap.png)

### Feature Importance
![Feature Importance](visualizations/feature_importance.png)

### Interactive Visualizations
For interactive exploration of the data, please check out:
- [Interactive Dashboard](visualizations/interactive_dashboard.html)
- [Success Patterns](visualizations/success_patterns.html)
- [Student Clusters](visualizations/student_clusters.html)
- [Scatter Matrix](visualizations/interactive/scatter_matrix.html)

## Recommendations

### For Students
1. Study Habits: Implement a consistent 6-7 hour daily study schedule with regular breaks
2. Sleep Management: Maintain 7-8 hours of quality sleep with minimal schedule variation
3. Social Media: Limit usage to 2 hours daily, avoiding study time
4. Physical Activity: Engage in regular exercise (3-4 times weekly)
5. Mental Health: Practice stress management techniques and seek support when needed

### For Educators
1. Workload Management: Implement balanced workload distribution
2. Study Skills: Provide training in effective study techniques
3. Mental Health: Offer comprehensive mental health support
4. Physical Activity: Encourage regular exercise and movement breaks
5. Engagement Monitoring: Track student engagement and provide early intervention

## Technical Details

### Tools Used
- Python 3.12
- Pandas, NumPy for data analysis
- Scikit-learn for machine learning
- Statsmodels for statistical analysis
- Plotly, Seaborn for visualization
- Dash for interactive dashboard

### Analysis Methods
1. Descriptive Statistics
2. Inferential Statistics
3. Machine Learning Models
4. Time Series Analysis
5. Cluster Analysis

## Interactive Dashboard
The project includes an interactive dashboard with:
- Real-time data visualization
- Profile analysis
- Success factor tracking
- Performance predictions
- Trend analysis

## Future Enhancements
1. Longitudinal study implementation
2. Additional behavioral metrics
3. Enhanced predictive modeling
4. Mobile application development
5. Real-time monitoring system

## Installation and Usage
```bash
# Clone the repository
git clone https://github.com/AJTHO21/Student-Habits-vs-Academic-Performance.git

# Install dependencies
pip install -r requirements.txt

# Generate sample data
python src/generate_sample_data.py

# Run analysis
python src/advanced_analysis.py
```

## Contributing
Contributions are welcome! Please read the contributing guidelines before submitting pull requests.

## License
This project is licensed under the MIT License - see the LICENSE file for details. 