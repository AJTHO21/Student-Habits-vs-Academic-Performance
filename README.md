# Student Habits vs Academic Performance Analysis

<img src="images/dashboard.png" alt="Analytics Dashboard" width="800"/>

## Project Overview
This project analyzes the relationship between student habits and academic performance using a comprehensive dataset of 1,000 students. The analysis includes various factors such as study habits, sleep patterns, social media usage, and other lifestyle factors that may impact academic success. Through advanced statistical analysis and machine learning techniques, we've uncovered meaningful patterns and actionable insights that can help both students and educators optimize academic performance.

### Distribution of Exam Scores
<img src="visualizations/distributions/distribution_exam_score.png" alt="Exam Score Distribution" width="600"/>
<img src="visualizations/distributions/distribution_age.png" alt="Age Distribution" width="600"/>

## Key Findings

### 1. Study Habits Impact
Our analysis revealed a robust positive correlation (r = 0.76, p < 0.001) between study hours and academic performance, making it the strongest predictor of success among all factors analyzed. Using a combination of linear regression and non-parametric tests, we found that students who maintain 6-7 hours of daily study time achieve optimal results.

<img src="visualizations/relationships/relationship_study_hours_per_day_exam_score_enhanced.png" alt="Study Hours Impact" width="600"/>
<img src="visualizations/distributions/distribution_study_hours_per_day.png" alt="Study Hours Distribution" width="600"/>

### 2. Sleep Patterns
Our comprehensive sleep analysis revealed a moderate but significant positive correlation (r = 0.42, p < 0.001) between sleep quality and academic performance.

<img src="visualizations/relationships/relationship_sleep_hours_exam_score_enhanced.png" alt="Sleep Patterns Impact" width="600"/>
<img src="visualizations/distributions/distribution_sleep_hours.png" alt="Sleep Hours Distribution" width="600"/>

### 3. Social Media Usage
Our analysis of social media impact revealed a complex relationship with academic performance.

<img src="visualizations/relationships/relationship_social_media_hours_exam_score_enhanced.png" alt="Social Media Impact" width="600"/>
<img src="visualizations/distributions/distribution_social_media_hours.png" alt="Social Media Usage Distribution" width="600"/>

### 4. Mental Health
Mental health emerged as a critical factor in academic success.

<img src="visualizations/relationships/relationship_mental_health_rating_exam_score_enhanced.png" alt="Mental Health Impact" width="600"/>
<img src="visualizations/distributions/distribution_mental_health_rating.png" alt="Mental Health Distribution" width="600"/>

### 5. Physical Activity and Exercise
Regular exercise showed a positive correlation with academic performance.

<img src="visualizations/relationships/relationship_exercise_frequency_exam_score_enhanced.png" alt="Exercise Impact" width="600"/>
<img src="visualizations/distributions/distribution_exercise_frequency.png" alt="Exercise Distribution" width="600"/>

### 6. Attendance and Engagement
Attendance proved to be a significant predictor of academic success.

<img src="visualizations/relationships/relationship_attendance_percentage_exam_score_enhanced.png" alt="Attendance Impact" width="600"/>
<img src="visualizations/distributions/distribution_attendance_percentage.png" alt="Attendance Distribution" width="600"/>

### 7. Entertainment and Leisure
Analysis of Netflix usage provided insights into entertainment habits.

<img src="visualizations/relationships/relationship_netflix_hours_exam_score_enhanced.png" alt="Netflix Usage Impact" width="600"/>
<img src="visualizations/distributions/distribution_netflix_hours.png" alt="Netflix Hours Distribution" width="600"/>

### 8. Demographic Factors

#### Age and Performance
Our analysis of age-related performance revealed interesting patterns in academic achievement across different age groups. The data showed that while age itself had a moderate correlation with academic performance (r = 0.38, p < 0.001), the relationship was non-linear, with peak performance occurring in specific age ranges. This suggests that maturity and experience play a significant role in academic success, but other factors like study habits and time management become increasingly important as students progress through their academic journey.

<img src="visualizations/relationships/relationship_age_exam_score_enhanced.png" alt="Age Impact" width="600"/>

#### Gender Distribution and Performance
<img src="visualizations/relationships/relationship_gender_exam_score_enhanced.png" alt="Gender Impact" width="600"/>
<img src="visualizations/distributions/distribution_gender_enhanced.png" alt="Gender Distribution" width="600"/>

#### Parental Education Level
<img src="visualizations/relationships/relationship_parental_education_level_exam_score_enhanced.png" alt="Parental Education Impact" width="600"/>
<img src="visualizations/distributions/distribution_parental_education_level_enhanced.png" alt="Parental Education Distribution" width="600"/>

#### Diet Quality
<img src="visualizations/relationships/relationship_diet_quality_exam_score_enhanced.png" alt="Diet Quality Impact" width="600"/>
<img src="visualizations/distributions/distribution_diet_quality_enhanced.png" alt="Diet Quality Distribution" width="600"/>

### 9. Environmental Factors

#### Internet Quality
<img src="visualizations/relationships/relationship_internet_quality_exam_score_enhanced.png" alt="Internet Quality Impact" width="600"/>
<img src="visualizations/distributions/distribution_internet_quality_enhanced.png" alt="Internet Quality Distribution" width="600"/>

#### Part-Time Job
<img src="visualizations/relationships/relationship_part_time_job_exam_score_enhanced.png" alt="Part-Time Job Impact" width="600"/>
<img src="visualizations/distributions/distribution_part_time_job_enhanced.png" alt="Part-Time Job Distribution" width="600"/>

#### Extracurricular Activities
<img src="visualizations/relationships/relationship_extracurricular_participation_exam_score_enhanced.png" alt="Extracurricular Impact" width="600"/>
<img src="visualizations/distributions/distribution_extracurricular_participation_enhanced.png" alt="Extracurricular Distribution" width="600"/>

## Statistical Analysis

### Correlation Analysis
<img src="visualizations/correlation_heatmap.png" alt="Correlation Matrix" width="800"/>

### Feature Importance
<img src="visualizations/numerical_vs_exam_score.png" alt="Feature Importance Analysis" width="800"/>

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