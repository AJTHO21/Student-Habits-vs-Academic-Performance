"""
Advanced Statistical Analysis for Student Habits and Academic Performance
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import os
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import statsmodels.api as sm
from scipy.stats import ttest_ind, f_oneway, chi2_contingency
import warnings

class AdvancedAnalysis:
    def __init__(self, data_path):
        """Initialize with the dataset and prepare numerical features."""
        self.df = pd.read_csv(data_path)
        self.numerical_features = [
            'study_hours_per_day', 'sleep_hours', 'social_media_hours',
            'netflix_hours', 'attendance_percentage', 'exercise_frequency',
            'mental_health_rating'
        ]
        self.target = 'exam_score'
        self.viz_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'visualizations')
        os.makedirs(self.viz_dir, exist_ok=True)

    def perform_multiple_regression(self):
        """Perform multiple regression analysis."""
        X = self.df[self.numerical_features]
        y = self.df[self.target]

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Fit the model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Get predictions and model performance
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        # Create coefficient analysis
        coef_df = pd.DataFrame({
            'Feature': self.numerical_features,
            'Coefficient': model.coef_
        }).sort_values('Coefficient', ascending=False)

        # Visualize feature importance
        fig = go.Figure(go.Bar(
            x=coef_df['Coefficient'],
            y=coef_df['Feature'],
            orientation='h'
        ))
        fig.update_layout(
            title='Feature Importance in Predicting Exam Scores',
            xaxis_title='Coefficient Value',
            yaxis_title='Feature',
            template='plotly_white'
        )
        fig.write_html(os.path.join(self.viz_dir, 'feature_importance.html'))

        return {
            'r2_score': r2,
            'rmse': rmse,
            'coefficients': coef_df.to_dict('records')
        }

    def perform_cluster_analysis(self):
        """Perform cluster analysis to identify student archetypes."""
        # Prepare data
        X = self.df[self.numerical_features]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Determine optimal number of clusters
        inertias = []
        K = range(1, 10)
        for k in K:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(X_scaled)
            inertias.append(kmeans.inertia_)

        # Perform clustering with optimal K
        optimal_k = 4  # We can make this dynamic based on elbow method
        kmeans = KMeans(n_clusters=optimal_k, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)
        self.df['Cluster'] = clusters

        # Create cluster profiles
        cluster_profiles = []
        for i in range(optimal_k):
            cluster_data = self.df[self.df['Cluster'] == i]
            profile = {
                'cluster_number': i,
                'size': len(cluster_data),
                'avg_exam_score': cluster_data[self.target].mean(),
                'characteristics': {
                    feature: cluster_data[feature].mean()
                    for feature in self.numerical_features
                }
            }
            cluster_profiles.append(profile)

        # Visualize clusters using PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        fig = px.scatter(
            x=X_pca[:, 0], y=X_pca[:, 1],
            color=clusters.astype(str),
            title='Student Clusters Based on Habits',
            labels={'x': 'First Principal Component', 'y': 'Second Principal Component'}
        )
        fig.write_html(os.path.join(self.viz_dir, 'student_clusters.html'))

        return cluster_profiles

    def identify_success_patterns(self):
        """Identify patterns associated with high performance."""
        # Define high performers (top 25%)
        high_score_threshold = self.df[self.target].quantile(0.75)
        high_performers = self.df[self.df[self.target] >= high_score_threshold]

        # Calculate success patterns
        success_patterns = {
            'threshold_score': high_score_threshold,
            'habit_ranges': {
                feature: {
                    'mean': high_performers[feature].mean(),
                    'std': high_performers[feature].std(),
                    'min': high_performers[feature].quantile(0.25),
                    'max': high_performers[feature].quantile(0.75)
                }
                for feature in self.numerical_features
            }
        }

        # Visualize success patterns
        fig = go.Figure()
        for feature in self.numerical_features:
            fig.add_trace(go.Box(
                y=high_performers[feature],
                name=feature,
                boxpoints='all',
                jitter=0.3,
                pointpos=-1.8
            ))

        fig.update_layout(
            title='Habit Patterns of High-Performing Students',
            yaxis_title='Value',
            template='plotly_white'
        )
        fig.write_html(os.path.join(self.viz_dir, 'success_patterns.html'))

        return success_patterns

    def create_interactive_dashboard(self):
        """Create an interactive dashboard for exploring relationships."""
        # Create subplot figure
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Study Hours vs Exam Score',
                'Sleep Hours vs Exam Score',
                'Mental Health vs Exam Score',
                'Exercise vs Exam Score'
            )
        )

        # Add traces
        key_features = ['study_hours_per_day', 'sleep_hours', 'mental_health_rating', 'exercise_frequency']
        positions = [(1, 1), (1, 2), (2, 1), (2, 2)]

        for feature, pos in zip(key_features, positions):
            fig.add_trace(
                go.Scatter(
                    x=self.df[feature],
                    y=self.df[self.target],
                    mode='markers',
                    name=feature,
                    marker=dict(size=8, opacity=0.6)
                ),
                row=pos[0], col=pos[1]
            )

        fig.update_layout(
            height=800,
            title_text="Interactive Student Performance Dashboard",
            template='plotly_white'
        )

        fig.write_html(os.path.join(self.viz_dir, 'interactive_dashboard.html'))

    def create_predictive_pipeline(self):
        """Create a predictive modeling pipeline for student performance."""
        # Define numerical and categorical features
        numerical_features = [
            'study_hours_per_day', 'sleep_hours', 'social_media_hours',
            'netflix_hours', 'attendance_percentage', 'exercise_frequency',
            'mental_health_rating'
        ]
        categorical_features = [
            'gender', 'diet_quality', 'parental_education_level',
            'internet_quality', 'part_time_job', 'extracurricular_participation'
        ]

        # Create preprocessing pipeline
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_features),
                ('cat', OneHotEncoder(), categorical_features)
            ]
        )

        # Create full pipeline
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
        ])

        # Prepare data
        X = self.df[numerical_features + categorical_features]
        y = self.df[self.target]

        # Perform cross-validation
        cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='r2')
        mean_r2 = cv_scores.mean()
        std_r2 = cv_scores.std()

        # Fit the model
        pipeline.fit(X, y)

        # Get feature importance
        feature_names = (
            numerical_features +
            list(pipeline.named_steps['preprocessor']
                .named_transformers_['cat']
                .get_feature_names_out(categorical_features))
        )
        importances = pipeline.named_steps['regressor'].feature_importances_

        # Create feature importance visualization
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False)

        fig = go.Figure(go.Bar(
            x=importance_df['Importance'],
            y=importance_df['Feature'],
            orientation='h'
        ))
        fig.update_layout(
            title='Feature Importance in Performance Prediction',
            xaxis_title='Importance Score',
            yaxis_title='Feature',
            template='plotly_white'
        )
        fig.write_html(os.path.join(self.viz_dir, 'predictive_feature_importance.html'))

        return {
            'mean_r2': mean_r2,
            'std_r2': std_r2,
            'feature_importance': importance_df.to_dict('records'),
            'pipeline': pipeline
        }

    def perform_detailed_statistical_analysis(self):
        """Perform detailed statistical analysis for all metrics."""
        warnings.filterwarnings('ignore')

        results = {}

        # Numerical metrics analysis
        for feature in self.numerical_features:
            # Correlation analysis
            corr, p_value = stats.pearsonr(self.df[feature], self.df[self.target])
            
            # Distribution analysis
            skewness = stats.skew(self.df[feature])
            kurtosis = stats.kurtosis(self.df[feature])
            
            # Quartile analysis
            q1 = self.df[feature].quantile(0.25)
            q3 = self.df[feature].quantile(0.75)
            iqr = q3 - q1
            
            results[feature] = {
                'correlation': {
                    'r': corr,
                    'p_value': p_value
                },
                'distribution': {
                    'skewness': skewness,
                    'kurtosis': kurtosis,
                    'q1': q1,
                    'q3': q3,
                    'iqr': iqr
                }
            }

        # Categorical metrics analysis
        categorical_features = [
            'gender', 'diet_quality', 'parental_education_level',
            'internet_quality', 'part_time_job', 'extracurricular_participation'
        ]

        for feature in categorical_features:
            # ANOVA for categorical vs numerical target
            groups = [self.df[self.df[feature] == val][self.target] 
                     for val in self.df[feature].unique()]
            f_stat, p_value = f_oneway(*groups)
            
            # Effect size calculation
            group_means = self.df.groupby(feature)[self.target].mean()
            group_sizes = self.df.groupby(feature).size()
            overall_mean = self.df[self.target].mean()
            
            # Calculate eta squared
            ss_between = sum(group_sizes * (group_means - overall_mean)**2)
            ss_total = sum((self.df[self.target] - overall_mean)**2)
            eta_squared = ss_between / ss_total
            
            results[feature] = {
                'anova': {
                    'f_stat': f_stat,
                    'p_value': p_value,
                    'eta_squared': eta_squared
                },
                'group_means': group_means.to_dict(),
                'group_sizes': group_sizes.to_dict()
            }

        return results

    def analyze_success_factors(self):
        """Analyze factors contributing to success and failure."""
        # Define success and failure thresholds
        success_threshold = self.df[self.target].quantile(0.75)
        failure_threshold = self.df[self.target].quantile(0.25)
        
        # Get success and failure groups
        success_group = self.df[self.df[self.target] >= success_threshold]
        failure_group = self.df[self.df[self.target] <= failure_threshold]
        
        # Calculate effect sizes for each feature
        effect_sizes = {}
        for feature in self.numerical_features:
            # Calculate Cohen's d
            mean_diff = success_group[feature].mean() - failure_group[feature].mean()
            pooled_std = np.sqrt(
                (success_group[feature].std()**2 + failure_group[feature].std()**2) / 2
            )
            cohens_d = mean_diff / pooled_std
            
            effect_sizes[feature] = {
                'cohens_d': cohens_d,
                'mean_success': success_group[feature].mean(),
                'mean_failure': failure_group[feature].mean(),
                'std_success': success_group[feature].std(),
                'std_failure': failure_group[feature].std()
            }
        
        # Calculate odds ratios for categorical features
        categorical_features = [
            'gender', 'diet_quality', 'parental_education_level',
            'internet_quality', 'part_time_job', 'extracurricular_participation'
        ]
        
        for feature in categorical_features:
            # Create contingency table
            contingency = pd.crosstab(self.df[feature], 
                                    self.df[self.target] >= success_threshold)
            odds_ratio = (contingency[True][0] * contingency[False][1]) / \
                        (contingency[True][1] * contingency[False][0])
            
            effect_sizes[feature] = {
                'odds_ratio': odds_ratio,
                'success_proportions': (success_group[feature].value_counts() / 
                                      len(success_group)).to_dict(),
                'failure_proportions': (failure_group[feature].value_counts() / 
                                      len(failure_group)).to_dict()
            }
        
        return effect_sizes

    def analyze_student_profiles(self):
        """Analyze different student profiles and provide specific recommendations."""
        # Define profile clusters based on key characteristics
        profile_features = [
            'study_hours_per_day', 'sleep_hours', 'mental_health_rating',
            'exercise_frequency', 'social_media_hours'
        ]
        
        # Perform clustering
        scaler = StandardScaler()
        X = scaler.fit_transform(self.df[profile_features])
        kmeans = KMeans(n_clusters=5, random_state=42)
        self.df['Profile'] = kmeans.fit_predict(X)
        
        # Analyze each profile
        profiles = {}
        for profile_id in range(5):
            profile_data = self.df[self.df['Profile'] == profile_id]
            
            # Calculate profile characteristics
            characteristics = {
                'size': len(profile_data),
                'avg_exam_score': profile_data[self.target].mean(),
                'habits': {
                    feature: {
                        'mean': profile_data[feature].mean(),
                        'std': profile_data[feature].std()
                    }
                    for feature in profile_features
                },
                'categorical_distributions': {
                    feature: profile_data[feature].value_counts(normalize=True).to_dict()
                    for feature in ['diet_quality', 'parental_education_level', 
                                  'internet_quality', 'extracurricular_participation']
                }
            }
            
            # Generate recommendations
            recommendations = self._generate_profile_recommendations(characteristics)
            
            profiles[profile_id] = {
                'characteristics': characteristics,
                'recommendations': recommendations
            }
        
        # Visualize profiles
        self._visualize_profiles(profiles)
        
        return profiles

    def _generate_profile_recommendations(self, characteristics):
        """Generate specific recommendations for a student profile."""
        recommendations = {
            'strengths': [],
            'weaknesses': [],
            'priority_actions': [],
            'long_term_goals': []
        }
        
        # Analyze study habits
        study_hours = characteristics['habits']['study_hours_per_day']['mean']
        if study_hours < 5:
            recommendations['weaknesses'].append('Insufficient study time')
            recommendations['priority_actions'].append(
                'Gradually increase study time to 6-7 hours daily'
            )
        elif study_hours > 8:
            recommendations['weaknesses'].append('Potential burnout risk')
            recommendations['priority_actions'].append(
                'Optimize study efficiency rather than increasing hours'
            )
        else:
            recommendations['strengths'].append('Good study time management')
        
        # Analyze sleep patterns
        sleep_hours = characteristics['habits']['sleep_hours']['mean']
        if sleep_hours < 6:
            recommendations['weaknesses'].append('Sleep deprivation')
            recommendations['priority_actions'].append(
                'Establish consistent sleep schedule (7-8 hours)'
            )
        elif sleep_hours > 9:
            recommendations['weaknesses'].append('Excessive sleep')
            recommendations['priority_actions'].append(
                'Maintain 7-8 hours of quality sleep'
            )
        else:
            recommendations['strengths'].append('Healthy sleep patterns')
        
        # Analyze mental health
        mental_health = characteristics['habits']['mental_health_rating']['mean']
        if mental_health < 6:
            recommendations['weaknesses'].append('Mental health concerns')
            recommendations['priority_actions'].append(
                'Implement stress management techniques'
            )
        else:
            recommendations['strengths'].append('Good mental health maintenance')
        
        # Analyze exercise
        exercise = characteristics['habits']['exercise_frequency']['mean']
        if exercise < 2:
            recommendations['weaknesses'].append('Insufficient physical activity')
            recommendations['priority_actions'].append(
                'Incorporate regular exercise (3-4 times weekly)'
            )
        else:
            recommendations['strengths'].append('Regular exercise routine')
        
        # Analyze social media usage
        social_media = characteristics['habits']['social_media_hours']['mean']
        if social_media > 3:
            recommendations['weaknesses'].append('Excessive social media use')
            recommendations['priority_actions'].append(
                'Implement social media time limits'
            )
        else:
            recommendations['strengths'].append('Balanced social media usage')
        
        # Set long-term goals based on profile
        if characteristics['avg_exam_score'] < 70:
            recommendations['long_term_goals'].append(
                'Focus on building strong foundational study habits'
            )
        elif characteristics['avg_exam_score'] < 85:
            recommendations['long_term_goals'].append(
                'Optimize existing habits for better performance'
            )
        else:
            recommendations['long_term_goals'].append(
                'Maintain current habits while exploring advanced techniques'
            )
        
        return recommendations

    def _visualize_profiles(self, profiles):
        """Create visualizations for student profiles."""
        # Create radar chart for each profile
        for profile_id, profile_data in profiles.items():
            fig = go.Figure()
            
            # Add radar trace
            fig.add_trace(go.Scatterpolar(
                r=[
                    profile_data['characteristics']['habits']['study_hours_per_day']['mean'],
                    profile_data['characteristics']['habits']['sleep_hours']['mean'],
                    profile_data['characteristics']['habits']['mental_health_rating']['mean'],
                    profile_data['characteristics']['habits']['exercise_frequency']['mean'],
                    profile_data['characteristics']['habits']['social_media_hours']['mean']
                ],
                theta=['Study Hours', 'Sleep Hours', 'Mental Health', 
                      'Exercise', 'Social Media'],
                fill='toself',
                name=f'Profile {profile_id}'
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 10]
                    )
                ),
                title=f'Student Profile {profile_id} Characteristics',
                template='plotly_white'
            )
            
            fig.write_html(os.path.join(self.viz_dir, f'profile_{profile_id}_radar.html'))

    def perform_advanced_statistical_tests(self):
        """Perform advanced statistical tests on the data."""
        from scipy.stats import mannwhitneyu, kruskal, chi2_contingency, spearmanr
        from scipy.stats import anderson, shapiro, levene, friedmanchisquare
        from statsmodels.stats.multitest import multipletests
        from statsmodels.stats.diagnostic import het_breuschpagan
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        from statsmodels.tsa.seasonal import seasonal_decompose
        from statsmodels.tsa.stattools import adfuller, kpss
        from statsmodels.stats.proportion import proportions_ztest
        import warnings
        warnings.filterwarnings('ignore')

        results = {
            'univariate_tests': {},
            'multivariate_tests': {},
            'time_series_tests': {},
            'regression_diagnostics': {},
            'effect_sizes': {}
        }

        # 1. Univariate Analysis
        for feature in self.numerical_features:
            # Normality tests
            shapiro_stat, shapiro_p = shapiro(self.df[feature])
            anderson_stat, anderson_crit, anderson_sig = anderson(self.df[feature])
            
            # Outlier detection using modified z-score
            median = np.median(self.df[feature])
            mad = np.median(np.abs(self.df[feature] - median))
            modified_zscores = 0.6745 * (self.df[feature] - median) / mad
            outliers = np.abs(modified_zscores) > 3.5
            
            results['univariate_tests'][feature] = {
                'normality': {
                    'shapiro': {'statistic': shapiro_stat, 'p_value': shapiro_p},
                    'anderson': {
                        'statistic': anderson_stat,
                        'critical_values': anderson_crit,
                        'significance_levels': anderson_sig
                    }
                },
                'outliers': {
                    'count': sum(outliers),
                    'percentage': (sum(outliers) / len(outliers)) * 100
                }
            }

        # 2. Multivariate Analysis
        # Correlation matrix using Spearman's rho
        corr_matrix = np.zeros((len(self.numerical_features), len(self.numerical_features)))
        p_values = np.zeros_like(corr_matrix)
        
        for i, feat1 in enumerate(self.numerical_features):
            for j, feat2 in enumerate(self.numerical_features):
                rho, p = spearmanr(self.df[feat1], self.df[feat2])
                corr_matrix[i, j] = rho
                p_values[i, j] = p
        
        results['multivariate_tests']['spearman_correlation'] = {
            'correlation_matrix': corr_matrix.tolist(),
            'p_values': p_values.tolist(),
            'features': self.numerical_features
        }

        # 3. Regression Diagnostics
        X = sm.add_constant(self.df[self.numerical_features])
        model = sm.OLS(self.df[self.target], X).fit()
        
        # Heteroscedasticity test
        _, bp_p_value, _, _ = het_breuschpagan(model.resid, model.model.exog)
        
        # Variance Inflation Factor
        vif_data = pd.DataFrame()
        vif_data["Feature"] = self.numerical_features
        vif_data["VIF"] = [variance_inflation_factor(X.values, i+1) for i in range(len(self.numerical_features))]
        
        results['regression_diagnostics'] = {
            'heteroscedasticity': {
                'breusch_pagan_p_value': bp_p_value
            },
            'multicollinearity': vif_data.to_dict('records')
        }

        # 4. Effect Size Calculations
        for feature in self.numerical_features:
            # Cohen's d for high vs low performers
            high_performers = self.df[self.df[self.target] >= self.df[self.target].quantile(0.75)][feature]
            low_performers = self.df[self.df[self.target] <= self.df[self.target].quantile(0.25)][feature]
            
            cohens_d = (high_performers.mean() - low_performers.mean()) / \
                      np.sqrt((high_performers.var() + low_performers.var()) / 2)
            
            # Glass's delta
            glass_delta = (high_performers.mean() - low_performers.mean()) / low_performers.std()
            
            results['effect_sizes'][feature] = {
                'cohens_d': cohens_d,
                'glass_delta': glass_delta
            }

        # 5. Advanced Group Comparisons
        # Levene's test for homogeneity of variances
        for feature in self.numerical_features:
            groups = [self.df[self.df[cat] == val][feature] 
                     for cat in ['diet_quality', 'parental_education_level']
                     for val in self.df[cat].unique()]
            levene_stat, levene_p = levene(*groups)
            
            results['multivariate_tests'][f'levene_test_{feature}'] = {
                'statistic': levene_stat,
                'p_value': levene_p
            }

        # 6. Proportion Tests
        for cat_feature in ['diet_quality', 'parental_education_level', 'internet_quality']:
            categories = sorted(self.df[cat_feature].unique())
            prop_tests = {}
            
            # Compare each pair of categories
            for i in range(len(categories)):
                for j in range(i + 1, len(categories)):
                    cat1, cat2 = categories[i], categories[j]
                    
                    # Get data for each category
                    cat1_data = self.df[self.df[cat_feature] == cat1]
                    cat2_data = self.df[self.df[cat_feature] == cat2]
                    
                    # Calculate success counts (above mean score)
                    mean_score = self.df[self.target].mean()
                    success_counts = [
                        sum(cat1_data[self.target] >= mean_score),
                        sum(cat2_data[self.target] >= mean_score)
                    ]
                    n_trials = [len(cat1_data), len(cat2_data)]
                    
                    # Perform proportion test
                    z_stat, p_value = proportions_ztest(success_counts, n_trials)
                    
                    prop_tests[f'{cat1}_vs_{cat2}'] = {
                        'z_statistic': z_stat,
                        'p_value': p_value,
                        'success_counts': success_counts,
                        'n_trials': n_trials
                    }
            
            results['multivariate_tests'][f'proportion_test_{cat_feature}'] = prop_tests

        # 7. Time-based Analysis (if applicable)
        if 'timestamp' in self.df.columns:
            self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
            self.df.set_index('timestamp', inplace=True)
            
            # Decomposition
            try:
                decomposition = seasonal_decompose(self.df[self.target], period=7)
                results['time_series_tests']['seasonal_decomposition'] = {
                    'trend': decomposition.trend.dropna().tolist(),
                    'seasonal': decomposition.seasonal.dropna().tolist(),
                    'residual': decomposition.resid.dropna().tolist()
                }
                
                # Stationarity tests
                adf_stat, adf_p, _, _, critical_values, _ = adfuller(self.df[self.target].dropna())
                kpss_stat, kpss_p, _, _ = kpss(self.df[self.target].dropna())
                
                results['time_series_tests']['stationarity'] = {
                    'adf_test': {
                        'statistic': adf_stat,
                        'p_value': adf_p,
                        'critical_values': critical_values
                    },
                    'kpss_test': {
                        'statistic': kpss_stat,
                        'p_value': kpss_p
                    }
                }
            except:
                pass

        # Multiple testing correction across all tests
        all_p_values = []
        p_value_sources = []
        
        # Collect all p-values
        for test_category in results.values():
            if isinstance(test_category, dict):
                for test_result in test_category.values():
                    if isinstance(test_result, dict):
                        for key, value in test_result.items():
                            if isinstance(value, dict) and 'p_value' in value:
                                all_p_values.append(value['p_value'])
                                p_value_sources.append(f"{key}_p_value")
        
        # Perform FDR correction
        if all_p_values:
            corrected_p_values = multipletests(all_p_values, method='fdr_bh')[1]
            results['multiple_testing_correction'] = {
                'original_p_values': dict(zip(p_value_sources, all_p_values)),
                'corrected_p_values': dict(zip(p_value_sources, corrected_p_values))
            }

        return results

    def visualize_statistical_results(self, results):
        """Create visualizations for statistical test results."""
        # 1. Effect Size Plot
        effect_sizes_df = pd.DataFrame.from_dict(
            {k: v['cohens_d'] for k, v in results['effect_sizes'].items()}, 
            orient='index',
            columns=['Effect Size']
        )
        
        fig = go.Figure(go.Bar(
            x=effect_sizes_df.index,
            y=effect_sizes_df['Effect Size'],
            text=effect_sizes_df['Effect Size'].round(3),
            textposition='auto',
        ))
        
        fig.update_layout(
            title='Effect Sizes (Cohen\'s d) Across Features',
            xaxis_title='Feature',
            yaxis_title='Effect Size',
            template='plotly_white'
        )
        
        fig.write_html(os.path.join(self.viz_dir, 'effect_sizes.html'))

        # 2. Correlation Heatmap
        corr_matrix = np.array(results['multivariate_tests']['spearman_correlation']['correlation_matrix'])
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix,
            x=self.numerical_features,
            y=self.numerical_features,
            colorscale='RdBu',
            zmid=0
        ))
        
        fig.update_layout(
            title='Spearman Correlation Heatmap',
            template='plotly_white'
        )
        
        fig.write_html(os.path.join(self.viz_dir, 'correlation_heatmap.html'))

        # 3. VIF Analysis
        vif_data = pd.DataFrame(results['regression_diagnostics']['multicollinearity'])
        
        fig = go.Figure(go.Bar(
            x=vif_data['Feature'],
            y=vif_data['VIF'],
            text=vif_data['VIF'].round(2),
            textposition='auto',
        ))
        
        fig.update_layout(
            title='Variance Inflation Factors',
            xaxis_title='Feature',
            yaxis_title='VIF',
            template='plotly_white'
        )
        
        fig.write_html(os.path.join(self.viz_dir, 'vif_analysis.html'))

        # 4. Normality Test Results
        normality_results = pd.DataFrame(
            [(feature, data['normality']['shapiro']['p_value']) 
             for feature, data in results['univariate_tests'].items()],
            columns=['Feature', 'Shapiro p-value']
        )
        
        fig = go.Figure(go.Bar(
            x=normality_results['Feature'],
            y=-np.log10(normality_results['Shapiro p-value']),
            text=normality_results['Shapiro p-value'].round(4),
            textposition='auto',
        ))
        
        fig.update_layout(
            title='Normality Test Results (-log10 p-value)',
            xaxis_title='Feature',
            yaxis_title='-log10(p-value)',
            template='plotly_white'
        )
        
        fig.write_html(os.path.join(self.viz_dir, 'normality_tests.html'))

def main():
    """Run advanced analysis."""
    analysis = AdvancedAnalysis('../data/student_habits_performance.csv')
    
    print("Performing detailed statistical analysis...")
    stats_results = analysis.perform_detailed_statistical_analysis()
    
    print("\nPerforming advanced statistical tests...")
    advanced_stats = analysis.perform_advanced_statistical_tests()
    
    print("\nVisualizing statistical results...")
    analysis.visualize_statistical_results(advanced_stats)
    
    print("\nAnalyzing student profiles...")
    profiles = analysis.analyze_student_profiles()
    
    print("\nAnalyzing success factors...")
    success_factors = analysis.analyze_success_factors()
    
    print("\nPerforming multiple regression analysis...")
    regression_results = analysis.perform_multiple_regression()
    
    print("\nPerforming cluster analysis...")
    cluster_profiles = analysis.perform_cluster_analysis()
    
    print("\nIdentifying success patterns...")
    success_patterns = analysis.identify_success_patterns()
    
    print("\nCreating interactive dashboard...")
    analysis.create_interactive_dashboard()
    
    print("\nAnalysis complete. All visualizations have been saved to the 'visualizations' directory.")
    
    return {
        'statistical_analysis': stats_results,
        'advanced_statistics': advanced_stats,
        'student_profiles': profiles,
        'success_factors': success_factors,
        'regression_results': regression_results,
        'cluster_profiles': cluster_profiles,
        'success_patterns': success_patterns
    }

if __name__ == "__main__":
    main() 