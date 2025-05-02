"""
Student Habits vs Academic Performance Analysis
This script analyzes the relationships between student habits and academic performance.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import os

class StudentAnalysis:
    def __init__(self, data_path):
        """Initialize the analysis with the dataset path."""
        self.df = pd.read_csv(data_path)
        self.numerical_columns = [
            'age', 'study_hours_per_day', 'sleep_hours', 'social_media_hours',
            'netflix_hours', 'attendance_percentage', 'exercise_frequency',
            'mental_health_rating', 'exam_score'
        ]
        self.categorical_columns = [
            'gender', 'diet_quality', 'parental_education_level',
            'internet_quality', 'part_time_job', 'extracurricular_participation'
        ]
        # Create visualizations directory if it doesn't exist
        self.viz_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'visualizations')
        os.makedirs(self.viz_dir, exist_ok=True)
        
    def save_plot(self, plot_type, fig=None):
        """Save the current plot or specified figure to the visualizations directory."""
        if fig is None:
            plt.savefig(os.path.join(self.viz_dir, f'{plot_type}.png'), 
                       bbox_inches='tight', dpi=300)
        else:
            fig.write_html(os.path.join(self.viz_dir, f'{plot_type}.html'))
        
    def show_basic_stats(self):
        """Display basic statistics and data quality information."""
        print("Dataset Overview:")
        print("-" * 50)
        print(f"Number of students: {self.df.shape[0]}")
        print(f"Number of features: {self.df.shape[1]}")
        print("\nBasic Statistics:")
        print(self.df[self.numerical_columns].describe())
        print("\nMissing Values:")
        print(self.df.isnull().sum())
        
    def plot_individual_numerical_distributions(self):
        """Create individual distribution plots for each numerical feature."""
        plt.style.use('seaborn-v0_8-darkgrid')
        
        for feature in self.numerical_columns:
            fig = plt.figure(figsize=(10, 6))
            
            # Create the histogram with KDE
            sns.histplot(data=self.df, x=feature, kde=True, color='#66B2FF', alpha=0.7)
            
            # Customize the plot
            plt.title(f'Distribution of {feature.replace("_", " ").title()}', 
                     pad=20, fontsize=12, fontweight='bold')
            
            # Add mean and median lines
            mean_val = self.df[feature].mean()
            median_val = self.df[feature].median()
            plt.axvline(mean_val, color='red', linestyle='--', alpha=0.5, label=f'Mean: {mean_val:.2f}')
            plt.axvline(median_val, color='green', linestyle='--', alpha=0.5, label=f'Median: {median_val:.2f}')
            
            plt.legend(loc='upper right', fontsize=10)
            plt.xlabel(feature.replace("_", " ").title(), fontsize=10, labelpad=10)
            plt.ylabel("Count", fontsize=10, labelpad=10)
            
            plt.tight_layout()
            self.save_plot(f'distribution_{feature}')
            plt.close()
        
        plt.style.use('seaborn-v0_8')
    
    def plot_individual_categorical_distributions(self):
        """Create individual distribution plots for each categorical feature."""
        plt.style.use('seaborn-v0_8-darkgrid')
        
        for feature in self.categorical_columns:
            fig = plt.figure(figsize=(10, 6))
            
            # Create the count plot
            ax = sns.countplot(data=self.df, x=feature, color='#FF9999', alpha=0.7)
            
            # Customize the plot
            plt.title(f'Distribution of {feature.replace("_", " ").title()}',
                     pad=20, fontsize=12, fontweight='bold')
            
            # Rotate x-axis labels if needed
            if max([len(str(item.get_text())) for item in ax.get_xticklabels()]) > 8:
                plt.xticks(rotation=45, ha='right')
            
            # Add value labels on top of each bar
            for p in ax.patches:
                ax.annotate(f'{int(p.get_height())}', 
                          (p.get_x() + p.get_width()/2., p.get_height()),
                          ha='center', va='bottom')
            
            plt.xlabel(feature.replace("_", " ").title(), fontsize=10, labelpad=10)
            plt.ylabel("Count", fontsize=10, labelpad=10)
            
            plt.tight_layout()
            self.save_plot(f'distribution_{feature}')
            plt.close()
        
        plt.style.use('seaborn-v0_8')
    
    def plot_individual_exam_relationships(self):
        """Create individual relationship plots for each feature vs exam score."""
        plt.style.use('seaborn-v0_8-darkgrid')
        
        # Numerical features
        for feature in self.numerical_columns[:-1]:  # Exclude exam_score
            fig = plt.figure(figsize=(10, 6))
            
            # Create scatter plot with regression line
            sns.regplot(data=self.df, x=feature, y='exam_score', 
                       color='#66B2FF', scatter_kws={'alpha':0.5})
            
            # Add correlation coefficient
            corr = self.df[feature].corr(self.df['exam_score'])
            plt.title(f'{feature.replace("_", " ").title()} vs Exam Score\n' +
                     f'Correlation: {corr:.2f}',
                     pad=20, fontsize=12, fontweight='bold')
            
            plt.xlabel(feature.replace("_", " ").title(), fontsize=10, labelpad=10)
            plt.ylabel("Exam Score", fontsize=10, labelpad=10)
            
            plt.tight_layout()
            self.save_plot(f'relationship_{feature}_exam_score')
            plt.close()
        
        # Categorical features
        for feature in self.categorical_columns:
            fig = plt.figure(figsize=(10, 6))
            
            # Create box plot
            sns.boxplot(data=self.df, x=feature, y='exam_score', color='#FF9999')
            
            # Rotate x-axis labels if needed
            if max([len(str(item.get_text())) for item in plt.gca().get_xticklabels()]) > 8:
                plt.xticks(rotation=45, ha='right')
            
            plt.title(f'{feature.replace("_", " ").title()} vs Exam Score',
                     pad=20, fontsize=12, fontweight='bold')
            plt.xlabel(feature.replace("_", " ").title(), fontsize=10, labelpad=10)
            plt.ylabel("Exam Score", fontsize=10, labelpad=10)
            
            plt.tight_layout()
            self.save_plot(f'relationship_{feature}_exam_score')
            plt.close()
        
        plt.style.use('seaborn-v0_8')
        
    def plot_distributions(self):
        """Plot distributions of numerical features with enhanced styling."""
        plt.style.use('seaborn-v0_8-darkgrid')
        
        fig = plt.figure(figsize=(20, 18))
        
        colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#FF99CC', '#99CCFF', '#FFB366', '#99FF99', '#FF99FF']
        
        for i, (feature, color) in enumerate(zip(self.numerical_columns, colors), 1):
            ax = plt.subplot(3, 3, i)
            
            sns.histplot(data=self.df, x=feature, kde=True, color=color, alpha=0.7)
            
            plt.title(f'Distribution of {feature.replace("_", " ").title()}', 
                     pad=20, fontsize=12, fontweight='bold')
            
            plt.xticks(rotation=30, ha='right')
            
            ax.grid(True, alpha=0.3)
            
            mean_val = self.df[feature].mean()
            median_val = self.df[feature].median()
            plt.axvline(mean_val, color='red', linestyle='--', alpha=0.5, label=f'Mean: {mean_val:.2f}')
            plt.axvline(median_val, color='green', linestyle='--', alpha=0.5, label=f'Median: {median_val:.2f}')
            
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
            
            plt.xlabel(feature.replace("_", " ").title(), fontsize=10, labelpad=10)
            plt.ylabel("Count", fontsize=10, labelpad=10)
            
        plt.tight_layout(pad=3.0, h_pad=0.8, w_pad=0.8)
        self.save_plot('numerical_distributions')
        plt.show()
        
        plt.style.use('seaborn-v0_8')
        
    def plot_categorical_distributions(self):
        """Plot distributions of categorical features with enhanced styling."""
        plt.style.use('seaborn-v0_8-darkgrid')
        
        fig = plt.figure(figsize=(20, 15))
        
        colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#FF99CC', '#99CCFF']
        
        for i, (feature, color) in enumerate(zip(self.categorical_columns, colors), 1):
            ax = plt.subplot(2, 3, i)
            
            sns.countplot(data=self.df, x=feature, color=color, alpha=0.7)
            
            plt.title(f'Distribution of {feature.replace("_", " ").title()}',
                     pad=20, fontsize=12, fontweight='bold')
            
            plt.xticks(rotation=45, ha='right')
            
            for p in ax.patches:
                ax.annotate(f'{int(p.get_height())}', 
                          (p.get_x() + p.get_width()/2., p.get_height()),
                          ha='center', va='bottom')
            
            ax.grid(True, alpha=0.3)
            
            plt.xlabel(feature.replace("_", " ").title(), fontsize=10, labelpad=10)
            plt.ylabel("Count", fontsize=10, labelpad=10)
        
        plt.tight_layout(pad=3.0, h_pad=1.0, w_pad=0.8)
        self.save_plot('categorical_distributions')
        plt.show()
        
        plt.style.use('seaborn-v0_8')
        
    def plot_correlation_heatmap(self):
        """Plot correlation heatmap for numerical features."""
        plt.figure(figsize=(12, 10))
        corr_matrix = self.df[self.numerical_columns].corr()
        mask = np.triu(np.ones_like(corr_matrix), k=1)
        sns.heatmap(corr_matrix, 
                   mask=mask,
                   annot=True,
                   fmt='.2f',
                   cmap='RdBu_r',
                   center=0,
                   square=True,
                   linewidths=1)
        plt.title('Correlation Heatmap of Student Habits', pad=20, size=16)
        plt.tight_layout()
        self.save_plot('correlation_heatmap')
        plt.show()
        
    def plot_exam_score_relationships(self):
        """Plot relationships between features and exam scores."""
        # Numerical features
        plt.figure(figsize=(20, 15))
        for i, feature in enumerate(self.numerical_columns[:-1], 1):
            plt.subplot(3, 3, i)
            sns.scatterplot(data=self.df, x=feature, y='exam_score')
            plt.title(f'{feature.replace("_", " ").title()} vs Exam Score')
            plt.xlabel(feature.replace("_", " ").title())
            plt.ylabel('Exam Score')
        plt.tight_layout()
        self.save_plot('numerical_vs_exam_score')
        plt.show()
        
        # Categorical features
        plt.figure(figsize=(20, 15))
        for i, feature in enumerate(self.categorical_columns, 1):
            plt.subplot(2, 3, i)
            sns.boxplot(data=self.df, x=feature, y='exam_score')
            plt.title(f'{feature.replace("_", " ").title()} vs Exam Score')
            plt.xticks(rotation=45)
            plt.xlabel(feature.replace("_", " ").title())
            plt.ylabel('Exam Score')
        plt.tight_layout()
        self.save_plot('categorical_vs_exam_score')
        plt.show()
        
    def create_scatter_matrix(self):
        """Create an interactive scatter matrix of key variables."""
        key_vars = ['exam_score', 'study_hours_per_day', 'mental_health_rating', 'sleep_hours']
        fig = px.scatter_matrix(
            self.df,
            dimensions=key_vars,
            color='diet_quality',
            title='Scatter Matrix of Key Variables'
        )
        fig.update_layout(title_x=0.5)
        self.save_plot('scatter_matrix', fig)
        fig.show()
        
    def analyze_correlations(self):
        """Print detailed correlation analysis with exam scores."""
        correlations = self.df[self.numerical_columns].corr()['exam_score'].sort_values(ascending=False)
        print("Correlations with Exam Score:")
        print("-" * 50)
        for feature, corr in correlations.items():
            if feature != 'exam_score':
                strength = 'strong' if abs(corr) > 0.5 else 'moderate' if abs(corr) > 0.3 else 'weak'
                direction = 'positive' if corr > 0 else 'negative'
                print(f"{feature.replace('_', ' ').title()}: {corr:.3f} ({direction} {strength} correlation)")

    def plot_individual_numerical_distributions_enhanced(self):
        """Create enhanced individual distribution plots for each numerical feature."""
        # Set style for enhanced plots
        plt.style.use('seaborn-v0_8-whitegrid')
        
        for feature in self.numerical_columns:
            fig = plt.figure(figsize=(12, 8))
            
            # Create the main plot
            ax = plt.gca()
            
            # Plot histogram with KDE
            sns.histplot(data=self.df, x=feature, kde=True, 
                        color='#4A90E2', alpha=0.6, 
                        line_kws={'color': '#2E5C88', 'linewidth': 2})
            
            # Add mean and median lines with annotations
            mean_val = self.df[feature].mean()
            median_val = self.df[feature].median()
            
            # Add vertical lines
            plt.axvline(mean_val, color='#E74C3C', linestyle='--', alpha=0.8, linewidth=2)
            plt.axvline(median_val, color='#2ECC71', linestyle='--', alpha=0.8, linewidth=2)
            
            # Add annotations with background
            bbox_props = dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.8)
            plt.annotate(f'Mean: {mean_val:.2f}', 
                        xy=(mean_val, plt.ylim()[1]), 
                        xytext=(10, 10), textcoords='offset points',
                        ha='left', va='bottom',
                        bbox=bbox_props, color='#E74C3C')
            plt.annotate(f'Median: {median_val:.2f}', 
                        xy=(median_val, plt.ylim()[1]*0.95), 
                        xytext=(10, -20), textcoords='offset points',
                        ha='left', va='bottom',
                        bbox=bbox_props, color='#2ECC71')
            
            # Customize the plot
            plt.title(f'Distribution of {feature.replace("_", " ").title()}', 
                     pad=20, fontsize=16, fontweight='bold', fontfamily='sans-serif')
            
            # Style the axes
            plt.xlabel(feature.replace("_", " ").title(), fontsize=12, fontweight='bold', fontfamily='sans-serif', labelpad=15)
            plt.ylabel("Frequency", fontsize=12, fontweight='bold', fontfamily='sans-serif', labelpad=15)
            
            # Customize grid
            ax.grid(True, alpha=0.3, linestyle='--')
            
            # Remove top and right spines
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            # Add a subtle background color
            ax.set_facecolor('#F8F9FA')
            fig.patch.set_facecolor('white')
            
            plt.tight_layout()
            self.save_plot(f'distribution_{feature}_enhanced')
            plt.close()
    
    def plot_individual_categorical_distributions_enhanced(self):
        """Create enhanced individual distribution plots for each categorical feature."""
        plt.style.use('seaborn-v0_8-whitegrid')
        
        colors = ['#4A90E2', '#50C878', '#E74C3C', '#F39C12', '#9B59B6', '#34495E']
        
        for feature in self.categorical_columns:
            fig = plt.figure(figsize=(12, 8))
            ax = plt.gca()
            
            # Calculate percentages
            value_counts = self.df[feature].value_counts()
            percentages = (value_counts / len(self.df) * 100).round(1)
            
            # Create the count plot
            sns.barplot(x=value_counts.index, y=value_counts.values, 
                       palette=colors[:len(value_counts)], alpha=0.8)
            
            # Add percentage labels on top of bars
            for i, (count, percentage) in enumerate(zip(value_counts, percentages)):
                ax.text(i, count, f'{count}\n({percentage}%)', 
                       ha='center', va='bottom', fontsize=10,
                       bbox=dict(boxstyle='round,pad=0.5', fc='white', ec='gray', alpha=0.8))
            
            # Customize the plot
            plt.title(f'Distribution of {feature.replace("_", " ").title()}',
                     pad=20, fontsize=16, fontweight='bold', fontfamily='sans-serif')
            
            # Style the axes
            plt.xlabel(feature.replace("_", " ").title(), fontsize=12, fontweight='bold', fontfamily='sans-serif', labelpad=15)
            plt.ylabel("Count", fontsize=12, fontweight='bold', fontfamily='sans-serif', labelpad=15)
            
            # Rotate x-axis labels if needed
            if max([len(str(item.get_text())) for item in ax.get_xticklabels()]) > 8:
                plt.xticks(rotation=45, ha='right')
            
            # Customize grid
            ax.grid(True, alpha=0.3, linestyle='--', axis='y')
            
            # Remove top and right spines
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            # Add a subtle background color
            ax.set_facecolor('#F8F9FA')
            fig.patch.set_facecolor('white')
            
            plt.tight_layout()
            self.save_plot(f'distribution_{feature}_enhanced')
            plt.close()
    
    def plot_individual_exam_relationships_enhanced(self):
        """Create enhanced individual relationship plots for each feature vs exam score."""
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # Numerical features
        for feature in self.numerical_columns[:-1]:  # Exclude exam_score
            fig = plt.figure(figsize=(12, 8))
            ax = plt.gca()
            
            # Create scatter plot with regression line
            sns.regplot(data=self.df, x=feature, y='exam_score',
                       scatter_kws={'alpha':0.5, 'color':'#4A90E2'},
                       line_kws={'color': '#E74C3C', 'linewidth': 2})
            
            # Calculate correlation and p-value
            corr, p_value = stats.pearsonr(self.df[feature], self.df['exam_score'])
            
            # Add correlation information
            plt.text(0.05, 0.95, f'Correlation: {corr:.2f}\np-value: {p_value:.3f}',
                    transform=ax.transAxes,
                    bbox=dict(boxstyle='round,pad=0.5', fc='white', ec='gray', alpha=0.8),
                    fontsize=10, fontfamily='sans-serif')
            
            # Customize the plot
            plt.title(f'Relationship between {feature.replace("_", " ").title()} and Exam Score',
                     pad=20, fontsize=16, fontweight='bold', fontfamily='sans-serif')
            
            # Style the axes
            plt.xlabel(feature.replace("_", " ").title(), fontsize=12, fontweight='bold', fontfamily='sans-serif', labelpad=15)
            plt.ylabel("Exam Score", fontsize=12, fontweight='bold', fontfamily='sans-serif', labelpad=15)
            
            # Customize grid
            ax.grid(True, alpha=0.3, linestyle='--')
            
            # Remove top and right spines
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            # Add a subtle background color
            ax.set_facecolor('#F8F9FA')
            fig.patch.set_facecolor('white')
            
            plt.tight_layout()
            self.save_plot(f'relationship_{feature}_exam_score_enhanced')
            plt.close()
        
        # Categorical features
        for feature in self.categorical_columns:
            fig = plt.figure(figsize=(12, 8))
            ax = plt.gca()
            
            # Create violin plot with box plot inside
            sns.violinplot(data=self.df, x=feature, y='exam_score',
                         palette='husl', inner='box', alpha=0.7)
            
            # Add individual points
            sns.stripplot(data=self.df, x=feature, y='exam_score',
                        color='#2C3E50', alpha=0.3, size=4, jitter=0.2)
            
            # Calculate and add mean scores for each category
            means = self.df.groupby(feature)['exam_score'].mean()
            for i, mean_val in enumerate(means):
                ax.text(i, ax.get_ylim()[0], f'Mean: {mean_val:.1f}',
                       ha='center', va='bottom', fontsize=10,
                       bbox=dict(boxstyle='round,pad=0.5', fc='white', ec='gray', alpha=0.8))
            
            # Customize the plot
            plt.title(f'Exam Scores by {feature.replace("_", " ").title()}',
                     pad=20, fontsize=16, fontweight='bold', fontfamily='sans-serif')
            
            # Style the axes
            plt.xlabel(feature.replace("_", " ").title(), fontsize=12, fontweight='bold', fontfamily='sans-serif', labelpad=15)
            plt.ylabel("Exam Score", fontsize=12, fontweight='bold', fontfamily='sans-serif', labelpad=15)
            
            # Rotate x-axis labels if needed
            if max([len(str(item.get_text())) for item in ax.get_xticklabels()]) > 8:
                plt.xticks(rotation=45, ha='right')
            
            # Customize grid
            ax.grid(True, alpha=0.3, linestyle='--', axis='y')
            
            # Remove top and right spines
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            # Add a subtle background color
            ax.set_facecolor('#F8F9FA')
            fig.patch.set_facecolor('white')
            
            plt.tight_layout()
            self.save_plot(f'relationship_{feature}_exam_score_enhanced')
            plt.close()

def main():
    """Main function to run the analysis."""
    # Set the style for all plots
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # Create analysis object
    analysis = StudentAnalysis('../data/student_habits_performance.csv')
    
    # Run analyses
    analysis.show_basic_stats()
    
    # Create enhanced individual plots
    print("\nGenerating enhanced distribution plots...")
    analysis.plot_individual_numerical_distributions_enhanced()
    analysis.plot_individual_categorical_distributions_enhanced()
    
    print("Generating enhanced relationship plots...")
    analysis.plot_individual_exam_relationships_enhanced()
    
    # Create overview plots
    print("Generating overview plots...")
    analysis.plot_correlation_heatmap()
    analysis.create_scatter_matrix()
    
    # Run correlation analysis
    print("\nPerforming correlation analysis...")
    analysis.analyze_correlations()
    
    print("\nAll visualizations have been saved to the 'visualizations' directory.")

if __name__ == "__main__":
    main() 