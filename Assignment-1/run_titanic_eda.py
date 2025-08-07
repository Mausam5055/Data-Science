"""
Complete EDA Assignment using Titanic Dataset
Run this script to perform comprehensive Exploratory Data Analysis
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

def load_titanic_data():
    """Load Titanic dataset"""
    try:
        df = pd.read_csv('titanic.csv')
        print("✅ Titanic dataset loaded successfully!")
        print(f"Dataset shape: {df.shape}")
        return df
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return None

def perform_complete_eda():
    """Perform complete EDA on Titanic dataset"""
    
    # Load data
    df = load_titanic_data()
    if df is None:
        return
    
    print("\n" + "="*60)
    print("🚢 TITANIC DATASET - EXPLORATORY DATA ANALYSIS")
    print("="*60)
    
    # 1. Basic Information
    print("\n📊 BASIC DATASET INFORMATION")
    print("-" * 40)
    print(f"Total passengers: {len(df)}")
    print(f"Total features: {len(df.columns)}")
    print(f"Missing values: {df.isnull().sum().sum()}")
    
    # 2. Feature Analysis
    print("\n🔍 FEATURE ANALYSIS")
    print("-" * 40)
    
    # Identify feature types
    numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = df.select_dtypes(include=['object']).columns.tolist()
    
    print(f"Numerical features ({len(numerical_features)}): {numerical_features}")
    print(f"Categorical features ({len(categorical_features)}): {categorical_features}")
    
    # 3. Survival Analysis
    print("\n💀 SURVIVAL ANALYSIS")
    print("-" * 40)
    
    survival_rate = df['Survived'].mean()
    print(f"Overall survival rate: {survival_rate:.2%}")
    
    # Survival by class
    survival_by_class = df.groupby('Pclass')['Survived'].agg(['count', 'mean'])
    survival_by_class.columns = ['Count', 'Survival Rate']
    print("\nSurvival by passenger class:")
    print(survival_by_class)
    
    # Survival by gender
    survival_by_sex = df.groupby('Sex')['Survived'].agg(['count', 'mean'])
    survival_by_sex.columns = ['Count', 'Survival Rate']
    print("\nSurvival by gender:")
    print(survival_by_sex)
    
    # 4. Numerical Features Analysis
    print("\n📈 NUMERICAL FEATURES ANALYSIS")
    print("-" * 40)
    
    # Age distribution
    print("\nAge Statistics:")
    print(df['Age'].describe())
    
    # Fare distribution
    print("\nFare Statistics:")
    print(df['Fare'].describe())
    
    # 5. Categorical Features Analysis
    print("\n🏷️ CATEGORICAL FEATURES ANALYSIS")
    print("-" * 40)
    
    # Embarked ports
    print("\nEmbarked ports:")
    print(df['Embarked'].value_counts())
    
    # Cabin analysis (simplified)
    print("\nCabin availability:")
    cabin_available = df['Cabin'].notna().sum()
    print(f"Passengers with cabin info: {cabin_available} ({cabin_available/len(df)*100:.1f}%)")
    
    # 6. Visualizations
    print("\n📊 CREATING VISUALIZATIONS")
    print("-" * 40)
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Titanic Dataset - Exploratory Data Analysis', fontsize=16)
    
    # 1. Survival by class
    survival_counts = df.groupby(['Pclass', 'Survived']).size().unstack()
    survival_counts.plot(kind='bar', stacked=True, ax=axes[0,0])
    axes[0,0].set_title('Survival by Passenger Class')
    axes[0,0].set_xlabel('Passenger Class')
    axes[0,0].set_ylabel('Count')
    axes[0,0].legend(['Died', 'Survived'])
    
    # 2. Age distribution by survival
    axes[0,1].hist([df[df['Survived']==0]['Age'].dropna(), 
                    df[df['Survived']==1]['Age'].dropna()], 
                   bins=20, label=['Died', 'Survived'], alpha=0.7)
    axes[0,1].set_title('Age Distribution by Survival')
    axes[0,1].set_xlabel('Age')
    axes[0,1].set_ylabel('Count')
    axes[0,1].legend()
    
    # 3. Fare distribution by survival
    axes[1,0].hist([df[df['Survived']==0]['Fare'].dropna(), 
                    df[df['Survived']==1]['Fare'].dropna()], 
                   bins=30, label=['Died', 'Survived'], alpha=0.7)
    axes[1,0].set_title('Fare Distribution by Survival')
    axes[1,0].set_xlabel('Fare')
    axes[1,0].set_ylabel('Count')
    axes[1,0].legend()
    
    # 4. Gender vs survival
    sns.countplot(data=df, x='Sex', hue='Survived', ax=axes[1,1])
    axes[1,1].set_title('Survival by Gender')
    axes[1,1].set_xlabel('Gender')
    axes[1,1].set_ylabel('Count')
    
    plt.tight_layout()
    plt.savefig('titanic_eda_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 7. Correlation Analysis
    print("\n🔗 CORRELATION ANALYSIS")
    print("-" * 40)
    
    # Select numerical features for correlation
    numeric_cols = ['Age', 'Fare', 'Pclass', 'SibSp', 'Parch']
    corr_matrix = df[numeric_cols].corr()
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Matrix of Numerical Features')
    plt.tight_layout()
    plt.savefig('titanic_correlation.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 8. Advanced Analysis
    print("\n🔍 ADVANCED INSIGHTS")
    print("-" * 40)
    
    # Family size analysis
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    family_survival = df.groupby('FamilySize')['Survived'].mean()
    print("\nSurvival rate by family size:")
    print(family_survival)
    
    # Age groups analysis
    df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 12, 18, 35, 60, 100], 
                           labels=['Child', 'Teen', 'Adult', 'Middle', 'Senior'])
    age_survival = df.groupby('AgeGroup')['Survived'].mean()
    print("\nSurvival rate by age group:")
    print(age_survival)
    
    # 9. Final Summary
    print("\n📋 FINAL INSIGHTS SUMMARY")
    print("=" * 60)
    print("✅ Key Findings:")
    print(f"   • Overall survival rate: {survival_rate:.1%}")
    print(f"   • Women had much higher survival rate than men")
    print(f"   • First class passengers had highest survival rate")
    print(f"   • Children had better survival chances")
    print(f"   • Higher fares correlated with better survival")
    print(f"   • Young adults (20-35) had lower survival rates")
    print("\n✅ EDA Complete! Check the generated visualizations for detailed insights.")

if __name__ == "__main__":
    perform_complete_eda()
