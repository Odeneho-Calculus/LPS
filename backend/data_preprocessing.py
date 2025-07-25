"""
Advanced Data Preprocessing Module for Loan Prediction System
Handles data cleaning, feature engineering, and preprocessing operations.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class LoanDataProcessor:
    """
    Professional-grade data preprocessing pipeline for loan prediction.
    Implements comprehensive data cleaning, feature engineering, and validation.
    """

    def __init__(self, data_path=None):
        self.data_path = data_path
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.target_column = 'Loan_Status'
        self.processed_data = None

    def load_data(self, data_path=None):
        """Load dataset with comprehensive error handling and validation."""
        try:
            path = data_path or self.data_path
            if not path:
                raise ValueError("Data path must be provided")

            self.raw_data = pd.read_csv(path)
            print(f"✓ Dataset loaded successfully: {self.raw_data.shape}")
            return self.raw_data

        except FileNotFoundError:
            raise FileNotFoundError(f"Dataset not found at: {path}")
        except Exception as e:
            raise Exception(f"Error loading data: {str(e)}")

    def explore_data(self):
        """Comprehensive data exploration and analysis."""
        if self.raw_data is None:
            raise ValueError("Data must be loaded first")

        print("\n" + "="*60)
        print("📊 COMPREHENSIVE DATA EXPLORATION REPORT")
        print("="*60)

        # Basic information
        print(f"\n📈 Dataset Shape: {self.raw_data.shape}")
        print(f"📈 Memory Usage: {self.raw_data.memory_usage(deep=True).sum() / 1024:.2f} KB")

        # Data types
        print("\n🔍 Data Types:")
        print(self.raw_data.dtypes)

        # Missing values analysis
        print("\n❌ Missing Values Analysis:")
        missing_data = self.raw_data.isnull().sum()
        missing_percent = (missing_data / len(self.raw_data)) * 100
        missing_df = pd.DataFrame({
            'Missing Count': missing_data,
            'Percentage': missing_percent
        }).sort_values('Missing Count', ascending=False)
        print(missing_df[missing_df['Missing Count'] > 0])

        # Target variable distribution
        print(f"\n🎯 Target Variable Distribution:")
        target_dist = self.raw_data[self.target_column].value_counts()
        print(target_dist)
        print(f"Approval Rate: {(target_dist['Y'] / target_dist.sum() * 100):.2f}%")

        # Statistical summary
        print("\n📊 Statistical Summary:")
        print(self.raw_data.describe())

        return missing_df

    def handle_missing_values(self):
        """Advanced missing value imputation with domain knowledge."""
        data = self.raw_data.copy()

        print("\n🔧 Handling Missing Values...")

        # Handle categorical variables
        categorical_columns = ['Gender', 'Married', 'Dependents', 'Self_Employed']

        for col in categorical_columns:
            if data[col].isnull().sum() > 0:
                mode_value = data[col].mode()[0]
                data[col].fillna(mode_value, inplace=True)
                print(f"  ✓ {col}: Filled {data[col].isnull().sum()} missing values with mode: {mode_value}")

        # Handle numerical variables with intelligent imputation
        if data['LoanAmount'].isnull().sum() > 0:
            # Use median based on employment status and income
            for emp_status in data['Self_Employed'].unique():
                mask = (data['Self_Employed'] == emp_status) & (data['LoanAmount'].isnull())
                if mask.sum() > 0:
                    median_amount = data[data['Self_Employed'] == emp_status]['LoanAmount'].median()
                    data.loc[mask, 'LoanAmount'] = median_amount
            print(f"  ✓ LoanAmount: Intelligent imputation based on employment status")

        # Handle Loan_Amount_Term
        if data['Loan_Amount_Term'].isnull().sum() > 0:
            mode_term = data['Loan_Amount_Term'].mode()[0]
            data['Loan_Amount_Term'].fillna(mode_term, inplace=True)
            print(f"  ✓ Loan_Amount_Term: Filled with mode: {mode_term}")

        # Handle Credit_History (critical feature)
        if data['Credit_History'].isnull().sum() > 0:
            # Use sophisticated imputation based on approval patterns
            approval_rate_with_history = data[data['Credit_History'] == 1]['Loan_Status'].value_counts(normalize=True)['Y']
            mode_credit = 1 if approval_rate_with_history > 0.5 else 0
            data['Credit_History'].fillna(mode_credit, inplace=True)
            print(f"  ✓ Credit_History: Filled with strategic value: {mode_credit}")

        self.processed_data = data
        print(f"✅ Missing value handling complete. Remaining nulls: {data.isnull().sum().sum()}")

        return data

    def feature_engineering(self):
        """Advanced feature engineering with domain expertise."""
        if self.processed_data is None:
            raise ValueError("Data preprocessing must be completed first")

        data = self.processed_data.copy()

        print("\n🏗️ Advanced Feature Engineering...")

        # Create derived features
        data['Total_Income'] = data['ApplicantIncome'] + data['CoapplicantIncome']
        data['Income_Loan_Ratio'] = data['Total_Income'] / (data['LoanAmount'] + 1)  # +1 to avoid division by zero
        data['Loan_Per_Month'] = data['LoanAmount'] / (data['Loan_Amount_Term'] / 12)
        data['Income_Per_Dependent'] = data['Total_Income'] / (data['Dependents'].replace('3+', '3').astype(int) + 1)

        # Create categorical features
        data['High_Income'] = (data['Total_Income'] > data['Total_Income'].median()).astype(int)
        data['High_Loan_Amount'] = (data['LoanAmount'] > data['LoanAmount'].median()).astype(int)

        # Log transformation for skewed features
        data['Log_LoanAmount'] = np.log1p(data['LoanAmount'])
        data['Log_Total_Income'] = np.log1p(data['Total_Income'])

        print("  ✓ Created derived income and ratio features")
        print("  ✓ Added categorical indicators")
        print("  ✓ Applied log transformations")

        self.processed_data = data
        return data

    def encode_categorical_variables(self):
        """Professional categorical encoding with proper handling."""
        if self.processed_data is None:
            raise ValueError("Feature engineering must be completed first")

        data = self.processed_data.copy()

        print("\n🔢 Encoding Categorical Variables...")

        # Define categorical columns for encoding
        categorical_cols = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area']

        # Handle Dependents separately (ordinal)
        data['Dependents'] = data['Dependents'].replace('3+', '3').astype(int)

        # Label encode categorical variables
        for col in categorical_cols:
            if col in data.columns:
                le = LabelEncoder()
                data[col] = le.fit_transform(data[col])
                self.label_encoders[col] = le
                print(f"  ✓ Encoded {col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")

        # Encode target variable
        le_target = LabelEncoder()
        data[self.target_column] = le_target.fit_transform(data[self.target_column])
        self.label_encoders[self.target_column] = le_target
        print(f"  ✓ Encoded target: {dict(zip(le_target.classes_, le_target.transform(le_target.classes_)))}")

        self.processed_data = data
        return data

    def prepare_features(self):
        """Prepare final feature set for model training."""
        if self.processed_data is None:
            raise ValueError("Data encoding must be completed first")

        data = self.processed_data.copy()

        # Remove identifier column
        if 'Loan_ID' in data.columns:
            data = data.drop('Loan_ID', axis=1)

        # Define feature columns (everything except target)
        self.feature_columns = [col for col in data.columns if col != self.target_column]

        print(f"\n📋 Final Feature Set ({len(self.feature_columns)} features):")
        for i, col in enumerate(self.feature_columns, 1):
            print(f"  {i:2d}. {col}")

        return data

    def split_data(self, test_size=0.2, random_state=42):
        """Create training and testing splits with stratification."""
        if self.processed_data is None or not self.feature_columns:
            raise ValueError("Data preparation must be completed first")

        data = self.processed_data

        X = data[self.feature_columns]
        y = data[self.target_column]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        print(f"\n✂️ Data Split Summary:")
        print(f"  Training Set: {X_train.shape[0]} samples ({(1-test_size)*100:.0f}%)")
        print(f"  Testing Set:  {X_test.shape[0]} samples ({test_size*100:.0f}%)")
        print(f"  Features: {X_train.shape[1]}")

        return X_train, X_test, y_train, y_test

    def create_visualization_report(self, save_path=None):
        """Generate comprehensive visualization report."""
        if self.processed_data is None:
            raise ValueError("Data must be processed first")

        data = self.raw_data  # Use raw data for better interpretability

        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        fig.suptitle('Comprehensive Loan Data Analysis Report', fontsize=16, fontweight='bold')

        # 1. Target distribution
        data['Loan_Status'].value_counts().plot(kind='bar', ax=axes[0,0], color=['red', 'green'])
        axes[0,0].set_title('Loan Approval Distribution')
        axes[0,0].set_xlabel('Loan Status')
        axes[0,0].set_ylabel('Count')

        # 2. Income distribution
        data['ApplicantIncome'].hist(bins=30, ax=axes[0,1], alpha=0.7)
        axes[0,1].set_title('Applicant Income Distribution')
        axes[0,1].set_xlabel('Income')

        # 3. Loan amount distribution
        data['LoanAmount'].hist(bins=30, ax=axes[0,2], alpha=0.7)
        axes[0,2].set_title('Loan Amount Distribution')
        axes[0,2].set_xlabel('Loan Amount')

        # 4. Credit history impact
        pd.crosstab(data['Credit_History'], data['Loan_Status']).plot(kind='bar', ax=axes[1,0])
        axes[1,0].set_title('Credit History vs Loan Status')
        axes[1,0].set_xlabel('Credit History')

        # 5. Property area impact
        pd.crosstab(data['Property_Area'], data['Loan_Status']).plot(kind='bar', ax=axes[1,1])
        axes[1,1].set_title('Property Area vs Loan Status')
        axes[1,1].set_xlabel('Property Area')

        # 6. Education impact
        pd.crosstab(data['Education'], data['Loan_Status']).plot(kind='bar', ax=axes[1,2])
        axes[1,2].set_title('Education vs Loan Status')
        axes[1,2].set_xlabel('Education')

        # 7. Gender impact
        pd.crosstab(data['Gender'], data['Loan_Status']).plot(kind='bar', ax=axes[2,0])
        axes[2,0].set_title('Gender vs Loan Status')
        axes[2,0].set_xlabel('Gender')

        # 8. Marriage impact
        pd.crosstab(data['Married'], data['Loan_Status']).plot(kind='bar', ax=axes[2,1])
        axes[2,1].set_title('Married vs Loan Status')
        axes[2,1].set_xlabel('Married')

        # 9. Dependents impact
        pd.crosstab(data['Dependents'], data['Loan_Status']).plot(kind='bar', ax=axes[2,2])
        axes[2,2].set_title('Dependents vs Loan Status')
        axes[2,2].set_xlabel('Dependents')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Visualization report saved: {save_path}")

        return fig

    def run_complete_pipeline(self, data_path=None):
        """Execute the complete data preprocessing pipeline."""
        print("🚀 EXECUTING COMPREHENSIVE DATA PREPROCESSING PIPELINE")
        print("="*70)

        try:
            # Step 1: Load data
            self.load_data(data_path)

            # Step 2: Explore data
            self.explore_data()

            # Step 3: Handle missing values
            self.handle_missing_values()

            # Step 4: Feature engineering
            self.feature_engineering()

            # Step 5: Encode categorical variables
            self.encode_categorical_variables()

            # Step 6: Prepare features
            self.prepare_features()

            # Step 7: Split data
            X_train, X_test, y_train, y_test = self.split_data()

            print("\n🎉 PREPROCESSING PIPELINE COMPLETED SUCCESSFULLY!")
            print("="*70)

            return X_train, X_test, y_train, y_test, self.label_encoders

        except Exception as e:
            print(f"\n❌ Pipeline failed: {str(e)}")
            raise

# Usage example and testing
if __name__ == "__main__":
    processor = LoanDataProcessor()
    try:
        X_train, X_test, y_train, y_test, encoders = processor.run_complete_pipeline(
            "../data/loan_dataset.csv"
        )
        print(f"\n✅ Ready for model training with {X_train.shape[1]} features")

    except Exception as e:
        print(f"Error: {e}")