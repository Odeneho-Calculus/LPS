"""
Advanced Machine Learning Model Module for Loan Prediction System
Implements Decision Tree with hyperparameter tuning, evaluation, and deployment features.
"""

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class LoanPredictionModel:
    """
    Professional-grade ML model for loan approval prediction.
    Implements comprehensive training, evaluation, and deployment pipeline.
    """

    def __init__(self, model_type='decision_tree'):
        self.model_type = model_type
        self.model = None
        self.best_params = None
        self.feature_importance = None
        self.feature_names = None
        self.label_encoders = None
        self.training_metrics = {}
        self.cross_val_scores = None

    def initialize_model(self, **kwargs):
        """Initialize the ML model with optimal configuration."""
        print(f"\n🤖 Initializing {self.model_type.title()} Model...")

        if self.model_type == 'decision_tree':
            self.model = DecisionTreeClassifier(
                random_state=42,
                class_weight='balanced',  # Handle class imbalance
                **kwargs
            )
        elif self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                random_state=42,
                class_weight='balanced',
                **kwargs
            )
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        print(f"✅ Model initialized with default parameters")
        return self.model

    def hyperparameter_tuning(self, X_train, y_train, cv_folds=5, scoring='f1', n_jobs=-1):
        """
        Comprehensive hyperparameter optimization using GridSearchCV.
        """
        print(f"\n🔧 HYPERPARAMETER OPTIMIZATION")
        print("="*50)

        if self.model_type == 'decision_tree':
            param_grid = {
                'max_depth': [3, 5, 7, 10, 15, None],
                'min_samples_split': [2, 5, 10, 20],
                'min_samples_leaf': [1, 2, 5, 10],
                'criterion': ['gini', 'entropy'],
                'max_features': ['sqrt', 'log2', None],
                'ccp_alpha': [0.0, 0.01, 0.02, 0.05]
            }
        elif self.model_type == 'random_forest':
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2']
            }

        print(f"🎯 Search space: {sum([len(v) for v in param_grid.values()])} parameter combinations")
        print(f"📊 Cross-validation folds: {cv_folds}")
        print(f"📈 Scoring metric: {scoring}")

        # Initialize model if not done
        if self.model is None:
            self.initialize_model()

        # Perform grid search
        cv_strategy = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

        grid_search = GridSearchCV(
            estimator=self.model,
            param_grid=param_grid,
            cv=cv_strategy,
            scoring=scoring,
            n_jobs=n_jobs,
            verbose=1,
            error_score='raise'
        )

        print("\n🚀 Starting hyperparameter search...")
        grid_search.fit(X_train, y_train)

        # Store results
        self.model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_

        print(f"\n✅ Optimization completed!")
        print(f"📊 Best CV Score: {grid_search.best_score_:.4f}")
        print(f"🎯 Best Parameters:")
        for param, value in self.best_params.items():
            print(f"   {param}: {value}")

        return grid_search

    def train_model(self, X_train, y_train, feature_names=None, label_encoders=None):
        """
        Train the model with comprehensive evaluation and monitoring.
        """
        print(f"\n🏋️ TRAINING {self.model_type.upper()} MODEL")
        print("="*50)

        self.feature_names = feature_names or [f"feature_{i}" for i in range(X_train.shape[1])]
        self.label_encoders = label_encoders

        # Initialize model if not done
        if self.model is None:
            self.initialize_model()

        # Train the model
        start_time = datetime.now()
        self.model.fit(X_train, y_train)
        training_time = (datetime.now() - start_time).total_seconds()

        print(f"✅ Model training completed in {training_time:.2f} seconds")

        # Extract feature importance
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)

            print(f"\n📊 Top 10 Most Important Features:")
            for idx, row in self.feature_importance.head(10).iterrows():
                print(f"   {row['feature']}: {row['importance']:.4f}")

        return self.model

    def cross_validate(self, X_train, y_train, cv_folds=5, scoring_metrics=None):
        """
        Perform comprehensive cross-validation analysis.
        """
        print(f"\n🔄 CROSS-VALIDATION ANALYSIS")
        print("="*40)

        if scoring_metrics is None:
            scoring_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

        cv_strategy = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        cv_results = {}

        for metric in scoring_metrics:
            scores = cross_val_score(
                self.model, X_train, y_train,
                cv=cv_strategy, scoring=metric, n_jobs=-1
            )
            cv_results[metric] = {
                'scores': scores,
                'mean': scores.mean(),
                'std': scores.std(),
                'min': scores.min(),
                'max': scores.max()
            }

            print(f"📈 {metric.upper()}:")
            print(f"   Mean: {scores.mean():.4f} (±{scores.std()*2:.4f})")
            print(f"   Range: [{scores.min():.4f}, {scores.max():.4f}]")

        self.cross_val_scores = cv_results
        return cv_results

    def evaluate_model(self, X_test, y_test, class_names=None):
        """
        Comprehensive model evaluation with detailed metrics and visualizations.
        """
        print(f"\n📊 MODEL EVALUATION REPORT")
        print("="*50)

        # Make predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1] if hasattr(self.model, 'predict_proba') else None

        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1_score': f1_score(y_test, y_pred, average='weighted'),
        }

        if y_pred_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)

        # Store metrics
        self.training_metrics = metrics

        # Print metrics
        print("🎯 Performance Metrics:")
        for metric, score in metrics.items():
            print(f"   {metric.title()}: {score:.4f}")

        # Classification report
        if class_names is None:
            class_names = ['Rejected', 'Approved'] if self.label_encoders else ['0', '1']

        print(f"\n📋 Detailed Classification Report:")
        print(classification_report(y_test, y_pred, target_names=class_names))

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"\n🔢 Confusion Matrix:")
        print(cm)

        return metrics, y_pred, y_pred_proba

    def create_evaluation_visualizations(self, X_test, y_test, save_dir=None):
        """
        Generate comprehensive evaluation visualizations.
        """
        print(f"\n🎨 Generating Evaluation Visualizations...")

        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1] if hasattr(self.model, 'predict_proba') else None

        fig = plt.figure(figsize=(20, 15))

        # 1. Confusion Matrix
        plt.subplot(3, 3, 1)
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Rejected', 'Approved'],
                   yticklabels=['Rejected', 'Approved'])
        plt.title('Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')

        # 2. ROC Curve
        if y_pred_proba is not None:
            plt.subplot(3, 3, 2)
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            plt.plot(fpr, tpr, color='darkorange', lw=2,
                    label=f'ROC Curve (AUC = {roc_auc:.3f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve')
            plt.legend(loc="lower right")

        # 3. Feature Importance
        if self.feature_importance is not None:
            plt.subplot(3, 3, 3)
            top_features = self.feature_importance.head(10)
            plt.barh(range(len(top_features)), top_features['importance'])
            plt.yticks(range(len(top_features)), top_features['feature'])
            plt.xlabel('Importance')
            plt.title('Top 10 Feature Importance')
            plt.gca().invert_yaxis()

        # 4. Prediction Distribution
        plt.subplot(3, 3, 4)
        if y_pred_proba is not None:
            plt.hist(y_pred_proba, bins=30, alpha=0.7, edgecolor='black')
            plt.xlabel('Prediction Probability')
            plt.ylabel('Frequency')
            plt.title('Prediction Probability Distribution')

        # 5. Cross-validation scores visualization
        if self.cross_val_scores:
            plt.subplot(3, 3, 5)
            metrics = list(self.cross_val_scores.keys())
            means = [self.cross_val_scores[m]['mean'] for m in metrics]
            stds = [self.cross_val_scores[m]['std'] for m in metrics]

            plt.bar(metrics, means, yerr=stds, capsize=5, alpha=0.7)
            plt.ylabel('Score')
            plt.title('Cross-Validation Scores')
            plt.xticks(rotation=45)

        # 6. Decision Tree Visualization (if applicable)
        if self.model_type == 'decision_tree' and hasattr(self.model, 'tree_'):
            plt.subplot(3, 3, 6)
            plot_tree(self.model, max_depth=3, feature_names=self.feature_names[:10],
                     class_names=['Rejected', 'Approved'], filled=True, fontsize=8)
            plt.title('Decision Tree Structure (Depth=3)')

        # 7. Model Performance Comparison
        plt.subplot(3, 3, 7)
        performance_metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        performance_values = [
            self.training_metrics.get('accuracy', 0),
            self.training_metrics.get('precision', 0),
            self.training_metrics.get('recall', 0),
            self.training_metrics.get('f1_score', 0)
        ]

        bars = plt.bar(performance_metrics, performance_values, color=['blue', 'green', 'orange', 'red'], alpha=0.7)
        plt.ylim(0, 1)
        plt.ylabel('Score')
        plt.title('Model Performance Metrics')
        plt.xticks(rotation=45)

        # Add value labels on bars
        for bar, value in zip(bars, performance_values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')

        plt.tight_layout()

        if save_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f"{save_dir}/model_evaluation_{timestamp}.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Visualizations saved: {save_path}")

        return fig

    def save_model(self, model_path, include_metadata=True):
        """
        Save trained model with comprehensive metadata.
        """
        print(f"\n💾 Saving Model...")

        model_data = {
            'model': self.model,
            'model_type': self.model_type,
            'feature_names': self.feature_names,
            'label_encoders': self.label_encoders,
            'best_params': self.best_params,
            'training_metrics': self.training_metrics,
            'feature_importance': self.feature_importance,
            'timestamp': datetime.now().isoformat()
        }

        if include_metadata:
            model_data.update({
                'cross_val_scores': self.cross_val_scores,
                'model_version': '1.0.0',
                'sklearn_version': '1.3.0'
            })

        joblib.dump(model_data, model_path)
        file_size = os.path.getsize(model_path) / 1024  # KB

        print(f"✅ Model saved successfully:")
        print(f"   Path: {model_path}")
        print(f"   Size: {file_size:.2f} KB")
        print(f"   Timestamp: {model_data['timestamp']}")

        return model_path

    def load_model(self, model_path):
        """
        Load saved model with all metadata.
        """
        print(f"\n📂 Loading Model from: {model_path}")

        try:
            model_data = joblib.load(model_path)

            self.model = model_data['model']
            self.model_type = model_data['model_type']
            self.feature_names = model_data.get('feature_names')
            self.label_encoders = model_data.get('label_encoders')
            self.best_params = model_data.get('best_params')
            self.training_metrics = model_data.get('training_metrics', {})
            self.feature_importance = model_data.get('feature_importance')

            print("✅ Model loaded successfully:")
            print(f"   Type: {self.model_type}")
            print(f"   Features: {len(self.feature_names) if self.feature_names else 'Unknown'}")
            print(f"   Timestamp: {model_data.get('timestamp', 'Unknown')}")

            if self.training_metrics:
                print(f"   Training Accuracy: {self.training_metrics.get('accuracy', 'N/A'):.4f}")

            return True

        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            return False

    def predict_single(self, input_data, return_probability=True):
        """
        Make prediction for a single loan application.
        """
        if self.model is None:
            raise ValueError("Model must be trained or loaded first")

        # Convert input to DataFrame if it's a dict
        if isinstance(input_data, dict):
            input_df = pd.DataFrame([input_data])
        else:
            input_df = pd.DataFrame([input_data], columns=self.feature_names)

        # Make prediction
        prediction = self.model.predict(input_df)[0]

        result = {
            'prediction': int(prediction),
            'prediction_label': 'Approved' if prediction == 1 else 'Rejected'
        }

        if return_probability and hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(input_df)[0]
            result.update({
                'probability': float(probabilities[1]),
                'confidence': float(max(probabilities)),
                'probabilities': {
                    'rejected': float(probabilities[0]),
                    'approved': float(probabilities[1])
                }
            })

        return result

    def get_feature_importance_dict(self):
        """Return feature importance as dictionary for API responses."""
        if self.feature_importance is None:
            return {}

        return dict(zip(
            self.feature_importance['feature'].tolist(),
            self.feature_importance['importance'].tolist()
        ))

    def get_model_info(self):
        """Return comprehensive model information for API."""
        return {
            'model_type': self.model_type,
            'training_metrics': self.training_metrics,
            'feature_count': len(self.feature_names) if self.feature_names else 0,
            'feature_names': self.feature_names,
            'best_params': self.best_params,
            'feature_importance': self.get_feature_importance_dict()
        }

# Usage example and testing
if __name__ == "__main__":
    from data_preprocessing import LoanDataProcessor

    # Load and preprocess data
    processor = LoanDataProcessor()
    X_train, X_test, y_train, y_test, encoders = processor.run_complete_pipeline(
        "../data/loan_dataset.csv"
    )

    # Initialize and train model
    model = LoanPredictionModel(model_type='decision_tree')

    # Hyperparameter tuning
    model.hyperparameter_tuning(X_train, y_train, cv_folds=5)

    # Train model
    model.train_model(X_train, y_train, processor.feature_columns, encoders)

    # Cross-validation
    model.cross_validate(X_train, y_train)

    # Evaluation
    metrics, y_pred, y_pred_proba = model.evaluate_model(X_test, y_test)

    # Create visualizations
    model.create_evaluation_visualizations(X_test, y_test)

    # Save model
    model.save_model("trained_model.pkl")

    print("\n🎉 Model training and evaluation completed successfully!")