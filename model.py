import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os

MODEL_FILE = 'attrition_model.pkl'
ENCODER_FILE = 'label_encoders.pkl'
SCALER_FILE = 'scaler.pkl'
FEATURE_ORDER_FILE = 'feature_order.pkl'


def prepare_data():
    """Load and prepare data from Dataset.csv"""
    try:
        # Load the dataset from CSV file
        df = pd.read_csv('Dataset.csv')

        print(f"Dataset loaded successfully with {len(df)} rows and {len(df.columns)} columns")
        print(f"Columns: {list(df.columns)}")

        # Check if the required columns exist
        required_columns = [
            'Age', 'Attrition', 'BusinessTravel', 'DailyRate', 'Department',
            'DistanceFromHome', 'Education', 'EducationField', 'EnvironmentSatisfaction',
            'Gender', 'HourlyRate', 'JobInvolvement', 'JobLevel', 'JobRole',
            'JobSatisfaction', 'MaritalStatus', 'MonthlyIncome', 'MonthlyRate',
            'NumCompaniesWorked', 'OverTime', 'PercentSalaryHike', 'PerformanceRating',
            'RelationshipSatisfaction', 'StockOptionLevel', 'TotalWorkingYears',
            'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany',
            'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager'
        ]

        # Check for missing columns
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Warning: Missing columns: {missing_columns}")

        # Remove columns that are constant or not useful for prediction
        columns_to_drop = ['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours']
        df = df.drop(columns=[col for col in columns_to_drop if col in df.columns], errors='ignore')

        # Display basic information about the dataset
        print(f"\nDataset shape: {df.shape}")
        print(f"Attrition distribution:\n{df['Attrition'].value_counts()}")

        return df

    except FileNotFoundError:
        print("Error: Dataset.csv file not found!")
        print("Please make sure Dataset.csv is in the same directory as model.py")
        raise
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        raise


def train_model():
    """Train the machine learning model"""
    # Load and prepare data
    df = prepare_data()

    # Separate features and target
    X = df.drop('Attrition', axis=1)
    y = df['Attrition']

    # Display data types and missing values
    print("\nData types:")
    print(X.dtypes)
    print(f"\nMissing values:\n{X.isnull().sum()}")

    # Encode categorical variables
    categorical_columns = X.select_dtypes(include=['object']).columns
    label_encoders = {}

    print(f"\nCategorical columns to encode: {list(categorical_columns)}")

    for col in categorical_columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le
        print(f"Encoded {col}: {len(le.classes_)} categories")

    # Scale numerical features
    scaler = StandardScaler()
    numerical_columns = X.select_dtypes(include=[np.number]).columns
    X[numerical_columns] = scaler.fit_transform(X[numerical_columns])

    # Save the feature order
    feature_order = X.columns.tolist()

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\nTraining set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Features: {X_train.shape[1]}")
    print(f"Feature order: {feature_order}")

    # Train model
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2
    )
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred) * 100

    # Additional metrics
    print("\n" + "=" * 50)
    print("MODEL EVALUATION RESULTS")
    print("=" * 50)
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    print(f"\nTop 10 Most Important Features:")
    print(feature_importance.head(10))

    # Save model and encoders
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump(model, f)

    with open(ENCODER_FILE, 'wb') as f:
        pickle.dump(label_encoders, f)

    with open(SCALER_FILE, 'wb') as f:
        pickle.dump(scaler, f)

    # Save feature order
    with open(FEATURE_ORDER_FILE, 'wb') as f:
        pickle.dump(feature_order, f)

    print(f"\nModel and encoders saved successfully!")

    model_info = {
        'algorithm': 'Random Forest',
        'features_used': len(X.columns),
        'training_samples': len(X_train),
        'test_samples': len(X_test),
        'accuracy': accuracy,
        'top_features': feature_importance.head(10).to_dict('records')
    }

    return accuracy, model_info


def predict_attrition(employee_data):
    """Predict attrition for a single employee"""
    if not os.path.exists(MODEL_FILE):
        raise Exception("Model not trained yet. Please train the model first.")

    # Load model, encoders, and scaler
    with open(MODEL_FILE, 'rb') as f:
        model = pickle.load(f)

    with open(ENCODER_FILE, 'rb') as f:
        label_encoders = pickle.load(f)

    with open(SCALER_FILE, 'rb') as f:
        scaler = pickle.load(f)

    # Load feature order
    if os.path.exists(FEATURE_ORDER_FILE):
        with open(FEATURE_ORDER_FILE, 'rb') as f:
            feature_order = pickle.load(f)
    else:
        feature_order = None

    # Convert to DataFrame
    df = pd.DataFrame([employee_data])

    # Encode categorical variables
    for col, encoder in label_encoders.items():
        if col in df.columns:
            # Handle unseen labels by using the first class
            original_value = str(df[col].iloc[0])
            if original_value not in encoder.classes_:
                print(f"Warning: Unseen label '{original_value}' for column '{col}'. Using default encoding.")
                df[col] = encoder.classes_[0]
            else:
                df[col] = original_value
            df[col] = encoder.transform(df[col])

    # Ensure all required columns are present
    required_cols = list(label_encoders.keys())
    missing_cols = set(required_cols) - set(df.columns)
    if missing_cols:
        # Add missing columns with default value 0
        for col in missing_cols:
            df[col] = 0

    # Reorder columns to match training data
    if feature_order is not None:
        # Ensure we have all expected features
        for feature in feature_order:
            if feature not in df.columns:
                df[feature] = 0  # Add missing features with default value

        # Reorder columns to match training order
        df = df[feature_order]
    else:
        # Use the order from label_encoders
        df = df[required_cols]

    # Scale numerical features
    numerical_columns = df.select_dtypes(include=[np.number]).columns
    df[numerical_columns] = scaler.transform(df[numerical_columns])

    # Make prediction
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]  # Probability of "Yes"

    return prediction, probability


def get_feature_options():
    """Get possible values for categorical features from the dataset"""
    try:
        df = pd.read_csv('Dataset.csv')

        feature_options = {}
        categorical_columns = ['BusinessTravel', 'Department', 'EducationField',
                               'Gender', 'JobRole', 'MaritalStatus', 'OverTime']

        for col in categorical_columns:
            if col in df.columns:
                feature_options[col] = sorted(df[col].astype(str).unique().tolist())

        # Numerical ranges
        numerical_ranges = {}
        numerical_columns = ['Age', 'DailyRate', 'DistanceFromHome', 'Education',
                             'EnvironmentSatisfaction', 'HourlyRate', 'JobInvolvement',
                             'JobLevel', 'JobSatisfaction', 'MonthlyIncome', 'MonthlyRate',
                             'NumCompaniesWorked', 'PercentSalaryHike', 'PerformanceRating',
                             'RelationshipSatisfaction', 'StockOptionLevel', 'TotalWorkingYears',
                             'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany',
                             'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager']

        for col in numerical_columns:
            if col in df.columns:
                numerical_ranges[col] = {
                    'min': int(df[col].min()),
                    'max': int(df[col].max()),
                    'mean': int(df[col].mean())
                }

        return feature_options, numerical_ranges

    except Exception as e:
        print(f"Error getting feature options: {str(e)}")
        # Return default values if dataset loading fails
        default_feature_options = {
            'BusinessTravel': ['Non-Travel', 'Travel_Rarely', 'Travel_Frequently'],
            'Department': ['Sales', 'Research & Development', 'Human Resources'],
            'EducationField': ['Life Sciences', 'Medical', 'Marketing', 'Technical Degree', 'Other', 'Human Resources'],
            'Gender': ['Male', 'Female'],
            'JobRole': ['Sales Executive', 'Research Scientist', 'Laboratory Technician', 'Manufacturing Director',
                        'Healthcare Representative', 'Manager', 'Sales Representative', 'Research Director',
                        'Human Resources'],
            'MaritalStatus': ['Single', 'Married', 'Divorced'],
            'OverTime': ['Yes', 'No']
        }

        default_numerical_ranges = {
            'Age': {'min': 18, 'max': 60, 'mean': 36},
            'DailyRate': {'min': 100, 'max': 1500, 'mean': 800},
            'DistanceFromHome': {'min': 1, 'max': 30, 'mean': 9},
            'Education': {'min': 1, 'max': 5, 'mean': 3},
            'EnvironmentSatisfaction': {'min': 1, 'max': 4, 'mean': 2},
            'HourlyRate': {'min': 30, 'max': 100, 'mean': 65},
            'JobInvolvement': {'min': 1, 'max': 4, 'mean': 2},
            'JobLevel': {'min': 1, 'max': 5, 'mean': 2},
            'JobSatisfaction': {'min': 1, 'max': 4, 'mean': 2},
            'MonthlyIncome': {'min': 1000, 'max': 20000, 'mean': 6500},
            'MonthlyRate': {'min': 2000, 'max': 27000, 'mean': 14000},
            'NumCompaniesWorked': {'min': 0, 'max': 10, 'mean': 2},
            'PercentSalaryHike': {'min': 0, 'max': 25, 'mean': 15},
            'PerformanceRating': {'min': 1, 'max': 4, 'mean': 3},
            'RelationshipSatisfaction': {'min': 1, 'max': 4, 'mean': 2},
            'StockOptionLevel': {'min': 0, 'max': 3, 'mean': 1},
            'TotalWorkingYears': {'min': 0, 'max': 40, 'mean': 11},
            'TrainingTimesLastYear': {'min': 0, 'max': 6, 'mean': 2},
            'WorkLifeBalance': {'min': 1, 'max': 4, 'mean': 2},
            'YearsAtCompany': {'min': 0, 'max': 40, 'mean': 7},
            'YearsInCurrentRole': {'min': 0, 'max': 18, 'mean': 4},
            'YearsSinceLastPromotion': {'min': 0, 'max': 15, 'mean': 2},
            'YearsWithCurrManager': {'min': 0, 'max': 17, 'mean': 4}
        }

        return default_feature_options, default_numerical_ranges