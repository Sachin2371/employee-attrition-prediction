from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
import pandas as pd
import numpy as np
import json
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from werkzeug.security import generate_password_hash, check_password_hash
import warnings

warnings.filterwarnings('ignore')

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

# File paths
USERS_FILE = 'users.json'
MODEL_FILE = 'attrition_model.pkl'

# Initialize data storage
if not os.path.exists(USERS_FILE):
    with open(USERS_FILE, 'w') as f:
        json.dump({}, f)


def create_training_data():
    """Load training data from Dataset.csv file"""
    try:
        if os.path.exists('Dataset.csv'):
            df = pd.read_csv('Dataset.csv')
            print(f"Loaded dataset from Dataset.csv with {len(df)} rows and {len(df.columns)} columns")
            print(f"Columns found: {df.columns.tolist()}")

            # Check if the required columns exist
            required_columns = ['Age', 'Attrition', 'BusinessTravel', 'DailyRate', 'Department',
                                'DistanceFromHome', 'Education', 'EducationField', 'EnvironmentSatisfaction',
                                'Gender', 'HourlyRate', 'JobInvolvement', 'JobLevel', 'JobRole',
                                'JobSatisfaction', 'MaritalStatus', 'MonthlyIncome', 'MonthlyRate',
                                'NumCompaniesWorked', 'OverTime', 'PercentSalaryHike', 'PerformanceRating',
                                'RelationshipSatisfaction', 'StockOptionLevel', 'TotalWorkingYears',
                                'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany',
                                'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager']

            # Check for missing columns
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                print(f"Warning: Missing columns in Dataset.csv: {missing_columns}")
                # Remove missing columns from required list
                required_columns = [col for col in required_columns if col not in missing_columns]

            # Select only the required columns that exist in the dataset
            df = df[required_columns]

            return df
        else:
            print("Dataset.csv not found. Please make sure the file exists in the project directory.")
            # Fallback to sample data if dataset doesn't exist
            return create_sample_data()

    except Exception as e:
        print(f"Error loading Dataset.csv: {str(e)}")
        # Fallback to sample data if there's an error
        return create_sample_data()


def create_sample_data():
    """Fallback sample data if Dataset.csv is not available"""
    print("Using sample data as fallback...")
    data = {
        'Age': [41, 49, 37, 33, 27, 32, 59, 30, 38, 36],
        'BusinessTravel': ['Travel_Rarely', 'Travel_Frequently', 'Travel_Rarely', 'Non-Travel', 'Travel_Rarely',
                           'Travel_Frequently', 'Travel_Rarely', 'Non-Travel', 'Travel_Rarely', 'Travel_Frequently'],
        'DailyRate': [1102, 279, 1373, 1392, 591, 1005, 1324, 1358, 216, 1299],
        'Department': ['Sales', 'Research & Development', 'Research & Development', 'Research & Development',
                       'Research & Development', 'Sales', 'Research & Development', 'Research & Development',
                       'Research & Development', 'Research & Development'],
        'DistanceFromHome': [1, 8, 2, 3, 2, 2, 3, 24, 23, 27],
        'Education': [2, 1, 2, 4, 1, 2, 3, 1, 3, 3],
        'EducationField': ['Life Sciences', 'Life Sciences', 'Other', 'Life Sciences', 'Medical',
                           'Life Sciences', 'Medical', 'Life Sciences', 'Life Sciences', 'Medical'],
        'EnvironmentSatisfaction': [2, 3, 4, 4, 1, 4, 3, 1, 2, 3],
        'Gender': ['Female', 'Male', 'Male', 'Female', 'Male', 'Male', 'Female', 'Male', 'Male', 'Female'],
        'HourlyRate': [94, 61, 92, 56, 40, 79, 81, 67, 44, 94],
        'JobInvolvement': [3, 2, 2, 3, 3, 3, 4, 3, 2, 3],
        'JobLevel': [2, 2, 1, 1, 1, 1, 1, 1, 3, 2],
        'JobRole': ['Sales Executive', 'Research Scientist', 'Laboratory Technician', 'Research Scientist',
                    'Laboratory Technician', 'Sales Representative', 'Laboratory Technician',
                    'Laboratory Technician', 'Manufacturing Director', 'Healthcare Representative'],
        'JobSatisfaction': [4, 2, 3, 3, 2, 4, 1, 3, 3, 3],
        'MaritalStatus': ['Single', 'Married', 'Single', 'Married', 'Married', 'Single', 'Married',
                          'Divorced', 'Single', 'Married'],
        'MonthlyIncome': [5993, 5130, 2090, 2909, 3468, 3068, 2670, 2693, 9526, 5237],
        'MonthlyRate': [19479, 24907, 2396, 23159, 16632, 11864, 9964, 13335, 8787, 16577],
        'NumCompaniesWorked': [8, 1, 6, 1, 9, 0, 4, 1, 0, 6],
        'OverTime': ['Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'No', 'No', 'No', 'Yes'],
        'PercentSalaryHike': [11, 23, 15, 11, 12, 13, 20, 22, 21, 13],
        'PerformanceRating': [3, 4, 3, 3, 3, 3, 4, 4, 4, 3],
        'RelationshipSatisfaction': [1, 4, 2, 3, 4, 3, 1, 2, 2, 2],
        'StockOptionLevel': [0, 1, 0, 0, 1, 0, 3, 1, 0, 2],
        'TotalWorkingYears': [8, 10, 7, 8, 6, 8, 12, 1, 10, 17],
        'TrainingTimesLastYear': [0, 3, 3, 3, 3, 2, 3, 2, 2, 3],
        'WorkLifeBalance': [1, 3, 3, 3, 3, 2, 2, 3, 3, 2],
        'YearsAtCompany': [6, 10, 0, 8, 2, 7, 1, 1, 9, 7],
        'YearsInCurrentRole': [4, 7, 0, 7, 2, 7, 0, 0, 5, 7],
        'YearsSinceLastPromotion': [0, 1, 0, 3, 2, 3, 0, 0, 1, 7],
        'YearsWithCurrManager': [5, 7, 0, 0, 2, 6, 0, 0, 8, 7],
        'Attrition': ['Yes', 'No', 'Yes', 'No', 'No', 'No', 'No', 'Yes', 'No', 'No']
    }
    return pd.DataFrame(data)


def train_and_save_model():
    """Train the model and save it using data from Dataset.csv"""
    try:
        print("Starting model training...")

        # Load data from Dataset.csv
        df = create_training_data()
        print(f"Training data shape: {df.shape}")
        print(f"Attrition distribution:\n{df['Attrition'].value_counts()}")

        # Define feature columns
        categorical_cols = ['BusinessTravel', 'Department', 'EducationField', 'Gender',
                            'JobRole', 'MaritalStatus', 'OverTime']

        numerical_cols = ['Age', 'DailyRate', 'DistanceFromHome', 'Education', 'EnvironmentSatisfaction',
                          'HourlyRate', 'JobInvolvement', 'JobLevel', 'JobSatisfaction', 'MonthlyIncome',
                          'MonthlyRate', 'NumCompaniesWorked', 'PercentSalaryHike', 'PerformanceRating',
                          'RelationshipSatisfaction', 'StockOptionLevel', 'TotalWorkingYears',
                          'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany',
                          'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager']

        # Remove any columns that don't exist in the dataset
        categorical_cols = [col for col in categorical_cols if col in df.columns]
        numerical_cols = [col for col in numerical_cols if col in df.columns]

        print(f"Using {len(categorical_cols)} categorical features: {categorical_cols}")
        print(f"Using {len(numerical_cols)} numerical features: {numerical_cols}")

        # Encode categorical variables
        label_encoders = {}
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
            print(f"Encoded {col}: {len(le.classes_)} classes")

        # Encode target variable
        target_encoder = LabelEncoder()
        df['Attrition'] = target_encoder.fit_transform(df['Attrition'])
        print(f"Encoded Attrition: {target_encoder.classes_}")

        # Prepare features and target
        X = df[numerical_cols + categorical_cols]
        y = df['Attrition']

        print(f"Features shape: {X.shape}")
        print(f"Target shape: {y.shape}")

        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
        model.fit(X, y)

        # Test the model
        train_score = model.score(X, y)
        print(f"Training accuracy: {train_score:.3f}")

        # Save model and encoders
        model_data = {
            'model': model,
            'label_encoders': label_encoders,
            'target_encoder': target_encoder,
            'feature_columns': X.columns.tolist(),
            'categorical_cols': categorical_cols,
            'numerical_cols': numerical_cols
        }

        with open(MODEL_FILE, 'wb') as f:
            pickle.dump(model_data, f)

        print("Model trained and saved successfully!")
        return True

    except Exception as e:
        print(f"Error in model training: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def load_model():
    """Load the trained model"""
    try:
        if os.path.exists(MODEL_FILE):
            with open(MODEL_FILE, 'rb') as f:
                model_data = pickle.load(f)
            print("Model loaded successfully!")
            return model_data
        else:
            print("Model file not found!")
            return None
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None


# Routes
@app.route('/')
def index():
    return redirect(url_for('login'))


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        with open(USERS_FILE, 'r') as f:
            users = json.load(f)

        if username in users and check_password_hash(users[username]['password'], password):
            session['user'] = username
            flash('Login successful!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password', 'error')

    return render_template('login.html')


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        if password != confirm_password:
            flash('Passwords do not match!', 'error')
            return render_template('signup.html')

        with open(USERS_FILE, 'r') as f:
            users = json.load(f)

        if username in users:
            flash('Username already exists!', 'error')
            return render_template('signup.html')

        users[username] = {
            'email': email,
            'password': generate_password_hash(password),
            'created_at': pd.Timestamp.now().isoformat()
        }

        with open(USERS_FILE, 'w') as f:
            json.dump(users, f, indent=4)

        flash('Registration successful! Please login.', 'success')
        return redirect(url_for('login'))

    return render_template('signup.html')


@app.route('/dashboard')
def dashboard():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('dashboard.html', username=session['user'])


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if 'user' not in session:
        return redirect(url_for('login'))

    prediction = None
    probability = None

    if request.method == 'POST':
        if 'csv_file' in request.files and request.files['csv_file'].filename != '':
            # CSV file prediction
            # CSV file prediction
            file = request.files['csv_file']
            if file and file.filename.endswith('.csv'):
                try:
                    # Read CSV file
                    df = pd.read_csv(file)
                    print(f"CSV loaded with {len(df)} rows and {len(df.columns)} columns")
                    print(f"Columns: {df.columns.tolist()}")

                    model_data = load_model()

                    if not model_data:
                        flash('Model not available. Please try again later.', 'error')
                        return render_template('prediction.html', username=session['user'])

                    # Prepare the data for prediction - use the same columns as training
                    categorical_cols = model_data['categorical_cols']
                    numerical_cols = model_data['numerical_cols']
                    feature_columns = model_data['feature_columns']

                    print(f"Model expects features: {feature_columns}")

                    # Create a copy for processing
                    prediction_df = df.copy()

                    # Encode categorical variables
                    for col in categorical_cols:
                        if col in prediction_df.columns and col in model_data['label_encoders']:
                            le = model_data['label_encoders'][col]
                            # Handle unknown categories
                            prediction_df[col] = prediction_df[col].astype(str)
                            prediction_df[col] = prediction_df[col].apply(
                                lambda x: x if x in le.classes_ else le.classes_[0]
                            )
                            prediction_df[col] = le.transform(prediction_df[col])
                        elif col not in prediction_df.columns:
                            # Add missing categorical column with default value
                            prediction_df[col] = 0

                    # Ensure all required numerical columns are present
                    for col in numerical_cols:
                        if col not in prediction_df.columns:
                            prediction_df[col] = 0

                    # Select only the required columns in correct order
                    prediction_df = prediction_df[feature_columns]

                    # Make predictions
                    predictions = model_data['model'].predict(prediction_df)
                    probabilities = model_data['model'].predict_proba(prediction_df)

                    # Prepare results
                    results = []
                    for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
                        attrition_label = model_data['target_encoder'].inverse_transform([pred])[0]
                        probability_value = prob[1] if pred == 1 else prob[0]
                        results.append({
                            'employee': i + 1,
                            'attrition': attrition_label,
                            'probability': f"{probability_value:.2%}"
                        })

                    return render_template('prediction_results.html',
                                           results=results,
                                           bulk_prediction=True,
                                           username=session['user'])

                except Exception as e:
                    error_msg = f'Error processing CSV file: {str(e)}'
                    print(error_msg)
                    import traceback
                    traceback.print_exc()
                    flash(error_msg, 'error')

        else:
            # Manual input prediction
            try:
                # Collect form data
                input_data = {}
                for field in ['Age', 'DailyRate', 'DistanceFromHome', 'Education', 'EnvironmentSatisfaction',
                              'HourlyRate', 'JobInvolvement', 'JobLevel', 'JobSatisfaction', 'MonthlyIncome',
                              'MonthlyRate', 'NumCompaniesWorked', 'PercentSalaryHike', 'PerformanceRating',
                              'RelationshipSatisfaction', 'StockOptionLevel', 'TotalWorkingYears',
                              'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany',
                              'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager']:
                    input_data[field] = int(request.form[field])

                for field in ['BusinessTravel', 'Department', 'EducationField', 'Gender',
                              'JobRole', 'MaritalStatus', 'OverTime']:
                    input_data[field] = request.form[field]

                model_data = load_model()

                if not model_data:
                    flash('Model not available. Please try again later.', 'error')
                    return render_template('prediction.html', username=session['user'])

                # Prepare input data
                categorical_cols = model_data['categorical_cols']
                feature_columns = model_data['feature_columns']

                # Create input DataFrame
                input_df = pd.DataFrame([input_data])

                # Encode categorical variables
                for col in categorical_cols:
                    if col in input_df.columns and col in model_data['label_encoders']:
                        le = model_data['label_encoders'][col]
                        input_value = input_data[col]
                        if input_value in le.classes_:
                            input_df[col] = le.transform([input_value])[0]
                        else:
                            input_df[col] = le.transform([le.classes_[0]])[0]

                # Ensure correct column order
                input_df = input_df[feature_columns]

                # Make prediction
                prediction_encoded = model_data['model'].predict(input_df)[0]
                probability_array = model_data['model'].predict_proba(input_df)[0]

                prediction = model_data['target_encoder'].inverse_transform([prediction_encoded])[0]
                probability_value = probability_array[1] if prediction_encoded == 1 else probability_array[0]

                return render_template('prediction.html',
                                       prediction=prediction,
                                       probability=f"{probability_value:.2%}",
                                       username=session['user'])

            except Exception as e:
                error_msg = f'Error making prediction: {str(e)}'
                print(error_msg)
                import traceback
                traceback.print_exc()
                flash(error_msg, 'error')

    return render_template('prediction.html', prediction=prediction, probability=probability, username=session['user'])


@app.route('/logout')
def logout():
    session.pop('user', None)
    flash('You have been logged out successfully!', 'success')
    return redirect(url_for('login'))


@app.route('/train_model')
def train_model_route():
    """Route to manually train the model"""
    if train_and_save_model():
        flash('Model trained successfully!', 'success')
    else:
        flash('Model training failed!', 'error')
    return redirect(url_for('dashboard'))


if __name__ == '__main__':
    # Train model on startup if not exists
    if not os.path.exists(MODEL_FILE):
        print("Model not found. Training new model...")
        train_and_save_model()
    else:
        print("Model found. Loading existing model...")
        load_model()

    app.run(debug=True, port=5000)