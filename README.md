## Insurance Fraud Detection
This project uses machine learning to predict potentially fraudulent insurance claims by analyzing patterns in customer data, policy information, and claim details.
Overview
Insurance fraud is a significant challenge for insurance companies. This project demonstrates how various machine learning algorithms can be leveraged to identify potentially fraudulent claims, which can help insurance companies save costs and improve operational efficiency.
Dataset
The project uses the insurance_claims.csv dataset that contains 1000 records with 40 columns including:

Customer information (age, gender, education level, occupation)
Policy details (policy number, bind date, annual premium, deductible)
Incident information (date, type, severity, location)
Claim details (injury claim, property claim, vehicle claim)
The target variable: fraud_reported (Y/N)

## Methodology
The project follows a standard machine learning pipeline:

Data Exploration and Preprocessing

Handling missing values for columns like collision_type, property_damage, and police_report_available
Visualizing data distributions and correlations
Dropping unnecessary columns like policy_number, incident_location, etc.
Removing multicollinear features (age and total_claim_amount)


## Feature Engineering

Converting categorical features to numerical using one-hot encoding
Scaling numerical features using StandardScaler


## Model Training and Evaluation

Splitting data into training (75%) and testing (25%) sets
Training multiple classification algorithms:

Support Vector Classifier (SVC)
K-Nearest Neighbors (KNN)
Decision Tree (with hyperparameter tuning)
Random Forest
AdaBoost
Gradient Boosting
Stochastic Gradient Boosting (SGB)
XGBoost
CatBoost
LightGBM
Extra Trees
Voting Classifier (ensemble of all models)




## Model Comparison

Evaluating models using accuracy, precision, recall, and F1-score
Visualizing model performance for comparison



## Results
The best performing models were:

Voting Classifier: 79.2% accuracy
Extra Trees Classifier: 78.4% accuracy
Random Forest: 74.0% accuracy

The Voting Classifier showed the best overall performance with a balance of precision (0.82 for non-fraud, 0.65 for fraud) and recall (0.91 for non-fraud, 0.45 for fraud).
## Requirements

Python 3.x
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
lightgbm
catboost
plotly

## Usage

Clone this repository
Install required packages: pip install -r requirements.txt
Run the Jupyter notebook: jupyter notebook InsuranceFraudDetection.ipynb
Follow the step-by-step analysis and model training process

## Future Improvements

Feature importance analysis to identify the most influential predictors
More extensive hyperparameter tuning for top-performing models
Address class imbalance with techniques like SMOTE or class weighting
Implement model explainability using SHAP or LIME
Deploy the model as a web service for real-time prediction

