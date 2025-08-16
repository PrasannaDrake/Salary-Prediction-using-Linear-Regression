#If salary_prediction_model.pkl is already there, then there is no need to train the model. Directly run app.py

import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

MODEL_PATH = "salary_prediction_model.pkl"

# 1. Load data
df = pd.read_csv("Salary Data.csv")
df = df.dropna()

X = df[['Age', 'Gender', 'Years of Experience', 'Education Level', 'Job Title']]
y = df['Salary']

# 2. Preprocessing
categorical_features = ['Gender', 'Education Level', 'Job Title']
numeric_features = ['Age', 'Years of Experience']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

# 3. Build pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# 4. Train model
model.fit(X, y)

# 5. Save model only if not already saved
if not os.path.exists(MODEL_PATH):
    joblib.dump(model, MODEL_PATH)
    print(f"✅ Model trained and saved as {MODEL_PATH}")
else:

    print(f"ℹ️ Model already exists at {MODEL_PATH}, not overwritten.")
