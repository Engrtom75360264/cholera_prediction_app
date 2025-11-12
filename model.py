# Model.py
import pandas as pd
import numpy as np
import pathlib
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score

# ------------------ Load dataset ------------------
df = pd.read_csv("C:\\Users\\HP\\ENGR.TOM/NGA-Table 1.csv")


# ------------------ Preprocessing ------------------
df = df.drop(['Entry ID', 'Local Government Area'], axis=1, errors='ignore')

# Convert numeric columns stored as object
numeric_cols = ['Cases', 'Deaths', 'CFR (%)', 'Population', 'Cases/100,000', 'Deaths/100,000']
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce')

# Fill missing values for numeric columns
for col in numeric_cols:
    if col in df.columns:
        df[col].fillna(df[col].median(), inplace=True)

# Fill missing values for categorical columns
if 'State' in df.columns:
    df['State'].fillna('Unknown', inplace=True)

# Keep only the features we want (ignore month_sin, month_cos)
feature_cols = ['Year', 'State','CFR (%)', 'Population']
target_col = 'Cases/100,000'

X = df[feature_cols]
y = df[target_col]

# ------------------ Preprocessing Pipeline ------------------
numeric_features = ['Year','CFR (%)', 'Population']
categorical_features = ['State']

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

# ------------------ Full Pipeline ------------------
pipe = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('gb', GradientBoostingRegressor(random_state=42))
])

# ------------------ Train/Test Split ------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ------------------ Grid Search ------------------
param_grid = {
    'gb__n_estimators': [100, 300, 500],
    'gb__learning_rate': [0.01, 0.05, 0.1],
    'gb__max_depth': [3, 4, 5],
    'gb__subsample': [0.8, 1.0]
}

grid_search = GridSearchCV(pipe, param_grid, cv=3, scoring='r2', n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

# ------------------ Evaluate ------------------
y_pred = best_model.predict(X_test)
print("✅ Best Parameters:", grid_search.best_params_)
print("✅ Test MSE:", mean_squared_error(y_test, y_pred))
print("✅ Test R²:", r2_score(y_test, y_pred))

# ------------------ Save Model ------------------
models_dir = pathlib.Path("models")
models_dir.mkdir(exist_ok=True)
joblib.dump(best_model, models_dir / "best_model.joblib")
print("💾 Model saved to models/best_model.joblib")
