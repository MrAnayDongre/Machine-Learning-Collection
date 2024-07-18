import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from sklearn.metrics import mean_squared_error, r2_score
import warnings

warnings.filterwarnings("ignore")

# Load data
df_train = pd.read_csv("/kaggle/input/testing-kagglex-dataset/data/train.csv")
df_test = pd.read_csv("/kaggle/input/testing-kagglex-dataset/data/test.csv")

# Feature engineering
def engineer_features(df):
    df['age'] = 2024 - df['model_year']
    df['brand_model'] = df['brand'] + '_' + df['model']
    df['engine_power'] = df['engine'].str.extract('(\d+\.?\d*)HP').astype(float)
    df['engine_size'] = df['engine'].str.extract('(\d+\.?\d*)L').astype(float)
    df['is_luxury'] = df['brand'].isin(['BMW', 'Mercedes-Benz', 'Audi', 'Lexus', 'Porsche']).astype(int)
    df['accident_binary'] = (df['accident'] != 'None reported').astype(int)
    return df

df_train = engineer_features(df_train)
df_test = engineer_features(df_test)

# Log transform the target variable
df_train['log_price'] = np.log1p(df_train['price'])

# Identify numeric and categorical columns
numeric_features = ['model_year', 'milage', 'age', 'engine_power', 'engine_size']
categorical_features = ['brand', 'fuel_type', 'transmission', 'ext_col', 'int_col', 'accident', 'clean_title', 'brand_model', 'is_luxury']

# Create preprocessing pipelines
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Prepare the data
X = df_train.drop(['id', 'price', 'log_price', 'engine'], axis=1)
y = df_train['log_price']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the preprocessor and transform the data
preprocessor.fit(X_train)
X_train_processed = preprocessor.transform(X_train)
X_val_processed = preprocessor.transform(X_val)
X_test_processed = preprocessor.transform(df_test.drop(['id', 'engine'], axis=1))

# Define the model
def create_model(input_dim):
    model = keras.Sequential([
        keras.layers.Dense(512, activation='relu', kernel_regularizer=l2(0.001), input_shape=(input_dim,)),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.4),
        keras.layers.Dense(256, activation='relu', kernel_regularizer=l2(0.001)),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.4),
        keras.layers.Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.4),
        keras.layers.Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.4),
        keras.layers.Dense(1)
    ])
    return model

# Create and compile the model
model = create_model(X_train_processed.shape[1])
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mean_absolute_error'])

# Define callbacks
early_stopping = EarlyStopping(patience=20, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(factor=0.5, patience=10, min_lr=1e-6)

# Train the model
history = model.fit(
    X_train_processed, y_train,
    validation_data=(X_val_processed, y_val),
    epochs=100,
    batch_size=64,
    callbacks=[early_stopping, lr_scheduler],
    verbose=1
)

# Evaluate the model
val_mse, val_mae = model.evaluate(X_val_processed, y_val)
val_rmse = np.sqrt(val_mse)
print(f"Validation RMSE (log scale): {val_rmse}")

# Make predictions on test data
y_pred_log = model.predict(X_test_processed)
y_pred = np.expm1(y_pred_log)  # Convert back to original scale

# Create submission file
submission = pd.DataFrame({'id': df_test['id'], 'price': y_pred.flatten()})
submission.to_csv('submission.csv', index=False)
print('Submission file created successfully')