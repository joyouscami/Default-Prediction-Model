from google.cloud import bigquery

# Use raw string for the file path
key_path = r"C:\Advanced Analytics\JSON KEY\master-charmer-472608-g7-5a81523269bb.json"

# Initialize the BigQuery client
client = bigquery.Client.from_service_account_json(key_path)

# Query table
query = """
    SELECT *
    FROM `bigquery-public-data.ml_datasets.credit_card_default`
    LIMIT 1000
"""

# Run the query and create a DataFrame
df = client.query(query).to_dataframe()

# Display the first few rows
print(df.head())

#Classification model using XG boost
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import xgboost as xgb

# Rename target if needed
df.rename(columns={"default_payment_next_month": "default"}, inplace=True)

# Drop ID column if present
if 'ID' in df.columns:
    df.drop(columns=["ID"], inplace=True)

# Handle missing values (if any)
df.dropna(inplace=True)
df.info()

# Features and target
X = df.drop("default", axis=1)
y = df["default"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train the XGBoost classifier
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train_scaled, y_train)
#sklearn.pipeline
# accuracy, precision, recall
# Make predictions
y_pred = model.predict(X_test_scaled)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


