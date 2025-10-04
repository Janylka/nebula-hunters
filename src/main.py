import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Load Kepler dataset
df = pd.read_csv("data/kepler_exoplanet_data.csv")

# Drop rows with missing values (basic cleaning)
df = df.dropna()

# Features and target
X = df.drop(columns=["koi_disposition"])
y = df["koi_disposition"]

# Encode target if needed (Confirmed / False Positive / Candidate)
y = y.astype("category").cat.codes

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Example prediction
print("🔮 Example prediction for first row:", model.predict([X_test.iloc[0]]))
