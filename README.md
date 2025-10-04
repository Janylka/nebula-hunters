# 🌌 Nebula Hunters

**Project:** Using AI to Discover New Worlds Beyond Our Solar System  
**Hackathon:** NASA Space Apps Challenge 2025  

---

## 🚀 Project Idea
Our project explores the use of **AI and machine learning** to classify potential exoplanets 
based on the Kepler mission dataset.  
We aim to build a prototype that learns from NASA’s open data and makes predictions on whether 
a celestial body is a confirmed planet or not.

---

## 📂 Repository Structure
```
nebula-hunters/
│── data/               # Dataset (Kepler CSV)
│── src/                # Source code for model & preprocessing
│── notebooks/          # Jupyter notebooks for exploration
│── models/             # Trained ML and LSTM models for prediction
│── requirements.txt    # Dependencies
│── README.md           # Project documentation
```

---

## ⚙️ Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/nebula-hunters.git
   cd nebula-hunters
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run:
   ```bash
   streamlit run app.py
   ```

---

## 🧠 Example Code (Prototype)
```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# Load dataset
df = pd.read_csv("data/kepler_exoplanet_data.csv")

# Features and target
X = df.drop(columns=["koi_disposition"])
y = df["koi_disposition"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Example prediction
print("Prediction:", model.predict(X_test[:1]))
```

---

## 📊 Data Source
We use the **Kepler Exoplanet Candidate Catalog** from NASA Exoplanet Archive.  
🔗 [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/)

---

## 👨‍🚀 Team
**Nebula Hunters** – Passionate explorers using AI to unlock the secrets of the cosmos.

---

✨ “Somewhere, something incredible is waiting to be known.” – Carl Sagan