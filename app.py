from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

app = Flask(__name__)

# Load dataset
file_path = "C:/waste-segrigation/data/design project.csv"
df = pd.read_csv(file_path)

# Convert Date column to datetime format
df["Date(MM/DD/YY)"] = pd.to_datetime(df["Date(MM/DD/YY)"], errors='coerce')
df = df.dropna(subset=["Date(MM/DD/YY)"])

# Fill missing values
columns_to_fill = ["Plastic(kg)", "Food(kg)", "Paper(kg)", "Electronic(kg)", "Hazard(kg)", "Sum"]
for col in columns_to_fill:
    df[col] = df[col].fillna(df[col].median())

# Extract features
df["Month"] = df["Date(MM/DD/YY)"].dt.month
df["Day"] = df["Date(MM/DD/YY)"].dt.day
df["Year"] = df["Date(MM/DD/YY)"].dt.year

# Define features and targets
features = ["Month", "Day", "Food(kg)", "Plastic(kg)", "Paper(kg)", "Electronic(kg)", "Hazard(kg)"]
targets = ["Food(kg)", "Plastic(kg)", "Paper(kg)", "Electronic(kg)", "Hazard(kg)", "Sum"]

X = df[features]
y = df[targets]

# Train the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict_year/<int:year>")
def predict_year(year):
    future_dates = pd.date_range(start=f"{year}-01-01", periods=365, freq='D')
    future_df = pd.DataFrame({"Month": future_dates.month, "Day": future_dates.day})

    for col in ["Food(kg)", "Plastic(kg)", "Paper(kg)", "Electronic(kg)", "Hazard(kg)"]:
        future_df[col] = df[col].median()

    future_predictions = model.predict(future_df[features])
    future_df[targets] = future_predictions
    monthly_predictions = future_df.groupby("Month")[targets].sum()

    response = {col: {"months": list(monthly_predictions.index), "values": list(monthly_predictions[col])} for col in targets}
    return jsonify(response)

@app.route("/predict_month/<int:month>")
def predict_month(month):
    future_df = pd.DataFrame({"Month": [month] * 30, "Day": list(range(1, 31))})

    for col in ["Food(kg)", "Plastic(kg)", "Paper(kg)", "Electronic(kg)", "Hazard(kg)"]:
        future_df[col] = df[col].median()

    future_predictions = model.predict(future_df[features])
    future_df[targets] = future_predictions
    monthly_total = future_df[targets].sum()

    response = {col: monthly_total[col] for col in targets}
    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True)
