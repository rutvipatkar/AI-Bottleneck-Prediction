import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Load waiting-time training data
data = pd.read_csv("waiting_time_data.csv")

# Input features
X = data[[
    "patients",
    "beds",
    "staff",
    "emergency_cases"
]]

# Target
y = data["waiting_time"]

# Create model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

# Train model
model.fit(X, y)

print("Waiting-time model trained successfully!")

# Test prediction
new_data = pd.DataFrame(
    [[60, 5, 12, 10]],
    columns=[
        "patients",
        "beds",
        "staff",
        "emergency_cases"
    ]
)

predicted_time = model.predict(new_data)

print(
    "Predicted waiting time:",
    round(predicted_time[0], 2),
    "minutes"
)