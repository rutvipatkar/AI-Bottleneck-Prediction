import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


data = pd.read_csv("hospital_data.csv")


X = data[["patients", "beds", "staff", "waiting_time", "emergency_cases"]]


y = data["bottleneck"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


model = RandomForestClassifier(random_state=42)


model.fit(X_train, y_train)


accuracy = model.score(X_test, y_test)

print("Model accuracy:", accuracy)


new_data = pd.DataFrame(
    [[70, 4, 9, 60, 20]],
    columns=["patients", "beds", "staff", "waiting_time", "emergency_cases"]
)


prediction = model.predict(new_data)

probability = model.predict_proba(new_data)


risk = probability[0][1] * 100

print("Bottleneck prediction:", prediction[0])
print("Bottleneck probability:", round(risk, 2), "%")


if risk >= 70:
    risk_level = "HIGH"
elif risk >= 40:
    risk_level = "MEDIUM"
else:
    risk_level = "LOW"

print("Risk level:", risk_level)


if risk_level == "HIGH":
    print("Recommendation: Increase staff and available beds immediately.")
elif risk_level == "MEDIUM":
    print("Recommendation: Monitor the department and prepare additional resources.")
else:
    print("Recommendation: Current resources are sufficient.")
