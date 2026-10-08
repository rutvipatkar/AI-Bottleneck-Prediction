from sklearn.tree import DecisionTreeClassifier


X = [
    [10],
    [20],
    [30],
    [40],
    [50]
]


y = [
    0,
    0,
    0,
    1,
    1
]


model = DecisionTreeClassifier()


model.fit(X, y)


prediction = model.predict([[35]])

print("Prediction:", prediction)