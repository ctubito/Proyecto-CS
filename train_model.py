import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

#Load data
data = pd.read_csv("data/penguins.csv")

#Visualize data
#print(data.head())

#Describe data
#print(data.info())
#print(data.describe())

#Preprocessing
data = data.dropna()

#Selectt features and target
features = [
    "bill_length_mm",
    "bill_depth_mm",
    "flipper_length_mm",
    "body_mass_g",
    "island",
    "sex",
]

target = "species"

#Divide x and y
X = data[features]
y = data[target]

#Divide data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Define preprocessing for numeric and categorical features
numeric_features = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
categorical_features = ["island", "sex"]

#Preprocessing pipelines
preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", "passthrough", numeric_features),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)
#train model
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier",
      RandomForestClassifier(n_estimators=100, random_state=42))
      ])

model.fit(X_train, y_train)

#Evaluate model
y_pred = model.predict(X_test)
accuracy = model.score(X_test, y_pred)

print(f"Model accuracy: {accuracy:.2f}")

print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

#Save model
joblib.dump(model, "models/penguin_model.pkl")
