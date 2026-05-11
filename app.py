

import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# -----------------------------
# LOAD DATASET
# -----------------------------
df = pd.read_excel("flower_dataset.xlsx")

X = df.drop("label", axis=1)
y = df["label"]

# -----------------------------
# TRAIN TEST SPLIT
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# TRAIN MODEL
# -----------------------------
model = RandomForestClassifier()
model.fit(X_train, y_train)

# -----------------------------
# TEST MODEL
# -----------------------------
pred = model.predict(X_test)
accuracy = accuracy_score(y_test, pred)

# -----------------------------
# STREAMLIT UI
# -----------------------------
st.title("🌸 Flower Prediction ML App")

st.write("This app is built using your Excel dataset")

st.subheader("Model Accuracy")
st.success(f"{accuracy * 100:.2f}%")

st.subheader("Enter Flower Features")

sl = st.slider("Sepal Length", 4.0, 8.0, 5.0)
sw = st.slider("Sepal Width", 2.0, 5.0, 3.0)
pl = st.slider("Petal Length", 1.0, 7.0, 4.0)
pw = st.slider("Petal Width", 0.1, 3.0, 1.0)

input_data = [[sl, sw, pl, pw]]

prediction = model.predict(input_data)

labels = ["Setosa", "Versicolor", "Virginica"]

st.subheader("Prediction")
st.info(labels[prediction[0]])

st.subheader("Dataset Preview")
st.dataframe(df.head())

