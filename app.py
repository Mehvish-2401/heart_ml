import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

# Load trained model
model = pickle.load(open("model.pkl", "rb"))

st.title("Heart Disease Risk Prediction App")

st.write("This AI application predicts heart disease risk using a trained Machine Learning model.")

# Tabs
tab1, tab2, tab3 = st.tabs(["Manual Prediction", "Upload CSV", "Dashboard"])

uploaded_predictions = None


# -------------------------
# TAB 1 : MANUAL PREDICTION
# -------------------------

with tab1:

    st.header("Enter Patient Data")

    age = st.number_input("Age", 20, 100)
    sex = st.selectbox("Sex (1 = Male, 0 = Female)", [1,0])
    cp = st.selectbox("Chest Pain Type (1-4)", [1,2,3,4])
    trestbps = st.number_input("Resting Blood Pressure", 80,200)
    chol = st.number_input("Cholesterol",100,600)
    fbs = st.selectbox("Fasting Blood Sugar >120 (1=True,0=False)",[0,1])
    restecg = st.selectbox("Rest ECG (0-2)",[0,1,2])
    thalach = st.number_input("Max Heart Rate",60,220)
    exang = st.selectbox("Exercise Induced Angina",[0,1])
    oldpeak = st.number_input("ST Depression",0.0,10.0)
    slope = st.selectbox("Slope (1-3)",[1,2,3])
    ca = st.selectbox("Major Vessels (0-3)",[0,1,2,3])
    thal = st.selectbox("Thal (3=Normal,6=Fixed,7=Reversible)",[3,6,7])

    if st.button("Predict"):

        input_data = np.array([[age,sex,cp,trestbps,chol,fbs,
                                restecg,thalach,exang,oldpeak,
                                slope,ca,thal]])

        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        st.subheader("Model Confidence")

        st.progress(int(probability * 100))
        st.write(f"Risk Probability: {probability:.2f}")

        if prediction == 1:
            st.error("⚠️ High Risk of Heart Disease")
        else:
            st.success("✅ Low Risk of Heart Disease")


# -------------------------
# TAB 2 : CSV UPLOAD
# -------------------------

with tab2:

    st.header("Batch Prediction via CSV")

    file = st.file_uploader("Upload CSV file")

    if file is not None:

        data = pd.read_csv(file)

        st.subheader("Uploaded Data")
        st.write(data)

        if st.button("Run Model on Uploaded Data"):

            predictions = model.predict(data)
            probabilities = model.predict_proba(data)[:,1]

            data["Prediction"] = predictions
            data["Risk Level"] = data["Prediction"].map({0:"Low Risk",1:"High Risk"})
            data["Risk Probability"] = probabilities

            uploaded_predictions = predictions

            st.subheader("Prediction Results")
            st.write(data)


# -------------------------
# TAB 3 : DASHBOARD
# -------------------------

with tab3:

    st.header("Prediction Dashboard")

    if uploaded_predictions is not None:

        low = list(uploaded_predictions).count(0)
        high = list(uploaded_predictions).count(1)

        labels = ["Low Risk","High Risk"]
        sizes = [low,high]

        fig, ax = plt.subplots()
        ax.pie(sizes, labels=labels, autopct='%1.1f%%')
        ax.set_title("Risk Distribution")

        st.pyplot(fig)

    else:

        st.info("Upload a dataset in the CSV tab to see analytics.")
