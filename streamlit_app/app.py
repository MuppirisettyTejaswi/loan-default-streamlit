import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

model = joblib.load("loan_default_app/model_training/loan_model.pkl")
train_columns = joblib.load("loan_default_app/model_training/train_columns.pkl")
train_mean = joblib.load("loan_default_app/model_training/train_mean.pkl")


st.title("Loan Default Risk Prediction")
st.markdown("Enter borrower information to predict the risk of loan default.")

st.sidebar.header("Enter Borrower Information")
age = st.sidebar.slider("Age", 18, 80, 30)
fare = st.sidebar.slider("Monthly Income", 0, 500, 50)
sex = st.sidebar.selectbox("Gender", ("Male", "Female"))
pclass = st.sidebar.selectbox("Credit Score Tier (1=High)", [1, 2, 3])
sibsp = st.sidebar.selectbox("Number of Dependents", [0, 1, 2, 3])

sex_encoded = 1 if sex == "Male" else 0

input_data = {
    'age': age,
    'fare': fare,
    'sex': sex_encoded,
    'pclass': pclass,
    'sibsp': sibsp
}
input_df = pd.DataFrame([input_data])
input_df = input_df[train_columns]

st.subheader(" Input Summary")
st.dataframe(input_df)

prediction = model.predict(input_df)[0]
proba = model.predict_proba(input_df)[0][1]

label = "❌ Likely to Default" if prediction == 1 else "✅ Not Likely to Default"
st.success(f"Prediction: **{label}**")
st.info(f"Probability of Default: **{proba:.2%}**")

st.subheader(" Input Feature Values")
fig1 = px.bar(x=input_df.columns, y=input_df.iloc[0], labels={'x': 'Feature', 'y': 'Value'})
st.plotly_chart(fig1)

st.subheader(" Input vs Dataset Averages")
mean_df = pd.DataFrame([train_mean, input_df.iloc[0]], index=["Dataset Avg", "Input"]).T
st.bar_chart(mean_df)

st.subheader(" Feature Importance")
try:
    importances = model.feature_importances_
    importance_df = pd.DataFrame({
        "Feature": train_columns,
        "Importance": importances
    }).sort_values(by="Importance", ascending=True)

    fig2 = px.bar(importance_df, x="Importance", y="Feature", orientation='h')
    st.plotly_chart(fig2)

except Exception as e:
    st.warning("Feature importance not available.")
    st.exception(e)
