import streamlit as st
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

st.set_page_config(layout = "wide")

scaler = joblib.load("Scaler.pkl")

st.title("Restaurant Rating Prediction App")

st.caption("This app help you to predict a restaurants review class")

st.divider()

averagecost = st.number_input("Please the estimated average cost for two", min_value = 50, max_value = 99999, value=1000, step=200)

tablebooking = st.checkbox("Restaurant has table booking?", ["Yes", "No"])

onlinedelivery = st.selectbox("restaurant has online booking?", ["Yes", "No"])

pricerange = st.selectbox("What is the price range (1 cheapest, 4 Most Expensive)", [1,2,3,4])

predictbutton = st.button("Predict the review")

st.divider()

model = joblib.load("mlmodel.pkl")

bookingstatus = 1 if tablebooking == "yes" else 0

deliverystatus = 1 if onlinedelivery == "yes" else 0

values = [[averagecost, bookingstatus, deliverystatus, pricerange]]
my_X_values = np.array(values)

X = scaler.transform(my_X_values)

if predictbutton:
 st.snow()
 
 prediction = model.predict(X)
 
 st.write(prediction)
 
 if prediction < 2.5:
  st.write("Poor")
 elif prediction < 3.5:
  st.write("Average")
 elif prediction < 4.0:
  st.write("Good")
 elif prediction < 4.5:
  st.write("Very Good")
 else:
  st.write("Excelent")