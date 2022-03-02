#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
import xgboost

st.write("""
# Cancellation Prediction App
This app predicts if a hotel booking will be canceled or not!
""")

st.sidebar.header('User Input Features')


def user_input_features():
    hotel = st.sidebar.selectbox("Hotel Type", ('Resort', 'City'))
    lead_time = st.sidebar.slider("Lead Time", 0, 296, 1)
    country = st.sidebar.text_input("Country", " ")
    market_segment = st.sidebar.selectbox("Market Segnment", ('Online TA', 'Online TA/TO', 'Groups', 'Direct', 'Corporate', 'Complementary', 'Aviation'))
    distribution_channel = st.sidebar.selectbox("Distribution Channel", ('Direct', 'Corporate', 'TA/TO', 'GDS'))
    is_repeated_guest = st.sidebar.selectbox("Repeated Guest", ('0', '1'))
    booking_changes = st.sidebar.slider("Booking Changes", 0, 100,1)
    deposit_type = st.sidebar.selectbox("Deposit Type", ('No Deposit', 'Refundable', 'Non Refund'))
    agent = st.sidebar.slider("Agent", 1, 300, 1)
    days_in_waiting_list = st.sidebar.slider("Days In Waiting List", 0, 391, 1)
    customer_type = st.sidebar.selectbox("Customer Type", ('Transient', 'Transient-Party', 'Contract', 'Group'))
    adr = st.sidebar.slider("ADR", -10, 230, 10)
    required_car_parking_spaces = st.sidebar.slider("Car Parking Spaces", 0, 8, 1)
    total_of_special_requests = st.sidebar.slider("Special Requests", 0, 5, 1)
    guests = st.sidebar.slider("Guests", 0, 55, 1)
    room = st.sidebar.slider("Room", 0, 1)
    net_canceled = st.sidebar.slider("Net Canceled", 0, 1)
    data = {'hotel': hotel,
            'lead_time': lead_time,
            'country': country,
            'market_segment': market_segment,
            'distribution_channel': distribution_channel,
            'is_repeated_guest': is_repeated_guest,
            'booking_changes': booking_changes,
            'deposit_type': deposit_type,
            'agent': agent,
            'days_in_waiting_list': days_in_waiting_list,
            'customer_type': customer_type,
            'adr': adr,
            'required_car_parking_spaces': required_car_parking_spaces,
            'total_of_special_requests': total_of_special_requests,
            'guests': guests,
            'room': room,
            'net_canceled': net_canceled }
    features = pd.DataFrame(data, index=[0])
    return features
input_df = user_input_features()

# Combines user input features with entire bookings dataset
# This will be useful for the encoding phase
bookings_raw = pd.read_csv('C:\\Users\\yasha\\OneDrive\\Desktop\\Deployment\\bookings_before_le.csv')
bookings = bookings_raw.drop(columns=['is_canceled'])
df = pd.concat([input_df,bookings],axis=0)

# Encoding of ordinal features
df['hotel'] = LabelEncoder().fit_transform((df['hotel']))
df['country'] = LabelEncoder().fit_transform((df['country']))
df['market_segment'] = LabelEncoder().fit_transform((df['market_segment']))
df['distribution_channel'] = LabelEncoder().fit_transform((df['distribution_channel']))
df['deposit_type'] = LabelEncoder().fit_transform((df['deposit_type']))
df['customer_type'] = LabelEncoder().fit_transform((df['customer_type']))
df = df[:1] # Selects only the first row (the user input data)

# Displays the user input features
st.subheader('User Input features')

st.write('Currently using example input parameters (shown below).')
st.write(df)

check=st.checkbox("Give Prediction")
if check:

    # Reads in saved classification model
    load_clf = pickle.load(open('model.pkl', 'rb'))

    # Apply model to make predictions
    prediction = load_clf.predict(df)
    #prediction_proba = load_clf.predict_proba(df)
    print(prediction)


    st.subheader('Prediction')
    booking_cancellations = np.array(['Not Canceled', 'Canceled'])
    st.write(booking_cancellations[prediction])

    #st.subheader('Prediction Probability')
    st.write(prediction)

else:
    st.markdown("No Prediction!")

