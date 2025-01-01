# 'streamlit_app.py' script

import streamlit as st
import requests
import json

# Azure Function endpoint URL
AZURE_FUNCTION_URL = "p9-recommendation-function.azurewebsites.net"
# AZURE_FUNCTION_URL = "http://localhost:7071/api/MyFunction" # Local testing

st.title("Article Recommendation System")

st.write("This interface queries an Azure Function that serves article recommendations for a given user.")

# Input for user_id
user_id = st.number_input("Enter a User ID", min_value=1, value=1, step=1)

if st.button("Get Recommendations"):
    # Call the Azure Function endpoint
    params = {'user': user_id}
    response = requests.get(AZURE_FUNCTION_URL, params=params)

    if response.status_code == 200:
        recommended_items = json.loads(response.text)
        st.write(f"Recommended Articles for User {user_id}:")
        for idx, item in enumerate(recommended_items, start=1):
            st.write(f"{idx}. Article ID: {item}")
    else:
        st.write(f"Error: {response.text}")