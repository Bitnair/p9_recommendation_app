# 'streamlit_app.py' script

import streamlit as st
import requests
import json

# Azure Function endpoint URL
AZURE_FUNCTION_URL = "https://p9-recommendation-function.azurewebsites.net/api/MyFunction"
# AZURE_FUNCTION_URL = "http://localhost:7071/api/MyFunction" # Local testing

st.title("Système de Recommandation d'Articles")

st.write("Veuillez entrer un ID d'utilisateur pour obtenir 5 recommandations d'articles personnalisées.")

# Input for user_id
user_id = st.number_input("ID utilisateur", min_value=1, value=1, step=1)

if st.button("Obtenir les recommandations"):
    # Call the Azure Function endpoint
    params = {'user': user_id}
    response = requests.get(AZURE_FUNCTION_URL, params=params)

    if response.status_code == 200:
        recommended_items = json.loads(response.text)
        st.write(f"Articles recommandés pour l'utilisateur {user_id}:")
        for idx, item in enumerate(recommended_items, start=1):
            st.write(f"{idx}. Article ID: {item}")
    else:
        st.write(f"Error: {response.text}")

if response.status_code == 200:
    try:
        recommended_items = json.loads(response.text)
        st.write(f"Articles recommandés pour l'utilisateur {user_id}:")
        for idx, item in enumerate(recommended_items, start=1):
            st.write(f"{idx}. Article ID: {item}")
    except json.decoder.JSONDecodeError:
        st.write("The response from the function was not valid JSON. Raw response:")
        st.write(response.text)
else:
    st.write(f"Error: {response.status_code} - {response.text}")