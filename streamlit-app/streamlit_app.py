# 'streamlit_app.py' script

import streamlit as st
import requests
import json

# Azure Function endpoint URL
AZURE_FUNCTION_URL = "https://p9-recommendation-function.azurewebsites.net/api/MyFunction"
# AZURE_FUNCTION_URL = "http://localhost:7071/api/MyFunction" # Local testing

st.title("Recommandation d'articles")

st.write("Entrez un ID d'utilisateur pour obtenir 5 recommandations d'articles personnalisées.")

# Input for user_id
user_id = st.number_input("ID utilisateur", min_value=1, value=1, step=1)

if st.button("Obtenir les recommandations"):
    # Call the Azure Function endpoint
    params = {'user': user_id}
    response = requests.get(AZURE_FUNCTION_URL, params=params, timeout=120)

    if response.status_code == 200:
        try:
            data = json.loads(response.text)
            st.write(f"ID d'articles recommandés pour cet utilisateur :")
            # Get the recommendations list from the response object.
            recommended_items = data.get("recommendations", [])
            for idx, item in enumerate(recommended_items, start=1):
                st.write(f"#{idx} : {item}")
        except json.decoder.JSONDecodeError:
            st.write("The response from the function was not a valid JSON. Raw response:")
            st.write(response.text)
    else:
        st.write(f"Error: {response.status_code} - {response.text}")
