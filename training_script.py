# 'training_script.py' script
"""
Load data from Blob Storage, train a collaborative filtering model, and store the artifacts back.
"""

import os
import math
import pickle
import tempfile
import numpy as np
import pandas as pd
import scipy.sparse as sparse
from azure.storage.blob import BlobServiceClient
from implicit.als import AlternatingLeastSquares
from main import CollaborativeFilteringRecommender
# from dotenv import load_dotenv  # Load environment variables from .env file for local testing


## For local testing: load environment variables from .env file
# load_dotenv('credentials.env')  

# CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
# if not CONNECTION_STRING:
#     raise ValueError("AZURE_STORAGE_CONNECTION_STRING is not set. Check the .env file.")
## For local testing: load environment variables from .env file


# Load environment variable from Azure 'Environment variables' settings 
CONNECTION_STRING = AZURE_STORAGE_CONNECTION_STRING

def load_csv_from_blob(blob_service_client, container_name, blob_name):
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
    download_stream = blob_client.download_blob()
    df = pd.read_csv(download_stream)
    return df

def save_object_to_blob(obj, blob_service_client, container_name, file_name):
    temp_dir = tempfile.gettempdir()
    file_path = os.path.join(temp_dir, file_name)
    if file_name.endswith('.npy'):
        np.save(file_path, obj)
    elif file_name.endswith('.npz'):
        sparse.save_npz(file_path, obj)
    elif file_name.endswith('.pkl'):
        with open(file_path, 'wb') as f:
            pickle.dump(obj, f)
    else:
        raise ValueError("Unsupported file format")
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=file_name)
    with open(file_path, 'rb') as data:
        blob_client.upload_blob(data, overwrite=True)
    os.remove(file_path)


def main():
    # Connect to Azure Blob Storage
    blob_service_client = BlobServiceClient.from_connection_string(CONNECTION_STRING)
    input_container = 'input-data'
    output_container = 'trained-model'

    # Load Data
    print("Loading data...")
    container_client = blob_service_client.get_container_client(input_container)
    df_list = []
    for blob in container_client.list_blobs():
        if 'hour' in blob.name:
            df = load_csv_from_blob(blob_service_client, input_container, blob.name)
            df_list.append(df)
        elif 'articles_metadata.csv' in blob.name:
            items_df = load_csv_from_blob(blob_service_client, input_container, blob.name)
    df = pd.concat(df_list, ignore_index=True)

    # Initialize and Train Model
    print("Training model...")
    cf_recommender = CollaborativeFilteringRecommender(df, items_df)
    cf_recommender.fit()

    # Save Model Components
    print("Saving model components...")
    components = {
        'user_factors.npy': cf_recommender.user_factors,
        'item_factors.npy': cf_recommender.item_factors,
        'interaction_matrix.npz': cf_recommender.interaction_matrix,
        'user_id_map.pkl': cf_recommender.user_id_map,
        'item_id_map.pkl': cf_recommender.item_id_map
    }
    for file_name, obj in components.items():
        save_object_to_blob(obj, blob_service_client, output_container, file_name)

    print("Model training and saving completed.")

    # ## For local testing
    # # Local directory to save model artifacts
    # local_dir = "/Users/tgeof/Documents/Documents/B - Travaux Perso/1 - Scolarité/Ingenieur IA/OpenClassroom/_Projets/9. Réalisez une application de recommandation de contenu/2_project/test4/local_data"
    # os.makedirs(local_dir, exist_ok=True)

    # np.save(os.path.join(local_dir, 'user_factors.npy'), cf_recommender.user_factors)
    # np.save(os.path.join(local_dir, 'item_factors.npy'), cf_recommender.item_factors)
    # sparse.save_npz(os.path.join(local_dir, 'interaction_matrix.npz'), cf_recommender.interaction_matrix)

    # with open(os.path.join(local_dir, 'user_id_map.pkl'), 'wb') as f:
    #     pickle.dump(cf_recommender.user_id_map, f)

    # with open(os.path.join(local_dir, 'item_id_map.pkl'), 'wb') as f:
    #     pickle.dump(cf_recommender.item_id_map, f)
    # ## For local testing

if __name__ == "__main__":
    main()


