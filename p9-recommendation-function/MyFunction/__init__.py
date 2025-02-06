# __init__.py: Entry point for the Azure Function App.
"""
Returns predictions as a list of recommended item IDs for a given user ID.
"""

import logging
import azure.functions as func
from azure.storage.blob import BlobServiceClient
print("BlobServiceClient successfully imported")

import ssl
import certifi
import os
import json
import numpy as np
import scipy.sparse as sparse
import pickle
import tempfile

# 1. DEBUG LOGGING FOR AZURE STORAGE
# Display HTTP requests/responses for BlobServiceClient
azure_logger = logging.getLogger("azure.core.pipeline.policies.http_logging_policy")
azure_logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
azure_logger.addHandler(handler)

def load_object(input_stream, file_name):
    """
    Saves an InputStream to a temporary file, then loads it based on file extension.
    For .npy files, it first tries loading with allow_pickle=False.
    """
    try:
        temp_dir = tempfile.gettempdir()
        file_path = os.path.join(temp_dir, file_name)
        
        # Read the bytes from the input stream.
        data = input_stream.read()
        # Log the first 10 bytes in hex.
        logging.info(f"First 10 bytes of {file_name}: {data[:10].hex()}")
        
        # (Optionally) Remove a BOM if it exists.
        bom = b'\xef\xbb\xbf'
        if data.startswith(bom):
            logging.info(f"Detected BOM in {file_name}; stripping it.")
            data = data[len(bom):]
        
        # Write the (possibly BOM-stripped) data to a temporary file.
        with open(file_path, 'wb') as f:
            f.write(data)
        
        if file_name.endswith('.npy'):
            try:
                obj = np.load(file_path, allow_pickle=False)
            except Exception as first_err:
                logging.warning(f"Failed to load {file_name} with allow_pickle=False: {first_err}. Trying with allow_pickle=True.")
                obj = np.load(file_path, allow_pickle=True)
            logging.info(f"Loaded .npy file: {file_name} with shape {obj.shape}")
            return obj
        elif file_name.endswith('.npz'):
            obj = sparse.load_npz(file_path)
            logging.info(f"Loaded .npz file: {file_name} with shape {obj.shape}")
            return obj
        elif file_name.endswith('.pkl'):
            with open(file_path, 'rb') as f:
                obj = pickle.load(f)
            logging.info(f"Loaded .pkl file: {file_name}")
            return obj
        else:
            raise ValueError(f"Unsupported file format: {file_name}")
    except Exception as e:
        logging.error(f"Error loading file {file_name}: {e}")
        raise


def main(
    req: func.HttpRequest,
    userFactors: func.InputStream,
    itemFactors: func.InputStream,
    interactionMatrix: func.InputStream,
    userIdMap: func.InputStream,
    itemIdMap: func.InputStream
) -> func.HttpResponse:
    """
    Returns predictions as a list of recommended item IDs for a given user ID.
    Combines SSL debugging, advanced logging, and recommendation logic.
    """

    logging.info("Azure Function is running correctly (merged version).")

    # 1) Log OpenSSL version and retrieve connection string
    logging.info(f"OpenSSL version: {ssl.OPENSSL_VERSION}")
    connection_string = os.getenv("AzureWebJobsStorage")
    if not connection_string or "DefaultEndpointsProtocol" not in connection_string:
        return func.HttpResponse("AzureWebJobsStorage is missing or invalid.", status_code=500)
    logging.info(f"AzureWebJobsStorage: {connection_string}")

    # 2) Create BlobServiceClient using a custom SSL certificate (certifi)
    try:
        blob_service_client = BlobServiceClient.from_connection_string(
            connection_string,
            connection_verify=certifi.where()
        )
        logging.info("✅ Successfully connected to Azure Blob Storage.")

        # (Optional) list containers for debugging
        containers = blob_service_client.list_containers()
        container_names = [c['name'] for c in containers]
        logging.info(f"Available containers: {container_names}")

    except Exception as e:
        logging.error("Error creating BlobServiceClient.")
        logging.error(str(e))
        return func.HttpResponse(f"Error connecting to Azure Blob Storage: {str(e)}", status_code=500)

    # 3) Extract user ID from the query parameter
    user_id_str = req.params.get('user')
    if not user_id_str:
        logging.warning("Missing 'user' parameter.")
        return func.HttpResponse("Missing 'user' parameter.", status_code=400)
    try:
        user_id = int(user_id_str)
        logging.info(f"Converted user_id: {user_id}")
    except ValueError:
        logging.warning(f"Invalid 'user' parameter: {user_id_str}")
        return func.HttpResponse("Invalid 'user' parameter. Must be an integer.", status_code=400)

    try:
        # 4) Load model components from InputStreams
        user_factors_arr = load_object(userFactors, "user_factors.npy")
        item_factors_arr = load_object(itemFactors, "item_factors.npy")
        interaction_mat   = load_object(interactionMatrix, "interaction_matrix.npz")  # expected to be a sparse matrix
        user_id_map       = load_object(userIdMap, "user_id_map.pkl")
        item_id_map       = load_object(itemIdMap, "item_id_map.pkl")

        # 5) Look up this user in user_id_map
        user_idx = user_id_map.get(user_id)
        if user_idx is None:
            logging.warning(f"User ID {user_id} not found in user_id_map.")
            return func.HttpResponse("User not found in user_id_map.", status_code=404)

        # 6) Generate recommendations:
        # Compute scores using the dot-product between the user's factor vector and each item’s factor vector.
        user_vec = user_factors_arr[user_idx]
        scores = np.dot(item_factors_arr, user_vec)  # scores is a 1D array with one score per item

        # Exclude items that the user has already interacted with.
        user_already_liked = interaction_mat[user_idx].indices  # assuming a CSR row format
        scores[user_already_liked] = -np.inf

        # 7) Get the top 5 recommendations.
        top_5_idx = np.argpartition(scores, -5)[-5:]
        top_5_idx = top_5_idx[np.argsort(scores[top_5_idx])[::-1]]

        # Map the item indices to real item IDs.
        recommended_items = []
        for i_idx in top_5_idx:
            if i_idx in item_id_map:
                recommended_items.append(item_id_map[i_idx])
            else:
                logging.warning(f"Item index {i_idx} not found in item_id_map; skipping.")

        logging.info(f"Recommended items for user {user_id}: {recommended_items}")

        # 8) Return the JSON response
        return func.HttpResponse(
            json.dumps({"user": user_id, "recommendations": recommended_items}),
            mimetype="application/json",
            status_code=200
        )

    except Exception as e:
        logging.error(f"Error processing recommendations for user {user_id}: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return func.HttpResponse(
            f"Internal server error while recommending for user {user_id}: {str(e)}",
            status_code=500
        )
