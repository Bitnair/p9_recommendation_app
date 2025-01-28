"""
Returns predictions as a list of recommended item IDs for a given user ID.
"""

import logging
import azure.functions as func
import numpy as np
import scipy.sparse as sparse
import os
import tempfile
import json
import pickle

def load_object(input_stream, file_name):
    try:
        temp_dir = tempfile.gettempdir()
        file_path = os.path.join(temp_dir, file_name)
        with open(file_path, 'wb') as f:
            f.write(input_stream.read())
        if file_name.endswith('.npy'):
            return np.load(file_path, allow_pickle=True)
        elif file_name.endswith('.npz'):
            return sparse.load_npz(file_path)
        elif file_name.endswith('.pkl'):
            with open(file_path, 'rb') as f:
                return pickle.load(f)
        else:
            raise ValueError(f"Unsupported file format: {file_name}")
    except Exception as e:
        logging.error(f"Error loading file {file_name}: {e}")
        raise

def main(req: func.HttpRequest, userFactors: func.InputStream,
         itemFactors: func.InputStream, interactionMatrix: func.InputStream,
         userIdMap: func.InputStream, itemIdMap: func.InputStream) -> func.HttpResponse:
    logging.info('Processing recommendation request.')

    # Parse and validate user parameter
    user_id_str = req.params.get('user')
    logging.info(f"Received user_id_str: {user_id_str}")
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
        # Load model components
        userFactors = load_object(userFactors, 'user_factors.npy')
        logging.info(f"Loaded user factors: {userFactors.shape}")
        itemFactors = load_object(itemFactors, 'item_factors.npy')
        logging.info(f"Loaded item factors: {itemFactors.shape}")
        interactionMatrix = load_object(interactionMatrix, 'interaction_matrix.npz')
        logging.info(f"Loaded interaction matrix with shape: {interactionMatrix.shape}")
        userIdMap = load_object(userIdMap, 'user_id_map.pkl')
        logging.info(f"Loaded user ID map: {len(userIdMap)} entries")
        itemIdMap = load_object(itemIdMap, 'item_id_map.pkl')
        logging.info(f"Loaded item ID map: {len(itemIdMap)} entries")

        # Generate Recommendations
        user_idx = userIdMap.get(user_id)
        if user_idx is None:
            logging.warning(f"User ID {user_id} not found in userIdMap.")
            return func.HttpResponse("User not found.", status_code=404)

        user_vector = userFactors[user_idx]
        scores = user_vector.dot(itemFactors.T)
        liked_items = interactionMatrix[user_idx].indices
        scores[liked_items] = -np.inf

        # Select top 5 recommendations
        recommended_indices = np.argpartition(scores, -5)[-5:]
        recommended_indices = recommended_indices[np.argsort(scores[recommended_indices])[::-1]]
        recommended_items = [itemIdMap[idx] for idx in recommended_indices]

        logging.info(f"Recommended items for user {user_id}: {recommended_items}")

        return func.HttpResponse(json.dumps(recommended_items), mimetype="application/json", status_code=200)

    except Exception as e:
        logging.error(f"Error processing recommendation: {e}")
        return func.HttpResponse(f"Internal server error: {e}", status_code=500)
