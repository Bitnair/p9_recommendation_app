# '__init__.py' script: entry point for the Azure Function App.
"""
Returns predictions as a list of recommended item IDs for a given user ID.
"""

import logging
import azure.functions as func
from azure.storage.blob import BlobServiceClient
import ssl
import certifi
import os

def main(
    req: func.HttpRequest,
    userFactors: func.InputStream,
    itemFactors: func.InputStream,
    interactionMatrix: func.InputStream,
    userIdMap: func.InputStream,
    itemIdMap: func.InputStream
) -> func.HttpResponse:
    
    logging.info("Azure Function is running correctly.")

    # Log OpenSSL version to ensure TLS 1.2+ support
    logging.info(f"OpenSSL version: {ssl.OPENSSL_VERSION}")

    # Retrieve and log environment variables
    connection_string = os.getenv("AzureWebJobsStorage")
    if not connection_string or "DefaultEndpointsProtocol" not in connection_string:
        logging.error("AzureWebJobsStorage is missing or incorrectly set.")
        return func.HttpResponse("Azure Storage connection string is missing or invalid.", status_code=500)
    logging.info(f"AzureWebJobsStorage: {connection_string}")

    try:
        # Initialize BlobServiceClient with connection_verify
        blob_service_client = BlobServiceClient.from_connection_string(
            connection_string, connection_verify=certifi.where()
        )
        logging.info("✅ Successfully connected to Azure Blob Storage.")

        # List containers to verify access
        containers = blob_service_client.list_containers()
        container_names = [container['name'] for container in containers]
        logging.info(f"Available containers: {container_names}")

        # Attempt to read one blob to test
        user_factors_data = userFactors.read()
        logging.info(f"Loaded user_factors.npy. Size: {len(user_factors_data)} bytes.")

        return func.HttpResponse(f"Blob data loaded successfully.", status_code=200)
    
    except Exception as e:
        logging.error("Error reading blobs:")
        logging.error(str(e))
        # Log the traceback for detailed debugging
        import traceback
        logging.error(traceback.format_exc())
        return func.HttpResponse(f"Error reading blob data: {str(e)}", status_code=500)
