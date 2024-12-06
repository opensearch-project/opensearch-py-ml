# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

import logging
import boto3
import json
from botocore.exceptions import ClientError

# Configure the logger for this module
logger = logging.getLogger(__name__)


class SecretHelper:
    """
    Helper class for managing secrets in AWS Secrets Manager.
    Provides methods to check existence, retrieve ARN, get secret values, and create new secrets.
    """

    def __init__(self, region: str):
        """
        Initialize the SecretHelper with the specified AWS region.

        :param region: AWS region where the Secrets Manager is located.
        """
        self.region = region

    def secret_exists(self, secret_name: str) -> bool:
        """
        Check if a secret with the given name exists in AWS Secrets Manager.

        :param secret_name: Name of the secret to check.
        :return: True if the secret exists, False otherwise.
        """
        # Initialize the Secrets Manager client
        secretsmanager = boto3.client('secretsmanager', region_name=self.region)
        try:
            # Attempt to retrieve the secret value
            secretsmanager.get_secret_value(SecretId=secret_name)
            return True
        except ClientError as e:
            # If the secret does not exist, return False
            if e.response['Error']['Code'] == 'ResourceNotFoundException':
                return False
            else:
                # Log other client errors and return False
                logger.error(f"An error occurred: {e}")
                return False

    def get_secret_arn(self, secret_name: str) -> str:
        """
        Retrieve the ARN of a secret in AWS Secrets Manager.

        :param secret_name: Name of the secret.
        :return: ARN of the secret if found, None otherwise.
        """
        # Initialize the Secrets Manager client
        secretsmanager = boto3.client('secretsmanager', region_name=self.region)
        try:
            # Describe the secret to get its details
            response = secretsmanager.describe_secret(SecretId=secret_name)
            return response['ARN']
        except ClientError as e:
            # Handle the case where the secret does not exist
            if e.response['Error']['Code'] == 'ResourceNotFoundException':
                logger.warning(f"The requested secret {secret_name} was not found")
                return None
            else:
                # Log other client errors and return None
                logger.error(f"An error occurred: {e}")
                return None

    def get_secret(self, secret_name: str) -> str:
        """
        Retrieve the secret value from AWS Secrets Manager.

        :param secret_name: Name of the secret.
        :return: Secret value as a string if found, None otherwise.
        """
        # Initialize the Secrets Manager client
        secretsmanager = boto3.client('secretsmanager', region_name=self.region)
        try:
            # Get the secret value
            response = secretsmanager.get_secret_value(SecretId=secret_name)
            return response.get('SecretString')
        except ClientError as e:
            # Handle the case where the secret does not exist
            if e.response['Error']['Code'] == 'ResourceNotFoundException':
                logger.warning("The requested secret was not found")
                return None
            else:
                # Log other client errors and return None
                logger.error(f"An error occurred: {e}")
                return None

    def create_secret(self, secret_name: str, secret_value: dict) -> str:
        """
        Create a new secret in AWS Secrets Manager.

        :param secret_name: Name of the secret to create.
        :param secret_value: Dictionary containing the secret data.
        :return: ARN of the created secret if successful, None otherwise.
        """
        # Initialize the Secrets Manager client
        secretsmanager = boto3.client('secretsmanager', region_name=self.region)
        try:
            # Create the secret with the provided name and value
            response = secretsmanager.create_secret(
                Name=secret_name,
                SecretString=json.dumps(secret_value),
            )
            # Log success and return the secret's ARN
            logger.info(f'Secret {secret_name} created successfully.')
            return response['ARN']
        except ClientError as e:
            # Log errors during secret creation and return None
            logger.error(f'Error creating secret: {e}')
            return None