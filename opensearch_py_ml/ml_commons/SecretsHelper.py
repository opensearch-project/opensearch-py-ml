# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

import json
import logging

import boto3
from botocore.exceptions import ClientError

# Configure the logger for this module
logger = logging.getLogger(__name__)


class SecretHelper:
    """
    Helper class for managing secrets in AWS Secrets Manager.
    Provides methods to check existence, retrieve details, and create secrets.
    """

    def __init__(
        self,
        region: str,
        aws_access_key: str,
        aws_secret_access_key: str,
        aws_session_token: str,
    ):
        """
        Initialize the SecretHelper with the specified AWS region.
        :param region: AWS region where the Secrets Manager is located.
        """
        self.region = region
        self.aws_access_key = aws_access_key
        self.aws_secret_access_key = aws_secret_access_key
        self.aws_session_token = aws_session_token

        # Create the Secrets Manager client once at the class level
        self.secretsmanager = boto3.client(
            "secretsmanager",
            region_name=self.region,
            aws_access_key_id=self.aws_access_key,
            aws_secret_access_key=self.aws_secret_access_key,
            aws_session_token=self.aws_session_token,
        )

    def secret_exists(self, secret_name: str) -> bool:
        """
        Check if a secret with the given name exists in AWS Secrets Manager.
        :param secret_name: Name of the secret to check.
        :return: True if the secret exists, False otherwise.
        """
        try:
            # Attempt to retrieve the secret value
            self.secretsmanager.get_secret_value(SecretId=secret_name)
            return True
        except ClientError as e:
            # If the secret does not exist, return False
            if e.response["Error"]["Code"] == "ResourceNotFoundException":
                return False
            else:
                # Log other client errors and return False
                logger.error(f"An error occurred: {e}")
                return False

    def get_secret_arn(self, secret_name: str) -> str:
        """
        Retrieve the ARN of a secret from AWS Secrets Manager.
        : param secret_name: Name of the secret to retrieve the ARN for.
        : return: ARN of the secret if found, None otherwise.
        """
        try:
            response = self.secretsmanager.describe_secret(SecretId=secret_name)
            return response["ARN"]
        except self.secretsmanager.exceptions.ResourceNotFoundException:
            print(f"The requested secret {secret_name} was not found")
            return None
        except Exception as e:
            print(f"An error occurred: {e}")
            return None

    def get_secret_details(self, secret_name: str, fetch_value: bool = False) -> dict:
        """
        Retrieve details of a secret from AWS Secrets Manager.
        Optionally fetch the secret value as well.
        :param secret_name: Name of the secret.
        :param fetch_value: Whether to also fetch the secret value (default is False).
        :return: A dictionary with secret details (ARN and optionally the secret value)
                 or an error dictionary if something went wrong.
        """
        try:
            # Describe the secret to get its ARN and metadata
            describe_response = self.secretsmanager.describe_secret(
                SecretId=secret_name
            )

            secret_details = {
                "ARN": describe_response["ARN"],
                # You can add more fields from `describe_response` if needed
            }

            # Fetch the secret value if requested
            if fetch_value:
                value_response = self.secretsmanager.get_secret_value(
                    SecretId=secret_name
                )
                secret_details["SecretValue"] = value_response.get("SecretString")

            return secret_details

        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code == "ResourceNotFoundException":
                logger.warning(f"The requested secret '{secret_name}' was not found")
            else:
                logger.error(
                    f"An error occurred while fetching secret '{secret_name}': {e}"
                )
            # Return a dictionary with error details
            return {"error": str(e), "error_code": error_code}

    def create_secret(self, secret_name: str, secret_value: dict) -> str:
        """
        Create a new secret in AWS Secrets Manager.
        :param secret_name: Name of the secret to create.
        :param secret_value: Dictionary containing the secret data.
        :return: ARN of the created secret if successful, None otherwise.
        """
        try:
            # Create the secret with the provided name and value
            response = self.secretsmanager.create_secret(
                Name=secret_name,
                SecretString=json.dumps(secret_value),
            )
            # Log success and return the secret's ARN
            logger.info(f"Secret '{secret_name}' created successfully.")
            return response["ARN"]
        except ClientError as e:
            # Log errors during secret creation and return None
            logger.error(f"Error creating secret '{secret_name}': {e}")
            return None
