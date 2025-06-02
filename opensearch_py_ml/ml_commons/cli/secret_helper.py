# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

import json
import logging
from typing import Any, Dict, Optional

import boto3
from botocore.exceptions import ClientError

from opensearch_py_ml.ml_commons.cli.aws_config import AWSConfig
from opensearch_py_ml.ml_commons.cli.opensearch_domain_config import (
    OpenSearchDomainConfig,
)

# Configure the logger for this module
logger = logging.getLogger(__name__)


class SecretHelper:
    """
    Helper class for managing secrets in AWS Secrets Manager.
    Provides methods to check existence, retrieve details, and create secrets.
    """

    def __init__(
        self, opensearch_config: OpenSearchDomainConfig, aws_config: AWSConfig
    ):
        """
        Initialize the SecretHelper with the specified AWS region.

        Args:
            region: AWS region where the Secrets Manager is located.
        """
        self.opensearch_config = opensearch_config
        self.aws_config = aws_config

        # Create the Secrets Manager client once at the class level
        self.secretsmanager = boto3.client(
            "secretsmanager",
            region_name=self.opensearch_config.opensearch_domain_region,
            aws_access_key_id=self.aws_config.aws_access_key,
            aws_secret_access_key=self.aws_config.aws_secret_access_key,
            aws_session_token=self.aws_config.aws_session_token,
        )

    def secret_exists(self, secret_name: str) -> bool:
        """
        Check if a secret with the given name exists in AWS Secrets Manager.

        Args:
            secret_name: Name of the secret to check.

        Returns:
            bool: True if the secret exists, False otherwise.
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

    def get_secret_arn(self, secret_name: str) -> Optional[str]:
        """
        Retrieve the ARN of a secret from AWS Secrets Manager.

        Args:
            secret_name: Name of the secret to retrieve the ARN for.

        Returns:
            Optional[str]:
                - str: ARN of the secret if found.
                - None: ARN of the secret not found.
        """
        try:
            response = self.secretsmanager.describe_secret(SecretId=secret_name)
            return response["ARN"]
        except self.secretsmanager.exceptions.ResourceNotFoundException:
            logger.warning(f"The requested secret {secret_name} was not found")
            return None
        except Exception as e:
            logger.error(f"An error occurred: {e}")
            return None

    def get_secret_details(
        self, secret_name: str, fetch_value: bool = False
    ) -> Dict[str, Any]:
        """
        Retrieve details of a secret from AWS Secrets Manager.
        Optionally fetch the secret value as well.

        Args:
            secret_name: Name of the secret.
            fetch_value (optional): Whether to also fetch the secret value (default is False).

        Returns:
            Dict[str, Any]: A dictionary with secret details (ARN and optionally the secret value) or an error dictionary if something went wrong.
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

    def create_secret(
        self, secret_name: str, secret_value: Dict[str, Any]
    ) -> Optional[str]:
        """
        Create a new secret in AWS Secrets Manager.

        Args:
            secret_name: Name of the secret to create.
            secret_value: Dictionary containing the secret data.

        Returns:
            Optional[str]:
                - str: ARN of created secret if successful
                - None: If creation fails
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
