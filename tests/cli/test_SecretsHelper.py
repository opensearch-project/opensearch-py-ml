# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.


import json
import logging
import unittest
from unittest.mock import MagicMock, patch

from botocore.exceptions import ClientError

from opensearch_py_ml.ml_commons.SecretsHelper import SecretHelper


class TestSecretHelper(unittest.TestCase):
    def setUp(self):
        self.region = "us-east-1"
        self.aws_access_key = "test-access-key"
        self.aws_secret_access_key = "test-secret-access-key"
        self.aws_session_token = "test-session-token"

    @classmethod
    def setUpClass(cls):
        # Suppress logging below ERROR level during tests
        logging.basicConfig(level=logging.ERROR)

    @patch("opensearch_py_ml.ml_commons.SecretsHelper.boto3.client")
    def test_create_secret_error_logging(self, mock_boto_client):
        mock_secretsmanager = MagicMock()
        mock_boto_client.return_value = mock_secretsmanager

        error_response = {
            "Error": {
                "Code": "InternalServiceError",
                "Message": "An unspecified error occurred",
            }
        }
        mock_secretsmanager.create_secret.side_effect = ClientError(
            error_response, "CreateSecret"
        )

        secret_helper = SecretHelper(
            region=self.region,
            aws_access_key=self.aws_access_key,
            aws_secret_access_key=self.aws_secret_access_key,
            aws_session_token=self.aws_session_token,
        )

        mock_boto_client.assert_called_once_with(
            "secretsmanager",
            region_name=self.region,
            aws_access_key_id=self.aws_access_key,
            aws_secret_access_key=self.aws_secret_access_key,
            aws_session_token=self.aws_session_token,
        )

        # Capture logs with a context manager
        with self.assertLogs(
            "opensearch_py_ml.ml_commons.SecretsHelper", level="ERROR"
        ) as cm:
            result = secret_helper.create_secret("new-secret", {"key": "value"})
            self.assertIsNone(result)
        # Confirm the error message was logged
        self.assertIn("Error creating secret 'new-secret'", cm.output[0])

    @patch("opensearch_py_ml.ml_commons.SecretsHelper.boto3.client")
    def test_create_secret_success(self, mock_boto_client):
        mock_secretsmanager = MagicMock()
        mock_boto_client.return_value = mock_secretsmanager

        mock_secretsmanager.create_secret.return_value = {
            "ARN": "arn:aws:secretsmanager:us-east-1:123456789012:secret:new-secret"
        }

        secret_helper = SecretHelper(
            region=self.region,
            aws_access_key=self.aws_access_key,
            aws_secret_access_key=self.aws_secret_access_key,
            aws_session_token=self.aws_session_token,
        )

        mock_boto_client.assert_called_once_with(
            "secretsmanager",
            region_name=self.region,
            aws_access_key_id=self.aws_access_key,
            aws_secret_access_key=self.aws_secret_access_key,
            aws_session_token=self.aws_session_token,
        )

        result = secret_helper.create_secret("new-secret", {"key": "value"})
        self.assertEqual(
            result, "arn:aws:secretsmanager:us-east-1:123456789012:secret:new-secret"
        )
        mock_secretsmanager.create_secret.assert_called_with(
            Name="new-secret", SecretString=json.dumps({"key": "value"})
        )

    @patch("opensearch_py_ml.ml_commons.SecretsHelper.boto3.client")
    def test_secret_exists_true(self, mock_boto_client):
        """Test that secret_exists returns True if secret is found."""
        mock_secretsmanager = MagicMock()
        mock_boto_client.return_value = mock_secretsmanager

        # If get_secret_value doesn't raise ResourceNotFoundException, assume secret exists
        mock_secretsmanager.get_secret_value.return_value = {
            "SecretString": "some-value"
        }

        secret_helper = SecretHelper(
            region=self.region,
            aws_access_key=self.aws_access_key,
            aws_secret_access_key=self.aws_secret_access_key,
            aws_session_token=self.aws_session_token,
        )

        mock_boto_client.assert_called_once_with(
            "secretsmanager",
            region_name=self.region,
            aws_access_key_id=self.aws_access_key,
            aws_secret_access_key=self.aws_secret_access_key,
            aws_session_token=self.aws_session_token,
        )

        exists = secret_helper.secret_exists("my-existing-secret")
        self.assertTrue(exists)
        mock_secretsmanager.get_secret_value.assert_called_with(
            SecretId="my-existing-secret"
        )

    @patch("opensearch_py_ml.ml_commons.SecretsHelper.boto3.client")
    def test_secret_exists_false(self, mock_boto_client):
        """Test that secret_exists returns False if secret is not found."""
        mock_secretsmanager = MagicMock()
        mock_boto_client.return_value = mock_secretsmanager

        error_response = {
            "Error": {
                "Code": "ResourceNotFoundException",
                "Message": "Secret not found",
            }
        }
        mock_secretsmanager.get_secret_value.side_effect = ClientError(
            error_response, "GetSecretValue"
        )

        secret_helper = SecretHelper(
            region=self.region,
            aws_access_key=self.aws_access_key,
            aws_secret_access_key=self.aws_secret_access_key,
            aws_session_token=self.aws_session_token,
        )

        mock_boto_client.assert_called_once_with(
            "secretsmanager",
            region_name=self.region,
            aws_access_key_id=self.aws_access_key,
            aws_secret_access_key=self.aws_secret_access_key,
            aws_session_token=self.aws_session_token,
        )

        exists = secret_helper.secret_exists("nonexistent-secret")
        self.assertFalse(exists)

    @patch("opensearch_py_ml.ml_commons.SecretsHelper.boto3.client")
    def test_secret_exists_other_error(self, mock_boto_client):
        """Test that secret_exists returns False on unexpected ClientError."""
        mock_secretsmanager = MagicMock()
        mock_boto_client.return_value = mock_secretsmanager

        error_response = {
            "Error": {
                "Code": "InternalServiceError",
                "Message": "An unspecified error occurred",
            }
        }
        mock_secretsmanager.get_secret_value.side_effect = ClientError(
            error_response, "GetSecretValue"
        )

        secret_helper = SecretHelper(
            region=self.region,
            aws_access_key=self.aws_access_key,
            aws_secret_access_key=self.aws_secret_access_key,
            aws_session_token=self.aws_session_token,
        )

        mock_boto_client.assert_called_once_with(
            "secretsmanager",
            region_name=self.region,
            aws_access_key_id=self.aws_access_key,
            aws_secret_access_key=self.aws_secret_access_key,
            aws_session_token=self.aws_session_token,
        )

        exists = secret_helper.secret_exists("problem-secret")
        self.assertFalse(exists)

    @patch("opensearch_py_ml.ml_commons.SecretsHelper.boto3.client")
    def test_get_secret_arn_success(self, mock_boto_client):
        """Test successful retrieval of secret ARN."""
        mock_secretsmanager = MagicMock()
        mock_boto_client.return_value = mock_secretsmanager

        mock_secretsmanager.describe_secret.return_value = {
            "ARN": "arn:aws:secretsmanager:us-east-1:123456789012:secret:my-secret"
        }

        secret_helper = SecretHelper(
            region=self.region,
            aws_access_key=self.aws_access_key,
            aws_secret_access_key=self.aws_secret_access_key,
            aws_session_token=self.aws_session_token,
        )

        mock_boto_client.assert_called_once_with(
            "secretsmanager",
            region_name=self.region,
            aws_access_key_id=self.aws_access_key,
            aws_secret_access_key=self.aws_secret_access_key,
            aws_session_token=self.aws_session_token,
        )

        arn = secret_helper.get_secret_arn("my-secret")
        self.assertEqual(
            arn, "arn:aws:secretsmanager:us-east-1:123456789012:secret:my-secret"
        )
        mock_secretsmanager.describe_secret.assert_called_with(SecretId="my-secret")

    @patch("opensearch_py_ml.ml_commons.SecretsHelper.boto3.client")
    def test_get_secret_arn_not_found(self, mock_boto_client):
        """Test get_secret_arn returns None when secret is not found."""
        mock_secretsmanager = MagicMock()
        mock_boto_client.return_value = mock_secretsmanager

        mock_secretsmanager.exceptions.ResourceNotFoundException = ClientError
        mock_secretsmanager.describe_secret.side_effect = ClientError(
            {
                "Error": {
                    "Code": "ResourceNotFoundException",
                    "Message": "Secret not found",
                }
            },
            "DescribeSecret",
        )

        secret_helper = SecretHelper(
            region=self.region,
            aws_access_key=self.aws_access_key,
            aws_secret_access_key=self.aws_secret_access_key,
            aws_session_token=self.aws_session_token,
        )

        mock_boto_client.assert_called_once_with(
            "secretsmanager",
            region_name=self.region,
            aws_access_key_id=self.aws_access_key,
            aws_secret_access_key=self.aws_secret_access_key,
            aws_session_token=self.aws_session_token,
        )

        arn = secret_helper.get_secret_arn("non-existent-secret")
        self.assertIsNone(arn)
        mock_secretsmanager.describe_secret.assert_called_with(
            SecretId="non-existent-secret"
        )

    @patch("opensearch_py_ml.ml_commons.SecretsHelper.boto3.client")
    def test_get_secret_details_arn_only_success(self, mock_boto_client):
        """Test get_secret_details returns ARN if fetch_value=False."""
        mock_secretsmanager = MagicMock()
        mock_boto_client.return_value = mock_secretsmanager

        mock_secretsmanager.describe_secret.return_value = {
            "ARN": "arn:aws:secretsmanager:us-east-1:123456789012:secret:my-secret"
        }

        secret_helper = SecretHelper(
            region=self.region,
            aws_access_key=self.aws_access_key,
            aws_secret_access_key=self.aws_secret_access_key,
            aws_session_token=self.aws_session_token,
        )

        mock_boto_client.assert_called_once_with(
            "secretsmanager",
            region_name=self.region,
            aws_access_key_id=self.aws_access_key,
            aws_secret_access_key=self.aws_secret_access_key,
            aws_session_token=self.aws_session_token,
        )

        details = secret_helper.get_secret_details("my-secret", fetch_value=False)
        self.assertIn("ARN", details)
        self.assertEqual(
            details["ARN"],
            "arn:aws:secretsmanager:us-east-1:123456789012:secret:my-secret",
        )
        self.assertNotIn("SecretValue", details)
        mock_secretsmanager.describe_secret.assert_called_with(SecretId="my-secret")

    @patch("opensearch_py_ml.ml_commons.SecretsHelper.boto3.client")
    def test_get_secret_details_with_value_success(self, mock_boto_client):
        """Test get_secret_details returns ARN and SecretValue if fetch_value=True."""
        mock_secretsmanager = MagicMock()
        mock_boto_client.return_value = mock_secretsmanager

        mock_secretsmanager.describe_secret.return_value = {
            "ARN": "arn:aws:secretsmanager:us-east-1:123456789012:secret:my-secret"
        }
        mock_secretsmanager.get_secret_value.return_value = {
            "SecretString": "my-secret-value"
        }

        secret_helper = SecretHelper(
            region=self.region,
            aws_access_key=self.aws_access_key,
            aws_secret_access_key=self.aws_secret_access_key,
            aws_session_token=self.aws_session_token,
        )

        mock_boto_client.assert_called_once_with(
            "secretsmanager",
            region_name=self.region,
            aws_access_key_id=self.aws_access_key,
            aws_secret_access_key=self.aws_secret_access_key,
            aws_session_token=self.aws_session_token,
        )

        details = secret_helper.get_secret_details("my-secret", fetch_value=True)
        self.assertIn("ARN", details)
        self.assertIn("SecretValue", details)
        self.assertEqual(details["SecretValue"], "my-secret-value")
        mock_secretsmanager.describe_secret.assert_called_with(SecretId="my-secret")
        mock_secretsmanager.get_secret_value.assert_called_with(SecretId="my-secret")

    @patch("opensearch_py_ml.ml_commons.SecretsHelper.boto3.client")
    def test_get_secret_details_not_found(self, mock_boto_client):
        """Test get_secret_details returns an error dict if secret is not found."""
        mock_secretsmanager = MagicMock()
        mock_boto_client.return_value = mock_secretsmanager

        error_response = {
            "Error": {
                "Code": "ResourceNotFoundException",
                "Message": "Secret not found",
            }
        }
        mock_secretsmanager.describe_secret.side_effect = ClientError(
            error_response, "DescribeSecret"
        )

        secret_helper = SecretHelper(
            region=self.region,
            aws_access_key=self.aws_access_key,
            aws_secret_access_key=self.aws_secret_access_key,
            aws_session_token=self.aws_session_token,
        )

        mock_boto_client.assert_called_once_with(
            "secretsmanager",
            region_name=self.region,
            aws_access_key_id=self.aws_access_key,
            aws_secret_access_key=self.aws_secret_access_key,
            aws_session_token=self.aws_session_token,
        )

        details = secret_helper.get_secret_details(
            "nonexistent-secret", fetch_value=True
        )
        self.assertIn("error", details)
        self.assertEqual(details["error_code"], "ResourceNotFoundException")
        mock_secretsmanager.describe_secret.assert_called_with(
            SecretId="nonexistent-secret"
        )


if __name__ == "__main__":
    unittest.main()
