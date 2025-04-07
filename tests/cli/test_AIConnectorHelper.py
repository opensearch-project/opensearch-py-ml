# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

import json
import unittest
from io import StringIO
from unittest.mock import MagicMock, call, patch
from urllib.parse import urlparse

from colorama import Fore, Style
from opensearchpy import RequestsHttpConnection
from requests.auth import HTTPBasicAuth

from opensearch_py_ml.ml_commons.cli.AIConnectorHelper import AIConnectorHelper
from opensearch_py_ml.ml_commons.cli.aws_config import AWSConfig
from opensearch_py_ml.ml_commons.cli.opensearch_domain_config import (
    OpenSearchDomainConfig,
)


class TestAIConnectorHelper(unittest.TestCase):
    def setUp(self):
        # Create OpenSearchDomainConfig
        self.opensearch_config = OpenSearchDomainConfig(
            opensearch_domain_region="us-east-1",
            opensearch_domain_name="test-domain",
            opensearch_domain_username="admin",
            opensearch_domain_password="password",
            opensearch_domain_endpoint="test-domain-url",
        )
        # Create AWSConfig
        self.aws_config = AWSConfig(
            aws_user_name="test-user",
            aws_role_name="test-role",
            aws_access_key="test-access-key",
            aws_secret_access_key="test-secret-access-key",
            aws_session_token="test-session-token",
        )
        self.service_type = "amazon-opensearch-service"
        self.domain_arn = "arn:aws:es:us-east-1:123456789012:domain/test-domain"
        self.test_data = {
            "secret_name": "test-secret",
            "secret_value": "test-secret-value",
            "connector_role_name": "test-connector-role",
            "create_connector_role_name": "test-create-connector-role",
            "create_connector_input": {"test": "payload"},
            "secret_arn": "arn:aws:secretsmanager:region:account:secret:test-secret",
            "connector_role_arn": "arn:aws:iam::account:role/test-connector-role",
            "create_connector_role_arn": "arn:aws:iam::account:role/test-create-connector-role",
            "connector_id": "test-connector-id",
            "connector_role_inline_policy": {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": "iam:PassRole",
                        "Resource": "arn:aws:iam::account:role/test-connector-role",
                    }
                ],
            },
        }

    @patch(
        "opensearch_py_ml.ml_commons.cli.AIConnectorHelper.AIConnectorHelper.get_opensearch_domain_info"
    )
    @patch("opensearch_py_ml.ml_commons.cli.AIConnectorHelper.OpenSearch")
    @patch("opensearch_py_ml.ml_commons.cli.AIConnectorHelper.SecretHelper")
    @patch("opensearch_py_ml.ml_commons.cli.AIConnectorHelper.IAMRoleHelper")
    def test___init__(
        self,
        mock_iam_role_helper,
        mock_secret_helper,
        mock_opensearch,
        mock_get_opensearch_domain_info,
    ):
        # Mock get_opensearch_domain_info
        mock_get_opensearch_domain_info.return_value = (
            self.opensearch_config.opensearch_domain_endpoint,
            self.domain_arn,
        )

        # Parse the URL
        parsed_url = urlparse(self.opensearch_config.opensearch_domain_endpoint)
        expected_host = parsed_url.hostname
        expected_port = parsed_url.port or (
            443 if parsed_url.scheme == "https" else 9200
        )
        expected_use_ssl = parsed_url.scheme == "https"

        # Instantiate AIConnectorHelper
        helper = AIConnectorHelper(
            self.service_type, self.opensearch_config, self.aws_config
        )

        # Assert basic attributes
        self.assertEqual(helper.service_type, self.service_type)
        self.assertEqual(
            helper.opensearch_config.opensearch_domain_endpoint,
            self.opensearch_config.opensearch_domain_endpoint,
        )
        self.assertEqual(helper.opensearch_domain_arn, self.domain_arn)

        # Assert OpenSearch client initialization
        mock_opensearch.assert_called_once_with(
            hosts=[{"host": expected_host, "port": expected_port}],
            http_auth=(
                self.opensearch_config.opensearch_domain_username,
                self.opensearch_config.opensearch_domain_password,
            ),
            use_ssl=expected_use_ssl,
            verify_certs=True,
            connection_class=RequestsHttpConnection,
        )

        # Assert helper initializations based on service_type
        if self.service_type == "open-source":
            self.assertIsNone(helper.iam_helper)
            self.assertIsNone(helper.secret_helper)
        else:
            # Assert IAMRoleHelper initialization
            mock_iam_role_helper.assert_called_once_with(
                opensearch_config=self.opensearch_config, aws_config=self.aws_config
            )

            # Assert SecretHelper initialization
            mock_secret_helper.assert_called_once_with(
                opensearch_config=self.opensearch_config, aws_config=self.aws_config
            )

    def test_open_source_service_type(self):
        """Test when service_type is open-source"""
        # Initialize helper with open-source service type
        helper = AIConnectorHelper(
            service_type="open-source",
            opensearch_config=OpenSearchDomainConfig(
                opensearch_domain_region="",
                opensearch_domain_name=self.opensearch_config.opensearch_domain_name,
                opensearch_domain_username=self.opensearch_config.opensearch_domain_username,
                opensearch_domain_password=self.opensearch_config.opensearch_domain_password,
                opensearch_domain_endpoint="https://localhost:9200",
            ),
            aws_config=AWSConfig(
                aws_user_name="",
                aws_role_name="",
                aws_access_key="",
                aws_secret_access_key="",
                aws_session_token="",
            ),
        )
        # Assert service_type is set correctly
        self.assertEqual(helper.service_type, "open-source")
        # Assert domain_arn, iam_helper, and secret_helper is None
        self.assertIsNone(helper.opensearch_domain_arn)
        self.assertIsNone(helper.iam_helper)
        self.assertIsNone(helper.secret_helper)

    @patch("boto3.client")
    def test_get_opensearch_domain_info_success(self, mock_boto3_client):
        """Test get_opensearch_domain_info successful"""
        # Mock the boto3 client
        mock_client_instance = MagicMock()
        mock_boto3_client.return_value = mock_client_instance

        # Mock the describe_domain response
        mock_client_instance.describe_domain.return_value = {
            "DomainStatus": {
                "Endpoint": self.opensearch_config.opensearch_domain_endpoint,
                "ARN": self.domain_arn,
            }
        }

        # Call the method
        endpoint, arn = AIConnectorHelper.get_opensearch_domain_info(
            self.opensearch_config.opensearch_domain_region,
            self.opensearch_config.opensearch_domain_name,
            self.aws_config.aws_access_key,
            self.aws_config.aws_secret_access_key,
            self.aws_config.aws_session_token,
        )

        # Assert the results
        self.assertEqual(endpoint, self.opensearch_config.opensearch_domain_endpoint)
        self.assertEqual(arn, self.domain_arn)
        mock_client_instance.describe_domain.assert_called_once_with(
            DomainName=self.opensearch_config.opensearch_domain_name
        )

    @patch("boto3.Session")
    @patch("sys.stdout", new_callable=StringIO)
    def test_get_opensearch_domain_info_no_credentials(self, mock_stdout, mock_session):
        """Test get_opensearch_domain_info when no valid credentials are provided"""
        # Mock the boto3 client to raise an exception
        mock_session_instance = MagicMock()
        mock_session.return_value = mock_session_instance
        mock_session_instance.get_credentials.return_value = None

        # Call the method with empty credentials
        endpoint, arn = AIConnectorHelper.get_opensearch_domain_info(
            self.opensearch_config.opensearch_domain_region,
            self.opensearch_config.opensearch_domain_name,
            "",  # empty access key
            "",  # empty secret key
            "",  # empty session token
        )

        # Assert the results
        self.assertIsNone(endpoint)
        self.assertIsNone(arn)

        # Assert the error message was printed
        expected_output = f"{Fore.RED}No valid credentials found.{Style.RESET_ALL}\n"
        self.assertEqual(mock_stdout.getvalue(), expected_output)

        # Verify session was created with empty credentials
        mock_session.assert_called_once_with(
            aws_access_key_id="", aws_secret_access_key="", aws_session_token=""
        )

        # Verify get_credentials was called
        mock_session_instance.get_credentials.assert_called_once()

    @patch("boto3.client")
    def test_get_opensearch_domain_info_exception(self, mock_boto3_client):
        """Test get_opensearch_domain_info exception handling"""
        # Mock the boto3 client to raise an exception
        mock_client_instance = MagicMock()
        mock_boto3_client.return_value = mock_client_instance
        mock_client_instance.describe_domain.side_effect = Exception("Test Exception")

        # Call the method
        endpoint, arn = AIConnectorHelper.get_opensearch_domain_info(
            self.opensearch_config.opensearch_domain_region,
            self.opensearch_config.opensearch_domain_name,
            self.aws_config.aws_access_key,
            self.aws_config.aws_secret_access_key,
            self.aws_config.aws_session_token,
        )

        # Assert the results are None
        self.assertIsNone(endpoint)
        self.assertIsNone(arn)

    @patch.object(AIConnectorHelper, "iam_helper", create=True)
    def test_get_ml_auth_success(self, mock_iam_helper):
        """Test get_ml_auth successful"""
        # Mock the get_role_arn to return a role ARN
        create_connector_role_name = "test-create-connector-role"
        create_connector_role_arn = (
            "arn:aws:iam::123456789012:role/test-create-connector-role"
        )
        mock_iam_helper.get_role_arn.return_value = create_connector_role_arn

        # Mock the assume_role to return temp credentials
        temp_credentials = {
            "AccessKeyId": "test-access-key",
            "SecretAccessKey": "test-secret-key",
            "SessionToken": "test-session-token",
        }
        mock_iam_helper.assume_role.return_value = temp_credentials

        # Instantiate helper
        with patch.object(AIConnectorHelper, "__init__", return_value=None):
            helper = AIConnectorHelper()
            helper.opensearch_config = self.opensearch_config
            helper.iam_helper = mock_iam_helper
            helper.opensearch_domain_arn = self.domain_arn

            # Call the method
            awsauth = helper.get_ml_auth(create_connector_role_name)

            # Assert that the IAM helper methods were called
            mock_iam_helper.get_role_arn.assert_called_with(create_connector_role_name)
            mock_iam_helper.assume_role.assert_called_with(create_connector_role_arn)

            # Since AWS4Auth is instantiated within the method, we can check if awsauth is not None
            self.assertIsNotNone(awsauth)

    @patch.object(AIConnectorHelper, "iam_helper", create=True)
    def test_get_ml_auth_role_not_found(self, mock_iam_helper):
        """Test get_ml_auth_role when role is not found"""
        # Mock the get_role_arn to return None
        create_connector_role_name = "test-create-connector-role"
        mock_iam_helper.get_role_arn.return_value = None

        # Instantiate helper
        with patch.object(AIConnectorHelper, "__init__", return_value=None):
            helper = AIConnectorHelper()
            helper.iam_helper = mock_iam_helper

            # Call the method and expect an exception
            with self.assertRaises(Exception) as context:
                helper.get_ml_auth(create_connector_role_name)

            self.assertTrue(
                f"IAM role '{create_connector_role_name}' not found."
                in str(context.exception)
            )

    @patch("opensearch_py_ml.ml_commons.cli.AIConnectorHelper.OpenSearch")
    @patch("opensearch_py_ml.ml_commons.cli.AIConnectorHelper.AWS4Auth")
    @patch.object(AIConnectorHelper, "iam_helper", create=True)
    def test_create_connector_managed(
        self, mock_iam_helper, mock_aws4auth, mock_opensearch
    ):
        """Test create_connector in managed service"""
        # Mock the IAM helper methods
        create_connector_role_name = "test-create-connector-role"
        create_connector_role_arn = (
            "arn:aws:iam::123456789012:role/test-create-connector-role"
        )
        mock_iam_helper.get_role_arn.return_value = create_connector_role_arn
        temp_credentials = {
            "credentials": {
                "AccessKeyId": "test-access-key",
                "SecretAccessKey": "test-secret-key",
                "SessionToken": "test-session-token",
            }
        }
        mock_iam_helper.assume_role.return_value = temp_credentials

        # Mock AWS4Auth
        mock_awsauth = MagicMock()
        mock_aws4auth.return_value = mock_awsauth

        # Mock OpenSearch client
        mock_os_client = MagicMock()
        mock_opensearch.return_value = mock_os_client
        mock_os_client.transport.perform_request.return_value = (
            200,
            {},
            '{"connector_id": "test-connector-id"}',
        )

        # Instantiate helper
        with patch.object(AIConnectorHelper, "__init__", return_value=None):
            helper = AIConnectorHelper()
            helper.service_type = self.service_type
            helper.opensearch_config = self.opensearch_config
            helper.iam_helper = mock_iam_helper

            # Mock the Connector class
            mock_connector = MagicMock()
            mock_connector.create_standalone_connector.return_value = {
                "connector_id": "test-connector-id"
            }
            with patch(
                "opensearch_py_ml.ml_commons.cli.AIConnectorHelper.Connector",
                return_value=mock_connector,
            ):
                # Call the method
                payload = {"key": "value"}
                connector_id = helper.create_connector(
                    create_connector_role_name, payload
                )

            # Assert that create_standalone_connector was called with the correct arguments
            mock_connector.create_standalone_connector.assert_called_once_with(payload)

            # Assert that the connector_id is returned
            self.assertEqual(connector_id, "test-connector-id")

    @patch("opensearch_py_ml.ml_commons.cli.AIConnectorHelper.OpenSearch")
    def test_create_connector_open_source(self, mock_opensearch):
        """Test create_connector in open-source service"""
        create_connector_role_name = None

        # Instantiate helper
        with patch.object(AIConnectorHelper, "__init__", return_value=None):
            helper = AIConnectorHelper()
            helper.service_type = "open-source"
            helper.opensearch_config = self.opensearch_config
            helper.opensearch_config.opensearch_domain_endpoint = (
                "https://localhost:9200"
            )

            # Mock the Connector class
            mock_connector = MagicMock()
            mock_connector.create_standalone_connector.return_value = {
                "connector_id": "test-connector-id"
            }

            with patch(
                "opensearch_py_ml.ml_commons.cli.AIConnectorHelper.Connector",
                return_value=mock_connector,
            ):
                # Call the method
                payload = {"key": "value"}
                connector_id = helper.create_connector(
                    create_connector_role_name, payload
                )

            # Verify OpenSearch client was created with correct parameters
            mock_opensearch.assert_called_once_with(
                hosts=[{"host": "localhost", "port": 9200}],
                http_auth=("admin", "password"),
                use_ssl=True,
                verify_certs=True,
                connection_class=RequestsHttpConnection,
            )

            # Assert that create_standalone_connector was called with the correct arguments
            mock_connector.create_standalone_connector.assert_called_once_with(payload)

            # Assert that the connector_id is returned
            self.assertEqual(connector_id, "test-connector-id")

    @patch("sys.stdout", new_callable=StringIO)
    def test_get_task(self, mock_stdout):
        """Test get_task with successful response"""
        # Mock task response
        mock_task_response = {
            "task_id": "test-task-id",
            "status": "COMPLETED",
            "task_type": "test-type",
        }

        # Mock MLCommonClient
        mock_ml_commons_client = MagicMock()
        mock_ml_commons_client.get_task_info.return_value = mock_task_response

        # Create helper instance
        with patch.object(AIConnectorHelper, "__init__", return_value=None):
            helper = AIConnectorHelper()
            helper.ml_commons_client = mock_ml_commons_client

            # Call get_task
            response = helper.get_task("test-task-id")

            # Verify response
            self.assertEqual(response, mock_task_response)

            # Verify get_task_info was called with correct parameters
            mock_ml_commons_client.get_task_info.assert_called_once_with(
                "test-task-id", False
            )

            # Verify printed output
            expected_output = f"Get Task Response: {json.dumps(mock_task_response)}\n"
            self.assertEqual(mock_stdout.getvalue(), expected_output)

    def test_get_task_exception(self):
        """Test get_task with exception"""
        # Mock MLCommonClient
        mock_ml_commons_client = MagicMock()
        mock_ml_commons_client.get_task_info.side_effect = Exception("Test Exception")

        # Create helper instance
        with patch.object(AIConnectorHelper, "__init__", return_value=None):
            helper = AIConnectorHelper()
            helper.ml_commons_client = mock_ml_commons_client

            # Call get_task and expect exception
            with self.assertRaises(Exception) as context:
                helper.get_task("test-task-id")

            self.assertTrue("Test Exception" in str(context.exception))

    @patch("requests.post")
    @patch.object(AIConnectorHelper, "get_ml_auth")
    @patch.object(AIConnectorHelper, "get_task")
    def test_register_model_direct_response(
        self, mock_get_task, mock_get_ml_auth, mock_requests_post
    ):
        """Test register_model when model_id is directly in the response"""
        # Mock get_ml_auth
        mock_awsauth = MagicMock()
        mock_get_ml_auth.return_value = mock_awsauth

        # Mock requests.post with task_id response
        mock_response = MagicMock()
        mock_response.text = json.dumps({"task_id": "test-task-id"})
        mock_requests_post.return_value = mock_response

        # Mock get_task
        mock_get_task.return_value = {"model_id": "task-model-id"}

        # Instantiate helper
        with patch.object(AIConnectorHelper, "__init__", return_value=None):
            helper = AIConnectorHelper()
            helper.opensearch_config = self.opensearch_config
            helper.model_access_control = MagicMock()

            # Call the method
            model_id = helper.register_model(
                "test-model",
                "test description",
                "test-connector-id",
                deploy=True,
            )

            # Assert correct URL
            expected_url = f"{helper.opensearch_config.opensearch_domain_endpoint}/_plugins/_ml/models/_register?deploy=true"
            mock_requests_post.assert_called_once_with(
                expected_url,
                auth=HTTPBasicAuth(
                    helper.opensearch_config.opensearch_domain_username,
                    helper.opensearch_config.opensearch_domain_password,
                ),
                json={
                    "name": "test-model",
                    "function_name": "remote",
                    "description": "test description",
                    "connector_id": "test-connector-id",
                },
                headers={"Content-Type": "application/json"},
            )

            # Verify get_task was called with correct parameters
            mock_get_task.assert_called_once_with(
                "test-task-id", wait_until_task_done=True
            )

            # Assert that model_id is returned
            self.assertEqual(model_id, "task-model-id")

    @patch("requests.post")
    @patch.object(AIConnectorHelper, "get_ml_auth")
    @patch.object(AIConnectorHelper, "get_task")
    def test_register_model_task_response(
        self, mock_get_task, mock_get_ml_auth, mock_requests_post
    ):
        """Test register_model when model_id comes from task response"""
        # Mock get_ml_auth
        mock_awsauth = MagicMock()
        mock_get_ml_auth.return_value = mock_awsauth

        # Mock requests.post
        mock_response = MagicMock()
        mock_response.text = json.dumps({"task_id": "test-task-id"})
        mock_requests_post.return_value = mock_response

        # Mock get_task
        mock_get_task.return_value = {"model_id": "test-model-id"}

        # Instantiate helper
        with patch.object(AIConnectorHelper, "__init__", return_value=None):
            helper = AIConnectorHelper()
            helper.opensearch_config = self.opensearch_config
            helper.model_access_control = MagicMock()

            # Call the method
            model_id = helper.register_model(
                "test-model",
                "test description",
                "test-connector-id",
                deploy=True,
            )

            # Assert correct URL
            expected_url = f"{helper.opensearch_config.opensearch_domain_endpoint}/_plugins/_ml/models/_register?deploy=true"
            mock_requests_post.assert_called_once_with(
                expected_url,
                auth=HTTPBasicAuth(
                    helper.opensearch_config.opensearch_domain_username,
                    helper.opensearch_config.opensearch_domain_password,
                ),
                json={
                    "name": "test-model",
                    "function_name": "remote",
                    "description": "test description",
                    "connector_id": "test-connector-id",
                },
                headers={"Content-Type": "application/json"},
            )
            mock_get_task.assert_called_once_with(
                "test-task-id", wait_until_task_done=True
            )

            # Assert that model_id is returned
            self.assertEqual(model_id, "test-model-id")

    @patch("requests.post")
    @patch.object(AIConnectorHelper, "get_ml_auth")
    @patch.object(AIConnectorHelper, "get_task")
    def test_register_model_no_model_id(
        self, mock_get_task, mock_get_ml_auth, mock_requests_post
    ):
        """Test register_model when no model_id is returned from task response"""
        # Mock get_ml_auth
        mock_awsauth = MagicMock()
        mock_get_ml_auth.return_value = mock_awsauth

        # Mock requests.post with task_id response
        mock_response = MagicMock()
        mock_response.text = json.dumps({"task_id": "test-task-id"})
        mock_requests_post.return_value = mock_response

        # Mock get_task with response missing model_id
        mock_get_task.return_value = {"status": "COMPLETED"}

        # Instantiate helper
        with patch.object(AIConnectorHelper, "__init__", return_value=None):
            helper = AIConnectorHelper()
            helper.opensearch_config = self.opensearch_config
            helper.model_access_control = MagicMock()

            # Verify KeyError is raised when no model_id is found
            with self.assertRaises(KeyError) as context:
                helper.register_model(
                    "test-model",
                    "test description",
                    "test-connector-id",
                    deploy=True,
                )

            # Verify error message
            self.assertIn(
                "'model_id' not found in task response", str(context.exception)
            )

    @patch("requests.post")
    @patch.object(AIConnectorHelper, "get_ml_auth")
    @patch.object(AIConnectorHelper, "get_task")
    def test_register_model_error_response(
        self, mock_get_task, mock_get_ml_auth, mock_requests_post
    ):
        """Test register_model with error response"""
        # Mock get_ml_auth
        mock_awsauth = MagicMock()
        mock_get_ml_auth.return_value = mock_awsauth

        # Mock requests.post with error response
        mock_response = MagicMock()
        error_message = "Invalid model configuration"
        mock_response.text = json.dumps({"error": error_message})
        mock_requests_post.return_value = mock_response

        # Instantiate helper
        with patch.object(AIConnectorHelper, "__init__", return_value=None):
            helper = AIConnectorHelper()
            helper.opensearch_config = self.opensearch_config
            helper.model_access_control = MagicMock()

            # Verify that the correct exception is raised with the error message
            with self.assertRaises(Exception) as context:
                helper.register_model(
                    "test-model",
                    "test description",
                    "test-connector-id",
                    deploy=True,
                )

            # Verify the error message
            self.assertEqual(
                str(context.exception), f"Error registering model: {error_message}"
            )

            # Assert correct URL and parameters were used
            expected_url = f"{helper.opensearch_config.opensearch_domain_endpoint}/_plugins/_ml/models/_register?deploy=true"
            mock_requests_post.assert_called_once_with(
                expected_url,
                auth=HTTPBasicAuth(
                    helper.opensearch_config.opensearch_domain_username,
                    helper.opensearch_config.opensearch_domain_password,
                ),
                json={
                    "name": "test-model",
                    "function_name": "remote",
                    "description": "test description",
                    "connector_id": "test-connector-id",
                },
                headers={"Content-Type": "application/json"},
            )

            # Verify get_task was not called
            mock_get_task.assert_not_called()

    @patch("requests.post")
    @patch.object(AIConnectorHelper, "get_ml_auth")
    @patch.object(AIConnectorHelper, "get_task")
    def test_register_model_key_error(
        self, mock_get_task, mock_get_ml_auth, mock_requests_post
    ):
        """Test register_model when response contains neither model_id nor task_id"""
        # Mock get_ml_auth
        mock_awsauth = MagicMock()
        mock_get_ml_auth.return_value = mock_awsauth

        # Mock requests.post with response missing both model_id and task_id
        mock_response = MagicMock()
        response_data = {"status": "success", "message": "Operation completed"}
        mock_response.text = json.dumps(response_data)
        mock_requests_post.return_value = mock_response

        # Instantiate helper
        with patch.object(AIConnectorHelper, "__init__", return_value=None):
            helper = AIConnectorHelper()
            helper.opensearch_config = self.opensearch_config
            helper.model_access_control = MagicMock()

            # Verify that KeyError is raised with correct message
            with self.assertRaises(KeyError) as context:
                helper.register_model(
                    "test-model",
                    "test description",
                    "test-connector-id",
                    deploy=True,
                )
            error_message = str(context.exception).strip('"')

            # Verify the error message
            expected_error_message = (
                f"The response does not contain 'model_id' or 'task_id'. "
                f"Response content: {response_data}"
            )
            self.assertEqual(error_message, expected_error_message)

            # Assert correct URL and parameters were used
            expected_url = f"{helper.opensearch_config.opensearch_domain_endpoint}/_plugins/_ml/models/_register?deploy=true"
            mock_requests_post.assert_called_once_with(
                expected_url,
                auth=HTTPBasicAuth(
                    helper.opensearch_config.opensearch_domain_username,
                    helper.opensearch_config.opensearch_domain_password,
                ),
                json={
                    "name": "test-model",
                    "function_name": "remote",
                    "description": "test description",
                    "connector_id": "test-connector-id",
                },
                headers={"Content-Type": "application/json"},
            )

            # Verify get_task was not called
            mock_get_task.assert_not_called()

    @patch("sys.stdout", new_callable=StringIO)
    @patch("requests.post")
    @patch.object(AIConnectorHelper, "get_ml_auth")
    def test_register_model_exception_handling(
        self, mock_get_ml_auth, mock_requests_post, mock_stdout
    ):
        """Test register_model exception handling and error output"""
        # Mock get_ml_auth
        mock_awsauth = MagicMock()
        mock_get_ml_auth.return_value = mock_awsauth

        # Mock requests.post to raise an exception
        test_error = Exception("Test error message")
        mock_requests_post.side_effect = test_error

        # Instantiate helper
        with patch.object(AIConnectorHelper, "__init__", return_value=None):
            helper = AIConnectorHelper()
            helper.opensearch_config = self.opensearch_config
            helper.model_access_control = MagicMock()

            # Verify that the original exception is re-raised
            with self.assertRaises(Exception) as context:
                helper.register_model(
                    "test-model",
                    "test description",
                    "test-connector-id",
                    deploy=True,
                )

            # Verify the original exception is preserved
            self.assertEqual(str(context.exception), "Test error message")

            # Verify the error message was printed
            expected_output = f"{Fore.RED}Error registering model: Test error message{Style.RESET_ALL}\n"
            self.assertEqual(mock_stdout.getvalue(), expected_output)

    @patch("requests.post")
    def test_deploy_model(self, mock_requests_post):
        """Test deploy_model successful"""
        # Mock requests.post
        response = MagicMock()
        response.text = "Deploy model response"
        mock_requests_post.return_value = response

        # Instantiate helper
        with patch.object(AIConnectorHelper, "__init__", return_value=None):
            helper = AIConnectorHelper()
            helper.opensearch_config = self.opensearch_config

            # Call the method
            result = helper.deploy_model("test-model-id")

            # Assert that the method was called once
            mock_requests_post.assert_called_once()

            # Extract call arguments
            args, kwargs = mock_requests_post.call_args

            # Assert URL
            expected_url = f"{helper.opensearch_config.opensearch_domain_endpoint}/_plugins/_ml/models/test-model-id/_deploy"
            self.assertEqual(args[0], expected_url)

            # Assert headers
            self.assertEqual(kwargs["headers"], {"Content-Type": "application/json"})

            # Assert auth
            self.assertIsInstance(kwargs["auth"], HTTPBasicAuth)
            self.assertEqual(
                kwargs["auth"].username,
                self.opensearch_config.opensearch_domain_username,
            )
            self.assertEqual(
                kwargs["auth"].password,
                self.opensearch_config.opensearch_domain_password,
            )

            # Assert that the response is returned
            self.assertEqual(result, response)

    @patch("requests.post")
    def test_predict(self, mock_requests_post):
        """Test predict successful"""
        # Mock requests.post
        response = MagicMock()
        response.text = "Predict response"
        mock_requests_post.return_value = response

        # Mock the json method to return a dictionary with inference_results
        mock_json = {
            "inference_results": [
                {"status_code": 200}  # or whatever status code you expect
            ]
        }
        response.json.return_value = mock_json

        # Instantiate helper
        with patch.object(AIConnectorHelper, "__init__", return_value=None):
            helper = AIConnectorHelper()
            helper.opensearch_config = self.opensearch_config

            # Call the method
            payload = {"input": "test input"}
            result = helper.predict("test-model-id", payload)

            # Assert that the method was called once
            mock_requests_post.assert_called_once()

            # Extract call arguments
            args, kwargs = mock_requests_post.call_args

            # Assert URL
            expected_url = f"{helper.opensearch_config.opensearch_domain_endpoint}/_plugins/_ml/models/test-model-id/_predict"
            self.assertEqual(args[0], expected_url)

            # Assert JSON payload
            self.assertEqual(kwargs["json"], payload)

            # Assert headers
            self.assertEqual(kwargs["headers"], {"Content-Type": "application/json"})

            # Assert auth
            self.assertIsInstance(kwargs["auth"], HTTPBasicAuth)
            self.assertEqual(
                kwargs["auth"].username,
                self.opensearch_config.opensearch_domain_username,
            )
            self.assertEqual(
                kwargs["auth"].password,
                self.opensearch_config.opensearch_domain_password,
            )

            # Assert that the response is returned
            expected_status = mock_json["inference_results"][0]["status_code"]
            expected_result = (response.text, expected_status)
            self.assertEqual(result, expected_result)

    @patch("requests.get")
    def test_get_connector(self, mock_requests_get):
        """Test get_connector successful"""
        # Mock requests.get
        response = MagicMock()
        response.text = "Get connector response"
        mock_requests_get.return_value = response

        # Mock the json method to return a dictionary with connector details
        mock_json = {
            "connector_id": "test-connector-id",
            "name": "test-connector",
            "description": "test description",
            "version": "1.0",
        }
        response.json.return_value = mock_json

        # Instantiate helper
        with patch.object(AIConnectorHelper, "__init__", return_value=None):
            helper = AIConnectorHelper()
            helper.opensearch_config = self.opensearch_config

            # Call the method
            result = helper.get_connector("test-connector-id")

            # Assert that the method was called once
            mock_requests_get.assert_called_once()

            # Extract call arguments
            args, kwargs = mock_requests_get.call_args

            # Assert URL
            expected_url = f"{helper.opensearch_config.opensearch_domain_endpoint}/_plugins/_ml/connectors/test-connector-id"
            self.assertEqual(args[0], expected_url)

            # Assert headers
            self.assertEqual(kwargs["headers"], {"Content-Type": "application/json"})

            # Assert auth
            self.assertIsInstance(kwargs["auth"], HTTPBasicAuth)
            self.assertEqual(
                kwargs["auth"].username,
                self.opensearch_config.opensearch_domain_username,
            )
            self.assertEqual(
                kwargs["auth"].password,
                self.opensearch_config.opensearch_domain_password,
            )

            # Assert that the response is returned
            self.assertEqual(result, response.text)

    @patch("opensearch_py_ml.ml_commons.cli.AIConnectorHelper.OpenSearch")
    @patch("opensearch_py_ml.ml_commons.cli.AIConnectorHelper.AWS4Auth")
    @patch.object(AIConnectorHelper, "iam_helper", create=True)
    @patch.object(AIConnectorHelper, "secret_helper", create=True)
    def test_create_connector_with_secret(
        self, mock_secret_helper, mock_iam_helper, mock_aws4auth, mock_opensearch
    ):
        """Test create_connector_with_secret method"""
        # Mock secret_helper methods
        mock_secret_helper.secret_exists.return_value = False
        mock_secret_helper.create_secret.return_value = self.test_data["secret_arn"]
        mock_secret_helper.get_secret_arn.return_value = self.test_data["secret_arn"]

        # Mock iam_helper methods
        mock_iam_helper.role_exists.return_value = False
        mock_iam_helper.create_iam_role.side_effect = [
            self.test_data["connector_role_arn"],
            self.test_data["create_connector_role_arn"],
        ]
        mock_iam_helper.get_role_arn.side_effect = [
            self.test_data["connector_role_arn"],
            self.test_data["create_connector_role_arn"],
        ]

        # Mock AWS4Auth
        mock_awsauth = MagicMock()
        mock_aws4auth.return_value = mock_awsauth

        # Mock OpenSearch client
        mock_os_client = MagicMock()
        mock_opensearch.return_value = mock_os_client
        mock_os_client.transport.perform_request.return_value = (
            200,
            {},
            '{"connector_id": "test-connector-id"}',
        )

        # Create helper instance
        with patch.object(AIConnectorHelper, "__init__", return_value=None):
            helper = AIConnectorHelper()
            helper.service_type = self.service_type
            helper.opensearch_config = self.opensearch_config
            helper.aws_config = self.aws_config
            helper.iam_helper = mock_iam_helper
            helper.secret_helper = mock_secret_helper
            helper.opensearch_domain_arn = self.domain_arn

            # Mock the Connector class
            mock_connector = MagicMock()
            mock_connector.create_standalone_connector.return_value = {
                "connector_id": "test-connector-id"
            }

            with patch(
                "opensearch_py_ml.ml_commons.cli.AIConnectorHelper.Connector",
                return_value=mock_connector,
            ):

                # Test the method
                connector_id, role_arn = helper.create_connector_with_secret(
                    self.test_data["secret_name"],
                    self.test_data["secret_value"],
                    self.test_data["connector_role_name"],
                    self.test_data["create_connector_role_name"],
                    self.test_data["create_connector_input"],
                    sleep_time_in_seconds=0,  # Set to 0 for testing
                )

                # Verify secret creation
                mock_secret_helper.secret_exists.assert_called_once_with(
                    self.test_data["secret_name"]
                )
                mock_secret_helper.create_secret.assert_called_once_with(
                    self.test_data["secret_name"], self.test_data["secret_value"]
                )

                # Verify connector role creation
                expected_connector_trust_policy = {
                    "Version": "2012-10-17",
                    "Statement": [
                        {
                            "Effect": "Allow",
                            "Principal": {"Service": "es.amazonaws.com"},
                            "Action": "sts:AssumeRole",
                        }
                    ],
                }
                expected_connector_inline_policy = {
                    "Version": "2012-10-17",
                    "Statement": [
                        {
                            "Action": [
                                "secretsmanager:GetSecretValue",
                                "secretsmanager:DescribeSecret",
                            ],
                            "Effect": "Allow",
                            "Resource": self.test_data["secret_arn"],
                        }
                    ],
                }
                mock_iam_helper.role_exists.assert_any_call(
                    self.test_data["connector_role_name"]
                )
                mock_iam_helper.create_iam_role.assert_any_call(
                    self.test_data["connector_role_name"],
                    expected_connector_trust_policy,
                    expected_connector_inline_policy,
                )

                # Verify create connector role creation
                expected_create_connector_trust_policy = {
                    "Version": "2012-10-17",
                    "Statement": [
                        {
                            "Effect": "Allow",
                            "Principal": {"AWS": self.aws_config.aws_user_name},
                            "Action": "sts:AssumeRole",
                        },
                        {
                            "Effect": "Allow",
                            "Principal": {"AWS": self.aws_config.aws_role_name},
                            "Action": "sts:AssumeRole",
                        },
                    ],
                }
                expected_create_connector_inline_policy = {
                    "Version": "2012-10-17",
                    "Statement": [
                        {
                            "Effect": "Allow",
                            "Action": "iam:PassRole",
                            "Resource": self.test_data["connector_role_arn"],
                        },
                        {
                            "Effect": "Allow",
                            "Action": "es:ESHttpPost",
                            "Resource": self.domain_arn,
                        },
                    ],
                }
                mock_iam_helper.role_exists.assert_any_call(
                    self.test_data["create_connector_role_name"]
                )
                mock_iam_helper.create_iam_role.assert_any_call(
                    self.test_data["create_connector_role_name"],
                    expected_create_connector_trust_policy,
                    expected_create_connector_inline_policy,
                )

                # Verify role mapping
                mock_iam_helper.map_iam_role_to_backend_role.assert_called_once_with(
                    self.test_data["create_connector_role_arn"]
                )

    @patch("opensearch_py_ml.ml_commons.cli.AIConnectorHelper.OpenSearch")
    @patch("opensearch_py_ml.ml_commons.cli.AIConnectorHelper.AWS4Auth")
    @patch.object(AIConnectorHelper, "iam_helper", create=True)
    @patch.object(AIConnectorHelper, "secret_helper", create=True)
    def test_create_connector_with_secret_existing_resources(
        self, mock_secret_helper, mock_iam_helper, mock_aws4auth, mock_opensearch
    ):
        """Test create_connector_with_secret method with existing resources"""
        # Mock existing resources
        mock_secret_helper.secret_exists.return_value = True
        mock_secret_helper.get_secret_arn.return_value = self.test_data["secret_arn"]

        # Mock IAM helper to indicate roles exist
        mock_iam_helper.role_exists.return_value = True
        mock_iam_helper.get_role_arn = MagicMock()
        mock_iam_helper.get_role_arn.side_effect = [
            self.test_data["connector_role_arn"],
            self.test_data["create_connector_role_arn"],
        ]

        # Mock AWS4Auth
        mock_awsauth = MagicMock()
        mock_aws4auth.return_value = mock_awsauth

        # Mock OpenSearch client
        mock_os_client = MagicMock()
        mock_opensearch.return_value = mock_os_client

        # Create helper instance
        with patch.object(AIConnectorHelper, "__init__", return_value=None):
            helper = AIConnectorHelper()
            helper.service_type = self.service_type
            helper.opensearch_config = self.opensearch_config
            helper.aws_config = self.aws_config
            helper.iam_helper = mock_iam_helper
            helper.secret_helper = mock_secret_helper
            helper.opensearch_domain_arn = self.domain_arn
            helper.opensearch_client = mock_os_client

            # Mock the create_connector method
            with patch.object(
                AIConnectorHelper, "create_connector", return_value="test-connector-id"
            ) as mock_create_connector:
                # Test the method
                connector_id, role_arn = helper.create_connector_with_secret(
                    self.test_data["secret_name"],
                    self.test_data["secret_value"],
                    self.test_data["connector_role_name"],
                    self.test_data["create_connector_role_name"],
                    self.test_data["create_connector_input"],
                    sleep_time_in_seconds=0,
                )

                # Verify existing resources were used
                mock_secret_helper.create_secret.assert_not_called()
                mock_iam_helper.create_iam_role.assert_not_called()

                # Verify secret operations
                mock_secret_helper.secret_exists.assert_called_once_with(
                    self.test_data["secret_name"]
                )
                mock_secret_helper.get_secret_arn.assert_called_once_with(
                    self.test_data["secret_name"]
                )

                # Verify role operations
                mock_iam_helper.role_exists.assert_any_call(
                    self.test_data["connector_role_name"]
                )
                mock_iam_helper.role_exists.assert_any_call(
                    self.test_data["create_connector_role_name"]
                )

                # Verify get_role_arn calls
                calls = [
                    call(self.test_data["connector_role_name"]),
                    call(self.test_data["create_connector_role_name"]),
                ]
                mock_iam_helper.get_role_arn.assert_has_calls(calls, any_order=False)

                # Verify create_connector was called
                mock_create_connector.assert_called_once_with(
                    self.test_data["create_connector_role_name"],
                    self.test_data["create_connector_input"],
                )

                # Verify role mapping
                mock_iam_helper.map_iam_role_to_backend_role.assert_called_once_with(
                    self.test_data["create_connector_role_arn"]
                )

                # Assert return values
                self.assertEqual(self.test_data["connector_id"], "test-connector-id")
                self.assertEqual(role_arn, self.test_data["connector_role_arn"])

                # Verify number of calls
                self.assertEqual(mock_iam_helper.get_role_arn.call_count, 2)
                self.assertEqual(mock_iam_helper.role_exists.call_count, 2)
                self.assertEqual(mock_secret_helper.secret_exists.call_count, 1)
                self.assertEqual(mock_secret_helper.get_secret_arn.call_count, 1)

    @patch("opensearch_py_ml.ml_commons.cli.AIConnectorHelper.OpenSearch")
    @patch("opensearch_py_ml.ml_commons.cli.AIConnectorHelper.AWS4Auth")
    @patch.object(AIConnectorHelper, "iam_helper", create=True)
    def test_create_connector_with_role(
        self, mock_iam_helper, mock_aws4auth, mock_opensearch
    ):
        """Test create_connector_with_role method"""
        # Mock iam_helper methods
        mock_iam_helper.role_exists.return_value = False
        mock_iam_helper.create_iam_role.side_effect = [
            self.test_data["connector_role_arn"],
            self.test_data["create_connector_role_arn"],
        ]
        mock_iam_helper.get_role_arn.side_effect = [
            self.test_data["connector_role_arn"],
            self.test_data["create_connector_role_arn"],
        ]

        # Mock AWS4Auth
        mock_awsauth = MagicMock()
        mock_aws4auth.return_value = mock_awsauth

        # Mock OpenSearch client
        mock_os_client = MagicMock()
        mock_opensearch.return_value = mock_os_client
        mock_os_client.transport.perform_request.return_value = (
            200,
            {},
            '{"connector_id": "test-connector-id"}',
        )

        # Create helper instance
        with patch.object(AIConnectorHelper, "__init__", return_value=None):
            helper = AIConnectorHelper()
            helper.service_type = self.service_type
            helper.opensearch_config = self.opensearch_config
            helper.aws_config = self.aws_config
            helper.iam_helper = mock_iam_helper
            helper.opensearch_domain_arn = self.domain_arn

            # Mock the Connector class
            mock_connector = MagicMock()
            mock_connector.create_standalone_connector.return_value = {
                "connector_id": "test-connector-id"
            }

            with patch(
                "opensearch_py_ml.ml_commons.cli.AIConnectorHelper.Connector",
                return_value=mock_connector,
            ):
                # Test the method
                connector_id, role_arn = helper.create_connector_with_role(
                    self.test_data["connector_role_inline_policy"],
                    self.test_data["connector_role_name"],
                    self.test_data["create_connector_role_name"],
                    self.test_data["create_connector_input"],
                    sleep_time_in_seconds=0,
                )

                # Verify connector role creation
                expected_connector_trust_policy = {
                    "Version": "2012-10-17",
                    "Statement": [
                        {
                            "Effect": "Allow",
                            "Principal": {"Service": "es.amazonaws.com"},
                            "Action": "sts:AssumeRole",
                        }
                    ],
                }
                mock_iam_helper.role_exists.assert_any_call(
                    self.test_data["connector_role_name"]
                )
                mock_iam_helper.create_iam_role.assert_any_call(
                    self.test_data["connector_role_name"],
                    expected_connector_trust_policy,
                    self.test_data["connector_role_inline_policy"],
                )

                # Verify create connector role creation
                expected_create_connector_trust_policy = {
                    "Version": "2012-10-17",
                    "Statement": [
                        {
                            "Effect": "Allow",
                            "Principal": {"AWS": self.aws_config.aws_user_name},
                            "Action": "sts:AssumeRole",
                        },
                        {
                            "Effect": "Allow",
                            "Principal": {"AWS": self.aws_config.aws_role_name},
                            "Action": "sts:AssumeRole",
                        },
                    ],
                }
                expected_create_connector_inline_policy = {
                    "Version": "2012-10-17",
                    "Statement": [
                        {
                            "Effect": "Allow",
                            "Action": "iam:PassRole",
                            "Resource": self.test_data["connector_role_arn"],
                        },
                        {
                            "Effect": "Allow",
                            "Action": "es:ESHttpPost",
                            "Resource": self.domain_arn,
                        },
                    ],
                }
                mock_iam_helper.role_exists.assert_any_call(
                    self.test_data["create_connector_role_name"]
                )
                mock_iam_helper.create_iam_role.assert_any_call(
                    self.test_data["create_connector_role_name"],
                    expected_create_connector_trust_policy,
                    expected_create_connector_inline_policy,
                )

                # Verify role mapping
                mock_iam_helper.map_iam_role_to_backend_role.assert_called_once_with(
                    self.test_data["create_connector_role_arn"]
                )

                # Verify connector creation
                expected_payload = {
                    **self.test_data["create_connector_input"],
                    "credential": {"roleArn": self.test_data["connector_role_arn"]},
                }
                mock_connector.create_standalone_connector.assert_called_once_with(
                    expected_payload
                )

                # Assert return values
                self.assertEqual(connector_id, "test-connector-id")
                self.assertEqual(role_arn, self.test_data["connector_role_arn"])

    @patch("opensearch_py_ml.ml_commons.cli.AIConnectorHelper.OpenSearch")
    @patch("opensearch_py_ml.ml_commons.cli.AIConnectorHelper.AWS4Auth")
    @patch.object(AIConnectorHelper, "iam_helper", create=True)
    def test_create_connector_with_role_existing_resources(
        self, mock_iam_helper, mock_aws4auth, mock_opensearch
    ):
        """Test create_connector_with_role method with existing resources"""
        # Mock IAM helper to indicate roles exist
        mock_iam_helper.role_exists.return_value = True
        mock_iam_helper.get_role_arn = MagicMock()
        mock_iam_helper.get_role_arn.side_effect = [
            self.test_data["connector_role_arn"],
            self.test_data["create_connector_role_arn"],
        ]

        # Mock AWS4Auth
        mock_awsauth = MagicMock()
        mock_aws4auth.return_value = mock_awsauth

        # Mock OpenSearch client
        mock_os_client = MagicMock()
        mock_opensearch.return_value = mock_os_client

        # Create helper instance
        with patch.object(AIConnectorHelper, "__init__", return_value=None):
            helper = AIConnectorHelper()
            helper.service_type = self.service_type
            helper.opensearch_config = self.opensearch_config
            helper.aws_config = self.aws_config
            helper.iam_helper = mock_iam_helper
            helper.opensearch_domain_arn = self.domain_arn
            helper.opensearch_client = mock_os_client

            # Mock the create_connector method
            with patch.object(
                AIConnectorHelper, "create_connector", return_value="test-connector-id"
            ) as mock_create_connector:
                # Test the method
                connector_id, role_arn = helper.create_connector_with_role(
                    self.test_data["connector_role_inline_policy"],
                    self.test_data["connector_role_name"],
                    self.test_data["create_connector_role_name"],
                    self.test_data["create_connector_input"],
                    sleep_time_in_seconds=0,
                )

                # Verify existing resources were used
                mock_iam_helper.create_iam_role.assert_not_called()

                # Verify role operations
                mock_iam_helper.role_exists.assert_any_call(
                    self.test_data["connector_role_name"]
                )
                mock_iam_helper.role_exists.assert_any_call(
                    self.test_data["create_connector_role_name"]
                )

                # Verify get_role_arn calls
                calls = [
                    call(self.test_data["connector_role_name"]),
                    call(self.test_data["create_connector_role_name"]),
                ]
                mock_iam_helper.get_role_arn.assert_has_calls(calls, any_order=False)

                # Verify create_connector was called with correct payload
                expected_payload = {
                    **self.test_data["create_connector_input"],
                    "credential": {"roleArn": self.test_data["connector_role_arn"]},
                }
                mock_create_connector.assert_called_once_with(
                    self.test_data["create_connector_role_name"], expected_payload
                )

                # Verify role mapping
                mock_iam_helper.map_iam_role_to_backend_role.assert_called_once_with(
                    self.test_data["create_connector_role_arn"]
                )

                # Assert return values
                self.assertEqual(connector_id, "test-connector-id")
                self.assertEqual(role_arn, self.test_data["connector_role_arn"])

                # Verify number of calls
                self.assertEqual(mock_iam_helper.get_role_arn.call_count, 2)
                self.assertEqual(mock_iam_helper.role_exists.call_count, 2)
                self.assertEqual(mock_iam_helper.create_iam_role.call_count, 0)


if __name__ == "__main__":
    unittest.main()
