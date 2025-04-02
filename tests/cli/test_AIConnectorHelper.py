# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

import json
import unittest
from io import StringIO
from unittest.mock import MagicMock, patch
from urllib.parse import urlparse

from colorama import Fore, Style
from opensearchpy import RequestsHttpConnection
from requests.auth import HTTPBasicAuth

from opensearch_py_ml.ml_commons.cli.AIConnectorHelper import AIConnectorHelper


class TestAIConnectorHelper(unittest.TestCase):
    def setUp(self):
        self.service_type = "amazon-opensearch-service"
        self.opensearch_domain_region = "us-east-1"
        self.opensearch_domain_name = "test-domain"
        self.opensearch_domain_username = "admin"
        self.opensearch_domain_password = "password"
        self.opensearch_domain_url = "test-domain-url"
        self.aws_user_name = "test-user"
        self.aws_role_name = "test-role"
        self.aws_access_key = "test-access-key"
        self.aws_secret_access_key = "test-secret-access-key"
        self.aws_session_token = "test-session-token"
        self.domain_arn = "arn:aws:es:us-east-1:123456789012:domain/test-domain"

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
            self.opensearch_domain_url,
            self.domain_arn,
        )

        # Parse the URL
        parsed_url = urlparse(self.opensearch_domain_url)
        expected_host = parsed_url.hostname
        expected_port = parsed_url.port or (
            443 if parsed_url.scheme == "https" else 9200
        )
        expected_use_ssl = parsed_url.scheme == "https"

        # Instantiate AIConnectorHelper
        helper = AIConnectorHelper(
            self.service_type,
            self.opensearch_domain_region,
            self.opensearch_domain_name,
            self.opensearch_domain_username,
            self.opensearch_domain_password,
            self.opensearch_domain_url,
            self.aws_user_name,
            self.aws_role_name,
            self.aws_access_key,
            self.aws_secret_access_key,
            self.aws_session_token,
        )

        # Assert basic attributes
        self.assertEqual(helper.service_type, self.service_type)
        self.assertEqual(helper.opensearch_domain_url, self.opensearch_domain_url)
        self.assertEqual(helper.opensearch_domain_arn, self.domain_arn)

        # Assert OpenSearch client initialization
        mock_opensearch.assert_called_once_with(
            hosts=[{"host": expected_host, "port": expected_port}],
            http_auth=(
                self.opensearch_domain_username,
                self.opensearch_domain_password,
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
                opensearch_domain_region=self.opensearch_domain_region,
                opensearch_domain_url=self.opensearch_domain_url,
                opensearch_domain_username=self.opensearch_domain_username,
                opensearch_domain_password=self.opensearch_domain_password,
                aws_access_key=self.aws_access_key,
                aws_secret_access_key=self.aws_secret_access_key,
                aws_session_token=self.aws_session_token,
            )

            # Assert SecretHelper initialization
            mock_secret_helper.assert_called_once_with(
                self.opensearch_domain_region,
                self.aws_access_key,
                self.aws_secret_access_key,
                self.aws_session_token,
            )

    def test_open_source_service_type(self):
        """Test when service_type is open-source"""
        # Initialize helper with open-source service type
        helper = AIConnectorHelper(
            service_type="open-source",
            opensearch_domain_region=self.opensearch_domain_region,
            opensearch_domain_name=self.opensearch_domain_name,
            opensearch_domain_username=self.opensearch_domain_username,
            opensearch_domain_password=self.opensearch_domain_password,
            opensearch_domain_url="https://localhost:9200",
            aws_user_name="",
            aws_role_name="",
            aws_access_key="",
            aws_secret_access_key="",
            aws_session_token="",
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
                "Endpoint": self.opensearch_domain_url,
                "ARN": self.domain_arn,
            }
        }

        # Call the method
        endpoint, arn = AIConnectorHelper.get_opensearch_domain_info(
            self.opensearch_domain_region,
            self.opensearch_domain_name,
            self.aws_access_key,
            self.aws_secret_access_key,
            self.aws_session_token,
        )

        # Assert the results
        self.assertEqual(endpoint, self.opensearch_domain_url)
        self.assertEqual(arn, self.domain_arn)
        mock_client_instance.describe_domain.assert_called_once_with(
            DomainName=self.opensearch_domain_name
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
            self.opensearch_domain_region,
            self.opensearch_domain_name,
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
            self.opensearch_domain_region,
            self.opensearch_domain_name,
            self.aws_access_key,
            self.aws_secret_access_key,
            self.aws_session_token,
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
            helper.opensearch_domain_region = self.opensearch_domain_region
            helper.iam_helper = mock_iam_helper
            helper.opensearch_domain_url = self.opensearch_domain_url
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
            helper.opensearch_domain_region = self.opensearch_domain_region
            helper.opensearch_domain_url = self.opensearch_domain_url
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
            helper.opensearch_domain_url = "https://localhost:9200"
            helper.opensearch_domain_username = "admin"
            helper.opensearch_domain_password = "password"

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
            helper.opensearch_domain_url = (
                "https://search-test-domain.us-east-1.es.amazonaws.com"
            )
            helper.opensearch_domain_username = "admin"
            helper.opensearch_domain_password = "password"
            helper.model_access_control = MagicMock()

            # Call the method
            model_id = helper.register_model(
                "test-model",
                "test description",
                "test-connector-id",
                deploy=True,
            )

            # Assert correct URL
            expected_url = f"{helper.opensearch_domain_url}/_plugins/_ml/models/_register?deploy=true"
            mock_requests_post.assert_called_once_with(
                expected_url,
                auth=HTTPBasicAuth(
                    helper.opensearch_domain_username, helper.opensearch_domain_password
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
            helper.opensearch_domain_url = (
                "https://search-test-domain.us-east-1.es.amazonaws.com"
            )
            helper.opensearch_domain_username = "admin"
            helper.opensearch_domain_password = "password"
            helper.model_access_control = MagicMock()

            # Call the method
            model_id = helper.register_model(
                "test-model",
                "test description",
                "test-connector-id",
                deploy=True,
            )

            # Assert correct URL
            expected_url = f"{helper.opensearch_domain_url}/_plugins/_ml/models/_register?deploy=true"
            mock_requests_post.assert_called_once_with(
                expected_url,
                auth=HTTPBasicAuth(
                    helper.opensearch_domain_username, helper.opensearch_domain_password
                ),
                json={
                    "name": "test-model",
                    "function_name": "remote",
                    "description": "test description",
                    "connector_id": "test-connector-id",
                },
                headers={"Content-Type": "application/json"},
            )

            # **Updated assertion** (include wait_until_task_done=True)
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
            helper.opensearch_domain_url = (
                "https://search-test-domain.us-east-1.es.amazonaws.com"
            )
            helper.opensearch_domain_username = "admin"
            helper.opensearch_domain_password = "password"
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
            helper.opensearch_domain_url = (
                "https://search-test-domain.us-east-1.es.amazonaws.com"
            )
            helper.opensearch_domain_username = "admin"
            helper.opensearch_domain_password = "password"
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
            expected_url = f"{helper.opensearch_domain_url}/_plugins/_ml/models/_register?deploy=true"
            mock_requests_post.assert_called_once_with(
                expected_url,
                auth=HTTPBasicAuth(
                    helper.opensearch_domain_username, helper.opensearch_domain_password
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
            helper.opensearch_domain_url = (
                "https://search-test-domain.us-east-1.es.amazonaws.com"
            )
            helper.opensearch_domain_username = "admin"
            helper.opensearch_domain_password = "password"
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
            expected_url = f"{helper.opensearch_domain_url}/_plugins/_ml/models/_register?deploy=true"
            mock_requests_post.assert_called_once_with(
                expected_url,
                auth=HTTPBasicAuth(
                    helper.opensearch_domain_username, helper.opensearch_domain_password
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
            helper.opensearch_domain_url = (
                "https://search-test-domain.us-east-1.es.amazonaws.com"
            )
            helper.opensearch_domain_username = "admin"
            helper.opensearch_domain_password = "password"
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
            helper.opensearch_domain_url = self.opensearch_domain_url
            helper.opensearch_domain_username = self.opensearch_domain_username
            helper.opensearch_domain_password = self.opensearch_domain_password

            # Call the method
            result = helper.deploy_model("test-model-id")

            # Assert that the method was called once
            mock_requests_post.assert_called_once()

            # Extract call arguments
            args, kwargs = mock_requests_post.call_args

            # Assert URL
            expected_url = f"{helper.opensearch_domain_url}/_plugins/_ml/models/test-model-id/_deploy"
            self.assertEqual(args[0], expected_url)

            # Assert headers
            self.assertEqual(kwargs["headers"], {"Content-Type": "application/json"})

            # Assert auth
            self.assertIsInstance(kwargs["auth"], HTTPBasicAuth)
            self.assertEqual(kwargs["auth"].username, self.opensearch_domain_username)
            self.assertEqual(kwargs["auth"].password, self.opensearch_domain_password)

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
            helper.opensearch_domain_url = self.opensearch_domain_url
            helper.opensearch_domain_username = self.opensearch_domain_username
            helper.opensearch_domain_password = self.opensearch_domain_password

            # Call the method
            payload = {"input": "test input"}
            result = helper.predict("test-model-id", payload)

            # Assert that the method was called once
            mock_requests_post.assert_called_once()

            # Extract call arguments
            args, kwargs = mock_requests_post.call_args

            # Assert URL
            expected_url = f"{helper.opensearch_domain_url}/_plugins/_ml/models/test-model-id/_predict"
            self.assertEqual(args[0], expected_url)

            # Assert JSON payload
            self.assertEqual(kwargs["json"], payload)

            # Assert headers
            self.assertEqual(kwargs["headers"], {"Content-Type": "application/json"})

            # Assert auth
            self.assertIsInstance(kwargs["auth"], HTTPBasicAuth)
            self.assertEqual(kwargs["auth"].username, self.opensearch_domain_username)
            self.assertEqual(kwargs["auth"].password, self.opensearch_domain_password)

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
            helper.opensearch_domain_url = self.opensearch_domain_url
            helper.opensearch_domain_username = self.opensearch_domain_username
            helper.opensearch_domain_password = self.opensearch_domain_password

            # Call the method
            result = helper.get_connector("test-connector-id")

            # Assert that the method was called once
            mock_requests_get.assert_called_once()

            # Extract call arguments
            args, kwargs = mock_requests_get.call_args

            # Assert URL
            expected_url = f"{helper.opensearch_domain_url}/_plugins/_ml/connectors/test-connector-id"
            self.assertEqual(args[0], expected_url)

            # Assert headers
            self.assertEqual(kwargs["headers"], {"Content-Type": "application/json"})

            # Assert auth
            self.assertIsInstance(kwargs["auth"], HTTPBasicAuth)
            self.assertEqual(kwargs["auth"].username, self.opensearch_domain_username)
            self.assertEqual(kwargs["auth"].password, self.opensearch_domain_password)

            # Assert that the response is returned
            self.assertEqual(result, response.text)


if __name__ == "__main__":
    unittest.main()
