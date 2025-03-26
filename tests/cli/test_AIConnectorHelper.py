# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

import json
import unittest
from unittest.mock import MagicMock, patch
from urllib.parse import urlparse

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
        self.opensearch_domain_url = "search-test-domain.us-east-1.es.amazonaws.com"
        self.aws_user_name = "test-user"
        self.aws_role_name = "test-role"
        self.aws_access_key = "test-access-key"
        self.aws_secret_access_key = "test-secret-access-key"
        self.aws_session_token = "test-session-token"
        # self.domain_endpoint = "search-test-domain.us-east-1.es.amazonaws.com"
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

    @patch("boto3.client")
    def test_get_opensearch_domain_info_success(self, mock_boto3_client):
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

    @patch("boto3.client")
    def test_get_opensearch_domain_info_exception(self, mock_boto3_client):
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
    def test_create_connector(self, mock_iam_helper, mock_aws4auth, mock_opensearch):
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

    @patch("requests.post")
    @patch.object(AIConnectorHelper, "get_ml_auth")
    @patch.object(AIConnectorHelper, "get_task")
    def test_register_model(self, mock_get_task, mock_get_ml_auth, mock_requests_post):
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

    def test_create_model_group_exists(self):
        # Mock the get_model_group_id_by_name to return an ID
        model_group_name = "test-model-group"
        model_group_id = "test-model-group-id"

        with patch.object(AIConnectorHelper, "__init__", return_value=None):
            helper = AIConnectorHelper()
            helper.model_access_control = MagicMock()
            helper.model_access_control.get_model_group_id_by_name.return_value = (
                model_group_id
            )

            # Call the method
            result = helper.model_access_control.get_model_group_id_by_name(
                model_group_name
            )

            # Assert that the ID is returned
            self.assertEqual(result, model_group_id)

    def test_create_model_group_new(self):
        # Mock the get_model_group_id_by_name to return None initially, then an ID
        model_group_name = "test-model-group"
        model_group_id = "test-model-group-id"

        with patch.object(AIConnectorHelper, "__init__", return_value=None):
            helper = AIConnectorHelper()
            helper.model_access_control = MagicMock()
            helper.model_access_control.get_model_group_id_by_name.side_effect = [
                None,
                model_group_id,
            ]
            helper.model_access_control.register_model_group.return_value = None

            # Call the method to get or create the model group
            result = helper.model_access_control.get_model_group_id_by_name(
                model_group_name
            )
            if result is None:
                helper.model_access_control.register_model_group(
                    name=model_group_name, description="test description"
                )
                result = helper.model_access_control.get_model_group_id_by_name(
                    model_group_name
                )

            # Assert that register_model_group was called
            helper.model_access_control.register_model_group.assert_called_once_with(
                name=model_group_name, description="test description"
            )

            # Assert that get_model_group_id_by_name was called twice
            self.assertEqual(
                helper.model_access_control.get_model_group_id_by_name.call_count, 2
            )

            # Assert that the ID is returned
        self.assertEqual(result, model_group_id)

    @patch("requests.post")
    def test_deploy_model(self, mock_requests_post):
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


if __name__ == "__main__":
    unittest.main()
