# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

import unittest
from unittest.mock import patch, MagicMock
import json
import requests

from opensearch_py_ml.ml_commons.rag_pipeline.rag.AIConnectorHelper import AIConnectorHelper

class TestAIConnectorHelper(unittest.TestCase):
    def setUp(self):
        self.region = 'us-east-1'
        self.opensearch_domain_name = 'test-domain'
        self.opensearch_domain_username = 'admin'
        self.opensearch_domain_password = 'password'
        self.aws_user_name = 'test-user'
        self.aws_role_name = 'test-role'

        self.domain_endpoint = 'search-test-domain.us-east-1.es.amazonaws.com'
        self.domain_arn = 'arn:aws:es:us-east-1:123456789012:domain/test-domain'

    @patch('opensearch_py_ml.ml_commons.rag_pipeline.rag.AIConnectorHelper.AIConnectorHelper.get_opensearch_domain_info')
    @patch('opensearch_py_ml.ml_commons.rag_pipeline.rag.AIConnectorHelper.OpenSearch')
    @patch('opensearch_py_ml.ml_commons.rag_pipeline.rag.AIConnectorHelper.SecretHelper')
    @patch('opensearch_py_ml.ml_commons.rag_pipeline.rag.AIConnectorHelper.IAMRoleHelper')
    def test___init__(self, mock_iam_role_helper, mock_secret_helper, mock_opensearch, mock_get_opensearch_domain_info):
        # Mock get_opensearch_domain_info
        mock_get_opensearch_domain_info.return_value = (self.domain_endpoint, self.domain_arn)

        # Instantiate AIConnectorHelper
        helper = AIConnectorHelper(
            self.region,
            self.opensearch_domain_name,
            self.opensearch_domain_username,
            self.opensearch_domain_password,
            self.aws_user_name,
            self.aws_role_name
        )

        # Assert domain URL
        expected_domain_url = f'https://{self.domain_endpoint}'
        self.assertEqual(helper.opensearch_domain_url, expected_domain_url)

        # Assert opensearch_client is initialized
        mock_opensearch.assert_called_once_with(
            hosts=[{'host': self.domain_endpoint, 'port': 443}],
            http_auth=(self.opensearch_domain_username, self.opensearch_domain_password),
            use_ssl=True,
            verify_certs=True,
            connection_class=unittest.mock.ANY
        )

        # Assert IAMRoleHelper and SecretHelper are initialized
        mock_iam_role_helper.assert_called_once_with(
            region=self.region,
            opensearch_domain_url=expected_domain_url,
            opensearch_domain_username=self.opensearch_domain_username,
            opensearch_domain_password=self.opensearch_domain_password,
            aws_user_name=self.aws_user_name,
            aws_role_name=self.aws_role_name,
            opensearch_domain_arn=self.domain_arn
        )
        mock_secret_helper.assert_called_once_with(self.region)

    @patch('boto3.client')
    def test_get_opensearch_domain_info_success(self, mock_boto3_client):
        # Mock the boto3 client
        mock_client_instance = MagicMock()
        mock_boto3_client.return_value = mock_client_instance

        # Mock the describe_domain response
        mock_client_instance.describe_domain.return_value = {
            'DomainStatus': {
                'Endpoint': self.domain_endpoint,
                'ARN': self.domain_arn
            }
        }

        # Call the method
        endpoint, arn = AIConnectorHelper.get_opensearch_domain_info(self.region, self.opensearch_domain_name)

        # Assert the results
        self.assertEqual(endpoint, self.domain_endpoint)
        self.assertEqual(arn, self.domain_arn)
        mock_client_instance.describe_domain.assert_called_once_with(DomainName=self.opensearch_domain_name)

    @patch('boto3.client')
    def test_get_opensearch_domain_info_exception(self, mock_boto3_client):
        # Mock the boto3 client to raise an exception
        mock_client_instance = MagicMock()
        mock_boto3_client.return_value = mock_client_instance
        mock_client_instance.describe_domain.side_effect = Exception('Test Exception')

        # Call the method
        endpoint, arn = AIConnectorHelper.get_opensearch_domain_info(self.region, self.opensearch_domain_name)

        # Assert the results are None
        self.assertIsNone(endpoint)
        self.assertIsNone(arn)

    @patch.object(AIConnectorHelper, 'iam_helper', create=True)
    def test_get_ml_auth_success(self, mock_iam_helper):
        # Mock the get_role_arn to return a role ARN
        create_connector_role_name = 'test-create-connector-role'
        create_connector_role_arn = 'arn:aws:iam::123456789012:role/test-create-connector-role'
        mock_iam_helper.get_role_arn.return_value = create_connector_role_arn

        # Mock the assume_role to return temp credentials
        temp_credentials = {
            "AccessKeyId": "test-access-key",
            "SecretAccessKey": "test-secret-key",
            "SessionToken": "test-session-token"
        }
        mock_iam_helper.assume_role.return_value = temp_credentials

        # Instantiate helper
        with patch.object(AIConnectorHelper, '__init__', return_value=None):
            helper = AIConnectorHelper()
            helper.region = self.region
            helper.iam_helper = mock_iam_helper
            helper.opensearch_domain_url = f'https://{self.domain_endpoint}'
            helper.opensearch_domain_arn = self.domain_arn

            # Call the method
            awsauth = helper.get_ml_auth(create_connector_role_name)

            # Assert that the IAM helper methods were called
            mock_iam_helper.get_role_arn.assert_called_with(create_connector_role_name)
            mock_iam_helper.assume_role.assert_called_with(create_connector_role_arn)

            # Since AWS4Auth is instantiated within the method, we can check if awsauth is not None
            self.assertIsNotNone(awsauth)

    @patch.object(AIConnectorHelper, 'iam_helper', create=True)
    def test_get_ml_auth_role_not_found(self, mock_iam_helper):
        # Mock the get_role_arn to return None
        create_connector_role_name = 'test-create-connector-role'
        mock_iam_helper.get_role_arn.return_value = None

        # Instantiate helper
        with patch.object(AIConnectorHelper, '__init__', return_value=None):
            helper = AIConnectorHelper()
            helper.iam_helper = mock_iam_helper

            # Call the method and expect an exception
            with self.assertRaises(Exception) as context:
                helper.get_ml_auth(create_connector_role_name)

            self.assertTrue(f"IAM role '{create_connector_role_name}' not found." in str(context.exception))

    @patch('requests.post')
    @patch('opensearch_py_ml.ml_commons.rag_pipeline.rag.AIConnectorHelper.AWS4Auth')
    @patch.object(AIConnectorHelper, 'iam_helper', create=True)
    def test_create_connector(self, mock_iam_helper, mock_aws4auth, mock_requests_post):
        # Mock the IAM helper methods
        create_connector_role_name = 'test-create-connector-role'
        create_connector_role_arn = 'arn:aws:iam::123456789012:role/test-create-connector-role'
        mock_iam_helper.get_role_arn.return_value = create_connector_role_arn
        temp_credentials = {
            "AccessKeyId": "test-access-key",
            "SecretAccessKey": "test-secret-key",
            "SessionToken": "test-session-token"
        }
        mock_iam_helper.assume_role.return_value = temp_credentials

        # Mock AWS4Auth
        mock_awsauth = MagicMock()
        mock_aws4auth.return_value = mock_awsauth

        # Mock requests.post
        response = MagicMock()
        response.text = json.dumps({'connector_id': 'test-connector-id'})
        mock_requests_post.return_value = response

        # Instantiate helper
        with patch.object(AIConnectorHelper, '__init__', return_value=None):
            helper = AIConnectorHelper()
            helper.region = self.region
            helper.opensearch_domain_url = f'https://{self.domain_endpoint}'
            helper.iam_helper = mock_iam_helper

            # Call the method
            payload = {'key': 'value'}
            connector_id = helper.create_connector(create_connector_role_name, payload)

            # Assert that the correct URL was used
            expected_url = f'{helper.opensearch_domain_url}/_plugins/_ml/connectors/_create'
            mock_requests_post.assert_called_once_with(
                expected_url,
                auth=mock_awsauth,
                json=payload,
                headers={"Content-Type": "application/json"}
            )

            # Assert that the connector_id is returned
            self.assertEqual(connector_id, 'test-connector-id')

    @patch.object(AIConnectorHelper, 'model_access_control', create=True)
    def test_search_model_group(self, mock_model_access_control):
        # Mock the response from model_access_control.search_model_group_by_name
        model_group_name = 'test-model-group'
        mock_response = {'hits': {'hits': []}}
        mock_model_access_control.search_model_group_by_name.return_value = mock_response

        # Instantiate helper
        with patch.object(AIConnectorHelper, '__init__', return_value=None):
            helper = AIConnectorHelper()
            helper.model_access_control = mock_model_access_control

            # Call the method
            response = helper.search_model_group(model_group_name, 'test-create-connector-role')

            # Assert that the method was called with correct parameters
            mock_model_access_control.search_model_group_by_name.assert_called_once_with(model_group_name, size=1)

            # Assert that the response is as expected
            self.assertEqual(response, mock_response)

    @patch.object(AIConnectorHelper, 'model_access_control', create=True)
    def test_create_model_group_exists(self, mock_model_access_control):
        # Mock the get_model_group_id_by_name to return an ID
        model_group_name = 'test-model-group'
        model_group_id = 'test-model-group-id'
        mock_model_access_control.get_model_group_id_by_name.return_value = model_group_id

        # Instantiate helper
        with patch.object(AIConnectorHelper, '__init__', return_value=None):
            helper = AIConnectorHelper()
            helper.model_access_control = mock_model_access_control

            # Call the method
            result = helper.create_model_group(model_group_name, 'test description', 'test-create-connector-role')

            # Assert that the ID is returned
            self.assertEqual(result, model_group_id)

    @patch.object(AIConnectorHelper, 'model_access_control', create=True)
    def test_create_model_group_new(self, mock_model_access_control):
        # Mock the get_model_group_id_by_name to return None initially, then an ID
        model_group_name = 'test-model-group'
        model_group_id = 'test-model-group-id'

        # First call returns None, second call returns the ID
        mock_model_access_control.get_model_group_id_by_name.side_effect = [None, model_group_id]

        # Instantiate helper
        with patch.object(AIConnectorHelper, '__init__', return_value=None):
            helper = AIConnectorHelper()
            helper.model_access_control = mock_model_access_control

            # Call the method
            result = helper.create_model_group(model_group_name, 'test description', 'test-create-connector-role')

            # Assert that register_model_group was called
            mock_model_access_control.register_model_group.assert_called_once_with(name=model_group_name, description='test description')

            # Assert that the ID is returned
            self.assertEqual(result, model_group_id)

    @patch.object(AIConnectorHelper, 'get_task')
    @patch('time.sleep', return_value=None)
    @patch('requests.post')
    @patch.object(AIConnectorHelper, 'get_ml_auth')
    @patch.object(AIConnectorHelper, 'create_model_group')
    def test_create_model(self, mock_create_model_group, mock_get_ml_auth, mock_requests_post, mock_sleep, mock_get_task):
        # Mock create_model_group
        model_group_id = 'test-model-group-id'
        mock_create_model_group.return_value = model_group_id

        # Mock get_ml_auth
        mock_awsauth = MagicMock()
        mock_get_ml_auth.return_value = mock_awsauth

        # Mock requests.post
        response = MagicMock()
        response.text = json.dumps({'model_id': 'test-model-id'})
        mock_requests_post.return_value = response

        # Instantiate helper
        with patch.object(AIConnectorHelper, '__init__', return_value=None):
            helper = AIConnectorHelper()
            helper.opensearch_domain_url = f'https://{self.domain_endpoint}'

            # Call the method
            model_id = helper.create_model('test-model', 'test description', 'test-connector-id', 'test-create-connector-role', deploy=True)

            # Assert that create_model_group was called
            mock_create_model_group.assert_called_once_with('test-model', 'test description', 'test-create-connector-role')

            # Assert that the correct URL was used
            expected_url = f'{helper.opensearch_domain_url}/_plugins/_ml/models/_register?deploy=true'
            payload = {
                "name": 'test-model',
                "function_name": "remote",
                "description": 'test description',
                "model_group_id": model_group_id,
                "connector_id": 'test-connector-id'
            }
            mock_requests_post.assert_called_once_with(
                expected_url,
                auth=mock_awsauth,
                json=payload,
                headers={"Content-Type": "application/json"}
            )

            # Assert that model_id is returned
            self.assertEqual(model_id, 'test-model-id')

    @patch('requests.post')
    def test_deploy_model(self, mock_requests_post):
        # Mock requests.post
        response = MagicMock()
        response.text = 'Deploy model response'
        mock_requests_post.return_value = response

        # Instantiate helper
        with patch.object(AIConnectorHelper, '__init__', return_value=None):
            helper = AIConnectorHelper()
            helper.opensearch_domain_url = f'https://{self.domain_endpoint}'
            helper.opensearch_domain_username = self.opensearch_domain_username
            helper.opensearch_domain_password = self.opensearch_domain_password

            # Call the method
            result = helper.deploy_model('test-model-id')

            # Assert that the method was called once
            mock_requests_post.assert_called_once()

            # Extract call arguments
            args, kwargs = mock_requests_post.call_args

            # Assert URL
            expected_url = f'{helper.opensearch_domain_url}/_plugins/_ml/models/test-model-id/_deploy'
            self.assertEqual(args[0], expected_url)

            # Assert headers
            self.assertEqual(kwargs['headers'], {"Content-Type": "application/json"})

            # Assert auth
            self.assertIsInstance(kwargs['auth'], requests.auth.HTTPBasicAuth)
            self.assertEqual(kwargs['auth'].username, self.opensearch_domain_username)
            self.assertEqual(kwargs['auth'].password, self.opensearch_domain_password)

            # Assert that the response is returned
            self.assertEqual(result, response)
    @patch('requests.post')
    def test_predict(self, mock_requests_post):
        # Mock requests.post
        response = MagicMock()
        response.text = 'Predict response'
        mock_requests_post.return_value = response

        # Instantiate helper
        with patch.object(AIConnectorHelper, '__init__', return_value=None):
            helper = AIConnectorHelper()
            helper.opensearch_domain_url = f'https://{self.domain_endpoint}'
            helper.opensearch_domain_username = self.opensearch_domain_username
            helper.opensearch_domain_password = self.opensearch_domain_password

            # Call the method
            payload = {'input': 'test input'}
            result = helper.predict('test-model-id', payload)

            # Assert that the method was called once
            mock_requests_post.assert_called_once()

            # Extract call arguments
            args, kwargs = mock_requests_post.call_args

            # Assert URL
            expected_url = f'{helper.opensearch_domain_url}/_plugins/_ml/models/test-model-id/_predict'
            self.assertEqual(args[0], expected_url)

            # Assert JSON payload
            self.assertEqual(kwargs['json'], payload)

            # Assert headers
            self.assertEqual(kwargs['headers'], {"Content-Type": "application/json"})

            # Assert auth
            self.assertIsInstance(kwargs['auth'], requests.auth.HTTPBasicAuth)
            self.assertEqual(kwargs['auth'].username, self.opensearch_domain_username)
            self.assertEqual(kwargs['auth'].password, self.opensearch_domain_password)

            # Assert that the response is returned
            self.assertEqual(result, response)

    @patch('time.sleep', return_value=None)
    @patch.object(AIConnectorHelper, 'create_connector')
    @patch.object(AIConnectorHelper, 'secret_helper', create=True)
    @patch.object(AIConnectorHelper, 'iam_helper', create=True)
    def test_create_connector_with_secret(self, mock_iam_helper, mock_secret_helper, mock_create_connector, mock_sleep):
        # Mock secret_helper methods
        secret_name = 'test-secret'
        secret_value = 'test-secret-value'
        secret_arn = 'arn:aws:secretsmanager:us-east-1:123456789012:secret:test-secret'
        mock_secret_helper.secret_exists.return_value = False
        mock_secret_helper.create_secret.return_value = secret_arn
        mock_secret_helper.get_secret_arn.return_value = secret_arn

        # Mock iam_helper methods
        connector_role_name = 'test-connector-role'
        create_connector_role_name = 'test-create-connector-role'
        connector_role_arn = 'arn:aws:iam::123456789012:role/test-connector-role'
        create_connector_role_arn = 'arn:aws:iam::123456789012:role/test-create-connector-role'
        mock_iam_helper.role_exists.side_effect = [False, False]
        mock_iam_helper.create_iam_role.side_effect = [connector_role_arn, create_connector_role_arn]
        mock_iam_helper.get_user_arn.return_value = 'arn:aws:iam::123456789012:user/test-user'
        mock_iam_helper.get_role_arn.side_effect = [connector_role_arn, create_connector_role_arn]
        mock_iam_helper.map_iam_role_to_backend_role.return_value = None

        # Mock create_connector
        connector_id = 'test-connector-id'
        mock_create_connector.return_value = connector_id

        # Instantiate helper
        with patch.object(AIConnectorHelper, '__init__', return_value=None):
            helper = AIConnectorHelper()
            helper.region = self.region
            helper.aws_user_name = self.aws_user_name
            helper.aws_role_name = self.aws_role_name
            helper.opensearch_domain_arn = self.domain_arn
            helper.opensearch_domain_url = f'https://{self.domain_endpoint}'
            helper.iam_helper = mock_iam_helper
            helper.secret_helper = mock_secret_helper

            # Prepare input
            create_connector_input = {'key': 'value'}

            # Call the method
            result = helper.create_connector_with_secret(
                secret_name,
                secret_value,
                connector_role_name,
                create_connector_role_name,
                create_connector_input,
                sleep_time_in_seconds=0  # For faster testing
            )

            # Assert that the methods were called
            mock_secret_helper.secret_exists.assert_called_once_with(secret_name)
            mock_secret_helper.create_secret.assert_called_once_with(secret_name, secret_value)

            self.assertEqual(mock_iam_helper.role_exists.call_count, 2)
            self.assertEqual(mock_iam_helper.create_iam_role.call_count, 2)
            mock_iam_helper.map_iam_role_to_backend_role.assert_called_once_with(create_connector_role_arn)

            # Assert that create_connector was called
            payload = create_connector_input.copy()
            payload['credential'] = {
                "secretArn": secret_arn,
                "roleArn": connector_role_arn
            }
            mock_create_connector.assert_called_once_with(create_connector_role_name, payload)

            # Assert that the connector_id is returned
            self.assertEqual(result, connector_id)

    @patch('time.sleep', return_value=None)
    @patch.object(AIConnectorHelper, 'create_connector')
    @patch.object(AIConnectorHelper, 'iam_helper', create=True)
    def test_create_connector_with_role(self, mock_iam_helper, mock_create_connector, mock_sleep):
        # Mock iam_helper methods
        connector_role_name = 'test-connector-role'
        create_connector_role_name = 'test-create-connector-role'
        connector_role_arn = 'arn:aws:iam::123456789012:role/test-connector-role'
        create_connector_role_arn = 'arn:aws:iam::123456789012:role/test-create-connector-role'
        mock_iam_helper.role_exists.side_effect = [False, False]
        mock_iam_helper.create_iam_role.side_effect = [connector_role_arn, create_connector_role_arn]
        mock_iam_helper.get_user_arn.return_value = 'arn:aws:iam::123456789012:user/test-user'
        mock_iam_helper.get_role_arn.side_effect = [connector_role_arn, create_connector_role_arn]
        mock_iam_helper.map_iam_role_to_backend_role.return_value = None

        # Mock create_connector
        connector_id = 'test-connector-id'
        mock_create_connector.return_value = connector_id

        # Instantiate helper
        with patch.object(AIConnectorHelper, '__init__', return_value=None):
            helper = AIConnectorHelper()
            helper.region = self.region
            helper.aws_user_name = self.aws_user_name
            helper.aws_role_name = self.aws_role_name
            helper.opensearch_domain_arn = self.domain_arn
            helper.opensearch_domain_url = f'https://{self.domain_endpoint}'
            helper.iam_helper = mock_iam_helper

            # Prepare input
            create_connector_input = {'key': 'value'}
            connector_role_inline_policy = {'Statement': []}

            # Call the method
            result = helper.create_connector_with_role(
                connector_role_inline_policy,
                connector_role_name,
                create_connector_role_name,
                create_connector_input,
                sleep_time_in_seconds=0  # For faster testing
            )

            # Assert that the methods were called
            self.assertEqual(mock_iam_helper.role_exists.call_count, 2)
            self.assertEqual(mock_iam_helper.create_iam_role.call_count, 2)
            mock_iam_helper.map_iam_role_to_backend_role.assert_called_once_with(create_connector_role_arn)

            # Assert that create_connector was called
            payload = create_connector_input.copy()
            payload['credential'] = {
                "roleArn": connector_role_arn
            }
            mock_create_connector.assert_called_once_with(create_connector_role_name, payload)

            # Assert that the connector_id is returned
            self.assertEqual(result, connector_id)

if __name__ == '__main__':
    unittest.main()