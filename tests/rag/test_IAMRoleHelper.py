# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

import unittest
from unittest.mock import patch, MagicMock
from botocore.exceptions import ClientError
import json
import logging

# Assuming IAMRoleHelper is defined in iam_role_helper.py
from opensearch_py_ml.ml_commons.IAMRoleHelper import IAMRoleHelper  # Replace with the actual module path if different

class TestIAMRoleHelper(unittest.TestCase):

    def setUp(self):
        self.region = 'us-east-1'
        self.iam_helper = IAMRoleHelper(region=self.region)

        # Configure logging to suppress error logs during tests
        logger = logging.getLogger('opensearch_py_ml.ml_commons.rag_pipeline.rag.IAMRoleHelper')
        logger.setLevel(logging.CRITICAL)  # Suppress logs below CRITICAL during tests

    @patch('opensearch_py_ml.ml_commons.rag_pipeline.rag.IAMRoleHelper.boto3.client')
    def test_role_exists_true(self, mock_boto_client):
        mock_iam_client = MagicMock()
        mock_boto_client.return_value = mock_iam_client

        mock_iam_client.get_role.return_value = {'Role': {'RoleName': 'test-role'}}

        result = self.iam_helper.role_exists('test-role')

        self.assertTrue(result)
        mock_iam_client.get_role.assert_called_with(RoleName='test-role')

    @patch('opensearch_py_ml.ml_commons.rag_pipeline.rag.IAMRoleHelper.boto3.client')
    def test_role_exists_false(self, mock_boto_client):
        mock_iam_client = MagicMock()
        mock_boto_client.return_value = mock_iam_client

        error_response = {
            'Error': {
                'Code': 'NoSuchEntity',
                'Message': 'Role does not exist'
            }
        }
        mock_iam_client.get_role.side_effect = ClientError(error_response, 'GetRole')

        result = self.iam_helper.role_exists('nonexistent-role')

        self.assertFalse(result)
        mock_iam_client.get_role.assert_called_with(RoleName='nonexistent-role')

    @patch('opensearch_py_ml.ml_commons.rag_pipeline.rag.IAMRoleHelper.boto3.client')
    def test_delete_role_success(self, mock_boto_client):
        mock_iam_client = MagicMock()
        mock_boto_client.return_value = mock_iam_client

        # Mock responses for list_attached_role_policies and list_role_policies
        mock_iam_client.list_attached_role_policies.return_value = {
            'AttachedPolicies': [{'PolicyArn': 'arn:aws:iam::aws:policy/ExamplePolicy'}]
        }
        mock_iam_client.list_role_policies.return_value = {
            'PolicyNames': ['InlinePolicy']
        }

        self.iam_helper.delete_role('test-role')

        mock_iam_client.detach_role_policy.assert_called_with(RoleName='test-role', PolicyArn='arn:aws:iam::aws:policy/ExamplePolicy')
        mock_iam_client.delete_role_policy.assert_called_with(RoleName='test-role', PolicyName='InlinePolicy')
        mock_iam_client.delete_role.assert_called_with(RoleName='test-role')

    @patch('opensearch_py_ml.ml_commons.rag_pipeline.rag.IAMRoleHelper.boto3.client')
    def test_delete_role_not_exist(self, mock_boto_client):
        mock_iam_client = MagicMock()
        mock_boto_client.return_value = mock_iam_client

        error_response = {
            'Error': {
                'Code': 'NoSuchEntity',
                'Message': 'Role does not exist'
            }
        }
        mock_iam_client.list_attached_role_policies.side_effect = ClientError(error_response, 'ListAttachedRolePolicies')

        self.iam_helper.delete_role('nonexistent-role')

        mock_iam_client.list_attached_role_policies.assert_called_with(RoleName='nonexistent-role')

    @patch('opensearch_py_ml.ml_commons.rag_pipeline.rag.IAMRoleHelper.boto3.client')
    def test_create_iam_role_success(self, mock_boto_client):
        mock_iam_client = MagicMock()
        mock_boto_client.return_value = mock_iam_client

        trust_policy = {"Version": "2012-10-17", "Statement": []}
        inline_policy = {"Version": "2012-10-17", "Statement": []}

        mock_iam_client.create_role.return_value = {
            'Role': {'Arn': 'arn:aws:iam::123456789012:role/test-role'}
        }

        role_arn = self.iam_helper.create_iam_role('test-role', trust_policy, inline_policy)

        self.assertEqual(role_arn, 'arn:aws:iam::123456789012:role/test-role')
        mock_iam_client.create_role.assert_called_with(
            RoleName='test-role',
            AssumeRolePolicyDocument=json.dumps(trust_policy),
            Description='Role with custom trust and inline policies',
        )
        mock_iam_client.put_role_policy.assert_called_with(
            RoleName='test-role',
            PolicyName='InlinePolicy',
            PolicyDocument=json.dumps(inline_policy)
        )

    @patch('opensearch_py_ml.ml_commons.rag_pipeline.rag.IAMRoleHelper.boto3.client')
    def test_create_iam_role_error(self, mock_boto_client):
        mock_iam_client = MagicMock()
        mock_boto_client.return_value = mock_iam_client

        trust_policy = {"Version": "2012-10-17", "Statement": []}
        inline_policy = {"Version": "2012-10-17", "Statement": []}

        error_response = {
            'Error': {
                'Code': 'EntityAlreadyExists',
                'Message': 'Role already exists'
            }
        }
        mock_iam_client.create_role.side_effect = ClientError(error_response, 'CreateRole')

        role_arn = self.iam_helper.create_iam_role('existing-role', trust_policy, inline_policy)

        self.assertIsNone(role_arn)
        mock_iam_client.create_role.assert_called_with(
            RoleName='existing-role',
            AssumeRolePolicyDocument=json.dumps(trust_policy),
            Description='Role with custom trust and inline policies',
        )

    @patch('opensearch_py_ml.ml_commons.rag_pipeline.rag.IAMRoleHelper.boto3.client')
    def test_get_role_arn_success(self, mock_boto_client):
        mock_iam_client = MagicMock()
        mock_boto_client.return_value = mock_iam_client

        mock_iam_client.get_role.return_value = {
            'Role': {'Arn': 'arn:aws:iam::123456789012:role/test-role'}
        }

        role_arn = self.iam_helper.get_role_arn('test-role')

        self.assertEqual(role_arn, 'arn:aws:iam::123456789012:role/test-role')
        mock_iam_client.get_role.assert_called_with(RoleName='test-role')

    @patch('opensearch_py_ml.ml_commons.rag_pipeline.rag.IAMRoleHelper.boto3.client')
    def test_get_role_arn_not_found(self, mock_boto_client):
        mock_iam_client = MagicMock()
        mock_boto_client.return_value = mock_iam_client

        error_response = {
            'Error': {
                'Code': 'NoSuchEntity',
                'Message': 'Role does not exist'
            }
        }
        mock_iam_client.get_role.side_effect = ClientError(error_response, 'GetRole')

        role_arn = self.iam_helper.get_role_arn('nonexistent-role')

        self.assertIsNone(role_arn)
        mock_iam_client.get_role.assert_called_with(RoleName='nonexistent-role')

    @patch('opensearch_py_ml.ml_commons.rag_pipeline.rag.IAMRoleHelper.boto3.client')
    def test_get_user_arn_success(self, mock_boto_client):
        mock_iam_client = MagicMock()
        mock_boto_client.return_value = mock_iam_client

        mock_iam_client.get_user.return_value = {
            'User': {'Arn': 'arn:aws:iam::123456789012:user/test-user'}
        }

        user_arn = self.iam_helper.get_user_arn('test-user')

        self.assertEqual(user_arn, 'arn:aws:iam::123456789012:user/test-user')
        mock_iam_client.get_user.assert_called_with(UserName='test-user')

    @patch('opensearch_py_ml.ml_commons.rag_pipeline.rag.IAMRoleHelper.boto3.client')
    def test_get_user_arn_not_found(self, mock_boto_client):
        mock_iam_client = MagicMock()
        mock_boto_client.return_value = mock_iam_client

        error_response = {
            'Error': {
                'Code': 'NoSuchEntity',
                'Message': 'User does not exist'
            }
        }
        mock_iam_client.get_user.side_effect = ClientError(error_response, 'GetUser')

        user_arn = self.iam_helper.get_user_arn('nonexistent-user')

        self.assertIsNone(user_arn)
        mock_iam_client.get_user.assert_called_with(UserName='nonexistent-user')

    @patch('opensearch_py_ml.ml_commons.rag_pipeline.rag.IAMRoleHelper.boto3.client')
    def test_assume_role_success(self, mock_boto_client):
        mock_sts_client = MagicMock()
        mock_boto_client.return_value = mock_sts_client

        mock_sts_client.assume_role.return_value = {
            'Credentials': {
                'AccessKeyId': 'ASIA...',
                'SecretAccessKey': 'secret',
                'SessionToken': 'token'
            }
        }

        role_arn = 'arn:aws:iam::123456789012:role/test-role'
        credentials = self.iam_helper.assume_role(role_arn, 'test-session')

        self.assertIsNotNone(credentials)
        self.assertEqual(credentials['AccessKeyId'], 'ASIA...')
        mock_sts_client.assume_role.assert_called_with(
            RoleArn=role_arn,
            RoleSessionName='test-session',
        )

    @patch('opensearch_py_ml.ml_commons.rag_pipeline.rag.IAMRoleHelper.boto3.client')
    def test_assume_role_failure(self, mock_boto_client):
        mock_sts_client = MagicMock()
        mock_boto_client.return_value = mock_sts_client

        error_response = {
            'Error': {
                'Code': 'AccessDenied',
                'Message': 'User is not authorized to perform: sts:AssumeRole'
            }
        }
        mock_sts_client.assume_role.side_effect = ClientError(error_response, 'AssumeRole')

        role_arn = 'arn:aws:iam::123456789012:role/unauthorized-role'
        credentials = self.iam_helper.assume_role(role_arn, 'test-session')

        self.assertIsNone(credentials)
        mock_sts_client.assume_role.assert_called_with(
            RoleArn=role_arn,
            RoleSessionName='test-session',
        )

    @patch('opensearch_py_ml.ml_commons.rag_pipeline.rag.IAMRoleHelper.requests.put')
    def test_map_iam_role_to_backend_role_success(self, mock_put):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_put.return_value = mock_response

        self.iam_helper.opensearch_domain_url = 'https://search-domain'
        self.iam_helper.opensearch_domain_username = 'user'
        self.iam_helper.opensearch_domain_password = 'pass'

        iam_role_arn = 'arn:aws:iam::123456789012:role/test-role'

        self.iam_helper.map_iam_role_to_backend_role(iam_role_arn)

        mock_put.assert_called_once()
        args, kwargs = mock_put.call_args
        self.assertIn('/_plugins/_security/api/rolesmapping/ml_full_access', args[0])
        self.assertEqual(kwargs['auth'], ('user', 'pass'))
        self.assertEqual(kwargs['json'], {'backend_roles': [iam_role_arn]})

    @patch('opensearch_py_ml.ml_commons.rag_pipeline.rag.IAMRoleHelper.requests.put')
    def test_map_iam_role_to_backend_role_failure(self, mock_put):
        mock_response = MagicMock()
        mock_response.status_code = 403
        mock_response.text = 'Forbidden'
        mock_put.return_value = mock_response

        self.iam_helper.opensearch_domain_url = 'https://search-domain'
        self.iam_helper.opensearch_domain_username = 'user'
        self.iam_helper.opensearch_domain_password = 'pass'

        iam_role_arn = 'arn:aws:iam::123456789012:role/test-role'

        self.iam_helper.map_iam_role_to_backend_role(iam_role_arn)

        mock_put.assert_called_once()
        args, kwargs = mock_put.call_args
        self.assertIn('/_plugins/_security/api/rolesmapping/ml_full_access', args[0])

    def test_get_iam_user_name_from_arn_valid(self):
        iam_principal_arn = 'arn:aws:iam::123456789012:user/test-user'
        user_name = self.iam_helper.get_iam_user_name_from_arn(iam_principal_arn)
        self.assertEqual(user_name, 'test-user')

    def test_get_iam_user_name_from_arn_invalid(self):
        iam_principal_arn = 'arn:aws:iam::123456789012:role/test-role'
        user_name = self.iam_helper.get_iam_user_name_from_arn(iam_principal_arn)
        self.assertIsNone(user_name)

if __name__ == '__main__':
    unittest.main()