# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

import unittest
from unittest.mock import patch, MagicMock, Mock
from opensearch_py_ml.ml_commons.rag_pipeline.rag.serverless import Serverless
from colorama import Fore, Style

class TestServerless(unittest.TestCase):
    def setUp(self):
        # Sample data
        self.collection_name = 'test-collection'
        self.iam_principal = 'arn:aws:iam::123456789012:user/test-user'
        self.aws_region = 'us-east-1'

        # Mock aoss_client
        self.aoss_client = MagicMock()

        # Define a custom ConflictException class
        class ConflictException(Exception):
            pass

        # Mock exceptions
        self.aoss_client.exceptions = MagicMock()
        self.aoss_client.exceptions.ConflictException = ConflictException

        # Initialize the Serverless instance
        self.serverless = Serverless(
            aoss_client=self.aoss_client,
            collection_name=self.collection_name,
            iam_principal=self.iam_principal,
            aws_region=self.aws_region
        )

        # Mock sleep to speed up tests
        self.sleep_patcher = patch('time.sleep', return_value=None)
        self.mock_sleep = self.sleep_patcher.start()

    def tearDown(self):
        self.sleep_patcher.stop()

    @patch('opensearch_py_ml.ml_commons.rag_pipeline.rag.serverless.Serverless.create_access_policy')
    @patch('opensearch_py_ml.ml_commons.rag_pipeline.rag.serverless.Serverless.create_security_policy')
    def test_create_security_policies_success(self, mock_create_security_policy, mock_create_access_policy):
        self.serverless.create_security_policies()
        # Check that create_security_policy is called twice (encryption and network)
        self.assertEqual(mock_create_security_policy.call_count, 2)
        # Check that create_access_policy is called once
        mock_create_access_policy.assert_called_once()

    def test_create_security_policy_success(self):
        policy_type = 'encryption'
        name = 'test-enc-policy'
        description = 'Test encryption policy'
        policy_body = '{}'
        self.aoss_client.create_security_policy.return_value = {}
        with patch('builtins.print') as mock_print:
            self.serverless.create_security_policy(policy_type, name, description, policy_body)
            self.aoss_client.create_security_policy.assert_called_with(
                description=description,
                name=name,
                policy=policy_body,
                type=policy_type
            )
            mock_print.assert_called_with(f"{Fore.GREEN}Encryption Policy '{name}' created successfully.{Style.RESET_ALL}")

    def test_create_security_policy_conflict(self):
        policy_type = 'network'
        name = 'test-net-policy'
        description = 'Test network policy'
        policy_body = '{}'
        # Simulate ConflictException
        conflict_exception = self.aoss_client.exceptions.ConflictException()
        self.aoss_client.create_security_policy.side_effect = conflict_exception
        with patch('builtins.print') as mock_print:
            self.serverless.create_security_policy(policy_type, name, description, policy_body)
            mock_print.assert_called_with(f"{Fore.YELLOW}Network Policy '{name}' already exists.{Style.RESET_ALL}")

    def test_create_security_policy_exception(self):
        policy_type = 'invalid'
        name = 'test-policy'
        description = 'Test policy'
        policy_body = '{}'
        with patch('builtins.print') as mock_print:
            self.serverless.create_security_policy(policy_type, name, description, policy_body)
            mock_print.assert_called_with(f"{Fore.RED}Error creating {policy_type} policy '{name}': Invalid policy type specified.{Style.RESET_ALL}")

    def test_create_access_policy_success(self):
        name = 'test-access-policy'
        description = 'Test access policy'
        policy_body = '{}'
        self.aoss_client.create_access_policy.return_value = {}
        with patch('builtins.print') as mock_print:
            self.serverless.create_access_policy(name, description, policy_body)
            self.aoss_client.create_access_policy.assert_called_with(
                description=description,
                name=name,
                policy=policy_body,
                type='data'
            )
            mock_print.assert_called_with(f"{Fore.GREEN}Data Access Policy '{name}' created successfully.{Style.RESET_ALL}\n")

    def test_create_access_policy_conflict(self):
        name = 'test-access-policy'
        description = 'Test access policy'
        policy_body = '{}'
        # Simulate ConflictException
        conflict_exception = self.aoss_client.exceptions.ConflictException()
        self.aoss_client.create_access_policy.side_effect = conflict_exception
        with patch('builtins.print') as mock_print:
            self.serverless.create_access_policy(name, description, policy_body)
            mock_print.assert_called_with(f"{Fore.YELLOW}Data Access Policy '{name}' already exists.{Style.RESET_ALL}\n")

    def test_create_collection_success(self):
        self.aoss_client.create_collection.return_value = {
            'createCollectionDetail': {'id': 'collection-id-123'}
        }
        with patch('builtins.print') as mock_print:
            collection_id = self.serverless.create_collection(self.collection_name)
            self.assertEqual(collection_id, 'collection-id-123')
            mock_print.assert_called_with(f"{Fore.GREEN}Collection '{self.collection_name}' creation initiated.{Style.RESET_ALL}")

    def test_create_collection_conflict(self):
        # Simulate ConflictException
        conflict_exception = self.aoss_client.exceptions.ConflictException()
        self.aoss_client.create_collection.side_effect = conflict_exception
        self.serverless.get_collection_id = MagicMock(return_value='existing-collection-id')
        with patch('builtins.print') as mock_print:
            collection_id = self.serverless.create_collection(self.collection_name)
            self.assertEqual(collection_id, 'existing-collection-id')
            mock_print.assert_called_with(f"{Fore.YELLOW}Collection '{self.collection_name}' already exists.{Style.RESET_ALL}")

    def test_create_collection_exception_retry(self):
        # Simulate Exception on first two attempts, success on third
        self.aoss_client.create_collection.side_effect = [
            Exception('Temporary error'),
            Exception('Temporary error'),
            {'createCollectionDetail': {'id': 'collection-id-123'}}
        ]
        with patch('builtins.print'):
            collection_id = self.serverless.create_collection(self.collection_name, max_retries=3)
            self.assertEqual(collection_id, 'collection-id-123')
            self.assertEqual(self.aoss_client.create_collection.call_count, 3)

    def test_get_collection_id_success(self):
        self.aoss_client.list_collections.return_value = {
            'collectionSummaries': [
                {'name': 'other-collection', 'id': 'other-id'},
                {'name': self.collection_name, 'id': 'collection-id-123'}
            ]
        }
        collection_id = self.serverless.get_collection_id(self.collection_name)
        self.assertEqual(collection_id, 'collection-id-123')

    def test_get_collection_id_not_found(self):
        self.aoss_client.list_collections.return_value = {
            'collectionSummaries': [
                {'name': 'other-collection', 'id': 'other-id'}
            ]
        }
        collection_id = self.serverless.get_collection_id(self.collection_name)
        self.assertIsNone(collection_id)

    def test_wait_for_collection_active_success(self):
        collection_id = 'collection-id-123'
        # Simulate 'CREATING' status, then 'ACTIVE'
        self.aoss_client.batch_get_collection.side_effect = [
            {'collectionDetails': [{'status': 'CREATING'}]},
            {'collectionDetails': [{'status': 'ACTIVE'}]}
        ]
        with patch('builtins.print'):
            result = self.serverless.wait_for_collection_active(collection_id, max_wait_minutes=1)
            self.assertTrue(result)
            self.assertEqual(self.aoss_client.batch_get_collection.call_count, 2)

    def test_wait_for_collection_active_timeout(self):
        collection_id = 'collection-id-123'
        # Simulate 'CREATING' status indefinitely
        self.aoss_client.batch_get_collection.return_value = {'collectionDetails': [{'status': 'CREATING'}]}
        with patch('builtins.print'):
            result = self.serverless.wait_for_collection_active(collection_id, max_wait_minutes=0.01)
            self.assertFalse(result)

    def test_get_collection_endpoint_success(self):
        collection_id = 'collection-id-123'
        self.serverless.get_collection_id = MagicMock(return_value=collection_id)
        self.aoss_client.batch_get_collection.return_value = {
            'collectionDetails': [{'collectionEndpoint': 'https://example-endpoint.com'}]
        }
        with patch('builtins.print'):
            endpoint = self.serverless.get_collection_endpoint()
            self.assertEqual(endpoint, 'https://example-endpoint.com')

    def test_get_collection_endpoint_collection_not_found(self):
        self.serverless.get_collection_id = MagicMock(return_value=None)
        with patch('builtins.print') as mock_print:
            endpoint = self.serverless.get_collection_endpoint()
            self.assertIsNone(endpoint)
            mock_print.assert_called_with(f"{Fore.RED}Collection '{self.collection_name}' not found.{Style.RESET_ALL}\n")

    def test_get_truncated_name_within_limit(self):
        name = 'short-name'
        truncated_name = self.serverless.get_truncated_name(name, max_length=32)
        self.assertEqual(truncated_name, name)

    def test_get_truncated_name_exceeds_limit(self):
        name = 'a' * 35
        truncated_name = self.serverless.get_truncated_name(name, max_length=32)
        self.assertEqual(truncated_name, 'a' * 29 + '...')

if __name__ == '__main__':
    unittest.main()