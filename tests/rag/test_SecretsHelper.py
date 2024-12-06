# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

#  Licensed to Elasticsearch B.V. under one or more contributor
#  license agreements. See the NOTICE file distributed with
#  this work for additional information regarding copyright
#  ownership. Elasticsearch B.V. licenses this file to you under
#  the Apache License, Version 2.0 (the "License"); you may
#  not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
# 	http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing,
#  software distributed under the License is distributed on an
#  "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
#  KIND, either express or implied.  See the License for the
#  specific language governing permissions and limitations
#  under the License.

import unittest
from unittest.mock import patch, MagicMock
from botocore.exceptions import ClientError
import json
import logging
# Adjust the import path as necessary
from opensearch_py_ml.ml_commons.SecretsHelper import SecretHelper

class TestSecretHelper(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Suppress logging below ERROR level during tests
        logging.basicConfig(level=logging.ERROR)

    @patch('opensearch_py_ml.ml_commons.rag_pipeline.rag.SecretsHelper.boto3.client')
    def test_create_secret_error_logging(self, mock_boto_client):
        mock_secretsmanager = MagicMock()
        mock_boto_client.return_value = mock_secretsmanager

        error_response = {
            'Error': {
                'Code': 'InternalServiceError',
                'Message': 'An unspecified error occurred'
            }
        }
        mock_secretsmanager.create_secret.side_effect = ClientError(error_response, 'CreateSecret')

        secret_helper = SecretHelper(region='us-east-1')
        with self.assertLogs('opensearch_py_ml.ml_commons.rag_pipeline.rag.SecretsHelper', level='ERROR') as cm:
            result = secret_helper.create_secret('new-secret', {'key': 'value'})
            self.assertIsNone(result)
        self.assertIn('Error creating secret', cm.output[0])

    @patch('opensearch_py_ml.ml_commons.rag_pipeline.rag.SecretsHelper.boto3.client')
    def test_get_secret_arn_success(self, mock_boto_client):
        mock_secretsmanager = MagicMock()
        mock_boto_client.return_value = mock_secretsmanager

        mock_secretsmanager.describe_secret.return_value = {
            'ARN': 'arn:aws:secretsmanager:us-east-1:123456789012:secret:my-secret'
        }

        secret_helper = SecretHelper(region='us-east-1')
        result = secret_helper.get_secret_arn('my-secret')
        self.assertEqual(result, 'arn:aws:secretsmanager:us-east-1:123456789012:secret:my-secret')
        mock_secretsmanager.describe_secret.assert_called_with(SecretId='my-secret')

    @patch('opensearch_py_ml.ml_commons.rag_pipeline.rag.SecretsHelper.boto3.client')
    def test_get_secret_arn_not_found(self, mock_boto_client):
        mock_secretsmanager = MagicMock()
        mock_boto_client.return_value = mock_secretsmanager

        error_response = {
            'Error': {
                'Code': 'ResourceNotFoundException',
                'Message': 'Secret not found'
            }
        }
        mock_secretsmanager.describe_secret.side_effect = ClientError(error_response, 'DescribeSecret')

        secret_helper = SecretHelper(region='us-east-1')
        result = secret_helper.get_secret_arn('nonexistent-secret')
        self.assertIsNone(result)
        mock_secretsmanager.describe_secret.assert_called_with(SecretId='nonexistent-secret')

    @patch('opensearch_py_ml.ml_commons.rag_pipeline.rag.SecretsHelper.boto3.client')
    def test_get_secret_success(self, mock_boto_client):
        mock_secretsmanager = MagicMock()
        mock_boto_client.return_value = mock_secretsmanager

        mock_secretsmanager.get_secret_value.return_value = {'SecretString': 'my-secret-value'}

        secret_helper = SecretHelper(region='us-east-1')
        result = secret_helper.get_secret('my-secret')
        self.assertEqual(result, 'my-secret-value')
        mock_secretsmanager.get_secret_value.assert_called_with(SecretId='my-secret')

    @patch('opensearch_py_ml.ml_commons.rag_pipeline.rag.SecretsHelper.boto3.client')
    def test_get_secret_not_found(self, mock_boto_client):
        mock_secretsmanager = MagicMock()
        mock_boto_client.return_value = mock_secretsmanager

        error_response = {
            'Error': {
                'Code': 'ResourceNotFoundException',
                'Message': 'Secret not found'
            }
        }
        mock_secretsmanager.get_secret_value.side_effect = ClientError(error_response, 'GetSecretValue')

        secret_helper = SecretHelper(region='us-east-1')
        result = secret_helper.get_secret('nonexistent-secret')
        self.assertIsNone(result)
        mock_secretsmanager.get_secret_value.assert_called_with(SecretId='nonexistent-secret')

    @patch('opensearch_py_ml.ml_commons.rag_pipeline.rag.SecretsHelper.boto3.client')
    def test_create_secret_success(self, mock_boto_client):
        mock_secretsmanager = MagicMock()
        mock_boto_client.return_value = mock_secretsmanager

        mock_secretsmanager.create_secret.return_value = {
            'ARN': 'arn:aws:secretsmanager:us-east-1:123456789012:secret:new-secret'
        }

        secret_helper = SecretHelper(region='us-east-1')
        result = secret_helper.create_secret('new-secret', {'key': 'value'})
        self.assertEqual(result, 'arn:aws:secretsmanager:us-east-1:123456789012:secret:new-secret')
        mock_secretsmanager.create_secret.assert_called_with(
            Name='new-secret',
            SecretString=json.dumps({'key': 'value'})
        )

if __name__ == '__main__':
    unittest.main()
