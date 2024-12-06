# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

import unittest
from unittest.mock import Mock, patch, call
import json
from io import StringIO
from opensearch_py_ml.ml_commons.rag_pipeline.rag.ml_models.SageMakerModel import SageMakerModel

class TestSageMakerModel(unittest.TestCase):

    def setUp(self):
        self.aws_region = "us-west-2"
        self.opensearch_domain_name = "test-domain"
        self.opensearch_username = "test-user"
        self.opensearch_password = "test-password"
        self.mock_iam_role_helper = Mock()
        self.sagemaker_model = SageMakerModel(
            self.aws_region,
            self.opensearch_domain_name,
            self.opensearch_username,
            self.opensearch_password,
            self.mock_iam_role_helper
        )

    def test_init(self):
        self.assertEqual(self.sagemaker_model.aws_region, self.aws_region)
        self.assertEqual(self.sagemaker_model.opensearch_domain_name, self.opensearch_domain_name)
        self.assertEqual(self.sagemaker_model.opensearch_username, self.opensearch_username)
        self.assertEqual(self.sagemaker_model.opensearch_password, self.opensearch_password)
        self.assertEqual(self.sagemaker_model.iam_role_helper, self.mock_iam_role_helper)

    @patch('builtins.input', side_effect=[
        'arn:aws:sagemaker:us-west-2:123456789012:endpoint/test-endpoint',
        'https://runtime.sagemaker.us-west-2.amazonaws.com/endpoints/test-endpoint/invocations',
        ''  # Empty string for region, to use default
    ])
    def test_register_sagemaker_model(self, mock_input):
        mock_helper = Mock()
        mock_helper.create_connector_with_role.return_value = "test-connector-id"
        mock_helper.create_model.return_value = "test-model-id"
        
        mock_config = {}
        mock_save_config = Mock()

        self.sagemaker_model.register_sagemaker_model(mock_helper, mock_config, mock_save_config)

        # Check if create_connector_with_role was called
        mock_helper.create_connector_with_role.assert_called_once()
        call_args, call_kwargs = mock_helper.create_connector_with_role.call_args

        # Check the arguments without assuming a specific order or number
        self.assertIn("my_test_sagemaker_connector_role", call_args)
        self.assertIn("my_test_create_sagemaker_connector_role", call_args)

        # Check the inline policy
        inline_policy = next(arg for arg in call_args if isinstance(arg, dict) and 'Statement' in arg)
        self.assertEqual(inline_policy['Statement'][0]['Action'], ["sagemaker:InvokeEndpoint"])
        self.assertEqual(inline_policy['Statement'][0]['Resource'], 
                        "arn:aws:sagemaker:us-west-2:123456789012:endpoint/test-endpoint")

        # Check the connector input
        connector_input = next(arg for arg in call_args if isinstance(arg, dict) and 'name' in arg)
        self.assertEqual(connector_input['name'], "SageMaker Embedding Model Connector")
        self.assertEqual(connector_input['parameters']['region'], "us-west-2")
        self.assertEqual(connector_input['actions'][0]['url'], 
                        "https://runtime.sagemaker.us-west-2.amazonaws.com/endpoints/test-endpoint/invocations")

        # Check if create_model was called with correct arguments
        mock_helper.create_model.assert_called_once_with(
            "SageMaker Embedding Model",
            "SageMaker embedding model for semantic search",
            "test-connector-id",
            "my_test_create_sagemaker_connector_role"
        )

        # Check if config was saved correctly
        mock_save_config.assert_called_once_with({'embedding_model_id': 'test-model-id'})

    @patch('builtins.input', side_effect=[
        'arn:aws:sagemaker:us-west-2:123456789012:endpoint/test-endpoint',
        'https://runtime.sagemaker.us-west-2.amazonaws.com/endpoints/test-endpoint/invocations',
        ''  # Empty string for region, to use default
    ])
    def test_register_sagemaker_model_connector_creation_failure(self, mock_input):
        mock_helper = Mock()
        mock_helper.create_connector_with_role.return_value = None
        
        mock_config = {}
        mock_save_config = Mock()

        self.sagemaker_model.register_sagemaker_model(mock_helper, mock_config, mock_save_config)

        mock_helper.create_model.assert_not_called()
        mock_save_config.assert_not_called()

    @patch('builtins.input', side_effect=[
        'arn:aws:sagemaker:us-west-2:123456789012:endpoint/test-endpoint',
        'https://runtime.sagemaker.us-west-2.amazonaws.com/endpoints/test-endpoint/invocations',
        ''  # Empty string for region, to use default
    ])
    def test_register_sagemaker_model_model_creation_failure(self, mock_input):
        mock_helper = Mock()
        mock_helper.create_connector_with_role.return_value = "test-connector-id"
        mock_helper.create_model.return_value = None
        
        mock_config = {}
        mock_save_config = Mock()

        self.sagemaker_model.register_sagemaker_model(mock_helper, mock_config, mock_save_config)

        mock_save_config.assert_not_called()

if __name__ == '__main__':
    unittest.main()
