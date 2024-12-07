# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

import unittest
from unittest.mock import Mock, patch

from opensearch_py_ml.ml_commons.rag_pipeline.rag.ml_models.BedrockModel import (
    BedrockModel,
)


class TestBedrockModel(unittest.TestCase):

    def setUp(self):
        self.aws_region = "us-west-2"
        self.opensearch_domain_name = "test-domain"
        self.opensearch_username = "test-user"
        self.opensearch_password = "test-password"
        self.mock_iam_role_helper = Mock()
        self.bedrock_model = BedrockModel(
            self.aws_region,
            self.opensearch_domain_name,
            self.opensearch_username,
            self.opensearch_password,
            self.mock_iam_role_helper,
        )

    def test_init(self):
        self.assertEqual(self.bedrock_model.aws_region, self.aws_region)
        self.assertEqual(
            self.bedrock_model.opensearch_domain_name, self.opensearch_domain_name
        )
        self.assertEqual(
            self.bedrock_model.opensearch_username, self.opensearch_username
        )
        self.assertEqual(
            self.bedrock_model.opensearch_password, self.opensearch_password
        )
        self.assertEqual(self.bedrock_model.iam_role_helper, self.mock_iam_role_helper)

    @patch("builtins.input", side_effect=["", "1"])
    def test_register_bedrock_model_default(self, mock_input):
        mock_helper = Mock()
        mock_helper.create_connector_with_role.return_value = "test-connector-id"
        mock_helper.create_model.return_value = "test-model-id"

        mock_config = {}
        mock_save_config = Mock()

        self.bedrock_model.register_bedrock_model(
            mock_helper, mock_config, mock_save_config
        )

        mock_helper.create_connector_with_role.assert_called_once()
        mock_helper.create_model.assert_called_once()
        mock_save_config.assert_called_once_with(
            {"embedding_model_id": "test-model-id"}
        )

    @patch(
        "builtins.input",
        side_effect=[
            "custom-region",
            "2",
            '{"name": "Custom Model", "description": "Custom description"}',
        ],
    )
    def test_register_bedrock_model_custom(self, mock_input):
        mock_helper = Mock()
        mock_helper.create_connector_with_role.return_value = "test-connector-id"
        mock_helper.create_model.return_value = "test-model-id"

        mock_config = {}
        mock_save_config = Mock()

        self.bedrock_model.register_bedrock_model(
            mock_helper, mock_config, mock_save_config
        )

        mock_helper.create_connector_with_role.assert_called_once()
        mock_helper.create_model.assert_called_once_with(
            "Custom Model",
            "Custom description",
            "test-connector-id",
            "my_test_create_bedrock_connector_role",
        )
        mock_save_config.assert_called_once_with(
            {"embedding_model_id": "test-model-id"}
        )

    def test_save_model_id(self):
        mock_config = {}
        mock_save_config = Mock()
        self.bedrock_model.save_model_id(mock_config, mock_save_config, "test-model-id")
        self.assertEqual(mock_config, {"embedding_model_id": "test-model-id"})
        mock_save_config.assert_called_once_with(mock_config)

    @patch("builtins.input", return_value="1")
    def test_get_custom_model_details_default(self, mock_input):
        default_input = {"name": "Default Model"}
        result = self.bedrock_model.get_custom_model_details(default_input)
        self.assertEqual(result, default_input)

    @patch("builtins.input", side_effect=["2", '{"name": "Custom Model"}'])
    def test_get_custom_model_details_custom(self, mock_input):
        default_input = {"name": "Default Model"}
        result = self.bedrock_model.get_custom_model_details(default_input)
        self.assertEqual(result, {"name": "Custom Model"})

    @patch("builtins.input", return_value="2\n{invalid json}")
    def test_get_custom_model_details_invalid_json(self, mock_input):
        default_input = {"name": "Default Model"}
        result = self.bedrock_model.get_custom_model_details(default_input)
        self.assertIsNone(result)

    @patch("builtins.input", return_value="3")
    def test_get_custom_model_details_invalid_choice(self, mock_input):
        default_input = {"name": "Default Model"}
        result = self.bedrock_model.get_custom_model_details(default_input)
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
