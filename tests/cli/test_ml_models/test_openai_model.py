# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

import unittest
from unittest.mock import Mock, patch

from colorama import Fore, Style

from opensearch_py_ml.ml_commons.cli.ml_models.openai_model import OpenAIModel


class TestOpenAIModel(unittest.TestCase):

    def setUp(self):
        self.mock_helper = Mock()
        self.mock_save_config = Mock()
        self.service_type = OpenAIModel.AMAZON_OPENSEARCH_SERVICE
        self.openai_model = OpenAIModel(self.service_type)
        self.connector_role_prefix = "test_role"
        self.api_key = "test_api_key"
        self.connector_secret_name = "test_secret_name"
        self.connector_body = {
            "name": "Custom Model",
            "description": "Custom description",
            "version": "1",
        }

    @patch(
        "opensearch_py_ml.ml_commons.cli.ml_models.model_base.ModelBase.set_trusted_endpoint"
    )
    @patch(
        "opensearch_py_ml.ml_commons.cli.ml_models.model_base.ModelBase.get_model_details"
    )
    def test_create_connector_embedding_model_managed(
        self, mock_get_model_details, mock_set_trusted_endpoint
    ):
        """Test creating an OpenAI connector with embedding model in managed service"""
        # Setup mocks
        mock_get_model_details.return_value = "1"
        self.mock_helper.create_connector_with_secret.return_value = (
            "test_connector_id",
            "test_role_arn",
            "test_secret_arn",
        )

        result = self.openai_model.create_connector(
            helper=self.mock_helper,
            save_config_method=self.mock_save_config,
            connector_role_prefix=self.connector_role_prefix,
            model_name="Embedding model",
            api_key=self.api_key,
            connector_secret_name=self.connector_secret_name,
        )

        # Verify method calls
        mock_set_trusted_endpoint.assert_called_once_with(
            self.mock_helper, "^https://api\\.openai\\.com/.*$"
        )
        mock_get_model_details.assert_called_once_with(
            "OpenAI", OpenAIModel.AMAZON_OPENSEARCH_SERVICE, "Embedding model"
        )
        self.assertTrue(result)

    @patch(
        "opensearch_py_ml.ml_commons.cli.ml_models.model_base.ModelBase.input_custom_model_details"
    )
    @patch(
        "opensearch_py_ml.ml_commons.cli.ml_models.model_base.ModelBase.set_trusted_endpoint"
    )
    @patch(
        "opensearch_py_ml.ml_commons.cli.ml_models.model_base.ModelBase.get_model_details"
    )
    def test_create_connector_custom_model_managed(
        self, mock_get_model_details, mock_set_trusted_endpoint, mock_custom_model
    ):
        """Test creating an OpenAI connector with custom model in managed service"""
        self.mock_helper.create_connector_with_secret.return_value = (
            "test_connector_id",
            "test_role_arn",
            "test_secret_arn",
        )
        mock_custom_model.return_value = self.connector_body

        result = self.openai_model.create_connector(
            helper=self.mock_helper,
            save_config_method=self.mock_save_config,
            connector_role_prefix=self.connector_role_prefix,
            model_name="Custom model",
            api_key=self.api_key,
            connector_secret_name=self.connector_secret_name,
        )

        # Verify method calls
        mock_set_trusted_endpoint.assert_called_once_with(
            self.mock_helper, "^https://api\\.openai\\.com/.*$"
        )
        mock_get_model_details.assert_called_once_with(
            "OpenAI", OpenAIModel.AMAZON_OPENSEARCH_SERVICE, "Custom model"
        )
        mock_custom_model.assert_called_once_with(external=True)
        self.assertTrue(result)

    @patch(
        "opensearch_py_ml.ml_commons.cli.ml_models.model_base.ModelBase.input_custom_model_details"
    )
    @patch(
        "opensearch_py_ml.ml_commons.cli.ml_models.model_base.ModelBase.get_model_details"
    )
    @patch("builtins.print")
    def test_create_connector_invalid_choice_managed(
        self, mock_print, mock_get_model_details, mock_custom_model
    ):
        """Test creating an OpenAI connector with invalid model choice in managed service"""
        self.mock_helper.create_connector_with_secret.return_value = (
            "mock_connector_id",
            "mock_role_arn",
            "test_secret_arn",
        )
        mock_custom_model.return_value = self.connector_body

        self.openai_model.create_connector(
            helper=self.mock_helper,
            save_config_method=self.mock_save_config,
            connector_role_prefix=self.connector_role_prefix,
            model_name="Invalid Model",
            api_key=self.api_key,
            connector_secret_name=self.connector_secret_name,
        )
        mock_print.assert_any_call(
            f"\n{Fore.YELLOW}Invalid choice. Defaulting to 'Custom model'.{Style.RESET_ALL}"
        )
        mock_custom_model.assert_called_once_with(external=True)

    def test_create_connector_failure(self):
        """Test creating an OpenAI connector in failure scenario"""
        self.mock_helper.create_connector_with_secret.return_value = None, None, None
        result = self.openai_model.create_connector(
            helper=self.mock_helper,
            save_config_method=self.mock_save_config,
            connector_role_prefix=self.connector_role_prefix,
            model_name="Embedding model",
            api_key=self.api_key,
            connector_secret_name=self.connector_secret_name,
        )
        self.assertFalse(result)

    @patch(
        "opensearch_py_ml.ml_commons.cli.ml_models.model_base.ModelBase.get_model_details"
    )
    def test_create_connector_chat_model_open_source(self, mock_get_model_details):
        """Test creating an OpenAI connector with chat model in open-source service"""
        # Create model with open-source service type
        open_source_model = OpenAIModel(service_type=OpenAIModel.OPEN_SOURCE)
        mock_get_model_details.return_value = "1"

        result = open_source_model.create_connector(
            helper=self.mock_helper,
            save_config_method=self.mock_save_config,
            connector_role_prefix=self.connector_role_prefix,
            model_name="Chat model",
            api_key=self.api_key,
        )
        # Verify method call
        mock_get_model_details.assert_called_once_with(
            "OpenAI", OpenAIModel.OPEN_SOURCE, "Chat model"
        )
        # Verify that create_connector was called instead of create_connector_with_secret
        self.mock_helper.create_connector.assert_called_once()
        self.assertTrue(result)

    @patch(
        "opensearch_py_ml.ml_commons.cli.ml_models.model_base.ModelBase.get_model_details"
    )
    def test_create_connector_embedding_model_open_source(self, mock_get_model_details):
        """Test creating a OpenAI connector with embedding model in open-source service"""
        # Create model with open-source service type
        open_source_model = OpenAIModel(service_type=OpenAIModel.OPEN_SOURCE)
        mock_get_model_details.return_value = "2"

        result = open_source_model.create_connector(
            helper=self.mock_helper,
            save_config_method=self.mock_save_config,
            connector_role_prefix=self.connector_role_prefix,
            model_name="Embedding model",
            api_key=self.api_key,
        )
        # Verify method call
        mock_get_model_details.assert_called_once_with(
            "OpenAI", OpenAIModel.OPEN_SOURCE, "Embedding model"
        )
        # Verify that create_connector was called instead of create_connector_with_secret
        self.mock_helper.create_connector.assert_called_once()
        self.assertTrue(result)

    @patch(
        "opensearch_py_ml.ml_commons.cli.ml_models.model_base.ModelBase.input_custom_model_details"
    )
    @patch(
        "opensearch_py_ml.ml_commons.cli.ml_models.model_base.ModelBase.get_model_details"
    )
    def test_create_connector_custom_model_open_source(
        self, mock_get_model_details, mock_custom_model
    ):
        """Test creating a OpenAI connector with custom model in open-source service"""
        # Create model with open-source service type
        open_source_model = OpenAIModel(service_type=OpenAIModel.OPEN_SOURCE)
        mock_custom_model.return_value = self.connector_body
        result = open_source_model.create_connector(
            helper=self.mock_helper,
            save_config_method=self.mock_save_config,
            connector_role_prefix=self.connector_role_prefix,
            model_name="Custom model",
            api_key=self.api_key,
        )
        # Verify method call
        mock_get_model_details.assert_called_once_with(
            "OpenAI", OpenAIModel.OPEN_SOURCE, "Custom model"
        )
        # Verify that create_connector was called instead of create_connector_with_secret
        self.mock_helper.create_connector.assert_called_once()
        mock_custom_model.assert_called_once_with(external=True)
        self.assertTrue(result)

    @patch(
        "opensearch_py_ml.ml_commons.cli.ml_models.model_base.ModelBase.input_custom_model_details"
    )
    @patch(
        "opensearch_py_ml.ml_commons.cli.ml_models.model_base.ModelBase.get_model_details"
    )
    @patch("builtins.print")
    def test_create_connector_invalid_choice_open_source(
        self, mock_print, mock_get_model_details, mock_custom_model
    ):
        """Test creating a OpenAI connector with an invalid model choice in open-source service"""
        # Create model with open-source service type
        open_source_model = OpenAIModel(service_type=OpenAIModel.OPEN_SOURCE)
        mock_custom_model.return_value = self.connector_body
        open_source_model.create_connector(
            helper=self.mock_helper,
            save_config_method=self.mock_save_config,
            connector_role_prefix=self.connector_role_prefix,
            model_name="Invalid model",
            api_key=self.api_key,
        )
        mock_print.assert_any_call(
            f"\n{Fore.YELLOW}Invalid choice. Defaulting to 'Custom model'.{Style.RESET_ALL}"
        )
        mock_custom_model.assert_called_once_with(external=True)

    @patch("builtins.input", side_effect=["1"])
    def test_create_connector_select_model_interactive(self, mock_input):
        """Test create_connector for selecting the model through the prompt"""
        self.mock_helper.create_connector_with_secret.return_value = (
            "mock_connector_id",
            "mock_role_arn",
            "test_secret_arn",
        )

        result = self.openai_model.create_connector(
            helper=self.mock_helper,
            save_config_method=self.mock_save_config,
            connector_role_prefix=self.connector_role_prefix,
            api_key=self.api_key,
            connector_secret_name=self.connector_secret_name,
        )

        self.mock_helper.create_connector_with_secret.assert_called_once()
        self.assertTrue(result)


if __name__ == "__main__":
    unittest.main()
