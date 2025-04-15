# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

import unittest
from unittest.mock import Mock, patch

from colorama import Fore, Style

from opensearch_py_ml.ml_commons.cli.ml_models.azure_openai_model import (
    AzureOpenAIModel,
)


class TestAzureOpenAIModel(unittest.TestCase):

    def setUp(self):
        self.mock_helper = Mock()
        self.mock_save_config = Mock()
        self.azure_openai_model = AzureOpenAIModel()
        self.api_key = "test_api_key"
        self.resource_name = "test_resource_name"
        self.deployment_name = "test_deployment_name"
        self.api_version = "test_api_version"

    @patch(
        "opensearch_py_ml.ml_commons.cli.ml_models.model_base.ModelBase.set_trusted_endpoint"
    )
    @patch(
        "opensearch_py_ml.ml_commons.cli.ml_models.model_base.ModelBase.get_model_details"
    )
    @patch("opensearch_py_ml.ml_commons.cli.ml_models.model_base.ModelBase.set_api_key")
    def test_create_connector_embedding(
        self, mock_set_api_key, mock_get_model_details, mock_set_trusted_endpoint
    ):
        """Test creating an Azure OpenAI connector with Embedding model"""
        # Set mock return values
        mock_get_model_details.return_value = "1"
        mock_set_api_key.return_value = self.api_key

        result = self.azure_openai_model.create_connector(
            helper=self.mock_helper,
            save_config_method=self.mock_save_config,
            model_name="Embedding model",
            api_key=self.api_key,
            resource_name=self.resource_name,
            deployment_name=self.deployment_name,
            api_version=self.api_version,
        )

        # Verify method calls
        mock_set_trusted_endpoint.assert_called_once_with(
            self.mock_helper, "^https://.*\\.openai\\.azure\\.com/.*$"
        )
        mock_get_model_details.assert_called_once_with(
            "Azure OpenAI", AzureOpenAIModel.OPEN_SOURCE, "Embedding model"
        )
        mock_set_api_key.assert_called_once_with(self.api_key, "OpenAI")
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
    @patch("opensearch_py_ml.ml_commons.cli.ml_models.model_base.ModelBase.set_api_key")
    def test_create_connector_custom_model(
        self,
        mock_set_api_key,
        mock_get_model_details,
        mock_set_trusted_endpoint,
        mock_custom_model,
    ):
        """Test creating an Azure OpenAI connector with custom model"""
        result = self.azure_openai_model.create_connector(
            helper=self.mock_helper,
            save_config_method=self.mock_save_config,
            model_name="Custom model",
            api_key=self.api_key,
            resource_name=self.resource_name,
            deployment_name=self.deployment_name,
            api_version=self.api_version,
        )

        # Verify method calls
        mock_set_trusted_endpoint.assert_called_once_with(
            self.mock_helper, "^https://.*\\.openai\\.azure\\.com/.*$"
        )
        mock_get_model_details.assert_called_once_with(
            "Azure OpenAI", AzureOpenAIModel.OPEN_SOURCE, "Custom model"
        )
        mock_set_api_key.assert_called_once_with("test_api_key", "OpenAI")
        mock_custom_model.assert_called_once()
        self.assertTrue(result)

    @patch(
        "opensearch_py_ml.ml_commons.cli.ml_setup.Setup.get_password_with_asterisks",
        return_value="test_api_key",
    )
    @patch(
        "builtins.input",
        side_effect=[
            "1",
            "test_resource_name",
            "test_deployment_name",
            "test_api_version",
        ],
    )
    def test_create_connector_select_model_interactive(
        self, mock_input, mock_get_password
    ):
        """Test create_connector for selecting the model through the prompt"""
        result = self.azure_openai_model.create_connector(
            helper=self.mock_helper, save_config_method=self.mock_save_config
        )
        self.mock_helper.create_connector.assert_called_once()
        self.assertTrue(result)

    @patch(
        "opensearch_py_ml.ml_commons.cli.ml_models.model_base.ModelBase.input_custom_model_details"
    )
    @patch(
        "opensearch_py_ml.ml_commons.cli.ml_models.model_base.ModelBase.get_model_details"
    )
    @patch("builtins.print")
    def test_create_connector_invalid_choice(
        self, mock_print, mock_get_model_details, mock_custom_model
    ):
        """Test creating an Azure OpenAi connector with invalid model choice"""
        self.azure_openai_model.create_connector(
            helper=self.mock_helper,
            save_config_method=self.mock_save_config,
            model_name="Invalid Model",
            api_key=self.api_key,
            resource_name=self.resource_name,
            deployment_name=self.deployment_name,
            api_version=self.api_version,
        )
        mock_print.assert_any_call(
            f"\n{Fore.YELLOW}Invalid choice. Defaulting to 'Custom model'.{Style.RESET_ALL}"
        )
        mock_custom_model.assert_called_once()

    def test_create_connector_failure(self):
        """Test creating an Azure OpenAI connector in failure scenario"""
        self.mock_helper.create_connector.return_value = None
        result = self.azure_openai_model.create_connector(
            helper=self.mock_helper,
            save_config_method=self.mock_save_config,
            model_name="Embedding model",
            api_key=self.api_key,
            resource_name=self.resource_name,
            deployment_name=self.deployment_name,
            api_version=self.api_version,
        )
        self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()
