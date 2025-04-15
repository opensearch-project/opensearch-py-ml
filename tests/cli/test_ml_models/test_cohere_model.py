# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

import unittest
from unittest.mock import Mock, patch

from colorama import Fore, Style

from opensearch_py_ml.ml_commons.cli.ml_models.cohere_model import CohereModel


class TestCohereModel(unittest.TestCase):

    def setUp(self):
        self.mock_helper = Mock()
        self.mock_save_config = Mock()
        self.service_type = CohereModel.AMAZON_OPENSEARCH_SERVICE
        self.cohere_model = CohereModel(service_type=self.service_type)
        self.connector_role_prefix = "test_role"
        self.api_key = "test_api_key"
        self.secret_name = "test_secret_name"

    @patch(
        "opensearch_py_ml.ml_commons.cli.ml_models.model_base.ModelBase.set_trusted_endpoint"
    )
    @patch(
        "opensearch_py_ml.ml_commons.cli.ml_models.model_base.ModelBase.get_model_details"
    )
    def test_create_connector_embedding_model_managed(
        self, mock_get_model_details, mock_set_trusted_endpoint
    ):
        """Test creating a Cohere connector with Embedding model in managed service"""
        # Setup mocks
        mock_get_model_details.return_value = "1"
        self.mock_helper.create_connector_with_secret.return_value = (
            "test_connector_id",
            "test_role_arn",
            "test_secret_arn",
        )

        result = self.cohere_model.create_connector(
            helper=self.mock_helper,
            save_config_method=self.mock_save_config,
            connector_role_prefix=self.connector_role_prefix,
            model_name="Embedding model",
            api_key=self.api_key,
            secret_name=self.secret_name,
        )

        # Verify method calls
        mock_set_trusted_endpoint.assert_called_once_with(
            self.mock_helper, "^https://api\\.cohere\\.ai/.*$"
        )
        mock_get_model_details.assert_called_once_with(
            "Cohere", CohereModel.AMAZON_OPENSEARCH_SERVICE, "Embedding model"
        )
        self.assertTrue(result)

    @patch(
        "opensearch_py_ml.ml_commons.cli.ml_models.model_base.ModelBase.get_model_details"
    )
    def test_create_connector_chat_model_open_source(self, mock_get_model_details):
        """Test creating a Cohere connector with Chat model in open-source service"""
        # Create model with open-source service type
        open_source_model = CohereModel(service_type=CohereModel.OPEN_SOURCE)
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
            "Cohere", CohereModel.OPEN_SOURCE, "Chat model"
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
    def test_create_connector_custom_model(
        self, mock_get_model_details, mock_set_trusted_endpoint, mock_custom_model
    ):
        """Test creating a Cohere connector with custom model"""
        self.mock_helper.create_connector_with_secret.return_value = (
            "test_connector_id",
            "test_role_arn",
            "test_secret_arn",
        )
        mock_custom_model.return_value = {
            "name": "Custom Model",
            "description": "Custom description",
            "version": "1",
        }

        result = self.cohere_model.create_connector(
            helper=self.mock_helper,
            save_config_method=self.mock_save_config,
            connector_role_prefix=self.connector_role_prefix,
            model_name="Custom model",
            api_key=self.api_key,
            secret_name=self.secret_name,
        )

        # Verify method calls
        mock_set_trusted_endpoint.assert_called_once_with(
            self.mock_helper, "^https://api\\.cohere\\.ai/.*$"
        )
        mock_get_model_details.assert_called_once_with(
            "Cohere", CohereModel.AMAZON_OPENSEARCH_SERVICE, "Custom model"
        )
        mock_custom_model.assert_called_once_with(external=True)
        self.assertTrue(result)

    @patch("builtins.input", side_effect=["1"])
    def test_create_connector_select_model_interactive(self, mock_input):
        """Test create_connector for selecting the model through the prompt"""
        self.mock_helper.create_connector_with_secret.return_value = (
            "mock_connector_id",
            "mock_role_arn",
            "test_secret_arn",
        )

        result = self.cohere_model.create_connector(
            helper=self.mock_helper,
            save_config_method=self.mock_save_config,
            connector_role_prefix=self.connector_role_prefix,
            api_key=self.api_key,
            secret_name=self.secret_name,
        )

        self.mock_helper.create_connector_with_secret.assert_called_once()
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
        """Test creating a Cohere connector with an invalid model choice"""
        self.mock_helper.create_connector_with_secret.return_value = (
            "mock_connector_id",
            "mock_role_arn",
            "test_secret_arn",
        )
        mock_custom_model.return_value = {
            "name": "Custom Model",
            "description": "Custom description",
            "version": "1",
        }
        self.cohere_model.create_connector(
            helper=self.mock_helper,
            save_config_method=self.mock_save_config,
            connector_role_prefix=self.connector_role_prefix,
            model_name="Invalid Model",
            api_key=self.api_key,
            secret_name=self.secret_name,
        )
        mock_print.assert_any_call(
            f"\n{Fore.YELLOW}Invalid choice. Defaulting to 'Custom model'.{Style.RESET_ALL}"
        )
        mock_custom_model.assert_called_once_with(external=True)

    def test_create_connector_failure(self):
        """Test creating a Cohere connector in failure scenario"""
        self.mock_helper.create_connector_with_secret.return_value = None, None, None
        result = self.cohere_model.create_connector(
            helper=self.mock_helper,
            save_config_method=self.mock_save_config,
            connector_role_prefix=self.connector_role_prefix,
            model_name="Embedding model",
            api_key=self.api_key,
            secret_name=self.secret_name,
        )
        self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()
