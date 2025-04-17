# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

import unittest
from unittest.mock import Mock, patch

from colorama import Fore, Style

from opensearch_py_ml.ml_commons.cli.ml_models.comprehend_model import ComprehendModel


class TestComprehendModel(unittest.TestCase):

    def setUp(self):
        self.region = "us-west-2"
        self.service_type = ComprehendModel.AMAZON_OPENSEARCH_SERVICE
        self.mock_helper = Mock()
        self.mock_save_config = Mock()
        self.comprehend_model = ComprehendModel(
            opensearch_domain_region=self.region, service_type=self.service_type
        )
        self.connector_role_prefix = "test_role"
        self.model_arn = "test_model_arn"
        self.connector_body = {
            "name": "Custom Model",
            "description": "Custom description",
            "version": "1",
            "parameters": {"api_name": "test_api_name"},
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
        """Test creating a Comprehend connector with embedding model in managed service"""
        # Setup mocks
        mock_get_model_details.return_value = "1"
        self.mock_helper.create_connector_with_role.return_value = (
            "test_connector_id",
            "test_role_arn",
            "",
        )

        result = self.comprehend_model.create_connector(
            helper=self.mock_helper,
            save_config_method=self.mock_save_config,
            connector_role_prefix=self.connector_role_prefix,
            region=self.region,
            model_name="Metadata embedding model",
            model_arn=self.model_arn,
        )

        # Verify method calls
        mock_set_trusted_endpoint.assert_called_once_with(
            self.mock_helper, "^https://comprehend\\..*[a-z0-9-]\\.amazonaws\\.com$"
        )
        mock_get_model_details.assert_called_once_with(
            "Amazon Comprehend",
            ComprehendModel.AMAZON_OPENSEARCH_SERVICE,
            "Metadata embedding model",
        )
        self.assertTrue(result)

    @patch(
        "opensearch_py_ml.ml_commons.cli.ml_models.model_base.ModelBase.input_custom_model_details"
    )
    @patch(
        "opensearch_py_ml.ml_commons.cli.ml_models.model_base.ModelBase.get_model_details"
    )
    @patch(
        "opensearch_py_ml.ml_commons.cli.ml_setup.Setup.get_password_with_asterisks",
        return_value=["access_key", "secret_key", "session_token"],
    )
    def test_create_connector_custom_model_open_source(
        self, mock_get_password, mock_get_model_details, mock_custom_model
    ):
        """Test creating a Comprehend connector with custom model in open-source service"""
        # Create model with open-source service type
        open_source_model = ComprehendModel(
            opensearch_domain_region=self.region,
            service_type=ComprehendModel.OPEN_SOURCE,
        )
        mock_custom_model.return_value = self.connector_body

        result = open_source_model.create_connector(
            helper=self.mock_helper,
            save_config_method=self.mock_save_config,
            connector_role_prefix=self.connector_role_prefix,
            region=self.region,
            model_name="Custom model",
            model_arn=self.model_arn,
        )

        # Verify method calls
        mock_get_model_details.assert_called_once_with(
            "Amazon Comprehend", ComprehendModel.OPEN_SOURCE, "Custom model"
        )
        mock_custom_model.assert_called_once_with()
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
        """Test creating a Comprehend connector with an invalid model choice in managed service"""
        # Setup mocks
        self.mock_helper.create_connector_with_role.return_value = (
            "test_connector_id",
            "test_role_arn",
            "",
        )
        mock_custom_model.return_value = self.connector_body

        self.comprehend_model.create_connector(
            helper=self.mock_helper,
            save_config_method=self.mock_save_config,
            connector_role_prefix=self.connector_role_prefix,
            region=self.region,
            model_name="Invalid Model",
            model_arn=self.model_arn,
        )
        mock_print.assert_any_call(
            f"\n{Fore.YELLOW}Invalid choice. Defaulting to 'Custom model'.{Style.RESET_ALL}"
        )
        mock_custom_model.assert_called_once_with()

    def test_create_connector_failure(self):
        """Test creating a Comprehend connector in failure scenario"""
        self.mock_helper.create_connector_with_role.return_value = None, None, None
        result = self.comprehend_model.create_connector(
            helper=self.mock_helper,
            save_config_method=self.mock_save_config,
            region=self.region,
            connector_role_prefix=self.connector_role_prefix,
            model_name="Metadata embedding model",
            model_arn=self.model_arn,
        )
        self.assertFalse(result)

    @patch("builtins.input", side_effect=["1"])
    def test_create_connector_select_model_interactive(self, mock_input):
        """Test create_connector for selecting the model through the prompt"""
        # Setup mock
        self.mock_helper.create_connector_with_role.return_value = (
            "test_connector_id",
            "test_role_arn",
            "",
        )

        result = self.comprehend_model.create_connector(
            helper=self.mock_helper,
            save_config_method=self.mock_save_config,
            connector_role_prefix=self.connector_role_prefix,
            region=self.region,
            model_arn=self.model_arn,
        )

        self.mock_helper.create_connector_with_role.assert_called_once()
        self.assertTrue(result)


if __name__ == "__main__":
    unittest.main()
