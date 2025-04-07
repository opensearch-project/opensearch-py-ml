# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

import unittest
from unittest.mock import Mock, patch

from colorama import Fore, Style

from opensearch_py_ml.ml_commons.cli.ml_models.AlephAlphaModel import AlephAlphaModel


class TestAlephAlphaModel(unittest.TestCase):

    def setUp(self):
        self.mock_helper = Mock()
        self.mock_save_config = Mock()
        self.aleph_alpha_model = AlephAlphaModel()

    @patch(
        "opensearch_py_ml.ml_commons.cli.ml_models.model_base.ModelBase.set_trusted_endpoint"
    )
    @patch(
        "opensearch_py_ml.ml_commons.cli.ml_models.model_base.ModelBase.get_model_details"
    )
    @patch("opensearch_py_ml.ml_commons.cli.ml_models.model_base.ModelBase.set_api_key")
    def test_create_connector_luminous_base(
        self, mock_set_api_key, mock_get_model_details, mock_set_trusted_endpoint
    ):
        """Test creating an Aleph Alpha connector with Luminous-Base embedding model"""
        # Set mock return values
        mock_get_model_details.return_value = "1"
        mock_set_api_key.return_value = "test_api_key"

        result = self.aleph_alpha_model.create_connector(
            helper=self.mock_helper,
            save_config_method=self.mock_save_config,
            model_name="Luminous-Base embedding model",
            api_key="test_api_key",
        )

        # Verify method calls
        mock_set_trusted_endpoint.assert_called_once_with(
            self.mock_helper, "^https://api\\.aleph-alpha\\.com/.*$"
        )
        mock_get_model_details.assert_called_once_with(
            "Aleph Alpha", "open-source", "Luminous-Base embedding model"
        )
        mock_set_api_key.assert_called_once_with("test_api_key", "Aleph Alpha")
        self.assertTrue(result)

    @patch(
        "opensearch_py_ml.ml_commons.cli.ml_models.model_base.ModelBase.set_trusted_endpoint"
    )
    @patch(
        "opensearch_py_ml.ml_commons.cli.ml_models.model_base.ModelBase.get_model_details"
    )
    @patch("opensearch_py_ml.ml_commons.cli.ml_models.model_base.ModelBase.set_api_key")
    def test_create_connector_custom_model(
        self, mock_set_api_key, mock_get_model_details, mock_set_trusted_endpoint
    ):
        """Test creating an Aleph Alpha connector with custom model"""
        custom_payload = {
            "name": "Custom Model",
            "description": "Custom description",
            "version": "1",
        }

        result = self.aleph_alpha_model.create_connector(
            helper=self.mock_helper,
            save_config_method=self.mock_save_config,
            model_name="Custom model",
            api_key="test_api_key",
            connector_body=custom_payload,
        )

        # Verify method calls
        mock_set_trusted_endpoint.assert_called_once_with(
            self.mock_helper, "^https://api\\.aleph-alpha\\.com/.*$"
        )
        mock_get_model_details.assert_called_once_with(
            "Aleph Alpha", "open-source", "Custom model"
        )
        mock_set_api_key.assert_called_once_with("test_api_key", "Aleph Alpha")
        self.assertTrue(result)

    @patch("builtins.input")
    def test_input_custom_model_details(self, mock_input):
        """Test create_connector for input_custom_model_details method"""
        mock_input.side_effect = [
            '{"name": "test-model",',
            '"description": "test description",',
            '"parameters": {"param": "value"}}',
            "",
        ]
        result = self.aleph_alpha_model.input_custom_model_details()
        expected_result = {
            "name": "test-model",
            "description": "test description",
            "parameters": {"param": "value"},
        }
        self.assertEqual(result, expected_result)

    @patch(
        "opensearch_py_ml.ml_commons.cli.ml_setup.Setup.get_password_with_asterisks",
        return_value="test_api_key",
    )
    @patch("builtins.input", side_effect=["1"])
    def test_create_connector_select_model_interactive(
        self, mock_input, mock_get_password
    ):
        """Test create_connector for selecting the model through the prompt"""
        result = self.aleph_alpha_model.create_connector(
            helper=self.mock_helper, save_config_method=self.mock_save_config
        )
        self.mock_helper.create_connector.assert_called_once()
        self.assertTrue(result)

    @patch(
        "opensearch_py_ml.ml_commons.cli.ml_models.model_base.ModelBase.get_model_details"
    )
    @patch("builtins.print")
    def test_create_connector_invalid_choice(self, mock_print, mock_get_model_details):
        """Test creating an Aleph Alpha connector with invalid model choice"""
        self.aleph_alpha_model.create_connector(
            helper=self.mock_helper,
            save_config_method=self.mock_save_config,
            model_name="Invalid Model",
            api_key="test_api_key",
            connector_body={"name": "test-model"},
        )
        mock_print.assert_any_call(
            f"\n{Fore.YELLOW}Invalid choice. Defaulting to 'Custom model'.{Style.RESET_ALL}"
        )

    def test_create_connector_failure(self):
        """Test creating an Aleph Alpha connector in failure scenario"""
        self.mock_helper.create_connector.return_value = None
        result = self.aleph_alpha_model.create_connector(
            helper=self.mock_helper,
            save_config_method=self.mock_save_config,
            model_name="Luminous-Base embedding model",
            api_key="test_api_key",
        )
        self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()
