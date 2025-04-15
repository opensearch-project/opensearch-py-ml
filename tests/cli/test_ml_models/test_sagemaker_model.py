# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

import unittest
from unittest.mock import Mock, patch

from colorama import Fore, Style

from opensearch_py_ml.ml_commons.cli.ml_models.sagemaker_model import SageMakerModel


class TestSageMakerModel(unittest.TestCase):

    def setUp(self):
        self.region = "us-west-2"
        self.mock_helper = Mock()
        self.mock_save_config = Mock()
        self.sagemaker_model = SageMakerModel(opensearch_domain_region=self.region)
        self.connector_role_prefix = "test_role"
        self.connector_endpoint_arn = "test_arn"
        self.connector_endpoint_url = "test_url"

    @patch(
        "opensearch_py_ml.ml_commons.cli.ml_models.model_base.ModelBase.set_trusted_endpoint"
    )
    @patch(
        "opensearch_py_ml.ml_commons.cli.ml_models.model_base.ModelBase.get_model_details"
    )
    @patch("builtins.input", side_effect=["test-url", ""])
    @patch("builtins.print")
    def test_create_connector_deepseek(
        self, mock_print, mock_input, mock_get_model_details, mock_set_trusted_endpoint
    ):
        """Test creating a SageMaker connector with DeepSeek R1 model"""
        # Setup mocks
        mock_get_model_details.return_value = "1"
        self.mock_helper.create_connector_with_role.return_value = (
            "test_connector_id",
            "test_role_arn",
            "",
        )
        result = self.sagemaker_model.create_connector(
            helper=self.mock_helper,
            save_config_method=self.mock_save_config,
            connector_role_prefix=self.connector_role_prefix,
            model_name="DeepSeek R1 model",
            endpoint_arn=self.connector_endpoint_arn,
        )

        # Verify method calls
        mock_set_trusted_endpoint.assert_called_once_with(
            self.mock_helper,
            "^https://runtime\\.sagemaker\\..*[a-z0-9-]\\.amazonaws\\.com/.*$",
        )
        mock_get_model_details.assert_called_once_with(
            "Amazon SageMaker", "amazon-opensearch-service", "DeepSeek R1 model"
        )
        mock_input.assert_any_call("Enter your SageMaker inference endpoint URL: ")
        mock_input.assert_any_call(f"Enter your SageMaker region [{self.region}]: ")
        self.assertTrue(result)

    @patch(
        "opensearch_py_ml.ml_commons.cli.ml_models.model_base.ModelBase.set_trusted_endpoint"
    )
    @patch(
        "opensearch_py_ml.ml_commons.cli.ml_models.model_base.ModelBase.get_model_details"
    )
    @patch("builtins.input", side_effect=["test-url", ""])
    @patch("builtins.print")
    def test_create_connector_embedding(
        self, mock_print, mock_input, mock_get_model_details, mock_set_trusted_endpoint
    ):
        """Test creating a SageMaker connector with embedding model"""
        # Setup mocks
        mock_get_model_details.return_value = "2"
        self.mock_helper.create_connector_with_role.return_value = (
            "test_connector_id",
            "test_role_arn",
            "",
        )
        result = self.sagemaker_model.create_connector(
            helper=self.mock_helper,
            save_config_method=self.mock_save_config,
            connector_role_prefix=self.connector_role_prefix,
            region=self.region,
            model_name="Embedding model",
            endpoint_arn=self.connector_endpoint_arn,
        )

        # Verify method cals
        mock_set_trusted_endpoint.assert_called_once_with(
            self.mock_helper,
            "^https://runtime\\.sagemaker\\..*[a-z0-9-]\\.amazonaws\\.com/.*$",
        )
        mock_get_model_details.assert_called_once_with(
            "Amazon SageMaker", "amazon-opensearch-service", "Embedding model"
        )
        mock_input.assert_any_call("Enter your SageMaker inference endpoint URL: ")
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
        """Test creating a SageMaker connector with custom model"""
        self.mock_helper.create_connector_with_role.return_value = (
            "test_connector_id",
            "test_role_arn",
            "",
        )
        mock_custom_model.return_value = {
            "name": "Custom Model",
            "description": "Custom description",
            "version": "1",
        }

        result = self.sagemaker_model.create_connector(
            helper=self.mock_helper,
            save_config_method=self.mock_save_config,
            connector_role_prefix=self.connector_role_prefix,
            region=self.region,
            model_name="Custom model",
            endpoint_arn=self.connector_endpoint_arn,
            endpoint_url=self.connector_endpoint_url,
        )

        # Verify method calls
        mock_set_trusted_endpoint.assert_called_once_with(
            self.mock_helper,
            "^https://runtime\\.sagemaker\\..*[a-z0-9-]\\.amazonaws\\.com/.*$",
        )
        mock_get_model_details.assert_called_once_with(
            "Amazon SageMaker", "amazon-opensearch-service", "Custom model"
        )
        mock_custom_model.assert_called_once()
        self.assertTrue(result)

    @patch("builtins.input", side_effect=["1"])
    def test_create_connector_select_model_interactive(self, mock_input):
        """Test create_connector for selecting the model through the prompt"""
        self.mock_helper.create_connector_with_role.return_value = (
            "test_connector_id",
            "test_role_arn",
            "",
        )

        result = self.sagemaker_model.create_connector(
            helper=self.mock_helper,
            save_config_method=self.mock_save_config,
            connector_role_prefix=self.connector_role_prefix,
            region=self.region,
            endpoint_arn=self.connector_endpoint_arn,
            endpoint_url=self.connector_endpoint_url,
        )
        self.mock_helper.create_connector_with_role.assert_called_once()
        self.assertTrue(result)

    @patch("builtins.input", side_effect=["test-endpoint-arn"])
    def test_create_connector_endpoint_arn(self, mock_input):
        """Test creating a SageMaker connector when user provides an endpoint ARN through the prompt"""
        self.mock_helper.create_connector_with_role.return_value = (
            "test_connector_id",
            "test_role_arn",
            "",
        )
        self.sagemaker_model.create_connector(
            helper=self.mock_helper,
            save_config_method=self.mock_save_config,
            connector_role_prefix=self.connector_role_prefix,
            region=self.region,
            model_name="Embedding model",
            endpoint_url=self.connector_endpoint_url,
        )
        mock_input.assert_any_call("Enter your SageMaker inference endpoint ARN: ")
        connector_role_inline_policy = (
            self.mock_helper.create_connector_with_role.call_args[0][0]
        )
        self.assertEqual(
            connector_role_inline_policy["Statement"][0]["Resource"],
            "test-endpoint-arn",
        )

    @patch("builtins.input", side_effect=[""])
    def test_create_connector_default_region(self, mock_input):
        """Test creating a SageMaker connector with default region"""
        self.mock_helper.create_connector_with_role.return_value = (
            "test_connector_id",
            "test_role_arn",
            "",
        )
        self.sagemaker_model.create_connector(
            helper=self.mock_helper,
            save_config_method=self.mock_save_config,
            connector_role_prefix=self.connector_role_prefix,
            model_name="Embedding model",
            endpoint_arn=self.connector_endpoint_arn,
            endpoint_url=self.connector_endpoint_url,
        )
        mock_input.assert_any_call(f"Enter your SageMaker region [{self.region}]: ")
        create_connector_calls = (
            self.mock_helper.create_connector_with_role.call_args_list
        )
        self.assertEqual(len(create_connector_calls), 1)
        _, _, _, connector_body = create_connector_calls[0][0]
        self.assertEqual(connector_body["parameters"]["region"], "us-west-2")

    @patch("builtins.input", side_effect=["us-east-1"])
    def test_create_connector_custom_region(self, mock_input):
        """Test creating a SageMaker connector with a custom region"""
        self.mock_helper.create_connector_with_role.return_value = (
            "test_connector_id",
            "test_role_arn",
            "",
        )
        self.sagemaker_model.create_connector(
            helper=self.mock_helper,
            save_config_method=self.mock_save_config,
            connector_role_prefix=self.connector_role_prefix,
            model_name="Embedding model",
            endpoint_arn=self.connector_endpoint_arn,
            endpoint_url=self.connector_endpoint_url,
        )
        mock_input.assert_any_call(f"Enter your SageMaker region [{self.region}]: ")
        create_connector_calls = (
            self.mock_helper.create_connector_with_role.call_args_list
        )
        self.assertEqual(len(create_connector_calls), 1)
        _, _, _, connector_body = create_connector_calls[0][0]
        self.assertEqual(connector_body["parameters"]["region"], "us-east-1")

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
        """Test creating a SageMaker connector with invalid model choice"""
        self.mock_helper.create_connector_with_role.return_value = (
            "test_connector_id",
            "test_role_arn",
            "",
        )
        mock_custom_model.return_value = {
            "name": "Custom Model",
            "description": "Custom description",
            "version": "1",
        }
        self.sagemaker_model.create_connector(
            helper=self.mock_helper,
            save_config_method=self.mock_save_config,
            connector_role_prefix=self.connector_role_prefix,
            region=self.region,
            model_name="Invalid Model",
            endpoint_arn=self.connector_endpoint_arn,
            endpoint_url=self.connector_endpoint_url,
        )
        mock_custom_model.assert_called_once()
        mock_print.assert_any_call(
            f"\n{Fore.YELLOW}Invalid choice. Defaulting to 'Custom model'.{Style.RESET_ALL}"
        )

    def test_create_connector_failure(self):
        """Test creating a SageMaker connector in failure scenario"""
        self.mock_helper.create_connector_with_role.return_value = None, None, None
        result = self.sagemaker_model.create_connector(
            helper=self.mock_helper,
            save_config_method=self.mock_save_config,
            connector_role_prefix=self.connector_role_prefix,
            region=self.region,
            model_name="Embedding model",
            endpoint_arn=self.connector_endpoint_arn,
            endpoint_url=self.connector_endpoint_url,
        )
        self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()
