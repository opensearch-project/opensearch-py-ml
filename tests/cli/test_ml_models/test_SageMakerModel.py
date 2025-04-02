# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

import unittest
from unittest.mock import Mock, patch

from colorama import Fore, Style

from opensearch_py_ml.ml_commons.cli.ml_models.SageMakerModel import SageMakerModel


class TestSageMakerModel(unittest.TestCase):

    def setUp(self):
        self.region = "us-west-2"
        self.mock_helper = Mock()
        self.mock_save_config = Mock()
        self.sagemaker_model = SageMakerModel(opensearch_domain_region=self.region)
        self.connector_role_prefix = "test_role"
        self.connector_endpoint_arn = "test_arn"
        self.connector_endpoint_url = "test_url"

    @patch("opensearch_py_ml.ml_commons.cli.ml_models.SageMakerModel.uuid")
    def test_create_sagemaker_connector_embedding(self, mock_uuid):
        """Test creating a SageMaker connector with embedding model"""
        mock_uuid.uuid1.return_value = Mock(__str__=lambda _: "12345678" * 4)
        self.mock_helper.create_connector_with_role.return_value = (
            "test_connector_id",
            "test_role_arn",
        )

        result = self.sagemaker_model.create_sagemaker_connector(
            helper=self.mock_helper,
            save_config_method=self.mock_save_config,
            connector_role_prefix=self.connector_role_prefix,
            region=self.region,
            model_name="Embedding model",
            endpoint_arn=self.connector_endpoint_arn,
            endpoint_url=self.connector_endpoint_url,
        )
        # Verify settings were set correctly
        self.mock_helper.opensearch_client.cluster.put_settings.assert_called_once()

        # Verify connector creation was called with correct parameters
        self.mock_helper.create_connector_with_role.assert_called_once()
        call_args = self.mock_helper.create_connector_with_role.call_args[0]

        # Verify role names
        self.assertEqual(
            call_args[1], f"{self.connector_role_prefix}_sagemaker_connector_12345678"
        )
        self.assertEqual(
            call_args[2],
            f"{self.connector_role_prefix}_sagemaker_connector_create_12345678",
        )

        # Verify connector payload
        connector_payload = call_args[3]
        self.assertEqual(
            connector_payload["name"], "SageMaker Embedding Model Connector"
        )
        self.assertEqual(connector_payload["protocol"], "aws_sigv4")
        self.assertEqual(connector_payload["parameters"]["region"], self.region)

        self.assertTrue(result)

    @patch("opensearch_py_ml.ml_commons.cli.ml_models.SageMakerModel.uuid")
    def test_create_sagemaker_connector_deepseek(self, mock_uuid):
        """Test creating a SageMaker connector with DeepSeek R1 model"""
        mock_uuid.uuid1.return_value = Mock(__str__=lambda _: "12345678" * 4)
        self.mock_helper.create_connector_with_role.return_value = (
            "test_connector_id",
            "test_role_arn",
        )

        result = self.sagemaker_model.create_sagemaker_connector(
            helper=self.mock_helper,
            save_config_method=self.mock_save_config,
            connector_role_prefix=self.connector_role_prefix,
            region=self.region,
            model_name="DeepSeek R1 model",
            endpoint_arn=self.connector_endpoint_arn,
            endpoint_url=self.connector_endpoint_url,
        )
        # Verify settings were set correctly
        self.mock_helper.opensearch_client.cluster.put_settings.assert_called_once()

        # Verify connector creation was called with correct parameters
        self.mock_helper.create_connector_with_role.assert_called_once()
        call_args = self.mock_helper.create_connector_with_role.call_args[0]

        # Verify role names
        self.assertEqual(
            call_args[1], f"{self.connector_role_prefix}_sagemaker_connector_12345678"
        )
        self.assertEqual(
            call_args[2],
            f"{self.connector_role_prefix}_sagemaker_connector_create_12345678",
        )

        # Verify connector payload
        connector_payload = call_args[3]
        self.assertEqual(connector_payload["name"], "DeepSeek R1 model connector")
        self.assertEqual(connector_payload["protocol"], "aws_sigv4")
        self.assertEqual(connector_payload["parameters"]["region"], self.region)

        self.assertTrue(result)

    @patch("opensearch_py_ml.ml_commons.cli.ml_models.SageMakerModel.uuid")
    def test_create_sagemaker_connector_custom(self, mock_uuid):
        """Test creating a SageMaker connector with custom model"""
        mock_uuid.uuid1.return_value = Mock(__str__=lambda _: "12345678" * 4)
        self.mock_helper.create_connector_with_role.return_value = (
            "test_connector_id",
            "test_role_arn",
        )

        connector_payload = {
            "name": "Custom SageMaker Connector",
            "description": "Test custom connector",
            "version": "1.0",
            "protocol": "aws_sigv4",
            "parameters": {"service_name": "sagemaker", "region": self.region},
            "actions": [
                {
                    "action_type": "predict",
                    "method": "POST",
                    "url": self.connector_endpoint_url,
                    "headers": {"Content-Type": "application/json"},
                    "request_body": "${parameters.input}",
                }
            ],
        }

        result = self.sagemaker_model.create_sagemaker_connector(
            helper=self.mock_helper,
            save_config_method=self.mock_save_config,
            connector_role_prefix=self.connector_role_prefix,
            region=self.region,
            model_name="Custom model",
            endpoint_arn=self.connector_endpoint_arn,
            endpoint_url=self.connector_endpoint_url,
            connector_payload=connector_payload,
        )

        call_args = self.mock_helper.create_connector_with_role.call_args[0]
        # Verify role names
        self.assertEqual(
            call_args[1], f"{self.connector_role_prefix}_sagemaker_connector_12345678"
        )
        self.assertEqual(
            call_args[2],
            f"{self.connector_role_prefix}_sagemaker_connector_create_12345678",
        )
        self.assertTrue(result)

    @patch("builtins.input", side_effect=["1"])
    def test_create_sagemaker_connector_select_model_interactive(self, mock_input):
        """Test create_sagemaker_connector for selecting the model through the prompt"""
        self.mock_helper.create_connector_with_role.return_value = (
            "mock_connector_id",
            "mock_role_arn",
        )

        result = self.sagemaker_model.create_sagemaker_connector(
            helper=self.mock_helper,
            save_config_method=self.mock_save_config,
            connector_role_prefix=self.connector_role_prefix,
            region=self.region,
            endpoint_arn=self.connector_endpoint_arn,
            endpoint_url=self.connector_endpoint_url,
        )
        self.mock_helper.create_connector_with_role.assert_called_once()
        self.assertTrue(result)

    @patch("builtins.input", side_effect=["test_prefix"])
    def test_valid_connector_role_prefix(self, mock_input):
        """Test creating a SageMaker connector with a valid connector role prefix"""
        self.mock_helper.create_connector_with_role.return_value = (
            "mock_connector_id",
            "mock_role_arn",
        )
        self.sagemaker_model.create_sagemaker_connector(
            helper=self.mock_helper,
            save_config_method=self.mock_save_config,
            region=self.region,
            model_name="Embedding model",
            endpoint_arn=self.connector_endpoint_arn,
            endpoint_url=self.connector_endpoint_url,
        )
        mock_input.assert_any_call("Enter your connector role prefix: ")
        create_connector_calls = (
            self.mock_helper.create_connector_with_role.call_args_list
        )
        _, connector_role_name, create_connector_role_name, _ = create_connector_calls[
            0
        ][0]
        self.assertTrue(
            connector_role_name.startswith("test_prefix_sagemaker_connector_")
        )
        self.assertTrue(
            create_connector_role_name.startswith(
                "test_prefix_sagemaker_connector_create_"
            )
        )

    @patch("builtins.input", side_effect=[""])
    def test_invalid_connector_role_prefix(self, mock_input):
        """Test creating a SageMaker connector with an invalid connector role prefix"""
        self.mock_helper.create_connector_with_role.return_value = (
            "mock_connector_id",
            "mock_role_arn",
        )
        with self.assertRaises(ValueError) as context:
            self.sagemaker_model.create_sagemaker_connector(
                helper=self.mock_helper,
                save_config_method=self.mock_save_config,
                region=self.region,
                model_name="Embedding model",
                endpoint_arn=self.connector_endpoint_arn,
                endpoint_url=self.connector_endpoint_url,
            )
        self.assertEqual(
            str(context.exception), "Connector role prefix cannot be empty."
        )
        mock_input.assert_any_call("Enter your connector role prefix: ")

    @patch("builtins.input", side_effect=["test-endpoint-arn"])
    def test_create_sagemaker_connector_endpoint_arn(self, mock_input):
        """Test creating a SageMaker connector when user provides an endpoint ARN through the prompt"""
        self.mock_helper.create_connector_with_role.return_value = (
            "mock_connector_id",
            "mock_role_arn",
        )
        self.sagemaker_model.create_sagemaker_connector(
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

    @patch("builtins.input", side_effect=["test-endpoint-url"])
    def test_create_sagemaker_connector_endpoint_url(self, mock_input):
        """Test creating a SageMaker connector when user provides an endpoint URL through the prompt"""
        self.mock_helper.create_connector_with_role.return_value = (
            "mock_connector_id",
            "mock_role_arn",
        )
        self.sagemaker_model.create_sagemaker_connector(
            helper=self.mock_helper,
            save_config_method=self.mock_save_config,
            connector_role_prefix=self.connector_role_prefix,
            region=self.region,
            model_name="Embedding model",
            endpoint_arn=self.connector_endpoint_arn,
        )
        mock_input.assert_any_call("Enter your SageMaker inference endpoint URL: ")
        _, _, _, connector_payload = (
            self.mock_helper.create_connector_with_role.call_args[0]
        )
        self.assertEqual(connector_payload["actions"][0]["url"], "test-endpoint-url")

    @patch("builtins.input", side_effect=[""])
    def test_create_sagemaker_connector_default_region(self, mock_input):
        """Test creating a SageMaker connector with default region"""
        self.mock_helper.create_connector_with_role.return_value = (
            "mock_connector_id",
            "mock_role_arn",
        )
        self.sagemaker_model.create_sagemaker_connector(
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
        _, _, _, connector_payload = create_connector_calls[0][0]
        self.assertEqual(connector_payload["parameters"]["region"], "us-west-2")

    @patch("builtins.input", side_effect=["us-east-1"])
    def test_create_sagemaker_connector_custom_region(self, mock_input):
        """Test creating a SageMaker connector with a custom region"""
        self.mock_helper.create_connector_with_role.return_value = (
            "mock_connector_id",
            "mock_role_arn",
        )
        self.sagemaker_model.create_sagemaker_connector(
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
        _, _, _, connector_payload = create_connector_calls[0][0]
        self.assertEqual(connector_payload["parameters"]["region"], "us-east-1")

    @patch("builtins.input")
    def test_input_custom_model_details(self, mock_input):
        """Test create_sagemaker_connector for input_custom_model_details method"""
        mock_input.side_effect = [
            '{"name": "test-model",',
            '"description": "test description",',
            '"parameters": {"param": "value"}}',
            "",
        ]
        result = self.sagemaker_model.input_custom_model_details()
        expected_result = {
            "name": "test-model",
            "description": "test description",
            "parameters": {"param": "value"},
        }
        self.assertEqual(result, expected_result)

    @patch("builtins.print")
    @patch("builtins.input")
    def test_create_sagemaker_connector_invalid_choice(self, mock_input, mock_print):
        """Test creating a SageMaker connector with invalid model choice"""
        self.mock_helper.create_connector_with_role.return_value = (
            "mock_connector_id",
            "mock_role_arn",
        )
        mock_input.side_effect = ['{"name": "test-model"}', ""]
        self.sagemaker_model.create_sagemaker_connector(
            helper=self.mock_helper,
            save_config_method=self.mock_save_config,
            connector_role_prefix=self.connector_role_prefix,
            region=self.region,
            model_name="Invalid Model",
            endpoint_arn=self.connector_endpoint_arn,
            endpoint_url=self.connector_endpoint_url,
        )

        mock_print.assert_any_call(
            f"\n{Fore.YELLOW}Invalid choice. Defaulting to 'Custom model'.{Style.RESET_ALL}"
        )
        self.mock_helper.create_connector_with_role.assert_called_once()

    def test_create_sagemaker_connector_failure(self):
        """Test creating a SageMaker connector in failure scenario"""
        self.mock_helper.create_connector_with_role.return_value = None, None

        result = self.sagemaker_model.create_sagemaker_connector(
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
