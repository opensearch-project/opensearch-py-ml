# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

import unittest
from unittest.mock import MagicMock, call, patch

from colorama import Fore, Style

from opensearch_py_ml.ml_commons.cli.connector_create import Create


class TestCreate(unittest.TestCase):
    def setUp(self):
        self.create_instance = Create()
        self.mock_config = {
            "service_type": "amazon-opensearch-service",
            "opensearch_config": {
                "opensearch_domain_region": "us-west-2",
                "opensearch_domain_endpoint": "https://domain.amazonaws.com",
                "opensearch_domain_name": "domain",
                "opensearch_domain_username": "admin",
                "opensearch_domain_password": "password",
            },
            "aws_credentials": {
                "aws_user_name": "test-user-arn",
                "aws_role_name": "test-role-arn",
                "aws_access_key": "test-access-key",
                "aws_secret_access_key": "test-secret-access-key",
                "aws_session_token": "test-session-token",
            },
        }

    @patch("opensearch_py_ml.ml_commons.cli.connector_create.BedrockModel")
    @patch("opensearch_py_ml.ml_commons.cli.connector_create.AIConnectorHelper")
    def test_create_command_managed_bedrock_with_connector_config(
        self, mock_ai_helper, mock_bedrock
    ):
        """Test create_command for creating a Bedrock connector with a connector configuration"""
        self.mock_connector_config = {
            "setup_config_path": "test_setup_config.yml",
            "connector_name": "Amazon Bedrock",
            "model_name": "test_model",
            "region": "us-west-2",
            "connector_role_prefix": "test_prefix",
        }

        self.mock_config["service_type"] = "amazon-opensearch-service"
        self.create_instance.config = self.mock_config
        mock_config_path = "test_config.yml"

        self.create_instance.load_config = MagicMock(return_value=self.mock_config)
        self.create_instance.load_connector_config = MagicMock(
            return_value=self.mock_connector_config
        )
        self.create_instance.update_config = MagicMock()

        mock_bedrock_instance = mock_bedrock.return_value
        mock_bedrock_instance.create_bedrock_connector = MagicMock(return_value=True)

        mock_helper_instance = mock_ai_helper.return_value

        result = self.create_instance.create_command(
            connector_config_path=mock_config_path
        )
        self.assertTrue(result)
        mock_bedrock.assert_called_once_with(
            opensearch_domain_region=self.mock_config["opensearch_config"][
                "opensearch_domain_region"
            ],
        )

        mock_bedrock_instance.create_bedrock_connector.assert_called_once_with(
            mock_helper_instance,
            self.create_instance.connector_output,
            connector_role_prefix=self.mock_connector_config["connector_role_prefix"],
            region=self.mock_connector_config["region"],
            model_name=self.mock_connector_config["model_name"],
            model_arn=None,
            connector_payload=None,
        )
        mock_ai_helper.assert_called_once()

    @patch("opensearch_py_ml.ml_commons.cli.connector_create.SageMakerModel")
    @patch("opensearch_py_ml.ml_commons.cli.connector_create.AIConnectorHelper")
    def test_create_command_managed_sagemaker_with_connector_config(
        self, mock_ai_helper, mock_sagemaker
    ):
        """Test create_command for creating a SageMaker connector with a connector configuration"""
        self.mock_connector_config = {
            "setup_config_path": "test_setup_config.yml",
            "connector_name": "Amazon SageMaker",
            "model_name": "test_model",
            "region": "us-west-2",
            "connector_role_prefix": "test_prefix",
        }

        self.mock_config["service_type"] = "amazon-opensearch-service"
        self.create_instance.config = self.mock_config
        mock_config_path = "test_config.yml"

        self.create_instance.load_config = MagicMock(return_value=self.mock_config)
        self.create_instance.load_connector_config = MagicMock(
            return_value=self.mock_connector_config
        )
        self.create_instance.update_config = MagicMock()

        mock_sagemaker_instance = mock_sagemaker.return_value
        mock_sagemaker_instance.create_sagemaker_connector = MagicMock(
            return_value=True
        )

        mock_helper_instance = mock_ai_helper.return_value

        result = self.create_instance.create_command(
            connector_config_path=mock_config_path
        )
        self.assertTrue(result)
        mock_sagemaker.assert_called_once_with(
            opensearch_domain_region=self.mock_config["opensearch_config"][
                "opensearch_domain_region"
            ],
        )

        mock_sagemaker_instance.create_sagemaker_connector.assert_called_once_with(
            mock_helper_instance,
            self.create_instance.connector_output,
            connector_role_prefix=self.mock_connector_config["connector_role_prefix"],
            region=self.mock_connector_config["region"],
            model_name=self.mock_connector_config["model_name"],
            endpoint_arn=None,
            endpoint_url=None,
            connector_payload=None,
        )
        mock_ai_helper.assert_called_once()

    @patch("opensearch_py_ml.ml_commons.cli.connector_create.CohereModel")
    @patch("opensearch_py_ml.ml_commons.cli.connector_create.AIConnectorHelper")
    def test_create_command_managed_cohere_with_connector_config(
        self, mock_ai_helper, mock_cohere
    ):
        """Test create_command for creating a Cohere connector with a connector configuration"""
        self.mock_connector_config = {
            "setup_config_path": "test_setup_config.yml",
            "connector_name": "Cohere",
            "model_name": "test_model",
            "connector_role_prefix": "test_prefix",
            "api_key": "test-api",
            "connector_secret_name": "test-secret-name",
        }

        self.mock_config["service_type"] = "amazon-opensearch-service"
        self.create_instance.config = self.mock_config
        mock_config_path = "test_config.yml"

        self.create_instance.load_config = MagicMock(return_value=self.mock_config)
        self.create_instance.load_connector_config = MagicMock(
            return_value=self.mock_connector_config
        )
        self.create_instance.update_config = MagicMock()

        mock_cohere_instance = mock_cohere.return_value
        mock_cohere_instance.create_cohere_connector = MagicMock(return_value=True)

        mock_helper_instance = mock_ai_helper.return_value

        result = self.create_instance.create_command(
            connector_config_path=mock_config_path
        )
        self.assertTrue(result)
        mock_cohere.assert_called_once()
        mock_cohere_instance.create_cohere_connector.assert_called_once_with(
            mock_helper_instance,
            self.create_instance.connector_output,
            connector_role_prefix=self.mock_connector_config["connector_role_prefix"],
            model_name=self.mock_connector_config["model_name"],
            api_key=self.mock_connector_config["api_key"],
            connector_payload=None,
            secret_name=self.mock_connector_config["connector_secret_name"],
        )
        mock_ai_helper.assert_called_once()

    @patch("opensearch_py_ml.ml_commons.cli.connector_create.DeepSeekModel")
    @patch("opensearch_py_ml.ml_commons.cli.connector_create.AIConnectorHelper")
    def test_create_command_managed_deepseek_with_connector_config(
        self, mock_ai_helper, mock_deepseek
    ):
        """Test create_command for creating a DeepSeek connector with a connector configuration"""
        self.mock_connector_config = {
            "setup_config_path": "test_setup_config.yml",
            "connector_name": "DeepSeek",
            "model_name": "test_model",
            "connector_role_prefix": "test_prefix",
            "api_key": "test-api",
            "connector_secret_name": "test-secret-name",
        }

        self.mock_config["service_type"] = "amazon-opensearch-service"
        self.create_instance.config = self.mock_config
        mock_config_path = "test_config.yml"

        self.create_instance.load_config = MagicMock(return_value=self.mock_config)
        self.create_instance.load_connector_config = MagicMock(
            return_value=self.mock_connector_config
        )
        self.create_instance.update_config = MagicMock()

        mock_deepseek_instance = mock_deepseek.return_value
        mock_deepseek_instance.create_deepseek_connector = MagicMock(return_value=True)

        mock_helper_instance = mock_ai_helper.return_value

        result = self.create_instance.create_command(
            connector_config_path=mock_config_path
        )
        self.assertTrue(result)
        mock_deepseek.assert_called_once_with(service_type="amazon-opensearch-service")
        mock_deepseek_instance.create_deepseek_connector.assert_called_once_with(
            mock_helper_instance,
            self.create_instance.connector_output,
            connector_role_prefix=self.mock_connector_config["connector_role_prefix"],
            model_name=self.mock_connector_config["model_name"],
            api_key=self.mock_connector_config["api_key"],
            connector_payload=None,
            secret_name=self.mock_connector_config["connector_secret_name"],
        )
        mock_ai_helper.assert_called_once()

    @patch("opensearch_py_ml.ml_commons.cli.connector_create.OpenAIModel")
    @patch("opensearch_py_ml.ml_commons.cli.connector_create.AIConnectorHelper")
    def test_create_command_managed_openai_with_connector_config(
        self, mock_ai_helper, mock_openai
    ):
        """Test create_command for creating an OpenAI connector with a connector configuration"""
        self.mock_connector_config = {
            "setup_config_path": "test_setup_config.yml",
            "connector_name": "OpenAI",
            "model_name": "test_model",
            "connector_role_prefix": "test_prefix",
            "api_key": "test-api",
            "connector_secret_name": "test-secret-name",
        }

        self.mock_config["service_type"] = "amazon-opensearch-service"
        self.create_instance.config = self.mock_config
        mock_config_path = "test_config.yml"

        self.create_instance.load_config = MagicMock(return_value=self.mock_config)
        self.create_instance.load_connector_config = MagicMock(
            return_value=self.mock_connector_config
        )
        self.create_instance.update_config = MagicMock()

        mock_openai_instance = mock_openai.return_value
        mock_openai_instance.create_openai_connector = MagicMock(return_value=True)

        mock_helper_instance = mock_ai_helper.return_value

        result = self.create_instance.create_command(
            connector_config_path=mock_config_path
        )
        self.assertTrue(result)
        mock_openai.assert_called_once()
        mock_openai_instance.create_openai_connector.assert_called_once_with(
            mock_helper_instance,
            self.create_instance.connector_output,
            connector_role_prefix=self.mock_connector_config["connector_role_prefix"],
            model_name=self.mock_connector_config["model_name"],
            api_key=self.mock_connector_config["api_key"],
            connector_payload=None,
            secret_name=self.mock_connector_config["connector_secret_name"],
        )
        mock_ai_helper.assert_called_once()

    @patch("opensearch_py_ml.ml_commons.cli.connector_create.AIConnectorHelper")
    @patch("builtins.print")
    def test_create_command_managed_invalid_connector_with_connector_config(
        self, mock_print, mock_ai_helper
    ):
        """Test create_command with an invalid connector name"""
        # Setup mock config with invalid connector
        self.mock_connector_config = {
            "setup_config_path": "test_setup_config.yml",
            "connector_name": "InvalidConnector",
            "model_name": "test_model",
        }

        self.mock_config["service_type"] = "amazon-opensearch-service"
        self.create_instance.config = self.mock_config
        mock_config_path = "test_config.yml"

        # Mock the configuration loading
        self.create_instance.load_config = MagicMock(return_value=self.mock_config)
        self.create_instance.load_connector_config = MagicMock(
            return_value=self.mock_connector_config
        )
        self.create_instance.update_config = MagicMock()

        # Call the create_command method
        result, _ = self.create_instance.create_command(
            connector_config_path=mock_config_path
        )

        # Verify the result is False
        self.assertFalse(result)

        # Verify the correct error message was printed
        expected_message = f"{Fore.RED}Invalid connector. Please make sure you provide the correct connector name.{Style.RESET_ALL}"
        mock_print.assert_called_once_with(expected_message)

    @patch("opensearch_py_ml.ml_commons.cli.connector_create.AlephAlphaModel")
    @patch("opensearch_py_ml.ml_commons.cli.connector_create.AIConnectorHelper")
    def test_create_command_open_source_aleph_alpha_with_connector_config(
        self, mock_ai_helper, mock_aleph_alpha
    ):
        """Test create_command for creating an Aleph Alpha connector with a connector configuration"""
        self.mock_connector_config = {
            "setup_config_path": "test_setup_config.yml",
            "connector_name": "Aleph Alpha",
            "model_name": "Luminous-Base embedding model",
            "api_key": "test-api",
        }

        self.mock_config["service_type"] = "open-source"
        self.create_instance.config = self.mock_config
        mock_config_path = "test_config.yml"

        self.create_instance.load_config = MagicMock(return_value=self.mock_config)
        self.create_instance.load_connector_config = MagicMock(
            return_value=self.mock_connector_config
        )
        self.create_instance.update_config = MagicMock()

        mock_aleph_alpha_instance = mock_aleph_alpha.return_value
        mock_aleph_alpha_instance.create_aleph_alpha_connector = MagicMock(
            return_value=True
        )

        mock_helper_instance = mock_ai_helper.return_value
        mock_helper_instance.validate_connector_name = MagicMock(return_value=True)
        mock_helper_instance.validate_model_name = MagicMock(return_value=True)

        result = self.create_instance.create_command(
            connector_config_path=mock_config_path
        )
        self.assertTrue(result)

        mock_aleph_alpha.assert_called_once()

        mock_aleph_alpha_instance.create_aleph_alpha_connector.assert_called_once_with(
            mock_helper_instance,
            self.create_instance.connector_output,
            model_name=self.mock_connector_config["model_name"],
            api_key=self.mock_connector_config["api_key"],
            connector_payload=None,
        )
        mock_ai_helper.assert_called_once()

    @patch("opensearch_py_ml.ml_commons.cli.connector_create.DeepSeekModel")
    @patch("opensearch_py_ml.ml_commons.cli.connector_create.AIConnectorHelper")
    def test_create_command_open_source_deepseek_with_connector_config(
        self, mock_ai_helper, mock_deepseek
    ):
        """Test create_command for creating a DeepSeek connector with a connector configuration"""
        self.mock_connector_config = {
            "setup_config_path": "test_setup_config.yml",
            "connector_name": "DeepSeek",
            "model_name": "DeepSeek Chat model",
            "api_key": "test-api",
        }

        self.mock_config["service_type"] = "open-source"
        self.create_instance.config = self.mock_config
        mock_config_path = "test_config.yml"

        self.create_instance.load_config = MagicMock(return_value=self.mock_config)
        self.create_instance.load_connector_config = MagicMock(
            return_value=self.mock_connector_config
        )
        self.create_instance.update_config = MagicMock()

        mock_deepseek_instance = mock_deepseek.return_value
        mock_deepseek_instance.create_deepseek_connector = MagicMock(return_value=True)

        mock_helper_instance = mock_ai_helper.return_value
        mock_helper_instance.validate_connector_name = MagicMock(return_value=True)
        mock_helper_instance.validate_model_name = MagicMock(return_value=True)

        result = self.create_instance.create_command(
            connector_config_path=mock_config_path
        )
        self.assertTrue(result)

        mock_deepseek.assert_called_once_with(service_type="open-source")

        mock_deepseek_instance.create_deepseek_connector.assert_called_once_with(
            mock_helper_instance,
            self.create_instance.connector_output,
            connector_role_prefix=None,
            model_name=self.mock_connector_config["model_name"],
            api_key=self.mock_connector_config["api_key"],
            connector_payload=None,
            secret_name=None,
        )
        mock_ai_helper.assert_called_once()

    @patch("opensearch_py_ml.ml_commons.cli.connector_create.OpenAIModel")
    @patch("opensearch_py_ml.ml_commons.cli.connector_create.AIConnectorHelper")
    def test_create_command_open_source_openai_with_connector_config(
        self, mock_ai_helper, mock_openai
    ):
        """Test create_command for creating an OpenAI connector with a connector configuration"""
        self.mock_connector_config = {
            "setup_config_path": "test_setup_config.yml",
            "connector_name": "OpenAI",
            "model_name": "Embedding model",
            "api_key": "test-api",
        }

        self.mock_config["service_type"] = "open-source"
        self.create_instance.config = self.mock_config
        mock_config_path = "test_config.yml"

        self.create_instance.load_config = MagicMock(return_value=self.mock_config)
        self.create_instance.load_connector_config = MagicMock(
            return_value=self.mock_connector_config
        )
        self.create_instance.update_config = MagicMock()

        mock_openai_instance = mock_openai.return_value
        mock_openai_instance.create_openai_connector = MagicMock(return_value=True)

        mock_helper_instance = mock_ai_helper.return_value
        mock_helper_instance.validate_connector_name = MagicMock(return_value=True)
        mock_helper_instance.validate_model_name = MagicMock(return_value=True)

        result = self.create_instance.create_command(
            connector_config_path=mock_config_path
        )
        self.assertTrue(result)

        mock_openai.assert_called_once_with(service_type="open-source")

        mock_openai_instance.create_openai_connector.assert_called_once_with(
            mock_helper_instance,
            self.create_instance.connector_output,
            connector_role_prefix=None,
            model_name=self.mock_connector_config["model_name"],
            api_key=self.mock_connector_config["api_key"],
            connector_payload=None,
            secret_name=None,
        )
        mock_ai_helper.assert_called_once()

    @patch("builtins.print")
    def test_create_command_open_source_invalid_connector_with_connector_config(
        self, mock_print
    ):
        """Test create_command with an invalid connector name"""
        # Setup mock config with invalid connector
        self.mock_connector_config = {
            "setup_config_path": "test_setup_config.yml",
            "connector_name": "InvalidConnector",
            "model_name": "test_model",
        }

        self.mock_config["service_type"] = "open-source"
        self.create_instance.config = self.mock_config
        mock_config_path = "test_config.yml"

        # Mock the configuration loading
        self.create_instance.load_config = MagicMock(return_value=self.mock_config)
        self.create_instance.load_connector_config = MagicMock(
            return_value=self.mock_connector_config
        )
        self.create_instance.update_config = MagicMock()

        # Call the create_command method
        result, _ = self.create_instance.create_command(
            connector_config_path=mock_config_path
        )

        # Verify the result is False
        self.assertFalse(result)

        # Verify the correct error message was printed
        expected_message = f"{Fore.RED}Invalid connector. Please make sure you provide the correct connector name.{Style.RESET_ALL}"
        mock_print.assert_called_once_with(expected_message)

    @patch("builtins.print")
    @patch("builtins.input")
    def test_create_command_no_connector_config(self, mock_input, mock_print):
        """Test create_command initial menu display when no config path is provided"""
        # Setup basic config
        self.mock_config["service_type"] = "amazon-opensearch-service"
        self.create_instance.config = self.mock_config
        self.create_instance.load_config = MagicMock(return_value=self.mock_config)

        # Mock only the initial inputs
        mock_input.side_effect = [
            "",  # setup configuration file path
            "",  # empty input for connector choice
        ]

        # Call the method without config path
        result = self.create_instance.create_command()

        # Verify only the initial prompts were made
        expected_input_prompts = [
            call("\nEnter the path to your existing setup configuration file: "),
            call("Enter your choice (1-5): "),
        ]

        # Verify only the expected inputs were prompted
        self.assertEqual(mock_input.call_args_list, expected_input_prompts)

        # Verify the menu prompts were printed
        expected_print_prompts = [
            "\nPlease select a supported connector to create:",
            "1. Amazon Bedrock",
            "2. Amazon SageMaker",
            "3. Cohere",
            "4. DeepSeek",
            "5. OpenAI",
        ]

        # Verify menu prompts were printed
        for prompt in expected_print_prompts:
            self.assertTrue(
                any(call[0][0] == prompt for call in mock_print.call_args_list),
                f"Expected prompt not found: {prompt}",
            )

        # Verify result is False since no valid selection was made
        self.assertFalse(result)

    @patch("builtins.print")
    @patch("builtins.input", side_effect=[""])
    def test_create_command_no_connector_config_found(self, mock_input, mock_print):
        """Test create_command with no connector configuration found"""
        self.create_instance.load_connector_config = MagicMock(return_value=None)
        result = self.create_instance.create_command("test-connector-config.yml")
        self.assertFalse(result)
        mock_print.assert_called_once()
        self.assertIn("No connector configuration found", mock_print.call_args[0][0])

    @patch("builtins.print")
    @patch("builtins.input", side_effect=[""])
    def test_create_command_no_setup_config(self, mock_input, mock_print):
        """Test create_command with no setup configuration found"""
        self.create_instance.load_config = MagicMock(return_value=None)
        result = self.create_instance.create_command()
        self.assertFalse(result)
        mock_print.assert_called_once()
        self.assertIn("No setup configuration found", mock_print.call_args[0][0])

    @patch("builtins.print")
    @patch("builtins.input", side_effect=[""])
    def test_create_command_no_domain_endpoint(self, mock_input, mock_print):
        """Test create_command with no domain endpoint"""
        self.mock_config["opensearch_config"]["opensearch_domain_endpoint"] = None
        self.create_instance.load_config = MagicMock(return_value=self.mock_config)
        result = self.create_instance.create_command()
        self.assertFalse(result)
        mock_print.assert_called_once()
        self.assertIn("OpenSearch endpoint not set.", mock_print.call_args[0][0])

    @patch("builtins.print")
    @patch("builtins.input", side_effect=[""])
    def test_create_command_no_username_password(self, mock_input, mock_print):
        """Test create_command with no domain username and password"""
        self.mock_config["service_type"] = "open-source"
        self.mock_config["opensearch_config"]["opensearch_domain_username"] = None
        self.mock_config["opensearch_config"]["opensearch_domain_password"] = None
        self.create_instance.config = self.mock_config
        self.create_instance.load_config = MagicMock(return_value=self.mock_config)
        result = self.create_instance.create_command()
        self.assertFalse(result)
        mock_print.assert_called_once()
        self.assertIn(
            "OpenSearch username or password not set", mock_print.call_args[0][0]
        )

    @patch("builtins.print")
    @patch("builtins.input", side_effect=[""])
    def test_create_command_no_aws_region_or_domain_name(self, mock_input, mock_print):
        """Test create_command with no AWS region and domain name"""
        self.mock_config["service_type"] = "amazon-opensearch-service"
        self.mock_config["opensearch_config"]["opensearch_domain_name"] = None
        self.mock_config["opensearch_config"]["opensearch_domain_region"] = None
        self.create_instance.config = self.mock_config
        self.create_instance.load_config = MagicMock(return_value=self.mock_config)
        result = self.create_instance.create_command()
        self.assertFalse(result)
        mock_print.assert_called_once()
        self.assertIn("AWS region or domain name not set", mock_print.call_args[0][0])

    @patch("builtins.input", side_effect=[""])
    def test_create_command_exception_handling(self, mock_input):
        """Test create_command for exception handling"""
        self.create_instance.load_config = MagicMock(
            side_effect=Exception("Test error")
        )
        result = self.create_instance.create_command()
        self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()
