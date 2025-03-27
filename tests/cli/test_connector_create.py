# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

import unittest
from unittest.mock import MagicMock, patch

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
    @patch("builtins.input", side_effect=["1"])
    def test_create_command_managed_bedrock(
        self, mock_input, mock_ai_helper, mock_bedrock
    ):
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
        mock_ai_helper.assert_called_once_with(
            service_type=self.mock_config["service_type"],
            opensearch_domain_region=self.mock_config["opensearch_config"][
                "opensearch_domain_region"
            ],
            opensearch_domain_name=self.mock_config["opensearch_config"][
                "opensearch_domain_name"
            ],
            opensearch_domain_username=self.mock_config["opensearch_config"][
                "opensearch_domain_username"
            ],
            opensearch_domain_password=self.mock_config["opensearch_config"][
                "opensearch_domain_password"
            ],
            aws_user_name=self.mock_config["aws_credentials"]["aws_user_name"],
            aws_role_name=self.mock_config["aws_credentials"]["aws_role_name"],
            opensearch_domain_url=self.mock_config["opensearch_config"][
                "opensearch_domain_endpoint"
            ],
            aws_access_key=self.mock_config["aws_credentials"]["aws_access_key"],
            aws_secret_access_key=self.mock_config["aws_credentials"][
                "aws_secret_access_key"
            ],
            aws_session_token=self.mock_config["aws_credentials"]["aws_session_token"],
        )

    @patch("opensearch_py_ml.ml_commons.cli.connector_create.DeepSeekModel")
    @patch("opensearch_py_ml.ml_commons.cli.connector_create.AIConnectorHelper")
    @patch("builtins.input", side_effect=["2"])
    def test_create_command_open_source_deepseek(
        self, mock_input, mock_ai_helper, mock_deepseek
    ):
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

        mock_ai_helper.assert_called_once_with(
            service_type=self.mock_config["service_type"],
            opensearch_domain_region=self.mock_config["opensearch_config"][
                "opensearch_domain_region"
            ],
            opensearch_domain_name=None,
            opensearch_domain_username=self.mock_config["opensearch_config"][
                "opensearch_domain_username"
            ],
            opensearch_domain_password=self.mock_config["opensearch_config"][
                "opensearch_domain_password"
            ],
            aws_user_name=self.mock_config["aws_credentials"]["aws_user_name"],
            aws_role_name=self.mock_config["aws_credentials"]["aws_role_name"],
            opensearch_domain_url=self.mock_config["opensearch_config"][
                "opensearch_domain_endpoint"
            ],
            aws_access_key=self.mock_config["aws_credentials"]["aws_access_key"],
            aws_secret_access_key=self.mock_config["aws_credentials"][
                "aws_secret_access_key"
            ],
            aws_session_token=self.mock_config["aws_credentials"]["aws_session_token"],
        )

    @patch("builtins.input", side_effect=[""])
    def test_create_command_no_config(self, mock_input):
        self.create_instance.load_config = MagicMock(return_value=None)
        result = self.create_instance.create_command()
        self.assertFalse(result)

    @patch("builtins.input", side_effect=[""])
    def test_create_command_exception_handling(self, mock_input):
        self.create_instance.load_config = MagicMock(
            side_effect=Exception("Test error")
        )
        result = self.create_instance.create_command()
        self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()
