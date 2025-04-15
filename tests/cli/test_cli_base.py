# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

import json
import os
import unittest
from unittest.mock import MagicMock, Mock, mock_open, patch

import yaml
from colorama import Fore, Style

from opensearch_py_ml.ml_commons.cli.cli_base import CLIBase


class TestCLIBase(unittest.TestCase):
    def setUp(self):
        if os.path.exists(CLIBase.CONFIG_FILE):
            os.remove(CLIBase.CONFIG_FILE)
        self.cli_base = CLIBase()
        self.cli_base.OUTPUT_FILE = "/default/path/output.yml"
        self.test_config = {
            "section1": {"key1": "value1"},
            "section2": {"key2": "value2"},
        }
        self.valid_config = {
            "service_type": CLIBase.AMAZON_OPENSEARCH_SERVICE,
            "opensearch_config": {
                "opensearch_domain_endpoint": "https://test-endpoint",
                "opensearch_domain_region": "us-west-2",
                "opensearch_domain_username": "admin",
                "opensearch_domain_password": "password",
            },
            "aws_credentials": {
                "aws_role_name": "test-role",
                "aws_user_name": "test-user",
                "aws_access_key": "test-key",
                "aws_secret_access_key": "test-secret",
                "aws_session_token": "test-token",
            },
        }

    def tearDown(self):
        if os.path.exists(CLIBase.CONFIG_FILE):
            os.remove(CLIBase.CONFIG_FILE)

    @patch("builtins.print")
    def test_load_config_setup(self, mock_print):
        """Test load_config to load setup config"""
        with open(CLIBase.CONFIG_FILE, "w") as f:
            yaml.dump(self.valid_config, f)
        config = self.cli_base.load_config(CLIBase.CONFIG_FILE, "setup")
        self.assertEqual(config, self.valid_config)
        self.assertEqual(self.cli_base.config, self.valid_config)
        mock_print.assert_called_once()
        self.assertIn(
            "Setup configuration loaded successfully", mock_print.call_args[0][0]
        )

    @patch("builtins.print")
    def test_load_config_connector(self, mock_print):
        """Test load_config to load connector config"""
        connector_config_path = "test_connector_config.yml"
        connector_config = {"connector": {"name": "test-connector", "type": "s3"}}

        with open(connector_config_path, "w") as f:
            yaml.dump(connector_config, f)
        config = self.cli_base.load_config(connector_config_path, "connector")
        self.assertEqual(config, connector_config)
        self.assertNotEqual(self.cli_base.config, connector_config)
        mock_print.assert_called_once()
        self.assertIn(
            "Connector configuration loaded successfully", mock_print.call_args[0][0]
        )

    @patch("opensearch_py_ml.ml_commons.cli.cli_base.logger")
    def test_load_config_no_file(self, mock_logger):
        """Test load_config with a non-existent file"""
        config = self.cli_base.load_config("nonexistent_config.yaml")
        self.assertEqual(config, {})
        mock_logger.warning.assert_called_once()
        warning_message = mock_logger.warning.call_args[0][0]
        self.assertIn("Configuration file not found at", warning_message)

    @patch("opensearch_py_ml.ml_commons.cli.cli_base.logger")
    def test_load_config_invalid_yaml(self, mock_logger):
        """Test load_config with an invalid YAML file"""
        with open(CLIBase.CONFIG_FILE, "w") as f:
            f.write("invalid: yaml: content:")

        config = self.cli_base.load_config(CLIBase.CONFIG_FILE)
        self.assertEqual(config, {})
        mock_logger.error.assert_called_once()
        error_message = mock_logger.error.call_args[0][0]
        self.assertIn("Error parsing YAML configuration:", error_message)

    @patch("opensearch_py_ml.ml_commons.cli.cli_base.logger")
    @patch("os.path.exists")
    def test_load_config_permission_error(self, mock_exists, mock_logger):
        """Test load_config with permission error"""
        mock_exists.return_value = True
        with patch("builtins.open", side_effect=PermissionError):
            config = self.cli_base.load_config(CLIBase.CONFIG_FILE)
            self.assertEqual(config, {})
            mock_logger.error.assert_called_once_with(
                f"{Fore.RED}Permission denied: Unable to read {CLIBase.CONFIG_FILE}{Style.RESET_ALL}"
            )

    @patch("opensearch_py_ml.ml_commons.cli.cli_base.logger")
    @patch("os.path.exists")
    def test_load_config_exception(self, mock_exists, mock_logger):
        """Test load_config with general exception"""
        mock_exists.return_value = True
        with patch("builtins.open", side_effect=Exception("Test error")):
            self.cli_base.load_config(CLIBase.CONFIG_FILE)
            mock_logger.error.assert_called_once_with(
                f"{Fore.RED}Error loading setup configuration: Test error{Style.RESET_ALL}"
            )

    @patch("os.path.abspath", return_value="/default/path/config.yml")
    @patch("os.path.exists", return_value=False)
    @patch("os.makedirs")
    @patch("builtins.print")
    @patch("builtins.input", return_value="")
    @patch("builtins.open", new_callable=mock_open)
    def test_save_yaml_file_config(
        self,
        mock_file,
        mock_input,
        mock_print,
        mock_makedirs,
        mock_exists,
        mock_abspath,
    ):
        """Test save_yaml_file for saving config to a YAML file"""
        # Execute
        save_result = self.cli_base.save_yaml_file(
            config=self.test_config, file_type="configuration"
        )

        # Verify result
        self.assertEqual(save_result, "/default/path/config.yml")
        self.assertEqual(self.cli_base.CONFIG_FILE, "/default/path/config.yml")

        # Verify file operation
        mock_file.assert_called_once_with("/default/path/config.yml", "w")

        # Verify success message
        mock_print.assert_called_once()
        self.assertIn(
            "Configuration information saved successfully", mock_print.call_args[0][0]
        )

    @patch("os.path.abspath", return_value="/default/path/config.yml")
    @patch("os.path.exists", return_value=True)
    @patch("os.makedirs")
    @patch("builtins.print")
    @patch("builtins.input", side_effect=["", "yes"])
    @patch("builtins.open", new_callable=mock_open)
    @patch.object(CLIBase, "_confirm_overwrite")
    def test_save_yaml_file_config_overwrite(
        self,
        mock_overwrite,
        mock_file,
        mock_input,
        mock_print,
        mock_makedirs,
        mock_exists,
        mock_abspath,
    ):
        """Test save_yaml_file for overwriting existing config file"""
        # Setup
        mock_overwrite.return_value = True

        # Execute
        save_result = self.cli_base.save_yaml_file(config=self.test_config)

        # Verify result and method call
        self.assertEqual(save_result, "/default/path/config.yml")
        mock_overwrite.assert_called_once()

    @patch("os.path.abspath", return_value="/default/path/config.yml")
    @patch("os.path.exists", return_value=True)
    @patch("builtins.print")
    @patch("builtins.input", side_effect=["", "no"])
    @patch.object(CLIBase, "_confirm_overwrite")
    def test_save_yaml_file_config_cancel_overwrite(
        self, mock_overwrite, mock_input, mock_print, mock_exists, mock_abspath
    ):
        """Test save_yaml_file when user cancels config overwrite"""
        # Setup
        mock_overwrite.return_value = False

        # Execute
        save_result = self.cli_base.save_yaml_file(config=self.test_config)

        # Verify result and method call
        self.assertIsNone(save_result)
        mock_overwrite.assert_called_once()

    @patch("os.path.abspath", return_value="/default/path/output.yml")
    @patch("os.path.exists", return_value=True)
    @patch("os.makedirs")
    @patch("builtins.print")
    @patch("builtins.input", return_value="")
    @patch("builtins.open", new_callable=mock_open)
    @patch.object(CLIBase, "_merge_configs")
    def test_save_yaml_file_output(
        self,
        mock_merge_configs,
        mock_file,
        mock_input,
        mock_print,
        mock_makedirs,
        mock_exists,
        mock_abspath,
    ):
        """Test save_yaml_file for saving output information to a YAML file"""
        # Setup
        test_config = {"test": "data"}
        existing_config = {"existing": "data"}
        merged_config = {"test": "data", "existing": "data"}
        mock_merge_configs.return_value = merged_config
        mock_file.return_value.read.return_value = yaml.dump(existing_config)

        # Execute
        save_result = self.cli_base.save_yaml_file(
            config=test_config, file_type="output", merge_existing=True
        )

        # Verify result
        self.assertEqual(save_result, "/default/path/output.yml")
        self.assertEqual(self.cli_base.OUTPUT_FILE, "/default/path/output.yml")

        # Verify output message
        mock_input.assert_called_once_with(
            "\nEnter the path to save the output information, "
            "or press Enter to save it in the current directory [/default/path/output.yml]: "
        )
        mock_merge_configs.assert_called_once_with(
            self.cli_base.OUTPUT_FILE, test_config
        )
        mock_print.assert_called_once()
        self.assertIn(
            "Output information saved successfully", mock_print.call_args[0][0]
        )

    @patch("opensearch_py_ml.ml_commons.cli.cli_base.logger")
    @patch("builtins.input", return_value="")
    def test_save_yaml_file_permission_error(self, mock_input, mock_logger):
        """Test save_yaml_file with permission error"""
        with patch("builtins.open", side_effect=PermissionError("Permission denied")):
            result = self.cli_base.save_yaml_file(self.test_config)
            self.assertIsNone(result)
            self.assertFalse(os.path.exists(CLIBase.CONFIG_FILE))
            mock_logger.error.assert_called_once_with(
                f"{Fore.RED}Error: Permission denied. Unable to write to {CLIBase.CONFIG_FILE}{Style.RESET_ALL}"
            )

    @patch("opensearch_py_ml.ml_commons.cli.cli_base.logger")
    @patch("builtins.input")
    def test_save_yaml_file_keyboard_interrupt(self, mock_input, mock_logger):
        """Test save_yaml_file keyboard interrupt handling"""
        mock_input.side_effect = KeyboardInterrupt()
        result = self.cli_base.save_yaml_file(self.test_config)
        self.assertIsNone(result)
        mock_logger.error.assert_called_once_with(
            f"\n{Fore.YELLOW}Operation cancelled by user.{Style.RESET_ALL}"
        )

    @patch("opensearch_py_ml.ml_commons.cli.cli_base.logger")
    @patch("builtins.input", return_value="")
    def test_save_yaml_file_exception(self, mock_input, mock_logger):
        """Test save_yaml_file with exception"""
        with patch("builtins.open", side_effect=Exception("Test error")):
            self.cli_base.save_yaml_file(self.test_config)
            self.assertFalse(os.path.exists(CLIBase.CONFIG_FILE))
            mock_logger.error.assert_called_once_with(
                f"{Fore.RED}Error saving configuration: Test error{Style.RESET_ALL}"
            )

    @patch("builtins.open", new_callable=mock_open)
    def test_merge_configs_empty_existing(self, mock_file):
        """Test _merge_configs when existing file is empty"""
        # Setup
        mock_file.return_value.read.return_value = ""
        new_config = {"key1": "value1"}

        # Execute
        result = self.cli_base._merge_configs(CLIBase.OUTPUT_FILE, new_config)

        # Verify
        self.assertEqual(result, new_config)

    @patch("builtins.input", return_value="yes")
    def test_confirm_overwrite_yes(self, mock_input):
        """Test _confirm_overwrite when user chooses to overwrite"""
        self.cli_base._confirm_overwrite(self.cli_base.CONFIG_FILE)
        mock_input.assert_any_call(
            f"{Fore.YELLOW}File already exists at {self.cli_base.CONFIG_FILE}. "
            f"Do you want to overwrite it? (yes/no): {Style.RESET_ALL}"
        )

    @patch("opensearch_py_ml.ml_commons.cli.cli_base.logger")
    @patch("builtins.input", return_value="no")
    def test_confirm_overwrite_no(self, mock_input, mock_logger):
        """Test _confirm_overwrite when user chooses not to overwrite"""
        result = self.cli_base._confirm_overwrite(self.cli_base.CONFIG_FILE)
        mock_input.assert_any_call(
            f"{Fore.YELLOW}File already exists at {self.cli_base.CONFIG_FILE}. "
            f"Do you want to overwrite it? (yes/no): {Style.RESET_ALL}"
        )
        mock_logger.warning.assert_called_once_with(
            f"{Fore.YELLOW}Operation cancelled. Please choose a different path.{Style.RESET_ALL}"
        )
        self.assertFalse(result)

    @patch("opensearch_py_ml.ml_commons.cli.cli_base.logger")
    @patch("builtins.input", side_effect=["invalid", "no"])
    def test_confirm_overwrite_invalid_response(self, mock_input, mock_logger):
        """Test _confirm_overwrite when user gives invalid response"""
        result = self.cli_base._confirm_overwrite(self.cli_base.CONFIG_FILE)
        mock_input.assert_any_call(
            f"{Fore.YELLOW}File already exists at {self.cli_base.CONFIG_FILE}. "
            f"Do you want to overwrite it? (yes/no): {Style.RESET_ALL}"
        )
        mock_logger.warning.assert_any_call(
            f"{Fore.YELLOW}Please enter 'yes' or 'no'.{Style.RESET_ALL}"
        )
        self.assertFalse(result)

    def test_update_config(self):
        """Test update_config successful"""
        test_config = {
            "setup_config_path": "test_path",
            "connector_name": "test_connector",
        }
        test_path = "test_config.yml"

        with patch("builtins.open", mock_open()) as mock_file, patch(
            "yaml.dump"
        ) as mock_yaml_dump, patch("builtins.print") as mock_print:

            result = self.cli_base.update_config(test_config, test_path)

            # Verify the result
            self.assertTrue(result)
            mock_file.assert_called_once_with(test_path, "w")
            mock_yaml_dump.assert_called_once_with(
                test_config, mock_file(), default_flow_style=False, sort_keys=False
            )
            mock_print.assert_called_once()
            self.assertIn(
                "Configuration saved successfully", mock_print.call_args[0][0]
            )

    @patch("opensearch_py_ml.ml_commons.cli.cli_base.logger")
    def test_update_config_permission_error(self, mock_logger):
        """Test update_config with permission error"""
        test_config = {"test": "config"}
        test_path = "test_config.yml"

        with patch("builtins.open", mock_open()) as mock_file:
            # Execute
            mock_file.side_effect = PermissionError("Permission denied")
            result = self.cli_base.update_config(test_config, test_path)

            # Verify result and error message
            self.assertFalse(result)
            mock_logger.error.assert_called_once_with(
                f"{Fore.RED}Error saving configuration: Permission denied{Style.RESET_ALL}"
            )

    @patch("opensearch_py_ml.ml_commons.cli.cli_base.logger")
    def test_update_config_yaml_error(self, mock_logger):
        """Test update_config with YAML error"""
        test_config = {"test": "config"}
        test_path = "test_config.yml"

        with patch("builtins.open", mock_open()), patch("yaml.dump") as mock_yaml_dump:
            # Execute
            mock_yaml_dump.side_effect = yaml.YAMLError("Invalid YAML")
            result = self.cli_base.update_config(test_config, test_path)

            # Verify result and error message
            self.assertFalse(result)
            mock_logger.error.assert_called_once_with(
                f"{Fore.RED}Error saving configuration: Invalid YAML{Style.RESET_ALL}"
            )

    @patch("opensearch_py_ml.ml_commons.cli.cli_base.logger")
    def test_update_config_invalid_path(self, mock_logger):
        """Test update_config with invalid path"""
        test_config = {"test": "config"}
        test_path = "/invalid/path/test_config.yml"

        with patch("builtins.open", mock_open()) as mock_file:
            # Execute
            mock_file.side_effect = FileNotFoundError("No such file or directory")
            result = self.cli_base.update_config(test_config, test_path)

            # Verify result and error message
            self.assertFalse(result)
            mock_logger.error.assert_called_once_with(
                f"{Fore.RED}Error saving configuration: No such file or directory{Style.RESET_ALL}"
            )

    def test_update_config_empty_config(self):
        """Test update_config with empty config"""
        test_config = {}
        test_path = "test_config.yml"

        with patch("builtins.open", mock_open()) as mock_file, patch(
            "yaml.dump"
        ) as mock_yaml_dump, patch("builtins.print"):
            # Execute
            result = self.cli_base.update_config(test_config, test_path)

            # Verify
            self.assertTrue(result)
            mock_yaml_dump.assert_called_once_with(
                {}, mock_file(), default_flow_style=False, sort_keys=False
            )

    @patch.object(CLIBase, "save_yaml_file", Mock())
    def test_connector_output(self):
        """Test connector_output with all parameters provided"""
        output_id = "test-id"
        output_config = json.dumps({"name": "test-connector"})
        role_name = "test-role"
        role_arn = "test-arn"
        secret_name = "test-secret"
        secret_arn = "test-secret-arn"

        self.cli_base.connector_output(
            output_id=output_id,
            output_config=output_config,
            role_name=role_name,
            role_arn=role_arn,
            secret_name=secret_name,
            secret_arn=secret_arn,
        )

        expected_update = [
            {
                "connector_id": "test-id",
                "connector_name": "test-connector",
                "connector_role_name": "test-role",
                "connector_role_arn": "test-arn",
                "connector_secret_name": "test-secret",
                "connector_secret_arn": "test-secret-arn",
            }
        ]

        # Verify
        self.assertEqual(
            self.cli_base.output_config["connector_create"], expected_update
        )
        self.cli_base.save_yaml_file.assert_called_once_with(
            self.cli_base.output_config, "output", merge_existing=True
        )

    @patch.object(CLIBase, "save_yaml_file", Mock())
    def test_connector_output_invalid_json(self):
        """Test connector_output with invalid JSON"""
        output_id = "test-id"
        output_config = "invalid json"

        with self.assertRaises(json.JSONDecodeError):
            self.cli_base.connector_output(
                output_id=output_id, output_config=output_config
            )

        # Verify save_yaml_file was not called
        self.cli_base.save_yaml_file.assert_not_called()

    @patch.object(CLIBase, "save_yaml_file", Mock())
    def test_register_model_output(self):
        """Test register_model_output with all parameters provided"""
        model_id = "test-id"
        model_name = "test-model"
        connector_id = "test-connector-id"
        expected_update = [
            {
                "model_id": "test-id",
                "model_name": "test-model",
                "connector_id": "test-connector-id",
            }
        ]

        # Execute
        self.cli_base.register_model_output(
            model_id=model_id, model_name=model_name, connector_id=connector_id
        )

        # Verify
        self.assertEqual(self.cli_base.output_config["register_model"], expected_update)
        self.cli_base.save_yaml_file.assert_called_once_with(
            self.cli_base.output_config, "output", merge_existing=True
        )

    @patch.object(CLIBase, "save_yaml_file", Mock())
    def test_predict_model_output(self):
        """Test predict_model_output with all parameters provided"""
        model_id = "test-model-id"
        response = "test-response"
        expected_update = [{"model_id": "test-model-id", "response": "test-response"}]

        # Execute
        self.cli_base.predict_model_output(model_id=model_id, response=response)

        # Verify
        self.assertEqual(self.cli_base.output_config["predict_model"], expected_update)
        self.cli_base.save_yaml_file.assert_called_once_with(
            self.cli_base.output_config, "output", merge_existing=True
        )

    def test_get_opensearch_domain_name(self):
        """Test get_opensearch_domain_name with various inputs, including edge cases"""
        test_cases = [
            # (input_url, expected_output)
            (
                "https://search-my-domain-abc123xyz.us-west-2.es.amazonaws.com",
                "my-domain",
            ),
            (
                "https://search-test-domain-def456uvw.eu-central-1.es.amazonaws.com",
                "test-domain",
            ),
            ("http://search-single-node-789.us-east-1.es.amazonaws.com", "single-node"),
            (
                "https://vpc-test-vpc-domain-abc123.us-west-2.es.amazonaws.com",
                "test-vpc-domain",
            ),
            ("http://vpc-my-vpc-abc123.us-west-2.es.amazonaws.com", "my-vpc"),
            # Edge cases
            ("", None),
            (None, None),
            # Malformed URLs
            ("http://", None),
            ("not_a_url", None),
            ("https://", None),
            # URLs without hostname
            ("file:///path/to/file", None),
            # Invalid OpenSearch domain URLs
            ("invalid-url", None),
            ("https://example.com", "example"),
            ("https://search-.domain.com", ""),
        ]
        for input_url, expected_output in test_cases:
            with self.subTest(input_url=input_url):
                result = self.cli_base.get_opensearch_domain_name(input_url)
                self.assertEqual(
                    result,
                    expected_output,
                    f"Failed for input '{input_url}'. Expected '{expected_output}', got '{result}'",
                )

    @patch("builtins.print")
    def test_check_config_opensource(self, mock_print):
        """Test _check_config with open-source service successful"""
        # Setup
        self.valid_config["service_type"] = CLIBase.OPEN_SOURCE
        service_type = self.valid_config["service_type"]
        opensearch_config = self.valid_config["opensearch_config"]

        # Execute
        self.cli_base._check_config(self.valid_config, service_type, opensearch_config)

        # Verify
        mock_print.assert_not_called()

    @patch("boto3.client")
    def test_check_config_managed(self, mock_boto3_client):
        """Test _check_config with managed service successful"""
        # Setup
        service_type = self.valid_config["service_type"]
        opensearch_config = self.valid_config["opensearch_config"]

        # Mock the AWS client
        mock_opensearch_client = MagicMock()
        mock_boto3_client.return_value = mock_opensearch_client

        # Mock the get_opensearch_domain_name method
        self.cli_base.get_opensearch_domain_name = Mock(return_value="test-domain")

        # Execute
        self.cli_base._check_config(self.valid_config, service_type, opensearch_config)

        # Verify
        self.assertEqual(self.cli_base.opensearch_domain_name, "test-domain")

    @patch("opensearch_py_ml.ml_commons.cli.cli_base.logger")
    def test_check_config_managed_no_region(self, mock_logger):
        """Test _check_config with managed service and no region"""
        # Setup
        self.valid_config["opensearch_config"]["opensearch_domain_region"] = ""
        service_type = self.valid_config["service_type"]
        opensearch_config = self.valid_config["opensearch_config"]

        # Execute
        result = self.cli_base._check_config(
            self.valid_config, service_type, opensearch_config
        )

        # Verify
        self.assertFalse(result)
        mock_logger.warning.assert_called_with(
            f"{Fore.RED}AWS region or domain name not set. Please run setup first.{Style.RESET_ALL}\n"
        )

    @patch("opensearch_py_ml.ml_commons.cli.cli_base.logger")
    def test_check_config_no_endpoint(self, mock_logger):
        """Test _check_config with no OpenSearch endpoint"""
        # Setup
        self.valid_config["opensearch_config"]["opensearch_domain_endpoint"] = ""
        service_type = self.valid_config["service_type"]
        opensearch_config = self.valid_config["opensearch_config"]

        # Execute
        result = self.cli_base._check_config(
            self.valid_config, service_type, opensearch_config
        )

        # Verify
        self.assertFalse(result)
        mock_logger.warning.assert_called_with(
            f"\n{Fore.RED}OpenSearch endpoint not set. Please run setup first.{Style.RESET_ALL}\n"
        )

    def test_load_and_check_config(self):
        """Test load_and_check_config successful"""
        # Setup
        mock_ai_helper = MagicMock()
        self.cli_base.load_config = MagicMock(return_value=self.valid_config)
        self.cli_base._check_config = MagicMock(return_value=mock_ai_helper)

        # Execute
        result = self.cli_base.load_and_check_config("test_config.yml")

        # Verify
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 4)
        ai_helper, config, service_type, opensearch_config = result

        self.assertEqual(ai_helper, mock_ai_helper)
        self.assertEqual(config, self.valid_config)

    def test_load_and_check_config_fail_load(self):
        """Test load_and_check_config when load_config fails"""
        # Setup
        self.cli_base.load_config = MagicMock(return_value=None)
        self.cli_base._check_config = MagicMock()

        # Execute
        result = self.cli_base.load_and_check_config("test_config.yml")

        # Verify
        self.assertFalse(result)
        self.cli_base.load_config.assert_called_once_with("test_config.yml")
        self.cli_base._check_config.assert_not_called()

    def test_load_and_check_config_invalid_config(self):
        """Test load_and_check_config when config is invalid"""
        # Setup
        self.cli_base.load_config = MagicMock(return_value={})
        self.cli_base._check_config = MagicMock(return_value=None)

        # Execute
        result = self.cli_base.load_and_check_config("test_config.yml")

        # Verify
        self.assertFalse(result)
        self.cli_base.load_config.assert_called_once_with("test_config.yml")


if __name__ == "__main__":
    unittest.main()
