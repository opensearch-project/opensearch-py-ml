# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

import json
import os
import unittest
from unittest.mock import Mock, mock_open, patch

import yaml
from colorama import Fore, Style

from opensearch_py_ml.ml_commons.cli.connector_base import ConnectorBase


class TestConnectorBase(unittest.TestCase):
    def setUp(self):
        if os.path.exists(ConnectorBase.CONFIG_FILE):
            os.remove(ConnectorBase.CONFIG_FILE)
        self.connector_base = ConnectorBase()
        self.connector_base.OUTPUT_FILE = "/default/path/output.yml"
        self.test_config = {
            "section1": {"key1": "value1"},
            "section2": {"key2": "value2"},
        }

    def tearDown(self):
        if os.path.exists(ConnectorBase.CONFIG_FILE):
            os.remove(ConnectorBase.CONFIG_FILE)

    @patch("builtins.print")
    def test_load_config_no_file(self, mock_print):
        """Test load_config with a non-existent file"""
        config = self.connector_base.load_config("nonexistent_config.yaml")
        self.assertEqual(config, {})
        mock_print.assert_called_once()
        self.assertIn("Configuration file not found", mock_print.call_args[0][0])

    @patch("builtins.print")
    def test_load_config_valid_yaml(self, mock_print):
        """Test load_config with a valid YAML file"""
        test_config = {
            "service_type": "amazon-opensearch-service",
            "opensearch_domain_region": "test-region",
        }
        with open(ConnectorBase.CONFIG_FILE, "w") as f:
            yaml.dump(test_config, f)
        config = self.connector_base.load_config(ConnectorBase.CONFIG_FILE)
        self.assertEqual(config, test_config)
        self.assertEqual(self.connector_base.config, test_config)
        mock_print.assert_called_once()
        self.assertIn(
            "Setup configuration loaded successfully", mock_print.call_args[0][0]
        )

    @patch("builtins.print")
    def test_load_config_invalid_yaml(self, mock_print):
        """Test load_config with an invalid YAML file"""
        with open(ConnectorBase.CONFIG_FILE, "w") as f:
            f.write("invalid: yaml: content:")

        config = self.connector_base.load_config(ConnectorBase.CONFIG_FILE)
        self.assertEqual(config, {})
        mock_print.assert_called_once()
        self.assertIn("Error parsing YAML configuration", mock_print.call_args[0][0])

    @patch("builtins.print")
    @patch("os.path.exists")
    def test_load_config_permission_error(self, mock_exists, mock_print):
        """Test load_config with permission error"""
        mock_exists.return_value = True
        with patch("builtins.open", side_effect=PermissionError):
            config = self.connector_base.load_config(ConnectorBase.CONFIG_FILE)

            # Verify empty dict is returned
            self.assertEqual(config, {})

            # Verify exact error message
            expected_message = f"{Fore.RED}Permission denied: Unable to read {ConnectorBase.CONFIG_FILE}{Style.RESET_ALL}"
            mock_print.assert_called_once_with(expected_message)

    @patch("builtins.print")
    @patch("os.path.exists")
    def test_load_config_exception(self, mock_exists, mock_print):
        """Test load_config with general exception"""
        mock_exists.return_value = True
        with patch("builtins.open", side_effect=Exception("Test error")):
            config = self.connector_base.load_config(ConnectorBase.CONFIG_FILE)
            mock_print.assert_called_once()
            self.assertEqual(config, {})
            self.assertIn(
                "Error loading setup configuration", mock_print.call_args[0][0]
            )

    def test_load_connector_config_valid_yaml(self):
        """Test load_connector_config with a valid YAML connector configuration"""
        test_connector_config = {
            "setup_config_path": "test_path",
            "connector_name": "test_connector",
        }
        mock_file_content = yaml.dump(test_connector_config)

        with patch(
            "builtins.open", mock_open(read_data=mock_file_content)
        ) as mock_file, patch("os.path.exists", return_value=True), patch(
            "os.access", return_value=True
        ):

            result = self.connector_base.load_connector_config("test_config.yml")

            mock_file.assert_called_once_with("test_config.yml", "r")
            self.assertEqual(result, test_connector_config)

    @patch("builtins.print")
    def test_load_connector_config_invalid_yaml(self, mock_print):
        """Test load_connector_config with an invalid YAML file"""
        with open(ConnectorBase.CONFIG_FILE, "w") as f:
            f.write("invalid: yaml: content:")

        config = self.connector_base.load_connector_config(ConnectorBase.CONFIG_FILE)
        self.assertEqual(config, {})
        mock_print.assert_called_once()
        self.assertIn("Error parsing YAML configuration", mock_print.call_args[0][0])

    def test_load_connector_config_file_not_found(self):
        """Test load_connector_config with a non-existent file"""
        with patch("builtins.open", mock_open()) as mock_file:
            mock_file.side_effect = FileNotFoundError()
            result = self.connector_base.load_connector_config("nonexistent.yml")
            self.assertEqual(result, {})

    @patch("builtins.print")
    @patch("os.path.exists")
    def test_load_connector_config_permission_error(self, mock_exists, mock_print):
        """Test load_connector_config with permission error"""
        mock_exists.return_value = True
        with patch("builtins.open", side_effect=PermissionError):
            config = self.connector_base.load_connector_config(
                ConnectorBase.CONFIG_FILE
            )

            # Verify empty dict is returned
            self.assertEqual(config, {})

            # Verify exact error message
            expected_message = f"{Fore.RED}Permission denied: Unable to read {ConnectorBase.CONFIG_FILE}{Style.RESET_ALL}"
            mock_print.assert_called_once_with(expected_message)

    @patch("builtins.print")
    @patch("os.path.exists")
    def test_load_connector_config_exception(self, mock_exists, mock_print):
        """Test load_connector_config with general exception"""
        mock_exists.return_value = True
        with patch("builtins.open", side_effect=Exception("Test error")):
            config = self.connector_base.load_connector_config(
                ConnectorBase.CONFIG_FILE
            )
            mock_print.assert_called_once()
            self.assertEqual(config, {})
            self.assertIn(
                "Error loading connector configuration", mock_print.call_args[0][0]
            )

    @patch("builtins.print")
    @patch("builtins.input", return_value="")
    def test_save_config(self, mock_input, mock_print):
        """Test save_config successful"""
        test_config = {"key": "value"}
        save_result = self.connector_base.save_config(test_config)
        mock_print.assert_called_once()
        self.assertTrue(save_result)
        self.assertTrue(os.path.exists(ConnectorBase.CONFIG_FILE))
        self.assertIn("Configuration saved successfully", mock_print.call_args[0][0])

    @patch("builtins.input")
    @patch("os.path.exists")
    @patch("builtins.print")
    def test_save_config_yes_overwrite(self, mock_print, mock_exists, mock_input):
        """Test save_config when file exists and user chooses to overwrite"""
        config = {"test": "data"}

        # Mock file exists
        mock_exists.return_value = True

        # Mock user inputs: first for file path (empty for default), then 'yes' for overwrite
        mock_input.side_effect = ["", "yes"]

        with patch("builtins.open", mock_open()), patch("yaml.dump") as mock_dump:

            result = self.connector_base.save_config(config)

            # Verify the overwrite confirmation was asked
            mock_input.assert_any_call(
                f"{Fore.YELLOW}File already exists at {self.connector_base.CONFIG_FILE}. "
                f"Do you want to overwrite it? (yes/no): {Style.RESET_ALL}"
            )

            # Verify file was saved
            self.assertIsNotNone(result)
            mock_dump.assert_called_once()

    @patch("builtins.input")
    @patch("os.path.exists")
    @patch("builtins.print")
    def test_save_config_no_overwrite(self, mock_print, mock_exists, mock_input):
        """Test save_config when file exists and user chooses not to overwrite"""
        config = {"test": "data"}

        # Mock file exists
        mock_exists.return_value = True

        # Mock user inputs: first for file path (empty for default), then 'no' for overwrite
        mock_input.side_effect = ["", "no"]

        with patch("builtins.open", mock_open()):
            result = self.connector_base.save_config(config)

            # Verify the overwrite confirmation was asked
            mock_input.assert_any_call(
                f"{Fore.YELLOW}File already exists at {self.connector_base.CONFIG_FILE}. "
                f"Do you want to overwrite it? (yes/no): {Style.RESET_ALL}"
            )

            # Verify operation was cancelled
            self.assertIsNone(result)
            mock_print.assert_any_call(
                f"{Fore.YELLOW}Operation cancelled. Please choose a different path.{Style.RESET_ALL}"
            )

    @patch("builtins.input")
    @patch("os.path.exists")
    @patch("builtins.print")
    def test_save_config_invalid_response(self, mock_print, mock_exists, mock_input):
        """Test save_config for invalid response"""
        config = {"test": "data"}
        mock_exists.return_value = True
        mock_input.side_effect = ["", "invalid"]

        with patch("builtins.open", mock_open()):
            self.connector_base.save_config(config)
            expected_message = (
                f"{Fore.YELLOW}Please enter 'yes or 'no'.{Style.RESET_ALL}"
            )
            mock_print.assert_any_call(expected_message)

    @patch("os.makedirs")
    @patch("os.path.exists")
    @patch("builtins.input")
    def test_save_config_create_directory(self, mock_input, mock_exists, mock_makedirs):
        """Test save_config with directory creation"""
        config = {"test": "data"}
        test_path = "/test/path/config.yaml"

        # Mock user input to return custom path
        mock_input.return_value = test_path

        # Mock directory doesn't exist
        mock_exists.side_effect = [
            False,
            False,
        ]  # First for file check, second for directory check

        with patch("builtins.open", mock_open()), patch("yaml.dump"):
            self.connector_base.save_config(config)

            # Verify makedirs was called with the correct directory
            expected_directory = "/test/path"
            mock_makedirs.assert_called_once_with(expected_directory)

    @patch("builtins.print")
    @patch("builtins.input", return_value="")
    def test_save_config_permission_error(self, mock_input, mock_print):
        """Test save_config with permission error"""
        with patch("builtins.open", side_effect=PermissionError("Permission denied")):
            config = {"key": "value"}
            result = self.connector_base.save_config(config)
            mock_print.assert_called_once()
            self.assertIsNone(result)
            self.assertFalse(os.path.exists(ConnectorBase.CONFIG_FILE))
            self.assertIn("Permission denied", mock_print.call_args[0][0])

    @patch("builtins.input")
    @patch("builtins.print")
    def test_save_config_keyboard_interrupt(self, mock_print, mock_input):
        """Test save_config keyboard interrupt handling"""
        config = {"test": "data"}
        mock_input.side_effect = KeyboardInterrupt()
        result = self.connector_base.save_config(config)
        self.assertIsNone(result)
        expected_message = (
            f"\n{Fore.YELLOW}Operation cancelled by user.{Style.RESET_ALL}"
        )
        mock_print.assert_called_once_with(expected_message)

    @patch("builtins.print")
    @patch("builtins.input", return_value="")
    def test_save_config_exception(self, mock_input, mock_print):
        """Test save_config with exception"""
        with patch("builtins.open", side_effect=Exception("Test error")):
            config = {"key": "value"}
            self.connector_base.save_config(config)
            mock_print.assert_called_once()
            self.assertFalse(os.path.exists(ConnectorBase.CONFIG_FILE))
            self.assertIn("Error saving configuration", mock_print.call_args[0][0])

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

            result = self.connector_base.update_config(test_config, test_path)

            # Verify the result
            self.assertTrue(result)

            # Verify file operations
            mock_file.assert_called_once_with(test_path, "w")

            # Verify yaml.dump was called with correct arguments
            mock_yaml_dump.assert_called_once_with(
                test_config, mock_file(), default_flow_style=False, sort_keys=False
            )

            # Verify success message
            mock_print.assert_called_once()
            self.assertIn(
                "Configuration saved successfully", mock_print.call_args[0][0]
            )

    def test_update_config_permission_error(self):
        """Test update_config with permission error"""
        test_config = {"test": "config"}
        test_path = "test_config.yml"

        with patch("builtins.open", mock_open()) as mock_file, patch(
            "builtins.print"
        ) as mock_print:

            # Simulate permission error
            mock_file.side_effect = PermissionError("Permission denied")

            result = self.connector_base.update_config(test_config, test_path)

            # Verify the result
            self.assertFalse(result)

            # Verify error message
            mock_print.assert_called_once()
            self.assertIn("Error saving configuration", mock_print.call_args[0][0])
            self.assertIn("Permission denied", mock_print.call_args[0][0])

    def test_update_config_yaml_error(self):
        """Test update_config with YAML error"""
        test_config = {"test": "config"}
        test_path = "test_config.yml"

        with patch("builtins.open", mock_open()), patch(
            "yaml.dump"
        ) as mock_yaml_dump, patch("builtins.print") as mock_print:

            # Simulate YAML error
            mock_yaml_dump.side_effect = yaml.YAMLError("Invalid YAML")

            result = self.connector_base.update_config(test_config, test_path)

            # Verify the result
            self.assertFalse(result)

            # Verify error message
            mock_print.assert_called_once()
            self.assertIn("Error saving configuration", mock_print.call_args[0][0])
            self.assertIn("Invalid YAML", mock_print.call_args[0][0])

    def test_update_config_invalid_path(self):
        """Test update_config with invalid path"""
        test_config = {"test": "config"}
        test_path = "/invalid/path/test_config.yml"

        with patch("builtins.open", mock_open()) as mock_file, patch(
            "builtins.print"
        ) as mock_print:

            # Simulate FileNotFoundError
            mock_file.side_effect = FileNotFoundError("No such file or directory")

            result = self.connector_base.update_config(test_config, test_path)

            # Verify the result
            self.assertFalse(result)

            # Verify error message
            mock_print.assert_called_once()
            self.assertIn("Error saving configuration", mock_print.call_args[0][0])
            self.assertIn("No such file or directory", mock_print.call_args[0][0])

    def test_update_config_empty_config(self):
        """Test update_config with empty config"""
        test_config = {}
        test_path = "test_config.yml"

        with patch("builtins.open", mock_open()) as mock_file, patch(
            "yaml.dump"
        ) as mock_yaml_dump, patch("builtins.print"):

            result = self.connector_base.update_config(test_config, test_path)

            # Verify the result
            self.assertTrue(result)

            # Verify yaml.dump was called with empty config
            mock_yaml_dump.assert_called_once_with(
                {}, mock_file(), default_flow_style=False, sort_keys=False
            )

    @patch.object(ConnectorBase, "save_output", Mock())
    def test_connector_output(self):
        """Test connector_output with all parameters provided"""
        output_id = "test-id"
        output_config = json.dumps({"name": "test-connector"})
        role_name = "test-role"
        secret_name = "test-secret"
        role_arn = "test-arn"

        self.connector_base.connector_output(
            output_id=output_id,
            output_config=output_config,
            role_name=role_name,
            secret_name=secret_name,
            role_arn=role_arn,
        )

        expected_update = {
            "connector_id": "test-id",
            "connector_name": "test-connector",
            "connector_role_arn": "test-arn",
            "connector_role_name": "test-role",
            "connector_secret_name": "test-secret",
        }

        # Verify the output_config was updated correctly
        self.assertEqual(
            self.connector_base.output_config["connector_create"], expected_update
        )

        # Verify save_output was called with the updated config
        self.connector_base.save_output.assert_called_once_with(
            self.connector_base.output_config
        )

    @patch.object(ConnectorBase, "save_output", Mock())
    def test_connector_output_invalid_json(self):
        """Test connector_output with invalid JSON"""
        output_id = "test-id"
        output_config = "invalid json"

        with self.assertRaises(json.JSONDecodeError):
            self.connector_base.connector_output(
                output_id=output_id, output_config=output_config
            )

        # Verify save_output was not called
        self.connector_base.save_output.assert_not_called()

    @patch.object(ConnectorBase, "save_output", Mock())
    def test_register_model_output(self):
        """Test register_model_output with all parameters provided"""
        model_id = "test-id"
        model_name = "test-model"

        self.connector_base.register_model_output(
            model_id=model_id, model_name=model_name
        )

        expected_update = {"model_id": "test-id", "model_name": "test-model"}

        # Verify the output_config was updated correctly
        self.assertEqual(
            self.connector_base.output_config["register_model"], expected_update
        )

        # Verify save_output was called with the updated config
        self.connector_base.save_output.assert_called_once_with(
            self.connector_base.output_config
        )

    @patch.object(ConnectorBase, "save_output", Mock())
    def test_predict_model_output(self):
        """Test predict_model_output with all parameters provided"""
        response = "test-response"
        self.connector_base.predict_model_output(response=response)

        expected_update = {"response": "test-response"}

        # Verify the output_config was updated correctly
        self.assertEqual(
            self.connector_base.output_config["predict_model"], expected_update
        )

        # Verify save_output was called with the updated config
        self.connector_base.save_output.assert_called_once_with(
            self.connector_base.output_config
        )

    @patch("builtins.input", return_value="")
    @patch("os.path.exists", return_value=False)
    @patch("os.path.abspath", return_value="/default/path/output.yml")
    @patch("os.makedirs")
    @patch("builtins.open", new_callable=mock_open)
    def test_save_output_default_path(
        self, mock_file, mock_makedirs, mock_abspath, mock_exists, mock_input
    ):
        """Test save_output with default path"""
        result = self.connector_base.save_output(self.test_config)

        # Verify the result
        self.assertEqual(result, "/default/path/output.yml")

        # Verify file operations
        mock_file.assert_called_with("/default/path/output.yml", "w")
        mock_file().write.assert_called()

        # Verify the OUTPUT_FILE was updated
        self.assertEqual(self.connector_base.OUTPUT_FILE, "/default/path/output.yml")

    @patch("builtins.input", return_value="/custom/path/config")
    @patch("os.path.exists", return_value=False)
    @patch("os.path.abspath", return_value="/custom/path/config.yaml")
    @patch("os.makedirs")
    @patch("builtins.open", new_callable=mock_open)
    def test_save_output_custom_path(
        self, mock_file, mock_makedirs, mock_abspath, mock_exists, mock_input
    ):
        """Test save_output with custom path"""
        result = self.connector_base.save_output(self.test_config)

        self.assertEqual(result, "/custom/path/config.yaml")
        mock_file.assert_called_with("/custom/path/config.yaml", "w")

    @patch("builtins.input")
    def test_save_output_adds_yaml_extension(self, mock_input):
        """Test save_output add .yaml extension when missing"""
        config = {"test": "data"}
        test_path = "/test/path/file"
        expected_path = "/test/path/file.yaml"

        mock_input.return_value = test_path

        with patch("os.path.exists") as mock_exists, patch(
            "os.path.abspath"
        ) as mock_abspath, patch("os.path.dirname") as mock_dirname, patch(
            "os.makedirs"
        ), patch(
            "builtins.open", mock_open()
        ), patch(
            "yaml.dump"
        ):

            mock_exists.return_value = False
            mock_abspath.side_effect = lambda x: x
            mock_dirname.return_value = "/test/path"
            result = self.connector_base.save_output(config)
            self.assertEqual(result, expected_path)

    @patch("builtins.input", return_value="")
    @patch("os.path.exists", return_value=True)
    @patch("os.path.abspath", return_value="/default/path/output.yml")
    @patch("os.path.dirname", return_value="/default/path")
    @patch("yaml.safe_load")
    @patch("builtins.open", new_callable=mock_open)
    def test_save_config_merge_config(
        self,
        mock_file,
        mock_yaml_load,
        mock_dirname,
        mock_abspath,
        mock_exists,
        mock_input,
    ):
        """Test save_output merging with existing config"""
        existing_config = {
            "section1": {"existing_key": "existing_value"},
            "section3": {"key3": "value3"},
        }
        mock_yaml_load.return_value = existing_config
        with patch("yaml.dump") as mock_yaml_dump:
            result = self.connector_base.save_output(self.test_config)
            self.assertEqual(result, "/default/path/output.yml")
            mock_yaml_dump.assert_called_once()
            actual_config = mock_yaml_dump.call_args[0][0]
            expected_merged_config = {
                "section1": {
                    "existing_key": "existing_value",
                    "key1": "value1",  # From test_config
                },
                "section2": {"key2": "value2"},  # From test_config
                "section3": {"key3": "value3"},  # From existing_config
            }
            self.assertEqual(actual_config, expected_merged_config)

    @patch("builtins.input", return_value="")
    @patch("os.path.exists", side_effect=PermissionError)
    @patch("os.path.abspath", return_value="/default/path/output.yml")
    def test_save_output_permission_error(self, mock_abspath, mock_exists, mock_input):
        """Test save_output with permission error"""
        result = self.connector_base.save_output(self.test_config)

        self.assertIsNone(result)

    @patch("builtins.input", side_effect=KeyboardInterrupt)
    def test_save_output_keyboard_interrupt(self, mock_input):
        """Test save_output with keyboard interrupt"""
        result = self.connector_base.save_output(self.test_config)

        self.assertIsNone(result)

    @patch("builtins.input", return_value="")
    @patch("os.path.exists", return_value=False)
    @patch("os.path.abspath", return_value="/default/path/output.yml")
    @patch("os.makedirs", side_effect=OSError("Test error"))
    def test_save_output_general_error(
        self, mock_makedirs, mock_abspath, mock_exists, mock_input
    ):
        """Test save_output with general error"""
        result = self.connector_base.save_output(self.test_config)

        self.assertIsNone(result)

    @patch("builtins.input", return_value="")
    @patch("os.path.exists", return_value=True)
    @patch("os.path.abspath", return_value="/default/path/output.yml")
    @patch(
        "builtins.open",
        side_effect=[
            mock_open(read_data="invalid: yaml: content: :").return_value,
            mock_open().return_value,
        ],
    )
    def test_save_output_invalid_existing_yaml(
        self, mock_file, mock_abspath, mock_exists, mock_input
    ):
        """Test save_output with invalid existing YAML"""
        result = self.connector_base.save_output(self.test_config)

        self.assertIsNone(result)

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
                result = self.connector_base.get_opensearch_domain_name(input_url)
                self.assertEqual(
                    result,
                    expected_output,
                    f"Failed for input '{input_url}'. Expected '{expected_output}', got '{result}'",
                )


if __name__ == "__main__":
    unittest.main()
