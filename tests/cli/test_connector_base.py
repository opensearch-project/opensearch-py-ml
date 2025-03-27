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

from opensearch_py_ml.ml_commons.cli.connector_base import ConnectorBase


class TestConnectorBase(unittest.TestCase):
    def setUp(self):
        if os.path.exists(ConnectorBase.CONFIG_FILE):
            os.remove(ConnectorBase.CONFIG_FILE)
        self.connector_base = ConnectorBase()
        # self.connector_base.save_output = Mock()
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
        config = self.connector_base.load_config("nonexistent_config.yaml")
        self.assertEqual(config, {})
        mock_print.assert_called_once()
        self.assertIn("Configuration file not found", mock_print.call_args[0][0])

    @patch("builtins.print")
    def test_load_config_valid_yaml(self, mock_print):
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
        with open(ConnectorBase.CONFIG_FILE, "w") as f:
            f.write("invalid: yaml: content:")

        config = self.connector_base.load_config(ConnectorBase.CONFIG_FILE)
        self.assertEqual(config, {})
        mock_print.assert_called_once()
        self.assertIn("Error parsing YAML configuration", mock_print.call_args[0][0])

    def test_load_connector_config_valid_yaml(self):
        """Test loading a valid YAML connector configuration"""
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

    def test_load_connector_config_invalid_yaml(self):
        """Test loading an invalid YAML file"""
        invalid_yaml = "invalid: yaml: content: :"

        with patch("builtins.open", mock_open(read_data=invalid_yaml)):
            result = self.connector_base.load_connector_config("invalid_config.yml")
            self.assertEqual(result, {})

    def test_load_connector_config_file_not_found(self):
        """Test loading a non-existent file"""
        with patch("builtins.open", mock_open()) as mock_file:
            mock_file.side_effect = FileNotFoundError()
            result = self.connector_base.load_connector_config("nonexistent.yml")
            self.assertEqual(result, {})

    @patch("builtins.print")
    @patch("builtins.input", return_value="")
    def test_save_config(self, mock_input, mock_print):
        test_config = {"key": "value"}
        save_result = self.connector_base.save_config(test_config)
        mock_print.assert_called_once()
        self.assertTrue(save_result)
        self.assertTrue(os.path.exists(ConnectorBase.CONFIG_FILE))
        self.assertIn("Configuration saved successfully", mock_print.call_args[0][0])

    @patch("builtins.print")
    @patch("builtins.input", return_value="")
    def test_save_config_error(self, mock_input, mock_print):
        with patch("builtins.open", side_effect=Exception("Test error")):
            config = {"key": "value"}
            self.connector_base.save_config(config)
            mock_print.assert_called_once()
            self.assertFalse(os.path.exists(ConnectorBase.CONFIG_FILE))
            self.assertIn("Error saving configuration", mock_print.call_args[0][0])

    def test_update_config(self):
        """Test successful config update"""
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
        """Test config update with permission error"""
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
        """Test config update with YAML error"""
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
        """Test config update with invalid path"""
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
        """Test config update with empty config"""
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

    @patch("builtins.input", return_value="")
    @patch("os.path.exists", return_value=True)
    @patch("os.path.abspath", return_value="/default/path/output.yml")
    @patch("builtins.open", new_callable=mock_open)
    def s(self, mock_file, mock_abspath, mock_exists, mock_input):
        """Test save_output merging with existing config"""
        # Setup existing config
        existing_config = {
            "section1": {"existing_key": "existing_value"},
            "section3": {"key3": "value3"},
        }
        mock_file.return_value.__enter__().read.return_value = yaml.dump(
            existing_config
        )

        result = self.connector_base.save_output(self.test_config)

        self.assertEqual(result, "/default/path/output.yml")
        # Verify the merged config was written
        mock_file.assert_called_with("/default/path/output.yml", "w")

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
        """Test get_opensearch_domain_name with various inputs, including edge cases and error cases"""
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
