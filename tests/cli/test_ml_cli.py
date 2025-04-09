# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

import os
import sys
import unittest
import warnings
from io import StringIO
from unittest.mock import mock_open, patch

from colorama import Fore, Style
from urllib3.exceptions import InsecureRequestWarning

from opensearch_py_ml.ml_commons.cli.ml_cli import main

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=InsecureRequestWarning)


class TestMLCLI(unittest.TestCase):
    def setUp(self):
        self.mock_config = {
            "service_type": "amazon-opensearch-service",
            "region": "us-west-2",
        }

        # Patch 'Setup', 'ConnectorManager', 'ModelManager' classes
        self.patcher_setup = patch("opensearch_py_ml.ml_commons.cli.ml_cli.Setup")
        self.mock_setup_class = self.patcher_setup.start()
        self.addCleanup(self.patcher_setup.stop)

        self.patcher_connector = patch(
            "opensearch_py_ml.ml_commons.cli.ml_cli.ConnectorManager"
        )
        self.mock_connector_class = self.patcher_connector.start()
        self.addCleanup(self.patcher_connector.stop)

        self.patcher_model = patch(
            "opensearch_py_ml.ml_commons.cli.ml_cli.ModelManager"
        )
        self.mock_model_class = self.patcher_model.start()
        self.addCleanup(self.patcher_model.stop)

        # Capture stdout
        self.held_stdout = StringIO()
        self.patcher_stdout = patch("sys.stdout", new=self.held_stdout)
        self.patcher_stdout.start()
        self.addCleanup(self.patcher_stdout.stop)

        # Capture stderr
        self.held_stderr = StringIO()
        self.patcher_stderr = patch("sys.stderr", new=self.held_stderr)
        self.patcher_stderr.start()
        self.addCleanup(self.patcher_stderr.stop)

    def test_setup_command(self):
        """Test setup command"""
        test_args = ["ml_cli.py", "setup"]
        self.mock_setup_class.return_value.setup_command.return_value = (
            "test_config.yml"
        )
        with patch.object(sys, "argv", test_args):
            with patch("os.makedirs") as mock_makedirs, patch(
                "builtins.open", mock_open()
            ) as mock_file:
                main()
                self.mock_setup_class.return_value.setup_command.assert_called_once()
                mock_makedirs.assert_called_once_with(
                    os.path.expanduser("~/.opensearch-ml"), exist_ok=True
                )
                mock_file.assert_called_once_with(
                    os.path.join(os.path.expanduser("~/.opensearch-ml"), "config_path"),
                    "w",
                )
                mock_file().write.assert_called_once_with("test_config.yml")

    def test_setup_command_with_path(self):
        """Test setup command with path"""
        test_args = ["ml_cli.py", "setup", "--path", "test_path"]
        with patch.object(sys, "argv", test_args), patch("builtins.open", mock_open()):
            main()
            self.mock_setup_class.return_value.setup_command.assert_called_once_with(
                config_path="test_path"
            )

    def test_connector_initialize_create_connector(self):
        """Test connector_create command"""
        test_args = ["ml_cli.py", "connector", "create"]
        self.mock_connector_class.return_value.initialize_create_connector.return_value = (
            None,
            "test_config.yml",
        )
        with patch.object(sys, "argv", test_args):
            main()
            self.mock_connector_class.return_value.initialize_create_connector.assert_called_once()

    def test_connector_initialize_create_connector_with_path(self):
        """Test connector_create command with path"""
        test_args = ["ml_cli.py", "connector", "create", "--path", "test_path"]
        self.mock_connector_class.return_value.initialize_create_connector.return_value = (
            None,
            "test_config.yml",
        )
        with patch.object(sys, "argv", test_args), patch("builtins.open", mock_open()):
            main()
            self.mock_connector_class.return_value.initialize_create_connector.assert_called_once_with(
                connector_config_path="test_path"
            )

    @patch("argparse.ArgumentParser.print_help")
    @patch("sys.exit")
    def test_invalid_connector_subcommand(self, mock_exit, mock_print_help):
        """Test invalid connector subcommand"""
        test_args = ["ml_cli.py", "connector", "invalid_arg"]
        with patch.object(sys, "argv", test_args):
            main()
            mock_print_help.assert_called_once()
            mock_exit.assert_any_call(1)

    def test_model_initialize_register_model(self):
        """Test model_register command"""
        test_args = ["ml_cli.py", "model", "register"]
        with patch.object(sys, "argv", test_args):
            main()
            self.mock_model_class.return_value.initialize_register_model.assert_called_once()

    def test_model_initialize_register_model_with_arguments(self):
        """Test model_register command with connector ID, model name, and model description"""
        test_args = [
            "ml_cli.py",
            "model",
            "register",
            "--connectorId",
            "connector123",
            "--name",
            "test model",
            "--description",
            "test description",
        ]
        with patch.object(sys, "argv", test_args):
            main()
            self.mock_model_class.return_value.initialize_register_model.assert_called_once_with(
                "test_config.yml",
                connector_id="connector123",
                model_name="test model",
                model_description="test description",
            )

    @patch("builtins.print")
    @patch("sys.exit")
    def test_model_initialize_register_model_file_not_found(
        self, mock_exit, mock_print
    ):
        """Test model_register command file not found handling"""
        test_args = ["ml_cli.py", "model", "register"]
        with patch.object(sys, "argv", test_args), patch(
            "os.path.expanduser", return_value="/mock/home/.opensearch-ml"
        ), patch("builtins.open", side_effect=FileNotFoundError()):

            main()
            mock_print.assert_called_once_with(
                f"{Fore.RED}No setup configuration found. Please run setup first.{Style.RESET_ALL}"
            )
            mock_exit.assert_called_once_with(1)

    def test_model_initialize_predict_model(self):
        """Test model_predict command"""
        test_args = ["ml_cli.py", "model", "predict"]
        with patch.object(sys, "argv", test_args):
            main()
            self.mock_model_class.return_value.initialize_predict_model.assert_called_once()

    def test_model_initialize_predict_model_with_arguments(self):
        """Test model_predict command with model ID and predict payload"""
        test_args = [
            "ml_cli.py",
            "model",
            "predict",
            "--modelId",
            "model123",
            "--body",
            '{"parameters": {}}',
        ]
        with patch.object(sys, "argv", test_args):
            main()
            self.mock_model_class.return_value.initialize_predict_model.assert_called_once_with(
                "test_config.yml",
                model_id="model123",
                body='{"parameters": {}}',
            )

    @patch("builtins.print")
    @patch("sys.exit")
    def test_model_initialize_predict_model_file_not_found(self, mock_exit, mock_print):
        """Test model_predict command file not found handling"""
        test_args = ["ml_cli.py", "model", "predict"]
        with patch.object(sys, "argv", test_args), patch(
            "os.path.expanduser", return_value="/mock/home/.opensearch-ml"
        ), patch("builtins.open", side_effect=FileNotFoundError()):

            main()
            mock_print.assert_called_once_with(
                f"{Fore.RED}No setup configuration found. Please run setup first.{Style.RESET_ALL}"
            )
            mock_exit.assert_called_once_with(1)

    @patch("argparse.ArgumentParser.print_help")
    @patch("sys.exit")
    def test_invalid_model_subcommand(self, mock_exit, mock_print_help):
        """Test invalid model subcommand"""
        test_args = ["ml_cli.py", "model", "invalid_arg"]
        with patch.object(sys, "argv", test_args):
            main()
            mock_print_help.assert_called_once()
            mock_exit.assert_any_call(1)

    def test_help_command(self):
        """Test help command"""
        test_args = ["ml_cli.py", "--help"]
        with patch.object(sys, "argv", test_args):
            with self.assertRaises(SystemExit) as cm:
                main()
            self.assertEqual(cm.exception.code, 0)
            output = self.held_stdout.getvalue()
            self.assertIn("OpenSearch ML CLI", output)
            self.assertIn("Available Commands", output)

    def test_no_command(self):
        """Test no command"""
        test_args = ["ml_cli.py"]
        with patch.object(sys, "argv", test_args):
            with self.assertRaises(SystemExit) as cm:
                main()
            self.assertEqual(cm.exception.code, 1)
            stdout_output = self.held_stdout.getvalue()
            stderr_output = self.held_stderr.getvalue()

            self.assertIn("OpenSearch ML CLI", stdout_output)
            self.assertIn("opensearch-ml setup", stdout_output)
            self.assertIn("opensearch-ml connector create", stdout_output)
            print("STDERR:", stderr_output)
            self.assertTrue(
                "usage: opensearch-ml" in stderr_output
                or "usage: opensearch-ml" in stdout_output
            )

    def test_invalid_command(self):
        """Test invalid command"""
        test_args = ["ml_cli.py", "invalid"]
        with patch.object(sys, "argv", test_args):
            with self.assertRaises(SystemExit) as cm:
                main()
            self.assertEqual(cm.exception.code, 2)
            stderr_output = self.held_stderr.getvalue()
            stdout_output = self.held_stdout.getvalue()
            print("STDERR:", stderr_output)
            print("STDOUT:", stdout_output)
            self.assertTrue(
                "invalid choice: 'invalid'" in stderr_output
                or "invalid choice: 'invalid'" in stdout_output
            )

    def test_dash_prefixed_connector_id(self):
        """Test dash-prefixed connector ID handling"""
        test_args = [
            "ml_cli.py",
            "model",
            "register",
            "--connectorId",
            "-connector123",
            "--name",
            "test model",
            "--description",
            "test description",
        ]
        with patch.object(sys, "argv", test_args):
            main()
            self.mock_model_class.return_value.initialize_register_model.assert_called_once_with(
                "test_config.yml",
                connector_id="-connector123",
                model_name="test model",
                model_description="test description",
            )

    @patch("argparse.ArgumentParser.error")
    def test_connector_id_missing_value(self, mock_error):
        """Test connector ID handling when value is missing"""
        test_args = ["ml_cli.py", "model", "register", "--connectorId"]
        with patch.object(sys, "argv", test_args):
            main()
            mock_error.assert_called_once_with("--connectorId requires a value")

    def test_dash_prefixed_model_id(self):
        """Test dash-prefixed model ID handling"""
        test_args = [
            "ml_cli.py",
            "model",
            "predict",
            "--modelId",
            "-model123",
            "--body",
            '{"parameters": {}}',
        ]
        with patch.object(sys, "argv", test_args):
            main()
            self.mock_model_class.return_value.initialize_predict_model.assert_called_once_with(
                "test_config.yml",
                model_id="-model123",
                body='{"parameters": {}}',
            )

    @patch("argparse.ArgumentParser.error")
    def test_model_id_missing_value(self, mock_error):
        """Test model ID handling when value is missing"""
        test_args = [
            "ml_cli.py",
            "model",
            "predict",
            "--modelId",
        ]
        with patch.object(sys, "argv", test_args):
            main()
            mock_error.assert_called_once_with("--modelId requires a value")


if __name__ == "__main__":
    unittest.main()
