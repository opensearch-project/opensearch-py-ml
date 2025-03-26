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

        # Patch 'Setup', 'Create', 'Register', 'Predict' classes
        self.patcher_setup = patch("opensearch_py_ml.ml_commons.cli.ml_cli.Setup")
        self.mock_setup_class = self.patcher_setup.start()
        self.addCleanup(self.patcher_setup.stop)

        self.patcher_create = patch("opensearch_py_ml.ml_commons.cli.ml_cli.Create")
        self.mock_create_class = self.patcher_create.start()
        self.addCleanup(self.patcher_create.stop)

        self.patcher_register = patch("opensearch_py_ml.ml_commons.cli.ml_cli.Register")
        self.mock_register_class = self.patcher_register.start()
        self.addCleanup(self.patcher_register.stop)

        self.patcher_predict = patch("opensearch_py_ml.ml_commons.cli.ml_cli.Predict")
        self.mock_predict_class = self.patcher_predict.start()
        self.addCleanup(self.patcher_predict.stop)

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
        test_args = ["ml_cli.py", "setup", "--path", "test_path"]
        with patch.object(sys, "argv", test_args), patch("builtins.open", mock_open()):
            main()
            self.mock_setup_class.return_value.setup_command.assert_called_once_with(
                config_path="test_path"
            )

    def test_connector_create_command(self):
        test_args = ["ml_cli.py", "connector", "create"]
        self.mock_create_class.return_value.create_command.return_value = (
            None,
            "test_config.yml",
        )
        with patch.object(sys, "argv", test_args):
            main()
            self.mock_create_class.return_value.create_command.assert_called_once()

    def test_connector_create_command_with_path(self):
        test_args = ["ml_cli.py", "connector", "create", "--path", "test_path"]
        self.mock_create_class.return_value.create_command.return_value = (
            None,
            "test_config.yml",
        )
        with patch.object(sys, "argv", test_args), patch("builtins.open", mock_open()):
            main()
            self.mock_create_class.return_value.create_command.assert_called_once_with(
                connector_config_path="test_path"
            )

    def test_model_register_command(self):
        test_args = ["ml_cli.py", "model", "register"]
        with patch.object(sys, "argv", test_args):
            main()
            self.mock_register_class.return_value.register_command.assert_called_once()

    def test_model_register_command_with_arguments(self):
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
            self.mock_register_class.return_value.register_command.assert_called_once_with(
                "test_config.yml",
                connector_id="connector123",
                model_name="test model",
                model_description="test description",
            )

    def test_model_predict_command(self):
        test_args = ["ml_cli.py", "model", "predict"]
        with patch.object(sys, "argv", test_args):
            main()
            self.mock_predict_class.return_value.predict_command.assert_called_once()

    def test_model_predict_command_with_arguments(self):
        test_args = [
            "ml_cli.py",
            "model",
            "predict",
            "--modelId",
            "model123",
            "--payload",
            '{"parameters": {}}',
        ]
        with patch.object(sys, "argv", test_args):
            main()
            self.mock_predict_class.return_value.predict_command.assert_called_once_with(
                "test_config.yml",
                model_id="model123",
                payload='{"parameters": {}}',
            )

    def test_help_command(self):
        test_args = ["ml_cli.py", "--help"]
        with patch.object(sys, "argv", test_args):
            with self.assertRaises(SystemExit) as cm:
                main()
            self.assertEqual(cm.exception.code, 0)
            output = self.held_stdout.getvalue()
            self.assertIn("OpenSearch ML CLI", output)
            self.assertIn("Available Commands", output)

    def test_no_command(self):
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


if __name__ == "__main__":
    unittest.main()
