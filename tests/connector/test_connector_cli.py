# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

import sys
import unittest
import warnings
from io import StringIO
from unittest.mock import patch

from urllib3.exceptions import InsecureRequestWarning

from opensearch_py_ml.ml_commons.connector.connector_cli import main

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=InsecureRequestWarning)


class TestConnectorCLI(unittest.TestCase):
    def setUp(self):
        # Mock the config to avoid actual file operations
        self.mock_config = {
            "service_type": "managed",
            "region": "us-west-2",
        }

        # Patch 'Setup' and 'Create' classes
        self.patcher_setup = patch(
            "opensearch_py_ml.ml_commons.connector.connector_cli.Setup"
        )
        self.mock_setup_class = self.patcher_setup.start()
        self.addCleanup(self.patcher_setup.stop)

        self.patcher_create = patch(
            "opensearch_py_ml.ml_commons.connector.connector_cli.Create"
        )
        self.mock_create_class = self.patcher_create.start()
        self.addCleanup(self.patcher_create.stop)

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
        test_args = ["connector_cli.py", "setup"]
        with patch.object(sys, "argv", test_args):
            main()
            # Ensure Setup.setup_command() is called
            self.mock_setup_class.return_value.setup_command.assert_called_once()

    def test_create_command(self):
        test_args = ["connector_cli.py", "create"]
        with patch.object(sys, "argv", test_args):
            main()
            # Ensure Create.create_command() is called
            self.mock_create_class.return_value.create_command.assert_called_once()

    def test_help_command(self):
        test_args = ["connector_cli.py", "--help"]
        with patch.object(sys, "argv", test_args):
            with self.assertRaises(SystemExit) as cm:
                main()
            self.assertEqual(cm.exception.code, 0)
            output = self.held_stdout.getvalue()
            self.assertIn("Connector Creation CLI", output)
            self.assertIn("Available Commands", output)

    def test_no_command(self):
        test_args = ["connector_cli.py"]
        with patch.object(sys, "argv", test_args):
            with self.assertRaises(SystemExit) as cm:
                main()
            self.assertEqual(cm.exception.code, 1)
            stdout_output = self.held_stdout.getvalue()
            stderr_output = self.held_stderr.getvalue()

            self.assertIn("Welcome to the Connector Creation", stdout_output)
            self.assertIn("connector setup", stdout_output)
            self.assertIn("connector create", stdout_output)
            self.assertTrue(
                "usage: connector_cli.py" in stderr_output
                or "usage: connector_cli.py" in stdout_output
            )

    def test_invalid_command(self):
        test_args = ["connector_cli.py", "invalid"]
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

    def test_multiple_commands(self):
        test_args = ["connector_cli.py", "setup", "create"]
        with patch.object(sys, "argv", test_args):
            with self.assertRaises(SystemExit):
                main()


if __name__ == "__main__":
    unittest.main()
