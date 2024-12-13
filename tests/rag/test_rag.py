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

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=InsecureRequestWarning)

# Import the main function from rag.py
from opensearch_py_ml.ml_commons.rag_pipeline.rag.rag import main


class TestRAGCLI(unittest.TestCase):
    def setUp(self):
        # Mock the config to avoid actual file operations
        self.mock_config = {
            "service_type": "managed",
            "region": "us-west-2",
            "default_search_method": "neural",
        }

        # Patch 'load_config' and 'save_config' functions
        self.patcher_load_config = patch(
            "opensearch_py_ml.ml_commons.rag_pipeline.rag.rag.load_config",
            return_value=self.mock_config,
        )
        self.mock_load_config = self.patcher_load_config.start()
        self.addCleanup(self.patcher_load_config.stop)

        self.patcher_save_config = patch(
            "opensearch_py_ml.ml_commons.rag_pipeline.rag.rag.save_config"
        )
        self.mock_save_config = self.patcher_save_config.start()
        self.addCleanup(self.patcher_save_config.stop)

        # Patch 'Setup', 'Ingest', and 'Query' classes
        self.patcher_setup = patch(
            "opensearch_py_ml.ml_commons.rag_pipeline.rag.rag.Setup"
        )
        self.mock_setup_class = self.patcher_setup.start()
        self.addCleanup(self.patcher_setup.stop)

        self.patcher_ingest = patch(
            "opensearch_py_ml.ml_commons.rag_pipeline.rag.rag.Ingest"
        )
        self.mock_ingest_class = self.patcher_ingest.start()
        self.addCleanup(self.patcher_ingest.stop)

        self.patcher_query = patch(
            "opensearch_py_ml.ml_commons.rag_pipeline.rag.rag.Query"
        )
        self.mock_query_class = self.patcher_query.start()
        self.addCleanup(self.patcher_query.stop)

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
        test_args = ["rag.py", "setup"]
        with patch.object(sys, "argv", test_args):
            main()
            # Ensure Setup.setup_command() is called
            self.mock_setup_class.return_value.setup_command.assert_called_once()
            # Ensure save_config is called
            self.mock_save_config.assert_called_once_with(
                self.mock_setup_class.return_value.config
            )

    def test_ingest_command_with_paths(self):
        test_args = ["rag.py", "ingest", "--paths", "/path/to/data1", "/path/to/data2"]
        with patch.object(sys, "argv", test_args):
            main()
            # Ensure Ingest.ingest_command() is called with correct paths
            self.mock_ingest_class.assert_called_once_with(self.mock_config)
            self.mock_ingest_class.return_value.ingest_command.assert_called_once_with(
                ["/path/to/data1", "/path/to/data2"]
            )

    def test_ingest_command_without_paths(self):
        test_args = ["rag.py", "ingest"]
        with patch.object(sys, "argv", test_args):
            with patch("rich.prompt.Prompt.ask", side_effect=["/path/to/data", ""]):
                main()
                # Ensure Ingest.ingest_command() is called with prompted paths
                self.mock_ingest_class.assert_called_once_with(self.mock_config)
                self.mock_ingest_class.return_value.ingest_command.assert_called_once_with(
                    ["/path/to/data"]
                )

    def test_query_command_with_queries(self):
        test_args = [
            "rag.py",
            "query",
            "--queries",
            "What is OpenSearch?",
            "How does Bedrock work?",
        ]
        with patch.object(sys, "argv", test_args):
            main()
            # Ensure Query.query_command() is called with correct queries
            self.mock_query_class.assert_called_once_with(self.mock_config)
            self.mock_query_class.return_value.query_command.assert_called_once_with(
                ["What is OpenSearch?", "How does Bedrock work?"], num_results=5
            )

    def test_query_command_without_queries(self):
        test_args = ["rag.py", "query"]
        with patch.object(sys, "argv", test_args):
            with patch(
                "rich.prompt.Prompt.ask", side_effect=["What is OpenSearch?", ""]
            ):
                main()
                # Ensure Query.query_command() is called with prompted queries
                self.mock_query_class.assert_called_once_with(self.mock_config)
                self.mock_query_class.return_value.query_command.assert_called_once_with(
                    ["What is OpenSearch?"], num_results=5
                )

    def test_query_command_with_num_results(self):
        test_args = [
            "rag.py",
            "query",
            "--queries",
            "What is OpenSearch?",
            "--num_results",
            "3",
        ]
        with patch.object(sys, "argv", test_args):
            main()
            # Ensure Query.query_command() is called with correct num_results
            self.mock_query_class.return_value.query_command.assert_called_once_with(
                ["What is OpenSearch?"], num_results=3
            )

    def test_no_command(self):
        test_args = ["rag.py"]
        with patch.object(sys, "argv", test_args):
            with self.assertRaises(SystemExit) as cm:
                main()
            self.assertEqual(cm.exception.code, 1)
            stderr_output = self.held_stderr.getvalue()
            stdout_output = self.held_stdout.getvalue()
            print("STDERR:", stderr_output)
            print("STDOUT:", stdout_output)
            self.assertTrue(
                "usage: rag.py" in stderr_output or "usage: rag.py" in stdout_output
            )

    def test_invalid_command(self):
        test_args = ["rag.py", "invalid"]
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
