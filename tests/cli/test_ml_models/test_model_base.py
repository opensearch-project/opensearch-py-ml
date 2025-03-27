# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

import unittest
from unittest.mock import patch

from colorama import Fore, Style

from opensearch_py_ml.ml_commons.cli.ml_models.model_base import ModelBase


class TestModelBase(unittest.TestCase):

    def setUp(self):
        self.model_base = ModelBase()
        self.valid_json = """
{
    "name": "Test Model",
    "description": "Test connector",
    "version": "1",
    "protocol": "http",
    "parameters": {
        "model": "test-model"
    },
    "actions": [
        {
            "action_type": "predict",
            "method": "POST",
            "url": "https://api.test.com/v1/predict",
            "headers": {
                "Authorization": "${auth}"
            }
        }
    ]
}"""
        self.invalid_json = """
{
    "name": Invalid JSON,
    "missing": "quotes"
}"""

    @patch("builtins.input")
    @patch("builtins.print")
    @patch("rich.console.Console.print")
    def test_input_custom_model_details_valid_json(
        self, mock_console_print, mock_print, mock_input
    ):
        """Test with valid JSON input."""
        json_lines = [
            line.strip() for line in self.valid_json.strip().split("\n") if line.strip()
        ]

        mock_input.side_effect = json_lines + [""]
        result = self.model_base.input_custom_model_details()

        # Verify the result is parsed correctly
        self.assertIsNotNone(result)
        self.assertEqual(result["name"], "Test Model")
        self.assertEqual(result["protocol"], "http")
        self.assertEqual(result["actions"][0]["headers"]["Authorization"], "${auth}")

        # Verify prints were called
        mock_print.assert_any_call("Please enter your model details as a JSON object.")
        mock_console_print.assert_any_call("[bold]Amazon OpenSearch Service:[/bold]")

    @patch("builtins.input")
    @patch("builtins.print")
    @patch("rich.console.Console.print")
    def test_input_custom_model_details_invalid_json(
        self, mock_console_print, mock_print, mock_input
    ):
        """Test with invalid JSON input."""
        json_lines = [
            line.strip()
            for line in self.invalid_json.strip().split("\n")
            if line.strip()
        ]

        mock_input.side_effect = json_lines + [""]
        result = self.model_base.input_custom_model_details()

        # Verify the result is None for invalid JSON
        self.assertIsNone(result)

        # Verify error message was printed
        error_calls = [
            call
            for call in mock_print.call_args_list
            if call[0]
            and isinstance(call[0][0], str)
            and call[0][0].startswith("Invalid JSON input")
        ]

        self.assertTrue(error_calls)

    @patch("builtins.input")
    @patch("builtins.print")
    @patch("rich.console.Console.print")
    def test_input_custom_model_details_external_true(
        self, mock_console_print, mock_print, mock_input
    ):
        """Test with external=True parameter."""
        json_lines = [
            line.strip() for line in self.valid_json.strip().split("\n") if line.strip()
        ]

        mock_input.side_effect = json_lines + [""]
        result = self.model_base.input_custom_model_details(external=True)

        # Verify the result is parsed correctly
        self.assertIsNotNone(result)

        # Verify external-specific messages were printed
        mock_print.assert_any_call(
            f"{Fore.YELLOW}\nIMPORTANT: When customizing the connector configuration, ensure you include the following in the 'headers' section:"
        )
        mock_print.assert_any_call(
            f'{Fore.YELLOW}{Style.BRIGHT}"Authorization": "${{auth}}"'
        )


if __name__ == "__main__":
    unittest.main()
