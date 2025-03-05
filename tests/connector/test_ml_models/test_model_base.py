# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

import unittest
from unittest.mock import patch

from opensearch_py_ml.ml_commons.connector.ml_models.model_base import ModelBase


class TestModelBase(unittest.TestCase):

    def setUp(self):
        self.setup_instance = ModelBase()

    @patch("builtins.input", return_value="1")
    def test_get_custom_model_details_default(self, mock_input):
        default_input = {"name": "Default Model"}
        result = self.setup_instance.get_custom_model_details(default_input)
        self.assertEqual(result, default_input)

    @patch("builtins.input", side_effect=["2", '{"name": "Custom Model"}'])
    def test_get_custom_model_details_custom(self, mock_input):
        default_input = {"name": "Default Model"}
        result = self.setup_instance.get_custom_model_details(default_input)
        self.assertEqual(result, {"name": "Custom Model"})

    @patch("builtins.input", return_value="2\n{invalid json}")
    def test_get_custom_model_details_invalid_json(self, mock_input):
        default_input = {"name": "Default Model"}
        result = self.setup_instance.get_custom_model_details(default_input)
        self.assertIsNone(result)

    @patch("builtins.input", return_value="3")
    def test_get_custom_model_details_invalid_choice(self, mock_input):
        default_input = {"name": "Default Model"}
        result = self.setup_instance.get_custom_model_details(default_input)
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
