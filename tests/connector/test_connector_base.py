# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

import os
import unittest
from unittest.mock import patch

import yaml

from opensearch_py_ml.ml_commons.connector.connector_base import ConnectorBase


class TestConnectorBase(unittest.TestCase):
    def setUp(self):
        if os.path.exists(ConnectorBase.CONFIG_FILE):
            os.remove(ConnectorBase.CONFIG_FILE)
        self.setup_instance = ConnectorBase()

    def tearDown(self):
        if os.path.exists(ConnectorBase.CONFIG_FILE):
            os.remove(ConnectorBase.CONFIG_FILE)

    def test_load_config_no_file(self):
        config = self.setup_instance.load_config("nonexistent_config.yaml")
        self.assertEqual(config, {})

    def test_load_config_valid_yaml(self):
        test_config = {
            "service_type": "managed",
            "opensearch_domain_region": "test-region",
        }
        with open(ConnectorBase.CONFIG_FILE, "w") as f:
            yaml.dump(test_config, f)
        config = self.setup_instance.load_config(ConnectorBase.CONFIG_FILE)
        self.assertEqual(config, test_config)
        self.assertEqual(self.setup_instance.config, test_config)

    def test_load_config_invalid_yaml(self):
        with open(ConnectorBase.CONFIG_FILE, "w") as f:
            f.write("invalid: yaml: content:")

        config = self.setup_instance.load_config(ConnectorBase.CONFIG_FILE)
        self.assertEqual(config, {})

    @patch("builtins.print")
    def test_load_config_file_not_found_message(self, mock_print):
        self.setup_instance.load_config("nonexistent_config.yaml")
        mock_print.assert_called_once()
        self.assertIn("Configuration file not found", mock_print.call_args[0][0])

    @patch("builtins.print")
    def test_save_config(self, mock_print):
        test_config = {"key": "value"}
        save_result = self.setup_instance.save_config(test_config)
        mock_print.assert_called_once()
        self.assertTrue(save_result)
        self.assertTrue(os.path.exists(ConnectorBase.CONFIG_FILE))
        self.assertIn("Configuration saved successfully", mock_print.call_args[0][0])

    @patch("builtins.print")
    def test_save_config_handles_error(self, mock_print):
        with patch("builtins.open", side_effect=Exception("Test error")):
            config = {"key": "value"}
            self.setup_instance.save_config(config)
            mock_print.assert_called_once()
            self.assertFalse(os.path.exists(ConnectorBase.CONFIG_FILE))
            self.assertIn("Error saving configuration", mock_print.call_args[0][0])


if __name__ == "__main__":
    unittest.main()
