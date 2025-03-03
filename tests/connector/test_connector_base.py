# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

import configparser
import os
import unittest
from unittest.mock import patch

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
        config = self.setup_instance.load_config()
        self.assertEqual(config, {})

    def test_load_config_with_file(self):
        parser = configparser.ConfigParser()
        parser["DEFAULT"] = {"region": "us-east-1", "service_type": "managed"}
        with open(ConnectorBase.CONFIG_FILE, "w") as f:
            parser.write(f)
        config = self.setup_instance.load_config()
        self.assertEqual(config.get("region"), "us-east-1")
        self.assertEqual(config.get("service_type"), "managed")

    def test_save_config(self):
        config = {"key": "value", "another_key": "another_value"}
        self.setup_instance.save_config(config)
        parser = configparser.ConfigParser()
        parser.read(ConnectorBase.CONFIG_FILE)
        self.assertTrue(os.path.exists(ConnectorBase.CONFIG_FILE))
        self.assertEqual(parser["DEFAULT"]["key"], "value")
        self.assertEqual(parser["DEFAULT"]["another_key"], "another_value")

    def test_save_config_handles_error(self):
        with patch("builtins.open", side_effect=Exception("Test error")):
            config = {"key": "value"}
            self.setup_instance.save_config(config)
            self.assertFalse(os.path.exists(ConnectorBase.CONFIG_FILE))
