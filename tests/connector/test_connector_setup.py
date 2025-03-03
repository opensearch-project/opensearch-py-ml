# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.


import os
import unittest
from unittest.mock import MagicMock, patch

from opensearch_py_ml.ml_commons.connector.connector_base import ConnectorBase
from opensearch_py_ml.ml_commons.connector.connector_setup import Setup


class TestSetup(unittest.TestCase):
    def setUp(self):
        if os.path.exists(ConnectorBase.CONFIG_FILE):
            os.remove(ConnectorBase.CONFIG_FILE)
        self.setup_instance = Setup()

    def tearDown(self):
        if os.path.exists(ConnectorBase.CONFIG_FILE):
            os.remove(ConnectorBase.CONFIG_FILE)

    @patch("builtins.input", return_value="fake_input")
    @patch("subprocess.run")
    def test_configure_aws(self, mock_subprocess, mock_input):
        with patch.object(
            self.setup_instance,
            "get_password_with_asterisks",
            return_value="fake_secret",
        ):
            self.setup_instance.configure_aws()
            self.assertEqual(mock_subprocess.call_count, 3)

    @patch("boto3.client")
    @patch("builtins.input")
    def test_aws_credentials_exist_no_reconfigure(self, mock_input, mock_boto_client):
        mock_boto_client.return_value.get_credentials.return_value = MagicMock()
        mock_input.return_value = "no"
        with patch.object(self.setup_instance, "configure_aws") as mock_configure:
            self.setup_instance.check_and_configure_aws()
            mock_configure.assert_not_called()

    @patch("builtins.input", side_effect=["2", "", "yes", "admin"])
    @patch.object(Setup, "get_password_with_asterisks", return_value="pass")
    def test_setup_configuration_open_source_with_auth(
        self, mock_get_password, mock_input
    ):
        self.setup_instance.setup_configuration()
        config = self.setup_instance.config
        self.assertEqual(config["service_type"], "open-source")
        self.assertEqual(config["opensearch_domain_username"], "admin")
        self.assertEqual(config["opensearch_domain_password"], "pass")

    @patch("builtins.input", side_effect=["2", "", "no", "2", ""])
    def test_setup_configuration_open_source_no_auth(self, mock_input):
        self.setup_instance.setup_configuration()
        config = self.setup_instance.config
        self.assertEqual(config["service_type"], "open-source")
        self.assertEqual(config["opensearch_domain_username"], "")
        self.assertEqual(config["opensearch_domain_password"], "")

    def test_get_opensearch_domain_name(self):
        self.setup_instance.opensearch_domain_endpoint = (
            "https://search-my-domain-name-abc123.us-west-2.es.amazonaws.com"
        )
        domain_name = self.setup_instance.get_opensearch_domain_name()
        self.assertEqual(domain_name, "my-domain-name")

    @patch("opensearch_py_ml.ml_commons.connector.connector_setup.OpenSearch")
    def test_initialize_opensearch_client_managed(self, mock_opensearch):
        self.setup_instance.service_type = "managed"
        self.setup_instance.opensearch_domain_endpoint = "https://test-domain:443"
        self.setup_instance.opensearch_domain_username = "admin"
        self.setup_instance.opensearch_domain_password = "pass"
        result = self.setup_instance.initialize_opensearch_client()
        self.assertTrue(result)
        mock_opensearch.assert_called_once()

    @patch("opensearch_py_ml.ml_commons.connector.connector_setup.OpenSearch")
    def test_initialize_opensearch_client_open_source_no_auth(self, mock_opensearch):
        self.setup_instance.service_type = "open-source"
        self.setup_instance.opensearch_domain_endpoint = "http://localhost:9200"
        self.setup_instance.opensearch_domain_username = ""
        self.setup_instance.opensearch_domain_password = ""
        result = self.setup_instance.initialize_opensearch_client()
        self.assertTrue(result)
        mock_opensearch.assert_called_once()

    @patch("opensearch_py_ml.ml_commons.connector.connector_setup.OpenSearch")
    def test_initialize_opensearch_client_open_source_with_auth(self, mock_opensearch):
        self.setup_instance.service_type = "open-source"
        self.setup_instance.opensearch_domain_endpoint = "http://localhost:9200"
        self.setup_instance.opensearch_domain_username = "admin"
        self.setup_instance.opensearch_domain_password = "pass"
        result = self.setup_instance.initialize_opensearch_client()
        self.assertTrue(result)
        mock_opensearch.assert_called_once()


if __name__ == "__main__":
    unittest.main()
