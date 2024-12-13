# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

import configparser
import os
import unittest
from unittest.mock import MagicMock, patch

# Adjust the import path to wherever Setup is actually defined
from opensearch_py_ml.ml_commons.rag_pipeline.rag.rag_setup import Setup


class TestSetup(unittest.TestCase):

    def setUp(self):
        if os.path.exists(Setup.CONFIG_FILE):
            os.remove(Setup.CONFIG_FILE)
        self.setup_instance = Setup()

    def tearDown(self):
        if os.path.exists(Setup.CONFIG_FILE):
            os.remove(Setup.CONFIG_FILE)

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

    def test_load_config_no_file(self):
        config = self.setup_instance.load_config()
        self.assertEqual(config, {})

    def test_load_config_with_file(self):
        parser = configparser.ConfigParser()
        parser["DEFAULT"] = {"region": "us-east-1", "service_type": "managed"}
        with open(Setup.CONFIG_FILE, "w") as f:
            parser.write(f)
        config = self.setup_instance.load_config()
        self.assertEqual(config.get("region"), "us-east-1")
        self.assertEqual(config.get("service_type"), "managed")

    @patch("builtins.input", side_effect=["2", "", "no", "2", ""])
    def test_setup_configuration_open_source_no_auth(self, mock_input):
        self.setup_instance.setup_configuration()
        config = self.setup_instance.config
        self.assertEqual(config["service_type"], "open-source")
        self.assertEqual(config["opensearch_username"], "")
        self.assertEqual(config["opensearch_password"], "")

    @patch("boto3.client")
    def test_initialize_clients_managed(self, mock_boto_client):
        self.setup_instance.service_type = "managed"
        self.setup_instance.aws_region = "us-west-2"
        mock_boto_client.return_value = MagicMock()
        result = self.setup_instance.initialize_clients()
        self.assertTrue(result)

    def test_initialize_clients_open_source(self):
        self.setup_instance.service_type = "open-source"
        result = self.setup_instance.initialize_clients()
        self.assertTrue(result)

    def test_get_opensearch_domain_name(self):
        self.setup_instance.opensearch_endpoint = (
            "https://search-my-domain-name-abc123.us-west-2.es.amazonaws.com"
        )
        domain_name = self.setup_instance.get_opensearch_domain_name()
        self.assertEqual(domain_name, "my-domain-name")

    @patch("boto3.client")
    def test_get_opensearch_domain_info(self, mock_boto_client):
        mock_client = MagicMock()
        mock_boto_client.return_value = mock_client
        mock_client.describe_domain.return_value = {
            "DomainStatus": {"Endpoint": "search-endpoint", "ARN": "test-arn"}
        }
        endpoint, arn = self.setup_instance.get_opensearch_domain_info(
            "us-west-2", "mydomain"
        )
        self.assertEqual(endpoint, "search-endpoint")
        self.assertEqual(arn, "test-arn")

    @patch("opensearch_py_ml.ml_commons.rag_pipeline.rag.rag_setup.OpenSearch")
    def test_initialize_opensearch_client_managed(self, mock_opensearch):
        self.setup_instance.service_type = "managed"
        self.setup_instance.opensearch_endpoint = "https://test-domain:443"
        self.setup_instance.opensearch_username = "admin"
        self.setup_instance.opensearch_password = "pass"
        result = self.setup_instance.initialize_opensearch_client()
        self.assertTrue(result)
        mock_opensearch.assert_called_once()

    @patch("opensearch_py_ml.ml_commons.rag_pipeline.rag.rag_setup.OpenSearch")
    def test_initialize_opensearch_client_open_source_no_auth(self, mock_opensearch):
        self.setup_instance.service_type = "open-source"
        self.setup_instance.opensearch_endpoint = "http://localhost:9200"
        self.setup_instance.opensearch_username = ""
        self.setup_instance.opensearch_password = ""
        result = self.setup_instance.initialize_opensearch_client()
        self.assertTrue(result)
        mock_opensearch.assert_called_once()

    @patch("builtins.input", side_effect=["", "", "", "", "", "", "", ""])
    def test_get_knn_index_details_all_defaults(self, mock_input):
        details = self.setup_instance.get_knn_index_details()
        self.assertEqual(
            details,
            (
                768,
                "l2",
                512,
                1,
                2,
                "passage_text",
                "passage_chunk",
                "passage_embedding",
            ),
        )

    def test_save_config(self):
        config = {"key": "value", "another_key": "another_value"}
        self.setup_instance.save_config(config)
        parser = configparser.ConfigParser()
        parser.read(Setup.CONFIG_FILE)
        self.assertEqual(parser["DEFAULT"]["key"], "value")
        self.assertEqual(parser["DEFAULT"]["another_key"], "another_value")


if __name__ == "__main__":
    unittest.main()
