# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.


import os
import unittest
from unittest.mock import MagicMock, call, patch

from colorama import Fore, Style

from opensearch_py_ml.ml_commons.cli.connector_base import ConnectorBase
from opensearch_py_ml.ml_commons.cli.ml_setup import Setup


# TODO: add more tests for new functions
class TestSetup(unittest.TestCase):
    def setUp(self):
        if os.path.exists(ConnectorBase.CONFIG_FILE):
            os.remove(ConnectorBase.CONFIG_FILE)
        self.setup_instance = Setup()

    def tearDown(self):
        if os.path.exists(ConnectorBase.CONFIG_FILE):
            os.remove(ConnectorBase.CONFIG_FILE)

    @patch("builtins.input", return_value="fake_input")
    @patch("builtins.print")
    def test_configure_aws(self, mock_print, mock_input):
        mock_get_password = MagicMock(
            side_effect=["fake_access_key", "fake_secret_key", "fake_session_token"]
        )

        with patch.multiple(
            self.setup_instance,
            get_password_with_asterisks=mock_get_password,
            update_aws_credentials=MagicMock(),
            check_credentials_validity=MagicMock(return_value=True),
        ):
            self.setup_instance.configure_aws()
            self.assertEqual(mock_get_password.call_count, 3)
            mock_get_password.assert_has_calls(
                [
                    call("Enter your AWS Access Key ID: "),
                    call("Enter your AWS Secret Access Key: "),
                    call("Enter your AWS Session Token: "),
                ]
            )
            self.setup_instance.update_aws_credentials.assert_called_once_with(
                "fake_access_key", "fake_secret_key", "fake_session_token"
            )
            self.setup_instance.check_credentials_validity.assert_called_once()
            mock_print.assert_any_call(
                f"{Fore.GREEN}New AWS credentials have been successfully configured and verified.{Style.RESET_ALL}"
            )

    @patch("boto3.Session")
    @patch("builtins.input", return_value="no")
    def test_aws_credentials_exist_no_reconfigure(self, mock_input, mock_session):
        mock_session_instance = mock_session.return_value
        mock_session_instance.get_credentials.return_value = MagicMock()
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
        self.assertEqual(
            config["opensearch_config"]["opensearch_domain_username"], "admin"
        )
        self.assertEqual(
            config["opensearch_config"]["opensearch_domain_password"], "pass"
        )

    @patch("builtins.input", side_effect=["2", "", "no", "2", ""])
    def test_setup_configuration_open_source_no_auth(self, mock_input):
        self.setup_instance.setup_configuration()
        config = self.setup_instance.config
        self.assertEqual(config["service_type"], "open-source")
        self.assertEqual(
            config["opensearch_config"]["opensearch_domain_username"], None
        )
        self.assertEqual(
            config["opensearch_config"]["opensearch_domain_password"], None
        )

    @patch("opensearch_py_ml.ml_commons.cli.ml_setup.OpenSearch")
    def test_initialize_opensearch_client_managed(self, mock_opensearch):
        self.setup_instance.service_type = "amazon-opensearch-service"
        self.setup_instance.opensearch_domain_endpoint = "https://test-domain:443"
        self.setup_instance.opensearch_domain_username = "admin"
        self.setup_instance.opensearch_domain_password = "pass"
        result = self.setup_instance.initialize_opensearch_client()
        self.assertTrue(result)
        mock_opensearch.assert_called_once()

    @patch("opensearch_py_ml.ml_commons.cli.ml_setup.OpenSearch")
    def test_initialize_opensearch_client_open_source_no_auth(self, mock_opensearch):
        self.setup_instance.service_type = "open-source"
        self.setup_instance.opensearch_domain_endpoint = "http://localhost:9200"
        self.setup_instance.opensearch_domain_username = ""
        self.setup_instance.opensearch_domain_password = ""
        result = self.setup_instance.initialize_opensearch_client()
        self.assertTrue(result)
        mock_opensearch.assert_called_once()

    @patch("opensearch_py_ml.ml_commons.cli.ml_setup.OpenSearch")
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
