# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.


import os
import unittest
from unittest.mock import MagicMock, call, patch

from botocore.exceptions import ClientError
from colorama import Fore, Style

from opensearch_py_ml.ml_commons.cli.cli_base import CLIBase
from opensearch_py_ml.ml_commons.cli.ml_setup import Setup


class TestSetup(unittest.TestCase):
    def setUp(self):
        if os.path.exists(CLIBase.CONFIG_FILE):
            os.remove(CLIBase.CONFIG_FILE)
        self.setup_instance = Setup()

    def tearDown(self):
        if os.path.exists(CLIBase.CONFIG_FILE):
            os.remove(CLIBase.CONFIG_FILE)

    @patch("boto3.Session")
    def test_check_credentials_validity(self, mock_session):
        """Test check_credentials_validity with valid credentials"""
        # Setup mock
        mock_sts_client = MagicMock()
        mock_session.return_value.client.return_value = mock_sts_client
        mock_sts_client.get_caller_identity.return_value = {"UserId": "test-user"}

        # Test with valid credentials
        result = self.setup_instance.check_credentials_validity(
            access_key="valid-access-key",
            secret_key="valid-secret-key",
            session_token="valid-session-token",
        )

        # Verify the result
        self.assertTrue(result)

        # Verify boto3.Session was called with correct credentials
        mock_session.assert_called_once_with(
            aws_access_key_id="valid-access-key",
            aws_secret_access_key="valid-secret-key",
            aws_session_token="valid-session-token",
        )

        # Verify STS client was created and called
        mock_session.return_value.client.assert_called_once_with("sts")
        mock_sts_client.get_caller_identity.assert_called_once()

    @patch("boto3.Session")
    def test_check_credentials_validity_client_error(self, mock_session):
        """Test check_credentials_validity with ClientError"""
        # Setup mock to raise ClientError
        mock_sts_client = MagicMock()
        mock_session.return_value.client.return_value = mock_sts_client
        mock_sts_client.get_caller_identity.side_effect = ClientError(
            error_response={"Error": {"Code": "ExpiredToken"}},
            operation_name="GetCallerIdentity",
        )

        # Test with credentials
        result = self.setup_instance.check_credentials_validity(
            access_key="test-access-key",
            secret_key="test-secret-key",
            session_token="test-session-token",
        )

        # Verify the result
        self.assertFalse(result)

    @patch("boto3.Session")
    def test_check_credentials_validity_from_config_file(self, mock_session):
        """Test check_credentials_validity with valid credentials in config"""
        # Setup mock
        mock_sts_client = MagicMock()
        mock_session.return_value.client.return_value = mock_sts_client
        mock_sts_client.get_caller_identity.return_value = {"UserId": "test-user"}

        # Setup config with valid credentials
        self.setup_instance.config = {
            "aws_credentials": {
                "aws_access_key": "valid-access-key",
                "aws_secret_access_key": "valid-secret-key",
                "aws_session_token": "valid-session-token",
            }
        }

        # Test credentials validation
        result = self.setup_instance.check_credentials_validity(use_config=True)

        # Verify the result
        self.assertTrue(result)

        # Verify boto3.Session was called with correct credentials
        mock_session.assert_called_once_with(
            aws_access_key_id="valid-access-key",
            aws_secret_access_key="valid-secret-key",
            aws_session_token="valid-session-token",
        )

    def test_update_aws_credentials_empty_config(self):
        """Test update_aws_credentials when config is empty"""
        self.setup_instance.update_aws_credentials(
            access_key="test-access-key",
            secret_key="test-secret-key",
            session_token="test-session-token",
        )

        # Verify credentials were added correctly
        self.assertIn("aws_credentials", self.setup_instance.config)
        self.assertEqual(
            self.setup_instance.config["aws_credentials"]["aws_access_key"],
            "test-access-key",
        )
        self.assertEqual(
            self.setup_instance.config["aws_credentials"]["aws_secret_access_key"],
            "test-secret-key",
        )
        self.assertEqual(
            self.setup_instance.config["aws_credentials"]["aws_session_token"],
            "test-session-token",
        )

    def test_update_aws_credentials_existing_config(self):
        """Test update_aws_credentials when aws_credentials already exists"""
        # Setup existing credentials
        self.setup_instance.config = {
            "aws_credentials": {
                "aws_access_key": "old-access-key",
                "aws_secret_access_key": "old-secret-key",
                "aws_session_token": "old-session-token",
            }
        }

        # Update credentials
        self.setup_instance.update_aws_credentials(
            access_key="new-access-key",
            secret_key="new-secret-key",
            session_token="new-session-token",
        )

        # Verify credentials were updated
        self.assertEqual(
            self.setup_instance.config["aws_credentials"]["aws_access_key"],
            "new-access-key",
        )
        self.assertEqual(
            self.setup_instance.config["aws_credentials"]["aws_secret_access_key"],
            "new-secret-key",
        )
        self.assertEqual(
            self.setup_instance.config["aws_credentials"]["aws_session_token"],
            "new-session-token",
        )

    @patch("opensearch_py_ml.ml_commons.cli.ml_setup.logger")
    def test_update_aws_credentials_exception(self, mock_logger):
        """Test update_aws_credentials exception handling"""
        # Make config an integer to force TypeError exception
        self.setup_instance.config = 123

        with self.assertRaises(TypeError):
            self.setup_instance.update_aws_credentials(
                access_key="test-access-key",
                secret_key="test-secret-key",
                session_token="test-session-token",
            )

        # Verify error message was printed with correct formatting
        mock_logger.error.assert_called_once_with(
            f"{Fore.RED}Failed to update AWS credentials: argument of type 'int' is not iterable{Style.RESET_ALL}"
        )

    @patch("builtins.input", return_value="fake_input")
    @patch("builtins.print")
    def test_configure_aws(self, mock_print, mock_input):
        """Test configure_aws successful"""
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

    @patch("opensearch_py_ml.ml_commons.cli.ml_setup.logger")
    def test_configure_aws_invalid_credentials(self, mock_logger):
        """Test configure_aws with invalid credentials"""
        mock_get_password = MagicMock(
            side_effect=["fake_access_key", "fake_secret_key", "fake_session_token"]
        )

        with patch.multiple(
            self.setup_instance,
            get_password_with_asterisks=mock_get_password,
            update_aws_credentials=MagicMock(),
            check_credentials_validity=MagicMock(return_value=False),
        ):
            self.setup_instance.configure_aws()
            mock_logger.warning.assert_called_once_with(
                f"{Fore.RED}The provided credentials are invalid or expired.{Style.RESET_ALL}"
            )

    @patch("opensearch_py_ml.ml_commons.cli.ml_setup.logger")
    def test_check_and_configure_aws_invalid_credentials(self, mock_logger):
        """Test check_and_configure_aws with invalid credentials"""
        config_path = "test_config.yml"

        # Setup the instance config
        self.setup_instance.config = {
            "aws_credentials": {
                "aws_access_key": "test-key",
                "aws_secret_access_key": "test-secret",
                "aws_session_token": "test-token",
            }
        }

        # Mock check_credentials_validity to return False
        with patch.object(
            self.setup_instance,
            "check_credentials_validity",
            return_value=False,
        ), patch.object(self.setup_instance, "configure_aws"):

            # Execute
            self.setup_instance.check_and_configure_aws(config_path)
            mock_logger.warning.assert_called_once_with(
                f"{Fore.YELLOW}Your AWS credentials are invalid or have expired.{Style.RESET_ALL}"
            )

    @patch("boto3.Session")
    @patch("builtins.input", return_value="yes")
    def test_check_and_configure_aws_yes_reconfigure(self, mock_input, mock_session):
        """Test check_and_configure_aws reconfigure"""
        config_path = "test_config.yml"

        # Setup the instance config
        self.setup_instance.config = {
            "aws_credentials": {
                "aws_access_key": "test-key",
                "aws_secret_access_key": "test-secret",
                "aws_session_token": "test-token",
            }
        }

        # Mock check_credentials_validity to return True
        with patch.object(
            self.setup_instance,
            "check_credentials_validity",
            return_value=True,
        ), patch.object(self.setup_instance, "configure_aws") as mock_configure:

            # Execute
            self.setup_instance.check_and_configure_aws(config_path)

            # Assert
            mock_configure.assert_called_once()  # configure_aws should be called
            mock_input.assert_called_once_with("Do you want to reconfigure? (yes/no): ")

    @patch("boto3.Session")
    @patch("builtins.input", return_value="no")
    def test_check_and_configure_aws_no_reconfigure(self, mock_input, mock_session):
        """Test check_and_configure_aws no reconfigure"""
        config_path = "test_config.yml"

        # Setup the instance config
        self.setup_instance.config = {
            "aws_credentials": {
                "aws_access_key": "test-key",
                "aws_secret_access_key": "test-secret",
                "aws_session_token": "test-token",
            }
        }

        # Mock check_credentials_validity to return True
        with patch.object(
            self.setup_instance,
            "check_credentials_validity",
            return_value=True,
        ), patch.object(
            self.setup_instance, "configure_aws"
        ) as mock_configure, patch.object(
            self.setup_instance, "update_config"
        ) as mock_update_config:

            # Execute
            self.setup_instance.check_and_configure_aws(config_path)

            # Assert
            mock_configure.assert_not_called()  # configure_aws should not be called
            mock_update_config.assert_not_called()  # update_config should not be called
            mock_input.assert_called_once_with("Do you want to reconfigure? (yes/no): ")

    @patch(
        "builtins.input",
        side_effect=[
            "1",
            "1",
            "test-iam-role-arn",
            "us-west-2",
            "test-domain-endpoint",
            "admin",
            "pass",
        ],
    )
    @patch.object(Setup, "get_password_with_asterisks", return_value="pass")
    @patch("boto3.Session")
    def test_setup_configuration_iam_role_managed_service(
        self, mock_session, mock_get_password, mock_input
    ):
        """Test setup_configuration with IAM role ARN in managed service"""
        # Mock the STS client
        mock_sts_client = MagicMock()
        mock_session.return_value.client.return_value = mock_sts_client
        mock_sts_client.get_caller_identity.return_value = {"UserId": "test-user"}

        self.setup_instance.setup_configuration()
        config = self.setup_instance.config
        self.assertEqual(config["service_type"], CLIBase.AMAZON_OPENSEARCH_SERVICE)
        self.assertEqual(
            config["opensearch_config"]["opensearch_domain_region"], "us-west-2"
        )
        self.assertEqual(
            config["opensearch_config"]["opensearch_domain_endpoint"],
            "test-domain-endpoint",
        )
        self.assertEqual(
            config["opensearch_config"]["opensearch_domain_username"], "admin"
        )
        self.assertEqual(
            config["opensearch_config"]["opensearch_domain_password"], "pass"
        )
        self.assertEqual(
            config["aws_credentials"]["aws_role_name"], "test-iam-role-arn"
        )
        self.assertEqual(config["aws_credentials"]["aws_user_name"], "")

    @patch(
        "builtins.input",
        side_effect=[
            "1",
            "2",
            "test-iam-user-arn",
            "us-west-2",
            "test-domain-endpoint",
            "admin",
            "pass",
        ],
    )
    @patch.object(Setup, "get_password_with_asterisks", return_value="pass")
    @patch("boto3.Session")
    def test_setup_configuration_iam_user_managed_service(
        self, mock_session, mock_get_password, mock_input
    ):
        """Test setup_configuration with IAM user ARN in managed service"""
        # Mock the STS client
        mock_sts_client = MagicMock()
        mock_session.return_value.client.return_value = mock_sts_client
        mock_sts_client.get_caller_identity.return_value = {"UserId": "test-user"}

        self.setup_instance.setup_configuration()
        config = self.setup_instance.config
        self.assertEqual(config["service_type"], CLIBase.AMAZON_OPENSEARCH_SERVICE)
        self.assertEqual(
            config["opensearch_config"]["opensearch_domain_region"], "us-west-2"
        )
        self.assertEqual(
            config["opensearch_config"]["opensearch_domain_endpoint"],
            "test-domain-endpoint",
        )
        self.assertEqual(
            config["opensearch_config"]["opensearch_domain_username"], "admin"
        )
        self.assertEqual(
            config["opensearch_config"]["opensearch_domain_password"], "pass"
        )
        self.assertEqual(config["aws_credentials"]["aws_role_name"], "")
        self.assertEqual(
            config["aws_credentials"]["aws_user_name"], "test-iam-user-arn"
        )

    @patch(
        "builtins.input",
        side_effect=[
            "1",
            "3",
            "test-iam-role-arn",
            "us-west-2",
            "test-domain-endpoint",
            "admin",
            "pass",
        ],
    )
    @patch.object(Setup, "get_password_with_asterisks", return_value="pass")
    @patch("boto3.Session")
    @patch("opensearch_py_ml.ml_commons.cli.ml_setup.logger")
    def test_setup_configuration_invalid_arn_managed_service(
        self, mock_logger, mock_session, mock_get_password, mock_input
    ):
        """Test setup_configuration with invalid ARN type in managed service"""
        # Mock the STS client
        mock_sts_client = MagicMock()
        mock_session.return_value.client.return_value = mock_sts_client
        mock_sts_client.get_caller_identity.return_value = {"UserId": "test-user"}

        self.setup_instance.setup_configuration()
        config = self.setup_instance.config
        self.assertEqual(config["service_type"], CLIBase.AMAZON_OPENSEARCH_SERVICE)
        self.assertEqual(
            config["opensearch_config"]["opensearch_domain_region"], "us-west-2"
        )
        self.assertEqual(
            config["opensearch_config"]["opensearch_domain_endpoint"],
            "test-domain-endpoint",
        )
        self.assertEqual(
            config["opensearch_config"]["opensearch_domain_username"], "admin"
        )
        self.assertEqual(
            config["opensearch_config"]["opensearch_domain_password"], "pass"
        )
        self.assertEqual(
            config["aws_credentials"]["aws_role_name"], "test-iam-role-arn"
        )
        self.assertEqual(config["aws_credentials"]["aws_user_name"], "")
        mock_logger.warning.assert_any_call(
            f"\n{Fore.YELLOW}Invalid choice. Defaulting to 'IAM Role ARN'.{Style.RESET_ALL}"
        )

    @patch("builtins.input", side_effect=["2", "", "", "yes", "admin"])
    @patch.object(Setup, "get_password_with_asterisks", return_value="pass")
    def test_setup_configuration_open_source_with_auth(
        self, mock_get_password, mock_input
    ):
        """Test setup_configuration in open-source service with authorization"""
        self.setup_instance.setup_configuration()
        config = self.setup_instance.config
        self.assertEqual(config["service_type"], CLIBase.OPEN_SOURCE)
        self.assertEqual(
            config["opensearch_config"]["opensearch_domain_username"], "admin"
        )
        self.assertEqual(
            config["opensearch_config"]["opensearch_domain_password"], "pass"
        )

    @patch("builtins.input", side_effect=["2", "", "no", "2", ""])
    def test_setup_configuration_open_source_no_auth(self, mock_input):
        """Test setup_configuration in open-source service without authorization"""
        self.setup_instance.setup_configuration()
        config = self.setup_instance.config
        self.assertEqual(config["service_type"], CLIBase.OPEN_SOURCE)
        self.assertEqual(
            config["opensearch_config"]["opensearch_domain_username"], None
        )
        self.assertEqual(
            config["opensearch_config"]["opensearch_domain_password"], None
        )

    @patch(
        "builtins.input",
        side_effect=[
            "invalid_choice",
            "1",
            "test-iam-role-arn",
            "us-west-2",
            "test-domain-endpoint",
            "admin",
            "pass",
        ],
    )
    @patch.object(Setup, "get_password_with_asterisks", return_value="pass")
    @patch("opensearch_py_ml.ml_commons.cli.ml_setup.logger")
    @patch.object(Setup, "configure_aws")
    def test_setup_configuration_invalid_service_type(
        self, mock_configure_aws, mock_logger, mock_get_password, mock_input
    ):
        """Test setup_configuration with invalid service type"""
        self.setup_instance.setup_configuration()
        config = self.setup_instance.config
        self.assertEqual(config["service_type"], CLIBase.AMAZON_OPENSEARCH_SERVICE)
        mock_logger.warning.assert_called_once_with(
            f"\n{Fore.YELLOW}Invalid choice. Defaulting to 'amazon-opensearch-service'.{Style.RESET_ALL}"
        )
        mock_configure_aws.assert_called_once()

    @patch("opensearch_py_ml.ml_commons.cli.ml_setup.OpenSearch")
    def test_initialize_opensearch_client_managed(self, mock_opensearch):
        """Test initialize_opensearch_client in managed service"""
        self.setup_instance.service_type = CLIBase.AMAZON_OPENSEARCH_SERVICE
        self.setup_instance.opensearch_config.opensearch_domain_endpoint = (
            "https://test-domain:443"
        )
        self.setup_instance.opensearch_config.opensearch_domain_username = "admin"
        self.setup_instance.opensearch_config.opensearch_domain_password = "pass"
        result = self.setup_instance.initialize_opensearch_client()
        self.assertTrue(result)
        mock_opensearch.assert_called_once()

    @patch("opensearch_py_ml.ml_commons.cli.ml_setup.OpenSearch")
    def test_initialize_opensearch_client_open_source_no_auth(self, mock_opensearch):
        """Test initialize_opensearch_client in open-source service without authorization"""
        self.setup_instance.service_type = CLIBase.OPEN_SOURCE
        self.setup_instance.opensearch_config.opensearch_domain_endpoint = (
            "http://localhost:9200"
        )
        self.setup_instance.opensearch_config.opensearch_domain_username = ""
        self.setup_instance.opensearch_config.opensearch_domain_password = ""
        result = self.setup_instance.initialize_opensearch_client()
        self.assertTrue(result)
        mock_opensearch.assert_called_once()

    @patch("opensearch_py_ml.ml_commons.cli.ml_setup.OpenSearch")
    def test_initialize_opensearch_client_open_source_with_auth(self, mock_opensearch):
        """Test initialize_opensearch_client in open-source service with authorization"""
        self.setup_instance.service_type = CLIBase.OPEN_SOURCE
        self.setup_instance.opensearch_config.opensearch_domain_endpoint = (
            "http://localhost:9200"
        )
        self.setup_instance.opensearch_config.opensearch_domain_username = "admin"
        self.setup_instance.opensearch_config.opensearch_domain_password = "pass"
        result = self.setup_instance.initialize_opensearch_client()
        self.assertTrue(result)
        mock_opensearch.assert_called_once()

    @patch("opensearch_py_ml.ml_commons.cli.ml_setup.logger")
    def test_initialize_opensearch_client_no_endpoint(self, mock_logger):
        """Test initialize_opensearch_client without domain endpoint"""
        self.setup_instance.service_type = CLIBase.AMAZON_OPENSEARCH_SERVICE
        self.setup_instance.opensearch_domain_username = "admin"
        self.setup_instance.opensearch_domain_password = "pass"
        result = self.setup_instance.initialize_opensearch_client()
        self.assertFalse(result)
        mock_logger.warning.assert_called_once_with(
            f"{Fore.RED}OpenSearch endpoint not set. Please run setup first.{Style.RESET_ALL}\n"
        )

    @patch("opensearch_py_ml.ml_commons.cli.ml_setup.logger")
    def test_initialize_opensearch_client_no_username_password(self, mock_logger):
        """Test initialize_opensearch_client without domain username and password"""
        self.setup_instance.service_type = CLIBase.AMAZON_OPENSEARCH_SERVICE
        self.setup_instance.opensearch_config.opensearch_domain_endpoint = (
            "https://test-domain:443"
        )
        result = self.setup_instance.initialize_opensearch_client()
        self.assertFalse(result)
        mock_logger.warning.assert_called_once_with(
            f"{Fore.RED}OpenSearch username or password not set. Please run setup first.{Style.RESET_ALL}\n"
        )

    @patch("opensearch_py_ml.ml_commons.cli.ml_setup.logger")
    def test_initialize_opensearch_client_invalid_service_type(self, mock_logger):
        """Test initialize_opensearch_client with invalid service type"""
        self.setup_instance.service_type = "invalid-service"
        self.setup_instance.opensearch_config.opensearch_domain_endpoint = (
            "http://localhost:9200"
        )
        result = self.setup_instance.initialize_opensearch_client()
        self.assertFalse(result)
        mock_logger.warning.assert_called_once_with(
            "Invalid service type. Please check your configuration."
        )

    @patch("opensearch_py_ml.ml_commons.cli.ml_setup.logger")
    @patch("opensearch_py_ml.ml_commons.cli.ml_setup.OpenSearch")
    def test_initialize_opensearch_client_exception(self, mock_opensearch, mock_logger):
        """Test initialize_opensearch_client exception handling"""
        self.setup_instance.service_type = CLIBase.OPEN_SOURCE
        self.setup_instance.opensearch_config.opensearch_domain_endpoint = (
            "http://localhost:9200"
        )
        mock_opensearch.side_effect = Exception("Connection failed")
        result = self.setup_instance.initialize_opensearch_client()
        self.assertFalse(result)
        mock_logger.error.assert_called_with(
            f"{Fore.RED}Error initializing OpenSearch client: Connection failed{Style.RESET_ALL}\n"
        )

    def test_update_from_config(self):
        """Test _update_from_config successful"""
        self.setup_instance.config = {
            "service_type": CLIBase.AMAZON_OPENSEARCH_SERVICE,
            "opensearch_config": {
                "opensearch_domain_region": "us-west-2",
                "opensearch_domain_endpoint": "https://test-endpoint",
                "opensearch_domain_username": "admin",
                "opensearch_domain_password": "password",
            },
            "aws_credentials": {
                "aws_role_name": "test-role",
                "aws_user_name": "test-user",
                "aws_access_key": "test-access-key",
                "aws_secret_access_key": "test-secret-key",
                "aws_session_token": "test-session-token",
            },
        }

        # Execute
        result = self.setup_instance._update_from_config()

        # Verify
        self.assertTrue(result)
        self.assertEqual(
            self.setup_instance.service_type, CLIBase.AMAZON_OPENSEARCH_SERVICE
        )

        # Verify OpenSearch config
        self.assertEqual(
            self.setup_instance.opensearch_config.opensearch_domain_region, "us-west-2"
        )
        self.assertEqual(
            self.setup_instance.opensearch_config.opensearch_domain_endpoint,
            "https://test-endpoint",
        )
        self.assertEqual(
            self.setup_instance.opensearch_config.opensearch_domain_username, "admin"
        )
        self.assertEqual(
            self.setup_instance.opensearch_config.opensearch_domain_password, "password"
        )

        # Verify AWS credentials
        self.assertEqual(self.setup_instance.aws_config.aws_role_name, "test-role")
        self.assertEqual(self.setup_instance.aws_config.aws_user_name, "test-user")
        self.assertEqual(
            self.setup_instance.aws_config.aws_access_key, "test-access-key"
        )
        self.assertEqual(
            self.setup_instance.aws_config.aws_secret_access_key, "test-secret-key"
        )
        self.assertEqual(
            self.setup_instance.aws_config.aws_session_token, "test-session-token"
        )

    def test_update_from_config_empty_config(self):
        """Test _update_from_config with empty config"""
        self.setup_instance.config = None
        result = self.setup_instance._update_from_config()
        self.assertFalse(result)

    @patch("builtins.print")
    @patch.object(Setup, "load_config")
    @patch.object(Setup, "_update_from_config")
    @patch.object(Setup, "check_and_configure_aws")
    def test_setup_command_valid_config_path(
        self, mock_configure_aws, mock_update_config, mock_load_config, mock_print
    ):
        """Test setup_command with valid config path"""
        # Setup
        config_path = "test_config.yml"
        mock_load_config.return_value = True
        mock_update_config.return_value = True
        self.setup_instance.service_type = CLIBase.AMAZON_OPENSEARCH_SERVICE

        # Execute
        result = self.setup_instance.setup_command(config_path)

        # Verify
        self.assertEqual(result, config_path)
        mock_load_config.assert_called_once_with(config_path)
        mock_update_config.assert_called_once()
        mock_configure_aws.assert_called_once_with(config_path)
        mock_print.assert_called_with(
            f"{Fore.GREEN}\nSetup complete. You are now ready to use the ML features.{Style.RESET_ALL}"
        )

    @patch("builtins.input", side_effect=["no"])
    @patch("opensearch_py_ml.ml_commons.cli.ml_setup.logger")
    @patch.object(Setup, "load_config")
    @patch.object(Setup, "setup_configuration")
    def test_setup_command_load_config_failure_with_config_path(
        self, mock_setup_configuration, mock_load_config, mock_logger, mock_input
    ):
        """Test setup_command with config path when load_config fails"""
        # Setup
        config_path = "invalid_config.yml"
        mock_load_config.return_value = False
        mock_setup_configuration.return_value = "new_config.yml"

        # Execute
        self.setup_instance.setup_command(config_path)

        # Verify
        mock_load_config.assert_called_once_with(config_path)
        mock_logger.warning.assert_any_call(
            f"{Fore.YELLOW}Could not load existing configuration. Creating new configuration...{Style.RESET_ALL}"
        )

    @patch("builtins.input", side_effect=["yes", ""])
    @patch("opensearch_py_ml.ml_commons.cli.ml_setup.logger")
    @patch.object(Setup, "load_config")
    @patch.object(Setup, "setup_configuration")
    def test_setup_command_load_config_failure(
        self, mock_setup_configuration, mock_load_config, mock_logger, mock_input
    ):
        """Test setup_command when load_config fails"""
        # Setup
        mock_load_config.return_value = False
        mock_setup_configuration.return_value = "new_config.yml"

        # Execute
        self.setup_instance.setup_command()

        # Verify
        mock_load_config.assert_called_once()
        mock_logger.warning.assert_any_call(
            f"{Fore.YELLOW}Could not load existing configuration. Creating new configuration...{Style.RESET_ALL}"
        )

    @patch("builtins.input", side_effect=["yes", "test_config.yml"])
    @patch("builtins.print")
    @patch.object(Setup, "load_config")
    @patch.object(Setup, "_update_from_config")
    @patch.object(Setup, "check_and_configure_aws")
    def test_setup_command_with_existing_config(
        self,
        mock_configure_aws,
        mock_update_config,
        mock_load_config,
        mock_print,
        mock_input,
    ):
        """Test setup_command with user having existing config"""
        # Setup
        mock_load_config.return_value = True
        mock_update_config.return_value = True
        self.setup_instance.service_type = CLIBase.AMAZON_OPENSEARCH_SERVICE

        # Execute
        result = self.setup_instance.setup_command()

        # Verify
        self.assertEqual(result, "test_config.yml")
        mock_load_config.assert_called_once_with("test_config.yml")
        mock_update_config.assert_called_once()
        mock_configure_aws.assert_called_once_with("test_config.yml")

    @patch("builtins.input", side_effect=["no"])
    @patch("builtins.print")
    @patch.object(Setup, "setup_configuration")
    @patch.object(Setup, "initialize_opensearch_client")
    def test_setup_command_new_configuration(
        self, mock_initialize_client, mock_setup_configuration, mock_print, mock_input
    ):
        """Test setup_command for new configuration"""
        # Setup
        mock_setup_configuration.return_value = "new_config.yml"
        mock_initialize_client.return_value = True

        # Execute
        result = self.setup_instance.setup_command()

        # Verify
        self.assertEqual(result, "new_config.yml")
        mock_setup_configuration.assert_called_once()
        mock_initialize_client.assert_called_once()
        mock_print.assert_any_call("Let's create a new configuration file.")

    @patch("builtins.input", side_effect=["yes", ""])
    @patch("opensearch_py_ml.ml_commons.cli.ml_setup.logger")
    @patch.object(Setup, "load_config")
    @patch.object(Setup, "_update_from_config")
    def test_setup_command_update_config_failure_with_config_path(
        self, mock_update_config, mock_load_config, mock_logger, mock_input
    ):
        """Test setup_command with config path when update_from_config fails"""
        # Setup
        config_path = "test_config.yml"
        mock_load_config.return_value = True
        mock_update_config.return_value = False
        self.setup_instance.service_type = CLIBase.AMAZON_OPENSEARCH_SERVICE

        # Execute
        self.setup_instance.setup_command(config_path)

        # Verify
        mock_update_config.assert_any_call()
        mock_logger.warning.assert_called_once_with(
            f"{Fore.RED}Failed to update configuration.{Style.RESET_ALL}"
        )

    @patch("builtins.input", side_effect=["yes", "test_config.yml"])
    @patch("opensearch_py_ml.ml_commons.cli.ml_setup.logger")
    @patch.object(Setup, "load_config")
    @patch.object(Setup, "_update_from_config")
    def test_setup_command_update_config_failure(
        self, mock_update_config, mock_load_config, mock_logger, mock_input
    ):
        """Test setup_command when update_from_config fails"""
        # Setup
        mock_load_config.return_value = True
        mock_update_config.return_value = False

        # Execute
        self.setup_instance.setup_command()

        # Verify
        mock_logger.warning.assert_any_call(
            f"{Fore.RED}Failed to update configuration.{Style.RESET_ALL}"
        )

    @patch("builtins.input", side_effect=["no"])
    @patch("builtins.print")
    @patch.object(Setup, "initialize_opensearch_client")
    @patch.object(Setup, "setup_configuration")
    def test_setup_command_initialize_opensearch_client(
        self, mock_setup_configuration, mock_initialize_client, mock_print, mock_input
    ):
        """Test setup_command when OpenSearch client initialization success"""
        # Mock the configuration loading process
        mock_initialize_client.return_value = True
        mock_setup_configuration.return_value = "new_config.yml"

        self.setup_instance.service_type = CLIBase.OPEN_SOURCE
        self.setup_instance.setup_command()

        # Verify
        mock_print.assert_called_with(
            f"{Fore.GREEN}Setup complete. You are now ready to use the ML features.{Style.RESET_ALL}"
        )

    @patch("builtins.input", side_effect=["no"])
    @patch("opensearch_py_ml.ml_commons.cli.ml_setup.logger")
    @patch.object(Setup, "initialize_opensearch_client")
    @patch.object(Setup, "setup_configuration")
    def test_setup_command_initialize_opensearch_client_failure(
        self, mock_setup_configuration, mock_initialize_client, mock_logger, mock_input
    ):
        """Test setup_command when OpenSearch client initialization fails"""
        # Mock the configuration loading process
        mock_initialize_client.return_value = False
        mock_setup_configuration.return_value = "new_config.yml"

        self.setup_instance.service_type = CLIBase.OPEN_SOURCE
        self.setup_instance.setup_command()

        # Verify
        mock_logger.warning.assert_called_with(
            f"\n{Fore.RED}Failed to initialize OpenSearch client. Setup incomplete.{Style.RESET_ALL}\n"
        )


if __name__ == "__main__":
    unittest.main()
