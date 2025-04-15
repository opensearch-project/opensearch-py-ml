# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

import configparser
import logging
import sys
import termios
import tty
from urllib.parse import urlparse

import boto3
from botocore.exceptions import ClientError
from colorama import Fore, Style, init
from opensearchpy import OpenSearch, RequestsHttpConnection

from opensearch_py_ml.ml_commons.cli.aws_config import AWSConfig
from opensearch_py_ml.ml_commons.cli.cli_base import CLIBase
from opensearch_py_ml.ml_commons.cli.opensearch_domain_config import (
    OpenSearchDomainConfig,
)

# Initialize colorama for colored terminal output
init(autoreset=True)

# Configure the logger for this module
logger = logging.getLogger(__name__)


class Setup(CLIBase):
    """
    Handles the setup and configuration of the OpenSearch environment.
    Manages AWS credentials (if amazon-opensearch-service) and OpenSearch client initialization.
    """

    def __init__(self):
        """
        Initialize the Setup class with default or existing configurations.
        """
        super().__init__()
        self.config = configparser.ConfigParser()
        self.service_type = ""
        self.ssl_check_enabled = True
        self.opensearch_client = None
        self.session = None
        self.opensearch_config = OpenSearchDomainConfig(
            opensearch_domain_region="",
            opensearch_domain_name="",
            opensearch_domain_username="",
            opensearch_domain_password="",
            opensearch_domain_endpoint="",
        )
        self.aws_config = AWSConfig(
            aws_user_name="",
            aws_role_name="",
            aws_access_key="",
            aws_secret_access_key="",
            aws_session_token="",
        )

    def check_credentials_validity(
        self, access_key=None, secret_key=None, session_token=None, use_config=False
    ):
        """
        Check if the provided AWS credentials are valid.
        """
        try:
            if use_config:
                # Check credentials from a config file
                aws_credentials = self.config.get("aws_credentials", {})
                credentials = {
                    "aws_access_key_id": aws_credentials.get("aws_access_key", ""),
                    "aws_secret_access_key": aws_credentials.get(
                        "aws_secret_access_key", ""
                    ),
                    "aws_session_token": aws_credentials.get("aws_session_token", ""),
                }
            else:
                credentials = {
                    "aws_access_key_id": access_key,
                    "aws_secret_access_key": secret_key,
                    "aws_session_token": session_token,
                }
            self.session = boto3.Session(**credentials)
            sts_client = self.session.client("sts")
            sts_client.get_caller_identity()
            return True
        except ClientError:
            return False

    def update_aws_credentials(self, access_key, secret_key, session_token):
        """
        Update AWS credentials in the config file.
        """
        try:
            if "aws_credentials" not in self.config:
                self.config["aws_credentials"] = {}

            self.config["aws_credentials"]["aws_access_key"] = access_key
            self.config["aws_credentials"]["aws_secret_access_key"] = secret_key
            self.config["aws_credentials"]["aws_session_token"] = session_token
        except Exception as e:
            logger.error(
                f"{Fore.RED}Failed to update AWS credentials: {e}{Style.RESET_ALL}"
            )
            raise

    def check_and_configure_aws(self, config_path):
        """
        Check if AWS credentials are configured and offer to reconfigure if needed.
        """
        if not self.check_credentials_validity(use_config=True):
            logger.warning(
                f"{Fore.YELLOW}Your AWS credentials are invalid or have expired.{Style.RESET_ALL}"
            )
            self.configure_aws()
            self.config["aws_credentials"][
                "aws_access_key"
            ] = self.aws_config.aws_access_key
            self.config["aws_credentials"][
                "aws_secret_access_key"
            ] = self.aws_config.aws_secret_access_key
            self.config["aws_credentials"][
                "aws_session_token"
            ] = self.aws_config.aws_session_token
            self.update_config(self.config, config_path)
        else:
            print(
                f"{Fore.GREEN}AWS credentials are already configured.{Style.RESET_ALL}"
            )
            reconfigure = input("Do you want to reconfigure? (yes/no): ").lower()
            if reconfigure == "yes":
                self.configure_aws()

    def configure_aws(self):
        """
        Configure AWS credentials by prompting the user for input.
        """
        print("Let's configure your AWS credentials.")
        self.aws_config.aws_access_key = self.get_password_with_asterisks(
            "Enter your AWS Access Key ID: "
        )
        self.aws_config.aws_secret_access_key = self.get_password_with_asterisks(
            "Enter your AWS Secret Access Key: "
        )
        self.aws_config.aws_session_token = self.get_password_with_asterisks(
            "Enter your AWS Session Token: "
        )

        self.update_aws_credentials(
            self.aws_config.aws_access_key,
            self.aws_config.aws_secret_access_key,
            self.aws_config.aws_session_token,
        )
        if self.check_credentials_validity(
            self.aws_config.aws_access_key,
            self.aws_config.aws_secret_access_key,
            self.aws_config.aws_session_token,
        ):
            print(
                f"{Fore.GREEN}New AWS credentials have been successfully configured and verified.{Style.RESET_ALL}"
            )
        else:
            logger.warning(
                f"{Fore.RED}The provided credentials are invalid or expired.{Style.RESET_ALL}"
            )

    def get_password_with_asterisks(self, prompt="Enter password: ") -> str:
        """
        Get password input from user, masking it with asterisks.
        """
        if sys.platform == "win32":
            import msvcrt

            print(prompt, end="", flush=True)
            password = ""
            while True:
                key = msvcrt.getch()
                if key == b"\r":  # Enter key
                    sys.stdout.write("\n")
                    return password
                elif key == b"\x08":  # Backspace key
                    if len(password) > 0:
                        password = password[:-1]
                        sys.stdout.write("\b \b")  # Erase the last asterisk
                        sys.stdout.flush()
                else:
                    try:
                        char = key.decode("utf-8")
                        password += char
                        sys.stdout.write("*")  # Mask input with '*'
                        sys.stdout.flush()
                    except UnicodeDecodeError:
                        continue
        else:
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            try:
                tty.setraw(fd)
                sys.stdout.write(prompt)
                sys.stdout.flush()
                password = ""
                while True:
                    ch = sys.stdin.read(1)
                    if ch in ("\r", "\n"):  # Enter key
                        sys.stdout.write("\r\n")
                        return password
                    elif ch == "\x7f":  # Backspace key
                        if len(password) > 0:
                            password = password[:-1]
                            sys.stdout.write("\b \b")  # Erase the last asterisk
                            sys.stdout.flush()
                    else:
                        password += ch
                        sys.stdout.write("*")  # Mask input with '*'
                        sys.stdout.flush()
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    def setup_configuration(self):
        """
        Set up the configuration by prompting the user for various settings.
        """
        # Prompt for service type
        print("\nChoose OpenSearch service type:")
        print("1. Amazon OpenSearch Service")
        print("2. Open-source")
        service_choice = input("Enter your choice (1-2): ").strip()

        if service_choice == "1":
            self.service_type = self.AMAZON_OPENSEARCH_SERVICE
        elif service_choice == "2":
            self.service_type = self.OPEN_SOURCE
        else:
            logger.warning(
                f"\n{Fore.YELLOW}Invalid choice. Defaulting to 'amazon-opensearch-service'.{Style.RESET_ALL}"
            )
            self.service_type = self.AMAZON_OPENSEARCH_SERVICE

        # Based on service type, prompt for different configurations
        if self.service_type == self.AMAZON_OPENSEARCH_SERVICE:
            print("\n--- Amazon OpenSearch Service Setup ---")
            self.configure_aws()

            # Prompt for ARN type
            print("\nChoose ARN type:")
            print("1. IAM Role ARN")
            print("2. IAM User ARN")
            arn_type = input("Enter your choice (1-2): ").strip()

            if arn_type == "1":
                self.aws_config.aws_role_name = (
                    input("Enter your AWS IAM Role ARN: ").strip()
                    or self.aws_config.aws_role_name
                )
            elif arn_type == "2":
                self.aws_config.aws_user_name = (
                    input("Enter your AWS IAM User ARN: ").strip()
                    or self.aws_config.aws_user_name
                )
            else:
                logger.warning(
                    f"\n{Fore.YELLOW}Invalid choice. Defaulting to 'IAM Role ARN'.{Style.RESET_ALL}"
                )
                self.aws_config.aws_role_name = (
                    input("Enter your AWS IAM Role ARN: ").strip()
                    or self.aws_config.aws_role_name
                )

            # Prompt for domain information
            default_region = "us-west-2"
            self.opensearch_config.opensearch_domain_region = (
                input(
                    f"\nEnter your AWS OpenSearch region, or press Enter for default [{default_region}]: "
                ).strip()
                or default_region
            )
            self.opensearch_config.opensearch_domain_endpoint = input(
                "Enter your AWS OpenSearch domain endpoint: "
            ).strip()
            self.opensearch_config.opensearch_domain_username = input(
                "Enter your AWS OpenSearch username: "
            ).strip()
            self.opensearch_config.opensearch_domain_password = (
                self.get_password_with_asterisks("Enter your AWS OpenSearch password: ")
            )
        elif self.service_type == self.OPEN_SOURCE:
            # For open-source, skip AWS configurations
            print("\n--- Open-source OpenSearch Setup ---")
            default_endpoint = "https://localhost:9200"
            self.opensearch_config.opensearch_domain_endpoint = (
                input(
                    f"\nEnter your custom endpoint, or press Enter for default [{default_endpoint}]: "
                ).strip()
                or default_endpoint
            )
            if self.opensearch_config.opensearch_domain_endpoint.startswith("https://"):
                ssl_check = (
                    input(
                        "Do you want to disable SSL certificate check? (yes/no), or press Enter for default 'Enable': "
                    )
                    .strip()
                    .lower()
                    or self.ssl_check_enabled
                )
                if ssl_check == "yes":
                    self.ssl_check_enabled = False
            auth_required = (
                input(
                    "Does your OpenSearch instance require authentication? (yes/no): "
                )
                .strip()
                .lower()
            )
            if auth_required == "yes":
                self.opensearch_config.opensearch_domain_username = input(
                    "Enter your OpenSearch username: "
                ).strip()
                self.opensearch_config.opensearch_domain_password = (
                    self.get_password_with_asterisks("Enter your OpenSearch password: ")
                )
            else:
                self.opensearch_config.opensearch_domain_username = None
                self.opensearch_config.opensearch_domain_password = None

            # AWS OpenSearch region and IAM principal not needed
            self.opensearch_config.opensearch_domain_region = ""
            self.aws_config.aws_role_name = ""
            self.aws_config.aws_user_name = ""

        # Update configuration dictionary
        self.config = {
            "service_type": self.service_type,
            "ssl_check_enabled": self.ssl_check_enabled,
            "opensearch_config": {
                "opensearch_domain_region": self.opensearch_config.opensearch_domain_region,
                "opensearch_domain_endpoint": self.opensearch_config.opensearch_domain_endpoint,
                "opensearch_domain_username": self.opensearch_config.opensearch_domain_username,
                "opensearch_domain_password": self.opensearch_config.opensearch_domain_password,
            },
            "aws_credentials": {
                "aws_role_name": self.aws_config.aws_role_name,
                "aws_user_name": self.aws_config.aws_user_name,
                "aws_access_key": self.aws_config.aws_access_key,
                "aws_secret_access_key": self.aws_config.aws_secret_access_key,
                "aws_session_token": self.aws_config.aws_session_token,
            },
        }
        config_path = self.save_yaml_file(self.config, "configuration", False)
        return config_path

    def initialize_opensearch_client(self) -> bool:
        """
        Initialize the OpenSearch client based on the service type and configuration.
        """
        if not self.opensearch_config.opensearch_domain_endpoint:
            logger.warning(
                f"{Fore.RED}OpenSearch endpoint not set. Please run setup first.{Style.RESET_ALL}\n"
            )
            return False

        parsed_url = urlparse(self.opensearch_config.opensearch_domain_endpoint)
        host = parsed_url.hostname
        is_aos = "amazonaws.com" in host
        port = parsed_url.port or (443 if is_aos else 9200)
        # Determine auth based on service type
        if self.service_type == self.AMAZON_OPENSEARCH_SERVICE:
            if (
                not self.opensearch_config.opensearch_domain_username
                or not self.opensearch_config.opensearch_domain_password
            ):
                logger.warning(
                    f"{Fore.RED}OpenSearch username or password not set. Please run setup first.{Style.RESET_ALL}\n"
                )
                return False
            auth = (
                self.opensearch_config.opensearch_domain_username,
                self.opensearch_config.opensearch_domain_password,
            )
        elif self.service_type == self.OPEN_SOURCE:
            if (
                self.opensearch_config.opensearch_domain_username
                and self.opensearch_config.opensearch_domain_password
            ):
                auth = (
                    self.opensearch_config.opensearch_domain_username,
                    self.opensearch_config.opensearch_domain_password,
                )
            else:
                auth = None
        else:
            logger.warning("Invalid service type. Please check your configuration.")
            return False

        use_ssl = parsed_url.scheme == "https"

        try:
            self.opensearch_client = OpenSearch(
                hosts=[{"host": host, "port": port}],
                http_auth=auth,
                use_ssl=use_ssl,
                verify_certs=self.ssl_check_enabled,
                ssl_show_warn=False,
                connection_class=RequestsHttpConnection,
                pool_maxsize=20,
            )
            print(
                f"{Fore.GREEN}Initialized OpenSearch client with host: {host} and port: {port}{Style.RESET_ALL}\n"
            )
            return True
        except Exception as e:
            logger.error(
                f"{Fore.RED}Error initializing OpenSearch client: {e}{Style.RESET_ALL}\n"
            )
            return False

    def _update_from_config(self):
        """
        Update instance variables from loaded config dictionary
        """
        if not self.config:
            return False

        self.service_type = self.config.get("service_type", "")
        self.ssl_check_enabled = self.config.get("ssl_check_enabled")
        # OpenSearch config
        opensearch_config = self.config.get("opensearch_config", {})
        self.opensearch_config.opensearch_domain_region = opensearch_config.get(
            "opensearch_domain_region", ""
        )
        self.opensearch_config.opensearch_domain_endpoint = opensearch_config.get(
            "opensearch_domain_endpoint", ""
        )
        self.opensearch_config.opensearch_domain_username = opensearch_config.get(
            "opensearch_domain_username", ""
        )
        self.opensearch_config.opensearch_domain_password = opensearch_config.get(
            "opensearch_domain_password", ""
        )
        # AWS credentials
        aws_credentials = self.config.get("aws_credentials", {})
        self.aws_config.aws_role_name = aws_credentials.get("aws_role_name", "")
        self.aws_config.aws_user_name = aws_credentials.get("aws_user_name", "")
        self.aws_config.aws_access_key = aws_credentials.get("aws_access_key", "")
        self.aws_config.aws_secret_access_key = aws_credentials.get(
            "aws_secret_access_key", ""
        )
        self.aws_config.aws_session_token = aws_credentials.get("aws_session_token", "")
        return True

    def _process_config(self, config_path):
        """
        Process the configuration file
        """
        if self.load_config(config_path):
            if self._update_from_config():
                if self.service_type == self.AMAZON_OPENSEARCH_SERVICE:
                    self.check_and_configure_aws(config_path)
                print(
                    f"{Fore.GREEN}\nSetup complete. You are now ready to use the ML features.{Style.RESET_ALL}"
                )
                return config_path
            else:
                logger.warning(
                    f"{Fore.RED}Failed to update configuration.{Style.RESET_ALL}"
                )
        else:
            logger.warning(
                f"{Fore.YELLOW}Could not load existing configuration. Creating new configuration...{Style.RESET_ALL}"
            )
            config_path = self.setup_configuration()

    def setup_command(self, config_path=None):
        """
        Main setup command that orchestrates the entire setup process.
        """
        # Check if setup config file path is given in the command
        if not config_path:
            config_exist = (
                input("\nDo you already have a configuration file? (yes/no): ")
                .strip()
                .lower()
            )
            if config_exist == "yes":
                print("\nGreat! Let's use your existing configuration.")
                config_path = input(
                    "Enter the path to your existing configuration file: "
                ).strip()
                return self._process_config(config_path)
            else:
                print("Let's create a new configuration file.")
                config_path = self.setup_configuration()

            # Initialize OpenSearch client
            if config_path:
                if self.initialize_opensearch_client():
                    print(
                        f"{Fore.GREEN}Setup complete. You are now ready to use the ML features.{Style.RESET_ALL}"
                    )
                    return config_path
                else:
                    # Handle failure to initialize OpenSearch client
                    logger.warning(
                        f"\n{Fore.RED}Failed to initialize OpenSearch client. Setup incomplete.{Style.RESET_ALL}\n"
                    )
                    return None

        return self._process_config(config_path)
