# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

import subprocess
import sys
import termios
import tty
from urllib.parse import urlparse

import boto3
from colorama import Fore, Style, init
from opensearchpy import OpenSearch, RequestsHttpConnection

from opensearch_py_ml.ml_commons.connector.connector_base import ConnectorBase

# Initialize colorama for colored terminal output
init(autoreset=True)


class Setup(ConnectorBase):
    """
    Handles the setup and configuration of the OpenSearch environment.
    Manages AWS credentials (if managed) and OpenSearch client initialization.
    """

    def __init__(self):
        """
        Initialize the Setup class with default or existing configurations.
        """
        super().__init__()
        self.config = {}
        self.service_type = ""
        self.opensearch_domain_region = ""
        self.opensearch_domain_endpoint = ""
        self.opensearch_domain_username = ""
        self.opensearch_domain_password = ""
        self.aws_user_name = ""
        self.aws_role_name = ""
        self.opensearch_client = None
        self.opensearch_domain_name = self.get_opensearch_domain_name()

    def check_and_configure_aws(self):
        """
        Check if AWS credentials are configured and offer to reconfigure if needed.
        """
        try:
            session = boto3.Session()
            credentials = session.get_credentials()

            if credentials is None:
                print(
                    f"{Fore.YELLOW}AWS credentials are not configured.{Style.RESET_ALL}"
                )
                self.configure_aws()
            else:
                print("\nAWS credentials are already configured.")
                reconfigure = input("Do you want to reconfigure? (yes/no): ").lower()
                if reconfigure == "yes":
                    self.configure_aws()
        except Exception as e:
            print(
                f"{Fore.RED}An error occurred while checking AWS credentials: {e}{Style.RESET_ALL}"
            )
            self.configure_aws()

    def configure_aws(self):
        """
        Configure AWS credentials using user input.
        """
        print("Let's configure your AWS credentials.")

        aws_access_key = self.get_password_with_asterisks(
            "Enter your AWS Access Key ID: "
        )
        aws_secret_access_key = self.get_password_with_asterisks(
            "Enter your AWS Secret Access Key: "
        )
        aws_session_token = self.get_password_with_asterisks(
            "Enter your AWS Session Token: "
        )

        try:
            # Configure AWS credentials using subprocess to call AWS CLI
            subprocess.run(
                ["aws", "configure", "set", "aws_access_key", aws_access_key],
                check=True,
            )

            subprocess.run(
                [
                    "aws",
                    "configure",
                    "set",
                    "aws_secret_access_key",
                    aws_secret_access_key,
                ],
                check=True,
            )

            subprocess.run(
                ["aws", "configure", "set", "aws_session_token", aws_session_token],
                check=True,
            )
            print(
                f"{Fore.GREEN}AWS credentials have been successfully configured.{Style.RESET_ALL}"
            )
        except subprocess.CalledProcessError as e:
            print(
                f"{Fore.RED}An error occurred while configuring AWS credentials: {e}{Style.RESET_ALL}"
            )
        except Exception as e:
            print(f"{Fore.RED}An unexpected error occurred: {e}{Style.RESET_ALL}")

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
        print("1. Managed")
        print("2. Open-source")
        service_choice = input("Enter your choice (1-2): ").strip()

        if service_choice == "1":
            self.service_type = "managed"
        elif service_choice == "2":
            self.service_type = "open-source"
        else:
            print(
                f"\n{Fore.YELLOW}Invalid choice. Defaulting to 'managed'.{Style.RESET_ALL}"
            )
            self.service_type = "managed"

        # Based on service type, prompt for different configurations
        if self.service_type == "managed":
            print("\n--- Managed OpenSearch Setup ---")
            self.check_and_configure_aws()

            # Prompt for ARN type
            print("\nChoose ARN type:")
            print("1. IAM Role ARN")
            print("2. IAM User ARN")
            arn_type = input("Enter your choice (1-2): ").strip()

            if arn_type == "1":
                self.aws_role_name = (
                    input(
                        f"Enter your AWS IAM Role ARN [{self.aws_role_name}]: "
                    ).strip()
                    or self.aws_role_name
                )
            elif arn_type == "2":
                self.aws_user_name = (
                    input(
                        f"Enter your AWS IAM User ARN [{self.aws_user_name}]: "
                    ).strip()
                    or self.aws_user_name
                )
            else:
                print(
                    f"\n{Fore.YELLOW}Invalid choice. Defaulting to 'IAM Role ARN'.{Style.RESET_ALL}"
                )
                self.aws_role_name = (
                    input(
                        f"Enter your AWS IAM Role ARN [{self.aws_role_name}]: "
                    ).strip()
                    or self.aws_role_name
                )

            default_region = "us-west-2"
            self.opensearch_domain_region = (
                input(
                    f"\nEnter your AWS OpenSearch region, or press Enter for default [{default_region}]: "
                ).strip()
                or default_region
            )
            self.opensearch_domain_endpoint = input(
                "Enter your AWS OpenSearch domain endpoint: "
            ).strip()
            self.opensearch_domain_username = input(
                "Enter your AWS OpenSearch username: "
            ).strip()
            self.opensearch_domain_password = self.get_password_with_asterisks(
                "Enter your AWS OpenSearch password: "
            )
        elif self.service_type == "open-source":
            # For open-source, skip AWS configurations
            print("\n--- Open-source OpenSearch Setup ---")
            default_endpoint = "https://localhost:9200"
            self.opensearch_domain_endpoint = (
                input(
                    f"\nEnter your custom endpoint, or press Enter for default [{default_endpoint}]: "
                ).strip()
                or default_endpoint
            )
            auth_required = (
                input(
                    "Does your OpenSearch instance require authentication? (yes/no): "
                )
                .strip()
                .lower()
            )
            if auth_required == "yes":
                self.opensearch_domain_username = input(
                    "Enter your OpenSearch username: "
                ).strip()
                self.opensearch_domain_password = self.get_password_with_asterisks(
                    "Enter your OpenSearch password: "
                )
            else:
                self.opensearch_domain_username = None
                self.opensearch_domain_password = None

            # AWS OpenSearch region and IAM principal not needed
            self.opensearch_domain_region = ""
            self.aws_role_name = ""
            self.aws_user_name = ""

        # Update configuration dictionary
        self.config = {
            "service_type": self.service_type,
            "opensearch_domain_region": self.opensearch_domain_region,
            "opensearch_domain_endpoint": (
                self.opensearch_domain_endpoint
                if self.opensearch_domain_endpoint
                else ""
            ),
            "opensearch_domain_username": (
                self.opensearch_domain_username
                if self.opensearch_domain_username
                else ""
            ),
            "opensearch_domain_password": (
                self.opensearch_domain_password
                if self.opensearch_domain_password
                else ""
            ),
            "aws_role_name": self.aws_role_name if self.aws_role_name else "",
            "aws_user_name": self.aws_user_name if self.aws_user_name else "",
            "opensearch_domain_name": (
                self.get_opensearch_domain_name()
                if self.get_opensearch_domain_name()
                else ""
            ),
        }
        self.save_config(self.config)

    def get_opensearch_domain_name(self) -> str:
        """
        Extract the domain name from the OpenSearch endpoint URL.
        """
        if self.opensearch_domain_endpoint:
            parsed_url = urlparse(self.opensearch_domain_endpoint)
            hostname = (
                parsed_url.hostname
            )  # e.g., 'search-your-domain-name-uniqueid.region.es.amazonaws.com'
            if hostname:
                # Split the hostname into parts
                parts = hostname.split(".")
                domain_part = parts[0]  # e.g., 'search-your-domain-name-uniqueid'
                # Remove the 'search-' prefix if present
                if domain_part.startswith("search-"):
                    domain_part = domain_part[len("search-") :]
                # Remove the unique ID suffix after the domain name
                domain_name = domain_part.rsplit("-", 1)[0]
                return domain_name
        return None

    def initialize_opensearch_client(self) -> bool:
        """
        Initialize the OpenSearch client based on the service type and configuration.
        """
        if not self.opensearch_domain_endpoint:
            print(
                f"{Fore.RED}OpenSearch endpoint not set. Please run setup first.{Style.RESET_ALL}\n"
            )
            return False

        parsed_url = urlparse(self.opensearch_domain_endpoint)
        host = parsed_url.hostname
        port = parsed_url.port or (443 if parsed_url.scheme == "https" else 9200)

        # Determine auth based on service type
        if self.service_type == "managed":
            if (
                not self.opensearch_domain_username
                or not self.opensearch_domain_password
            ):
                print(
                    f"{Fore.RED}OpenSearch username or password not set. Please run setup first.{Style.RESET_ALL}\n"
                )
                return False
            auth = (self.opensearch_domain_username, self.opensearch_domain_password)
        elif self.service_type == "open-source":
            if self.opensearch_domain_username and self.opensearch_domain_password:
                auth = (
                    self.opensearch_domain_username,
                    self.opensearch_domain_password,
                )
            else:
                auth = None
        else:
            print("Invalid service type. Please check your configuration.")
            return False

        use_ssl = parsed_url.scheme == "https"
        verify_certs = True

        try:
            self.opensearch_client = OpenSearch(
                hosts=[{"host": host, "port": port}],
                http_auth=auth,
                use_ssl=use_ssl,
                verify_certs=verify_certs,
                ssl_show_warn=False,
                connection_class=RequestsHttpConnection,
                pool_maxsize=20,
            )
            print(
                f"{Fore.GREEN}Initialized OpenSearch client with host: {host} and port: {port}{Style.RESET_ALL}\n"
            )
            return True
        except Exception as ex:
            print(
                f"{Fore.RED}Error initializing OpenSearch client: {ex}{Style.RESET_ALL}\n"
            )
            return False

    def _update_from_config(self):
        """
        Update instance variables from loaded config dictionary
        """
        if not self.config:
            return False

        self.service_type = self.config.get("service_type", "")
        self.opensearch_domain_region = self.config.get("opensearch_domain_region", "")
        self.opensearch_domain_endpoint = self.config.get(
            "opensearch_domain_endpoint", ""
        )
        self.opensearch_domain_username = self.config.get(
            "opensearch_domain_username", ""
        )
        self.opensearch_domain_password = self.config.get(
            "opensearch_domain_password", ""
        )
        self.aws_role_name = self.config.get("aws_role_name", "")
        self.aws_user_name = self.config.get("aws_user_name", "")
        self.opensearch_domain_name = self.get_opensearch_domain_name()
        return True

    def setup_command(self):
        """
        Main setup command that orchestrates the entire setup process.
        """
        # Ask user if they already have a configuration file
        config_exist = (
            input("\nDo you already have a configuration file? (yes/no): ")
            .strip()
            .lower()
        )
        if config_exist == "yes":
            print("\nGreat! Let's use your existing configuration.")
            config_path = input(
                "Enter the path to your existing configuration file (connector_config.yml): "
            ).strip()

            if self.load_config(config_path):
                if self._update_from_config():
                    print(f"{Fore.GREEN}\nConfiguration file loaded successfully.")
                    return True
                else:
                    print(f"{Fore.RED}Failed to update configuration.{Style.RESET_ALL}")
            else:
                print(
                    f"{Fore.YELLOW}Could not load existing configuration. Creating new configuration...{Style.RESET_ALL}"
                )
        else:
            print("Let's create a new configuration file.")
            self.setup_configuration()

        if not self.opensearch_domain_endpoint:
            print(
                f"\n{Fore.RED}OpenSearch endpoint not set. Setup incomplete.{Style.RESET_ALL}\n"
            )
            return
        else:
            if self.service_type == "managed":
                self.opensearch_domain_name = self.get_opensearch_domain_name()
            else:
                self.opensearch_domain_name = None

        # Initialize OpenSearch client
        if self.initialize_opensearch_client():
            print(
                f"{Fore.GREEN}Setup complete. You are now ready to begin the connector creation process{Style.RESET_ALL}"
            )
        else:
            # Handle failure to initialize OpenSearch client
            print(
                f"\n{Fore.RED}Failed to initialize OpenSearch client. Setup incomplete.{Style.RESET_ALL}\n"
            )
