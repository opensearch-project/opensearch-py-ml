# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

"""
Main CLI script for connector creation
"""

import argparse
import os
import sys

from colorama import Fore, Style, init
from rich.console import Console

from opensearch_py_ml.ml_commons.cli.connector_manager import ConnectorManager
from opensearch_py_ml.ml_commons.cli.ml_setup import Setup
from opensearch_py_ml.ml_commons.cli.model_manager import ModelManager

# Initialize colorama for colored terminal output
init(autoreset=True)

# Initialize Rich console for enhanced CLI outputs
console = Console()


class AllowDashActionConnector(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        """
        Enable argparse to correctly handle connector IDs that begin with a dash ('-')
        instead of treating them as argument separators.
        """
        if values is None:
            args = sys.argv
            try:
                connector_idx = args.index("--connectorId")
                if connector_idx + 1 < len(args):
                    value = args[connector_idx + 1]
                    setattr(namespace, self.dest, value)
                    sys.argv.remove("--connectorId")
                    sys.argv.remove(value)
                    return
            except ValueError:
                pass
            parser.error("--connectorId requires a value")
        else:
            setattr(namespace, self.dest, values)


class AllowDashActionModel(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        """
        Enable argparse to correctly handle model IDs that begin with a dash ('-')
        instead of treating them as argument separators.
        """
        if values is None:
            args = sys.argv
            try:
                model_idx = args.index("--modelId")
                if model_idx + 1 < len(args):
                    value = args[model_idx + 1]
                    setattr(namespace, self.dest, value)
                    sys.argv.remove("--modelId")
                    sys.argv.remove(value)
                    return
            except ValueError:
                pass
            parser.error("--modelId requires a value")
        else:
            setattr(namespace, self.dest, values)


def main():
    """
    Main function to handle opensearch-ml CLI commands
    """
    parser = argparse.ArgumentParser(
        prog="opensearch-ml",
        description="OpenSearch ML CLI",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Command Groups:
  connector
    create    Create a new connector

  model
    register  Register a new model
    predict   Predict a model

Examples:
  Initialize the setup:
    opensearch-ml setup

  Initialize the setup from a setup configuration file:
    opensearch-ml setup --path /file/setup_config.yml

  Create a connector:
    opensearch-ml connector create

  Create a connector from a connector configuration file:
    opensearch-ml connector create --path /file/connector_config.yml

  Register a model:
    opensearch-ml model register

  Register a model with a model name, description, and the connector ID
    opensearch-ml model register --connectorId 'connector123' --name 'Test model' --description 'This is a test model'

  Predict a model:
    opensearch-ml model predict

  Predict a model with a model ID and the request payload:
    opensearch-ml model predict --modelId 'model123' --body '{"parameters": {"texts": ["hello world"]}}'
""",
    )
    # Create subparsers for different command groups
    subparsers = parser.add_subparsers(
        title="Available Commands", dest="command", metavar="command"
    )

    # Setup command
    setup = subparsers.add_parser(
        "setup", help="Initialize and configure OpenSearch setup and AWS credentials"
    )
    setup.add_argument(
        "--path", nargs="+", help="Path to the setup configuration file."
    )

    # Create the 'connector' command group
    connector_parser = subparsers.add_parser("connector", help="Manage ML connectors")

    # Create the 'model' command group
    model_parser = subparsers.add_parser("model", help="Manage ML models")

    # Create subcommands for 'connector'
    connector_subparsers = connector_parser.add_subparsers(
        title="Connector Commands", dest="subcommand", metavar="subcommand"
    )

    # Create command
    connector_create = connector_subparsers.add_parser(
        "create", help="Create a new connector"
    )
    connector_create.add_argument(
        "--path", nargs="+", help="Path to the connector configuration file."
    )

    # Create subcommands for 'model'
    model_subparsers = model_parser.add_subparsers(
        title="Model Commands", dest="subcommand", metavar="subcommand"
    )

    # Register command
    model_register = model_subparsers.add_parser(
        "register", help="Register a new model"
    )
    model_register.add_argument(
        "--connectorId",
        action=AllowDashActionConnector,
        help="The connector ID to register the model with.",
        metavar="CONNECTOR_ID",
        nargs="?",
    )
    model_register.add_argument(
        "--name",
        help="Name of the model to register.",
        metavar="MODEL_NAME",
    )
    model_register.add_argument(
        "--description",
        help="Description of the model to register.",
        metavar="MODEL_DESCRIPTION",
    )

    # Predict command
    model_predict = model_subparsers.add_parser("predict", help="Predict a model")
    model_predict.add_argument(
        "--modelId",
        action=AllowDashActionModel,
        help="ID of the model to predict.",
        metavar="MODEL_ID",
        nargs="?",
    )
    model_predict.add_argument(
        "--body", help="Payload of the predict request", metavar="PREDICT_BODY"
    )

    args, unknown = parser.parse_known_args()
    config_path = None
    setup_config_path = None
    connector_manager = ConnectorManager()
    model_manager = ModelManager()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == "setup":
        setup = Setup()
        console.print("\n[bold blue]Starting connector setup...[/bold blue]")
        setup_config_path = args.path[0] if args.path else None
        config_path = setup.setup_command(config_path=setup_config_path)
        # Save the config path to a file for later use
        if config_path:
            config_dir = os.path.expanduser("~/.opensearch-ml")
            os.makedirs(config_dir, exist_ok=True)
            with open(os.path.join(config_dir, "config_path"), "w") as f:
                f.write(config_path)

    if args.command == "connector":
        if not args.subcommand:
            connector_parser.print_help()
            sys.exit(1)

        if args.subcommand == "create":
            console.print("\n[bold blue]Starting connector creation...[/bold blue]")
            connector_config_path = args.path[0] if args.path else None
            create_connector_result = connector_manager.initialize_create_connector(
                connector_config_path=connector_config_path
            )
            if create_connector_result:
                _, setup_config_path = create_connector_result
            # Save the setup config path after creation
            if setup_config_path:
                config_dir = os.path.expanduser("~/.opensearch-ml")
                os.makedirs(config_dir, exist_ok=True)
                with open(os.path.join(config_dir, "config_path"), "w") as f:
                    f.write(setup_config_path)

    if args.command == "model":
        if not args.subcommand:
            model_parser.print_help()
            sys.exit(1)

        if args.subcommand == "register":
            # Read the saved setup config path
            try:
                config_dir = os.path.expanduser("~/.opensearch-ml")
                with open(os.path.join(config_dir, "config_path"), "r") as f:
                    config_path = f.read().strip()
            except FileNotFoundError:
                print(
                    f"{Fore.RED}No setup configuration found. Please run setup first.{Style.RESET_ALL}"
                )
                sys.exit(1)

            console.print("\n[bold blue]Starting model registration...[/bold blue]")
            connector_id = getattr(args, "connectorId", None)
            model_name = args.name if args.name else None
            model_description = args.description if args.description else None
            model_manager.initialize_register_model(
                config_path,
                connector_id=connector_id,
                model_name=model_name,
                model_description=model_description,
            )
        elif args.subcommand == "predict":
            # Read the saved setup config path
            try:
                config_dir = os.path.expanduser("~/.opensearch-ml")
                with open(os.path.join(config_dir, "config_path"), "r") as f:
                    config_path = f.read().strip()
            except FileNotFoundError:
                print(
                    f"{Fore.RED}No setup configuration found. Please run setup first.{Style.RESET_ALL}"
                )
                sys.exit(1)
            console.print("\n[bold blue]Starting model prediction...[/bold blue]")
            model_id = args.modelId if args.modelId else None
            body = args.body if args.body else None
            model_manager.initialize_predict_model(
                config_path, model_id=model_id, body=body
            )


if __name__ == "__main__":
    main()
