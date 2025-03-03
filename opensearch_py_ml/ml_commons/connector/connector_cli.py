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
import sys

from colorama import init
from rich.console import Console

from opensearch_py_ml.ml_commons.connector.connector_base import ConnectorBase
from opensearch_py_ml.ml_commons.connector.connector_create import Create
from opensearch_py_ml.ml_commons.connector.connector_setup import Setup

# Initialize colorama for colored terminal output
init(autoreset=True)

# Initialize Rich console for enhanced CLI outputs
console = Console()


def main():
    """
    Main function to parse arguments and execute commands.
    """
    # Set up argument parser for CLI with Rich help formatting
    parser = argparse.ArgumentParser(
        description="Connector Creation CLI",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Examples:
Initialize the setup:
connector setup

Creat connector:
connector create
""",
    )
    subparsers = parser.add_subparsers(title="Available Commands", dest="command")

    # Setup command
    subparsers.add_parser("setup", help="Initialize and configure connector creation.")

    # Create command
    subparsers.add_parser("create", help="Create connector.")

    # Parse arguments
    args = parser.parse_args()

    # Only display the banner if no command is executed
    if not args.command:
        console.print("[bold cyan]Welcome to the Connector Creation[/bold cyan]")
        console.print(
            "Use [bold blue]connector setup[/bold blue], or [bold blue]connector create[/bold blue] to begin.\n"
        )

    # Handle commands
    if args.command == "setup":
        # Handle setup command
        setup = Setup()
        console.print("[bold blue]Starting connector setup...[/bold blue]")
        setup.setup_command()
        ConnectorBase.save_config(setup.config)
    elif args.command == "create":
        # Handle create command
        create = Create()
        console.print("[bold blue]Starting connector creation...[/bold blue]")
        create.create_command()
        ConnectorBase.save_config(create.config)
    else:
        # If an invalid command is provided, print help
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
