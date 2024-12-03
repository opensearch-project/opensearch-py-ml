#!/usr/bin/env python3  # SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors.
# See GitHub history for details.

# Licensed to Elasticsearch B.V. under one or more contributor
# license agreements. See the NOTICE file distributed with
# this work for additional information regarding copyright
# ownership. Elasticsearch B.V. licenses this file to you under
# the Apache License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for
# the specific language governing permissions and limitations
# under the License.
"""
Main CLI script for OpenSearch PY ML
"""

import argparse
import configparser
import sys
from colorama import Fore, Style, init
from rich.console import Console
from rich.prompt import Prompt
from rag_setup import Setup
from ingest import Ingest
from query import Query

# Initialize colorama
init(autoreset=True)

# Initialize Rich console
console = Console()

CONFIG_FILE = 'config.ini'


def load_config():
    """
    Load configuration from the config file.
    """
    config = configparser.ConfigParser()
    config.read(CONFIG_FILE)
    if 'DEFAULT' not in config:
        console.print(f"[{Fore.RED}ERROR{Style.RESET_ALL}] 'DEFAULT' section missing in {CONFIG_FILE}. Please run the setup command first.")
        sys.exit(1)
    return config['DEFAULT']


def save_config(config):
    """
    Save configuration to the config file.
    """
    parser = configparser.ConfigParser()
    parser['DEFAULT'] = config
    try:
        with open(CONFIG_FILE, 'w') as f:
            parser.write(f)
        console.print(f"[{Fore.GREEN}SUCCESS{Style.RESET_ALL}] Configuration saved to {CONFIG_FILE}.")
    except Exception as e:
        console.print(f"[{Fore.RED}ERROR{Style.RESET_ALL}] Failed to save configuration: {e}")


def main():
    """
    Main function to parse arguments and execute commands.
    """
    # Set up argument parser for CLI with Rich help formatting
    parser = argparse.ArgumentParser(
        description="RAG Pipeline CLI",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Examples:
Initialize the setup:
  rag setup

Ingest documents from multiple paths:
  rag ingest --paths /data/docs /data/reports

Execute queries with default number of results:
  rag query --queries "What is OpenSearch?" "How does Bedrock work?"

Execute queries with a specified number of results:
  rag query --queries "What is OpenSearch?" --num_results 3
"""
    )
    subparsers = parser.add_subparsers(title="Available Commands", dest="command")

    # Setup command
    setup_parser = subparsers.add_parser(
        'setup',
        help='Initialize and configure the RAG pipeline.'
    )

    # Ingest command
    ingest_parser = subparsers.add_parser(
        'ingest',
        help='Ingest documents into OpenSearch.'
    )
    ingest_parser.add_argument(
        '--paths',
        nargs='+',
        help='Paths to files or directories for ingestion.'
    )

    # Query command
    query_parser = subparsers.add_parser(
        'query',
        help='Execute queries and generate answers.'
    )
    query_parser.add_argument(
        '--queries',
        nargs='+',
        help='Query texts for search and answer generation.'
    )
    query_parser.add_argument(
        '--num_results',
        type=int,
        default=5,
        help='Number of top results to retrieve for each query. (default: 5)'
    )

    # Parse arguments
    args = parser.parse_args()

    # Only display the banner if no command is executed
    if not args.command:
        console.print("[bold cyan]Welcome to the RAG Pipeline[/bold cyan]")
        console.print("Use [bold blue]rag setup[/bold blue], [bold blue]rag ingest[/bold blue], or [bold blue]rag query[/bold blue] to begin.\n")

    # Load existing configuration
    if args.command != 'setup' and args.command:
        config = load_config()
    else:
        config = None  # Setup may create the config

    # Handle commands
    if args.command == 'setup':
        # Run setup process
        setup = Setup()
        console.print("[bold blue]Starting setup process...[/bold blue]")
        setup.setup_command()
        save_config(setup.config)
    elif args.command == 'ingest':
        # Handle ingestion command
        if not args.paths:
            # If no paths provided as arguments, prompt user for input
            paths = []
            while True:
                path = Prompt.ask("Enter a file or directory path (or press Enter to finish)", default="", show_default=False)
                if not path:
                    break
                paths.append(path)
        else:
            paths = args.paths
        if not paths:
            console.print(f"[{Fore.RED}ERROR{Style.RESET_ALL}] No paths provided for ingestion. Aborting.")
            sys.exit(1)
        ingest = Ingest(config)
        ingest.ingest_command(paths)
    elif args.command == 'query':
        # Handle query command
        if not args.queries:
            # If no queries provided as arguments, prompt user for input
            queries = []
            while True:
                query = Prompt.ask("Enter a query (or press Enter to finish)", default="", show_default=False)
                if not query:
                    break
                queries.append(query)
        else:
            queries = args.queries
        if not queries:
            console.print(f"[{Fore.RED}ERROR{Style.RESET_ALL}] No queries provided. Aborting.")
            sys.exit(1)
        query = Query(config)
        query.query_command(queries, num_results=args.num_results)
    else:
        # If an invalid command is provided, print help
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
