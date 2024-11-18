#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.


#  Licensed to Elasticsearch B.V. under one or more contributor
#  license agreements. See the NOTICE file distributed with
#  this work for additional information regarding copyright
#  ownership. Elasticsearch B.V. licenses this file to you under
#  the Apache License, Version 2.0 (the "License"); you may
#  not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
# 	http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing,
#  software distributed under the License is distributed on an
#  "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
#  KIND, either express or implied.  See the License for the
#  specific language governing permissions and limitations
#  under the License.



"""
Main CLI script for OpenSearch with Bedrock Integration
"""

import argparse
import configparser
from rag_setup import Setup
from ingest import Ingest
from query import Query

CONFIG_FILE = 'config.ini'

def load_config():
    # Load configuration from the config file
    config = configparser.ConfigParser()
    config.read(CONFIG_FILE)
    return config['DEFAULT']

def save_config(config):
    # Save configuration to the config file
    parser = configparser.ConfigParser()
    parser['DEFAULT'] = config
    with open(CONFIG_FILE, 'w') as f:
        parser.write(f)
    
def main():
    # Set up argument parser for CLI
    parser = argparse.ArgumentParser(description="RAG Pipeline CLI")
    parser.add_argument('command', choices=['setup', 'ingest', 'query'], help='Command to run')
    parser.add_argument('--paths', nargs='+', help='Paths to files or directories for ingestion')
    parser.add_argument('--queries', nargs='+', help='Query texts for search and answer generation')
    parser.add_argument('--num_results', type=int, default=5, help='Number of top results to retrieve for each query')

    args = parser.parse_args()

    # Load existing configuration
    config = load_config()

    if args.command == 'setup':
        # Run setup process
        setup = Setup()
        setup.setup_command()
        save_config(setup.config)
    elif args.command == 'ingest':
        # Handle ingestion command
        if not args.paths:
            # If no paths provided as arguments, prompt user for input
            paths = []
            while True:
                path = input("Enter a file or directory path (or press Enter to finish): ")
                if not path:
                    break
                paths.append(path)
        else:
            paths = args.paths
        ingest = Ingest(config)
        ingest.ingest_command(paths)
    elif args.command == 'query':
        # Handle query command
        if not args.queries:
            # If no queries provided as arguments, prompt user for input
            queries = []
            while True:
                query = input("Enter a query (or press Enter to finish): ")
                if not query:
                    break
                queries.append(query)
        else:
            queries = args.queries
        query = Query(config)
        query.query_command(queries, num_results=args.num_results)
    else:
        # If an invalid command is provided, print help
        parser.print_help()

if __name__ == "__main__":
    main()
