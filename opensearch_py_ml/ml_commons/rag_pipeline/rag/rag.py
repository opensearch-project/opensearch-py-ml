#!/usr/bin/env python3

"""
Main CLI script for OpenSearch with Bedrock Integration
"""

import argparse
import configparser
from rag_setup import SetupClass
from ingest import IngestClass
from query import QueryClass

CONFIG_FILE = 'config.ini'

def load_config():
    config = configparser.ConfigParser()
    config.read(CONFIG_FILE)
    return config['DEFAULT']

def save_config(config):
    parser = configparser.ConfigParser()
    parser['DEFAULT'] = config
    with open(CONFIG_FILE, 'w') as f:
        parser.write(f)
    
def main():
    parser = argparse.ArgumentParser(description="RAG Pipeline CLI")
    parser.add_argument('command', choices=['setup', 'ingest', 'query'], help='Command to run')
    parser.add_argument('--paths', nargs='+', help='Paths to files or directories for ingestion')
    parser.add_argument('--queries', nargs='+', help='Query texts for search and answer generation')
    parser.add_argument('--num_results', type=int, default=5, help='Number of top results to retrieve for each query')

    args = parser.parse_args()

    config = load_config()

    if args.command == 'setup':
        setup = SetupClass()
        setup.setup_command()
        save_config(setup.config)
    elif args.command == 'ingest':
        if not args.paths:
            paths = []
            while True:
                path = input("Enter a file or directory path (or press Enter to finish): ")
                if not path:
                    break
                paths.append(path)
        else:
            paths = args.paths
        ingest = IngestClass(config)
        ingest.ingest_command(paths)
    elif args.command == 'query':
        if not args.queries:
            queries = []
            while True:
                query = input("Enter a query (or press Enter to finish): ")
                if not query:
                    break
                queries.append(query)
        else:
            queries = args.queries
        query = QueryClass(config)
        query.query_command(queries, num_results=args.num_results)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()