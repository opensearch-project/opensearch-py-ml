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

import os
import glob
import json
import tiktoken
from tqdm import tqdm
from colorama import Fore, Style, init
from typing import List, Dict
import csv
import PyPDF2
import boto3
import botocore
import time
import random


from opensearch_connector import OpenSearchConnector

init(autoreset=True)  # Initialize colorama

class Ingest:

    def __init__(self, config):
        # Initialize the Ingest class with configuration
        self.config = config
        self.aws_region = config.get('region')
        self.index_name = config.get('index_name')
        self.bedrock_client = None
        self.opensearch = OpenSearchConnector(config)
        self.embedding_model_id = config.get('embedding_model_id')

        if not self.embedding_model_id:
            print("Embedding model ID is not set. Please run setup first.")
            return

    def initialize_clients(self):
        # Initialize OpenSearch client
        if self.opensearch.initialize_opensearch_client():
            print("OpenSearch client initialized successfully.")
            return True
        else:
            print("Failed to initialize OpenSearch client.")
            return False


    def process_file(self, file_path: str) -> List[Dict[str, str]]:
        # Process a file based on its extension
        # Supports CSV, TXT, and PDF files
        # Returns a list of dictionaries containing extracted text
        _, file_extension = os.path.splitext(file_path)
        
        if file_extension.lower() == '.csv':
            return self.process_csv(file_path)
        elif file_extension.lower() == '.txt':
            return self.process_txt(file_path)
        elif file_extension.lower() == '.pdf':
            return self.process_pdf(file_path)
        else:
            print(f"Unsupported file type: {file_extension}")
            return []

    def process_csv(self, file_path: str) -> List[Dict[str, str]]:
        # Process a CSV file
        # Extracts information and returns a list of dictionaries
        # Each dictionary contains the entire row content
        documents = []
        with open(file_path, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                documents.append({"text": json.dumps(row)})
        return documents


    def process_txt(self, file_path: str) -> List[Dict[str, str]]:
        # Process a TXT file
        # Reads the entire content of the file
        # Returns a list with a single dictionary containing the file content
        with open(file_path, 'r') as txtfile:
            content = txtfile.read()
        return [{"text": content}]

    def process_pdf(self, file_path: str) -> List[Dict[str, str]]:
        # Process a PDF file
        # Extracts text from each page of the PDF
        # Returns a list of dictionaries, each containing text from a page
        documents = []
        with open(file_path, 'rb') as pdffile:
            pdf_reader = PyPDF2.PdfReader(pdffile)
            for page in pdf_reader.pages:
                extracted_text = page.extract_text()
                if extracted_text:  # Ensure that text was extracted
                    documents.append({"text": extracted_text})
        return documents

    def text_embedding(self, text, max_retries=5, initial_delay=1, backoff_factor=2):
        if self.opensearch is None:
            print("OpenSearch client is not initialized. Please run setup first.")
            return None

        delay = initial_delay
        for attempt in range(max_retries):
            try:
                payload = {
                    "text_docs": [text]
                }
                response = self.opensearch.opensearch_client.transport.perform_request(
                    method="POST",
                    url=f"/_plugins/_ml/_predict/text_embedding/{self.embedding_model_id}",
                    body=payload
                )
                inference_results = response.get('inference_results', [])
                if not inference_results:
                    print(f"No inference results returned for text: {text}")
                    return None
                output = inference_results[0].get('output')

                # Remove or comment out the debugging print statements
                # print(f"Output type: {type(output)}")
                # print(f"Output content: {output}")

                # Adjust the extraction of embedding data
                if isinstance(output, list) and len(output) > 0:
                    embedding_dict = output[0]
                    if isinstance(embedding_dict, dict) and 'data' in embedding_dict:
                        embedding = embedding_dict['data']
                    else:
                        print(f"Unexpected embedding output format: {output}")
                        return None
                elif isinstance(output, dict) and 'data' in output:
                    embedding = output['data']
                else:
                    print(f"Unexpected embedding output format: {output}")
                    return None

                # Optionally, you can also remove this print statement if you prefer
                # print(f"Extracted embedding of length {len(embedding)}")
                return embedding
            except Exception as ex:
                print(f"Error on attempt {attempt + 1}: {ex}")
                if attempt == max_retries - 1:
                    raise
                time.sleep(delay)
                delay *= backoff_factor
        return None

    def process_and_ingest_data(self, file_paths: List[str]):
        # Process and ingest data from multiple files
        # Generates embeddings for each document and ingests into OpenSearch
        # Displays progress and results of the ingestion process
        if not self.initialize_clients():
            print("Failed to initialize clients. Aborting ingestion.")
            return

        all_documents = []
        for file_path in file_paths:
            print(f"Processing file: {file_path}")
            documents = self.process_file(file_path)
            all_documents.extend(documents)
        
        total_documents = len(all_documents)
        print(f"Total documents to process: {total_documents}")
        
        print("Generating embeddings for the documents...")
        success_count = 0
        error_count = 0
        with tqdm(total=total_documents, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as pbar:
            for doc in all_documents:
                try:
                    embedding = self.text_embedding(doc['text'])
                    if embedding is not None:
                        doc['embedding'] = embedding
                        success_count += 1
                    else:
                        error_count += 1
                        print(f"{Fore.RED}Error generating embedding for document: {doc['text'][:50]}...{Style.RESET_ALL}")
                except Exception as e:
                    error_count += 1
                    print(f"{Fore.RED}Error processing document: {str(e)}{Style.RESET_ALL}")
                pbar.update(1)
                pbar.set_postfix({'Success': success_count, 'Errors': error_count})
        
        print(f"\n{Fore.GREEN}Documents with successful embeddings: {success_count}{Style.RESET_ALL}")
        print(f"{Fore.RED}Documents with failed embeddings: {error_count}{Style.RESET_ALL}")
        
        if success_count == 0:
            print(f"{Fore.RED}No documents to ingest. Aborting ingestion.{Style.RESET_ALL}")
            return
        
        print(f"{Fore.YELLOW}Ingesting data into OpenSearch...{Style.RESET_ALL}")
        actions = []
        for doc in all_documents:
            if 'embedding' in doc and doc['embedding'] is not None:
                action = {
                    "_op_type": "index",
                    "_index": self.index_name,
                    "_source": {
                        "nominee_text": doc['text'],
                        "nominee_vector": doc['embedding']  # This is now a list of floats
                    }
                }
                actions.append(action)
        
        success, failed = self.opensearch.bulk_index(actions)
        print(f"{Fore.GREEN}Successfully ingested {success} documents.{Style.RESET_ALL}")
        print(f"{Fore.RED}Failed to ingest {failed} documents.{Style.RESET_ALL}")

    def ingest_command(self, paths: List[str]):
        # Main ingestion command
        # Processes all valid files in the given paths and initiates ingestion
        all_files = []
        for path in paths:
            if os.path.isfile(path):
                all_files.append(path)
            elif os.path.isdir(path):
                for root, dirs, files in os.walk(path):
                    for file in files:
                        all_files.append(os.path.join(root, file))
            else:
                print(f"{Fore.YELLOW}Invalid path: {path}{Style.RESET_ALL}")
        
        supported_extensions = ['.csv', '.txt', '.pdf']
        valid_files = [f for f in all_files if any(f.lower().endswith(ext) for ext in supported_extensions)]
        
        if not valid_files:
            print(f"{Fore.RED}No valid files found for ingestion.{Style.RESET_ALL}")
            return
        
        print(f"{Fore.GREEN}Found {len(valid_files)} valid files for ingestion.{Style.RESET_ALL}")
        
        self.process_and_ingest_data(valid_files)