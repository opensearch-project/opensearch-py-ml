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
    EMBEDDING_MODEL_ID = 'amazon.titan-embed-text-v2:0'

    def __init__(self, config):
        self.config = config
        self.aws_region = config.get('region')
        self.index_name = config.get('index_name')
        self.bedrock_client = None
        self.opensearch = OpenSearchConnector(config)

    def initialize_clients(self):
        try:
            self.bedrock_client = boto3.client('bedrock-runtime', region_name=self.aws_region)
            if self.opensearch.initialize_opensearch_client():
                print("Clients initialized successfully.")
                return True
            else:
                print("Failed to initialize OpenSearch client.")
                return False
        except Exception as e:
            print(f"Failed to initialize clients: {e}")
            return False

    def process_file(self, file_path: str) -> List[Dict[str, str]]:
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
        documents = []
        with open(file_path, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                text = f"{row['name']} got nominated under the category, {row['category']}, for the film {row['film']}"
                if row.get('winner', '').lower() != 'true':
                    text += " but did not win"
                documents.append({"text": text})
        return documents

    def process_txt(self, file_path: str) -> List[Dict[str, str]]:
        with open(file_path, 'r') as txtfile:
            content = txtfile.read()
        return [{"text": content}]

    def process_pdf(self, file_path: str) -> List[Dict[str, str]]:
        documents = []
        with open(file_path, 'rb') as pdffile:
            pdf_reader = PyPDF2.PdfReader(pdffile)
            for page in pdf_reader.pages:
                extracted_text = page.extract_text()
                if extracted_text:  # Ensure that text was extracted
                    documents.append({"text": extracted_text})
        return documents

    def text_embedding(self, text, max_retries=5, initial_delay=1, backoff_factor=2):
        if self.bedrock_client is None:
            print("Bedrock client is not initialized. Please run setup first.")
            return None
        
        delay = initial_delay
        for attempt in range(max_retries):
            try:
                payload = {"inputText": text}
                response = self.bedrock_client.invoke_model(modelId=self.EMBEDDING_MODEL_ID, body=json.dumps(payload))
                response_body = json.loads(response['body'].read())
                embedding = response_body.get('embedding')
                if embedding is None:
                    print(f"No embedding returned for text: {text}")
                    print(f"Response body: {response_body}")
                    return None
                return embedding
            except botocore.exceptions.ClientError as e:
                error_code = e.response['Error']['Code']
                error_message = e.response['Error']['Message']
                print(f"ClientError on attempt {attempt + 1}: {error_code} - {error_message}")
                if error_code == 'ThrottlingException':
                    if attempt == max_retries - 1:
                        raise
                    time.sleep(delay + random.uniform(0, 1))
                    delay *= backoff_factor
                else:
                    raise
            except Exception as ex:
                print(f"Unexpected error on attempt {attempt + 1}: {ex}")
                if attempt == max_retries - 1:
                    raise
        return None

    def process_and_ingest_data(self, file_paths: List[str]):
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
                        "nominee_vector": doc['embedding']
                    }
                }
                actions.append(action)
        
        success, failed = self.opensearch.bulk_index(actions)
        print(f"{Fore.GREEN}Successfully ingested {success} documents.{Style.RESET_ALL}")
        print(f"{Fore.RED}Failed to ingest {failed} documents.{Style.RESET_ALL}")

    def ingest_command(self, paths: List[str]):
        all_files = []
        for path in paths:
            if os.path.isfile(path):
                all_files.append(path)
            elif os.path.isdir(path):
                all_files.extend(glob.glob(os.path.join(path, '*')))
            else:
                print(f"{Fore.YELLOW}Invalid path: {path}{Style.RESET_ALL}")
        
        supported_extensions = ['.csv', '.txt', '.pdf']
        valid_files = [f for f in all_files if any(f.lower().endswith(ext) for ext in supported_extensions)]
        
        if not valid_files:
            print(f"{Fore.RED}No valid files found for ingestion.{Style.RESET_ALL}")
            return
        
        print(f"{Fore.GREEN}Found {len(valid_files)} valid files for ingestion.{Style.RESET_ALL}")
        
        self.process_and_ingest_data(valid_files)
