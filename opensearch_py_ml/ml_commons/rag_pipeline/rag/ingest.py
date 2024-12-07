# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

import csv
import json
import os
from typing import Dict, List

import PyPDF2
from colorama import Fore, Style, init
from opensearchpy import exceptions as opensearch_exceptions
from tqdm import tqdm

from opensearch_py_ml.ml_commons.rag_pipeline.rag.embedding_client import (
    EmbeddingClient,
)
from opensearch_py_ml.ml_commons.rag_pipeline.rag.opensearch_connector import (
    OpenSearchConnector,
)

# Initialize colorama for colored terminal output
init(autoreset=True)  # Initialize colorama


class Ingest:
    """
    Helper class for ingesting various file types into OpenSearch.
    """

    def __init__(self, config):
        """
        Initialize the Ingest class with configuration.

        :param config: Configuration dictionary containing necessary parameters.
        """
        self.config = config
        self.aws_region = config.get("region")
        self.index_name = config.get("index_name")
        self.bedrock_client = None
        self.opensearch = OpenSearchConnector(config)
        self.embedding_model_id = config.get("embedding_model_id")
        self.embedding_client = (
            None  # Will be initialized after OpenSearch client is ready
        )
        self.pipeline_name = config.get(
            "ingest_pipeline_name", "text-chunking-ingest-pipeline"
        )

    def initialize_clients(self) -> bool:
        """
        Initialize the OpenSearch client and the EmbeddingClient.

        :return: True if initialization is successful, False otherwise.
        """
        if self.opensearch.initialize_opensearch_client():
            print("OpenSearch client initialized successfully.")

            # Now that OpenSearch client is initialized, initialize the embedding client
            if not self.embedding_model_id:
                print("Embedding model ID is not set. Please run setup first.")
                return False

            self.embedding_client = EmbeddingClient(
                self.opensearch.opensearch_client, self.embedding_model_id
            )
            return True
        else:
            print("Failed to initialize OpenSearch client.")
            return False

    def ingest_command(self, paths: List[str]):
        """
        Main ingestion command that processes and ingests all valid files from the provided paths.

        :param paths: List of file or directory paths to ingest.
        """
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

        # Define supported file extensions
        supported_extensions = [".csv", ".txt", ".pdf"]
        valid_files = [
            f
            for f in all_files
            if any(f.lower().endswith(ext) for ext in supported_extensions)
        ]

        # Check if there are valid files to ingest
        if not valid_files:
            print(f"{Fore.RED}No valid files found for ingestion.{Style.RESET_ALL}")
            return

        print(
            f"{Fore.GREEN}Found {len(valid_files)} valid files for ingestion.{Style.RESET_ALL}"
        )

        # Process and ingest data from valid files
        self.process_and_ingest_data(valid_files)

    def process_and_ingest_data(self, file_paths: List[str]):
        """
        Processes the provided files, generates embeddings, and ingests the data into OpenSearch.
        """
        # Initialize clients before ingestion
        if not self.initialize_clients():
            print("Failed to initialize clients. Aborting ingestion.")
            return

        # Create the ingest pipeline
        self.create_ingest_pipeline(self.pipeline_name)

        # Retrieve field names from the config
        passage_text_field = self.config.get("passage_text_field", "passage_text")
        self.config.get("passage_chunk_field", "passage_chunk")
        embedding_field = self.config.get("embedding_field", "passage_embedding")

        all_documents = []
        for file_path in file_paths:
            print(f"\nProcessing file: {file_path}")
            documents = self.process_file(file_path)
            all_documents.extend(documents)

        total_documents = len(all_documents)
        print(f"\nTotal documents to process: {total_documents}")

        print("\nGenerating embeddings for the documents...")
        success_count = 0
        error_count = 0

        # Progress bar for embedding generation
        with tqdm(
            total=total_documents,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
        ) as pbar:
            for doc in all_documents:
                try:
                    embedding = self.embedding_client.get_text_embedding(doc["text"])
                    if embedding is not None:
                        doc["embedding"] = embedding
                        success_count += 1
                    else:
                        error_count += 1
                        print(
                            f"{Fore.RED}Error generating embedding for document: {doc['text'][:50]}...{Style.RESET_ALL}"
                        )
                except Exception as e:
                    error_count += 1
                    print(
                        f"{Fore.RED}Error processing document: {str(e)}{Style.RESET_ALL}"
                    )
                pbar.update(1)
                pbar.set_postfix({"Success": success_count, "Errors": error_count})

        print(
            f"\n{Fore.GREEN}Documents with successful embeddings: {success_count}{Style.RESET_ALL}"
        )
        print(
            f"{Fore.RED}Documents with failed embeddings: {error_count}{Style.RESET_ALL}"
        )

        # Check if there are documents to ingest
        if success_count == 0:
            print(
                f"{Fore.RED}No documents to ingest. Aborting ingestion.{Style.RESET_ALL}"
            )
            return

        print(f"\n{Fore.YELLOW}Ingesting data into OpenSearch...{Style.RESET_ALL}")
        actions = []
        for doc in all_documents:
            if "embedding" in doc and doc["embedding"] is not None:
                action = {
                    "_op_type": "index",
                    "_index": self.index_name,
                    "_source": {
                        passage_text_field: doc["text"],
                        embedding_field: {"knn": doc["embedding"]},
                    },
                    "pipeline": self.pipeline_name,
                }
                actions.append(action)

        # Bulk index the documents into OpenSearch
        success, failed = self.opensearch.bulk_index(actions)
        print(
            f"\n{Fore.GREEN}Successfully ingested {success} documents.{Style.RESET_ALL}"
        )
        print(f"{Fore.RED}Failed to ingest {failed} documents.{Style.RESET_ALL}")

    def create_ingest_pipeline(self, pipeline_id: str):
        """
        Creates an ingest pipeline in OpenSearch if it does not already exist.
        :param pipeline_id: ID of the ingest pipeline to create.
        """
        try:
            # Check if the pipeline already exists
            self.opensearch.opensearch_client.ingest.get_pipeline(id=pipeline_id)
            print(f"\nIngest pipeline '{pipeline_id}' already exists.")
        except opensearch_exceptions.NotFoundError:
            # Pipeline does not exist, create it
            source_field = self.config.get("passage_text_field", "passage_text")
            target_field = self.config.get("passage_chunk_field", "passage_chunk")
            embedding_field = self.config.get("embedding_field", "passage_embedding")
            model_id = self.embedding_model_id

            pipeline_body = {
                "description": "A text chunking and embedding ingest pipeline",
                "processors": [
                    {
                        "text_chunking": {
                            "algorithm": {"delimiter": {"delimiter": "."}},
                            "field_map": {source_field: target_field},
                        }
                    },
                    {
                        "text_embedding": {
                            "model_id": model_id,
                            "field_map": {target_field: embedding_field},
                        }
                    },
                ],
            }
            # Create the ingest pipeline
            self.opensearch.opensearch_client.ingest.put_pipeline(
                id=pipeline_id, body=pipeline_body
            )
            print(f"\nIngest pipeline '{pipeline_id}' created successfully.")
        except Exception as e:
            print(f"\nError checking or creating ingest pipeline: {e}")

    def process_file(self, file_path: str) -> List[Dict[str, str]]:
        """
        Processes a file based on its extension and extracts text.

        :param file_path: Path to the file to process.
        :return: List of dictionaries containing extracted text.
        """
        _, file_extension = os.path.splitext(file_path)

        if file_extension.lower() == ".csv":
            return self.process_csv(file_path)
        elif file_extension.lower() == ".txt":
            return self.process_txt(file_path)
        elif file_extension.lower() == ".pdf":
            return self.process_pdf(file_path)
        else:
            print(f"Unsupported file type: {file_extension}")
            return []

    def process_csv(self, file_path: str) -> List[Dict[str, str]]:
        """
        Processes a CSV file and extracts each row as a JSON string.

        :param file_path: Path to the CSV file.
        :return: List of dictionaries with extracted text.
        """
        documents = []
        with open(file_path, "r", newline="", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                documents.append({"text": json.dumps(row)})
        return documents

    def process_txt(self, file_path: str) -> List[Dict[str, str]]:
        """
        Processes a TXT file and reads its entire content.

        :param file_path: Path to the TXT file.
        :return: List containing a single dictionary with the file content.
        """
        with open(file_path, "r") as txtfile:
            content = txtfile.read()
        return [{"text": content}]

    def process_pdf(self, file_path: str) -> List[Dict[str, str]]:
        """
        Processes a PDF file and extracts text from each page.

        :param file_path: Path to the PDF file.
        :return: List of dictionaries, each containing text from a page.
        """
        documents = []
        with open(file_path, "rb") as pdffile:
            pdf_reader = PyPDF2.PdfReader(pdffile)
            for page in pdf_reader.pages:
                extracted_text = page.extract_text()
                if extracted_text:  # Ensure that text was extracted
                    documents.append({"text": extracted_text})
        return documents
