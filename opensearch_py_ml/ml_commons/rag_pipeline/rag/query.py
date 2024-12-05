# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

import json
from colorama import Fore, Style, init
from typing import List
from opensearch_py_ml.ml_commons.rag_pipeline.rag.opensearch_connector import OpenSearchConnector
from opensearch_py_ml.ml_commons.rag_pipeline.rag.embedding_client import EmbeddingClient
import requests
import os
import urllib3
import boto3
import time
import tiktoken

# Disable insecure request warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Initialize colorama for colored terminal output
init(autoreset=True)  # Initialize colorama


class Query:
    """
    Handles querying operations using OpenSearch and integrates with Large Language Models (LLMs) for generating responses.
    Supports both neural and semantic search methods.
    """

    def __init__(self, config):
        """
        Initialize the Query class with the provided configuration.

        :param config: Configuration dictionary containing necessary parameters.
        """
        # Store the configuration
        self.config = config
        self.index_name = config.get('index_name')
        self.opensearch = OpenSearchConnector(config)
        self.embedding_model_id = config.get('embedding_model_id')
        self.llm_model_id = config.get('llm_model_id')  # Get the LLM model ID from config
        self.aws_region = config.get('region')
        self.bedrock_client = None
        self.embedding_client = None  # Will be initialized after OpenSearch client is ready

        # Load LLM configurations from config
        self.llm_config = {
            "maxTokenCount": int(config.get('llm_max_token_count', '1000')),
            "temperature": float(config.get('llm_temperature', '0.7')),
            "topP": float(config.get('llm_top_p', '0.9')),
            "stopSequences": [s.strip() for s in config.get('llm_stop_sequences', '').split(',') if s.strip()]
        }

        # Set the default search method
        self.default_search_method = self.config.get('default_search_method', 'neural')

        # Initialize clients
        if not self.initialize_clients():
            print("Failed to initialize clients. Aborting.")
            return

        # Check OpenSearch connection
        if not self.opensearch.check_connection():
            print("Failed to connect to OpenSearch. Please check your configuration.")
            return

    def initialize_clients(self) -> bool:
        """
        Initialize the OpenSearch client and Bedrock client if LLM is configured.

        :return: True if clients are initialized successfully, False otherwise.
        """
        # Initialize OpenSearch client
        if self.opensearch.initialize_opensearch_client():
            print("OpenSearch client initialized successfully.")

            # Initialize EmbeddingClient now that OpenSearch client is ready
            if not self.embedding_model_id:
                print("Embedding model ID is not set. Please run setup first.")
                return False

            self.embedding_client = EmbeddingClient(self.opensearch.opensearch_client, self.embedding_model_id)

            # Initialize Bedrock client only if LLM model ID is provided
            if self.llm_model_id:
                try:
                    self.bedrock_client = boto3.client('bedrock-runtime', region_name=self.aws_region)
                    print("Bedrock client initialized successfully.")
                except Exception as e:
                    print(f"{Fore.RED}Failed to initialize Bedrock client: {e}{Style.RESET_ALL}")
                    return False
            return True
        else:
            print(f"{Fore.RED}Failed to initialize OpenSearch client.{Style.RESET_ALL}")
            return False

    def extract_relevant_sentences(self, query: str, text: str) -> List[str]:
        """
        Extract relevant sentences from the text based on the query.

        :param query: The user's query.
        :param text: The text from which to extract sentences.
        :return: A list of relevant sentences.
        """
        # Lowercase and remove punctuation from query
        query_processed = ''.join(c.lower() if c.isalnum() or c.isspace() else ' ' for c in query)
        query_words = set(query_processed.split())

        # Split text into sentences based on punctuation and newlines
        import re
        sentences = re.split(r'[\n.!?]+', text)

        sentence_scores = []
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            # Lowercase and remove punctuation from sentence
            sentence_processed = ''.join(c.lower() if c.isalnum() or c.isspace() else ' ' for c in sentence)
            sentence_words = set(sentence_processed.split())
            common_words = query_words.intersection(sentence_words)
            score = len(common_words) / (len(query_words) + 1e-6)  # Normalized score
            if score > 0:
                sentence_scores.append((score, sentence))

        # Sort sentences by score in descending order
        sentence_scores.sort(reverse=True)

        # Return the sentences with highest scores
        top_sentences = [sentence for score, sentence in sentence_scores]
        return top_sentences

    def bulk_query_neural(self, queries: List[str], k: int = 5) -> List[dict]:
        """
        Perform bulk neural searches for a list of queries.

        :param queries: List of query strings.
        :param k: Number of top results to retrieve per query.
        :return: List of results containing query, documents, and number of results.
        """
        results = []
        for query_text in queries:
            try:
                # Perform search using the neural method
                hits = self.opensearch.search(query_text, self.embedding_model_id, k)
                if hits:
                    # Collect the content from the retrieved documents
                    documents = []
                    for hit in hits:
                        source = hit['_source']
                        document = {
                            'score': hit['_score'],
                            'source': source
                        }
                        documents.append(document)
                    num_results = len(hits)
                else:
                    documents = []
                    num_results = 0
                    print(f"{Fore.YELLOW}Warning: No hits found for query '{query_text}'.{Style.RESET_ALL}")

                results.append({
                    'query': query_text,
                    'documents': documents,
                    'num_results': num_results
                })
            except Exception as ex:
                # Handle search errors
                print(f"{Fore.RED}Error performing search for query '{query_text}': {str(ex)}{Style.RESET_ALL}")
                results.append({
                    'query': query_text,
                    'documents': [],
                    'num_results': 0
                })

        return results

    def bulk_query_semantic(self, queries: List[str], k: int = 5) -> List[dict]:
        """
        Perform bulk semantic searches for a list of queries by generating embeddings.

        :param queries: List of query strings.
        :param k: Number of top results to retrieve per query.
        :return: List of results containing query, context, and number of results.
        """
        # Generate embeddings for queries and search OpenSearch index
        # Returns a list of results containing query, context, and number of results
        query_vectors = []
        for query in queries:
            embedding = self.embedding_client.get_text_embedding(query)
            if embedding:
                query_vectors.append(embedding)
            else:
                print(f"{Fore.RED}Failed to generate embedding for query: {query}{Style.RESET_ALL}")
                query_vectors.append(None)

        results = []
        for i, vector in enumerate(query_vectors):
            if vector is None:
                # Handle cases where embedding generation failed
                results.append({
                    'query': queries[i],
                    'context': "",
                    'num_results': 0
                })
                continue
            try:
                # Perform vector-based search
                hits = self.opensearch.search_by_vector(vector, k)
                # Concatenate the retrieved passages as context
                context = '\n'.join([hit['_source']['nominee_text'] for hit in hits])
                results.append({
                    'query': queries[i],
                    'context': context,
                    'num_results': len(hits)
                })
            except Exception as ex:
                # Handle search errors
                print(f"{Fore.RED}Error performing search for query '{queries[i]}': {ex}{Style.RESET_ALL}")
                results.append({
                    'query': queries[i],
                    'context': "",
                    'num_results': 0
                })
        return results


    def generate_answer(self, prompt: str, llm_config: dict) -> str:
        """
        Generate an answer using the configured Large Language Model (LLM).

        :param prompt: The prompt to send to the LLM.
        :param llm_config: Configuration dictionary for the LLM parameters.
        :return: Generated answer as a string or None if generation fails.
        """
        try:
            max_input_tokens = 8192  # Max tokens for the model
            expected_output_tokens = llm_config.get('maxTokenCount', 1000)
            # Adjust the encoding based on the model
            encoding = tiktoken.get_encoding("cl100k_base")  # Use appropriate encoding

            # Encode the prompt to count tokens
            prompt_tokens = encoding.encode(prompt)
            allowable_input_tokens = max_input_tokens - expected_output_tokens

            if len(prompt_tokens) > allowable_input_tokens:
                # Truncate the prompt to fit within the model's token limit
                prompt_tokens = prompt_tokens[:allowable_input_tokens]
                prompt = encoding.decode(prompt_tokens)
                print(f"{Fore.YELLOW}Prompt truncated to {allowable_input_tokens} tokens.{Style.RESET_ALL}")

            # Simplified LLM config with only supported parameters
            llm_config_simplified = {
                'maxTokenCount': expected_output_tokens,
                'temperature': llm_config.get('temperature', 0.7),
                'topP': llm_config.get('topP', 1.0),
                'stopSequences': llm_config.get('stopSequences', [])
            }

            # Prepare the body for the LLM inference request
            body = json.dumps({
                'inputText': prompt,
                'textGenerationConfig': llm_config_simplified
            })

            # Invoke the LLM model using Bedrock client
            response = self.bedrock_client.invoke_model(modelId=self.llm_model_id, body=body)
            response_body = json.loads(response['body'].read())
            results = response_body.get('results', [])
            if not results:
                print(f"{Fore.YELLOW}No results returned from LLM.{Style.RESET_ALL}")
                return None
            answer = results[0].get('outputText', '').strip()
            return answer
        except Exception as ex:
            # Handle errors during answer generation
            print(f"{Fore.RED}Error generating answer from LLM: {ex}{Style.RESET_ALL}")
            return None

    def query_command(self, queries: List[str], num_results: int = 5):
        """
        Handle the querying process by performing either neural or semantic searches and generating answers using LLM.

        :param queries: List of query strings.
        :param num_results: Number of top results to retrieve per query.
        """
        # Retrieve the default search method from config
        search_method = self.default_search_method

        print(f"\nUsing the default search method: {search_method.capitalize()} Search")

        # Process each query until the user decides to exit
        while True:
            if not queries:
                # Prompt the user for a new query
                query_text = input("\nEnter a query (or type 'exit' to finish): ").strip()
                if not query_text or query_text.lower() == 'exit':
                    print("\nExiting query session.")
                    break
                queries = [query_text]

            if search_method == 'neural':
                # Proceed with neural search
                results = self.bulk_query_neural(queries, k=num_results)

                for result in results:
                    print(f"\nQuery: {result['query']}")
                    if result['documents']:
                        all_relevant_sentences = []
                        for doc in result['documents']:
                            passage_chunks = doc['source'].get('passage_chunk', [])
                            if not passage_chunks:
                                continue
                            for passage in passage_chunks:
                                # Extract relevant sentences from each passage
                                relevant_sentences = self.extract_relevant_sentences(result['query'], passage)
                                all_relevant_sentences.extend(relevant_sentences)

                        if all_relevant_sentences:
                            # Output the top relevant sentences
                            print("\nAnswer:")
                            for sentence in all_relevant_sentences[:1]:  # Display the top sentence
                                print(sentence)
                        else:
                            print("\nNo relevant sentences found.")
                    else:
                        print("\nNo documents found for this query.")
            elif search_method == 'semantic':
                # Proceed with semantic search
                if not self.bedrock_client or not self.llm_model_id:
                    print(f"\n{Fore.RED}LLM model is not configured. Please run setup to select an LLM model.{Style.RESET_ALL}")
                    return

                # Use the LLM configurations from setup
                llm_config = self.llm_config

                # Perform semantic search
                results = self.bulk_query_semantic(queries, k=num_results)

                for result in results:
                    print(f"\nQuery: {result['query']}")

                    if not result['context']:
                        print(f"\n{Fore.RED}No context available for this query.{Style.RESET_ALL}")
                        continue

                    # Prepare the augmented prompt with context
                    augmented_prompt = f"""Context: {result['context']}
Based on the above context, please provide a detailed and insightful answer to the following question. Feel free to make reasonable inferences or connections if the context doesn't provide all the information:

Question: {result['query']}

Answer:"""

                    print("\nGenerating answer using LLM...")
                    # Generate the answer using the LLM
                    answer = self.generate_answer(augmented_prompt, llm_config)

                    if answer:
                        # Display the generated answer
                        print("\nGenerated Answer:")
                        print(answer)
                    else:
                        print("\nFailed to generate an answer.")

            # After processing, reset queries to allow for the next input
            queries = []