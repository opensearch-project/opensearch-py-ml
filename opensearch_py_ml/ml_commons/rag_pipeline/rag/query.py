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

import json
from colorama import Fore, Style, init
from typing import List
from opensearch_connector import OpenSearchConnector
import requests
import os
import urllib3
import boto3
import time
import tiktoken

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

init(autoreset=True)  # Initialize colorama

class Query:
    def __init__(self, config):
        # Initialize the Query class with configuration
        self.config = config
        self.index_name = config.get('index_name')
        self.opensearch = OpenSearchConnector(config)
        self.embedding_model_id = config.get('embedding_model_id')
        self.llm_model_id = config.get('llm_model_id')  # Get the LLM model ID from config
        self.aws_region = config.get('region')
        self.bedrock_client = None

        # Initialize the default search method from config
        self.default_search_method = self.config.get('default_search_method', 'neural')

        # Load LLM configurations from config
        self.llm_config = {
            "maxTokenCount": int(config.get('llm_max_token_count', '1000')),
            "temperature": float(config.get('llm_temperature', '0.7')),
            "topP": float(config.get('llm_top_p', '0.9')),
            "stopSequences": [s.strip() for s in config.get('llm_stop_sequences', '').split(',') if s.strip()]
        }

        # Initialize OpenSearch client
        if not self.initialize_clients():
            print("Failed to initialize clients. Aborting.")
            return

        # Check OpenSearch connection
        if not self.opensearch.check_connection():
            print("Failed to connect to OpenSearch. Please check your configuration.")
            return

    def initialize_clients(self):
        # Initialize OpenSearch client and Bedrock client if needed
        if self.opensearch.initialize_opensearch_client():
            print("OpenSearch client initialized successfully.")
            # Initialize Bedrock client only if needed
            if self.llm_model_id:
                try:
                    self.bedrock_client = boto3.client('bedrock-runtime', region_name=self.aws_region)
                    print("Bedrock client initialized successfully.")
                except Exception as e:
                    print(f"Failed to initialize Bedrock client: {e}")
                    return False
            return True
        else:
            print("Failed to initialize OpenSearch client.")
            return False

    def extract_relevant_sentences(self, query, text):
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

    def bulk_query_neural(self, queries, k=5):
        results = []
        for query_text in queries:
            try:
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
                print(f"{Fore.RED}Error performing search for query '{query_text}': {str(ex)}{Style.RESET_ALL}")
                results.append({
                    'query': query_text,
                    'documents': [],
                    'num_results': 0
                })

        return results

    def bulk_query_semantic(self, queries, k=5):
        # Generate embeddings for queries and search OpenSearch index
        # Returns a list of results containing query, context, and number of results
        query_vectors = []
        for query in queries:
            embedding = self.text_embedding(query)
            if embedding:
                query_vectors.append(embedding)
            else:
                print(f"{Fore.RED}Failed to generate embedding for query: {query}{Style.RESET_ALL}")
                query_vectors.append(None)

        results = []
        for i, vector in enumerate(query_vectors):
            if vector is None:
                results.append({
                    'query': queries[i],
                    'context': "",
                    'num_results': 0
                })
                continue
            try:
                hits = self.opensearch.search_by_vector(vector, k)
                context = '\n'.join([hit['_source']['nominee_text'] for hit in hits])
                results.append({
                    'query': queries[i],
                    'context': context,
                    'num_results': len(hits)
                })
            except Exception as ex:
                print(f"{Fore.RED}Error performing search for query '{queries[i]}': {ex}{Style.RESET_ALL}")
                results.append({
                    'query': queries[i],
                    'context': "",
                    'num_results': 0
                })
        return results

    def text_embedding(self, text, max_retries=5, initial_delay=1, backoff_factor=2):
        if self.opensearch.opensearch_client is None:
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

                # Verify that embedding is a list of floats
                if not isinstance(embedding, list) or not all(isinstance(x, (float, int)) for x in embedding):
                    print(f"Embedding is not a list of floats: {embedding}")
                    return None

                return embedding
            except Exception as ex:
                print(f"Error on attempt {attempt + 1}: {ex}")
                if attempt == max_retries - 1:
                    raise
                time.sleep(delay)
                delay *= backoff_factor
        return None

    def generate_answer(self, prompt, llm_config):
        # Generate an answer using the LLM model
        # Handles token limit and configures LLM parameters
        # Returns the generated answer or None if an error occurs
        try:
            max_input_tokens = 8192  # Max tokens for the model
            expected_output_tokens = llm_config.get('maxTokenCount', 1000)
            # Adjust the encoding based on the model
            encoding = tiktoken.get_encoding("cl100k_base")  # Use appropriate encoding

            prompt_tokens = encoding.encode(prompt)
            allowable_input_tokens = max_input_tokens - expected_output_tokens

            if len(prompt_tokens) > allowable_input_tokens:
                # Truncate the prompt to fit within the model's token limit
                prompt_tokens = prompt_tokens[:allowable_input_tokens]
                prompt = encoding.decode(prompt_tokens)
                print(f"Prompt truncated to {allowable_input_tokens} tokens.")

            # Simplified LLM config with only supported parameters
            llm_config = {
                'maxTokenCount': expected_output_tokens,
                'temperature': llm_config.get('temperature', 0.7),
                'topP': llm_config.get('topP', 1.0),
                'stopSequences': llm_config.get('stopSequences', [])
            }

            body = json.dumps({
                'inputText': prompt,
                'textGenerationConfig': llm_config
            })
            response = self.bedrock_client.invoke_model(modelId=self.llm_model_id, body=body)
            response_body = json.loads(response['body'].read())
            results = response_body.get('results', [])
            if not results:
                print("No results returned from LLM.")
                return None
            answer = results[0].get('outputText', '').strip()
            return answer
        except Exception as ex:
            print(f"Error generating answer from LLM: {ex}")
            return None

    def query_command(self, queries: List[str], num_results=5):
        search_method = self.default_search_method

        print(f"\nUsing the default search method: {search_method.capitalize()} Search")

        # Keep the session active until the user types 'exit' or presses Enter without input
        while True:
            if not queries:
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

                results = self.bulk_query_semantic(queries, k=num_results)

                for result in results:
                    print(f"\nQuery: {result['query']}")
                    print(f"Found {result['num_results']} results.")

                    if not result['context']:
                        print(f"\n{Fore.RED}No context available for this query.{Style.RESET_ALL}")
                        continue

                    augmented_prompt = f"""Context: {result['context']}
Based on the above context, please provide a detailed and insightful answer to the following question. Feel free to make reasonable inferences or connections if the context doesn't provide all the information:

Question: {result['query']}

Answer:"""

                    print("\nGenerating answer using LLM...")
                    answer = self.generate_answer(augmented_prompt, llm_config)

                    if answer:
                        print("\nGenerated Answer:")
                        print(answer)
                    else:
                        print("\nFailed to generate an answer.")

            # After processing, reset queries to allow for the next input
            queries = []