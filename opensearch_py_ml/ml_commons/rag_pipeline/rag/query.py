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
import tiktoken
from colorama import Fore, Style, init
from typing import List
import boto3
import botocore
import time
import random
from opensearch_connector import OpenSearchConnector

init(autoreset=True)  # Initialize colorama

class Query:
    LLM_MODEL_ID = 'amazon.titan-text-express-v1'

    def __init__(self, config):
        # Initialize the Query class with configuration
        self.config = config
        self.aws_region = config.get('region')
        self.index_name = config.get('index_name')
        self.bedrock_client = None
        self.opensearch = OpenSearchConnector(config)
        self.embedding_model_id = config.get('embedding_model_id')
    def initialize_clients(self):
        # Initialize AWS Bedrock and OpenSearch clients
        # Returns True if successful, False otherwise
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

                # Optionally, remove debugging print statements
                # print(f"Extracted embedding of length {len(embedding)}")
                return embedding
            except Exception as ex:
                print(f"Error on attempt {attempt + 1}: {ex}")
                if attempt == max_retries - 1:
                    raise
                time.sleep(delay)
                delay *= backoff_factor
        return None

    def bulk_query(self, queries, k=5):
        # Perform bulk semantic search for multiple queries
        # Generates embeddings for queries and searches OpenSearch index
        # Returns a list of results containing query, context, and number of results
        print("Generating embeddings for queries...")
        query_vectors = []
        for query in queries:
            embedding = self.text_embedding(query)
            if embedding:
                query_vectors.append(embedding)
            else:
                print(f"{Fore.RED}Failed to generate embedding for query: {query}{Style.RESET_ALL}")
                query_vectors.append(None)
        
        print("Performing bulk semantic search...")
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
                hits = self.opensearch.search(vector, k)
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

    def generate_answer(self, prompt, config):
        # Generate an answer using the LLM model
        # Handles token limit and configures LLM parameters
        # Returns the generated answer or None if an error occurs
        try:
            max_input_tokens = 8192  # Max tokens for the model
            expected_output_tokens = config.get('maxTokenCount', 1000)
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
                'temperature': config.get('temperature', 0.7),
                'topP': config.get('topP', 1.0),
                'stopSequences': config.get('stopSequences', [])
            }

            body = json.dumps({
                'inputText': prompt,
                'textGenerationConfig': llm_config
            })
            response = self.bedrock_client.invoke_model(modelId=self.LLM_MODEL_ID, body=body)
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
        # Main query command to process multiple queries
        # Performs semantic search and generates answers using LLM
        # Prints results for each query
        if not self.initialize_clients():
            print("Failed to initialize clients. Aborting query.")
            return
        
        results = self.bulk_query(queries, k=num_results)
        
        llm_config = {
            "maxTokenCount": 1000,
            "temperature": 0.7,
            "topP": 0.9,
            "stopSequences": []
        }
        
        for result in results:
            print(f"\nQuery: {result['query']}")
            print(f"Found {result['num_results']} results.")
            
            if not result['context']:
                print(f"{Fore.RED}No context available for this query.{Style.RESET_ALL}")
                continue
            
            augmented_prompt = f"""Context: {result['context']}
Based on the above context, please provide a detailed and insightful answer to the following question. Feel free to make reasonable inferences or connections if the context doesn't provide all the information:

Question: {result['query']}

Answer:"""
        
            print("Generating answer using LLM...")
            answer = self.generate_answer(augmented_prompt, llm_config)
        
            if answer:
                print("Generated Answer:")
                print(answer)
            else:
                print("Failed to generate an answer.")
