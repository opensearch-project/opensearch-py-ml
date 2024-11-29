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

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

init(autoreset=True)  # Initialize colorama

class Query:
    def __init__(self, config):
        # Initialize the Query class with configuration
        self.config = config
        self.index_name = config.get('index_name')
        self.opensearch = OpenSearchConnector(config)
        self.embedding_model_id = config.get('embedding_model_id')

        # Initialize OpenSearch client
        if not self.initialize_clients():
            print("Failed to initialize clients. Aborting.")
            return

        # Check OpenSearch connection
        if not self.opensearch.check_connection():
            print("Failed to connect to OpenSearch. Please check your configuration.")
            return

    def initialize_clients(self):
        # Initialize OpenSearch client only
        if self.opensearch.initialize_opensearch_client():
            print("OpenSearch client initialized successfully.")
            return True
        else:
            print("Failed to initialize OpenSearch client.")
            return False

    def bulk_query(self, queries, k=5):
        print("Performing bulk semantic search...")

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
    def query_command(self, queries: List[str], num_results=5):
        results = self.bulk_query(queries, k=num_results)

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
                    print("Answer:")
                    for sentence in all_relevant_sentences[:1]:  # Display the top sentence
                        print(sentence)
                else:
                    print("No relevant sentences found.")
            else:
                print("No documents found for this query.")