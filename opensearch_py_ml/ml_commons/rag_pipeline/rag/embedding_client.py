# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

import time


class EmbeddingClient:
    def __init__(self, opensearch_client, embedding_model_id):
        self.opensearch_client = opensearch_client
        self.embedding_model_id = embedding_model_id

    def get_text_embedding(
        self, text, max_retries=5, initial_delay=1, backoff_factor=2
    ):
        """
        Generate a text embedding using OpenSearch's ML API with retry logic.

        :param text: Text to generate embedding for.
        :param max_retries: Maximum number of retry attempts.
        :param initial_delay: Initial delay between retries in seconds.
        :param backoff_factor: Factor by which the delay increases after each retry.
        :return: Embedding vector or None if generation fails.
        """
        delay = initial_delay
        for attempt in range(max_retries):
            try:
                payload = {"text_docs": [text]}
                response = self.opensearch_client.transport.perform_request(
                    method="POST",
                    url=f"/_plugins/_ml/_predict/text_embedding/{self.embedding_model_id}",
                    body=payload,
                )
                inference_results = response.get("inference_results", [])
                if not inference_results:
                    print(f"No inference results returned for text: {text}")
                    return None
                output = inference_results[0].get("output")

                # Adjust the extraction of embedding data
                if isinstance(output, list) and len(output) > 0:
                    embedding_dict = output[0]
                    if isinstance(embedding_dict, dict) and "data" in embedding_dict:
                        embedding = embedding_dict["data"]
                    else:
                        print(f"Unexpected embedding output format: {output}")
                        return None
                elif isinstance(output, dict) and "data" in output:
                    embedding = output["data"]
                else:
                    print(f"Unexpected embedding output format: {output}")
                    return None

                return embedding
            except Exception as ex:
                print(f"Error on attempt {attempt + 1}: {ex}")
                if attempt == max_retries - 1:
                    raise
                time.sleep(delay)
                delay *= backoff_factor
        return None
