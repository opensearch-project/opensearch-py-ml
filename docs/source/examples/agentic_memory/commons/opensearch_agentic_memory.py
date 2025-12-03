import requests
import os
from typing import Dict, Optional, Any


class OpenSearchAgenticMemory:
    def __init__(self, cluster_url: str,
                 username: str,
                 password: str,
                 verify_ssl: str,
                 memory_container_id: str = None,
                 memory_container_name: str = "Strands agent memory container",
                 memory_container_description: str = "default",
                 embedding_model_id: Optional[str] = None,
                 llm_id: Optional[str] = None,
                 infer: bool = False,
                 long_term: bool = False):
        self.memory_container_id = memory_container_id
        self.memory_container_name = memory_container_name
        self.base_url = cluster_url
        self.auth = (username, password)
        self.verify = verify_ssl
        self.headers = {"Content-Type": "application/json"}
        self.embedding_model_id = embedding_model_id
        self.llm_id = llm_id
        self.long_term = long_term

        if memory_container_id is None:
            default_container_id = self.get_memory_container(memory_container_name)
            if default_container_id is None:                
                self.create_memory_container(memory_container_name, memory_container_description, memory_container_name, embedding_model_id, llm_id, long_term)
            else:
                print("Find memory container with id '{}' by name '{}'".format(default_container_id, memory_container_name))
                self.memory_container_id = default_container_id


    def get_memory_container(self, name: str) -> Dict:
        url = f"{self.base_url}/_plugins/_ml/memory_containers/_search"
        body = {
            "query": {
                "bool": {
                    "filter": [
                        {
                            "term": {
                                "name.keyword": name
                            }
                        }
                    ]
                }
            },
            "sort": [
                {
                    "created_time": {
                        "order": "asc"
                    }
                }
            ],
            "size": 1
        }

        response = self._make_request("GET", url, json=body)
        first_hit = self._get_first_hit(response)
        if first_hit is None:
            return None
        return first_hit['_id']

    def create_memory_container(self, name: str, description: str, index_prefix: str,
                                embedding_model_id: Optional[str] = None,
                                llm_id: Optional[str] = None,
                                long_term: bool = False) -> Dict:
        url = f"{self.base_url}/_plugins/_ml/memory_containers/_create"

        # Long-term memory
        if long_term:
            # Auto-create models if not provided
            if not embedding_model_id:
                embedding_model_id = self._create_embedding_model()
            if not llm_id:
                llm_id = self._create_llm_model()
            body = {
                "name": name,
                "description": description,
                "configuration": {
                    "index_prefix": index_prefix,
                    "embedding_model_type": "TEXT_EMBEDDING",
                    "embedding_model_id": embedding_model_id,
                    "embedding_dimension": 1024,
                    "llm_id": llm_id,
                    "use_system_index": False,
                    "disable_session": False,
                    "strategies": [
                        {
                            "enabled": True,
                            "type": "SEMANTIC",
                            "namespace": ["user_id"]
                        }
                    ],
                    "index_settings": {
                        "session_index": {
                            "index": {
                            "number_of_shards": "1",
                            "auto_expand_replicas": "0-all"
                            }
                        },
                        "working_memory_index": {
                            "index": {
                            "number_of_shards": "1",
                            "auto_expand_replicas": "0-all"
                            }
                        },
                        "long_term_memory_index": {
                            "index": {
                            "number_of_shards": "1",
                            "auto_expand_replicas": "0-all"
                            }
                        },
                        "long_term_memory_history_index": {
                            "index": {
                            "number_of_shards": "1",
                            "auto_expand_replicas": "0-all"
                            }
                        }
                    }
                }
            }
        else:
            # Short-term memory
            body = {
                "name": name,
                "description": description,
                "configuration": {
                    "index_prefix": index_prefix,
                    "use_system_index": False,
                    "disable_session": False,
                    "index_settings": {
                        "session_index": {
                            "index": {
                            "number_of_shards": "1",
                            "auto_expand_replicas": "0-all"
                            }
                        },
                        "working_memory_index": {
                            "index": {
                            "number_of_shards": "1",
                            "auto_expand_replicas": "0-all"
                            }
                        }
                    }
                }
            }

        response = self._make_request("POST", url, json=body)
        self.memory_container_id = response['memory_container_id']
        print("Created memory container with id '{}'".format(self.memory_container_id))
        return response

    def create_session(self, session_id: str, metadata: Dict[str, Any], agents: Dict[str, Any] = None) -> Dict:
        url = f"{self.base_url}/_plugins/_ml/memory_containers/{self.memory_container_id}/memories/sessions"
        body = {
            "session_id": session_id,
        }
        if metadata:
            body["metadata"] = metadata
        if agents:
            body["agents"] = agents

        return self._make_request("POST", url, json=body)

    def update_session(self, session_id: str, metadata: Dict[str, Any], agents: Dict[str, Any]) -> Dict:
        url = f"{self.base_url}/_plugins/_ml/memory_containers/{self.memory_container_id}/memories/sessions/{session_id}"
        body = {
            "name": session_id,
        }
        if metadata:
            body["metadata"] = metadata
        if agents:
            body["agents"] = agents

        return self._make_request("PUT", url, json=body)

    def get_session(self, session_id: str) -> Dict:
        url = f"{self.base_url}/_plugins/_ml/memory_containers/{self.memory_container_id}/memories/sessions/{session_id}"
        return self._make_request("GET", url)

    def delete_session(self, session_id: str) -> Dict:
        url = f"{self.base_url}/_plugins/_ml/memory_containers/{self.memory_container_id}/memories/sessions/{session_id}"

        return self._make_request("DELETE", url)

    def add_message(self, session_id: str, agent_id: str, message: Dict[str, Any], infer: bool = False, user_id: str = None) -> Dict:
        url = f"{self.base_url}/_plugins/_ml/memory_containers/{self.memory_container_id}/memories"
        namespace = {
            "session_id": session_id,
            "agent_id": agent_id,
        }
        if user_id:
            namespace["user_id"] = user_id
            
        body = {
            "namespace": namespace,
            "infer": infer,
            "memory_type": "conversational",
        }
        if "message" in message:
            body["messages"] = [
                message['message']
            ]
            message.pop('message', None)
        if "message_id" in message:
            body["message_id"] = message['message_id']
            message.pop('message_id', None)
        message = {k: v for k, v in message.items() if v is not None}
        if message:
            body['metadata'] = message

        return self._make_request("POST", url, json=body)

    def search_session(self, session_id: str) -> Dict:
        url = f"{self.base_url}/_plugins/_ml/memory_containers/{self.memory_container_id}/memories/sessions/_search"
        body = {
            "query": {
                "bool": {
                    "filter": [
                        {
                            "term": {
                                "namespace.session_id": session_id
                            }
                        }
                    ]
                }
            },
            "sort": [
                {
                    "message_id": {
                        "order": "asc"
                    }
                }
            ],
            "size": 1
        }

        response = self._make_request("GET", url, json=body)
        messages: list[Dict[str, Any]] = []
        search_response = self._get_hits(response)
        if search_response:
            for doc in search_response:
                messages.append(self._parse_message_from_source(doc['_source']))
            return messages
        return None

    def list_message(self, session_id: str, agent_id: str, limit: Optional[int] = None, offset: int = 0) -> Dict:
        url = f"{self.base_url}/_plugins/_ml/memory_containers/{self.memory_container_id}/memories/working/_search"
        body = {
            "query": {
                "bool": {
                    "filter": [
                        {
                            "term": {
                                "namespace.session_id": session_id
                            }
                        },
                        {
                            "term": {
                                "namespace.agent_id": agent_id
                            }
                        },
                        {
                            "term": {
                                "namespace_size": 2
                            }
                        }
                    ]
                }
            },
            "sort": [
                {
                    "message_id": {
                        "order": "asc"
                    }
                }
            ]
        }

        if limit:
            body['size'] = limit
        if offset:
            body['from'] = offset

        response = self._make_request("GET", url, json=body)
        messages: list[Dict[str, Any]] = []
        search_response = self._get_hits(response)
        if search_response:
            for doc in search_response:
                messages.append(self._parse_message_from_source(doc['_source']))
            return messages
        return None

    def get_message(self, session_id: str, agent_id:str, message_id: int) -> Dict:
        url = f"{self.base_url}/_plugins/_ml/memory_containers/{self.memory_container_id}/memories/working/_search"
        body = {
            "query": {
                "bool": {
                    "filter": [
                        {
                            "term": {
                                "namespace.session_id": session_id
                            }
                        },
                        {
                            "term": {
                                "namespace.agent_id": agent_id
                            }
                        },
                        {
                            "term": {
                                "namespace_size": 2
                            }
                        },
                        {
                            "term": {
                                "message_id": message_id
                            }
                        }
                    ]
                }
            },
            "sort": [
                {
                    "created_time": {
                        "order": "desc"
                    }
                }
            ]
        }

        response = self._make_request("GET", url, json=body)
        result = self._parse_message_from_source(response)
        return result

    def update_message(self, session_id: str, message_id: int, new_message:Dict[str, Any]) -> Dict:
        message_doc = self.get_message(session_id, message_id)

        if message_doc is None:
            return None


        message_doc_id = message_doc['_id']
        message_source = message_doc['_source']

        created_at = message_source['metadata']['created_at']
        new_message['created_at'] = created_at

        if "message" in new_message:
            message_source["messages"] = [
                new_message['message']
            ]
            new_message.pop('message', None)
        if new_message:
            message_source['metadata'] = new_message

        url = f"{self.base_url}/_plugins/_ml/memory_containers/{self.memory_container_id}/memories/working/{message_doc_id}"

        response = self._make_request("PUT", url, json=message_source)
        return response

    def _make_request(self, method: str, url: str, **kwargs) -> Dict:
        """Make HTTP request with error handling"""
        try:
            response = requests.request(
                method=method,
                url=url,
                auth=self.auth,
                headers=self.headers,
                verify=self.verify,
                **kwargs
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            if '404' in str(e):
                return None
            error_details = ""
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_details = f" - Response: {e.response.text}"
                except:
                    pass
            raise Exception(f"API request failed: {str(e)}{error_details}")

    def _parse_message_from_source(self, response: Dict[str, Any]) -> Dict[str, Any]:
        result = {
            "message": response['messages'][0],
            "message_id": response['message_id']
        }
        result.update(response['metadata'])
        return result

    def _get_hits(self, response: Dict[str, Any]) -> Optional[list[Dict[str, Any]]]:
        try:
            # Check if response exists and is a dict
            if not response or not isinstance(response, dict):
                return None

            # Navigate through the nested structure safely
            hits = response.get('hits', {})
            if not hits or not isinstance(hits, dict):
                return None

            hits_array = hits.get('hits', [])
            if not hits_array or not isinstance(hits_array, list):
                return None

            return hits_array

        except Exception as e:
            print(f"Error occurred while extracting _source: {str(e)}")
            return None

    def _get_first_hit(self, response: Dict[str, Any]) -> Optional[str]:
        try:
            # Check if response exists and is a dict
            if not response or not isinstance(response, dict):
                return None

            # Navigate through the nested structure safely
            hits = response.get('hits', {})
            if not hits or not isinstance(hits, dict):
                return None

            hits_array = hits.get('hits', [])
            if not hits_array or not isinstance(hits_array, list):
                return None

            # Get the first hit
            first_hit = hits_array[0] if hits_array else None
            if not first_hit or not isinstance(first_hit, dict):
                return None

            # Return the _source
            return first_hit #.get('_source')

        except Exception as e:
            print(f"Error occurred while extracting _source: {str(e)}")
            return None
    
    def _create_embedding_model(self) -> str:
        """Create Amazon Bedrock Titan embedding model for long-term memory"""
        url = f"{self.base_url}/_plugins/_ml/models/_register"
        aws_region = os.getenv('AWS_REGION') or 'us-east-1'
        aws_access_key = os.getenv('AWS_ACCESS_KEY_ID')
        aws_secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
        aws_session_token = os.getenv('AWS_SESSION_TOKEN')

        body = {
            "name": "Bedrock embedding model",
            "function_name": "remote",
            "description": "test model",
            "connector": {
                "name": "Amazon Bedrock Connector: embedding",
                "description": "The connector to bedrock Titan embedding model",
                "version": 1,
                "protocol": "aws_sigv4",
                "parameters": {
                    "region": aws_region,
                    "service_name": "bedrock",
                    "model": "amazon.titan-embed-text-v2:0",
                    "dimensions": 1024,
                    "normalize": True,
                    "embeddingTypes": [
                        "float"
                    ]
                },
                "credential": {
                    "access_key": aws_access_key,
                    "secret_key": aws_secret_key,
                    "session_token": aws_session_token
                },
                "actions": [
                    {
                        "action_type": "predict",
                        "method": "POST",
                        "url": "https://bedrock-runtime.${parameters.region}.amazonaws.com/model/${parameters.model}/invoke",
                        "headers": {
                            "content-type": "application/json",
                            "x-amz-content-sha256": "required"
                        },
                        "request_body": "{ \"inputText\": \"${parameters.inputText}\", \"dimensions\": ${parameters.dimensions}, \"normalize\": ${parameters.normalize}, \"embeddingTypes\": ${parameters.embeddingTypes} }",
                        "pre_process_function": "connector.pre_process.bedrock.embedding",
                        "post_process_function": "connector.post_process.bedrock.embedding"
                    }
                ]
            }
        }

        response = self._make_request("POST", url, json=body)
        self.embedding_model_id = response['model_id']
        print("Created embedding model with id '{}'".format(self.embedding_model_id))
        return self.embedding_model_id
        
    def _create_llm_model(self) -> str:
        """Create Amazon Bedrock LLM model for long-term memory"""
        aws_region = os.getenv('AWS_REGION') or 'us-east-1'
        aws_access_key = os.getenv('AWS_ACCESS_KEY_ID')
        aws_secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
        aws_session_token = os.getenv('AWS_SESSION_TOKEN')
        
        url = f"{self.base_url}/_plugins/_ml/models/_register"
        body = {
        "name": "Bedrock infer model",
        "function_name": "remote",
        "description": "LLM model for memory processing",
        "connector": {
                "name": "Amazon Bedrock Connector: LLM",
                "description": "The connector to bedrock Claude Opus 4.5 model",
                "version": 1,
                "protocol": "aws_sigv4",
                "parameters": {
                    "region": aws_region,
                    "service_name": "bedrock",
                    "max_tokens": 8000,
                    "temperature": 1,
                    "anthropic_version": "bedrock-2023-05-31",
                    "model": "global.anthropic.claude-opus-4-5-20251101-v1:0"
                },
                "credential": {
                    "access_key": aws_access_key,
                    "secret_key": aws_secret_key,
                    "session_token": aws_session_token
                },
                "actions": [{
                    "action_type": "predict",
                    "method": "POST",
                    "headers": {"content-type": "application/json"},
                    "url": "https://bedrock-runtime.${parameters.region}.amazonaws.com/model/${parameters.model}/converse",
                    "request_body": "{  \"anthropic_version\": \"${parameters.anthropic_version}\", \"max_tokens\": ${parameters.max_tokens}, \"temperature\": ${parameters.temperature}, \"system\": [{\"text\": \"${parameters.system_prompt}\"}], \"messages\": [ { \"role\": \"user\", \"content\": [ {\"text\": \"${parameters.user_prompt}\" }] }]}"
                }]
            }
        }
        
        model_response = self._make_request("POST", url, json=body)
        self.llm_id = model_response['model_id']
        print(f"Created LLM model with id '{self.llm_id}'")
        return self.llm_id

    def search_long_term_memories(self, query: str, user_id: str) -> list[Dict[str, Any]]:
        """Search long-term memories using semantic search"""
        url = f"{self.base_url}/_plugins/_ml/memory_containers/{self.memory_container_id}/memories/long-term/_search"
        body = {
            "query": {
                "bool": {
                    "filter": [
                        {
                            "term": {
                                "namespace.user_id": user_id
                            }
                        }
                    ]
                }
            },
            "sort": [
                {
                    "created_time": {
                        "order": "desc"
                    }
                }
            ],
            "size": 10
        }
        
        response = self._make_request("GET", url, json=body)
        return self._get_hits(response) or []

    def get_long_term_memory(self, memory_id: str) -> Dict[str, Any]:
        """Get a specific long-term memory by ID"""
        url = f"{self.base_url}/_plugins/_ml/memory_containers/{self.memory_container_id}/memories/long-term/{memory_id}"
        return self._make_request("GET", url)

    def delete_long_term_memory(self, memory_id: str) -> Dict[str, Any]:
        """Delete a specific long-term memory by ID"""
        url = f"{self.base_url}/_plugins/_ml/memory_containers/{self.memory_container_id}/memories/long-term/{memory_id}"
        return self._make_request("DELETE", url)