"""OpenSearch checkpoint saver implementation using Agentic Memory API."""

from __future__ import annotations

import base64
import json
import requests
from collections.abc import AsyncIterator, Iterator, Sequence
from typing import Any, cast
from urllib.parse import urljoin

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    SerializerProtocol,
    get_checkpoint_id,
    get_checkpoint_metadata,
)
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

__all__ = ["OpenSearchSaver"]


class OpenSearchSaver(BaseCheckpointSaver[str]):
    """Checkpoint saver using OpenSearch Agentic Memory API.

    This implementation maps LangGraph concepts to OpenSearch Agentic Memory:
    - Memory Container: One per application (created once, reused)
    - Session: Maps to LangGraph thread_id (one per conversation)
    - Working Memory: Maps to LangGraph checkpoint (one per step)

    Uses payload_type="data" with metadata to distinguish checkpoint types:
    - metadata.type="checkpoint" for state snapshots
    - metadata.type="write" for intermediate writes

    Args:
        base_url: OpenSearch base URL (e.g., "http://localhost:9200")
        memory_container_id: ID of the memory container to use
        auth: Optional tuple of (username, password) for authentication
        verify_ssl: Whether to verify SSL certificates (default: True)
        headers: Optional additional headers for requests
        serde: Optional serializer for checkpoints

    Example:
        >>> # Create container first (one-time setup)
        >>> container_id = OpenSearchSaver.create_memory_container(
        ...     base_url="http://localhost:9200",
        ...     name="langgraph_checkpoints",
        ...     description="LangGraph checkpoint storage",
        ...     auth=("admin", "admin")
        ... )
        >>> # Use the saver
        >>> checkpointer = OpenSearchSaver(
        ...     base_url="http://localhost:9200",
        ...     memory_container_id=container_id,
        ...     auth=("admin", "admin")
        ... )
        >>> graph = builder.compile(checkpointer=checkpointer)
    """

    base_url: str
    memory_container_id: str
    auth: tuple[str, str] | None
    verify_ssl: bool
    headers: dict[str, str]

    def __init__(
            self,
            base_url: str,
            memory_container_id: str,
            *,
            auth: tuple[str, str] | None = None,
            verify_ssl: bool = True,
            headers: dict[str, str] | None = None,
            serde: SerializerProtocol | None = None,
    ) -> None:
        super().__init__(serde=serde)
        self.base_url = base_url.rstrip("/")
        self.memory_container_id = memory_container_id
        self.auth = auth
        self.verify_ssl = verify_ssl
        self.headers = headers or {}
        self.jsonplus_serde = JsonPlusSerializer()

        # Create a session for reusing connections
        self.session = requests.Session()
        if auth:
            self.session.auth = auth
        self.session.headers.update({"Content-Type": "application/json"})
        self.session.headers.update(self.headers)
        self.session.verify = verify_ssl

    @classmethod
    def create_memory_container(
            cls,
            base_url: str,
            name: str,
            description: str = "",
            configuration: dict[str, Any] | None = None,
            *,
            auth: tuple[str, str] | None = None,
            verify_ssl: bool = True,
    ) -> str:
        """Create a memory container for storing checkpoints.

        This should be called once during application setup.

        Args:
            base_url: OpenSearch base URL
            name: Name for the memory container
            description: Description of the container
            configuration: Optional container configuration
            auth: Optional (username, password) tuple
            verify_ssl: Whether to verify SSL certificates

        Returns:
            str: The memory_container_id to use with OpenSearchSaver
        """
        url = urljoin(base_url, "/_plugins/_ml/memory_containers/_create")
        response = requests.post(
            url,
            json={
                "name": name,
                "description": description,
                "configuration": configuration or {},
            },
            auth=auth,
            verify=verify_ssl,
            headers={"Content-Type": "application/json"},
        )
        response.raise_for_status()
        return response.json()["memory_container_id"]

    def _ensure_session(self, thread_id: str) -> None:
        """Ensure a session exists for the given thread_id."""
        # Check if session exists by trying to get it
        url = urljoin(
            self.base_url,
            f"/_plugins/_ml/memory_containers/{self.memory_container_id}/memories/sessions/{thread_id}"
        )
        try:
            response = self.session.get(url)
            response.raise_for_status()
        except requests.exceptions.RequestException:
            # Session doesn't exist, create it
            url = urljoin(
                self.base_url,
                f"/_plugins/_ml/memory_containers/{self.memory_container_id}/memories/sessions"
            )
            response = self.session.post(
                url,
                json={
                    "session_id": thread_id,
                    "metadata": {"created_by": "langgraph"},
                }
            )
            response.raise_for_status()

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        """Get a checkpoint tuple from OpenSearch.

        Args:
            config: Configuration containing thread_id and optionally checkpoint_id

        Returns:
            CheckpointTuple if found, None otherwise
        """
        thread_id = str(config["configurable"]["thread_id"])
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = get_checkpoint_id(config)

        # Build query to find the checkpoint
        # Using payload_type="data" with metadata.type="checkpoint"
        query: dict[str, Any] = {
            "query": {
                "bool": {
                  "filter": [
                    {
                        "term": {
                            "memory_container_id": self.memory_container_id,
                        }
                    },
                    {
                      "term": {
                        "namespace.thread_id": thread_id
                      }
                    },
                    {
                        "term": {
                            "namespace.checkpoint_ns": checkpoint_ns
                        }
                    },
                    {
                      "term": {
                        "payload_type": "data"
                      }
                    },
                    {
                      "term": {
                        "metadata.type": "checkpoint"
                      }
                    }
                  ]
                }
              },
              "sort": [
                {
                  "checkpoint_id": {
                    "order": "desc"
                  }
                }
              ],
              "size": 1
        }

        if checkpoint_id:
            # Get specific checkpoint
            query["query"]["bool"]["filter"].append(
                {"term": {"checkpoint_id": checkpoint_id}}
            )

        # Search in memories
        url = urljoin(
            self.base_url,
            f"/_plugins/_ml/memory_containers/{self.memory_container_id}/memories/working/_search"
        )

        try:
            response = self.session.post(url, json=query)
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.RequestException as e:
            print(f"⚠️  Failed to retrieve checkpoint: {e}")
            return None

        hits = data.get("hits", {}).get("hits", [])
        if not hits:
            print(f"⚠️  No checkpoint found for thread_id={thread_id}, checkpoint_ns={checkpoint_ns}, checkpoint_id={checkpoint_id}")
            return None

        doc = hits[0]["_source"]

        # Extract checkpoint data from binary_data
        binary_data = doc["binary_data"]
        parent_checkpoint_id = doc["namespace"].get("parent_checkpoint_id")

        # Decode base64 and parse JSON
        decoded_json = base64.b64decode(binary_data).decode('utf-8')
        data = json.loads(decoded_json)

        # Extract checkpoint and metadata
        checkpoint_b64 = data["checkpoint"]
        checkpoint_type = data["checkpoint_type"]
        metadata_b64 = data["metadata"]

        # Decode base64 and deserialize
        checkpoint_bytes = base64.b64decode(checkpoint_b64)
        checkpoint = self.serde.loads_typed((checkpoint_type, checkpoint_bytes))

        metadata_bytes = base64.b64decode(metadata_b64)
        if metadata_bytes:
            try:
                # Try direct JSON decode first
                decoded_metadata = json.loads(metadata_bytes.decode('utf-8'))
                # Ensure required fields exist
                if 'step' not in decoded_metadata:
                    decoded_metadata['step'] = 0
                if 'source' not in decoded_metadata:
                    decoded_metadata['source'] = 'unknown'
                metadata = cast(CheckpointMetadata, decoded_metadata)
            except Exception:
                # Fallback to default metadata with required fields
                metadata = cast(CheckpointMetadata, {'step': 0, 'source': 'unknown'})
        else:
            metadata = cast(CheckpointMetadata, {'step': 0, 'source': 'unknown'})

        # Build config for this checkpoint
        checkpoint_config = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": doc["namespace"]["checkpoint_id"],
            }
        }

        # Get pending writes for this checkpoint
        writes_query = {
          "query": {
            "bool": {
              "filter": [
                {
                  "term": {
                    "memory_container_id": self.memory_container_id,
                  }
                },
                {
                  "term": {
                    "namespace.thread_id": thread_id
                  }
                },
                {
                  "term": {
                    "payload_type": "data"
                  }
                },
                {
                  "term": {
                    "namespace.checkpoint_id": doc["namespace"]["checkpoint_id"]
                  }
                },
                {
                  "term": {
                    "metadata.type": "write"
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
          "size": 1000
        }

        writes_url = urljoin(
            self.base_url,
            f"/_plugins/_ml/memory_containers/{self.memory_container_id}/memories/working/_search"
        )
        try:
            writes_response = self.session.post(writes_url, json=writes_query)
            writes_response.raise_for_status()
            writes_data = writes_response.json()
            writes_hits = writes_data.get("hits", {}).get("hits", [])
            pending_writes = []
            for w in writes_hits:
                binary_data = w["_source"]["binary_data"]

                # Decode base64 and parse JSON
                decoded_json = base64.b64decode(binary_data).decode('utf-8')
                data = json.loads(decoded_json)

                # Extract write data
                channel = data["channel"]
                value_b64 = data["value"]
                value_type = data["value_type"]

                # Decode base64 and deserialize
                value_bytes = base64.b64decode(value_b64)
                deserialized_value = self.serde.loads_typed((value_type, value_bytes))

                pending_writes.append((
                    w["_source"]["namespace"]["task_id"],
                    channel,
                    deserialized_value,
                ))
        except Exception as e:
            print(f"⚠️  Failed to get write checkpoint for thread_id={thread_id}: {e}")
            pending_writes = []

        return CheckpointTuple(
            config=checkpoint_config,
            checkpoint=checkpoint,
            metadata=metadata,
            parent_config=(
                {
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_ns": checkpoint_ns,
                        "checkpoint_id": parent_checkpoint_id,
                    }
                }
                if parent_checkpoint_id
                else None
            ),
            pending_writes=pending_writes,
        )

    def list(
            self,
            config: RunnableConfig | None,
            *,
            filter: dict[str, Any] | None = None,
            before: RunnableConfig | None = None,
            limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        """List checkpoints from OpenSearch.

        Args:
            config: Base configuration with thread_id
            filter: Additional metadata filters
            before: List checkpoints before this checkpoint
            limit: Maximum number of checkpoints to return

        Yields:
            CheckpointTuple instances
        """
        if not config:
            return

        thread_id = str(config["configurable"]["thread_id"])
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")

        # Build query - using payload_type="data" with metadata.type="checkpoint"
        must_clauses = [
            {"term": {"namespace.thread_id": thread_id}},
            {"term": {"payload_type": "data"}},
            {"term": {"metadata.type": "checkpoint"}},
        ]

        # Add before filter
        if before:
            before_checkpoint_id = before["configurable"].get("checkpoint_id")
            if before_checkpoint_id:
                must_clauses.append(
                    {"range": {"checkpoint_id": {"lt": before_checkpoint_id}}}
                )

        # Add metadata filters
        if filter:
            for key, value in filter.items():
                must_clauses.append(
                    {"term": {f"metadata.{key}": value}}
                )

        query = {
            "query": {"bool": {"must": must_clauses}},
            "sort": [
                {"checkpoint_id": {"order": "desc"}}
            ],
            "size": limit or 100,
        }

        url = urljoin(
            self.base_url,
            f"/_plugins/_ml/memory_containers/{self.memory_container_id}/memories/working/_search"
        )

        try:
            response = self.session.post(url, json=query)
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.RequestException:
            return

        for hit in data.get("hits", {}).get("hits", []):
            doc = hit["_source"]
            binary_data = doc["binary_data"]
            parent_checkpoint_id = doc["namespace"].get("parent_checkpoint_id")

            # Decode base64 and parse JSON
            decoded_json = base64.b64decode(binary_data).decode('utf-8')
            checkpoint_data = json.loads(decoded_json)

            # Extract checkpoint and metadata
            checkpoint_b64 = checkpoint_data["checkpoint"]
            checkpoint_type = checkpoint_data["checkpoint_type"]
            metadata_b64 = checkpoint_data["metadata"]

            # Decode base64 and deserialize (same as SqliteSaver)
            checkpoint_bytes = base64.b64decode(checkpoint_b64)
            checkpoint = self.serde.loads_typed((checkpoint_type, checkpoint_bytes))

            metadata_bytes = base64.b64decode(metadata_b64)
            metadata = cast(
                CheckpointMetadata,
                self.jsonplus_serde.loads_typed(("json", metadata_bytes))[1] if metadata_bytes else {}
            )

            checkpoint_config = {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": doc["namespace"]["checkpoint_id"],
                }
            }

            # Get writes for this checkpoint
            writes_query = {
                "query": {
                    "bool": {
                        "must": [
                            {"term": {"namespace.thread_id.keyword": thread_id}},
                            {"term": {"namespace.checkpoint_ns.keyword": checkpoint_ns}},
                            {
                                "term": {
                                    "namespace.checkpoint_id.keyword": doc["namespace"][
                                        "checkpoint_id"
                                    ]
                                }
                            },
                            {"term": {"payload_type": "data"}},
                            {"term": {"metadata.type": "write"}},
                        ]
                    }
                },
                "sort": [
                    {"message_id": {"order": "asc"}},
                ],
                "size": 1000,
            }

            try:
                writes_response = self.session.post(url, json=writes_query)
                writes_response.raise_for_status()
                writes_data = writes_response.json()
                writes_hits = writes_data.get("hits", {}).get("hits", [])
                pending_writes = []
                for w in writes_hits:
                    binary_data = w["_source"]["binary_data"]

                    # Decode base64 and parse JSON
                    decoded_json = base64.b64decode(binary_data).decode('utf-8')
                    write_data = json.loads(decoded_json)

                    # Extract write data
                    channel = write_data["channel"]
                    value_b64 = write_data["value"]
                    value_type = write_data["value_type"]

                    # Decode base64 and deserialize (same as SqliteSaver)
                    value_bytes = base64.b64decode(value_b64)
                    deserialized_value = self.serde.loads_typed((value_type, value_bytes))

                    pending_writes.append((
                        w["_source"]["namespace"]["task_id"],
                        channel,
                        deserialized_value,
                    ))
            except requests.exceptions.RequestException:
                pending_writes = []

            yield CheckpointTuple(
                config=checkpoint_config,
                checkpoint=checkpoint,
                metadata=metadata,
                parent_config=(
                    {
                        "configurable": {
                            "thread_id": thread_id,
                            "checkpoint_ns": checkpoint_ns,
                            "checkpoint_id": parent_checkpoint_id,
                        }
                    }
                    if parent_checkpoint_id
                    else None
                ),
                pending_writes=pending_writes,
            )

    def put(
            self,
            config: RunnableConfig,
            checkpoint: Checkpoint,
            metadata: CheckpointMetadata,
            new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Save a checkpoint to OpenSearch.

        Args:
            config: Configuration with thread_id
            checkpoint: The checkpoint to save
            metadata: Checkpoint metadata
            new_versions: New channel versions

        Returns:
            Updated configuration with checkpoint_id
        """
        thread_id = str(config["configurable"]["thread_id"])
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        parent_checkpoint_id = config["configurable"].get("checkpoint_id")

        # Ensure session exists
        self._ensure_session(thread_id)

        # Serialize checkpoint and metadata (same as SqliteSaver approach)
        type_, serialized_checkpoint = self.serde.dumps_typed(checkpoint)
        metadata_type, serialized_metadata = self.jsonplus_serde.dumps_typed(
            get_checkpoint_metadata(config, metadata)
        )

        messages = []
        # Debug: Print what's being saved
        if "channel_values" in checkpoint and "messages" in checkpoint["channel_values"]:
            for msg in checkpoint["channel_values"]["messages"]:
                messages.append({"role": msg.type, "content": msg.content})

        # Convert bytes to base64 string for JSON transport
        checkpoint_b64 = base64.b64encode(serialized_checkpoint).decode('utf-8')
        metadata_b64 = base64.b64encode(serialized_metadata).decode('utf-8')

        # Create JSON structure and encode to binary_data
        data = {
            "checkpoint": checkpoint_b64,
            "checkpoint_type": type_,
            "metadata": metadata_b64,
            "messages": messages,
        }
        encoded_json = json.dumps(data)
        binary_data_b64 = base64.b64encode(encoded_json.encode('utf-8')).decode('utf-8')

        # Create working memory document with payload_type="data"
        # Use metadata.type="checkpoint" to distinguish from writes
        # checkpoint_id contains UUID v6 which is lexicographically sortable by timestamp
        memory_doc = {
            "payload_type": "data",
            "checkpoint_id": checkpoint["id"],  # UUID v6 with embedded timestamp
            "binary_data": binary_data_b64,
            "namespace": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint["id"],
            },
            "metadata": {
                "type": "checkpoint",
                "source": metadata.get("source", "unknown"),
                "step": metadata.get("step", -1),
            },
            "tags": {
                "source": metadata.get("source", "unknown"),
                "step": str(metadata.get("step", -1)),
            },
        }

        # Add parent checkpoint ID if exists
        if parent_checkpoint_id:
            memory_doc["namespace"]["parent_checkpoint_id"] = parent_checkpoint_id

        # Store in OpenSearch
        url = urljoin(
            self.base_url,
            f"/_plugins/_ml/memory_containers/{self.memory_container_id}/memories"
        )
        try:
            response = self.session.post(url, json=memory_doc)
            response.raise_for_status()
        except Exception as e:
            print(f"❌ Failed to save checkpoint: {e}")
            # Print response details if available
            if hasattr(e, 'response') and e.response is not None:
                print(f"   Response status: {e.response.status_code}")
                print(f"   Response body: {e.response.text[:500]}")

        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint["id"],
            }
        }

    def put_writes(
            self,
            config: RunnableConfig,
            writes: Sequence[tuple[str, Any]],
            task_id: str,
            task_path: str = "",
    ) -> None:
        """Store intermediate writes linked to a checkpoint.

        Args:
            config: Configuration with thread_id and checkpoint_id
            writes: List of writes as (channel, value) pairs
            task_id: Task identifier
            task_path: Task path
        """
        thread_id = str(config["configurable"]["thread_id"])
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = str(config["configurable"]["checkpoint_id"])

        # Ensure session exists
        self._ensure_session(thread_id)

        url = urljoin(
            self.base_url,
            f"/_plugins/_ml/memory_containers/{self.memory_container_id}/memories"
        )

        # Store each write as a separate memory document
        # Use payload_type="data" with metadata.type="write"
        for idx, (channel, value) in enumerate(writes):
            # Serialize (same as SqliteSaver)
            type_, serialized_value = self.serde.dumps_typed(value)

            # Convert bytes to base64 string
            value_b64 = base64.b64encode(serialized_value).decode('utf-8')

            # Create JSON structure and encode to binary_data
            data = {
                "channel": channel,
                "value": value_b64,
                "value_type": type_,
            }
            encoded_json = json.dumps(data)
            binary_data_b64 = base64.b64encode(encoded_json.encode('utf-8')).decode('utf-8')

            # Use WRITES_IDX_MAP for special write types (errors, interrupts, etc.)
            write_idx = WRITES_IDX_MAP.get(channel, idx)

            write_doc = {
                "payload_type": "data",
                "checkpoint_id": checkpoint_id,  # UUID for identification
                "binary_data": binary_data_b64,
                "namespace": {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": checkpoint_id,
                    "task_id": task_id,
                    "idx": write_idx,
                    "channel": channel,
                },
                "message_id": write_idx,
                "metadata": {
                    "type": "write",
                    "channel": channel,
                    "task_id": task_id,
                },
                "tags": {"task_path": task_path} if task_path else {},
            }

            try:
                response = self.session.post(url, json=write_doc)
                response.raise_for_status()
            except Exception as e:
                print(f"An unexpected error occurred: {e}")

    def delete_thread(self, thread_id: str) -> None:
        """Delete all checkpoints and writes for a thread.

        Args:
            thread_id: The thread ID to delete
        """
        # Delete all memories for this thread using delete by query
        query = {
            "query": {
                "term": {"namespace.thread_id.keyword": str(thread_id)}
            }
        }

        url = urljoin(
            self.base_url,
            f"/_plugins/_ml/memory_containers/{self.memory_container_id}/memories/_delete_by_query"
        )

        try:
            response = self.session.post(url, json=query)
            response.raise_for_status()
        except requests.exceptions.RequestException:
            # If delete by query is not supported, search and delete individually
            try:
                search_url = urljoin(
                    self.base_url,
                    f"/_plugins/_ml/memory_containers/{self.memory_container_id}/memories/working/_search"
                )
                search_response = self.session.post(
                    search_url,
                    json={**query, "size": 10000}
                )
                search_response.raise_for_status()
                search_data = search_response.json()

                for hit in search_data.get("hits", {}).get("hits", []):
                    memory_id = hit["_id"]
                    delete_url = urljoin(
                        self.base_url,
                        f"/_plugins/_ml/memory_containers/{self.memory_container_id}/memories/{memory_id}"
                    )
                    delete_response = self.session.delete(delete_url)
                    delete_response.raise_for_status()
            except requests.exceptions.RequestException:
                pass

    def get_next_version(self, current: str | None, channel: None) -> str:
        """Generate the next version ID for a channel.

        Args:
            current: Current version identifier
            channel: Deprecated, kept for compatibility

        Returns:
            Next version identifier
        """
        import random

        if current is None:
            current_v = 0
        elif isinstance(current, int):
            current_v = current
        else:
            current_v = int(current.split(".")[0])
        next_v = current_v + 1
        next_h = random.random()
        return f"{next_v:032}.{next_h:016}"

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        """Async version not implemented. Use get_tuple() instead."""
        raise NotImplementedError(
            "Async methods are not supported by OpenSearchSaver. "
            "Use the synchronous methods instead."
        )

    async def alist(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[CheckpointTuple]:
        """Async version not implemented. Use list() instead."""
        raise NotImplementedError(
            "Async methods are not supported by OpenSearchSaver. "
            "Use the synchronous methods instead."
        )
        yield

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Async version not implemented. Use put() instead."""
        raise NotImplementedError(
            "Async methods are not supported by OpenSearchSaver. "
            "Use the synchronous methods instead."
        )

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Async version not implemented. Use put_writes() instead."""
        raise NotImplementedError(
            "Async methods are not supported by OpenSearchSaver. "
            "Use the synchronous methods instead."
        )

    async def adelete_thread(self, thread_id: str) -> None:
        """Async version not implemented. Use delete_thread() instead."""
        raise NotImplementedError(
            "Async methods are not supported by OpenSearchSaver. "
            "Use the synchronous methods instead."
        )