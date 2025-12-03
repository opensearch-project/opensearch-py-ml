import json
import logging
import os
from enum import Enum
from typing import Dict, List, Optional

from .opensearch_agentic_memory import OpenSearchAgenticMemory
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from strands import tool
from strands.types.tools import AgentTool

# Set up logging
logger = logging.getLogger(__name__)

# Initialize Rich console
console = Console()


# Define memory actions as an Enum
class MemoryAction(str, Enum):
    """Enum for memory actions."""

    STORE = "store"
    GET = "get"
    SEARCH = "search"
    DELETE = "delete"


# Define required parameters for each action
REQUIRED_PARAMS = {
    MemoryAction.STORE: ["user_id", "content"],
    MemoryAction.SEARCH: ["user_id", "query"],
    MemoryAction.GET: ["memory_id"],
    MemoryAction.DELETE: ["memory_id"],
}


class OpenSearchMemoryToolProvider:
    """Provider for OpenSearch Agentic Memory tools."""

    def __init__(
        self,
        cluster_url: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        verify_ssl: bool = False,
        session_id: str = None,
        agent_id: str = None,
        user_id: str = None,
        memory_container_id: Optional[str] = None,
        memory_container_name: str = "strands_agent_memory",
        memory_container_description: str = "Strands agent memory container",
        embedding_model_id: Optional[str] = None,
        llm_id: Optional[str] = None,
        infer: bool = False,
    ):
        """
        Initialize the OpenSearch Memory tool provider.

        Args:
            cluster_url: OpenSearch cluster URL
            username: OpenSearch username
            password: OpenSearch password
            verify_ssl: Whether to verify SSL
            session_id: Default session ID to use for operations
            agent_id: Default agent ID to use for operations
            user_id: Default user ID to use for operations
            memory_container_id: Optional memory container ID
            memory_container_name: Name for the memory container
            memory_container_description: Description for the memory container
            embedding_model_id: Optional embedding model ID
            llm_id: Optional LLM model ID
            infer: Whether to enable inference

        Raises:
            ValueError: If required credentials are missing
        """
        self.cluster_url = cluster_url
        self.username = username
        self.password = password
        self.verify = verify_ssl
        self.memory_container_id = memory_container_id
        self.memory_container_name = memory_container_name
        self.session_id = session_id
        self.agent_id = agent_id
        self.user_id = user_id
        self.embedding_model_id = embedding_model_id
        self.llm_id = llm_id

        if not all([self.cluster_url, self.username, self.password]):
            raise ValueError(
                "OpenSearch credentials required. Set OPENSEARCH_CLUSTER_URL, "
                "OPENSEARCH_USERNAME, and OPENSEARCH_PASSWORD environment variables "
                "or provide them as parameters."
            )
        
        self.memory = OpenSearchAgenticMemory(
            cluster_url=self.cluster_url,
            username=self.username,
            password=self.password,
            verify_ssl=self.verify,
            memory_container_id=memory_container_id,
            memory_container_name=memory_container_name,
            memory_container_description=memory_container_description,
            embedding_model_id=self.embedding_model_id,
            llm_id=self.llm_id,
            infer=infer,
            long_term=True
        )

    @property
    def tools(self) -> list[AgentTool]:
        """Extract all @tool decorated methods from this instance."""
        tools = []

        for attr_name in dir(self):
            if attr_name == "tools":
                continue
            attr = getattr(self, attr_name)
            if isinstance(attr, AgentTool):
                tools.append(attr)

        return tools

    @tool
    def opensearch_memory(
        self,
        action: str,
        content: Optional[str] = None,
        query: Optional[str] = None,
        session_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        user_id: Optional[str] = None,
        memory_id: Optional[str] = None,
        metadata: Optional[Dict] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> Dict:
        """
        Work with OpenSearch memories - store, search, retrieve, and delete memory records.

        This tool helps agents store and access memories in OpenSearch, allowing them to remember
        important information across conversations and interactions.

        Key Capabilities:
        - Store new memories (text conversations or structured data)
        - Search for memories
        - Retrieve specific memory by ID
        - Delete specific memory by ID

        Supported Actions:
        -----------------
        Memory Management:
        - store: Store a new memory (conversation or data)
          Use this when you need to save information for later recall.

        - get: Fetch a specific memory by memory ID
          Use this when you already know the exact message ID.

        - search: search memories with a query
          Use this when searching for all memories

        - delete: Remove a specific memory by memory ID
          Use this to delete memory that are no longer needed.

        Args:
            action: The memory operation to perform (one of: "store", "get", "search", "delete")
            content: For store action: Simple text string to store as a memory
                     (e.g., "User prefers vegetarian pizza with extra cheese")
            query: For search action: Simple text string to search for
            session_id: Session ID (uses default from initialization if not provided)
            agent_id: Agent ID (uses default from initialization if not provided)
            user_id: User ID (uses default from initialization if not provided)
            memory_id: ID of a specific memory (required for get/delete action)
            metadata: Optional metadata to store with the memory (for store action)
            limit: Maximum number of results to return (optional, for search action)
            offset: Offset for pagination (optional, for search action)

        Returns:
            Dict: Response containing the requested memory information or operation status
        """
        try:
            # Use provided values or defaults from initialization
            session_id = session_id or self.session_id
            agent_id = agent_id or self.agent_id
            user_id = user_id or self.user_id

            # Try to convert string action to Enum
            try:
                action_enum = MemoryAction(action)
            except ValueError:
                return {
                    "status": "error",
                    "content": [
                        {
                            "text": f"Action '{action}' is not supported. "
                            f"Supported actions: {', '.join([a.value for a in MemoryAction])}"
                        }
                    ],
                }

            # Create a dictionary mapping parameter names to their values
            param_values = {
                "session_id": session_id,
                "agent_id": agent_id,
                "user_id": user_id,
                "content": content,
                "query": query,
                "memory_id": memory_id,
            }

            # Check which required parameters are missing
            missing_params = [param for param in REQUIRED_PARAMS[action_enum] if not param_values.get(param)]

            if missing_params:
                return {
                    "status": "error",
                    "content": [
                        {
                            "text": (
                                f"The following parameters are required for {action_enum.value} action: "
                                f"{', '.join(missing_params)}"
                            )
                        }
                    ],
                }

            # Check if we're in development mode
            strands_dev = os.environ.get("BYPASS_TOOL_CONSENT", "").lower() == "true"

            # For mutative operations, show confirmation dialog unless in BYPASS_TOOL_CONSENT mode
            mutative_actions = {MemoryAction.STORE, MemoryAction.DELETE}
            needs_confirmation = action_enum in mutative_actions and not strands_dev

            if needs_confirmation:
                if action_enum == MemoryAction.STORE:
                    # Preview what will be stored
                    content_preview = content[:15000] + "..." if len(content) > 15000 else content
                    preview_title = f"Add long-term memory for session {session_id}, agent {agent_id}"
                    console.print(
                        Panel(
                            content_preview,
                            title=f"[bold green]{preview_title}",
                            border_style="green",
                        )
                    )

                elif action_enum == MemoryAction.DELETE:
                    console.print(
                        Panel(
                            f"Long-term memory ID: {memory_id}",
                            title="[bold red]⚠️ Long term memory to be permanently deleted",
                            border_style="red",
                        )
                    )

            # Execute the appropriate action
            try:
                if action_enum == MemoryAction.STORE:
                    result = self._store_memory(session_id, agent_id, user_id, content, metadata)
                    panel = self._format_store_response(result)
                    console.print(panel)
                    return {
                        "status": "success",
                        "content": [{"text": f"Memory stored successfully: {json.dumps(result, default=str)}"}],
                    }

                elif action_enum == MemoryAction.GET:
                    memory = self._get_memory(memory_id)
                    panel = self._format_get_response(memory)
                    console.print(panel)
                    return {
                        "status": "success",
                        "content": [{"text": f"Message retrieved successfully: {json.dumps(memory, default=str)}"}],
                    }

                elif action_enum == MemoryAction.SEARCH:
                    memories = self._search_long_term_memories(query, user_id)
                    panel = self._format_search_response(memories or [])
                    console.print(panel)
                    return {
                        "status": "success",
                        "content": [
                            {"text": f"Memories searched successfully: {json.dumps(memories or [], default=str)}"}
                        ],
                    }

                elif action_enum == MemoryAction.DELETE:
                    self._delete_long_term_memory(memory_id)
                    panel = self._format_delete_response(session_id)
                    console.print(panel)
                    return {
                        "status": "success",
                        "content": [{"text": f"Session {session_id} deleted successfully"}],
                    }

            except Exception as e:
                error_msg = f"API error: {str(e)}"
                logger.error(error_msg)
                return {"status": "error", "content": [{"text": error_msg}]}

        except Exception as e:
            logger.error(f"Unexpected error in opensearch_memory tool: {str(e)}")
            error_panel = Panel(
                Text(str(e), style="red"),
                title="❌ Memory Operation Error",
                border_style="red",
            )
            console.print(error_panel)
            return {"status": "error", "content": [{"text": str(e)}]}

    def _store_memory(
        self,
        session_id: str,
        agent_id: str,
        user_id: str,
        content: str,
        metadata: Optional[Dict] = None,
    ) -> Dict:
        """Store a memory in OpenSearch."""
        message = {"message": {"role": "user", "content": [{"text": content}]}}
        if metadata:
            message.update(metadata)
        return self.memory.add_message(session_id, agent_id, message, infer=True, user_id=user_id)

    def _get_memory(self, memory_id: str) -> Dict:
        """Get a specific message by ID."""
        return self.memory.get_long_term_memory(memory_id)

    def _search_long_term_memories(self, question: str, user_id: str) -> List[Dict]:
        """Search for messages in a session."""
        return self.memory.search_long_term_memories(question, user_id=user_id)

    def _delete_long_term_memory(self, memory_id: str) -> Dict:
        """Delete a session and all its messages."""
        return self.memory.delete_long_term_memory(memory_id)

    def _format_store_response(self, result: Dict) -> Panel:
        """Format store memory response."""
        memory_id = result.get("working_memory_id", "unknown")
        content = ["✅ Memory stored successfully:", f"🔑 Memory ID: {memory_id}"]
        return Panel("\n".join(content), title="[bold green]Memory Stored", border_style="green")

    def _format_get_response(self, message: Dict) -> Panel:
        """Format get message response."""
        message_id = message.get("message_id", "unknown")
        content_data = message.get("message", {})

        # Extract text content
        text_content = "No content available"
        if isinstance(content_data, dict):
            content_list = content_data.get("content", [])
            if content_list and isinstance(content_list, list):
                text_content = content_list[0].get("text", "No content available")

        created_at = message.get("created_at", "Unknown")

        result = [
            "✅ Message retrieved successfully:",
            f"🔑 Message ID: {message_id}",
            f"🕒 Created: {created_at}",
            f"\n📄 Content: {text_content}",
        ]

        # Add metadata if present
        metadata_keys = [k for k in message.keys() if k not in ["message", "message_id", "created_at"]]
        if metadata_keys:
            metadata = {k: message[k] for k in metadata_keys}
            result.insert(3, f"📋 Metadata: {json.dumps(metadata, indent=2)}")

        return Panel("\n".join(result), title="[bold green]Message Retrieved", border_style="green")

    def _format_list_response(self, messages: List[Dict]) -> Panel:
        """Format list messages response."""
        if not messages:
            return Panel("No messages found.", title="[bold yellow]No Messages", border_style="yellow")

        table = Table(title="Messages", show_header=True, header_style="bold magenta")
        table.add_column("Message ID", style="cyan")
        table.add_column("Content", style="yellow", width=50)
        table.add_column("Created At", style="blue")

        for message in messages:
            message_id = str(message.get("message_id", "unknown"))
            content_data = message.get("message", {})

            # Extract text content
            text_content = "No content available"
            if isinstance(content_data, dict):
                content_list = content_data.get("content", [])
                if content_list and isinstance(content_list, list):
                    text_content = content_list[0].get("text", "No content available")

            created_at = message.get("created_at", "Unknown")

            # Truncate content if too long
            content_preview = text_content[:100] + "..." if len(text_content) > 100 else text_content

            table.add_row(message_id, content_preview, created_at)

        return Panel(table, title="[bold green]Messages List", border_style="green")

    def _format_search_response(self, memories: List[Dict]) -> Panel:
        """Format search session response."""
        if not memories:
            return Panel("No long term memories found in session.", title="[bold yellow]No Messages", border_style="yellow")

        table = Table(title="Long term memories", show_header=True, header_style="bold magenta")
        table.add_column("Memory ID", style="cyan", width=20)
        table.add_column("Content", style="yellow", width=50)

        for memory in memories:
            memory_id = str(memory.get("_id", "unknown"))
            content_data = memory.get("_source", {}).get("memory", "unknown")

            # Truncate content if too long
            content_preview = content_data[:100] + "..." if len(content_data) > 100 else content_data

            table.add_row(memory_id, content_preview)

        return Panel(table, title="[bold green]Search Results", border_style="green")

    def _format_delete_response(self, session_id: str) -> Panel:
        """Format delete session response."""
        content = ["✅ Session deleted successfully:", f"🔑 Session ID: {session_id}"]
        return Panel("\n".join(content), title="[bold green]Session Deleted", border_style="green")