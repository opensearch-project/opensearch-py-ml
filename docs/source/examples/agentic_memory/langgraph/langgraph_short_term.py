import os
from datetime import datetime
from langchain_aws import ChatBedrock
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from opensearch_checkpoint_saver import OpenSearchSaver

# Suppress SSL warnings
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '.env.langgraph-short'))

# Opensearch cluster configuration
cluster_url = os.getenv('OPENSEARCH_URL')
username = os.getenv('OPENSEARCH_USERNAME')
password = os.getenv('OPENSEARCH_PASSWORD')
verify_ssl = os.getenv('OPENSEARCH_VERIFY_SSL', 'false').lower() == 'true'

# Memory configuration
memory_container_name = os.getenv('MEMORY_CONTAINER_NAME') or 'langgraph_short_term'

# AWS Bedrock configuration
bedrock_model_id = os.getenv('BEDROCK_MODEL_ID') or 'global.anthropic.claude-opus-4-5-20251101-v1:0'
aws_region = os.getenv('AWS_REGION') or 'us-east-1'

def create_chatbot(checkpointer):
    """Create a minimal chatbot with checkpoint support."""
    
    model = ChatBedrock(
        model_id=bedrock_model_id,
        region_name=aws_region,
        model_kwargs={"temperature": 0.7, "max_tokens": 1024}
    )
    
    def chat_node(state: MessagesState):
        return {"messages": [model.invoke(state["messages"])]}
    
    graph = StateGraph(MessagesState)
    graph.add_node("chat", chat_node)
    graph.add_edge(START, "chat")
    graph.add_edge("chat", END)
    
    return graph.compile(checkpointer=checkpointer)


def send_message(app, thread_id: str, message: str):
    """Send a message and return the response."""
    config = {"configurable": {"thread_id": thread_id}}
    result = app.invoke({"messages": [HumanMessage(content=message)]}, config)
    return result["messages"][-1].content


def get_message_count(app, thread_id: str) -> int:
    """Get the number of messages in the conversation."""
    config = {"configurable": {"thread_id": thread_id}}
    state = app.get_state(config)
    if state and state.values:
        return len(state.values.get("messages", []))
    return 0

def setup_opensearch_checkpointer(container_name: str, index_prefix: str) -> OpenSearchSaver:
    """Setup OpenSearch checkpoint saver.

    Args:
        container_name: Name for the memory container

    Returns:
        Configured OpenSearchSaver instance
    """
    auth = (username, password)

    # Try to find existing container first
    container_id = find_existing_container(container_name)
    
    if container_id:
        print(f"✅ Find memory container with id '{container_id}' by name '{container_name}'")
    else:
        try:
            # Create new container
            container_id = OpenSearchSaver.create_memory_container(
                base_url=cluster_url,
                name=container_name,
                description="Checkpoint storage for Bedrock Claude chatbot",
                configuration={
                    "index_prefix": index_prefix,
                    "disable_history": False,
                    "disable_session": False,
                    "use_system_index": False,
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
                },
                auth=auth,
                verify_ssl=verify_ssl,
            )
            print(f"✅ Created memory container: {container_id}")
        except Exception as e:
            raise e

    # Create and return checkpointer
    checkpointer = OpenSearchSaver(
        base_url=cluster_url,
        memory_container_id=container_id,
        auth=auth,
        verify_ssl=verify_ssl,
    )

    return checkpointer

def find_existing_container(name: str) -> str:
    """Find existing memory container by name."""
    import requests
    
    url = f"{cluster_url}/_plugins/_ml/memory_containers/_search"
    body = {
        "query": {"bool": {"filter": [{"term": {"name.keyword": name}}]}},
        "sort": [{"created_time": {"order": "asc"}}],
        "size": 1
    }
    
    try:
        response = requests.get(url, json=body, auth=(username, password), verify=verify_ssl)
        response.raise_for_status()
        data = response.json()
        hits = data.get("hits", {}).get("hits", [])
        return hits[0]["_id"] if hits else None
    except Exception:
        return None

def find_existing_thread(container_id: str, user_prefix: str = "demo_") -> str:
    """Find the most recent thread ID for this user."""
    import requests
    
    auth = (username, password)
    url = f"{cluster_url}/_plugins/_ml/memory_containers/{container_id}/memories/working/_search"
    
    body = {
        "query": {
            "bool": {
                "filter": [
                    {"term": {"payload_type": "data"}},
                    {"term": {"metadata.type": "checkpoint"}},
                    {"wildcard": {"namespace.thread_id": f"{user_prefix}*"}}
                ]
            }
        },
        "sort": [{"checkpoint_id": {"order": "desc"}}],
        "size": 1
    }
    
    try:
        response = requests.post(url, json=body, auth=auth, verify=verify_ssl)
        response.raise_for_status()
        data = response.json()
        
        hits = data.get("hits", {}).get("hits", [])
        if hits:
            return hits[0]["_source"]["namespace"]["thread_id"]
        return None
    except Exception:
        return None


if __name__ == "__main__":
    """Create a new conversation session."""
    #1. Setup OpenSearch checkpointer
    print("\n1. Setting up OpenSearch checkpointer...")
    container_name = memory_container_name
    index_prefix = memory_container_name
    checkpointer = setup_opensearch_checkpointer(container_name, index_prefix)
    
    #2. Create chatbot
    print("\n2. Creating chatbot...")
    app = create_chatbot(checkpointer)
    print("✅ Chatbot ready\n")
    
    #3. Start interactive session
    existing_thread = find_existing_thread(checkpointer.memory_container_id)

    if existing_thread:
        print(f"Found existing thread: {existing_thread}")
        choice = input("Resume existing conversation? (y/n): ").strip().lower()
        if choice == 'y':
            thread_id = existing_thread
            print(f"Resuming thread: {thread_id}")
        else:
            thread_id = f"demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            print(f"Starting new thread: {thread_id}")
    else:
        thread_id = f"demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        print(f"Starting new thread: {thread_id}")

    print("LangGraph Interactive Demo")
    print("Type 'q' or 'quit' to end the conversation\n")
    
    while True:
        question = input("👤 You: ").strip()
        
        if question.lower() in ['q', 'quit']:
            print("🤖 Assistant: Goodbye!")
            
            # Show session summary
            msg_count = get_message_count(app, thread_id)
            print("\n" + "=" * 70)
            print(f"\n📊 Session Summary:")
            print(f"Thread ID: {thread_id}")
            print(f"Total messages: {msg_count}")
            print(f"Session ended: {datetime.now()}")
            break
        
        if not question:
            continue
        
        response = send_message(app, thread_id, question)
        print(f"🤖 Assistant: {response}")
        print()