import os
import urllib3
from dotenv import load_dotenv

# Load environment variables
load_dotenv('.env.strands-short')

# Suppress SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

from strands.session.repository_session_manager import RepositorySessionManager
from strands import Agent
from opensearch_session_manager import OpenSearchSessionRepository

# Opensearch cluster configuration
cluster_url = os.getenv('OPENSEARCH_URL')
username = os.getenv('OPENSEARCH_USERNAME')
password = os.getenv('OPENSEARCH_PASSWORD')
verify_ssl = os.getenv("OPENSEARCH_VERIFY_SSL", "false").lower() == "true"

# Memory and session configuration
memory_container_name = os.getenv('MEMORY_CONTAINER_NAME') or 'strands_short_term'
memory_container_description = os.getenv('MEMORY_CONTAINER_DESCRIPTION') or 'OpenSearch Strands demo memory container'
session_id = os.getenv('SESSION_ID') or 'strands_short_term_session'

repo = OpenSearchSessionRepository(cluster_url, username, password, verify_ssl,
                                   memory_container_name=memory_container_name,
                                   memory_container_description=memory_container_description)
session_manager = RepositorySessionManager(session_id=session_id, session_repository=repo)

agent = Agent(
    session_manager=session_manager,
    system_prompt="You are a helpful assistant.",
    # ... other args like tools, model, etc.
)

print(f"OpenSearch Agentic Memory Demo - Session: {session_id}")
print("Type 'q' or 'quit' to end the conversation\n")

while True:
    question = input("👤 You: ").strip()
    
    print(f"🤖 Assistant: ", end='') 
    
    if question.lower() in ['q', 'quit']:
        print("Goodbye!")
        break
    
    if not question:
        continue
    
    response = agent(question)
    print("\n")