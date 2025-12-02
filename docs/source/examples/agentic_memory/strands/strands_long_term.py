import os
import sys
import urllib3
from dotenv import load_dotenv
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv('.env.strands-long')

# Suppress SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

from strands import Agent
from commons.opensearch_memory_tool import OpenSearchMemoryToolProvider

# Opensearch cluster configuration
cluster_url = os.getenv('OPENSEARCH_URL')
username = os.getenv('OPENSEARCH_USERNAME')
password = os.getenv('OPENSEARCH_PASSWORD')
verify_ssl = os.getenv("OPENSEARCH_VERIFY_SSL", "false").lower() == "true"

# Model configuration
embedding_model_id = os.getenv('EMBEDDING_MODEL_ID')
llm_id = os.getenv('LLM_MODEL_ID')

# Memory and session configuration
memory_container_name = os.getenv('MEMORY_CONTAINER_NAME') or 'strands_long_term'
session_id = os.getenv('SESSION_ID') or 'strands_long_term_session'
user_id = os.getenv('USER_ID') or 'strands_user'
agent_id = os.getenv('AGENT_ID') or 'strands_agent'

# Initialize the tool provider
provider = OpenSearchMemoryToolProvider(
    cluster_url=cluster_url,
    username=username,
    password=password,
    verify_ssl=verify_ssl,
    memory_container_name=memory_container_name,
    session_id=session_id,
    agent_id=agent_id,
    user_id=user_id,
    embedding_model_id=embedding_model_id,
    llm_id=llm_id
)

agent = Agent(tools=provider.tools)

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