import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Optional, Any
from strands.session.session_repository import SessionRepository
from strands.types.exceptions import SessionException
from strands.types.session import Session, SessionAgent, SessionMessage
from commons.opensearch_agentic_memory import OpenSearchAgenticMemory


class OpenSearchSessionRepository(SessionRepository):

    def __init__(self,
                 cluster_url: str,
                 username: str,
                 password: str,
                 verify_ssl: bool = False,
                 memory_container_id: str = None,
                 memory_container_name: str = "default",
                 memory_container_description: str = "Strands agent memory container"):
        self.osam = OpenSearchAgenticMemory(cluster_url, username, password, verify_ssl, memory_container_id, memory_container_name, memory_container_description)

    def create_session(self, session: Session, **kwargs: Any) -> Session:
        self.osam.create_session(session.session_id, session.to_dict(),)
        return session

    def read_session(self, session_id: str, **kwargs: Any) -> Optional[Session]:
        session_data = self.osam.get_session(session_id)
        if session_data is None:
            return None
        metadata = session_data.get('metadata')
        return Session.from_dict(metadata)

    def delete_session(self, session_id: str, **kwargs: Any) -> None:
        """Delete session and all associated data."""
        self.osam.delete_session(session_id)

    def create_agent(self, session_id: str, session_agent: SessionAgent, **kwargs: Any) -> None:
        self.osam.update_session(session_id, None, session_agent.to_dict())

    def read_agent(self, session_id: str, agent_id: str, **kwargs: Any) -> Optional[SessionAgent]:
        session_data = self.osam.get_session(session_id)
        agents = session_data.get('agents')
        if agents is None:
            return None
        return SessionAgent.from_dict(agents)

    def update_agent(self, session_id: str, session_agent: SessionAgent, **kwargs: Any) -> None:
        agent_id = session_agent.agent_id
        previous_agent = self.read_agent(session_id=session_id, agent_id=agent_id)
        if previous_agent is None:
            raise SessionException(f"Agent {agent_id} in session {session_id} does not exist")
        session_agent.created_at = previous_agent.created_at
        # update session with new agents data
        self.osam.update_session(session_id, None, session_agent.to_dict())

    def create_message(self, session_id: str, agent_id: str, session_message: SessionMessage, **kwargs: Any) -> None:
        self.osam.add_message(session_id, agent_id, session_message.to_dict())

    def read_message(self, session_id: str, agent_id: str, message_id: int, **kwargs: Any) -> Optional[SessionMessage]:
        message_data = self.osam.get_message(session_id, agent_id, message_id)
        return SessionMessage.from_dict(message_data)

    def update_message(self, session_id: str, agent_id: str, session_message: SessionMessage, **kwargs: Any) -> None:
        self.osam.update_message(session_id, agent_id, session_message.to_dict())

    def list_messages(self, session_id: str, agent_id: str, limit: Optional[int] = None, offset: int = 0,
                      **kwargs: Any) -> list[SessionMessage]:
        docs = self.osam.list_message(session_id, agent_id, limit, offset)
        messages: list[SessionMessage] = []
        if docs:
            for doc in docs:
                messages.append(SessionMessage.from_dict(doc))
        return messages