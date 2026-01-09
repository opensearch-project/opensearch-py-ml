"""API type definitions for embedding model inference"""
from typing import TypedDict, List, Union, Optional


class EmbeddingRequest(TypedDict):
    """Single embedding request"""
    texts: List[str]
    content_type: Optional[str]  # "query" or "passage"


class BatchEmbeddingRequest(TypedDict):
    """Batch embedding request (OpenSearch connector format)"""
    parameters: EmbeddingRequest


class EmbeddingResponse(TypedDict):
    """Standard embedding response format"""
    embeddings: Union[List[float], List[List[float]]]
    processing_time_ms: float
    device: str
