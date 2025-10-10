"""API type definitions for semantic highlighting inference"""
from typing import TypedDict, List, Union


class HighlightSpan(TypedDict):
    """Character-level highlight span"""
    start: int
    end: int


class SingleRequest(TypedDict):
    """Single document request"""
    question: str
    context: str


class BatchRequest(TypedDict):
    """Batch request"""
    inputs: List[SingleRequest]


class InferenceResponse(TypedDict):
    """Standard response format"""
    highlights: Union[List[HighlightSpan], List[List[HighlightSpan]]]
    processing_time_ms: float
    device: str
