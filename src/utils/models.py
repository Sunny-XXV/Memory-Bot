from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field
from tabulate import tabulate


class MemoryItemRaw(BaseModel):
    """Model for creating a new MemoryItem via the API"""

    content_type: str = Field(..., description="'text', 'image', 'audio', 'video', 'web_link'")
    text_content: Optional[str] = Field(None, description="Raw text content")
    data_uri: Optional[str] = Field(None, description="Pointer to binary data like S3 URL")
    event_timestamp: datetime = Field(..., description="Real-world timestamp of the event")
    meta: Optional[Dict[str, Any]] = Field(None, description="Metadata about the origin")
    reply_to_id: Optional[UUID] = Field(None, description="ID of the item this is replying to")


class MemoryItem(BaseModel):
    """Full MemoryItem model as stored in database"""

    model_config = ConfigDict(from_attributes=True)

    id: UUID = Field(default_factory=uuid4)
    parent_id: Optional[UUID] = None
    content_type: str
    text_content: Optional[str] = None
    analyzed_text: Optional[str] = None
    data_uri: Optional[str] = None
    embedding: Optional[List[float]] = None
    embedding_model_version: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None
    event_timestamp: datetime
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def __str__(self) -> str:
        headers = ["Field", "Value"]
        table = [
            ["id", str(self.id)],
            ["parent_id", str(self.parent_id) if self.parent_id else "None"],
            ["content_type", self.content_type],
            ["text_content", self.text_content[:60] if self.text_content else "None"],
            ["analyzed_text", self.analyzed_text[:60] if self.analyzed_text else "None"],
            ["data_uri", self.data_uri[:60] if self.data_uri else "None"],
            ["embedding_model_version", self.embedding_model_version or "None"],
            ["meta", str(self.meta)[:60] if self.meta else "None"],
            ["event_timestamp", self.event_timestamp.isoformat()],
            ["created_at", self.created_at.isoformat()],
            ["updated_at", self.updated_at.isoformat()],
        ]
        return "\n" + tabulate(table, headers=headers, tablefmt="rounded_outline")


class MemoryItemResponse(BaseModel):
    """Search result with score."""

    item: MemoryItem
    score: Optional[float] = None


class RetrievalResponse(BaseModel):
    """Response from retrieve API."""

    query: str
    results: List[MemoryItemResponse]


class IngestionResponse(BaseModel):
    """Response from ingestion API."""

    status: str = Field(..., description="Ingestion status")
    item_id: UUID = Field(..., description="Generated item ID")


class TaskStatus(BaseModel):
    """Background task status."""

    task_id: UUID = Field(..., description="Task ID")
    task_type: str = Field(..., description="Type of task")
    status: str = Field(..., description="Task status")
    source_item_id: UUID = Field(..., description="Source item ID")
    created_at: datetime = Field(..., description="When task was created")
    started_at: Optional[datetime] = Field(None, description="When task started")
    completed_at: Optional[datetime] = Field(None, description="When task completed")
    error_message: Optional[str] = Field(None, description="Error message if failed")


class Relationship(BaseModel):
    """Relationship between memory items."""

    id: UUID = Field(default_factory=uuid4, description="Unique relationship ID")
    source_node_id: UUID = Field(..., description="Source item ID")
    target_node_id: UUID = Field(..., description="Target item ID")
    relationship_type: str = Field(..., description="Type of relationship")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class RelatedItemResult(BaseModel):
    """Related item with relationship information."""

    item: MemoryItem = Field(..., description="The related memory item")
    relationship: Relationship = Field(..., description="The relationship")


class RelatedItemsResponse(BaseModel):
    """Response for related items API."""

    item_id: UUID = Field(..., description="Original item ID")
    related_items: List[RelatedItemResult] = Field(..., description="Related items")
