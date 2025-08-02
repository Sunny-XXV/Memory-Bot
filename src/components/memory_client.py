import json
from datetime import datetime
from typing import Dict, List, Optional
from uuid import UUID

import httpx
from loguru import logger
from pydantic import ValidationError

from src.utils.config import MemoryAPIConfig
from src.utils.models import (
    IngestionResponse,
    MemoryItem,
    MemoryItemRaw,
    RelatedItemsResponse,
    RetrievalResponse,
    TaskStatus,
)


class MemoryAPIError(Exception):
    pass


class MemoryAPIClient:
    def __init__(self, config: MemoryAPIConfig) -> None:
        self._config = config
        self._client = httpx.AsyncClient(timeout=config.timeout)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self._client.aclose()

    async def close(self) -> None:
        await self._client.aclose()

    async def ingest(self, item: MemoryItemRaw) -> IngestionResponse:
        try:
            response = await self._client.post(
                self._config.ingest_url,
                json=item.model_dump(mode="json"),
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()

            data = response.json()
            return IngestionResponse.model_validate(data)

        except httpx.HTTPStatusError as e:
            logger.error(
                f"HTTP error during ingestion: {e.response.status_code} - {e.response.text}"
            )
            raise MemoryAPIError(f"Failed to ingest item: {e.response.text}") from e
        except ValidationError as e:
            logger.error(f"Validation error in ingestion response: {e}")
            raise MemoryAPIError(f"Invalid response format: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error during ingestion: {e}")
            raise MemoryAPIError(f"Ingestion failed: {e}") from e

    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        content_types: Optional[List[str]] = None,
        include_context: bool = False,
        enable_reranking: bool = True,
    ) -> RetrievalResponse:
        params = {
            "query": query,
            "top_k": top_k,
            "include_context": include_context,
            "enable_reranking": enable_reranking,
        }

        if filters:
            params["filters"] = json.dumps(filters)
        if start_date:
            params["start_date"] = start_date.isoformat()
        if end_date:
            params["end_date"] = end_date.isoformat()
        if content_types:
            params["content_types"] = ",".join(content_types)

        try:
            response = await self._client.get(self._config.retrieve_url, params=params)
            response.raise_for_status()

            data = response.json()
            return RetrievalResponse.model_validate(data)

        except httpx.HTTPStatusError as e:
            logger.error(
                f"HTTP error during retrieval: {e.response.status_code} - {e.response.text}"
            )
            raise MemoryAPIError(f"Failed to retrieve items: {e.response.text}") from e
        except ValidationError as e:
            logger.error(f"Validation error in retrieval response: {e}")
            raise MemoryAPIError(f"Invalid response format: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error during retrieval: {e}")
            raise MemoryAPIError(f"Retrieval failed: {e}") from e

    async def get_item(self, item_id: UUID) -> MemoryItem:
        try:
            response = await self._client.get(self._config.item_url(str(item_id)))
            response.raise_for_status()

            data = response.json()
            return MemoryItem.model_validate(data)

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                raise MemoryAPIError(f"Item not found: {item_id}") from e
            logger.error(f"HTTP error getting item: {e.response.status_code} - {e.response.text}")
            raise MemoryAPIError(f"Failed to get item: {e.response.text}") from e
        except ValidationError as e:
            logger.error(f"Validation error in item response: {e}")
            raise MemoryAPIError(f"Invalid response format: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error getting item: {e}")
            raise MemoryAPIError(f"Get item failed: {e}") from e

    async def get_related_items(
        self, item_id: UUID, relationship_types: Optional[List[str]] = None
    ) -> RelatedItemsResponse:
        url = f"{self._config.item_url(str(item_id))}/related"
        params = {}

        if relationship_types:
            params["relationship_types"] = ",".join(relationship_types)

        try:
            response = await self._client.get(url, params=params)
            response.raise_for_status()

            data = response.json()
            return RelatedItemsResponse.model_validate(data)

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                raise MemoryAPIError(f"Item not found: {item_id}") from e
            logger.error(
                f"HTTP error getting related items: {e.response.status_code} - {e.response.text}"
            )
            raise MemoryAPIError(f"Failed to get related items: {e.response.text}") from e
        except ValidationError as e:
            logger.error(f"Validation error in related items response: {e}")
            raise MemoryAPIError(f"Invalid response format: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error getting related items: {e}")
            raise MemoryAPIError(f"Get related items failed: {e}") from e

    async def get_task_status(self, task_id: UUID) -> TaskStatus:
        try:
            response = await self._client.get(self._config.task_url(str(task_id)))
            response.raise_for_status()

            data = response.json()
            return TaskStatus.model_validate(data)

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                raise MemoryAPIError(f"Task not found: {task_id}") from e
            logger.error(
                f"HTTP error getting task status: {e.response.status_code} - {e.response.text}"
            )
            raise MemoryAPIError(f"Failed to get task status: {e.response.text}") from e
        except ValidationError as e:
            logger.error(f"Validation error in task status response: {e}")
            raise MemoryAPIError(f"Invalid response format: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error getting task status: {e}")
            raise MemoryAPIError(f"Get task status failed: {e}") from e

    async def health_check(self) -> bool:
        """Check if the Memory API is healthy."""
        try:
            response = await self._client.get(f"{self._config.base_url.rstrip('/')}/health")
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"Health check failed: {e}")
            return False
