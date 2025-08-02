import hashlib
import mimetypes
import os
from datetime import datetime, timedelta
from io import BytesIO
from typing import Dict, List, Optional, Tuple
from uuid import uuid4

from loguru import logger
from minio import Minio
from minio.error import S3Error

from src.utils.config import MinIOConfig


class MinIOClientError(Exception):
    pass


class MinIOClient:
    def __init__(self, config: MinIOConfig) -> None:
        self._config = config
        self._client = Minio(
            endpoint=config.endpoint,
            access_key=config.access_key,
            secret_key=config.secret_key,
            secure=config.secure,
        )
        self._bucket_name = config.bucket_name

    async def initialize(self) -> None:
        try:
            if not self._client.bucket_exists(self._bucket_name):
                self._client.make_bucket(self._bucket_name)
                logger.info(f"Created MinIO bucket: {self._bucket_name}")
            else:
                logger.info(f"MinIO bucket exists: {self._bucket_name}")
        except S3Error as e:
            logger.error(f"Failed to initialize MinIO bucket: {e}")
            raise MinIOClientError(f"MinIO initialization failed: {e}") from e

    def store_file(
        self,
        data: bytes,
        file_path: str,
        metadata: Optional[dict],
    ) -> str:
        """
        Store binary data in MinIO and return the object URL.

        Args:
            data: Binary data to store
            file_name: Original filename for generating object name
            metadata: Additional metadata for the object

        Returns:
            The object URL for accessing the data
        """
        try:
            # Generate object name
            file_extension = os.path.splitext(file_path)[1]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_id = str(uuid4())[:8]
            object_name = f"telegram/{timestamp}_{unique_id}{file_extension}"

            data_hash = hashlib.sha256(data).hexdigest()
            content_type = mimetypes.guess_type(file_path)[0] or "application/octet-stream"

            # Prepare metadata
            object_metadata: Dict[str, str | List[str] | Tuple[str]] = {
                "uploaded_at": datetime.now().isoformat(),
                "original_name": file_path,
                "content_type": content_type,
                "file_size": str(len(data)),
                "sha256": data_hash,
            }
            if metadata:
                object_metadata.update(metadata)

            data_stream = BytesIO(data)

            self._client.put_object(
                bucket_name=self._bucket_name,
                object_name=object_name,
                data=data_stream,
                length=len(data),
                content_type=content_type,
                metadata=object_metadata,
            )

            # Generate accessible URL
            object_url = self._generate_object_url(object_name)

            logger.info(f"Stored data in MinIO: {file_path} ({len(data)} bytes) -> {object_url}")
            return object_url

        except S3Error as e:
            logger.error(f"Failed to store data in MinIO: {e}")
            raise MinIOClientError(f"Failed to store data: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error storing data: {e}")
            raise MinIOClientError(f"Unexpected error: {e}") from e

    def get_presigned_url(self, object_name: str, expires: timedelta = timedelta(hours=1)) -> str:
        """
        Generate a presigned URL for accessing an object.

        Args:
            object_name: Name of the object in MinIO
            expires: URL expiration time

        Returns:
            Presigned URL for accessing the object
        """
        try:
            return self._client.presigned_get_object(
                bucket_name=self._bucket_name, object_name=object_name, expires=expires
            )
        except S3Error as e:
            logger.error(f"Failed to generate presigned URL: {e}")
            raise MinIOClientError(f"Failed to generate presigned URL: {e}") from e

    def delete_object(self, object_name: str) -> None:
        """
        Delete an object from MinIO.

        Args:
            object_name: Name of the object to delete
        """
        try:
            self._client.remove_object(self._bucket_name, object_name)
            logger.info(f"Deleted object from MinIO: {object_name}")
        except S3Error as e:
            logger.error(f"Failed to delete object: {e}")
            raise MinIOClientError(f"Failed to delete object: {e}") from e

    def get_object_info(self, object_name: str) -> dict:
        """
        Get information about an object.

        Args:
            object_name: Name of the object

        Returns:
            Object information including metadata
        """
        try:
            stat = self._client.stat_object(self._bucket_name, object_name)
            return {
                "object_name": object_name,
                "size": stat.size,
                "content_type": stat.content_type,
                "last_modified": stat.last_modified,
                "etag": stat.etag,
                "metadata": stat.metadata,
            }
        except S3Error as e:
            logger.error(f"Failed to get object info: {e}")
            raise MinIOClientError(f"Failed to get object info: {e}") from e

    def _generate_object_url(self, object_name: str) -> str:
        protocol = "https" if self._config.secure else "http"
        return f"{protocol}://{self._config.endpoint}/{self._bucket_name}/{object_name}"

    def _extract_object_name_from_url(self, url: str) -> Optional[str]:
        try:
            # Expected format: http(s)://endpoint/bucket/object_name
            if f"/{self._bucket_name}/" in url:
                return url.split(f"/{self._bucket_name}/", 1)[1]
            return None
        except Exception:
            return None

    def health_check(self) -> bool:
        try:
            self._client.bucket_exists(self._bucket_name)
            return True
        except Exception as e:
            logger.warning(f"MinIO health check failed: {e}")
            return False
