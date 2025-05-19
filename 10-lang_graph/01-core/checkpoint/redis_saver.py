from contextlib import contextmanager
from typing import Optional, Iterator, List, Tuple, Any

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver, Checkpoint, CheckpointMetadata, ChannelVersions, WRITES_IDX_MAP, CheckpointTuple, get_checkpoint_id, \
    PendingWrite
from redis import Redis

from redis_saver_helper import _make_redis_checkpoint_key, _make_redis_checkpoint_writes_key, _parse_redis_checkpoint_key, \
    _parse_redis_checkpoint_data, _filter_keys, _parse_redis_checkpoint_writes_key, _load_writes


class RedisSaver(BaseCheckpointSaver):
    """Redis-based checkpoint saver implementation."""

    conn: Redis
    expiration_seconds: int

    def __init__(self, conn: Redis, expiration_seconds: Optional[int] = 60 * 5):
        super().__init__()
        self.conn = conn
        self.expiration_seconds = expiration_seconds

    @classmethod
    @contextmanager
    def from_conn_info(cls, *, host: str, port: int, db: int, expiration_seconds: Optional[int] = 60 * 5) -> Iterator["RedisSaver"]:
        conn = None
        try:
            conn = Redis(host=host, port=port, db=db)
            yield RedisSaver(conn, expiration_seconds)
        finally:
            if conn:
                conn.close()

    @classmethod
    @contextmanager
    def from_conn_url(cls, *, url: str, expiration_seconds: Optional[int] = 60 * 5) -> Iterator["RedisSaver"]:
        conn = None
        try:
            conn = Redis.from_url(url)
            yield RedisSaver(conn, expiration_seconds)
        finally:
            if conn:
                conn.close()

    def put(
            self,
            config: RunnableConfig,
            checkpoint: Checkpoint,
            metadata: CheckpointMetadata,
            new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Save a checkpoint to Redis.

        Args:
            config (RunnableConfig): The config to associate with the checkpoint.
            checkpoint (Checkpoint): The checkpoint to save.
            metadata (CheckpointMetadata): Additional metadata to save with the checkpoint.
            new_versions (ChannelVersions): New channel versions as of this write.

        Returns:
            RunnableConfig: Updated configuration after storing the checkpoint.
        """
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"]["checkpoint_ns"]
        checkpoint_id = checkpoint["id"]
        parent_checkpoint_id = config["configurable"].get("checkpoint_id")
        key = _make_redis_checkpoint_key(thread_id, checkpoint_ns, checkpoint_id)

        type_, serialized_checkpoint = self.serde.dumps_typed(checkpoint)
        serialized_metadata = self.serde.dumps(metadata)
        data = {
            "checkpoint": serialized_checkpoint,
            "type": type_,
            "metadata": serialized_metadata,
            "parent_checkpoint_id": parent_checkpoint_id
            if parent_checkpoint_id
            else "",
        }
        self.conn.hset(key, mapping=data)
        self.conn.expire(key, self.expiration_seconds)
        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
            }
        }

    def put_writes(
            self,
            config: RunnableConfig,
            writes: List[Tuple[str, Any]],
            task_id: str,
    ) -> None:
        """Store intermediate writes linked to a checkpoint.

        Args:
            config (RunnableConfig): Configuration of the related checkpoint.
            writes (Sequence[Tuple[str, Any]]): List of writes to store, each as (channel, value) pair.
            task_id (str): Identifier for the task creating the writes.
        """
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"]["checkpoint_ns"]
        checkpoint_id = config["configurable"]["checkpoint_id"]

        for idx, (channel, value) in enumerate(writes):
            key = _make_redis_checkpoint_writes_key(
                thread_id,
                checkpoint_ns,
                checkpoint_id,
                task_id,
                WRITES_IDX_MAP.get(channel, idx),
            )
            type_, serialized_value = self.serde.dumps_typed(value)
            data = {"channel": channel, "type": type_, "value": serialized_value}
            if all(w[0] in WRITES_IDX_MAP for w in writes):
                # Use HSET which will overwrite existing values
                self.conn.hset(key, mapping=data)
                self.conn.expire(key, self.expiration_seconds)
            else:
                # Use HSETNX which will not overwrite existing values
                for field, value in data.items():
                    self.conn.hsetnx(key, field, value)
                    self.conn.expire(key, self.expiration_seconds)

    def get_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        """Get a checkpoint tuple from Redis.

        This method retrieves a checkpoint tuple from Redis based on the
        provided config. If the config contains a "checkpoint_id" key, the checkpoint with
        the matching thread ID and checkpoint ID is retrieved. Otherwise, the latest checkpoint
        for the given thread ID is retrieved.

        Args:
            config (RunnableConfig): The config to use for retrieving the checkpoint.

        Returns:
            Optional[CheckpointTuple]: The retrieved checkpoint tuple, or None if no matching checkpoint was found.
        """
        thread_id = config["configurable"]["thread_id"]
        checkpoint_id = get_checkpoint_id(config)
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")

        checkpoint_key = self._get_checkpoint_key(
            self.conn, thread_id, checkpoint_ns, checkpoint_id
        )
        if not checkpoint_key:
            return None

        checkpoint_data = self.conn.hgetall(checkpoint_key)

        # load pending writes
        checkpoint_id = (
                checkpoint_id
                or _parse_redis_checkpoint_key(checkpoint_key)["checkpoint_id"]
        )
        pending_writes = self._load_pending_writes(
            thread_id, checkpoint_ns, checkpoint_id
        )
        return _parse_redis_checkpoint_data(
            self.serde, checkpoint_key, checkpoint_data, pending_writes=pending_writes
        )

    def list(
            self,
            config: Optional[RunnableConfig],
            *,
            # TODO: implement filtering
            filter: Optional[dict[str, Any]] = None,
            before: Optional[RunnableConfig] = None,
            limit: Optional[int] = None,
    ) -> Iterator[CheckpointTuple]:
        """List checkpoints from the database.

        This method retrieves a list of checkpoint tuples from Redis based
        on the provided config. The checkpoints are ordered by checkpoint ID in descending order (newest first).

        Args:
            config (RunnableConfig): The config to use for listing the checkpoints.
            filter (Optional[Dict[str, Any]]): Additional filtering criteria for metadata. Defaults to None.
            before (Optional[RunnableConfig]): If provided, only checkpoints before the specified checkpoint ID are returned. Defaults to None.
            limit (Optional[int]): The maximum number of checkpoints to return. Defaults to None.

        Yields:
            Iterator[CheckpointTuple]: An iterator of checkpoint tuples.
        """
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        pattern = _make_redis_checkpoint_key(thread_id, checkpoint_ns, "*")

        keys = _filter_keys(self.conn.keys(pattern), before, limit)
        for key in keys:
            data = self.conn.hgetall(key)
            if data and b"checkpoint" in data and b"metadata" in data:
                # load pending writes
                checkpoint_id = _parse_redis_checkpoint_key(key.decode())[
                    "checkpoint_id"
                ]
                pending_writes = self._load_pending_writes(
                    thread_id, checkpoint_ns, checkpoint_id
                )
                yield _parse_redis_checkpoint_data(
                    self.serde, key.decode(), data, pending_writes=pending_writes
                )

    def _load_pending_writes(
            self, thread_id: str, checkpoint_ns: str, checkpoint_id: str
    ) -> List[PendingWrite]:
        writes_key = _make_redis_checkpoint_writes_key(
            thread_id, checkpoint_ns, checkpoint_id, "*", None
        )
        matching_keys = self.conn.keys(pattern=writes_key)
        parsed_keys = [
            _parse_redis_checkpoint_writes_key(key.decode()) for key in matching_keys
        ]
        pending_writes = _load_writes(
            self.serde,
            {
                (parsed_key["task_id"], parsed_key["idx"]): self.conn.hgetall(key)
                for key, parsed_key in sorted(
                zip(matching_keys, parsed_keys), key=lambda x: x[1]["idx"]
            )
            },
        )
        return pending_writes

    def _get_checkpoint_key(
            self, conn, thread_id: str, checkpoint_ns: str, checkpoint_id: Optional[str]
    ) -> Optional[str]:
        """Determine the Redis key for a checkpoint."""
        if checkpoint_id:
            return _make_redis_checkpoint_key(thread_id, checkpoint_ns, checkpoint_id)

        all_keys = conn.keys(_make_redis_checkpoint_key(thread_id, checkpoint_ns, "*"))
        if not all_keys:
            return None

        latest_key = max(
            all_keys,
            key=lambda k: _parse_redis_checkpoint_key(k.decode())["checkpoint_id"],
        )
        return latest_key.decode()

    def clear_history(self, thread_id):
        checkpoint_deleted_count = writes_deleted_count = 0
        redis_conn = self.conn
        # 删除 checkpoint
        checkpoint_matching_keys = redis_conn.keys(f"checkpoint${thread_id}*")
        if checkpoint_matching_keys:
            checkpoint_deleted_count = redis_conn.delete(*checkpoint_matching_keys)
        # 删除 writes
        writes_matching_keys = redis_conn.keys(f"writes${thread_id}*")
        if writes_matching_keys:
            writes_deleted_count = redis_conn.delete(*writes_matching_keys)
        # 返回删除数量
        return {
            "checkpoint_deleted_count": checkpoint_deleted_count,
            "writes_deleted_count": writes_deleted_count,
        }
