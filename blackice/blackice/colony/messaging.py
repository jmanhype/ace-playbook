"""Messaging module for BLACKICE 3.0 colony.

Implements durable message passing between agents:
- Threaded conversations
- Message persistence
- Delivery guarantees
- Topic-based routing
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import uuid4

import structlog

from blackice.primitives.types import AgentId


logger = structlog.get_logger(__name__)


class MessagePriority(str, Enum):
    """Message priority levels."""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class MessageStatus(str, Enum):
    """Message delivery status."""

    PENDING = "pending"
    DELIVERED = "delivered"
    READ = "read"
    FAILED = "failed"


@dataclass
class Message:
    """A message between agents."""

    id: str
    sender: AgentId
    recipient: AgentId
    content: str
    thread_id: str | None = None
    priority: MessagePriority = MessagePriority.NORMAL
    status: MessageStatus = MessageStatus.PENDING
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    delivered_at: datetime | None = None
    read_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for persistence."""
        return {
            "id": self.id,
            "sender": str(self.sender),
            "recipient": str(self.recipient),
            "content": self.content,
            "thread_id": self.thread_id,
            "priority": self.priority.value,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "delivered_at": self.delivered_at.isoformat() if self.delivered_at else None,
            "read_at": self.read_at.isoformat() if self.read_at else None,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Message:
        """Create from dictionary."""
        return cls(
            id=data["id"],
            sender=AgentId(data["sender"]),
            recipient=AgentId(data["recipient"]),
            content=data["content"],
            thread_id=data.get("thread_id"),
            priority=MessagePriority(data.get("priority", "normal")),
            status=MessageStatus(data.get("status", "pending")),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.now(timezone.utc),
            delivered_at=datetime.fromisoformat(data["delivered_at"]) if data.get("delivered_at") else None,
            read_at=datetime.fromisoformat(data["read_at"]) if data.get("read_at") else None,
            metadata=data.get("metadata", {}),
        )


@dataclass
class Thread:
    """A conversation thread between agents."""

    id: str
    topic: str
    participants: list[AgentId]
    messages: list[Message] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    closed_at: datetime | None = None

    @property
    def is_closed(self) -> bool:
        """Check if thread is closed."""
        return self.closed_at is not None

    @property
    def message_count(self) -> int:
        """Get number of messages in thread."""
        return len(self.messages)


class MessageStore:
    """Durable storage for messages.

    Provides persistence for messages to survive restarts.
    """

    def __init__(self, storage_path: Path | None = None) -> None:
        """Initialize the message store.

        Args:
            storage_path: Path for persistent storage
        """
        self.storage_path = storage_path
        self._messages: dict[str, Message] = {}
        self._threads: dict[str, Thread] = {}

        if storage_path:
            self._load()

    def _load(self) -> None:
        """Load messages from storage."""
        if not self.storage_path or not self.storage_path.exists():
            return

        try:
            data = json.loads(self.storage_path.read_text())

            for msg_data in data.get("messages", []):
                msg = Message.from_dict(msg_data)
                self._messages[msg.id] = msg

            logger.info("messages_loaded", count=len(self._messages))

        except Exception as e:
            logger.error("failed_to_load_messages", error=str(e))

    def _save(self) -> None:
        """Save messages to storage."""
        if not self.storage_path:
            return

        try:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)

            data = {
                "messages": [msg.to_dict() for msg in self._messages.values()],
            }

            self.storage_path.write_text(json.dumps(data, indent=2))

        except Exception as e:
            logger.error("failed_to_save_messages", error=str(e))

    async def save(self, message: Message) -> None:
        """Save a message.

        Args:
            message: The message to save
        """
        self._messages[message.id] = message
        self._save()

    async def get(self, message_id: str) -> Message | None:
        """Get a message by ID.

        Args:
            message_id: The message ID

        Returns:
            The message if found
        """
        return self._messages.get(message_id)

    async def get_for_recipient(
        self,
        recipient: AgentId,
        status: MessageStatus | None = None,
        limit: int = 100,
    ) -> list[Message]:
        """Get messages for a recipient.

        Args:
            recipient: The recipient agent ID
            status: Optional status filter
            limit: Maximum messages to return

        Returns:
            List of messages
        """
        messages = [
            msg for msg in self._messages.values()
            if msg.recipient == recipient
            and (status is None or msg.status == status)
        ]
        messages.sort(key=lambda m: m.created_at, reverse=True)
        return messages[:limit]


class Messaging:
    """Agent messaging system.

    Provides reliable message passing between agents:
    - Send messages with delivery tracking
    - Receive with filtering and pagination
    - Threaded conversations
    - Message persistence
    """

    def __init__(self, storage_path: Path | None = None) -> None:
        """Initialize the messaging system.

        Args:
            storage_path: Path for persistent storage
        """
        self._store = MessageStore(storage_path)
        self._threads: dict[str, Thread] = {}
        self._queues: dict[AgentId, asyncio.Queue[Message]] = {}
        self._lock = asyncio.Lock()

    async def send(
        self,
        sender: AgentId,
        recipient: AgentId,
        content: str,
        *,
        thread_id: str | None = None,
        priority: MessagePriority = MessagePriority.NORMAL,
        metadata: dict[str, Any] | None = None,
    ) -> Message:
        """Send a message to another agent.

        Args:
            sender: Sending agent ID
            recipient: Recipient agent ID
            content: Message content
            thread_id: Optional thread to add to
            priority: Message priority
            metadata: Optional metadata

        Returns:
            The sent message
        """
        message = Message(
            id=str(uuid4()),
            sender=sender,
            recipient=recipient,
            content=content,
            thread_id=thread_id,
            priority=priority,
            metadata=metadata or {},
        )

        # Save to store
        await self._store.save(message)

        # Add to thread if specified
        if thread_id and thread_id in self._threads:
            self._threads[thread_id].messages.append(message)

        # Add to recipient queue
        async with self._lock:
            if recipient not in self._queues:
                self._queues[recipient] = asyncio.Queue()
            await self._queues[recipient].put(message)

        message.status = MessageStatus.DELIVERED
        message.delivered_at = datetime.now(timezone.utc)
        await self._store.save(message)

        logger.info(
            "message_sent",
            message_id=message.id,
            sender=str(sender),
            recipient=str(recipient),
            thread_id=thread_id,
        )

        return message

    async def receive(
        self,
        recipient: AgentId,
        timeout: float | None = None,
    ) -> Message | None:
        """Receive a message for an agent.

        Args:
            recipient: The agent to receive for
            timeout: Optional timeout in seconds

        Returns:
            The received message or None if timeout
        """
        async with self._lock:
            if recipient not in self._queues:
                self._queues[recipient] = asyncio.Queue()
            queue = self._queues[recipient]

        try:
            if timeout:
                message = await asyncio.wait_for(queue.get(), timeout=timeout)
            else:
                message = await queue.get()

            message.status = MessageStatus.READ
            message.read_at = datetime.now(timezone.utc)
            await self._store.save(message)

            return message

        except asyncio.TimeoutError:
            return None

    async def receive_all(
        self,
        recipient: AgentId,
        max_messages: int = 100,
    ) -> list[Message]:
        """Receive all pending messages for an agent.

        Args:
            recipient: The agent to receive for
            max_messages: Maximum messages to receive

        Returns:
            List of received messages
        """
        messages = []

        async with self._lock:
            if recipient not in self._queues:
                return messages
            queue = self._queues[recipient]

        while len(messages) < max_messages:
            try:
                message = queue.get_nowait()
                message.status = MessageStatus.READ
                message.read_at = datetime.now(timezone.utc)
                await self._store.save(message)
                messages.append(message)
            except asyncio.QueueEmpty:
                break

        return messages

    async def create_thread(
        self,
        topic: str,
        participants: list[AgentId],
    ) -> Thread:
        """Create a new conversation thread.

        Args:
            topic: Thread topic
            participants: Participating agent IDs

        Returns:
            The created thread
        """
        thread = Thread(
            id=str(uuid4()),
            topic=topic,
            participants=participants,
        )

        async with self._lock:
            self._threads[thread.id] = thread

        logger.info(
            "thread_created",
            thread_id=thread.id,
            topic=topic,
            participants=[str(p) for p in participants],
        )

        return thread

    async def get_thread(self, thread_id: str) -> Thread | None:
        """Get a thread by ID.

        Args:
            thread_id: The thread ID

        Returns:
            The thread if found
        """
        return self._threads.get(thread_id)

    async def get_thread_messages(
        self,
        thread_id: str,
        limit: int = 100,
    ) -> list[Message]:
        """Get messages in a thread.

        Args:
            thread_id: The thread ID
            limit: Maximum messages to return

        Returns:
            List of messages in the thread
        """
        thread = self._threads.get(thread_id)
        if thread is None:
            return []

        messages = thread.messages[-limit:]
        return messages

    async def close_thread(self, thread_id: str) -> bool:
        """Close a thread.

        Args:
            thread_id: The thread to close

        Returns:
            True if closed, False if not found
        """
        thread = self._threads.get(thread_id)
        if thread is None:
            return False

        thread.closed_at = datetime.now(timezone.utc)

        logger.info(
            "thread_closed",
            thread_id=thread_id,
            message_count=thread.message_count,
        )

        return True

    async def get_pending_count(self, recipient: AgentId) -> int:
        """Get count of pending messages for an agent.

        Args:
            recipient: The agent to check

        Returns:
            Number of pending messages
        """
        async with self._lock:
            if recipient not in self._queues:
                return 0
            return self._queues[recipient].qsize()

    async def broadcast(
        self,
        sender: AgentId,
        recipients: list[AgentId],
        content: str,
        *,
        priority: MessagePriority = MessagePriority.NORMAL,
    ) -> list[Message]:
        """Send a message to multiple recipients.

        Args:
            sender: Sending agent ID
            recipients: List of recipient IDs
            content: Message content
            priority: Message priority

        Returns:
            List of sent messages
        """
        messages = []
        for recipient in recipients:
            message = await self.send(
                sender=sender,
                recipient=recipient,
                content=content,
                priority=priority,
            )
            messages.append(message)
        return messages
