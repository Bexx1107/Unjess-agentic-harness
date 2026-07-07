"""Inter-agent messaging — send and receive messages between agents."""

import logging
import threading
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class Message:
    """A message between agents."""

    id: str = ""
    sender: str = ""  # conversation ID of sender
    recipient: str = ""  # conversation ID of recipient
    content: str = ""
    timestamp: float = 0.0
    message_type: str = "text"  # "text", "status", "result", "error"

    def __post_init__(self) -> None:
        if not self.id:
            self.id = uuid.uuid4().hex[:12]
        if not self.timestamp:
            self.timestamp = time.time()


class MessageBus:
    """Thread-safe message bus for inter-agent communication.

    Each agent has a mailbox identified by its conversation ID.
    Messages are stored until the recipient reads them.
    """

    def __init__(self) -> None:
        self._mailboxes: dict[str, list[Message]] = defaultdict(list)
        self._lock = threading.Lock()

    def send(self, message: Message) -> None:
        """Send a message to a recipient's mailbox.

        Args:
            message: The message to send.
        """
        with self._lock:
            self._mailboxes[message.recipient].append(message)
            logger.debug(
                "Message %s → %s: %s",
                message.sender,
                message.recipient,
                message.content[:80],
            )

    def receive(self, recipient_id: str) -> list[Message]:
        """Read and clear all messages for a recipient.

        Args:
            recipient_id: The conversation ID of the recipient.

        Returns:
            List of pending messages (clears the mailbox).
        """
        with self._lock:
            messages = self._mailboxes.pop(recipient_id, [])
            return messages

    def peek(self, recipient_id: str) -> list[Message]:
        """Peek at messages without clearing them.

        Args:
            recipient_id: The conversation ID.

        Returns:
            List of pending messages (does NOT clear).
        """
        with self._lock:
            return list(self._mailboxes.get(recipient_id, []))

    def has_messages(self, recipient_id: str) -> bool:
        """Check if a recipient has pending messages."""
        with self._lock:
            return bool(self._mailboxes.get(recipient_id))

    def message_count(self, recipient_id: str) -> int:
        """Count pending messages for a recipient."""
        with self._lock:
            return len(self._mailboxes.get(recipient_id, []))

    def broadcast(self, sender_id: str, recipient_ids: list[str], content: str) -> None:
        """Send the same message to multiple recipients.

        Args:
            sender_id: Sender's conversation ID.
            recipient_ids: List of recipient conversation IDs.
            content: Message content.
        """
        for rid in recipient_ids:
            self.send(Message(
                sender=sender_id,
                recipient=rid,
                content=content,
            ))
