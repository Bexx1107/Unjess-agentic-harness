"""Tests for the inter-agent messaging module — Message and MessageBus."""

import threading
import time
import uuid
from unittest.mock import patch

import pytest

from unjess.subagents.messaging import Message, MessageBus


# ---------------------------------------------------------------------------
# Message dataclass
# ---------------------------------------------------------------------------

class TestMessage:
    """Tests for the Message dataclass and its auto-populated fields."""

    def test_auto_generates_id(self):
        msg = Message()
        assert len(msg.id) == 12
        assert isinstance(msg.id, str)

    def test_auto_generated_ids_are_unique(self):
        ids = {Message().id for _ in range(100)}
        assert len(ids) == 100

    def test_preserves_explicit_id(self):
        msg = Message(id="custom-id-123")
        assert msg.id == "custom-id-123"

    def test_auto_sets_timestamp(self):
        before = time.time()
        msg = Message()
        after = time.time()
        assert before <= msg.timestamp <= after

    def test_preserves_explicit_timestamp(self):
        msg = Message(timestamp=1000.5)
        assert msg.timestamp == 1000.5

    def test_default_message_type(self):
        msg = Message()
        assert msg.message_type == "text"

    def test_custom_message_type(self):
        msg = Message(message_type="error")
        assert msg.message_type == "error"

    def test_all_fields_set(self):
        msg = Message(
            id="abc",
            sender="agent-1",
            recipient="agent-2",
            content="hello",
            timestamp=42.0,
            message_type="status",
        )
        assert msg.id == "abc"
        assert msg.sender == "agent-1"
        assert msg.recipient == "agent-2"
        assert msg.content == "hello"
        assert msg.timestamp == 42.0
        assert msg.message_type == "status"

    def test_defaults_are_empty_strings(self):
        msg = Message(id="x", timestamp=1.0)
        assert msg.sender == ""
        assert msg.recipient == ""
        assert msg.content == ""


# ---------------------------------------------------------------------------
# MessageBus basics
# ---------------------------------------------------------------------------

class TestMessageBus:
    """Tests for MessageBus — send, receive, peek, broadcast, and queries."""

    def test_send_and_receive(self):
        bus = MessageBus()
        msg = Message(sender="a", recipient="b", content="hi")
        bus.send(msg)
        received = bus.receive("b")
        assert len(received) == 1
        assert received[0].content == "hi"
        assert received[0].sender == "a"

    def test_receive_clears_mailbox(self):
        bus = MessageBus()
        bus.send(Message(sender="a", recipient="b", content="1"))
        bus.send(Message(sender="a", recipient="b", content="2"))
        first = bus.receive("b")
        assert len(first) == 2
        second = bus.receive("b")
        assert second == []

    def test_receive_empty_mailbox(self):
        bus = MessageBus()
        assert bus.receive("nobody") == []

    def test_peek_does_not_clear(self):
        bus = MessageBus()
        bus.send(Message(sender="a", recipient="b", content="peeked"))
        peeked = bus.peek("b")
        assert len(peeked) == 1
        peeked_again = bus.peek("b")
        assert len(peeked_again) == 1

    def test_peek_empty_mailbox(self):
        bus = MessageBus()
        assert bus.peek("nobody") == []

    def test_peek_returns_copy(self):
        bus = MessageBus()
        bus.send(Message(sender="a", recipient="b", content="c"))
        peeked = bus.peek("b")
        peeked.clear()  # mutate returned list
        assert bus.peek("b") == [bus.peek("b")[0]]  # original unchanged

    def test_has_messages_true(self):
        bus = MessageBus()
        bus.send(Message(sender="a", recipient="b", content="x"))
        assert bus.has_messages("b") is True

    def test_has_messages_false(self):
        bus = MessageBus()
        assert bus.has_messages("b") is False

    def test_has_messages_false_after_receive(self):
        bus = MessageBus()
        bus.send(Message(sender="a", recipient="b", content="x"))
        bus.receive("b")
        assert bus.has_messages("b") is False

    def test_message_count_zero(self):
        bus = MessageBus()
        assert bus.message_count("nobody") == 0

    def test_message_count_increments(self):
        bus = MessageBus()
        bus.send(Message(sender="a", recipient="b", content="1"))
        assert bus.message_count("b") == 1
        bus.send(Message(sender="a", recipient="b", content="2"))
        assert bus.message_count("b") == 2

    def test_message_count_after_receive(self):
        bus = MessageBus()
        bus.send(Message(sender="a", recipient="b", content="1"))
        bus.receive("b")
        assert bus.message_count("b") == 0

    def test_broadcast(self):
        bus = MessageBus()
        bus.broadcast("sender", ["r1", "r2", "r3"], "hello all")
        assert bus.message_count("r1") == 1
        assert bus.message_count("r2") == 1
        assert bus.message_count("r3") == 1
        msg_r1 = bus.receive("r1")[0]
        assert msg_r1.sender == "sender"
        assert msg_r1.content == "hello all"

    def test_broadcast_empty_recipients(self):
        bus = MessageBus()
        bus.broadcast("sender", [], "nobody listening")
        # No error, no messages anywhere

    def test_broadcast_creates_distinct_messages(self):
        bus = MessageBus()
        bus.broadcast("s", ["r1", "r2"], "same content")
        m1 = bus.receive("r1")[0]
        m2 = bus.receive("r2")[0]
        assert m1.id != m2.id
        assert m1.recipient == "r1"
        assert m2.recipient == "r2"

    def test_multiple_recipients_isolated(self):
        bus = MessageBus()
        bus.send(Message(sender="a", recipient="x", content="for x"))
        bus.send(Message(sender="a", recipient="y", content="for y"))
        x_msgs = bus.receive("x")
        y_msgs = bus.receive("y")
        assert len(x_msgs) == 1
        assert x_msgs[0].content == "for x"
        assert len(y_msgs) == 1
        assert y_msgs[0].content == "for y"

    def test_send_preserves_message_order(self):
        bus = MessageBus()
        for i in range(10):
            bus.send(Message(sender="a", recipient="b", content=str(i)))
        msgs = bus.receive("b")
        assert [m.content for m in msgs] == [str(i) for i in range(10)]

    def test_message_types_preserved(self):
        bus = MessageBus()
        bus.send(Message(sender="a", recipient="b", content="ok", message_type="result"))
        bus.send(Message(sender="a", recipient="b", content="bad", message_type="error"))
        msgs = bus.receive("b")
        assert msgs[0].message_type == "result"
        assert msgs[1].message_type == "error"


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------

class TestMessageBusThreadSafety:
    """Tests for MessageBus thread safety under concurrent access."""

    def test_concurrent_sends_no_lost_messages(self):
        bus = MessageBus()
        num_threads = 10
        msgs_per_thread = 100
        barrier = threading.Barrier(num_threads)

        def sender(thread_id: int) -> None:
            barrier.wait()
            for i in range(msgs_per_thread):
                bus.send(Message(
                    sender=f"t{thread_id}",
                    recipient="collector",
                    content=f"t{thread_id}-msg{i}",
                ))

        threads = [threading.Thread(target=sender, args=(t,)) for t in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        msgs = bus.receive("collector")
        assert len(msgs) == num_threads * msgs_per_thread

    def test_concurrent_send_and_receive(self):
        bus = MessageBus()
        total_sent = 200
        received: list[Message] = []
        lock = threading.Lock()
        done = threading.Event()

        def sender() -> None:
            for i in range(total_sent):
                bus.send(Message(sender="s", recipient="r", content=str(i)))
                time.sleep(0.001)
            done.set()

        def receiver() -> None:
            while not done.is_set() or bus.has_messages("r"):
                batch = bus.receive("r")
                if batch:
                    with lock:
                        received.extend(batch)
                time.sleep(0.005)

        s = threading.Thread(target=sender)
        r = threading.Thread(target=receiver)
        s.start()
        r.start()
        s.join()
        r.join()

        assert len(received) == total_sent

    def test_concurrent_peek_and_send(self):
        bus = MessageBus()
        errors: list[str] = []

        def sender() -> None:
            for i in range(50):
                bus.send(Message(sender="s", recipient="target", content=str(i)))
                time.sleep(0.001)

        def peeker() -> None:
            for _ in range(50):
                try:
                    bus.peek("target")
                except Exception as e:
                    errors.append(str(e))
                time.sleep(0.001)

        threads = [threading.Thread(target=sender), threading.Thread(target=peeker)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []

    def test_concurrent_broadcast(self):
        bus = MessageBus()
        recipients = [f"r{i}" for i in range(5)]
        num_broadcasts = 20

        def broadcaster(thread_id: int) -> None:
            for i in range(num_broadcasts):
                bus.broadcast(f"sender-{thread_id}", recipients, f"msg-{i}")

        threads = [threading.Thread(target=broadcaster, args=(t,)) for t in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        for rid in recipients:
            msgs = bus.receive(rid)
            assert len(msgs) == num_broadcasts * 4

    def test_concurrent_has_messages_and_count(self):
        bus = MessageBus()

        def writer() -> None:
            for _ in range(100):
                bus.send(Message(sender="w", recipient="x", content="d"))
                time.sleep(0.001)

        def reader() -> None:
            for _ in range(100):
                bus.has_messages("x")
                bus.message_count("x")
                time.sleep(0.001)

        threads = [threading.Thread(target=writer), threading.Thread(target=reader)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        # No assertion needed beyond no exceptions raised
