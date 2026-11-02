"""Tests for refcounted MicCaptureHub."""

from __future__ import annotations

import unittest

import numpy as np

from whisper_timestamped.audio_capture import MicCaptureHub, get_shared_mic_hub, reset_shared_mic_hub


class _FakeStream:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.started = False
        self.stopped = False
        self.closed = False

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def close(self):
        self.closed = True


class TestMicCaptureHub(unittest.TestCase):
    def tearDown(self):
        reset_shared_mic_hub()

    def test_first_subscribe_opens_last_unsub_closes(self):
        streams = []

        def factory(**kwargs):
            s = _FakeStream(**kwargs)
            streams.append(s)
            return s

        hub = MicCaptureHub(stream_factory=factory)

        def on_a(data):
            pass

        def on_b(data):
            pass

        hub.subscribe(on_a, sample_rate=16000)
        self.assertEqual(len(streams), 1)
        self.assertTrue(streams[0].started)
        hub.subscribe(on_b, sample_rate=16000)
        self.assertEqual(len(streams), 1)
        hub.unsubscribe(on_a)
        self.assertFalse(streams[0].closed)
        self.assertEqual(hub.listener_count, 1)
        hub.unsubscribe(on_b)
        self.assertTrue(streams[0].stopped)
        self.assertTrue(streams[0].closed)
        self.assertFalse(hub.is_running)

    def test_fanout_delivers_copy_to_all_listeners(self):
        hub = MicCaptureHub(stream_factory=lambda **kw: _FakeStream(**kw))
        received = []
        hub.subscribe(received.append)
        block = np.linspace(0, 1, 8, dtype=np.float32)
        hub._audio_callback(block.reshape(-1, 1), 8, None, None)
        self.assertEqual(len(received), 1)
        np.testing.assert_array_equal(received[0], block)
        self.assertIsNot(received[0], block)

    def test_shared_singleton_reset(self):
        a = get_shared_mic_hub()
        b = get_shared_mic_hub()
        self.assertIs(a, b)
        reset_shared_mic_hub()
        c = get_shared_mic_hub()
        self.assertIsNot(a, c)


if __name__ == "__main__":
    unittest.main()
