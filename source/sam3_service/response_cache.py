"""Bounded CPU-only exact-response cache for the remote SAM service.

The cache deliberately stores compact JSON bytes rather than tensors.  This
keeps GPU memory ownership inside the model process and makes every returned
value an independent object that callers cannot mutate in place.
"""

from __future__ import annotations

import json
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from time import perf_counter


@dataclass
class _Flight:
    event: threading.Event = field(default_factory=threading.Event)
    payload: bytes | None = None
    error: BaseException | None = None


@dataclass(frozen=True)
class CacheClaim:
    role: str
    payload: dict | None = None
    flight: _Flight | None = None


class ExactResponseCoordinator:
    """LRU cache plus identical in-flight request coalescing.

    ``claim`` returns one of three roles:

    * ``hit``: an exact cached response is available;
    * ``wait``: another thread owns the same exact request;
    * ``owner``: the caller must compute and then call ``complete``/``fail``.
    """

    def __init__(self, max_entries=24, max_bytes=128 * 1024 * 1024):
        if int(max_entries) < 1 or int(max_bytes) < 1:
            raise ValueError("cache limits must be positive")
        self.max_entries = int(max_entries)
        self.max_bytes = int(max_bytes)
        self._lock = threading.Lock()
        self._entries = OrderedDict()
        self._entry_bytes = 0
        self._flights = {}
        self._hits = 0
        self._misses = 0
        self._coalesced = 0
        self._evictions = 0

    @staticmethod
    def _encode(payload):
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"),
                          allow_nan=False).encode("utf-8")

    @staticmethod
    def _decode(payload):
        return json.loads(payload.decode("utf-8"))

    def peek(self, key):
        """Inspect without creating a new in-flight owner."""
        with self._lock:
            payload = self._entries.get(key)
            if payload is not None:
                self._entries.move_to_end(key)
                self._hits += 1
                return CacheClaim("hit", self._decode(payload))
            flight = self._flights.get(key)
            if flight is not None:
                self._coalesced += 1
                return CacheClaim("wait", flight=flight)
            return CacheClaim("miss")

    def claim(self, key):
        with self._lock:
            payload = self._entries.get(key)
            if payload is not None:
                self._entries.move_to_end(key)
                self._hits += 1
                return CacheClaim("hit", self._decode(payload))
            flight = self._flights.get(key)
            if flight is not None:
                self._coalesced += 1
                return CacheClaim("wait", flight=flight)
            flight = _Flight()
            self._flights[key] = flight
            self._misses += 1
            return CacheClaim("owner", flight=flight)

    def complete(self, key, flight, payload):
        encoded = self._encode(payload)
        with self._lock:
            if self._flights.get(key) is not flight:
                raise RuntimeError("request ownership changed before completion")
            if len(encoded) <= self.max_bytes:
                previous = self._entries.pop(key, None)
                if previous is not None:
                    self._entry_bytes -= len(previous)
                self._entries[key] = encoded
                self._entry_bytes += len(encoded)
                while (len(self._entries) > self.max_entries
                       or self._entry_bytes > self.max_bytes):
                    _, removed = self._entries.popitem(last=False)
                    self._entry_bytes -= len(removed)
                    self._evictions += 1
            flight.payload = encoded
            self._flights.pop(key, None)
            flight.event.set()

    def fail(self, key, flight, error):
        with self._lock:
            if self._flights.get(key) is flight:
                flight.error = error
                self._flights.pop(key, None)
                flight.event.set()

    def wait(self, flight, timeout=None):
        started = perf_counter()
        if not flight.event.wait(timeout):
            raise TimeoutError("timed out waiting for identical SAM request")
        waited = perf_counter() - started
        if flight.error is not None:
            raise flight.error
        if flight.payload is None:
            raise RuntimeError("identical SAM request completed without a payload")
        return self._decode(flight.payload), waited

    def stats(self):
        with self._lock:
            return {
                "entries": len(self._entries),
                "bytes": self._entry_bytes,
                "max_entries": self.max_entries,
                "max_bytes": self.max_bytes,
                "in_flight": len(self._flights),
                "hits": self._hits,
                "misses": self._misses,
                "coalesced_requests": self._coalesced,
                "evictions": self._evictions,
                "storage": "cpu_json",
            }
