"""Event system for Aura — reactive callbacks on store, recall, and maintenance.

Usage:
    from aura import Aura
    from aura.events import AuraEvents

    brain = AuraEvents("./data")

    # Subscribe to events
    handle = brain.on_store(lambda record_id, content, level, tags: print(f"Stored: {content}"))
    brain.on_recall(lambda query, results: print(f"Recalled {len(results)} results for '{query}'"))
    brain.on_maintenance(lambda report: print(f"Maintenance done"))

    brain.store("Hello world")  # triggers on_store callback
    brain.recall_structured("hello", top_k=5)  # triggers on_recall callback
    brain.run_maintenance()  # triggers on_maintenance callback

    # Unsubscribe
    brain.off(handle)

AuraEvents wraps the Rust Aura class and intercepts store/recall/maintenance
calls to fire registered callbacks. All other methods are proxied transparently.
"""

from aura._core import Aura as _RustAura


class AuraEvents:
    """Aura with event callbacks.

    Drop-in replacement for Aura that adds on_store, on_recall,
    and on_maintenance hooks.
    """

    def __init__(self, *args, **kwargs):
        self._brain = _RustAura(*args, **kwargs)
        self._listeners = {}  # handle -> (event_type, callback, filter)
        self._next_handle = 0

    # ── Event registration ──

    def on_store(self, callback, tags=None, level=None):
        """Register a callback fired after every store().

        Args:
            callback: fn(record_id: str, content: str, level, tags: list)
            tags: Only fire if stored record has any of these tags (None = all)
            level: Only fire if stored record is at this level (None = all)

        Returns:
            int: Handle for unsubscribing via off().
        """
        handle = self._next_handle
        self._next_handle += 1
        self._listeners[handle] = ("store", callback, {"tags": tags, "level": level})
        return handle

    def on_recall(self, callback):
        """Register a callback fired after every recall_structured().

        Args:
            callback: fn(query: str, results: list[dict])

        Returns:
            int: Handle for unsubscribing via off().
        """
        handle = self._next_handle
        self._next_handle += 1
        self._listeners[handle] = ("recall", callback, {})
        return handle

    def on_maintenance(self, callback):
        """Register a callback fired after every run_maintenance().

        Args:
            callback: fn(report)

        Returns:
            int: Handle for unsubscribing via off().
        """
        handle = self._next_handle
        self._next_handle += 1
        self._listeners[handle] = ("maintenance", callback, {})
        return handle

    def off(self, handle):
        """Unsubscribe a callback by its handle.

        Returns:
            bool: True if the handle was found and removed.
        """
        return self._listeners.pop(handle, None) is not None

    # ── Intercepted methods ──

    def store(self, content, level=None, tags=None, **kwargs):
        """Store with on_store event firing."""
        record_id = self._brain.store(content, level=level, tags=tags, **kwargs)
        self._fire_store(record_id, content, level, tags or [])
        return record_id

    def recall_structured(self, query, top_k=None, **kwargs):
        """Recall with on_recall event firing."""
        results = self._brain.recall_structured(query, top_k=top_k, **kwargs)
        self._fire_recall(query, results)
        return results

    def recall(self, query, **kwargs):
        """Recall (text) with on_recall event firing."""
        result = self._brain.recall(query, **kwargs)
        self._fire_recall(query, result)
        return result

    def run_maintenance(self):
        """Maintenance with on_maintenance event firing."""
        report = self._brain.run_maintenance()
        self._fire_maintenance(report)
        return report

    # ── Event dispatch ──

    def _fire_store(self, record_id, content, level, tags):
        for _, (event_type, callback, filt) in self._listeners.items():
            if event_type != "store":
                continue
            # Apply tag filter
            if filt.get("tags") and not set(filt["tags"]) & set(tags):
                continue
            # Apply level filter
            if filt.get("level") and filt["level"] != level:
                continue
            try:
                callback(record_id, content, level, tags)
            except Exception:
                pass  # don't let listener errors break store

    def _fire_recall(self, query, results):
        for _, (event_type, callback, _) in self._listeners.items():
            if event_type != "recall":
                continue
            try:
                callback(query, results)
            except Exception:
                pass

    def _fire_maintenance(self, report):
        for _, (event_type, callback, _) in self._listeners.items():
            if event_type != "maintenance":
                continue
            try:
                callback(report)
            except Exception:
                pass

    # ── Proxy everything else to the Rust Aura ──

    def __getattr__(self, name):
        return getattr(self._brain, name)
