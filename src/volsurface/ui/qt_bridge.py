"""The only place where the asyncio event bus meets Qt.

Under ``qasync`` the Qt and asyncio loops are one, so bus handlers already run on the GUI thread.
The bridge subscribes to the application events the window cares about and re-emits them as Qt
signals, which keeps widgets free of any knowledge of the bus.
"""

from PySide6 import QtCore

from volsurface.app.events import (
    AlertRaised,
    ComputeFailed,
    SessionStateChanged,
    SurfaceComputed,
)
from volsurface.domain.events import HandlerFailed
from volsurface.domain.ports import DeliveryPolicy, EventSubscriber, Subscription


class EventBridge(QtCore.QObject):
    update_ready = QtCore.Signal(object)
    state_changed = QtCore.Signal(object)
    alert_raised = QtCore.Signal(object)
    failure = QtCore.Signal(str)

    def __init__(self, parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent)
        self._subscriptions: list[Subscription] = []

    def bind(self, bus: EventSubscriber) -> None:
        """Start forwarding events from ``bus``; call once."""
        self._subscriptions += [
            bus.subscribe(
                SurfaceComputed,
                self._on_update,
                policy=DeliveryPolicy.LATEST,
                name="ui.update",
            ),
            bus.subscribe(SessionStateChanged, self.state_changed.emit, name="ui.state"),
            bus.subscribe(AlertRaised, self._on_alert, name="ui.alert"),
            bus.subscribe(ComputeFailed, self._on_compute_failed, name="ui.compute_failed"),
            bus.subscribe(HandlerFailed, self._on_handler_failed, name="ui.handler_failed"),
        ]

    def close(self) -> None:
        for subscription in self._subscriptions:
            subscription.cancel()
        self._subscriptions.clear()

    def _on_update(self, event: SurfaceComputed) -> None:
        self.update_ready.emit(event.update)

    def _on_alert(self, event: AlertRaised) -> None:
        self.alert_raised.emit(event.alert)

    def _on_compute_failed(self, event: ComputeFailed) -> None:
        self.failure.emit(f"{event.source}: {event.error}")

    def _on_handler_failed(self, event: HandlerFailed) -> None:
        self.failure.emit(f"{event.handler}: {event.error}")
