from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import List

router = APIRouter()


class ConnectionManager:
    """Manages active WebSocket connections with error-isolated broadcast."""

    def __init__(self) -> None:
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket) -> None:
        try:
            self.active_connections.remove(websocket)
        except ValueError:
            pass  # Already removed

    async def broadcast(self, message: dict) -> None:
        """Send to all clients. Dead connections are pruned automatically."""
        dead: list[WebSocket] = []
        for conn in self.active_connections:
            try:
                await conn.send_json(message)
            except Exception:
                dead.append(conn)
        for conn in dead:
            self.disconnect(conn)


manager = ConnectionManager()


@router.websocket("/events")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            await websocket.send_json({"echo": data})
    except WebSocketDisconnect:
        manager.disconnect(websocket)


async def broadcast_event(event_type: str, payload: dict) -> None:
    """Broadcast an event to all connected WebSocket clients."""
    await manager.broadcast({
        "type": event_type,
        "payload": payload,
    })
