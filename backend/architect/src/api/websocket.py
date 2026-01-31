from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import List
import json
import asyncio

router = APIRouter()

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            await connection.send_json(message)

manager = ConnectionManager()

@router.websocket("/events")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        # Keep connection open and send occasional heartbeats
        while True:
            # In a real app, this would listen to a Redis/PubSub bus
            # For now, we just wait for disconnection
            data = await websocket.receive_text()
            # Echo for testing
            await websocket.send_json({"echo": data})
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# Helper function to broadcast events from other modules
async def broadcast_event(event_type: str, payload: dict):
    await manager.broadcast({
        "type": event_type,
        "payload": payload
    })
