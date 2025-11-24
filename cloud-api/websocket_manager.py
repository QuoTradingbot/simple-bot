"""
WebSocket Manager for QuoTrading Cloud API
Handles real-time signal broadcasting to 1,000+ connected bots

Features:
- Push signals to all connected bots instantly (no polling)
- Connection pool management
- Automatic reconnection handling
- Redis pub/sub for multi-server broadcasting
"""

from fastapi import WebSocket, WebSocketDisconnect
from typing import Dict, Set, Optional, Any
import json
import asyncio
import logging
from datetime import datetime
from redis_manager import RedisManager

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manages WebSocket connections for real-time signal broadcasting"""
    
    def __init__(self, redis_manager: Optional[RedisManager] = None):
        """
        Initialize connection manager
        
        Args:
            redis_manager: Redis manager for pub/sub (optional)
        """
        # Active connections per user
        self.active_connections: Dict[str, Set[WebSocket]] = {}
        
        # Total connection count
        self.connection_count = 0
        
        # Redis for multi-server pub/sub
        self.redis_manager = redis_manager
        self.pubsub_task: Optional[asyncio.Task] = None
        
        # Message queue for broadcast
        self.broadcast_queue: asyncio.Queue = asyncio.Queue()
        
        logger.info("🔌 WebSocket Connection Manager initialized")
    
    async def connect(self, websocket: WebSocket, user_id: str):
        """
        Accept a new WebSocket connection
        
        Args:
            websocket: WebSocket connection
            user_id: User's license key or account ID
        """
        await websocket.accept()
        
        # Add to active connections
        if user_id not in self.active_connections:
            self.active_connections[user_id] = set()
        
        self.active_connections[user_id].add(websocket)
        self.connection_count += 1
        
        logger.info(f"✅ WebSocket connected: {user_id} (Total: {self.connection_count})")
        
        # Send welcome message
        await websocket.send_json({
            "type": "connected",
            "user_id": user_id,
            "timestamp": datetime.utcnow().isoformat(),
            "message": "QuoTrading Signal Stream - Connected"
        })
    
    def disconnect(self, websocket: WebSocket, user_id: str):
        """
        Remove a disconnected WebSocket
        
        Args:
            websocket: WebSocket connection
            user_id: User's license key or account ID
        """
        if user_id in self.active_connections:
            self.active_connections[user_id].discard(websocket)
            
            # Remove user entry if no connections left
            if not self.active_connections[user_id]:
                del self.active_connections[user_id]
        
        self.connection_count -= 1
        logger.info(f"❌ WebSocket disconnected: {user_id} (Total: {self.connection_count})")
    
    async def send_personal_message(self, message: Dict[str, Any], user_id: str):
        """
        Send message to specific user's connections
        
        Args:
            message: Message dict to send
            user_id: Target user ID
        """
        if user_id not in self.active_connections:
            return
        
        # Send to all connections for this user
        disconnected = set()
        for websocket in self.active_connections[user_id]:
            try:
                await websocket.send_json(message)
            except Exception as e:
                logger.warning(f"Failed to send to {user_id}: {e}")
                disconnected.add(websocket)
        
        # Clean up disconnected sockets
        for ws in disconnected:
            self.disconnect(ws, user_id)
    
    async def broadcast(self, message: Dict[str, Any], user_filter: Optional[Set[str]] = None):
        """
        Broadcast message to all connected users (or filtered subset)
        
        Args:
            message: Message dict to broadcast
            user_filter: Optional set of user_ids to send to (None = all)
        """
        # Add timestamp if not present
        if 'timestamp' not in message:
            message['timestamp'] = datetime.utcnow().isoformat()
        
        # Track send stats
        sent_count = 0
        failed_count = 0
        
        # Determine recipients
        recipients = user_filter if user_filter else set(self.active_connections.keys())
        
        # Send to all matching users
        for user_id in recipients:
            if user_id not in self.active_connections:
                continue
            
            # Send to all connections for this user
            disconnected = set()
            for websocket in self.active_connections[user_id]:
                try:
                    await websocket.send_json(message)
                    sent_count += 1
                except Exception as e:
                    logger.warning(f"Broadcast failed to {user_id}: {e}")
                    disconnected.add(websocket)
                    failed_count += 1
            
            # Clean up disconnected sockets
            for ws in disconnected:
                self.disconnect(ws, user_id)
        
        logger.info(f"📡 Broadcast: {message.get('type', 'unknown')} → {sent_count} clients ({failed_count} failed)")
        
        # Publish to Redis for multi-server scenarios
        if self.redis_manager:
            try:
                self.redis_manager.publish('quotrading:signals', json.dumps(message))
            except Exception as e:
                logger.warning(f"Redis publish failed: {e}")
    
    async def broadcast_signal(self, signal: Dict[str, Any]):
        """
        Broadcast trading signal to all connected bots
        
        Args:
            signal: Signal data dict
        """
        message = {
            "type": "signal",
            "data": signal,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        await self.broadcast(message)
        
        logger.info(f"📢 Signal broadcast: {signal.get('symbol')} {signal.get('action')} @ {signal.get('price')}")
    
    async def broadcast_market_update(self, market_data: Dict[str, Any]):
        """
        Broadcast market data update (VWAP, RSI, etc.)
        
        Args:
            market_data: Market data dict
        """
        message = {
            "type": "market_update",
            "data": market_data,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        await self.broadcast(message)
    
    async def send_notification(self, user_id: str, notification: Dict[str, Any]):
        """
        Send notification to specific user
        
        Args:
            user_id: Target user ID
            notification: Notification data
        """
        message = {
            "type": "notification",
            "data": notification,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        await self.send_personal_message(message, user_id)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get connection statistics"""
        return {
            "total_connections": self.connection_count,
            "unique_users": len(self.active_connections),
            "connections_per_user": {
                user_id: len(connections)
                for user_id, connections in self.active_connections.items()
            }
        }
    
    async def start_redis_listener(self):
        """Start listening to Redis pub/sub for multi-server broadcasting"""
        if not self.redis_manager:
            logger.warning("Redis not available - multi-server broadcasting disabled")
            return
        
        logger.info("🎧 Starting Redis pub/sub listener for multi-server broadcasting")
        
        try:
            pubsub = self.redis_manager.redis_client.pubsub()
            await pubsub.subscribe('quotrading:signals')
            
            async for message in pubsub.listen():
                if message['type'] == 'message':
                    try:
                        data = json.loads(message['data'])
                        await self.broadcast(data)
                    except Exception as e:
                        logger.error(f"Error processing Redis message: {e}")
        
        except Exception as e:
            logger.error(f"Redis listener error: {e}")


# Global connection manager instance
connection_manager: Optional[ConnectionManager] = None


def get_connection_manager() -> ConnectionManager:
    """Get global connection manager instance"""
    return connection_manager


def init_connection_manager(redis_manager: Optional[RedisManager] = None):
    """Initialize global connection manager"""
    global connection_manager
    connection_manager = ConnectionManager(redis_manager)
    return connection_manager
