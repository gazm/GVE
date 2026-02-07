# backend/architect/src/torch_preloader.py
"""
Torch Preloader - Lazy loads PyTorch/ROCm in background thread at startup.

Solves the problem of slow torch+ROCm initialization (~5-10s) by:
1. Starting preload in background thread at server startup
2. Broadcasting status via optional callback so UI shows progress
3. Providing ensure_loaded() for code that needs torch

Status states: "cold" -> "loading" -> "ready" | "unavailable"
Once loaded, torch stays in memory for the lifetime of the process.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any, Awaitable, Callable, Literal

TorchStatus = Literal["cold", "loading", "ready", "unavailable"]

# Type alias for the broadcast callback: async fn(event_type, payload) -> None
BroadcastCallback = Callable[[str, dict[str, Any]], Awaitable[None]]


class TorchPreloader:
    """
    Singleton that manages lazy torch loading.
    
    Usage:
        from src.torch_preloader import preloader
        
        # At startup (in FastAPI lifespan):
        preloader.set_broadcast(broadcast_event, asyncio.get_running_loop())
        preloader.start_preload()
        
        # Before torch-dependent code:
        preloader.ensure_loaded()
        
        # Or get torch directly:
        torch = preloader.get_torch()
    """
    
    _instance: TorchPreloader | None = None
    
    def __new__(cls) -> TorchPreloader:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self) -> None:
        if self._initialized:
            return
        
        self._status: TorchStatus = "cold"
        self._torch: Any = None
        self._device: str = "cpu"
        self._device_name: str = "CPU"
        self._error: str | None = None
        self._ready_event = threading.Event()
        self._preload_thread: threading.Thread | None = None
        self._broadcast_fn: BroadcastCallback | None = None
        self._main_loop: asyncio.AbstractEventLoop | None = None
        self._initialized = True
    
    def get_status(self) -> dict[str, Any]:
        """Get current torch status as dict (for API/WebSocket)."""
        return {
            "status": self._status,
            "device": self._device,
            "device_name": self._device_name,
            "error": self._error,
        }
    
    def set_broadcast(
        self,
        callback: BroadcastCallback,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        """
        Register an async broadcast callback and the event loop it runs on.
        
        Call from async context (e.g. FastAPI lifespan) BEFORE start_preload().
        The preloader will schedule broadcasts on this loop from its thread.
        """
        self._broadcast_fn = callback
        self._main_loop = loop
    
    def start_preload(self) -> None:
        """
        Start background thread to preload torch.
        
        Safe to call multiple times - only first call starts the thread.
        """
        if self._preload_thread is not None:
            return  # Already started
        
        if self._status == "ready":
            return  # Already loaded
        
        self._preload_thread = threading.Thread(
            target=self._preload_worker,
            name="torch-preloader",
            daemon=True,
        )
        self._preload_thread.start()
        print("🔥 [torch_preloader] Started background preload thread")
    
    def _broadcast_status(self) -> None:
        """Broadcast status change via registered callback (fire-and-forget, thread-safe)."""
        if self._broadcast_fn is None or self._main_loop is None:
            return  # No callback registered — skip silently
        
        try:
            coro = self._broadcast_fn("torch:status", self.get_status())
            asyncio.run_coroutine_threadsafe(coro, self._main_loop)
        except Exception as e:
            # Don't fail preload if broadcast fails (e.g., no connections yet)
            print(f"⚠️ [torch_preloader] Broadcast failed (non-fatal): {e}")
    
    def _preload_worker(self) -> None:
        """Background worker that imports torch."""
        print("🔄 [torch_preloader] Loading PyTorch/ROCm...")
        self._status = "loading"
        self._broadcast_status()
        
        try:
            import torch
            
            self._torch = torch
            self._status = "ready"
            
            # Detect device
            if torch.cuda.is_available():
                self._device = "cuda"
                try:
                    self._device_name = torch.cuda.get_device_name(0)
                except Exception:
                    self._device_name = "GPU"
                print(f"✅ [torch_preloader] Torch ready: {self._device_name}")
            else:
                self._device = "cpu"
                self._device_name = "CPU"
                print("✅ [torch_preloader] Torch ready: CPU (no GPU detected)")
            
        except Exception as e:
            self._status = "unavailable"
            self._error = str(e)
            print(f"❌ [torch_preloader] Failed to load torch: {e}")
        
        # Signal completion
        self._ready_event.set()
        self._broadcast_status()
    
    def ensure_loaded(self, timeout: float = 120.0) -> bool:
        """
        Block until torch is loaded or timeout.
        
        Args:
            timeout: Max seconds to wait (default 120s for slow ROCm init)
            
        Returns:
            True if torch is ready, False if unavailable or timeout
        """
        if self._status == "ready":
            return True
        
        if self._status == "cold":
            # Start preload if not started yet
            self.start_preload()
        
        # Wait for completion
        loaded = self._ready_event.wait(timeout)
        
        if not loaded:
            print(f"⚠️ [torch_preloader] Timeout waiting for torch ({timeout}s)")
            return False
        
        return self._status == "ready"
    
    def get_torch(self) -> Any:
        """
        Get the torch module, waiting for load if needed.
        
        Raises:
            RuntimeError: If torch is unavailable
        """
        if not self.ensure_loaded():
            raise RuntimeError(f"Torch unavailable: {self._error or 'timeout'}")
        return self._torch
    
    def get_device(self) -> str:
        """Get the detected device string ("cuda", "cpu")."""
        self.ensure_loaded()
        return self._device
    
    def is_ready(self) -> bool:
        """Check if torch is loaded without blocking."""
        return self._status == "ready"


# Global singleton instance
preloader = TorchPreloader()
