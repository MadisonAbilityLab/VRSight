import asyncio
import websockets
import json
from dataclasses import dataclass, asdict
import numpy as np
import base64
# import os

@dataclass
class AudioPacket:
    x: float
    y: float
    z: float
    audio_data: str  # Base64 encoded audio data
    label: str
    asset_id: int
    # description: str
    #timestamp: float
        
class AudioStreamServer:
    def __init__(self, host="localhost", port=8765):
        self.host = host
        self.port = port
        self.clients = set()
        self.loop = None

    async def register(self, websocket):
        """Enhanced WebSocket client registration with better error handling"""
        self.clients.add(websocket)
        try:
            # Set a ping interval to detect disconnected clients
            websocket.ping_interval = 30  # (seconds)
            websocket.ping_timeout = 10   # (seconds)
            
            # Keep connection alive
            await websocket.wait_closed()
        except websockets.exceptions.ConnectionClosedError as e:
            print(f"Connection closed with error: {e}")
        except websockets.exceptions.ConnectionClosedOK:
            print("Connection closed normally")
        except Exception as e:
            print(f"Unexpected error in WebSocket connection: {e}")
        finally:
            self.clients.remove(websocket)

    async def broadcast_audio_packet(self, packet):
        """Improved broadcast with better error handling and timeout protection"""
        if not self.clients:
            return
        
        # Create message
        try:
            message = json.dumps(asdict(packet))
        except Exception as e:
            print(f"Error serializing packet: {e}")
            return
        
        # Create a copy of clients to avoid issues if the set changes during iteration
        clients_copy = set(self.clients)
        
        # Send to each client with individual error handling and timeout
        failed_clients = set()
        for client in clients_copy:
            try:
                # Use asyncio.wait_for with a shorter timeout (0.5 seconds instead of 1.0)
                await asyncio.wait_for(client.send(message), timeout=0.5)
            except (asyncio.TimeoutError, websockets.exceptions.WebSocketException) as e:
                print(f"Error sending to client: {e}")
                failed_clients.add(client)
            except Exception as e:
                print(f"Unexpected error broadcasting to client: {e}")
                failed_clients.add(client)
        
        # Remove failed clients
        self.clients.difference_update(failed_clients)
        if failed_clients:
            print(f"Removed {len(failed_clients)} disconnected WebSocket clients")

    def schedule_broadcast(self, coro):
        """
        Schedules an async coroutine on the server's loop 
        so that it executes in the correct thread/event-loop.
        """
        if self.loop is None:
            raise RuntimeError("Server loop is not running yet.")
        # Use call_soon_threadsafe to schedule the coroutine on the server loop
        self.loop.call_soon_threadsafe(asyncio.create_task, coro)

    async def _run_websocket(self):
        """Main coroutine to handle websocket connections."""
        async with websockets.serve(self.register, self.host, self.port):
            print(f"AudioStreamServer started on ws://{self.host}:{self.port}")
            await asyncio.Future()  # Run forever

    def start_server(self):
        """Sets up the server in the current thread; call this in a dedicated thread."""
        # Each thread needs its own event loop
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        try:
            self.loop.run_until_complete(self._run_websocket())
        finally:
            try:
                # Cancel all running tasks
                tasks = asyncio.all_tasks(self.loop)
                for task in tasks:
                    task.cancel()
                # Run the event loop one last time to complete cancellation
                self.loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))
                # Finally close the loop
                self.loop.close()
            except Exception as e:
                print(f"Error during cleanup: {e}")
    
    def shutdown(self):
        """
        Properly shut down the WebSocket server by closing connections
        and stopping the event loop.
        """
        print("Shutting down AudioStreamServer...")
        
        if self.loop is None:
            print("No event loop to shut down")
            return
        
        try:
            # Close all client connections
            close_tasks = []
            for client in self.clients.copy():
                try:
                    # Schedule close tasks
                    close_tasks.append(self.loop.create_task(client.close()))
                except Exception as e:
                    print(f"Error closing client connection: {e}")
            
            # Clear clients set
            self.clients.clear()
            
            # Create shutdown task
            shutdown_task = self.loop.create_task(self._shutdown())
            
            # Schedule shutdown in the loop
            if self.loop.is_running():
                self.loop.call_soon_threadsafe(lambda: None)  # Wake up the loop
        except Exception as e:
            print(f"Error during AudioStreamServer shutdown: {e}")
        
    async def _shutdown(self):
        """
        Internal method for shutdown sequence
        """
        # Wait a little to allow connections to close
        await asyncio.sleep(0.5)
        
        # Get all tasks
        tasks = [t for t in asyncio.all_tasks(self.loop) if t is not asyncio.current_task()]
        
        # Cancel all tasks
        for task in tasks:
            task.cancel()
        
        # Wait for tasks to complete cancellation
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        
        # Stop the loop
        self.loop.stop()
