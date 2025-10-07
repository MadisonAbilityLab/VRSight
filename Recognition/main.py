from dotenv import load_dotenv
import argparse
import numpy as np
import sys
import torch
import time
import cv2
import threading
import queue
from imutils.video import VideoStream

# Import new modular engines
from settings import (
    THREAD_HEARTBEAT_TIMEOUT, THREAD_ERROR_THRESHOLD,
    MEMORY_CLEANUP_THRESHOLD_MB, MEMORY_CLEANUP_INTERVAL,
    WEBCAM_INDEX, DEVICE, COOLDOWNS, SPECIAL_LABELS, EDGE_LABELS
)
from object_detection_engine import ObjectDetectionEngine, run_object_detection_thread
from depth_detection_engine import DepthDetectionEngine, run_depth_detection_thread
from edge_detection_engine import EdgeDetectionEngine, run_edge_detection_thread
from geometry_utils import (
    calculate_box_midpoint, check_location_near_bbox, check_bbox_overlap,
    calculate_distance, expand_bbox, clamp_bbox_to_frame, calculate_bbox_area,
    point_to_line_distance, line_intersects_bbox
)
from scene_sweep_processor import process_scenesweep
from audio_utils import (
    create_speech_function, create_centered_speech_function,
    broadcast_immediate_centered_audio, play_user_command_feedback
)
from aim_assist_processor import process_command_1_2_combo
from aim_assist_menu_pilot_processor import process_aim_assist_menu_pilot
from interaction_detection import (
    enhance_check_line_location, check_line_location, check_hand_proximity,
    ray_bbox_intersect
)
from thread_manager import get_thread_manager
from memory_manager import get_memory_manager
from config import get_config_manager
from label_handler_framework import handle_label, background_processor, initialize_handler
from server import AudioStreamServer
import websockets
import json
import gpt_functions
from server import AudioPacket
from tts_engine import synthesize_speech, prepare_tts_text
from speech_cooldown_manager import CategoryCooldownManager
from vr_interaction_debug import DebugVisualizer, InteractionDebugVisualizer, add_debug_arguments
import math
from ocr_functions import perform_ocr_on_frame
from simple_logger import log_error, log_info
from env_validator import validate_environment
from model_validator import validate_model_files
from performance_monitor import update_frame_count, report_performance
from label_handle_functions import calc_box_location
import logging
import os

# load environment vars
load_dotenv()

# Modernized thread management using new managers
thread_manager = get_thread_manager()
memory_manager = get_memory_manager()

# Setup memory optimizations
memory_manager.optimize_for_performance()

# Legacy compatibility - using new memory manager
class LegacyMemoryMonitor:
    """Legacy wrapper for backward compatibility"""
    def __init__(self, cleanup_threshold_mb=MEMORY_CLEANUP_THRESHOLD_MB):
        pass  # Configuration handled by memory_manager

    def check_and_cleanup(self, force=False):
        return memory_manager.check_and_cleanup(force=force)

    def clear_stale_queues(self, queues):
        for queue_name, queue_obj in queues.items():
            try:
                while not queue_obj.empty():
                    queue_obj.get_nowait()
            except Exception:
                pass

# Default cooldowns moved to config.py as COOLDOWNS
# edge_labels moved to config.py as EDGE_LABELS

# WebSocket sender coroutine
async def send_ocr_result_via_websocket(text, websocket_url="ws://localhost:8765"):
    try:
        async with websockets.connect(websocket_url) as websocket:
            message = {"type": "ocr_text", "text": text, "timestamp": time.time()}
            await websocket.send(json.dumps(message))
            # OCR result sent to WebSocket
    except Exception as e:
        print(f"WebSocket error: {e}")

# Global instances for detection engines
object_engine = None
depth_engine = None
edge_engine = None


logging.basicConfig(
    filename='key_press.log',
    level=logging.INFO,  # Log all levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format='%(asctime)s - %(message)s'
)

# Initialize resources in thread manager
resource_manager = thread_manager.resource_manager

# Initialize interaction detection with resource manager
# Note: Depth map access now uses resource_manager.copy_resource('depth_map')
descriptions_prev_selection = ""

# Use managed queues from thread manager
object_detection_queue = thread_manager.get_queue('object_detection')
depth_detection_queue = thread_manager.get_queue('depth_detection')
edge_detection_queue = thread_manager.get_queue('edge_detection')
gpt_request_queue = thread_manager.get_queue('gpt_request')

# Legacy compatibility
thread_coordinator = thread_manager
memory_monitor = memory_manager

# set torch options
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

def list_available_cameras(max_cameras=5):
    """
    Lists available camera indices by attempting to open them.
    """
    available_cameras = []
    for index in range(max_cameras):
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            available_cameras.append(index)
            cap.release()
    return available_cameras

def select_camera():
    """
    Prompts the user to select a camera index from the available cameras.
    """
    available_cameras = list_available_cameras()
    # Available cameras detected
    if not available_cameras:
        print("No cameras found.")
        exit()
    # Available cameras listed
    for idx in available_cameras:
        print(f"Camera {idx}: Available")
    while True:
        try:
            selected_index = int(input("Select the camera index to use: "))
            if selected_index in available_cameras:
                return selected_index
            else:
                print("Invalid selection. Please choose from the available cameras.")
        except ValueError:
            print("Please enter a valid integer.")

# Prompt the user to select a webcam index
# WEBCAM_INDEX = select_camera()
# WEBCAM_INDEX moved to config.py

# Initialize the video stream with the selected webcam index
raw_video = None  # Will be initialized in initialize_system()

def frame_reader():
    """Optimized frame reader using new buffering system"""
    thread_coordinator.register_thread("frame_reader", is_critical=True)

    retry_count = 0
    max_retries = 5
    frame_buffer = memory_manager.get_frame_buffer('raw_frames')

    while not thread_coordinator.should_shutdown():
        try:
            thread_coordinator.heartbeat("frame_reader")

            global raw_video
            if raw_video is None:
                time.sleep(0.5)
                continue

            new_frame = raw_video.read()

            if new_frame is None:
                retry_count += 1
                if retry_count > max_retries:
                    print("WARNING: Multiple failed frame reads. Reinitializing video stream...")
                    try:
                        global WEBCAM_INDEX
                        new_video = VideoStream(src=WEBCAM_INDEX).start()
                        time.sleep(1)  # Allow camera to warm up
                        test_frame = new_video.read()
                        if test_frame is not None:
                            # Video stream reinitialized
                            raw_video.stop()
                            raw_video = new_video
                            retry_count = 0
                    except Exception as e:
                        print(f"Failed to reinitialize video stream: {e}")
                time.sleep(0.1)
                continue

            retry_count = 0

            frame_buffer.add_frame(new_frame)
            resource_manager.update_resource('frame', new_frame.copy())

            # Periodic memory check
            if int(time.time()) % 30 == 0:  # Every 30 seconds
                memory_manager.check_and_cleanup()

        except Exception as e:
            thread_coordinator.increment_error("frame_reader")
            print(f"ERROR in frame reader: {e}")
            time.sleep(0.5)

parser = argparse.ArgumentParser()
parser.add_argument('--encoder', type=str, default='vits', choices=['vits', 'vitb', 'vitl'])
parser.add_argument('--pred-only')
# parser.add_argument('--debug', action='store_true', help='Enable debug mode')

# Add debug mode arguments
add_debug_arguments(parser)

args = parser.parse_args()

# Initialize debug visualization system
def initialize_debug_system(args=None):
    """
    Initialize the interaction debugging system with optional args.
    
    Args:
        args: Parsed command-line arguments (optional)
    
    Returns:
        InteractionDebugVisualizer: Configured debug visualization system
    """
    # Default to 32abled debug if no args provided
    enable_debug = False
    
    if args is not None and hasattr(args, 'debug'):
        enable_debug = args.debug
    
    # Create debug visualizer
    debug_visualizer = InteractionDebugVisualizer(enable_debug)
    
    # Parse command-line args if provided
    if args is not None:
        debug_visualizer.parse_debug_arguments(args)
    
    # Force window creation for active modes (critical for Issue C fix)
    if debug_visualizer.debug_mode:
        for mode, window in debug_visualizer.debug_windows.items():
            if window['enabled'] and not debug_visualizer.windows_created.get(mode, False):
                # Create window
                win_name = window['name']
                cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
                cv2.moveWindow(win_name, 
                               debug_visualizer.window_positions[mode][0], 
                               debug_visualizer.window_positions[mode][1])
                cv2.resizeWindow(win_name, 640, 480)
                debug_visualizer.windows_created[mode] = True
                
                # Create initial blank frame
                blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(blank_frame, f"Initializing {mode} debug view...", 
                            (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                window['frame'] = blank_frame
                cv2.imshow(win_name, blank_frame)
                cv2.waitKey(1)  # Critical to actually show the window
                
                # Debug window created
    
    return debug_visualizer

debug_visualizer = initialize_debug_system(args)
# Debug visualization system initialized

# Device detection and model configurations moved to engine modules
# Device configuration loaded

def initialize_system():
    """Initialize all system components using new modular engines"""
    components = {}
    initialization_errors = []

    # Initialize object detection engine
    try:
        # Initialize object detection engine
        global object_engine
        object_engine = ObjectDetectionEngine()
        if object_engine.initialize_model():
            components["object_engine"] = object_engine
            components["yolo_model"] = object_engine  # Add for critical check
            # Object detection engine ready
        else:
            raise Exception("Failed to initialize object detection model")
    except Exception as e:
        error_msg = f"Object detection initialization error: {e}"
        initialization_errors.append(error_msg)
        log_error(error_msg)
        components["object_engine"] = None
        components["yolo_model"] = None

    # Initialize depth detection engine
    try:
        # Initialize depth detection engine
        global depth_engine
        depth_engine = DepthDetectionEngine()
        if depth_engine.initialize_model():
            components["depth_engine"] = depth_engine
            components["depth_model"] = depth_engine  # Add for critical check
            # Depth detection engine ready
        else:
            raise Exception("Failed to initialize depth detection model")
    except Exception as e:
        error_msg = f"Depth detection initialization error: {e}"
        initialization_errors.append(error_msg)
        log_error(error_msg)
        components["depth_engine"] = None
        components["depth_model"] = None

    # Initialize edge detection engine
    try:
        # Initialize edge detection engine
        global edge_engine
        edge_engine = EdgeDetectionEngine()
        components["edge_engine"] = edge_engine
        # Edge detection engine ready
    except Exception as e:
        initialization_errors.append(f"Edge detection initialization error: {e}")
        components["edge_engine"] = None
    
    # Initialize video stream
    try:
        # Initialize video stream
        global raw_video
        raw_video = VideoStream(src=WEBCAM_INDEX).start()
        time.sleep(1)  # Allow camera to warm up
        
        # Check if video stream is working
        test_frame = raw_video.read()
        if test_frame is None:
            raise Exception("Could not read from video stream")
            
        components["video_stream"] = raw_video
        # Video stream ready
    except Exception as e:
        initialization_errors.append(f"Video stream initialization error: {e}")
        components["video_stream"] = None
    
    # Initialize audio server
    try:
        # Initialize audio server
        audio_server = AudioStreamServer(host="localhost", port=8765)
        components["audio_server"] = audio_server
        # Audio server ready
    except Exception as e:
        initialization_errors.append(f"Audio server initialization error: {e}")
        components["audio_server"] = None
    
    # Check if critical components were initialized
    critical_failure = False
    for component_name in ["yolo_model", "depth_model", "video_stream"]:
        if components.get(component_name) is None:
            critical_failure = True
            print(f"CRITICAL: {component_name} initialization failed")
    
    # Report initialization status
    if initialization_errors:
        print("Initialization completed with errors:")
        for error in initialization_errors:
            print(f"  - {error}")
    # else:
    #     # System initialization completed
    #     os.system('cls')
    
    return components, critical_failure

# Replace the cleanup_system function in main.py with this improved version

def cleanup_system(components):
    """
    Thoroughly clean up all system resources to ensure proper termination.
    Force exits if clean shutdown fails.
    """
    print("Cleaning up system resources...")
    
    # Set global running flag to False to signal threads to exit
    global running
    running = False
    
    # Phase 1: Signal all threads to exit gracefully
    try:
        # Shutdown background processor first
        if 'background_processor' in globals() and background_processor is not None:
            background_processor.shutdown()
            # Background processor shutdown requested
    except Exception as e:
        print(f"Error shutting down background processor: {e}")
    
    if "audio_server" in components and components["audio_server"] is not None:
        try:
            # Shutting down audio server
            components["audio_server"].shutdown()
        except Exception as e:
            print(f"Error shutting down audio server: {e}")
    
    # Phase 2: Clean up queues
    try:
        queues = {
            "object_detection": object_detection_queue,
            "depth_detection": depth_detection_queue,
            "edge_detection": edge_detection_queue,
            "gpt_request": gpt_request_queue
        }
        for name, queue_obj in queues.items():
            try:
                # Clearing queue
                while not queue_obj.empty():
                    try:
                        queue_obj.get_nowait()
                    except:
                        break
            except Exception as e:
                print(f"Error clearing {name} queue: {e}")
    except Exception as e:
        print(f"Error in queue cleanup: {e}")
    
    # Phase 3: Signal threads to exit and wait briefly
    try:
        # Signal the GPT worker thread to exit with sentinel
        gpt_request_queue.put(None)
        
        # Brief wait to allow threads to process exit signals
        # Waiting for threads to exit gracefully
        time.sleep(2)
    except Exception as e:
        print(f"Error signaling thread exit: {e}")
    
    # Phase 4: Resource cleanup
    try:
        # Clear CUDA cache first if available
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                # CUDA cache cleared
            except Exception as e:
                print(f"Error clearing CUDA cache: {e}")
                
        # Close OpenCV windows
        try:
            cv2.destroyAllWindows()
            # Force window destruction with extra calls
            for _ in range(5):
                cv2.waitKey(1)
            # OpenCV windows closed
        except Exception as e:
            print(f"Error closing OpenCV windows: {e}")
            
        # Stop video stream
        if "video_stream" in components and components["video_stream"] is not None:
            try:
                components["video_stream"].stop()
                # Video stream stopped
            except Exception as e:
                print(f"Error stopping video stream: {e}")
    except Exception as e:
        print(f"Error in resource cleanup: {e}")
    
    # Phase 5: Final thread termination - wait with timeout
    # Final thread termination
    
    # Phase 6: Resource cleanup verification
    try:
        # Track whether any non-daemon threads are still running
        non_daemon_threads = []
        for thread in threading.enumerate():
            if not thread.daemon and thread is not threading.current_thread():
                non_daemon_threads.append(thread.name)

        if non_daemon_threads:
            log_info(f"Waiting for {len(non_daemon_threads)} threads to exit")
            # Wait up to 5 seconds for remaining threads
            timeout = 5.0
            start_time = time.time()
        while time.time() - start_time < timeout:
            remaining = []
            for thread in threading.enumerate():
                if not thread.daemon and thread is not threading.current_thread():
                    remaining.append(thread.name)
            
            if not remaining:
                break
                
            time.sleep(0.5)
            
        still_running = []
        for thread in threading.enumerate():
            if not thread.daemon and thread is not threading.current_thread():
                still_running.append(thread.name)
                
        if still_running:
            warning_msg = f"WARNING: These non-daemon threads did not exit: {still_running}"
            print(warning_msg)
            log_error(warning_msg)
            print("Forcing exit with os._exit(0)")
        else:
            log_info("All non-daemon threads exited cleanly")
    except Exception as e:
        log_error(f"Error in resource cleanup verification: {e}")
        # Use os._exit for hard exit that will terminate all threads
        import os
        os._exit(0)
    
    # System cleanup completed

# Object detection functionality moved to object_detection_engine.py

# Depth detection functionality moved to depth_detection_engine.py

def safe_depth_diff(depth1, depth2):
    """
    Calculate depth difference with protection against overflow and NaN values.
    
    Args:
        depth1, depth2: Depth values to compare
        
    Returns:
        float: Absolute difference between depths, or a large value if calculation fails
    """
    try:
        # Check for None or invalid values
        if depth1 is None or depth2 is None:
            return float('inf')
            
        # Convert to float to avoid integer overflow
        d1 = float(depth1)
        d2 = float(depth2)
        
        # Check for NaN or infinity
        if math.isnan(d1) or math.isnan(d2) or math.isinf(d1) or math.isinf(d2):
            return float('inf')
            
        # Calculate difference with bounds checking
        if abs(d1) > 1e6 or abs(d2) > 1e6:  # Unrealistically large values
            return float('inf')
            
        return abs(d1 - d2)
    except Exception as e:
        print(f"Error calculating depth difference: {e}")
        return float('inf')

# Edge detection functionality moved to edge_detection_engine.py

def enhance_check_line_location(objects, lines, frame_copy, current_depth_map, audio_server, debug_frame=None):
    """
    Enhanced line interaction detection that processes all detected lines
    to identify objects being pointed at, prioritizing from hands/controllers.
    """
    if objects is None or not objects:
        return None
        
    if lines is None or len(lines) == 0:
        return None
        
    try:
        # Find hands and controllers
        hand_objects = [obj for obj in objects if obj is not None and obj.get('label') in EDGE_LABELS]
        
        # Find interaction-relevant objects (targets)
        interaction_objects = [obj for obj in objects if obj is not None and obj.get('label') in 
                              ['button', 'interactable', 'portal', 'menu', 
                               'sign-text', 'ui-text', 'sign-graphic', 'ui-graphic',
                               'progress bar']]
                               
        # Return early if no interaction objects
        if not interaction_objects:
            return None
        
        # Get depth for all objects - with defensive error handling
        for obj in objects:
            if obj is None or 'bbox' not in obj:
                continue
                
            bbox = obj.get('bbox')
            if len(bbox) != 4:
                continue
                
            mx, my = calculate_box_midpoint(bbox)
            
            # Ensure coordinates are within bounds
            if current_depth_map is not None:
                h, w = current_depth_map.shape[:2]
                if 0 <= my < h and 0 <= mx < w:
                    obj['depth'] = int(current_depth_map[my, mx])
                else:
                    obj['depth'] = None
            else:
                obj['depth'] = None
        
        # Process all lines with the objects
        results = []
        
        # First, process lines from hands/controllers
        for hand in hand_objects:
            if hand is None or 'bbox' not in hand:
                continue
                
            hand_bbox = hand.get('bbox')
            hand_depth = hand.get('depth')
            hand_label = hand.get('label', '')
            h_mx, h_my = calculate_box_midpoint(hand_bbox)
            
            # Find lines that start from this hand/controller
            hand_lines = []
            for line in lines:
                if line is None:
                    continue
                    
                # Get line coordinates - handling different structures
                # Sometimes lines come as [[[x1, y1, x2, y2]]], sometimes as [[x1, y1, x2, y2]]
                if len(line) == 1 and isinstance(line[0], (list, np.ndarray)):
                    line_coords = line[0]
                else:
                    line_coords = line
                    
                if len(line_coords) != 4:
                    continue
                    
                x1, y1, x2, y2 = line_coords
                
                # Check if either end of the line is near the hand
                if check_location_near_bbox(hand_bbox, x1, y1, 15) or check_location_near_bbox(hand_bbox, x2, y2, 15):
                    # Determine direction - ensure x1,y1 is at the hand end
                    if check_location_near_bbox(hand_bbox, x2, y2, 15):
                        # Swap coordinates to make hand the starting point
                        x1, y1, x2, y2 = x2, y2, x1, y1
                    
                    # Calculate line length
                    line_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                    
                    # Only consider lines of reasonable length
                    if line_length > 10:
                        hand_lines.append({
                            'line': [x1, y1, x2, y2],
                            'hand': hand,
                            'distance': line_length
                        })
            
            # Skip if no valid hand lines found
            if not hand_lines:
                continue
                
            # Sort hand lines by distance (longer lines first)
            hand_lines.sort(key=lambda x: -x['distance'])
            
            # Process each hand line to find pointing targets
            for hand_line in hand_lines[:3]:  # Only check top 3 lines per hand
                x1, y1, x2, y2 = hand_line['line']
                
                # Calculate normalized direction vector
                line_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                if line_length > 0:
                    dx, dy = (x2 - x1) / line_length, (y2 - y1) / line_length
                    
                    # Extend the line for better detection
                    extension_factor = min(100, line_length * 1.5)  # Extend by 50% or max 100px
                    extended_x = int(x2 + dx * extension_factor)
                    extended_y = int(y2 + dy * extension_factor)
                    
                    # Find objects this line might intersect with
                    best_hit_object = None
                    best_hit_distance = float('inf')
                    
                    for obj in interaction_objects:
                        if obj is None or 'bbox' not in obj:
                            continue
                            
                        bbox = obj.get('bbox')
                        if len(bbox) != 4:
                            continue
                            
                        # Check for ray intersection with bbox
                        hit, distance = ray_bbox_intersect(x2, y2, dx, dy, bbox)
                        
                        # Track if this is the closest hit
                        if hit and distance < best_hit_distance and distance < 200:  # Limit to reasonable distance
                            best_hit_object = obj
                            best_hit_distance = distance
                    
                    # If we found a target, check if it's within reach
                    if best_hit_object is not None:
                        obj_depth = best_hit_object.get('depth')
                        max_reach = 15.0  # Maximum reach distance
                        
                        # Check depth constraints if both hand and object depths are available
                        depth_valid = True
                        if hand_depth is not None and obj_depth is not None:
                            depth_diff = abs(hand_depth - obj_depth)
                            depth_valid = depth_diff <= max_reach
                            best_hit_object['depth_valid'] = depth_valid
                        
                        # Process if within reach or depth info not available
                        if depth_valid:
                            # Handle interaction - calculate world coordinates for spatial audio
                            obj_mx, obj_my = calculate_box_midpoint(best_hit_object.get('bbox'))
                            obj_depth = best_hit_object.get('depth', 128)  # Default to mid-depth if missing
                            
                            # Calculate world coordinates for the object
                            world_x, world_y, world_z = calc_box_location(obj_mx, obj_my, obj_depth)
                            
                            # Use world coordinates for handling
                            handle_detected_object(best_hit_object, world_x, world_y, world_z, audio_server)
                            
                            # Store the result
                            results.append(best_hit_object)
                            
                            # Only process the first valid target
                            break
        
        return results if results else None
            
    except Exception as e:
        print(f"Error in enhance_check_line_location: {e}")
        import traceback
        traceback.print_exc()
        
        if debug_vis:
            return debug_vis.get_frame()
        
        return None


audio_server = None  # Will be initialized in initialize_system()

def start_audio_stream_server():
    # This will block in the thread, so we put it in its own function.
    thread_coordinator.register_thread("audio_server")
    # Audio server thread started
    
    try:
        audio_server.start_server()
    except Exception as e:
        thread_coordinator.increment_error("audio_server")
        print(f"ERROR in audio stream server: {e}")
        
def grab_results(results, frame):
    """
    Extracts detections from YOLO results.

    Args:
        results (list): YOLO model output.
        frame: the entire frame. 

    Returns:
        list: List of detection dictionaries containing bbox, confidence, class_id, and label.
    """
    detections = []
    try:
        for result in results:
            for detection in result.boxes:
                try:
                    x1,y1,x2,y2 = map(int, detection.xyxy[0].tolist())
                    
                    # Ensure coordinates are within frame boundaries
                    x1 = max(0, min(x1, frame.shape[1]-1))
                    y1 = max(0, min(y1, frame.shape[0]-1))
                    x2 = max(0, min(x2, frame.shape[1]-1))
                    y2 = max(0, min(y2, frame.shape[0]-1))
                    
                    # Skip invalid boxes
                    if x1 >= x2 or y1 >= y2:
                        continue
                        
                    confidence = float(detection.conf)
                    class_id = int(detection.cls)
                    label = result.names[class_id]

                    # Extract the pixel values for the bounding box region
                    object_pixels = frame[y1:y2, x1:x2]

                    detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'confidence': confidence,
                        'class_id': class_id,
                        'label': label,
                        'pixels': object_pixels
                    })
                except Exception as e:
                    print(f"Error processing individual detection: {e}")
                    continue
    except Exception as e:
        print(f"ERROR in grab_results: {e}")
    
    return detections


def run_periodic_tone_detection():
    global last_auto_detection_time
    thread_coordinator.register_thread("tone_detection")
    # Tone detection thread started
    
    while running:
        try:
            thread_coordinator.heartbeat("tone_detection")
            
            current_time = time.time()
            if (current_time - last_auto_detection_time) >= 60 or last_auto_detection_time == 0:
                frame_copy = resource_manager.copy_resource('frame')

                if frame_copy is not None:
                    # Instead of calling ask_gpt_about_tone here, we enqueue the task
                    gpt_request_queue.put({
                        "type": "tone_detection",
                        "frame": frame_copy
                    })
                    last_auto_detection_time = current_time

            time.sleep(1)  # check once per second
        except Exception as e:
            thread_coordinator.increment_error("tone_detection")
            print(f"ERROR in tone detection thread: {e}")
            time.sleep(5)  # Wait longer after an error

gpt_request_queue = queue.Queue()
def gpt_worker():
    """
    Dedicated thread to handle all GPT calls in the background,
    preventing them from blocking other threads.
    
    Enhanced to queue speech output and support callbacks.
    """
    thread_coordinator.register_thread("gpt_worker")
    # GPT worker thread started
    
    while running:
        try:
            thread_coordinator.heartbeat("gpt_worker")
            
            # Wait for tasks with timeout to allow for graceful shutdown
            try:
                task = gpt_request_queue.get(timeout=1.0)
            except queue.Empty:
                continue
                
            if task is None:
                # None is a sentinel indicating we should exit
                break

            task_type = task.get("type")
            callback = task.get("callback")  # Optional callback function
            
            try:
                if task_type == "tone_detection":
                    frame_data = task.get("frame")
                    if frame_data is not None:
                        tone = gpt_functions.ask_gpt_about_tone(frame_data)
                        # Tone detection completed
                        
                        # Call callback if provided
                        if callback and callable(callback):
                            callback(tone)

                elif task_type == "frame_description":
                    frame_data = task.get("frame")
                    detections = task.get("detections", [])
                    if frame_data is not None:
                        # Process GPT request for frame description
                        description = gpt_functions.ask_gpt_about_frame(detections, frame_data)
                        # Frame description generated
                        
                        # Split long descriptions into manageable chunks
                        if description:
                            # Process the description
                            def split_into_chunks(text, max_length=150):
                                """Split text into chunks for TTS processing"""
                                # Attempt to split at sentence endings
                                sentences = []
                                start = 0
                                for i in range(len(text)):
                                    if text[i] in '.!?' and (i+1 == len(text) or text[i+1] == ' '):
                                        sentences.append(text[start:i+1].strip())
                                        start = i+1
                                
                                # Add any remaining text
                                if start < len(text):
                                    sentences.append(text[start:].strip())
                                
                                # Combine sentences into chunks
                                chunks = []
                                current_chunk = ""
                                for sentence in sentences:
                                    if len(current_chunk) + len(sentence) <= max_length:
                                        current_chunk += " " + sentence if current_chunk else sentence
                                    else:
                                        if current_chunk:
                                            chunks.append(current_chunk)
                                        current_chunk = sentence
                                
                                if current_chunk:
                                    chunks.append(current_chunk)
                                
                                return chunks
                            
                            # Split into chunks
                            chunks = split_into_chunks(description)
                            
                            # Process each chunk sequentially
                            for chunk_idx, chunk in enumerate(chunks):
                                audio_b64 = synthesize_speech(chunk, gpt_functions.get_current_emotion())
                                
                                # Skip if synthesis failed
                                if audio_b64 is None:
                                    print(f"Warning: Failed to synthesize chunk {chunk_idx+1}")
                                    continue
                                
                                if audio_server is not None:
                                    packet = AudioPacket(
                                        x=0.0,  # Center position
                                        y=2.5,
                                        z=30.0,
                                        audio_data=audio_b64,
                                        label=f"scene_description_part{chunk_idx+1}",
                                        asset_id=-1
                                    )
                                    
                                    # Wait for previous chunk to finish (estimated)
                                    if chunk_idx > 0:
                                        # Rough estimate: 100ms per character
                                        prev_chunk_length = len(chunks[chunk_idx-1])
                                        wait_time = min(max(0.5, prev_chunk_length * 0.05), 5.0)  # Between 0.5 and 7 seconds
                                        time.sleep(wait_time)
                                    
                                    try:
                                        # Send the audio packet
                                        audio_server.schedule_broadcast(audio_server.broadcast_audio_packet(packet))
                                        
                                        # Wait for the current chunk to be processed
                                        estimated_duration = len(chunk) * 0.05  # Rough estimate
                                        wait_time = min(max(0.5, estimated_duration), 5.0)  # Between 0.5 and 7 seconds
                                        time.sleep(wait_time)
                                    except Exception as send_error:
                                        print(f"Error sending description chunk {chunk_idx+1}: {send_error}")
                                else:
                                    print(f"Warning: Audio server not available for chunk {chunk_idx+1}")
                            
                        else:
                            # No description text returned from GPT
                            pass
                        
                        # Call callback if provided
                        if callback and callable(callback):
                            callback(description)
                    else:
                        print("No frame data provided for frame description")

            except Exception as e:
                thread_coordinator.increment_error("gpt_worker")
                print(f"[GPT Worker] Exception while processing {task_type} task:", e)
                import traceback
                traceback.print_exc()
                
                # Still call callback with error result if provided
                if callback and callable(callback):
                    callback(f"Error processing {task_type}")

            # Mark task as done
            gpt_request_queue.task_done()
        except Exception as e:
            thread_coordinator.increment_error("gpt_worker")
            print(f"ERROR in GPT worker thread: {e}")
            time.sleep(1)


def assign_nametags_to_avatars(avatar_detections, nametag_detections):
    """
    Assign nametags to closest avatars and return list of avatars without nametags
    """
    try:
        assigned_avatars = set()
        avatars_without_nametags = []

        for nametag in nametag_detections:
            nametag_bbox = nametag.get('bbox')
            if nametag_bbox is None:
                continue
                
            closest_avatar = None
            min_distance = float('inf')

            for avatar in avatar_detections:
                avatar_bbox = avatar.get('bbox')
                if avatar_bbox is None:
                    continue
                    
                distance = calculate_distance(nametag_bbox, avatar_bbox)
                if distance < min_distance and avatar not in assigned_avatars:
                    min_distance = distance
                    closest_avatar = avatar

            if closest_avatar:
                # assign nametag to the closest avatar.
                assigned_avatars.add(closest_avatar)
                closest_avatar['nametag'] = nametag

        # Find avatars without nametags
        for avatar in avatar_detections:
            if avatar is not None and avatar not in assigned_avatars:
                avatars_without_nametags.append(avatar)

        return avatars_without_nametags
    except Exception as e:
        print(f"Error in assign_nametags_to_avatars: {e}")
        return []

def assign_signs_to_portals(portal_detections, sign_text_detections):
    """
    finds the closest sign-text detection by comparing
    the midpoints of the bounding boxes. Modifies each portal detection
    in place by adding an 'associated_sign_text' key if one is found.
    """
    try:
        for _, portal in portal_detections:
            if portal is None or 'bbox' not in portal:
                continue
                
            portal_center = calculate_box_midpoint(portal['bbox'])
            best_distance = float('inf')
            associated_sign = None
            
            for _, sign in sign_text_detections:
                if sign is None or 'bbox' not in sign:
                    continue
                    
                sign_center = calculate_box_midpoint(sign['bbox'])
                distance = np.sqrt((portal_center[0] - sign_center[0]) ** 2 + (portal_center[1] - sign_center[1]) ** 2)
                if distance < best_distance:
                    best_distance = distance
                    associated_sign = sign
                    
            if associated_sign is not None:
                portal['associated_sign_text'] = associated_sign
    except Exception as e:
        print(f"Error in assign_signs_to_portals: {e}")

# Functions to integrate with main.py

def check_line_location(objects, x1, y1, x2, y2, hand_bbox=[], hand_label="", audio_server=None, debug_frame=None):
    """
    Enhanced check for line interactions with improved ray casting, depth validation, and priority handling.
    
    Args:
        objects: List of objects in the scene
        x1, y1, x2, y2: Coordinates of the line
        hand_bbox: Bounding box of the hand/controller
        hand_label: Label of the hand/controller
        audio_server: Server for sending audio packets
        debug_frame: Optional frame for drawing debug visualizations
    
    Returns:
        Debug visualization frame if debug_frame is provided, otherwise None
    """
    try:
        # Skip if no objects
        if objects is None or len(objects) == 0:
            return None
        
        # Initialize debug visualization if a frame is provided
        debug_vis = None
        if debug_frame is not None:
            from vr_interaction_debug import DebugVisualizer
            debug_vis = DebugVisualizer(debug_frame)
            # Draw the original line
            debug_vis.draw_line(x1, y1, x2, y2, (0, 255, 0), 2)
            if hand_bbox:
                debug_vis.draw_box(hand_bbox, (0, 255, 0), 2, f"{hand_label}")
            
        # Basic parameters for detection
        threshold = 10  # Pixel proximity threshold
        max_reach = 15.0  # Maximum reach distance in world units
        extended_distance_threshold = 60  # Max pixel distance for ray casting
        
        # Get hand depth for reach calculation
        hand_z = None
        hand_mx, hand_my = None, None
        if hand_bbox:
            hand_mx, hand_my = calculate_box_midpoint(hand_bbox)
            depth_map = resource_manager.copy_resource('depth_map')
            if depth_map is not None and 0 <= hand_my < depth_map.shape[0] and 0 <= hand_mx < depth_map.shape[1]:
                hand_z = depth_map[hand_my, hand_mx]
            
            if debug_vis:
                debug_vis.add_text(f"Hand depth: {hand_z}", 10, 30)
        
        # Prioritize important interactive elements (lower number = higher priority)
        priority_labels = ['menu', 'button', 'interactable', 'portal', 'spawner', 'target']
        
        # Pre-filter objects by priority labels for efficiency
        priority_objects = {}
        for target_label in priority_labels:
            priority_objects[target_label] = [
                o for o in objects 
                if o is not None and o.get('label') == target_label
            ]
            
        # Draw all objects in debug frame with their depth values
        if debug_vis:
            debug_vis.add_text(f"Total objects: {len(objects)}", 10, 50)
            
            y_pos = 70
            for target_label in priority_labels:
                obj_count = len(priority_objects.get(target_label, []))
                if obj_count > 0:
                    debug_vis.add_text(f"{target_label}: {obj_count}", 10, y_pos)
                    y_pos += 20
                    
            # Draw all candidate objects with their depth
            for obj in objects:
                if obj is None or 'bbox' not in obj:
                    continue
                    
                bbox = obj.get('bbox')
                label = obj.get('label', 'unknown')
                mx, my = calculate_box_midpoint(bbox)
                
                # Get depth
                obj_z = None
                depth_map = resource_manager.copy_resource('depth_map')
                if depth_map is not None and 0 <= my < depth_map.shape[0] and 0 <= mx < depth_map.shape[1]:
                    obj_z = depth_map[my, mx]
                
                # Store depth in object for later use
                obj['depth'] = obj_z
                
                # Color based on depth difference
                color = (255, 255, 255)  # White for objects without depth
                if hand_z is not None and obj_z is not None:
                    depth_diff = safe_depth_diff(hand_z, obj_z)
                    if depth_diff <= max_reach:
                        # Green for objects within reach
                        color = (0, 255, 0)
                    else:
                        # Red for objects out of reach
                        # Brighter red for closer objects
                        red_intensity = max(50, min(255, 255 - (depth_diff - max_reach) * 10))
                        color = (int(red_intensity), 0, 0)
                
                debug_vis.draw_box(bbox, color, 1, f"{label} d:{obj_z}")
        
        # Step 1: Check for direct hits at line endpoints
        direct_hit_objects = []
        
        for (x, y) in [(x1, y1), (x2, y2)]:
            for target_label in priority_labels:
                for obj in priority_objects.get(target_label, []):
                    obj_bbox = obj.get('bbox')
                    if obj_bbox and check_location_near_bbox(obj_bbox, x, y, threshold):
                        # Calculate priority score (label priority + distance if hand position is known)
                        priority = priority_labels.index(target_label)
                        distance_score = 0
                        if hand_mx is not None and hand_my is not None:
                            obj_mx, obj_my = calculate_box_midpoint(obj_bbox)
                            distance_score = ((obj_mx - hand_mx) ** 2 + (obj_my - hand_my) ** 2) ** 0.5
                        
                        direct_hit_objects.append((obj, priority, distance_score))
                        
                        # For debugging
                        if debug_vis:
                            obj['ray_hit'] = True
                            obj['hit_distance'] = distance_score
        
        # If we have direct hits, select the highest priority one
        if direct_hit_objects:
            # Sort by priority first, then by distance (if available)
            direct_hit_objects.sort(key=lambda x: (x[1], x[2]))
            best_obj = direct_hit_objects[0][0]
            
            obj_mx, obj_my = calculate_box_midpoint(best_obj.get('bbox'))
            
            # Perform depth check for direct hit if depth data available
            obj_z = best_obj.get('depth')
            
            # Only enforce depth check if both hand and object depths are available
            depth_valid = False
            if hand_z is not None and obj_z is not None:
                depth_diff = safe_depth_diff(hand_z, obj_z)
                depth_valid = depth_diff <= max_reach
                best_obj['depth_valid'] = depth_valid
                
                if depth_valid:
                    if debug_vis:
                        best_obj['selected'] = True
                    handle_detected_object(best_obj, obj_mx, obj_my, audio_server)
                    
                    # Return debug visualization if created
                    if debug_vis:
                        return debug_vis.get_frame()
                    return None
            else:
                # If depth data unavailable, proceed with the interaction
                if debug_vis:
                    best_obj['selected'] = True
                handle_detected_object(best_obj, obj_mx, obj_my, audio_server)
                
                # Return debug visualization if created
                if debug_vis:
                    return debug_vis.get_frame()
                return None
        
        # Step 2: Ray casting for extended pointing with reasonable range
        dx, dy = x2 - x1, y2 - y1
        length = np.sqrt(dx**2 + dy**2)
        
        if length > 0:
            # Use a smaller scale for more reasonable pointing distance
            scale = 15 / length  # Reduced from 50 to 15 for more realistic pointing
            extended_x = int(x2 + dx * scale)
            extended_y = int(y2 + dy * scale)
            
            # Draw the extended line in debug visualization
            if debug_vis:
                debug_vis.draw_line(x2, y2, extended_x, extended_y, (255, 0, 0), 2, "Extended ray")
                debug_vis.add_circle(extended_x, extended_y, 5, (255, 0, 0), -1)
                debug_vis.add_text(f"Max reach: {max_reach} units", 10, 350)
            
            # Calculate ray direction (normalized)
            ray_dir_x, ray_dir_y = dx/length, dy/length
            
            # Find all objects that the ray might intersect with
            ray_hit_objects = []
            
            for target_label in priority_labels:
                for obj in priority_objects.get(target_label, []):
                    obj_bbox = obj.get('bbox')
                    if not obj_bbox:
                        continue
                    
                    # Perform simplified ray-bbox intersection test
                    # This checks if the ray from (x2,y2) in direction (dx,dy) intersects the bbox
                    is_hit, hit_dist = ray_bbox_intersect(x2, y2, ray_dir_x, ray_dir_y, obj_bbox)
                    
                    if is_hit and hit_dist <= extended_distance_threshold:
                        obj_mx, obj_my = calculate_box_midpoint(obj_bbox)
                        
                        # Get object depth from earlier lookup
                        obj_z = obj.get('depth')
                        
                        # Calculate priority score
                        priority = priority_labels.index(target_label)
                        
                        # Add to list of potential hits with priority, distance, and depth info
                        ray_hit_objects.append((obj, priority, hit_dist, obj_z))
                        
                        # For debugging
                        if debug_vis:
                            obj['ray_hit'] = True
                            obj['hit_distance'] = hit_dist
            
            # If we have ray hits, process them by priority
            if ray_hit_objects:
                # Sort by priority first, then by distance
                ray_hit_objects.sort(key=lambda x: (x[1], x[2]))
                
                # Process hits in order until we find a valid one
                for obj, _, hit_dist, obj_z in ray_hit_objects:
                    obj_mx, obj_my = calculate_box_midpoint(obj.get('bbox'))
                    
                    # Check depth constraints if both hand and object depths are available
                    depth_valid = False
                    if hand_z is not None and obj_z is not None:
                        depth_diff = safe_depth_diff(hand_z, obj_z)
                        depth_valid = depth_diff <= max_reach
                        obj['depth_valid'] = depth_valid
                        
                        if depth_valid:
                            if debug_vis:
                                obj['selected'] = True
                            handle_detected_object(obj, obj_mx, obj_my, audio_server)
                            
                            # Return debug visualization if created
                            if debug_vis:
                                return debug_vis.get_frame()
                            return None
                    else:
                        # If depth data unavailable but distance is reasonable, proceed
                        if hit_dist <= 100:  # Reasonable pixel distance
                            if debug_vis:
                                obj['selected'] = True
                            handle_detected_object(obj, obj_mx, obj_my, audio_server)
                            
                            # Return debug visualization if created
                            if debug_vis:
                                return debug_vis.get_frame()
                            return None
            
            # Additional fallback: check if any object is near the extended point
            for target_label in priority_labels:
                for obj in priority_objects.get(target_label, []):
                    obj_bbox = obj.get('bbox')
                    if not obj_bbox:
                        continue
                        
                    if check_location_near_bbox(obj_bbox, extended_x, extended_y, threshold):
                        obj_mx, obj_my = calculate_box_midpoint(obj_bbox)
                        
                        # Get object depth from earlier lookup
                        obj_z = obj.get('depth')
                                
                        # Check if within depth range
                        depth_valid = False
                        if hand_z is not None and obj_z is not None:
                            depth_diff = safe_depth_diff(hand_z, obj_z)
                            depth_valid = depth_diff <= max_reach
                            obj['depth_valid'] = depth_valid
                            
                            if depth_valid:
                                if debug_vis:
                                    obj['ray_hit'] = True
                                    obj['hit_distance'] = np.sqrt((extended_x - obj_mx)**2 + (extended_y - obj_my)**2)
                                    obj['selected'] = True
                                handle_detected_object(obj, extended_x, extended_y, audio_server)
                                
                                # Return debug visualization if created
                                if debug_vis:
                                    return debug_vis.get_frame()
                                return None
                        else:
                            # Fallback if depth data not available
                            if debug_vis:
                                obj['ray_hit'] = True
                                obj['hit_distance'] = np.sqrt((extended_x - obj_mx)**2 + (extended_y - obj_my)**2)
                                obj['selected'] = True
                            handle_detected_object(obj, extended_x, extended_y, audio_server)
                            
                            # Return debug visualization if created
                            if debug_vis:
                                return debug_vis.get_frame()
                            return None
                            
    except Exception as e:
        print(f"Error in check_line_location: {e}")
        import traceback
        traceback.print_exc()
        
    # Return the debug frame if it exists
    if debug_vis:
        return debug_vis.get_frame()
    return None

def ray_bbox_intersect(ray_origin_x, ray_origin_y, ray_dir_x, ray_dir_y, bbox):
    """
    Check if a ray intersects with a bounding box. Handles edge cases and numerical issues.
    
    Args:
        ray_origin_x, ray_origin_y: Origin point of the ray
        ray_dir_x, ray_dir_y: Direction vector of the ray (normalized)
        bbox: Bounding box [x1, y1, x2, y2]
    
    Returns:
        tuple: (hit_detected, distance) - Boolean indicating if ray hits the bbox and distance to hit
    """
    try:
        x1, y1, x2, y2 = bbox
        
        # Small epsilon to handle numerical precision issues
        epsilon = 1e-6
        
        # Check for zero direction to avoid division by zero
        if abs(ray_dir_x) < epsilon and abs(ray_dir_y) < epsilon:
            return False, float('inf')
            
        # Handle division by near-zero with robust approach
        if abs(ray_dir_x) < epsilon:
            # Ray is vertical - check if x is within box horizontally
            if ray_origin_x < x1 - epsilon or ray_origin_x > x2 + epsilon:
                return False, float('inf')
            tx_min, tx_max = -float('inf'), float('inf')
        else:
            # Calculate intersections with vertical edges
            tx1 = (x1 - ray_origin_x) / ray_dir_x
            tx2 = (x2 - ray_origin_x) / ray_dir_x
            tx_min = min(tx1, tx2)
            tx_max = max(tx1, tx2)
        
        if abs(ray_dir_y) < epsilon:
            # Ray is horizontal - check if y is within box vertically
            if ray_origin_y < y1 - epsilon or ray_origin_y > y2 + epsilon:
                return False, float('inf')
            ty_min, ty_max = -float('inf'), float('inf')
        else:
            # Calculate intersections with horizontal edges
            ty1 = (y1 - ray_origin_y) / ray_dir_y
            ty2 = (y2 - ray_origin_y) / ray_dir_y
            ty_min = min(ty1, ty2)
            ty_max = max(ty1, ty2)
        
        # Find intersection range
        t_min = max(tx_min, ty_min)
        t_max = min(tx_max, ty_max)
        
        # Check if intersection occurs in positive ray direction
        if t_max < -epsilon or t_min > t_max:
            return False, float('inf')
            
        # Intersection distance (use t_min if positive, else t_max)
        t = t_min if t_min >= -epsilon else t_max
        
        # Only return hits in forward direction
        if t < epsilon:
            return False, float('inf')
        
        # Calculate intersection point for debugging/logging
        ix = ray_origin_x + t * ray_dir_x
        iy = ray_origin_y + t * ray_dir_y
        
        # Verify intersection point is actually within the bbox (numerical stability check)
        if ix < x1 - epsilon or ix > x2 + epsilon or iy < y1 - epsilon or iy > y2 + epsilon:
            # This should not happen with correct math, but guards against numerical errors
            return False, float('inf')
        
        # Calculate distance from ray origin to intersection point
        distance = ((ix - ray_origin_x) ** 2 + (iy - ray_origin_y) ** 2) ** 0.5
        
        return True, distance
    except Exception as e:
        print(f"Error in ray_bbox_intersect: {e}")
        return False, float('inf')


def handle_detected_object(obj, world_x, world_y, world_z, audio_server=None):
    """
    Handle detected object interaction based on its label and the interaction point.
    """
    try:
        if obj is None:
            return
            
        label = obj.get('label')
        if not label:
            return
        
        # Check cooldown for this label (using original label)
        if not cooldown_manager.can_trigger(label):
            return

        # Calculate relative coordinates for menus
        bbox = obj.get('bbox')
        if label == 'menu' and bbox:
            # Get the relative position within the menu
            if 'pixels' in obj:
                menu_frame = obj['pixels']
                rel_x = world_x - bbox[0]  # Relative x within menu
                rel_y = world_y - bbox[1]  # Relative y within menu
                gpt_response = gpt_functions.ask_gpt_about_contents(menu_frame, rel_x, rel_y)
            else:
                gpt_response = ""
        else:
            gpt_response = ""
        
        # IMPORTANT: Store original_label but DO NOT convert label to semantic version here
        original_label = label  # Keep track of original label
        
        # Coordinate calculation completed
        
        label_parameters = {"frame":obj.get('pixels'),
            "audio_server":audio_server,
            "detected_label":original_label,  # Use original technical label for processing
            "depth":0,          # Skip depth since we already have world coordinates
            "emotion":gpt_functions.get_current_emotion(),
            "text":gpt_response,
            # Pass the pre-calculated world coordinates directly
            "world_coords":(world_x, world_y, world_z)}
        # Trigger audio feedback using the provided world coordinates
        # Pass the ORIGINAL technical label, not the semantic label
        # handle_label(
        #     obj.get('pixels'),
        #     audio_server,
        #     original_label,  # Use original technical label for processing
        #     None, None,  # We skip mx, my since we already have world coordinates
        #     0,          # Skip depth since we already have world coordinates
        #     gpt_functions.get_current_emotion(),
        #     text=gpt_response,
        #     # Pass the pre-calculated world coordinates directly
        #     world_coords=(world_x, world_y, world_z)
        # )
        #parameterized version
        handle_label(**label_parameters)
        
        if label != "menu":
            cooldown_manager.trigger(original_label)
    except Exception as e:
        print(f"Error handling detected object: {e}")
        
        

def check_hand_proximity(hand_obj, objects, audio_server, debug_frame=None):
    """
    Check for objects near a hand that should be interacted with.
    Uses depth and spatial proximity.
    """
    if hand_obj is None or objects is None or 'bbox' not in hand_obj:
        return None
        
    try:
        h_bbox = hand_obj.get('bbox')
        h_mx, h_my = calculate_box_midpoint(h_bbox)
        
        # Get hand depth
        hand_depth = None
        current_depth_map = resource_manager.copy_resource('depth_map')
        if current_depth_map is not None:
            if 0 <= h_my < current_depth_map.shape[0] and 0 <= h_mx < current_depth_map.shape[1]:
                hand_depth = current_depth_map[h_my, h_mx]
        
        # Parameters for detection
        proximity_threshold = 60  # Pixel distance
        depth_threshold = 50      # Depth difference threshold
        best_object = None
        
        # Default priority weight
        default_priority_weight = 1.2
        
        # Collect all candidates
        candidates = []
        
        # Find candidate objects based on proximity
        for obj in objects:
            if obj == hand_obj or obj is None or 'bbox' not in obj:
                continue
                
            obj_bbox = obj.get('bbox')
            obj_label = obj.get('label', '')
            
            # Skip non-interactable objects
            if obj_label not in ['button', 'interactable', 'menu', 'watch', 'writing utensil']:
                continue
                
            obj_mx, obj_my = calculate_box_midpoint(obj_bbox)
            
            # Get object depth
            obj_depth = None
            if current_depth_map is not None:
                if 0 <= obj_my < current_depth_map.shape[0] and 0 <= obj_mx < current_depth_map.shape[1]:
                    obj_depth = current_depth_map[obj_my, obj_mx]
            
            # Calculate 2D distance
            distance_2d = np.sqrt((obj_mx - h_mx)**2 + (obj_my - h_my)**2)
            
            # Skip objects too far away immediately
            if distance_2d > proximity_threshold:
                continue
            
            # Calculate depth difference if both depths are available
            depth_diff = float('inf')
            if hand_depth is not None and obj_depth is not None:
                depth_diff = abs(hand_depth - obj_depth)
            
            # Combined score (weighted sum of normalized distances)
            # Lower is better
            distance_score = distance_2d / proximity_threshold
            depth_score = depth_diff / depth_threshold if depth_diff != float('inf') else 10.0
            
            # Weighting: 50% spatial proximity, 30% depth alignment, 20% priority
            combined_score = (0.5 * distance_score) + (0.3 * depth_score)
            
            # Store original values for debugging
            obj['combined_score'] = combined_score
            obj['distance_2d'] = distance_2d
            obj['depth_diff'] = depth_diff
            obj['priority_weight'] = default_priority_weight
            
            # Add to candidates
            candidates.append(obj)
        
        # Sort candidates by score (lower is better)
        candidates.sort(key=lambda x: x.get('combined_score', float('inf')))
        
        # Get the best object if any candidates exist
        if candidates:
            best_object = candidates[0]
            # Calculate world coordinates for the object
            obj_mx, obj_my = calculate_box_midpoint(best_object.get('bbox'))
            
            obj_depth = best_object.get('depth', 128)  # Default depth if missing
            
            # Calculate world coordinates for spatial audio
            world_x, world_y, world_z = calc_box_location(obj_mx, obj_my, obj_depth)
            
            # Only interact if not on cooldown
            if cooldown_manager.can_trigger(best_object.get('label')):
                # Use the world coordinates for the audio packet
                handle_detected_object(best_object, world_x, world_y, world_z, audio_server)
        
        return None
            
    except Exception as e:
        print(f"Error in hand proximity check: {e}")
        import traceback
        traceback.print_exc()
        
    return None


def process_debug_input(key, debug_visualizer):
    """
    Process keyboard input for debugging mode toggling.
    
    Args:
        key: The pressed key code
        debug_visualizer: The debug visualization system
    """
    if debug_visualizer is None:
        return
        
    try:
        # Toggle main debug mode
        if key == ord('d'):
            debug_visualizer.toggle_debug_mode('all')
            
        # Toggle interaction debug
        elif key == ord('i'):
            debug_visualizer.toggle_debug_mode('interaction')
            
        # Toggle proximity debug
        elif key == ord('h'):
            debug_visualizer.toggle_debug_mode('proximity')
            
    except Exception as e:
        print(f"Error processing debug input: {e}")


def create_debug_audio_feedback(toggle_type, audio_server):
    """
    Create audio feedback when debug mode is toggled.
    
    Args:
        toggle_type: Type of toggle ('on', 'off', 'interaction', 'proximity')
        audio_server: Audio server for broadcasting
    """
    if audio_server is None:
        return
    
    try:
        from tts_engine import synthesize_speech
        
        feedback_text = "Debug mode"
        if toggle_type == 'on':
            feedback_text += " enabled"
        elif toggle_type == 'off':
            feedback_text += " disabled"
        elif toggle_type == 'interaction':
            feedback_text += " interaction visualization toggled"
        elif toggle_type == 'proximity':
            feedback_text += " proximity visualization toggled"
        
        # Generate audio
        audio_data = synthesize_speech(feedback_text, "neutral")
        
        if audio_data:
            # Create and send packet
            from server import AudioPacket
            packet = AudioPacket(
                x=0.0,
                y=2.5,
                z=10.0,
                audio_data=audio_data,
                label="debug_toggle",
                asset_id=-1
            )
            
            # Schedule broadcast
            audio_server.schedule_broadcast(audio_server.broadcast_audio_packet(packet))
    except Exception as e:
        print(f"Error creating debug audio feedback: {e}")

def memory_cleanup():
    """Periodically cleans up GPU and CPU memory"""
    thread_coordinator.register_thread("memory_cleanup")
    
    while running:
        try:
            thread_coordinator.heartbeat("memory_cleanup")
            
            # Use memory monitor to perform cleanup
            queues = {
                'object_detection': object_detection_queue,
                'depth_detection': depth_detection_queue,
                'edge_detection': edge_detection_queue,
                'gpt_request': gpt_request_queue
            }
            memory_monitor.clear_stale_queues(queues)
            memory_monitor.check_and_cleanup()
            
            
        except Exception as e:
            thread_coordinator.increment_error("memory_cleanup")
            print(f"Error in memory cleanup: {e}")
            
        # Run every minute
        time.sleep(60)

def thread_watchdog():
    """Monitors critical threads and restarts them if they've stopped"""
    thread_coordinator.register_thread("thread_watchdog")
    # Thread watchdog started
    
    # Thread functions now handled by engine modules
    # Old watchdog system may need updates for new architecture
    
    # Thread monitoring now handled by ThreadCoordinator
    # This watchdog function needs refactoring for new engine architecture
    
    # Track global system health
    system_health = {
        'last_system_check': time.time(),
        'thread_failures': 0,
        'critical_failures': 0,
    }
    global running  
    while running:
        try:
            thread_coordinator.heartbeat("thread_watchdog")
            
            # Check if we should shutdown
            if thread_coordinator.should_shutdown():
                print("Watchdog detected shutdown flag. Initiating system shutdown...")
                running = False
                break
            
            # Check for critical system health
            # Engine threads now handle their own restart logic
            critical_threads = ["object_detection", "depth_detection", "edge_detection"]
            for thread_name in critical_threads:
                if thread_coordinator.needs_restart(thread_name):
                    # Update system health metrics
                    system_health['thread_failures'] += 1

                    # If failures are excessive, consider system shutdown
                    if system_health['thread_failures'] > 20:
                        system_health['critical_failures'] += 1
                        if system_health['critical_failures'] > 5:
                            print(f"CRITICAL: Excessive failures in {thread_name}")
                            thread_coordinator.safe_shutdown()
                            continue

                    # Engine threads handle their own restart logic
                    # Monitoring thread health
                    thread_coordinator.record_restart(thread_name)
            
            # Check for deadlocks
            potential_deadlocks = thread_coordinator.detect_deadlock()
            if potential_deadlocks:
                print(f"Potential deadlocks detected in: {potential_deadlocks}")
                thread_coordinator.emergency_restart(potential_deadlocks)
            
            # Reset health metrics periodically
            current_time = time.time()
            if current_time - system_health['last_system_check'] > 600:  # 10 minutes
                system_health['thread_failures'] = max(0, system_health['thread_failures'] - 5)
                system_health['critical_failures'] = max(0, system_health['critical_failures'] - 1)
                system_health['last_system_check'] = current_time
                
            # Log status periodically
            if int(current_time) % 60 == 0:
                status_report = thread_coordinator.get_status_report()
                
        except Exception as e:
            thread_coordinator.increment_error("thread_watchdog")
            
        time.sleep(5)

def process_keyboard_commands(key, frame_copy, objects, current_depth_map, audio_server, debug_visualizer=None):
    """
    Process keyboard commands with improved combo detection and race condition handling.
    
    Args:
        key: The key that was pressed
        frame_copy: The current frame
        objects: Detected objects in the frame
        current_depth_map: The current depth map
        audio_server: Server for sending audio packets
        debug_visualizer: Optional debug visualization system
    
    Returns:
        bool: True to continue, False to exit
    """
    # Static variables to track key state
    if not hasattr(process_keyboard_commands, "state"):
        process_keyboard_commands.state = {
            "key_state": {},               # Track currently pressed keys
            "key_timestamps": {},          # When each key was pressed
            "last_command_time": 0,        # Last time a command was executed
            "combo_window": 1.0,           # Window for combo detection (seconds) - increased from 0.3 to 0.5
            "command_cooldown": 0.5,       # Cooldown between commands (seconds)
            "last_message_time": 0,        # For cooldown messages
            "combo_executed": False,       # Flag to indicate combo has been executed
            "combo_executed_time": 0,      # When combo was executed
            "key_expiry": 1.5              # Time after which keys are considered released (seconds) - increased from 0.4 to 0.8
        }
    
    # Local reference to state for cleaner code
    state = process_keyboard_commands.state
    
    # Current time for various checks
    current_time = time.time()
    
    # Automatic key expiry - remove keys that haven't been refreshed recently
    keys_to_remove = []
    for k, timestamp in state["key_timestamps"].items():
        if current_time - timestamp > state["key_expiry"]:
            keys_to_remove.append(k)
    
    for k in keys_to_remove:
        if k in state["key_state"]:
            del state["key_state"][k]
        if k in state["key_timestamps"]:
            del state["key_timestamps"][k]
    
    # Check if user commands are currently being processing
    if cooldown_manager.is_user_command_active():
        if current_time - state["last_message_time"] > 2:
            state["last_message_time"] = current_time
        return True
    
    # Process debug mode toggle keys if debug_visualizer is available
    if debug_visualizer is not None:
        if key == ord('d'):
            debug_visualizer.toggle_debug_mode('all')
            return True
        elif key == ord('i'):
            debug_visualizer.toggle_debug_mode('interaction')
            return True
        elif key == ord('h'):
            debug_visualizer.toggle_debug_mode('proximity')
            return True
    
    # Reset combo executed flag after a timeout
    # if state["combo_executed"] and (current_time - state["combo_executed_time"] > 1.0):
    #     state["combo_executed"] = False
    #     # Clear key state to prevent retriggering
    #     if '1' in state["key_state"]:
    #         del state["key_state"]['1']
    #     if '2' in state["key_state"]:
    #         del state["key_state"]['2']
    
    # Process key press
    if key > 0 and key < 255:  # Only process valid keys
        key_char = chr(key) if 32 <= key <= 126 else str(key)  # Convert to printable char if possible
        
        
        # Record key press with timestamp
        state["key_state"][key_char] = True
        state["key_timestamps"][key_char] = current_time
        
        # Check for command cooldown
        if current_time - state["last_command_time"] < state["command_cooldown"]:
            return True
        
        # First, check if both '1' and '2' are pressed (combo detection)
        # This is the critical improvement - check both keys BEFORE individual handlers
        # if ('1' in state["key_state"] and '2' in state["key_state"] and 
        #     abs(state["key_timestamps"].get('1', 0) - state["key_timestamps"].get('2', 0)) < state["combo_window"]):
            
        #     # Execute 1+2 combo directly, don't proceed to individual key handlers
        #     if not state["combo_executed"]:
        #         state["combo_executed"] = True
        #         state["combo_executed_time"] = current_time
        #         process_command_1_2_combo(objects, frame_copy, current_depth_map, audio_server)
        #         state["last_command_time"] = current_time
        #     return True
        
        if key == ord('3'):
            # Execute 1+2 combo directly, don't proceed to individual key handlers
            # if not state["combo_executed"]:
            state["combo_executed_time"] = current_time
            # Execute 1+2 combo command
            edge_detections = resource_manager.copy_resource('edge_detections')
            process_command_1_2_combo(objects, frame_copy, current_depth_map, audio_server, cooldown_manager, edge_detections, None)
            state["last_command_time"] = current_time
            return True

        # If not a combo, handle individual key presses
        # Handle key '2' press
        if key == ord('2'):
            # if not state["combo_executed"]:
            process_scenesweep(objects, frame_copy, current_depth_map, audio_server, cooldown_manager)
            state["last_command_time"] = current_time
            return True
            
        # Handle contextcompass press
        if key == ord('1'):
            # if not state["combo_executed"]:
            try:
                # Clear any pending speech
                cooldown_manager.clear_queue()
                
                # Play audio feedback
                play_user_command_feedback("2", audio_server)
                
                # Mark start of user command execution
                cooldown_manager.start_user_command("2")
                
                print("Starting ContextCompass!")
                logging.info("ContextCompass triggered")
                
                if frame_copy is None:
                    print("ERROR: No frame available for GPT query")
                    cooldown_manager.end_user_command()
                    return True
                    
                # Safe copy of objects for GPT
                current_objects_copy = []
                if objects:
                    for obj in objects:
                        if obj is not None:
                            current_objects_copy.append(obj)
                
                # Callback function to end command when GPT processing is complete
                def gpt_callback(result):
                    cooldown_manager.end_user_command()
                
                # Enqueue the task with our callback
                gpt_request_queue.put({
                    "type": "frame_description",
                    "frame": frame_copy.copy(),
                    "detections": current_objects_copy,
                    "callback": gpt_callback
                })
                
                # Update command time
                state["last_command_time"] = current_time
                
            except Exception as e:
                print(f"ERROR processing key '2' action: {e}")
                cooldown_manager.end_user_command()
            
            return True
    
    # Handle other keys
    if key == ord('c'):
        # Toggle cooldown manager debug mode
        if hasattr(cooldown_manager, 'toggle_debug'):
            cooldown_manager.toggle_debug()
        return True
        
    elif key == ord('p'):
        # Toggle proximity visualization
        if debug_visualizer:
            debug_visualizer.debug_windows['proximity']['enabled'] = not debug_visualizer.debug_windows['proximity']['enabled']
            print(f"Proximity debug: {'ON' if debug_visualizer.debug_windows['proximity']['enabled'] else 'OFF'}")
        return True

    elif key == ord('s'):
        if debug_visualizer:
            debug_visualizer.toggle_stats_display()
            print(f"Stats display: {'Enabled' if debug_visualizer.show_stats else 'Disabled'}")
        return True

    elif key == ord('q'):  # Exit the program
        print("Exiting...")
        cv2.destroyAllWindows()
        return False  # Signal to exit the main loop
        
    return True  # Continue main loop

def play_user_command_feedback(command, audio_server):
    """
    Play audio feedback for a user command immediately.
    
    Args:
        command: The command that was pressed ('1', '2', 'aim_assist', or 'combo')
        audio_server: Server for sending audio packets
    """
    if audio_server is None:
        return
    
    try:
        # Simple feedback to confirm command reception
        feedback = None
        if command == "1":
            feedback = synthesize_speech("Reading all objects", "neutral")
        elif command == "2":
            feedback = synthesize_speech("Describing scene", "neutral")
        elif command == "aim_assist":
            feedback = synthesize_speech("Detecting nearby objects", "neutral")
        elif command == "combo":
            feedback = synthesize_speech("Enhanced object reading", "neutral")
            
        if feedback:
            # Send the audio packet to provide immediate feedback
            packet = AudioPacket(
                x=0.0,  # Center position
                y=2.5,
                z=30.0,
                audio_data=feedback,
                label=f"command_announcement_{command}",
                asset_id=-1
            )
            
            audio_server.schedule_broadcast(audio_server.broadcast_audio_packet(packet))
    except Exception as e:
        print(f"Error in play_user_command_feedback: {e}")
def monitor_command_completion():
    """
    Monitor speech queue and end the user command when all speech is done.
    Includes timeout safety to prevent hanging.
    """
    # Wait until speech queue is empty and not speaking
    max_wait_time = 60  # 30 seconds maximum wait time
    check_interval = 0.5  # seconds
    start_time = time.time()
    last_queue_size = -1
    last_progress_time = start_time
    
    
    try:
        while time.time() - start_time < max_wait_time:
            # Get current queue size
            current_queue_size = cooldown_manager.speech_queue.qsize() if hasattr(cooldown_manager.speech_queue, 'qsize') else 0
            is_speaking = cooldown_manager.is_speaking()
            
            # Log if queue size changes
            if current_queue_size != last_queue_size:
                last_queue_size = current_queue_size
                last_progress_time = time.time()
            
            # Check if we're done
            if cooldown_manager.is_queue_empty() and not is_speaking:
                # Double-check after a short delay to ensure no speech is pending
                time.sleep(1.0)
                if cooldown_manager.is_queue_empty() and not cooldown_manager.is_speaking():
                    break
                    
            # Check for lack of progress - if queue size hasn't changed in a while
            current_time = time.time()
            if current_time - last_progress_time > 20:  # No progress for 20 seconds
                print("WARNING: No progress in speech queue for 20 seconds - possible hang")
                # Force completion if we're clearly stuck
                if current_queue_size == 0 and not is_speaking:
                    print("Queue appears empty but command not completing - forcing completion")
                    break
                elif current_queue_size > 0:
                    # Try to unstick by removing one item
                    try:
                        if not cooldown_manager.speech_queue.empty():
                            cooldown_manager.speech_queue.get_nowait()
                            cooldown_manager.speech_queue.task_done()
                            print("Removed one item from speech queue to unstick processing")
                    except Exception:
                        pass  # Ignore errors in emergency unsticking
                
                last_progress_time = current_time  # Reset timer
            
            # Short sleep before checking again
            time.sleep(check_interval)
        
        # Check if we timed out
        if time.time() - start_time >= max_wait_time:
            print(f"Command monitor timed out after {max_wait_time}s - ending command anyway")
            
            # Try to clear queue as a precaution
            cooldown_manager.clear_queue()
        else:
            pass
        
    except Exception as e:
        print(f"Error in command completion monitor: {e}")
    finally:
        # Always end the user command, even if there was an error
        cooldown_manager.end_user_command()

if __name__ == '__main__':
    log_info("Starting VR-AI Scene Recognition System")

    # Validate environment variables
    if not validate_environment():
        log_error("Environment validation failed - exiting")
        sys.exit(1)

    # Validate model files
    if not validate_model_files():
        log_error("Model file validation failed - exiting")
        sys.exit(1)

    running = True
    last_auto_detection_time = 0  # auto detection timer
    components = None
    
    # Setup signal handlers for cleaner termination
    import signal
    
    def signal_handler(sig, frame):
        print(f"Received signal {sig}, initiating shutdown...")
        global running
        running = False
        
        # Cleanup without waiting for the main loop to finish
        if components:
            cleanup_system(components)
            
        # Force exit after cleanup
        import os
        os._exit(0)
        
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # Termination request
    
    try:
        # Initialize system components
        components, critical_failure = initialize_system()

        # Extract critical components
        raw_video = components["video_stream"]
        audio_server = components["audio_server"]
        
        # Start all threads
        threads = []
        
        # Helper function to create and start a thread
        def start_thread(target, name, daemon=True):
            thread = threading.Thread(target=target, name=name, daemon=daemon)
            thread.start()
            threads.append(thread)
            return thread
        
        # Start helper threads
        gpt_thread = start_thread(gpt_worker, "gpt_worker")
        frame_reader_thread = start_thread(frame_reader, "frame_reader")
        memory_cleanup_thread = start_thread(memory_cleanup, "memory_cleanup")
        thread_watchdog_thread = start_thread(thread_watchdog, "thread_watchdog")
        
        # Start main processing threads using new engines
        # Sequential processing coordinator for same frame
        import threading
        import queue

        # Create a coordination system
        frame_coordinator = threading.Event()
        shared_frame = [None]
        models_finished = [0]

        def get_current_frame():
            return resource_manager.get_resource('frame')

        def get_frame_sequential():
            current_frame = resource_manager.get_resource('frame')
            if current_frame is not None:
                shared_frame[0] = current_frame.copy()
                models_finished[0] = 0
                frame_coordinator.clear()
            return shared_frame[0]

        def get_current_objects():
            return resource_manager.copy_resource('object_detections')

        def object_result_callback(detections, annotated_frame):
            resource_manager.update_resource('object_detections', detections)

            try:
                object_detection_queue.put(annotated_frame, block=False)
            except queue.Full:
                try:
                    object_detection_queue.get_nowait()
                    object_detection_queue.put(annotated_frame, block=False)
                except:
                    pass

        def depth_result_callback(depth_map_result, depth_visualization):
            resource_manager.update_resource('depth_map', depth_map_result)

            try:
                depth_detection_queue.put(depth_visualization, block=False)
            except queue.Full:
                try:
                    depth_detection_queue.get_nowait()
                    depth_detection_queue.put(depth_visualization, block=False)
                except:
                    pass

        def edge_result_callback(detected_lines, vis_frame):
            resource_manager.update_resource('edge_detections', detected_lines)

            try:
                edge_detection_queue.put(vis_frame, block=False)
            except queue.Full:
                pass

        running_flag = [True]

        object_detection_thread = start_thread(
            lambda: run_object_detection_thread(
                object_engine, get_current_frame, object_result_callback,
                thread_coordinator, running_flag
            ), "object_detection", daemon=False)


        depth_detection_thread = start_thread(
            lambda: run_depth_detection_thread(
                depth_engine, get_current_frame, depth_result_callback,
                thread_coordinator, running_flag
            ), "depth_detection", daemon=False)

        edge_detection_thread = start_thread(
            lambda: run_edge_detection_thread(
                edge_engine, get_current_frame, get_current_objects, edge_result_callback,
                thread_coordinator, running_flag
            ), "edge_detection", daemon=False)
        server_thread = start_thread(start_audio_stream_server, "audio_server", daemon=False)
        tone_detection_thread = start_thread(run_periodic_tone_detection, "tone_detection")

        # Initialize cooldown manager
        cooldown_manager = CategoryCooldownManager()
        
        # Register cooldowns for special labels
        for label, cooldown_time in COOLDOWNS.items():
            cooldown_manager.register_special_label(label, cooldown_time)
            
        initialize_handler(audio_server, cooldown_manager)
        
        # Labels now imported from config.py as SPECIAL_LABELS and EDGE_LABELS

        # Main display loop
        while True:
            try:
                # Update performance monitoring
                update_frame_count()
                report_performance()

                # Display frames from queues (optimized)
                try:
                    od_frame = object_detection_queue.get_nowait()
                    cv2.imshow('Object Detection', od_frame)
                except queue.Empty:
                    pass

                try:
                    dd_frame = depth_detection_queue.get_nowait()
                    cv2.imshow('Depth Detection', dd_frame)
                except queue.Empty:
                    pass

                try:
                    ed_frame = edge_detection_queue.get_nowait()

                    # Add debug status overlay if debug mode is enabled
                    if debug_visualizer and debug_visualizer.debug_mode:
                        ed_frame = debug_visualizer.add_status_overlay(ed_frame)

                    cv2.imshow('Edge Detection', ed_frame)
                except queue.Empty:
                    pass

                # Safely acquire current state
                objects = None
                objects = resource_manager.copy_resource('object_detections')
                current_depth_map = resource_manager.copy_resource('depth_map')
                current_lines = resource_manager.copy_resource('edge_detections')
                frame_copy = resource_manager.copy_resource('frame')

                # Skip iteration if we don't have necessary data
                if frame_copy is None:
                    print("WARNING: no frame received. continuing...")
                    time.sleep(0.1)
                    continue

                current_time = time.time()

                # Process automatic special labels
                if objects is not None:
                    # Pre-filter special labels for efficiency
                    special_detections = [obj for obj in objects if obj is not None and obj.get('label') in SPECIAL_LABELS]
                    
                    for detection in special_detections:
                        try:
                            label = detection.get('label')
                            if label is None:
                                continue
                                
                            bbox = detection.get('bbox')
                            if bbox is None:
                                continue
                                
                            # Check cooldown for this label
                            if not cooldown_manager.can_trigger(label):
                                continue
                            
                            mx, my = calculate_box_midpoint(bbox)
                            
                            # Safely get depth value
                            depth = 0
                            if current_depth_map is not None:
                                try:
                                    if 0 <= my < current_depth_map.shape[0] and 0 <= mx < current_depth_map.shape[1]:
                                        depth = current_depth_map[my, mx]
                                    else:
                                        print(f"WARNING: Coordinates ({mx}, {my}) out of depth map bounds")
                                except Exception as e:
                                    print(f"ERROR accessing depth map: {e}")
                            
                            # Safely extract frame data
                            try:
                                if (0 <= bbox[1] < frame_copy.shape[0] and 
                                    0 <= bbox[3] <= frame_copy.shape[0] and 
                                    0 <= bbox[0] < frame_copy.shape[1] and 
                                    0 <= bbox[2] <= frame_copy.shape[1] and
                                    bbox[1] < bbox[3] and bbox[0] < bbox[2]):  # Ensure valid bbox dimensions
                                    
                                    object_framedata = frame_copy[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                                    
                                    # Call handle_label safely
                                    try:
                                        handle_label(
                                            object_framedata,
                                            audio_server,
                                            label,
                                            mx, my,
                                            depth,
                                            gpt_functions.get_current_emotion()
                                        )
                                        
                                        # Update cooldown
                                        cooldown_manager.trigger(label)
                                    except Exception as e:
                                        print(f"ERROR in handle_label for {label}: {e}")
                                else:
                                    print(f"WARNING: Invalid bbox dimensions {bbox} for frame of shape {frame_copy.shape}")
                            except Exception as e:
                                print(f"ERROR extracting frame data for {label}: {e}")
                        except Exception as e:
                            print(f"ERROR processing special label: {e}")

                # Process lines using our enhanced detection if we have lines
                if current_lines is not None and objects is not None and current_depth_map is not None:
                    try:
                        # Only create debug frame if in debug mode
                        debug_frame = None
                        if debug_visualizer and debug_visualizer.debug_mode:
                            debug_frame = frame_copy.copy()
                            
                        enhance_check_line_location(
                            objects, 
                            current_lines, 
                            frame_copy, 
                            current_depth_map, 
                            audio_server, 
                            debug_frame
                        )
                    except Exception as e:
                        print(f"ERROR in enhanced line detection: {e}")
                        import traceback
                        traceback.print_exc()

                # Handle proximity visualization if enabled
                if debug_visualizer and debug_visualizer.debug_mode and debug_visualizer.debug_windows['proximity']['enabled']:
                    try:
                        # Find hand objects
                        hand_objects = [obj for obj in objects if obj is not None and obj.get('label') in EDGE_LABELS]
                        
                        # Only process the first hand for proximity visualization
                        if hand_objects:
                            hand_obj = hand_objects[0]
                            
                            # Get hand depth
                            hand_bbox = hand_obj.get('bbox')
                            h_mx, h_my = calculate_box_midpoint(hand_bbox)
                            hand_depth = None
                            
                            if current_depth_map is not None:
                                if 0 <= h_my < current_depth_map.shape[0] and 0 <= h_mx < current_depth_map.shape[1]:
                                    hand_depth = current_depth_map[h_my, h_mx]
                            
                            hand_obj['depth'] = hand_depth
                            
                            # Find nearby objects
                            nearby_objects = []
                            selected_obj = None
                            
                            # Define proximity threshold
                            proximity_threshold = 60  # pixels
                            
                            for obj in objects:
                                if obj == hand_obj or obj is None or 'bbox' not in obj:
                                    continue
                                    
                                # Get object center and depth
                                obj_bbox = obj.get('bbox')
                                obj_mx, obj_my = calculate_box_midpoint(obj_bbox)
                                
                                # Calculate 2D distance
                                distance_2d = np.sqrt((obj_mx - h_mx)**2 + (obj_my - h_my)**2)
                                
                                # Get object depth
                                obj_depth = None
                                if current_depth_map is not None:
                                    if 0 <= obj_my < current_depth_map.shape[0] and 0 <= obj_mx < current_depth_map.shape[1]:
                                        obj_depth = current_depth_map[obj_my, obj_mx]
                                
                                obj['depth'] = obj_depth
                                
                                # Check if object is within proximity threshold
                                if distance_2d <= proximity_threshold:
                                    # Calculate a score based on distance and type
                                    score = distance_2d
                                    
                                    # Add to nearby objects list
                                    nearby_objects.append((obj, score, distance_2d))
                            
                            # Sort nearby objects by score
                            nearby_objects.sort(key=lambda x: x[1])
                            
                            # Select the closest object
                            if nearby_objects:
                                selected_obj = nearby_objects[0][0]
                            
                            # Create and queue debug visualization
                            debug_visualizer.visualize_hand_proximity(
                                frame_copy, 
                                hand_obj, 
                                objects, 
                                nearby_objects, 
                                selected_obj
                            )
                        else:
                            # No hands found, still create visualization with empty hand
                            debug_visualizer.visualize_hand_proximity(
                                frame_copy, 
                                None, 
                                objects, 
                                [], 
                                None
                            )
                    except Exception as e:
                        print(f"Error in proximity visualization: {e}")
                        import traceback
                        traceback.print_exc()

                # Handle interaction visualization if enabled
                if debug_visualizer and debug_visualizer.debug_mode and debug_visualizer.debug_windows['interaction']['enabled']:
                    try:
                        # Create a debug frame if none exists yet
                        if frame_copy is None:
                            continue
                            
                        debug_frame = frame_copy.copy()
                        
                        # Find hand objects and lines for interaction visualization
                        hand_objects = [obj for obj in objects if obj is not None and obj.get('label') in EDGE_LABELS]
                        
                        # Initialize visualizer
                        debug_vis = DebugVisualizer(debug_frame)
                        debug_vis.add_title("Ray Interaction Detection", "Interaction Debug")
                        
                        # If we have hands and lines, visualize interaction
                        if hand_objects and current_lines is not None and len(current_lines) > 0:
                            hand_obj = hand_objects[0]
                            hand_bbox = hand_obj.get('bbox')
                            h_mx, h_my = calculate_box_midpoint(hand_bbox)
                            
                            # Get hand depth
                            hand_depth = None
                            if current_depth_map is not None:
                                if 0 <= h_my < current_depth_map.shape[0] and 0 <= h_mx < current_depth_map.shape[1]:
                                    hand_depth = current_depth_map[h_my, h_mx]
                            
                            # Draw hand
                            debug_vis.draw_box(hand_bbox, (0, 255, 0), 2, f"Hand/Controller d:{hand_depth}")
                            
                            # Find a line that starts from this hand
                            for line in current_lines:
                                if line is None:
                                    continue
                                    
                                # Get line coordinates
                                if len(line) == 1 and isinstance(line[0], (list, np.ndarray)):
                                    line_coords = line[0]
                                else:
                                    line_coords = line
                                    
                                if len(line_coords) != 4:
                                    continue
                                    
                                x1, y1, x2, y2 = line_coords
                                
                                # Check if either end of the line is near the hand
                                if check_location_near_bbox(hand_bbox, x1, y1, 15) or check_location_near_bbox(hand_bbox, x2, y2, 15):
                                    # Make sure x1,y1 is at the hand end
                                    if check_location_near_bbox(hand_bbox, x2, y2, 15):
                                        x1, y1, x2, y2 = x2, y2, x1, y1
                                    
                                    # Draw the line
                                    debug_vis.draw_line(x1, y1, x2, y2, (0, 255, 0), 2, "Detected Line")
                                    
                                    # Calculate and draw extended ray
                                    dx, dy = x2 - x1, y2 - y1
                                    length = np.sqrt(dx**2 + dy**2)
                                    if length > 0:
                                        scale = 15 / length
                                        extended_x = int(x2 + dx * scale)
                                        extended_y = int(y2 + dy * scale)
                                        
                                        debug_vis.draw_line(x2, y2, extended_x, extended_y, (255, 0, 0), 2, "Extended Ray")
                                        debug_vis.add_circle(extended_x, extended_y, 5, (255, 0, 0), -1)
                                        
                                        # Draw all objects
                                        for obj in objects:
                                            if obj is None or 'bbox' not in obj or obj == hand_obj:
                                                continue
                                                
                                            bbox = obj.get('bbox')
                                            label = obj.get('label', 'unknown')
                                            depth = obj.get('depth')
                                            
                                            # Regular objects in white
                                            color = (255, 255, 255)
                                            thickness = 1
                                            
                                            # Draw the object
                                            debug_vis.draw_box(bbox, color, thickness, f"{label} d:{depth}")
                                    
                                    # Only process the first valid line
                                    break
                        else:
                            # No hands or lines to visualize
                            debug_vis.add_text("No hands or lines detected", 10, 70, (0, 0, 255))
                            
                        # Add explanatory text
                        debug_vis.add_text("Ray interaction debugging", 10, 50)
                        
                        # Queue the frame for display
                        debug_visualizer.queue_debug_frame('interaction', debug_vis.get_frame())
                        
                    except Exception as e:
                        print(f"Error in interaction visualization: {e}")
                        import traceback
                        traceback.print_exc()
        
                # Process key presses
                key = cv2.waitKey(1) & 0xFF
                if not process_keyboard_commands(key, frame_copy, objects, current_depth_map, audio_server, debug_visualizer):
                    break  # Exit the main loop if process_keyboard_commands returns False
                
            except KeyboardInterrupt:
                print("Keyboard interrupt received. Shutting down...")
                break
            except Exception as e:
                print(f"ERROR in main loop iteration: {e}")
                # Continue running despite errors
                continue

    except KeyboardInterrupt:
        print("Exiting due to keyboard interrupt...")
    except Exception as e:
        print(f"CRITICAL ERROR in main thread: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Thorough cleanup
        if components:
            cleanup_system(components)
        print("Program terminated")
        
        # Force clean exit
        import os
        os._exit(0)
