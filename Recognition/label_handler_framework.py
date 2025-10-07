import gpt_functions
from label_handle_functions import (
    action_avatars,
    action_gpt, 
    action_ocr, 
    action_spatial,
    find_action_with_label, 
    parameter_removal,
    send_audio_packet,
    calc_box_location
)
from server import AudioPacket
import concurrent.futures
import time
import threading
from label_handler_list import (
    avatar_list,
    interactable_list,
    user_safety_list,
    informational_list,
    important_list,
    seating_area_list,
    user_hud_list,
    other_list,
    special_labels,
    action_list,
    action_dictionary,
)
from ocr_functions import perform_ocr_on_frame, is_similar
from tts_engine import synthesize_speech
from speech_cooldown_manager import CategoryCooldownManager
from settings import CAM_WIDTH, CAM_HEIGHT, ASSET_ID, PLAYCANVAS_CAMERA_HEIGHT, COOLDOWN_INTERACTABLES, COOLDOWN_USER_SAFETY, COOLDOWN_INFORMATIONAL

# cool-down array for interactable categories
# array that keeps track of last text for each category

COOLDOWN_INTERACTABLES_LIST = COOLDOWN_INTERACTABLES
COOLDOWN_USER_SAFETY_LIST = COOLDOWN_USER_SAFETY
COOLDOWN_INFORMATIONAL_LIST = COOLDOWN_INFORMATIONAL

# Maps each label type to its appropriate action function
action_map = {
    "ocr": action_ocr,
    "gpt": action_gpt,
    "spatial": action_spatial,
    "avatars": action_avatars,
}

# Label to action mapping for more specific handling
label_to_action = {
    # Avatar category
    "avatar": "avatars",
    "avatar-nonhuman": "avatars",
    "chat box": "ocr",
    "chat bubble": "ocr",
    "nametag": "ocr",
    
    # Informational category
    "progress bar": "ocr",
    "sign-graphic": "gpt",
    "ui-graphic": "gpt",
    "HUD": "gpt",
    "menu": "gpt",
    "sign-text": "ocr",
    "ui-text": "ocr",
    
    # Interactables category
    "button": "spatial",
    "interactable": "spatial",
    "portal": "spatial",
    "spawner": "spatial",
    "target": "spatial",
    "watch": "spatial",
    "writing surface": "spatial",
    "writing utensil": "spatial",
    
    # Safety category
    "guardian": "spatial",
    "out of bounds": "spatial",
    
    # Seating category
    "campfire": "spatial",
    "seat-single": "spatial",
    "seat-multiple": "spatial",
    "table": "spatial",
    
    # HUD category
    "controller": "spatial",
    "hand": "spatial",
    "dashboard": "spatial",
    "pointer-target": "spatial"
}
#retrieve the function associated with each string action key
action_map = {
    "ocr": action_ocr,
    "gpt": action_gpt,
    "spatial":action_spatial,
    "avatars":action_avatars,
}


#boiler plate function instead of 20 different functions
def boiler_plate(frame=None, audio_server=None,x=0,y=0,z=0, label=None, emotion="neutral"):
    action = find_action_with_label(label)
    params = {"frame":frame,"audio_server":audio_server,"x":x,"y":y,"z":z, "emotion":emotion}
    # Remove parameters not needed for this action type
    
    parameter_removal(action,params)

    #switch condition via dictionary function value retrieval
    action_function = action_map.get(action) 
    #call function if function is not None
    if action_function:
        #i.e, ocr
        action_function(**params) #** unpacks the parameters into multiple parameters
    else: 
        print("No action")

    


# Legacy maps removed - cooldown and similarity checking now handled in main.py
# Removed: interactables_map, interactables_text_map, safety_map, informational_map,
# informational_text_map, important_map, user_hud_map, dashboard_text_map

# Global variables
_audio_server = None
cooldown_manager = None  # Will be initialized in initialize_handler

# TTS request limiter
class TTSRateLimiter:
    def __init__(self, max_concurrent=2, cooldown_time=0.5):
        self.semaphore = threading.Semaphore(max_concurrent)
        self.cooldown_time = cooldown_time
        
    def acquire(self):
        """Acquire permission to make a TTS request"""
        acquired = self.semaphore.acquire(blocking=False)
        if not acquired:
            print("TTS rate limit reached, skipping request")
        return acquired
        
    def release(self):
        """Release the semaphore after TTS request completes"""
        try:
            self.semaphore.release()
            # Small cooldown to avoid overwhelming the TTS service
            time.sleep(self.cooldown_time)
        except Exception as e:
            print(f"Error releasing TTS semaphore: {e}")

# Initialize TTS rate limiter
tts_limiter = TTSRateLimiter()

def initialize_handler(server, cooldown_mgr=None):
    global _audio_server, cooldown_manager
    _audio_server = server
    
    # Initialize the cooldown manager if provided
    if cooldown_mgr is not None:
        cooldown_manager = cooldown_mgr
    else:
        # Fallback to create a new instance if none provided
        from speech_cooldown_manager import CategoryCooldownManager
        cooldown_manager = CategoryCooldownManager()

# Background processing for API-heavy tasks
class BackgroundProcessor:
    # Add this to the BackgroundProcessor class in label_handler_framework.py
    def __init__(self):
        # Create a daemon thread pool that won't block program exit
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=3, 
            thread_name_prefix="background_worker"
        )
        # Set attribute that allows for graceful shutdown
        self._shutdown_requested = False
        
    def shutdown(self):
        """Shut down the thread pool gracefully with improved cleanup"""
        self._shutdown_requested = True
        
        # Complete all submitted tasks before shutdown
        try:
            self.thread_pool.shutdown(wait=True)
        except Exception as e:
            print(f"Error during thread pool shutdown: {e}")
            # Fallback to non-waiting shutdown
            try:
                self.thread_pool.shutdown(wait=False)
            except Exception:
                pass
        
        print("Background processor shutdown complete") 
        
    def process_async(self, task_func, *args, **kwargs):
        """Execute a task asynchronously in the background"""
        self.thread_pool.submit(task_func, *args, **kwargs)
    
    def process_with_fallback(self, primary_func, fallback_func, *args, **kwargs):
        """Process with primary function, falling back to secondary if primary fails"""
        try:
            return primary_func(*args, **kwargs)
        except Exception as e:
            print(f"Primary function failed: {e}, using fallback")
            try:
                return fallback_func(*args, **kwargs)
            except Exception as fallback_error:
                print(f"Fallback function also failed: {fallback_error}")
                return None

        # Example usage for GPT queries
    def process_gpt_with_fallback(self, frame, prompt, label, original_label=None):
        """Process GPT query with fallback to simpler description"""
        
        def primary_func():
            if original_label == "menu":
                return gpt_functions.ask_gpt_about_object(frame, label, original_label)
            else:
                return gpt_functions.ask_gpt_about_object(frame, label)
            
        def fallback_func():
            # Simple fallback that doesn't require API call
            return f"{label} detected"
            
        return self.process_with_fallback(primary_func, fallback_func)
    
    def process_ocr_in_background(self, frame, audio_server, x, y, z, label, category, original_label=None):
        """Process OCR in background to prevent blocking.
        Uses pre-calculated world coordinates for consistent positioning.
        
        Args:
            frame: Image frame to process
            audio_server: The audio server
            x, y, z: World coordinates (already calculated using calc_box_location)
            label: The object label
            category: The object category
        """
        if frame is None or audio_server is None:
            return
            
        def background_task():
            try:
                # Add a small delay to prevent overlapping with initial notification
                name = original_label if original_label == "menu" else label

                # Attempt OCR with a timeout
                text = None
                ocr_text = perform_ocr_on_frame(frame)
                if ocr_text:
                    text = f"{label}: {ocr_text}"
                else:
                    text = f"{label} in view"
                
                # Send audio if we got a result
                try:
                    audio_data = synthesize_speech(text, gpt_functions.get_current_emotion())
                    
                    # Skip if synthesis failed
                    if audio_data is None:
                        print(f"Warning: Failed to synthesize OCR result for {label}")
                        return
                    time.sleep(0.3)
                    if audio_server is not None:
                        # Create audio packet using the provided world coordinates
                        # We don't recalculate coordinates here, ensuring consistency
                        packet = AudioPacket(
                            x=float(x),
                            y=float(y),
                            z=float(z),
                            audio_data=audio_data,
                            label=f"{label}_details",
                            asset_id=-1
                        )
                        
                        audio_server.schedule_broadcast(audio_server.broadcast_audio_packet(packet))
                except Exception as tts_error:
                    print(f"TTS error in background OCR: {tts_error}")
                    # Send a non-speech notification as fallback - using the same coordinates
                    send_audio_packet(x, y, z, audio_server, 207906643, f"{label}_details", "")
            except Exception as e:
                print(f"Error in background OCR processing for {label}: {e}")
                
        # Start the background task
        self.thread_pool.submit(background_task)
    
    def process_gpt_in_background(self, frame, audio_server, x, y, z, label, category, original_label=None):
        """
        Process GPT query in background to prevent blocking.
        Uses pre-calculated world coordinates for consistent positioning.
        
        Args:
            frame: Image frame to process
            audio_server: The audio server
            x, y, z: World coordinates (already calculated using calc_box_location)
            label: The object label
            category: The object category
        """
        if frame is None or audio_server is None:
            return
            
        def background_task():
            try:
                # Add a small delay to prevent overlapping with initial notification
                
                
                # Use our fallback system for more robust API calls
                text = self.process_gpt_with_fallback(frame, "", label, original_label)
                if not text:
                    text = f"{label} in view"
                
                # Send audio if we got a result
                try:
                    audio_data = synthesize_speech(text, gpt_functions.get_current_emotion())
                    
                    # Skip if synthesis failed
                    if audio_data is None:
                        print(f"Warning: Failed to synthesize GPT result for {label}")
                        return
                    time.sleep(0.7)
                    if audio_server is not None:
                        # Create audio packet using the provided world coordinates
                        # We don't recalculate coordinates here, ensuring consistency
                        packet = AudioPacket(
                            x=float(x),
                            y=float(y),
                            z=float(z),
                            audio_data=audio_data,
                            label=f"{label}_details",
                            asset_id=-1
                        )
                        
                        audio_server.schedule_broadcast(audio_server.broadcast_audio_packet(packet))
                    else:
                        print(f"Warning: Audio server not available for GPT result")
                except Exception as tts_error:
                    print(f"TTS error in background GPT: {tts_error}")
                    # Send a non-speech notification as fallback - using the same coordinates
                    send_audio_packet(x, y, z, audio_server, 207906643, f"{label}_details", "")
            except Exception as e:
                print(f"Error in background GPT processing for {label}: {e}")
                
        # Start the background task
        self.thread_pool.submit(background_task)

# Initialize background processor
background_processor = BackgroundProcessor()

def handle_unknown(label: str, emotion="neutral"):
    """Handle unknown label types safely"""
    try:
        if not tts_limiter.acquire():
            return None
            
        try:
            return synthesize_speech(label, emotion)
        finally:
            tts_limiter.release()
    except Exception as e:
        print(f"Error in handle_unknown: {e}")
        return None

label_map = {label: boiler_plate for label in action_list}

def handle_label(
    frame, audio_server, detected_label: str, mx=0, my=0, depth=0, emotion="neutral", text="",
    world_coords=None
) -> None:
    """
    Main label handling function that routes to appropriate handler based on label type.
    
    Args:
        frame: Image frame containing the detected object
        audio_server: Server for sending audio packets
        detected_label: Type of object detected
        mx, my: Screen coordinates of the detected object
        depth: Depth value of the detected object
        emotion: Current emotional context
        text: Optional text to include with the label
        world_coords: Optional pre-calculated world coordinates (x, y, z)
    """
    print(f"handle_label called: {detected_label}")
    try:
        # Skip if null label or missing coordinates and no world coordinates provided
        if (detected_label is None or (mx is None or my is None)) and world_coords is None:
            print("WARNING: Null label or coordinates provided to handle_label")
            return None
        
        # Calculate world coordinates for spatial audio if not provided
        if world_coords is None:
            world_x, world_y, world_z = calc_box_location(mx, my, depth)
        else:
            # Use the provided world coordinates
            world_x, world_y, world_z = world_coords
            
        if detected_label == "menu" and not cooldown_manager.is_user_command_active():
            print("Skipping automatic menu processing - will only process when Command 1 is pressed")
            return None
        
        # SPECIAL HANDLING FOR GUARDIAN: Just play sound with 5-second cooldown
        if detected_label == "guardian":
            # Check cooldown specifically for guardian
            if not cooldown_manager.can_execute_special_label("guardian"):
                # print("Guardian announcement on cooldown, skipping")
                return None
                
            # Record execution to start the cooldown
            cooldown_manager.record_special_label_execution("guardian")
            
            # Send direct audio packet without TTS to play sound effect
            send_audio_packet(world_x, world_y, world_z, audio_server, 218331816, "guardian", "")
            print("Guardian boundary sound effect played")
            
            # Return early - no further processing needed
            return None
        
        # Determine category for the label
        category = "other"
        if detected_label in avatar_list:
            category = "avatar"
        elif detected_label in informational_list:
            category = "informational"
        elif detected_label in interactable_list:
            category = "interactables"
        elif detected_label in seating_area_list:
            category = "seating"
        elif detected_label in user_safety_list:
            category = "user_safety"
        elif detected_label in user_hud_list:
            category = "user_hud"
            
        # Special handling for other special labels
        if detected_label in special_labels:
            # Check cooldown for special labels
            if not cooldown_manager.can_execute_special_label(detected_label):
                print(f"{detected_label} announcement on cooldown, skipping")
                return None
                
            # Record execution for special label
            cooldown_manager.record_special_label_execution(detected_label)
        else:
            # Check category cooldown for regular labels
            if not cooldown_manager.can_execute_category(category, detected_label):
                print(f"{category}:{detected_label} announcement on cooldown, skipping")
                return None
                
            # Record execution for category
            cooldown_manager.record_execution(category, detected_label)
        
        # Look up the action type for this label
        action_type = label_to_action.get(detected_label)
        
        if action_type:
            # Get the actual function to call
            action_func = action_map.get(action_type)
            
            if action_func:
                # Set up parameters for the function - always use the world coordinates
                params = {
                    "frame": frame,
                    "audio_server": audio_server,
                    "x": world_x,
                    "y": world_y,
                    "z": world_z,
                    "label": detected_label,
                    "emotion": emotion
                }
                
                # Remove any parameters that aren't needed for this action
                action = action_type  # action is the function name string
                parameter_removal(action, params)
                
                # Start asynchronous processing for API-heavy actions
                if action_type == "ocr":
                    # Try to send immediate notification
                    try:
                        basic_audio = synthesize_speech(f"{detected_label}")
                        if basic_audio and audio_server:
                            send_audio_packet(world_x, world_y, world_z, audio_server, 207906643, detected_label, basic_audio)
                    except Exception as e:
                        print(f"Warning: Could not synthesize initial speech for {detected_label}: {e}")
                        # Just send spatial audio packet instead
                        send_audio_packet(world_x, world_y, world_z, audio_server, 207906643, detected_label, "")
                    
                    # Process details in background if we have a frame
                    if frame is not None:
                        try:
                            # Pass the world coordinates to background processor
                            background_processor.process_ocr_in_background(
                                frame.copy(), audio_server, world_x, world_y, world_z, detected_label, category
                            )
                        except Exception as e:
                            print(f"Warning: Could not start background OCR for {detected_label}: {e}")
                elif action_type == "gpt":
                    # Try to send immediate notification
                    try:
                        basic_audio = synthesize_speech(f"{detected_label}")
                        if basic_audio and audio_server:
                            send_audio_packet(world_x, world_y, world_z, audio_server, 207906643, detected_label, basic_audio)
                    except Exception as e:
                        print(f"Warning: Could not synthesize initial speech for {detected_label}: {e}")
                        # Just send spatial audio packet instead
                        send_audio_packet(world_x, world_y, world_z, audio_server, 207906643, detected_label, "")
                    
                    # Process details in background if we have a frame
                    if frame is not None:
                        try:
                            # Pass the world coordinates to background processor
                            background_processor.process_gpt_in_background(
                                frame.copy(), audio_server, world_x, world_y, world_z, detected_label, category
                            )
                        except Exception as e:
                            print(f"Warning: Could not start background GPT for {detected_label}: {e}")
                else:
                    # For other action types, call directly with error handling
                    try:
                        action_func(**params)
                    except Exception as action_error:
                        print(f"Error in {action_type} for {detected_label}: {action_error}")
                        # Try a simple spatial notification
                        send_audio_packet(world_x, world_y, world_z, audio_server, 207906643, detected_label, "")
            else:
                print(f"No action function found for type: {action_type}")
                # Fall back to direct spatial notification
                send_audio_packet(world_x, world_y, world_z, audio_server, 207906643, detected_label, "")
        else:
            print(f"No action type defined for label: {detected_label}")
            # Fall back to direct spatial notification
            send_audio_packet(world_x, world_y, world_z, audio_server, 207906643, detected_label, "")
                
    except Exception as e:
        print(f"Error in handle_label for '{detected_label}': {e}")
        import traceback
        traceback.print_exc()
        
        # Try to provide some audio output even if there was an error
        try:
            if audio_server:
                # Use direct spatial audio without TTS to avoid more errors
                send_audio_packet(0, 0, 0, audio_server, 207906643, detected_label, "")
        except:
            pass  # Don't let error handling cause more errors