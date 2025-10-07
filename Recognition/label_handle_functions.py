import math
from ocr_functions import perform_ocr_on_frame, is_similar
from tts_engine import synthesize_speech
from gpt_functions import ask_gpt_about_object, ask_gpt_about_menu
from label_handler_list import (
    asset_dictionary,
    parameters_dictionary,
    action_dictionary,
)
from server import AudioPacket
import concurrent.futures

#converts 640x,640x view into coordinates on playcanvas
def calc_box_location(screen_x, screen_y, depth, viewport_width=640, viewport_height=640, fov=90):
    """
    Convert screen coordinates and depth to PlayCanvas world space coordinates.
    
    Args:
        screen_x, screen_y: Screen coordinates (pixels)
        depth: Depth value (0-255)
        viewport_width, viewport_height: Viewport dimensions (pixels)
        fov: Field of view (degrees)
        
    Returns:
        tuple: (x, y, z) world coordinates for PlayCanvas
    """
    # Convert screen coordinates to normalized device coordinates (-1 to 1)
    x_ndc = (2 * screen_x / viewport_width) - 1
    y_ndc = 1 - (2 * screen_y / viewport_height)  # Flip y for screen coordinates
    
    # Calculate field of view in radians and perspective scale
    fov_rad = math.radians(fov / 2)
    scale = math.tan(fov_rad)
    
    intermediate_depth = 0
    
    # Map depth from 0-255 to world units with better calibration
    min_distance = 1.0     # original val 3.5 - closer minimum
    max_distance = 80.0    # original val 30.0 - closer maximum
    
    #intermediate_depth = depth
    if (depth > max_distance):
        intermediate_depth = max_distance
    
    #if abs(screen_y) < max_distance:
    #screen_y = 320
    #if abs(screen_x) < max_distance:
    #screen_x = 320
        
    # Normalize depth and invert (255 = closest, 0 = furthest)
    normalized_depth = min(max(intermediate_depth, 0), 255) / 255.0
    # Convert to actual distance
    #z_distance = max_distance - normalized_depth
    z_distance = max_distance - normalized_depth * (max_distance - min_distance)
    
    # Scale factor to bring objects generally closer (reduced from previous calculations)
    scale_factor = 0.75
    screen_width = 320 # Original screen width in pixels
    
    # Calculate raw world x,y based on z distance and FOV
    x_world = x_ndc * z_distance * scale * scale_factor
    y_world = y_ndc * z_distance * scale * scale_factor
    
    x_final = -x_world   # Flip X because of 180 degree Y rotation
    y_final = y_world    # Keep Y as is (up is still up)
    z_final = z_distance * scale_factor  # Scale Z to bring objects closer
    
    # Print debug info if depth is significant
    # if depth > 30:
    #     print(f"Screen: ({screen_x},{screen_y}) Depth: {depth} → World: ({x_final:.2f},{y_final:.2f},{z_final:.2f})")
    
    #if (screen_x < screen_width): ## If the object is on the left side of the screen
            
         
    #print(f"Screen: ({screen_x},{screen_y}) Depth: {depth}")
    return (x_final, y_final, z_final)


#linear search algorithm to find action corresponding to label
def find_action_with_label(label):
    for key, value in action_dictionary.items():
        if label in value:
            return key
    return None


#linear search algorithm to find asset_id corresponding to label + emotion
def find_asset_with_label_emotion(label, emotion="neutral"):
    for key, value in asset_dictionary.items():
        if label in value and emotion in value:
            return key
    return None


#Uses dictionary nested within parameters_dictionary to remove parameters within boiler_plate
def parameter_removal(action, params):
    param_values = parameters_dictionary.get(action, {})

    keys_to_remove = [key for key in params if key in param_values and param_values[key] is False]

    for key in keys_to_remove:
        del params[key] #the original params in boiler_plate is automatically affected

def send_audio_packet(x=0, y=0, z=0, audio_server=None, asset_id=218331816, label="", audio_data=""):
    """Enhanced version with better error handling"""
    if audio_server is None:
        print("audio_server: None")
        return None
    
    try:
        # Ensure asset_id is always an integer (use default if None)
        if asset_id is None:
            asset_id = 218331816  # Default asset ID
            
        # Validate coordinates are actual numbers
        x, y, z = float(x), float(y), float(z)
        
        # Create and send packet
        packet = AudioPacket(
            x=x, 
            y=y, 
            z=z, 
            audio_data=audio_data, 
            label=label, 
            asset_id=int(asset_id))
            
        audio_server.schedule_broadcast(audio_server.broadcast_audio_packet(packet))
        return True
    except Exception as e:
        print(f"Error sending audio packet: {e}")
        # Try to send a fallback packet without audio data for basic notification
        try:
            if label:
                fallback_packet = AudioPacket(
                    x=float(x), 
                    y=float(y), 
                    z=float(z), 
                    audio_data="", 
                    label=f"{label}_fallback", 
                    asset_id=int(asset_id))
                audio_server.schedule_broadcast(audio_server.broadcast_audio_packet(fallback_packet))
        except:
            pass
        return False


#Handle all audio directly in this action function instead of passing it back
def action_ocr(frame=None, audio_server=None, x=0, y=0, z=0, label=None, emotion="neutral"):
    if audio_server is None:
        print("audio_server: None")
        return None

    else:
        text = None
        audio = None
        if frame is None:
            text = f"{label} in view"
        else:
            text = f"{label} detected"  # Immediate response
            
        audio = synthesize_speech(text, emotion)
            
        # Use default asset_id if not specified
        asset_id = find_asset_with_label_emotion(label, emotion)
        if asset_id is None:
            asset_id = 218331816  # Default value
            
        params = {"x":x, "y":y, "z":z, "audio_server":audio_server, "asset_id":asset_id, "label":label, "audio_data":audio}

        if audio:
            send_audio_packet(**params)
        else:
            print("audio: None")


def action_gpt(frame=None, audio_server=None, x=0, y=0, z=0, label=None, emotion="neutral"):
    if audio_server is None:
        print("audio_server: None")
        return None
    
    else:
        text = None
        audio = None
        if frame is None:
            text = f"{label} in view"
        else:
            text = f"{label} detected"  # Immediate response
            
        audio = synthesize_speech(text, emotion)
            
        # Use default asset_id if not specified
        asset_id = find_asset_with_label_emotion(label, emotion)
        if asset_id is None:
            asset_id = 218331816  # Default value
            
        params = {"x":x, "y":y, "z":z, "audio_server":audio_server, "asset_id":asset_id, "label":label, "audio_data":audio}

        if audio:
            send_audio_packet(**params)
        else:
            print("audio: None")


def action_spatial(audio_server=None, x=0, y=0, z=0, label=None, emotion="neutral"):
    if audio_server is None:
        print("audio_server: None")
        return None
    
    else: 
        # Retrieve asset_id, using default if not found
        asset_id = find_asset_with_label_emotion(label, emotion)
        if asset_id is None:
            asset_id = 218331816  # Default value
            
        params = {"x":x, "y":y, "z":z, "audio_server":audio_server, "label":label, "asset_id":asset_id}

        send_audio_packet(**params)


def action_avatars(x=0, y=0, z=0, audio_server=None, label=None, emotion="neutral"):
    if audio_server is None:
        print("audio_server: None")
        return None
    
    else:
        distance = z  # Use actual z coordinate
        audio = None
        text_distance = "far away"
        if distance > 200:
            text_distance = "close"
        elif distance > 50:
            text_distance = "nearby"
        text = f"Someone is {text_distance}"
        audio = synthesize_speech(text, emotion)
        
        # Use default asset_id if not specified
        asset_id = find_asset_with_label_emotion(label, emotion)
        if asset_id is None:
            asset_id = 218331816  # Default value
            
        params = {"x":x, "y":y, "z":z, "audio_server":audio_server, "asset_id":asset_id, "label":label, "audio_data":audio}

        if audio:
            send_audio_packet(**params)
        else:
            print("audio: None")
