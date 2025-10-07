"""
Audio utilities handling audioPacket creation, spatial audio positioning, and TTS management.
"""

from server import AudioPacket
from tts_engine import synthesize_speech, prepare_tts_text
from label_handle_functions import calc_box_location
import gpt_functions


def create_audio_packet(x, y, z, audio_data, label, asset_id=-1):
    """
    Create an AudioPacket with the given parameters.

    Args:
        x, y, z (float): 3D coordinates for spatial audio
        audio_data: Audio data from TTS synthesis
        label (str): Label for the audio packet
        asset_id (int): Asset ID for the audio packet

    Returns:
        AudioPacket: Configured audio packet
    """
    return AudioPacket(
        x=float(x),
        y=float(y),
        z=float(z),
        audio_data=audio_data,
        label=label,
        asset_id=asset_id
    )


def create_centered_audio_packet(audio_data, label, height=2.5, distance=10.0, asset_id=-1):
    """
    Create an AudioPacket positioned at center with specified height and distance.

    Args:
        audio_data: Audio data from TTS synthesis
        label (str): Label for the audio packet
        height (float): Y coordinate (height)
        distance (float): Z coordinate (distance)
        asset_id (int): Asset ID for the audio packet

    Returns:
        AudioPacket: Configured audio packet at center position
    """
    return AudioPacket(
        x=0.0,
        y=height,
        z=distance,
        audio_data=audio_data,
        label=label,
        asset_id=asset_id
    )


def create_spatial_audio_packet(bbox_x, bbox_y, depth, audio_data, label, asset_id=-1):
    """
    Create an AudioPacket with spatial positioning based on bounding box coordinates.

    Args:
        bbox_x, bbox_y (int): Bounding box center coordinates
        depth (float): Depth value from depth map
        audio_data: Audio data from TTS synthesis
        label (str): Label for the audio packet
        asset_id (int): Asset ID for the audio packet

    Returns:
        AudioPacket: Configured spatial audio packet
    """
    world_x, world_y, world_z = calc_box_location(bbox_x, bbox_y, depth)
    return create_audio_packet(world_x, world_y, world_z, audio_data, label, asset_id)


def synthesize_and_create_packet(text, bbox_x, bbox_y, depth, label, asset_id=-1):
    """
    Synthesize speech and create a spatial audio packet in one step.

    Args:
        text (str): Text to synthesize
        bbox_x, bbox_y (int): Bounding box center coordinates
        depth (float): Depth value from depth map
        label (str): Label for the audio packet
        asset_id (int): Asset ID for the audio packet

    Returns:
        AudioPacket or None: Configured audio packet, or None if synthesis failed
    """
    try:
        semantic_text = prepare_tts_text(text, label)
        audio_data = synthesize_speech(semantic_text, gpt_functions.get_current_emotion())

        if audio_data:
            return create_spatial_audio_packet(bbox_x, bbox_y, depth, audio_data, label, asset_id)
        return None
    except Exception as e:
        print(f"ERROR in synthesize_and_create_packet: {e}")
        return None


def synthesize_and_create_centered_packet(text, label, height=2.5, distance=10.0, asset_id=-1):
    """
    Synthesize speech and create a centered audio packet in one step.

    Args:
        text (str): Text to synthesize
        label (str): Label for the audio packet
        height (float): Y coordinate (height)
        distance (float): Z coordinate (distance)
        asset_id (int): Asset ID for the audio packet

    Returns:
        AudioPacket or None: Configured audio packet, or None if synthesis failed
    """
    try:
        semantic_text = prepare_tts_text(text, label)
        audio_data = synthesize_speech(semantic_text, gpt_functions.get_current_emotion())

        if audio_data:
            return create_centered_audio_packet(audio_data, label, height, distance, asset_id)
        return None
    except Exception as e:
        print(f"ERROR in synthesize_and_create_centered_packet: {e}")
        return None


def create_speech_function(text, bbox_x, bbox_y, depth, label, audio_server, asset_id=-1):
    """
    Create a speech function that can be queued for later execution.

    Args:
        text (str): Text to speak
        bbox_x, bbox_y (int): Bounding box center coordinates
        depth (float): Depth value from depth map
        label (str): Label for the audio packet
        audio_server: Audio server for broadcasting
        asset_id (int): Asset ID for the audio packet

    Returns:
        function: Speech function ready for queue execution
    """
    def speech_func():
        try:
            packet = synthesize_and_create_packet(text, bbox_x, bbox_y, depth, label, asset_id)
            if packet and audio_server:
                audio_server.schedule_broadcast(audio_server.broadcast_audio_packet(packet))
                print(f"Speaking: {text}")
        except Exception as e:
            print(f"ERROR in speech function: {e}")

    return speech_func


def create_centered_speech_function(text, label, audio_server, height=2.5, distance=10.0, asset_id=-1):
    """
    Create a centered speech function that can be queued for later execution.

    Args:
        text (str): Text to speak
        label (str): Label for the audio packet
        audio_server: Audio server for broadcasting
        height (float): Y coordinate (height)
        distance (float): Z coordinate (distance)
        asset_id (int): Asset ID for the audio packet

    Returns:
        function: Speech function ready for queue execution
    """
    def speech_func():
        try:
            packet = synthesize_and_create_centered_packet(text, label, height, distance, asset_id)
            if packet and audio_server:
                audio_server.schedule_broadcast(audio_server.broadcast_audio_packet(packet))
                print(f"Speaking: {text}")
        except Exception as e:
            print(f"ERROR in centered speech function: {e}")

    return speech_func


def create_enhanced_speech_function(text, bbox_x, bbox_y, depth, label, audio_server,
                                   frame_crop=None, use_gpt_analysis=False, asset_id=-1):
    """
    Create an enhanced speech function with optional GPT analysis for interactive objects.

    Args:
        text (str): Base text to speak
        bbox_x, bbox_y (int): Bounding box center coordinates
        depth (float): Depth value from depth map
        label (str): Label for the audio packet
        audio_server: Audio server for broadcasting
        frame_crop: Cropped frame for GPT analysis (optional)
        use_gpt_analysis (bool): Whether to use GPT for enhanced descriptions
        asset_id (int): Asset ID for the audio packet

    Returns:
        function: Enhanced speech function ready for queue execution
    """
    def speech_func():
        try:
            final_text = text

            if use_gpt_analysis and frame_crop is not None:
                interactive_labels = ["interactable", "sign-text", "ui-text", "sign-graphic",
                                    "ui-graphic", "menu", "button"]

                if label in interactive_labels:
                    try:
                        if label == "menu":
                            gpt_description = gpt_functions.ask_gpt_about_menu(frame_crop)
                        elif label in ["sign-text", "ui-text"]:
                            gpt_description = gpt_functions.ask_gpt_about_sign_object(frame_crop, label)
                        else:
                            gpt_description = gpt_functions.ask_gpt_about_object(frame_crop, label)

                        final_text = f"{text}: {gpt_description}"
                    except Exception as e:
                        print(f"GPT analysis failed for {label}: {e}")
                        final_text = text

            packet = synthesize_and_create_packet(final_text, bbox_x, bbox_y, depth, label, asset_id)
            if packet and audio_server:
                audio_server.schedule_broadcast(audio_server.broadcast_audio_packet(packet))
                print(f"Speaking: {final_text}")
        except Exception as e:
            print(f"ERROR in enhanced speech function: {e}")

    return speech_func


def broadcast_immediate_audio(text, bbox_x, bbox_y, depth, label, audio_server, asset_id=-1):
    """
    Synthesize and immediately broadcast audio without queueing.

    Args:
        text (str): Text to speak
        bbox_x, bbox_y (int): Bounding box center coordinates
        depth (float): Depth value from depth map
        label (str): Label for the audio packet
        audio_server: Audio server for broadcasting
        asset_id (int): Asset ID for the audio packet

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        packet = synthesize_and_create_packet(text, bbox_x, bbox_y, depth, label, asset_id)
        if packet and audio_server:
            audio_server.schedule_broadcast(audio_server.broadcast_audio_packet(packet))
            print(f"Speaking immediately: {text}")
            return True
        return False
    except Exception as e:
        print(f"ERROR in immediate audio broadcast: {e}")
        return False


def broadcast_immediate_centered_audio(text, label, audio_server, height=2.5, distance=10.0, asset_id=-1):
    """
    Synthesize and immediately broadcast centered audio without queueing.

    Args:
        text (str): Text to speak
        label (str): Label for the audio packet
        audio_server: Audio server for broadcasting
        height (float): Y coordinate (height)
        distance (float): Z coordinate (distance)
        asset_id (int): Asset ID for the audio packet

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        packet = synthesize_and_create_centered_packet(text, label, height, distance, asset_id)
        if packet and audio_server:
            audio_server.schedule_broadcast(audio_server.broadcast_audio_packet(packet))
            print(f"Speaking immediately: {text}")
            return True
        return False
    except Exception as e:
        print(f"ERROR in immediate centered audio broadcast: {e}")
        return False


def play_user_command_feedback(command_id, audio_server):
    """
    Play audio feedback for user commands.

    Args:
        command_id (str): ID of the command for feedback
        audio_server: Audio server for broadcasting

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        feedback_text = f"Command {command_id}"
        return broadcast_immediate_centered_audio(
            feedback_text,
            f"command_{command_id}_feedback",
            audio_server
        )
    except Exception as e:
        print(f"ERROR in play_user_command_feedback: {e}")
        return False


# Common audio packet creation patterns for specific scenarios

def create_no_objects_speech_function(audio_server):
    """Create speech function for 'no objects found' scenario."""
    return create_centered_speech_function("No objects found", "no_objects", audio_server)


def create_error_speech_function(error_message, audio_server):
    """Create speech function for error messages."""
    return create_centered_speech_function(f"ERROR: {error_message}", "error", audio_server)


def create_status_speech_function(status_message, audio_server):
    """Create speech function for status messages."""
    return create_centered_speech_function(status_message, "status", audio_server)