"""
Aim Assist Processor handling hand/controller detection with GPT analysis and aim assist functionality.
"""

import time
import threading
import cv2
import logging

from geometry_utils import calculate_box_midpoint, expand_bbox
from settings import EDGE_LABELS, SPECIAL_LABELS
from audio_utils import (
    create_speech_function, create_enhanced_speech_function,
    broadcast_immediate_centered_audio, play_user_command_feedback,
    create_no_objects_speech_function
)
import gpt_functions


def monitor_command_completion():
    """Monitor speech queue completion and end command when done."""
    # This would need to be implemented based on the cooldown_manager interface
    pass


def detect_hand_objects(objects):
    """
    Extract hand/controller objects from detection list.

    Args:
        objects: List of detected objects

    Returns:
        list: Hand/controller objects with valid bounding boxes
    """
    return [obj for obj in objects
            if obj is not None and
            obj.get('label') in EDGE_LABELS and
            'bbox' in obj]


def process_hand_analysis(hand_obj, frame_copy, cooldown_manager, audio_server):
    """
    Analyze hand/controller region using GPT and queue speech.

    Args:
        hand_obj: Hand/controller detection object
        frame_copy: Current frame
        cooldown_manager: Speech queue manager
        audio_server: Audio broadcasting server

    Returns:
        bool: True if analysis was successful
    """
    try:
        hand_bbox = hand_obj.get('bbox')
        if not hand_bbox:
            return False

        h_mx, h_my = calculate_box_midpoint(hand_bbox)

        # Expand bounding box for better context
        expanded_bbox = expand_bbox(hand_bbox, 25)
        expanded_x1, expanded_y1, expanded_x2, expanded_y2 = expanded_bbox

        # Clamp to frame boundaries
        expanded_x1 = max(0, expanded_x1)
        expanded_y1 = max(0, expanded_y1)
        expanded_x2 = min(frame_copy.shape[1], expanded_x2)
        expanded_y2 = min(frame_copy.shape[0], expanded_y2)

        # Validate dimensions
        if expanded_x1 >= expanded_x2 or expanded_y1 >= expanded_y2:
            return False

        # Extract expanded region
        expanded_hand_region = frame_copy[expanded_y1:expanded_y2, expanded_x1:expanded_x2]

        # Get GPT analysis
        hand_label = hand_obj.get('label')
        description = gpt_functions.ask_gpt_about_hand(expanded_hand_region, hand_label)

        if description:
            # Get depth information
            depth = 0
            # This would need current_depth_map passed in if needed

            # Create speech function
            speech_func = create_speech_function(
                description, h_mx, h_my, depth, "hand_analysis", audio_server
            )

            # Queue the speech
            cooldown_manager.queue_speech(
                category="user_command",
                label="hand_analysis",
                text=description,
                speech_func=speech_func,
                priority=0
            )

            # Start monitoring for completion
            threading.Thread(target=monitor_command_completion, daemon=True).start()
            return True

    except Exception as e:
        print(f"Error in hand analysis: {e}")

    return False


def process_objects_with_gpt(objects, frame_copy, current_depth_map, cooldown_manager, audio_server):
    """
    Process all objects with GPT analysis in left-to-right order.

    Args:
        objects: List of detected objects
        frame_copy: Current frame
        current_depth_map: Current depth map
        cooldown_manager: Speech queue manager
        audio_server: Audio broadcasting server
    """
    try:
        # Filter and sort objects (excluding special labels)
        filtered_objects = []
        for obj in objects:
            if obj is None:
                continue

            label = obj.get('label')
            bbox = obj.get('bbox')

            if label is None or bbox is None:
                continue

            # Skip special labels that are processed elsewhere
            if label in SPECIAL_LABELS:
                continue

            mx, my = calculate_box_midpoint(bbox)

            # Get depth information
            depth = 0
            if current_depth_map is not None:
                if 0 <= my < current_depth_map.shape[0] and 0 <= mx < current_depth_map.shape[1]:
                    depth = current_depth_map[my, mx]

            filtered_objects.append((mx, my, depth, obj))

        # Sort by x-coordinate (left to right)
        filtered_objects.sort(key=lambda x: x[0])

        # Process each object with GPT analysis
        for i, (mx, my, depth, obj) in enumerate(filtered_objects):
            label = obj.get('label')
            bbox = obj.get('bbox')

            # Extract frame crop for GPT analysis
            frame_crop = None
            if (bbox and
                0 <= bbox[1] < frame_copy.shape[0] and bbox[3] <= frame_copy.shape[0] and
                0 <= bbox[0] < frame_copy.shape[1] and bbox[2] <= frame_copy.shape[1] and
                bbox[1] < bbox[3] and bbox[0] < bbox[2]):

                frame_crop = frame_copy[bbox[1]:bbox[3], bbox[0]:bbox[2]]

                # Debug frame saving
                if frame_crop.size > 0:
                    debug_filename = f"debug_combo_{label}_{i}.png"
                    cv2.imwrite(debug_filename, frame_crop)

            # Create enhanced speech function with GPT analysis
            speech_func = create_enhanced_speech_function(
                label, mx, my, depth, label, audio_server,
                frame_crop=frame_crop, use_gpt_analysis=True
            )

            # Queue speech
            cooldown_manager.queue_speech(
                category="user_command",
                label=label,
                text=label,
                speech_func=speech_func,
                priority=i
            )

        print(f"AimAssist processing completed for {len(filtered_objects)} objects")

        # Start monitoring for completion
        threading.Thread(target=monitor_command_completion, daemon=True).start()

    except Exception as e:
        print(f"Error in process_objects_with_gpt: {e}")
        import traceback
        traceback.print_exc()


def process_command_1_2_combo(objects, frame_copy, current_depth_map, audio_server,
                             cooldown_manager, edge_detections=None, edge_lock=None):
    """
    Process Command 1+2 combo: Hand detection with GPT analysis or object reading.

    Args:
        objects: Detected objects in the frame
        frame_copy: The current frame
        current_depth_map: The current depth map
        audio_server: Server for sending audio packets
        cooldown_manager: Speech queue and cooldown manager
        edge_detections: Current edge detections (optional)
        edge_lock: Lock for edge detection access (optional)

    Returns:
        bool: True to keep main loop running
    """
    try:
        # Check for edge/line detections if available
        current_lines = None
        if edge_lock and edge_detections is not None:
            with edge_lock:
                current_lines = edge_detections.copy() if edge_detections is not None else None

        # Check if there are hands/controllers detected
        hand_objects = detect_hand_objects(objects)

        # If we have hands, delegate to aim assist/menu pilot functionality
        if hand_objects:
            # Import here to avoid circular imports
            from aim_assist_menu_pilot_processor import process_aim_assist_menu_pilot
            return process_aim_assist_menu_pilot(objects, frame_copy, current_depth_map, audio_server)

        # Otherwise, fall back to reading all objects with GPT analysis
        print("No pointers detected. Processing all objects with GPT...")

        # Clear any pending speech
        cooldown_manager.clear_queue()

        # Play audio feedback
        play_user_command_feedback("combo", audio_server)

        # Mark start of user command execution
        cooldown_manager.start_user_command("combo")

        print(f"Starting AimAssist with {len(objects) if objects else 0} objects")
        logging.info("Combo triggered")

        # Skip if no objects
        if not objects or len(objects) == 0:
            print("No objects to process in Command combo")

            # Queue no objects speech
            cooldown_manager.queue_speech(
                category="user_command",
                label="no_objects",
                text="No objects detected",
                speech_func=create_no_objects_speech_function(audio_server),
                priority=0
            )

            # Start monitoring for completion
            threading.Thread(target=monitor_command_completion, daemon=True).start()
            return True

        # Check for hand objects again for fallback hand analysis
        hand_objects = detect_hand_objects(objects)

        if hand_objects:
            # Process first detected hand
            hand_obj = hand_objects[0]
            if process_hand_analysis(hand_obj, frame_copy, cooldown_manager, audio_server):
                return True

        # Process all objects with GPT analysis
        process_objects_with_gpt(objects, frame_copy, current_depth_map, cooldown_manager, audio_server)

    except Exception as e:
        print(f"ERROR in process_command_1_2_combo: {e}")
        import traceback
        traceback.print_exc()
        cooldown_manager.end_user_command()

    return True