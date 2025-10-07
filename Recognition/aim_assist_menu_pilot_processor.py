"""
Aim Assist Menu Pilot Processor handling precise pointer-based object interaction and menu navigation.
"""

import time
import threading
import math
import cv2
import logging

from geometry_utils import (
    calculate_box_midpoint, check_location_near_bbox, line_intersects_bbox,
    point_to_line_distance
)
from settings import EDGE_LABELS, SPECIAL_LABELS, INTERACTION_LABELS
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


def find_closest_line_intersection(lines, objects, frame_shape):
    """
    Find objects that intersect with detected lines.

    Args:
        lines: List of detected line coordinates
        objects: List of detected objects
        frame_shape: Shape of the frame (height, width)

    Returns:
        list: Objects that intersect with lines, sorted by proximity
    """
    intersecting_objects = []

    if not lines or not objects:
        return intersecting_objects

    for line in lines:
        try:
            x1, y1, x2, y2 = line[0]

            for obj in objects:
                if obj is None:
                    continue

                bbox = obj.get('bbox')
                label = obj.get('label')

                if not bbox or not label:
                    continue

                # Skip special labels
                if label in SPECIAL_LABELS:
                    continue

                # Check if line intersects with object bounding box
                if line_intersects_bbox(x1, y1, x2, y2, bbox):
                    # Calculate distance from line midpoint to object center
                    line_mid_x = (x1 + x2) / 2
                    line_mid_y = (y1 + y2) / 2
                    obj_x, obj_y = calculate_box_midpoint(bbox)

                    distance = math.sqrt((line_mid_x - obj_x)**2 + (line_mid_y - obj_y)**2)

                    intersecting_objects.append((distance, obj))

        except Exception as e:
            print(f"Error processing line intersection: {e}")
            continue

    # Sort by distance (closest first) and remove duplicates
    intersecting_objects.sort(key=lambda x: x[0])
    unique_objects = []
    seen_objects = set()

    for distance, obj in intersecting_objects:
        obj_id = id(obj)
        if obj_id not in seen_objects:
            unique_objects.append(obj)
            seen_objects.add(obj_id)

    return unique_objects


def find_objects_near_hands(hand_objects, objects, proximity_threshold=100):
    """
    Find objects near hand/controller positions.

    Args:
        hand_objects: List of hand/controller objects
        objects: List of all detected objects
        proximity_threshold: Maximum distance for proximity detection

    Returns:
        list: Objects near hands, sorted by proximity
    """
    nearby_objects = []

    if not hand_objects or not objects:
        return nearby_objects

    for hand in hand_objects:
        if hand is None:
            continue

        hand_bbox = hand.get('bbox')
        if not hand_bbox:
            continue

        hand_x, hand_y = calculate_box_midpoint(hand_bbox)

        for obj in objects:
            if obj is None or obj == hand:
                continue

            obj_bbox = obj.get('bbox')
            obj_label = obj.get('label')

            if not obj_bbox or not obj_label:
                continue

            # Skip special labels and hand/controller labels
            if obj_label in SPECIAL_LABELS or obj_label in EDGE_LABELS:
                continue

            obj_x, obj_y = calculate_box_midpoint(obj_bbox)
            distance = math.sqrt((hand_x - obj_x)**2 + (hand_y - obj_y)**2)

            if distance <= proximity_threshold:
                nearby_objects.append((distance, obj))

    # Sort by distance and remove duplicates
    nearby_objects.sort(key=lambda x: x[0])
    unique_objects = []
    seen_objects = set()

    for distance, obj in nearby_objects:
        obj_id = id(obj)
        if obj_id not in seen_objects:
            unique_objects.append(obj)
            seen_objects.add(obj_id)

    return unique_objects


def process_targeted_objects(target_objects, frame_copy, current_depth_map,
                           cooldown_manager, audio_server, use_enhanced_gpt=True):
    """
    Process targeted objects with spatial audio and optional GPT analysis.

    Args:
        target_objects: List of objects to process
        frame_copy: Current frame
        current_depth_map: Current depth map
        cooldown_manager: Speech queue manager
        audio_server: Audio broadcasting server
        use_enhanced_gpt: Whether to use enhanced GPT analysis
    """
    try:
        for i, obj in enumerate(target_objects):
            if obj is None:
                continue

            label = obj.get('label')
            bbox = obj.get('bbox')

            if not label or not bbox:
                continue

            # Calculate position and depth
            obj_x, obj_y = calculate_box_midpoint(bbox)
            depth = 0

            if current_depth_map is not None:
                if 0 <= obj_y < current_depth_map.shape[0] and 0 <= obj_x < current_depth_map.shape[1]:
                    depth = current_depth_map[obj_y, obj_x]

            # Extract frame crop for GPT analysis if requested
            frame_crop = None
            if use_enhanced_gpt:
                if (0 <= bbox[1] < frame_copy.shape[0] and bbox[3] <= frame_copy.shape[0] and
                    0 <= bbox[0] < frame_copy.shape[1] and bbox[2] <= frame_copy.shape[1] and
                    bbox[1] < bbox[3] and bbox[0] < bbox[2]):

                    frame_crop = frame_copy[bbox[1]:bbox[3], bbox[0]:bbox[2]]

                    # Debug frame saving
                    if frame_crop.size > 0:
                        debug_filename = f"debug_aim_assist_{label}_{i}.png"
                        cv2.imwrite(debug_filename, frame_crop)

            # Create appropriate speech function
            if use_enhanced_gpt and frame_crop is not None:
                speech_func = create_enhanced_speech_function(
                    label, obj_x, obj_y, depth, label, audio_server,
                    frame_crop=frame_crop, use_gpt_analysis=True
                )
            else:
                speech_func = create_speech_function(
                    label, obj_x, obj_y, depth, label, audio_server
                )

            # Queue speech with priority based on relevance
            priority = i
            if label in INTERACTION_LABELS:
                priority -= 10  # Higher priority for interactive objects

            cooldown_manager.queue_speech(
                category="user_command",
                label=label,
                text=label,
                speech_func=speech_func,
                priority=priority
            )

        print(f"Processed {len(target_objects)} targeted objects")

    except Exception as e:
        print(f"Error processing targeted objects: {e}")
        import traceback
        traceback.print_exc()


def detect_hand_objects(objects):
    """Extract hand/controller objects from detection list."""
    return [obj for obj in objects
            if obj is not None and
            obj.get('label') in EDGE_LABELS and
            'bbox' in obj]


def process_aim_assist_menu_pilot(objects, frame_copy, current_depth_map, audio_server,
                                 cooldown_manager=None, edge_detections=None, edge_lock=None):
    """
    Process 1+2 combo to implement AimAssist and MenuPilot functionality.

    AimAssist: Reads objects near the pointer position without needing direct pointing
    MenuPilot: Reads out precise elements around the pointer when over a menu
    Falls back to objects near hand/controller if pointers aren't available

    Args:
        objects: Detected objects in the frame
        frame_copy: The current frame
        current_depth_map: The current depth map
        audio_server: Server for sending audio packets
        cooldown_manager: Speech queue and cooldown manager (optional for legacy compatibility)
        edge_detections: Current edge detections (optional)
        edge_lock: Lock for edge detection access (optional)

    Returns:
        bool: True to keep main loop running
    """
    try:
        # Get current lines if available
        current_lines = None
        if edge_lock and edge_detections is not None:
            with edge_lock:
                current_lines = edge_detections.copy() if edge_detections is not None else None

        # For legacy compatibility, handle missing cooldown_manager
        if cooldown_manager is None:
            print("Warning: cooldown_manager not provided, using immediate audio")
            # Fall back to immediate audio broadcast
            return process_aim_assist_legacy(objects, frame_copy, current_depth_map, audio_server, current_lines)

        # Clear any pending speech
        cooldown_manager.clear_queue()

        # Play audio feedback
        play_user_command_feedback("aim_assist", audio_server)

        # Mark start of user command execution
        cooldown_manager.start_user_command("aim_assist")

        print(f"Starting AimAssist/MenuPilot with {len(objects) if objects else 0} objects")
        logging.info("AimAssist/MenuPilot triggered")

        # Skip if no objects
        if not objects or len(objects) == 0:
            print("No objects to process in AimAssist")

            cooldown_manager.queue_speech(
                category="user_command",
                label="no_objects",
                text="No objects detected",
                speech_func=create_no_objects_speech_function(audio_server),
                priority=0
            )

            threading.Thread(target=monitor_command_completion, daemon=True).start()
            return True

        # Strategy 1: Look for line-object intersections first
        target_objects = []
        if current_lines and len(current_lines) > 0:
            print(f"Found {len(current_lines)} lines, checking for intersections...")
            target_objects = find_closest_line_intersection(current_lines, objects, frame_copy.shape)

        # Strategy 2: Fall back to objects near hands/controllers
        if not target_objects:
            print("No line intersections found, looking for objects near hands...")
            hand_objects = detect_hand_objects(objects)
            if hand_objects:
                target_objects = find_objects_near_hands(hand_objects, objects)

        # Strategy 3: Fall back to general object processing if no specific targets
        if not target_objects:
            print("No targeted objects found, processing all interactive objects...")
            # Filter for interactive objects first
            interactive_objects = [obj for obj in objects
                                 if obj is not None and
                                 obj.get('label') in INTERACTION_LABELS]

            if interactive_objects:
                target_objects = interactive_objects[:5]  # Limit to first 5
            else:
                # Process all non-special objects
                target_objects = [obj for obj in objects
                                if obj is not None and
                                obj.get('label') not in SPECIAL_LABELS][:3]  # Limit to first 3

        # Process the targeted objects
        if target_objects:
            print(f"Processing {len(target_objects)} targeted objects with enhanced analysis")
            process_targeted_objects(
                target_objects, frame_copy, current_depth_map,
                cooldown_manager, audio_server, use_enhanced_gpt=True
            )
        else:
            print("No suitable objects found for AimAssist processing")

        # Start monitoring for completion
        threading.Thread(target=monitor_command_completion, daemon=True).start()

    except Exception as e:
        print(f"ERROR in process_aim_assist_menu_pilot: {e}")
        import traceback
        traceback.print_exc()
        if cooldown_manager:
            cooldown_manager.end_user_command()

    return True


def process_aim_assist_legacy(objects, frame_copy, current_depth_map, audio_server, current_lines=None):
    """
    Legacy version for backward compatibility when cooldown_manager is not available.

    Args:
        objects: Detected objects in the frame
        frame_copy: The current frame
        current_depth_map: The current depth map
        audio_server: Server for sending audio packets
        current_lines: Current edge detections

    Returns:
        bool: True to keep main loop running
    """
    try:
        print("Running AimAssist in legacy mode (immediate audio)")

        if not objects:
            broadcast_immediate_centered_audio("No objects detected", "no_objects", audio_server)
            return True

        # Find target objects using the same strategies
        target_objects = []

        if current_lines and len(current_lines) > 0:
            target_objects = find_closest_line_intersection(current_lines, objects, frame_copy.shape)

        if not target_objects:
            hand_objects = detect_hand_objects(objects)
            if hand_objects:
                target_objects = find_objects_near_hands(hand_objects, objects)

        if not target_objects:
            # Process interactive objects
            target_objects = [obj for obj in objects
                            if obj is not None and
                            obj.get('label') in INTERACTION_LABELS][:3]

        # Broadcast immediate audio for each target
        for i, obj in enumerate(target_objects[:3]):  # Limit to 3 to avoid spam
            if obj is None:
                continue

            label = obj.get('label')
            bbox = obj.get('bbox')

            if not label or not bbox:
                continue

            obj_x, obj_y = calculate_box_midpoint(bbox)
            depth = 0

            if current_depth_map is not None:
                if 0 <= obj_y < current_depth_map.shape[0] and 0 <= obj_x < current_depth_map.shape[1]:
                    depth = current_depth_map[obj_y, obj_x]

            # Use immediate audio broadcast
            from audio_utils import broadcast_immediate_audio
            broadcast_immediate_audio(label, obj_x, obj_y, depth, label, audio_server)

            # Small delay between items
            time.sleep(0.5)

    except Exception as e:
        print(f"Error in legacy AimAssist: {e}")

    return True