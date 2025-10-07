"""
Handles SceneSweep/Command 2: Read all labels from left to right with queued speech.
"""

import time
import threading
import cv2

from geometry_utils import calculate_box_midpoint, check_bbox_overlap
from settings import SPECIAL_LABELS, EDGE_LABELS
from ocr_functions import perform_ocr_on_frame
from audio_utils import (
    create_speech_function, create_centered_speech_function,
    create_enhanced_speech_function, play_user_command_feedback,
    create_no_objects_speech_function
)
import gpt_functions


def monitor_command_completion():
    """Monitor speech queue completion and end command when done."""
    # This would need to be implemented based on the cooldown_manager interface
    pass


def looks_like_username(text):
    """Check if OCR text looks like a username."""
    if not text:
        return False
    return len(text) < 30 and text.count(' ') <= 1


def process_scenesweep(objects, frame_copy, current_depth_map, audio_server, cooldown_manager):
    """
    Process Command 2: Read all labels from left to right with queued speech.
    No depth filtering - reads all objects regardless of depth.

    Args:
        objects: Detected objects in the frame
        frame_copy: The current frame
        current_depth_map: The current depth map
        audio_server: Server for sending audio packets
        cooldown_manager: Speech queue and cooldown manager

    Returns:
        bool: True to keep main loop running
    """
    try:
        # Clear any pending speech
        cooldown_manager.clear_queue()

        print(f"Starting SceneSweep with {len(objects) if objects else 0} objects")

        play_user_command_feedback("1", audio_server)

        # Mark start of user command execution
        cooldown_manager.start_user_command("1")

        # Skip if no objects
        if not objects or len(objects) == 0:
            print("No objects to process; skipping...")

            # Queue speech for no objects
            cooldown_manager.queue_speech(
                category="user_command",
                label="no_objects",
                text="No objects found",
                speech_func=create_no_objects_speech_function(audio_server),
                priority=0
            )

            # Start monitoring for completion
            threading.Thread(target=monitor_command_completion, daemon=True).start()
            return True

        time.sleep(0.3)  # Short delay to allow speech to queue

        # Setup detection categories
        avatar_detections = []
        sign_text_detections = []
        seat_detections = []
        portal_detections = []
        sorted_detections = []

        # First pass: categorize detections (WITHOUT depth filtering)
        for detection in objects:
            try:
                if detection is None:
                    continue

                label = detection.get('label')
                if label is None:
                    continue

                bbox = detection.get('bbox')
                if bbox is None:
                    continue

                mx, my = calculate_box_midpoint(bbox)

                # Skip special labels (already processed)
                if label in SPECIAL_LABELS:
                    continue

                # Get depth for information only (NOT for filtering)
                depth = 0
                if current_depth_map is not None:
                    if 0 <= my < current_depth_map.shape[0] and 0 <= mx < current_depth_map.shape[1]:
                        depth = current_depth_map[my, mx]

                # Categorize by label type without any depth filtering
                if label in ['avatar', 'avatar-nonhuman']:
                    avatar_detections.append((mx, my, depth, detection))
                elif label == 'sign-text':
                    sign_text_detections.append((mx, my, depth, detection))
                elif label in ['seat-single', 'seat-multiple']:
                    seat_detections.append((mx, my, depth, detection))
                elif label == 'portal':
                    portal_detections.append((mx, my, depth, detection))
                else:
                    # Everything else gets sorted left-to-right
                    sorted_detections.append((mx, my, depth, detection))
            except Exception as e:
                print(f"ERROR categorizing detection: {e}")
                continue

        # Process avatar-sign attribution
        avatars_with_names, avatars_without_names, attributed_sign_indices = process_avatar_name_attribution(
            avatar_detections, sign_text_detections, frame_copy
        )

        # Process portal-sign attribution
        portals_with_destinations, portals_without_destinations = process_portal_destination_attribution(
            portal_detections, sign_text_detections, attributed_sign_indices, frame_copy
        )

        # Process seat occupancy
        vacant_singles, vacant_multiples = process_seat_occupancy(seat_detections, avatar_detections)

        # Queue all speech in proper order
        priority_counter = queue_scenesweep_speech(
            avatars_with_names, avatars_without_names,
            portals_with_destinations, portals_without_destinations,
            vacant_singles, vacant_multiples,
            sign_text_detections, attributed_sign_indices,
            sorted_detections, frame_copy,
            cooldown_manager, audio_server
        )

        print("SceneSweep Finished Processing!")

        # Start monitoring for completion
        threading.Thread(target=monitor_command_completion, daemon=True).start()

    except Exception as e:
        print(f"ERROR in process_scenesweep: {e}")
        import traceback
        traceback.print_exc()
        cooldown_manager.end_user_command()

    return True


def process_avatar_name_attribution(avatar_detections, sign_text_detections, frame_copy):
    """Attribute sign-text to avatars as names."""
    avatars_with_names = []
    avatars_without_names = []
    attributed_sign_indices = set()

    for mx, my, depth, avatar in avatar_detections:
        avatar_bbox = avatar.get('bbox')
        best_sign_idx = None
        best_sign_text = None
        best_distance = float('inf')

        # Find the closest sign-text above the avatar
        for idx, (sx, sy, s_depth, sign) in enumerate(sign_text_detections):
            sign_bbox = sign.get('bbox')

            # Check if sign is above avatar (sign bottom y < avatar top y)
            if sign_bbox[3] < avatar_bbox[1] + 20:  # Allow slight overlap
                # Check horizontal overlap
                if (sign_bbox[0] < avatar_bbox[2] and sign_bbox[2] > avatar_bbox[0]):
                    # Calculate vertical distance
                    vertical_dist = avatar_bbox[1] - sign_bbox[3]

                    # Check if this is closer than previous best
                    if vertical_dist < best_distance and vertical_dist < 50:
                        # Try to perform OCR to check if it looks like a username
                        if (0 <= sign_bbox[1] < frame_copy.shape[0] and
                            0 <= sign_bbox[3] <= frame_copy.shape[0] and
                            0 <= sign_bbox[0] < frame_copy.shape[1] and
                            0 <= sign_bbox[2] <= frame_copy.shape[1] and
                            sign_bbox[1] < sign_bbox[3] and sign_bbox[0] < sign_bbox[2]):

                            sign_frame = frame_copy[sign_bbox[1]:sign_bbox[3], sign_bbox[0]:sign_bbox[2]]
                            ocr_text = perform_ocr_on_frame(sign_frame)

                            if ocr_text and looks_like_username(ocr_text):
                                best_sign_idx = idx
                                best_sign_text = ocr_text
                                best_distance = vertical_dist

        # Store avatar with or without name
        if best_sign_idx is not None and best_sign_text:
            avatars_with_names.append((mx, my, depth, avatar, best_sign_text))
            attributed_sign_indices.add(best_sign_idx)
        else:
            avatars_without_names.append((mx, my, depth, avatar))

    return avatars_with_names, avatars_without_names, attributed_sign_indices


def process_portal_destination_attribution(portal_detections, sign_text_detections, attributed_sign_indices, frame_copy):
    """Attribute sign-text to portals as destinations."""
    portals_with_destinations = []
    portals_without_destinations = []

    for mx, my, depth, portal in portal_detections:
        portal_bbox = portal.get('bbox')
        best_sign_idx = None
        best_sign_text = None
        best_distance = float('inf')

        # Find the closest unattributed sign-text near the portal
        for idx, (sx, sy, s_depth, sign) in enumerate(sign_text_detections):
            if idx in attributed_sign_indices:
                continue  # Skip already attributed sign-text

            sign_bbox = sign.get('bbox')

            # Calculate center-to-center distance
            distance = math.sqrt((mx - sx)**2 + (my - sy)**2)

            if distance < 100 and distance < best_distance:
                # Try to perform OCR
                if (0 <= sign_bbox[1] < frame_copy.shape[0] and
                    0 <= sign_bbox[3] <= frame_copy.shape[0] and
                    0 <= sign_bbox[0] < frame_copy.shape[1] and
                    0 <= sign_bbox[2] <= frame_copy.shape[1] and
                    sign_bbox[1] < sign_bbox[3] and sign_bbox[0] < sign_bbox[2]):

                    sign_frame = frame_copy[sign_bbox[1]:sign_bbox[3], sign_bbox[0]:sign_bbox[2]]
                    ocr_text = perform_ocr_on_frame(sign_frame)

                    if ocr_text:
                        best_sign_idx = idx
                        best_sign_text = ocr_text
                        best_distance = distance

        # Store portal with or without destination
        if best_sign_idx is not None and best_sign_text:
            portals_with_destinations.append((mx, my, depth, portal, best_sign_text))
            attributed_sign_indices.add(best_sign_idx)
        else:
            portals_without_destinations.append((mx, my, depth, portal))

    return portals_with_destinations, portals_without_destinations


def process_seat_occupancy(seat_detections, avatar_detections):
    """Determine which seats are vacant."""
    occupied_seat_ids = set()

    for idx, (mx, my, depth, seat) in enumerate(seat_detections):
        seat_bbox = seat.get('bbox')
        seat_id = id(seat)

        # Check which avatars overlap with this seat
        for _, _, _, avatar in avatar_detections:
            avatar_bbox = avatar.get('bbox')
            if check_bbox_overlap(seat_bbox, avatar_bbox, threshold=0.3):
                occupied_seat_ids.add(seat_id)
                break

    # Separate vacant seats by type
    vacant_singles = []
    vacant_multiples = []

    for idx, (mx, my, depth, seat) in enumerate(seat_detections):
        seat_id = id(seat)
        if seat_id not in occupied_seat_ids:
            if seat.get('label') == 'seat-single':
                vacant_singles.append((mx, my, depth, seat))
            elif seat.get('label') == 'seat-multiple':
                vacant_multiples.append((mx, my, depth, seat))

    return vacant_singles, vacant_multiples


def queue_scenesweep_speech(avatars_with_names, avatars_without_names,
                           portals_with_destinations, portals_without_destinations,
                           vacant_singles, vacant_multiples,
                           sign_text_detections, attributed_sign_indices,
                           sorted_detections, frame_copy,
                           cooldown_manager, audio_server):
    """Queue all speech for scene sweep in proper order."""
    priority = 0

    # 1. Process avatars with names
    for i, (mx, my, depth, avatar, name) in enumerate(avatars_with_names):
        distance_desc = "far"
        if depth > 200:
            distance_desc = "close"
        elif depth > 50:
            distance_desc = "nearby"

        speech_text = f"{name} is {distance_desc}"
        speech_func = create_speech_function(speech_text, mx, my, depth, "avatar_name", audio_server)

        cooldown_manager.queue_speech(
            category="user_command",
            label="avatar_name",
            text=speech_text,
            speech_func=speech_func,
            priority=priority + i
        )

    priority += len(avatars_with_names)

    # 2. Process avatars without names
    for i, (mx, my, depth, avatar) in enumerate(avatars_without_names):
        distance_desc = "far"
        if depth > 200:
            distance_desc = "close"
        elif depth > 50:
            distance_desc = "nearby"

        speech_text = f"Someone is {distance_desc}"
        speech_func = create_speech_function(speech_text, mx, my, depth, "avatar_unnamed", audio_server)

        cooldown_manager.queue_speech(
            category="user_command",
            label="avatar_unnamed",
            text=speech_text,
            speech_func=speech_func,
            priority=priority + i
        )

    priority += len(avatars_without_names)

    # 3. Process portals with destinations
    for i, (mx, my, depth, portal, destination) in enumerate(portals_with_destinations):
        speech_text = f"Portal to {destination}"
        speech_func = create_speech_function(speech_text, mx, my, depth, "portal_destination", audio_server)

        cooldown_manager.queue_speech(
            category="user_command",
            label="portal_destination",
            text=speech_text,
            speech_func=speech_func,
            priority=priority + i
        )

    priority += len(portals_with_destinations)

    # 4. Process portals without destinations
    for i, (mx, my, depth, portal) in enumerate(portals_without_destinations):
        speech_text = "Portal"
        speech_func = create_speech_function(speech_text, mx, my, depth, "portal", audio_server)

        cooldown_manager.queue_speech(
            category="user_command",
            label="portal",
            text=speech_text,
            speech_func=speech_func,
            priority=priority + i
        )

    priority += len(portals_without_destinations)

    # 5. Process vacant seats
    if vacant_singles:
        seat_count = len(vacant_singles)
        speech_text = f"{seat_count} open single seats" if seat_count > 1 else "Open seat"
        mx, my, depth, first_seat = vacant_singles[0]
        speech_func = create_speech_function(speech_text, mx, my, depth, "vacant_seats", audio_server)

        cooldown_manager.queue_speech(
            category="user_command",
            label="vacant_seats",
            text=speech_text,
            speech_func=speech_func,
            priority=priority
        )
        priority += 1

    if vacant_multiples:
        seat_count = len(vacant_multiples)
        speech_text = f"{seat_count} open group seats" if seat_count > 1 else "Open group seat"
        mx, my, depth, first_seat = vacant_multiples[0]
        speech_func = create_speech_function(speech_text, mx, my, depth, "vacant_group_seats", audio_server)

        cooldown_manager.queue_speech(
            category="user_command",
            label="vacant_group_seats",
            text=speech_text,
            speech_func=speech_func,
            priority=priority
        )
        priority += 1

    # 6. Process remaining sign-text
    remaining_sign_text = []
    for idx, (mx, my, depth, sign) in enumerate(sign_text_detections):
        if idx not in attributed_sign_indices:
            remaining_sign_text.append((mx, my, depth, sign))

    # Sort remaining sign-text left-to-right
    remaining_sign_text.sort(key=lambda x: x[0])

    for i, (mx, my, depth, sign) in enumerate(remaining_sign_text):
        sign_bbox = sign.get('bbox')

        sign_text = "Sign text"
        if (0 <= sign_bbox[1] < frame_copy.shape[0] and
            0 <= sign_bbox[3] <= frame_copy.shape[0] and
            0 <= sign_bbox[0] < frame_copy.shape[1] and
            0 <= sign_bbox[2] <= frame_copy.shape[1] and
            sign_bbox[1] < sign_bbox[3] and sign_bbox[0] < sign_bbox[2]):

            sign_frame = frame_copy[sign_bbox[1]:sign_bbox[3], sign_bbox[0]:sign_bbox[2]]
            ocr_text = perform_ocr_on_frame(sign_frame)

            if ocr_text:
                sign_text = f"Sign text: {ocr_text}"

        speech_func = create_speech_function(sign_text, mx, my, depth, "sign_text", audio_server)

        cooldown_manager.queue_speech(
            category="user_command",
            label="sign_text",
            text=sign_text,
            speech_func=speech_func,
            priority=priority + i
        )

    priority += len(remaining_sign_text)

    # 7. Process all other detections left-to-right
    sorted_detections.sort(key=lambda x: x[0])

    for i, (mx, my, depth, detection) in enumerate(sorted_detections):
        label = detection.get('label')
        print("Label:", label, "mx:", mx, "my:", my, "depth:", depth)

        # Extract frame crop for GPT analysis
        bbox = detection.get('bbox')
        frame_crop = None
        if bbox is not None and bbox[1] < frame_copy.shape[0] and bbox[3] <= frame_copy.shape[0] and \
           bbox[0] < frame_copy.shape[1] and bbox[2] <= frame_copy.shape[1]:
            frame_crop = frame_copy[bbox[1]:bbox[3], bbox[0]:bbox[2]]

            # Debug frame saving
            if frame_crop.size > 0:
                debug_filename = f"debug_bbox_{label}_{i}.png"
                cv2.imwrite(debug_filename, frame_crop)

        speech_func = create_enhanced_speech_function(
            label, mx, my, depth, label, audio_server,
            frame_crop=frame_crop, use_gpt_analysis=True
        )

        cooldown_manager.queue_speech(
            category="user_command",
            label=label,
            text=label,
            speech_func=speech_func,
            priority=priority + i
        )

    return priority + len(sorted_detections)