"""
Interaction Detection handling line-object interactions, ray casting, and spatial proximity detection.
"""

import math
import threading
from geometry_utils import (
    calculate_box_midpoint, check_location_near_bbox, line_intersects_bbox,
    point_to_line_distance
)
from settings import EDGE_LABELS, INTERACTION_LABELS
from audio_utils import broadcast_immediate_audio


# Note: Depth map access now uses resource_manager.copy_resource('depth_map')


def get_object_depth(obj, current_depth_map=None):
    """
    Get depth value for an object from its bounding box center.

    Args:
        obj: Object with bbox information
        current_depth_map: Optional depth map to use instead of global

    Returns:
        int or None: Depth value or None if unavailable
    """
    try:
        bbox = obj.get('bbox')
        if not bbox or len(bbox) != 4:
            return None

        mx, my = calculate_box_midpoint(bbox)

        # Use provided depth map or get from resource manager
        depth_source = current_depth_map
        if depth_source is None:
            # Import here to avoid circular imports
            from main import resource_manager
            depth_source = resource_manager.copy_resource('depth_map')

        if depth_source is not None:
            h, w = depth_source.shape[:2]
            if 0 <= my < h and 0 <= mx < w:
                return int(depth_source[my, mx])

        return None
    except Exception as e:
        print(f"Error getting object depth: {e}")
        return None


def filter_interactive_objects(objects):
    """
    Filter objects to find interaction-relevant targets.

    Args:
        objects: List of detected objects

    Returns:
        list: Objects that can be interacted with
    """
    if not objects:
        return []

    interactive_labels = [
        'button', 'interactable', 'portal', 'menu',
        'sign-text', 'ui-text', 'sign-graphic', 'ui-graphic',
        'progress bar'
    ]

    return [obj for obj in objects
            if obj is not None and
            obj.get('label') in interactive_labels]


def filter_hand_objects(objects):
    """
    Filter objects to find hands and controllers.

    Args:
        objects: List of detected objects

    Returns:
        list: Hand/controller objects
    """
    if not objects:
        return []

    return [obj for obj in objects
            if obj is not None and
            obj.get('label') in EDGE_LABELS]


def calculate_line_object_distance(x1, y1, x2, y2, obj):
    """
    Calculate the minimum distance between a line and an object.

    Args:
        x1, y1, x2, y2: Line coordinates
        obj: Object with bbox

    Returns:
        float: Minimum distance in pixels
    """
    try:
        bbox = obj.get('bbox')
        if not bbox:
            return float('inf')

        obj_x, obj_y = calculate_box_midpoint(bbox)
        return point_to_line_distance(obj_x, obj_y, x1, y1, x2, y2)
    except Exception as e:
        print(f"Error calculating line-object distance: {e}")
        return float('inf')


def ray_bbox_intersect(ray_origin_x, ray_origin_y, ray_dir_x, ray_dir_y, bbox):
    """
    Check if a ray intersects with a bounding box using ray-box intersection.

    Args:
        ray_origin_x, ray_origin_y: Ray origin coordinates
        ray_dir_x, ray_dir_y: Ray direction vector
        bbox: Bounding box [x1, y1, x2, y2]

    Returns:
        tuple: (intersects: bool, distance: float)
    """
    try:
        if not bbox or len(bbox) != 4:
            return False, float('inf')

        x1, y1, x2, y2 = bbox

        # Avoid division by zero
        if abs(ray_dir_x) < 1e-6:
            ray_dir_x = 1e-6
        if abs(ray_dir_y) < 1e-6:
            ray_dir_y = 1e-6

        # Calculate intersection parameters
        t_min_x = (x1 - ray_origin_x) / ray_dir_x
        t_max_x = (x2 - ray_origin_x) / ray_dir_x
        t_min_y = (y1 - ray_origin_y) / ray_dir_y
        t_max_y = (y2 - ray_origin_y) / ray_dir_y

        # Ensure min/max are in correct order
        if t_min_x > t_max_x:
            t_min_x, t_max_x = t_max_x, t_min_x
        if t_min_y > t_max_y:
            t_min_y, t_max_y = t_max_y, t_min_y

        # Check for intersection
        t_min = max(t_min_x, t_min_y)
        t_max = min(t_max_x, t_max_y)

        # Ray intersects if t_max >= 0 and t_min <= t_max
        if t_max >= 0 and t_min <= t_max:
            # Distance is the closest intersection point
            distance = max(0, t_min) if t_min >= 0 else 0
            return True, distance

        return False, float('inf')
    except Exception as e:
        print(f"Error in ray_bbox_intersect: {e}")
        return False, float('inf')


def check_line_location(objects, x1, y1, x2, y2, hand_bbox=None, hand_label="",
                       audio_server=None, debug_frame=None, current_depth_map=None):
    """
    Enhanced check for line interactions with improved ray casting and depth validation.

    Args:
        objects: List of objects in the scene
        x1, y1, x2, y2: Coordinates of the line
        hand_bbox: Bounding box of the hand/controller
        hand_label: Label of the hand/controller
        audio_server: Server for sending audio packets
        debug_frame: Optional frame for drawing debug visualizations
        current_depth_map: Optional depth map for depth calculations

    Returns:
        Debug visualization frame if debug_frame is provided, otherwise None
    """
    try:
        # Skip if no objects
        if not objects:
            return debug_frame

        # Initialize debug visualization if a frame is provided
        debug_vis = None
        if debug_frame is not None:
            try:
                from vr_interaction_debug import DebugVisualizer
                debug_vis = DebugVisualizer(debug_frame)
                # Draw the original line
                debug_vis.draw_line(x1, y1, x2, y2, (0, 255, 0), 2)
                if hand_bbox:
                    debug_vis.draw_box(hand_bbox, (0, 255, 0), 2, f"{hand_label}")
            except ImportError:
                print("Debug visualization not available")

        # Basic parameters for detection
        threshold = 10  # Pixel proximity threshold
        max_reach = 15.0  # Maximum reach distance in world units
        extended_distance_threshold = 60  # Max pixel distance for ray casting

        # Get hand depth for reach calculation
        hand_z = None
        hand_mx, hand_my = None, None
        if hand_bbox:
            hand_mx, hand_my = calculate_box_midpoint(hand_bbox)

            # Use provided depth map or global reference
            depth_source = current_depth_map
            if depth_source is None:
                # Import here to avoid circular imports
                from main import resource_manager
                depth_source = resource_manager.copy_resource('depth_map')

            if depth_source is not None and 0 <= hand_my < depth_source.shape[0] and 0 <= hand_mx < depth_source.shape[1]:
                hand_z = depth_source[hand_my, hand_mx]

            if debug_vis:
                debug_vis.add_text(f"Hand depth: {hand_z}", 10, 30)

        # Prioritize important interactive elements
        priority_labels = ['menu', 'button', 'interactable', 'portal', 'spawner', 'target']

        # Calculate ray direction (normalized)
        line_dx = x2 - x1
        line_dy = y2 - y1
        line_length = math.sqrt(line_dx*line_dx + line_dy*line_dy)
        if line_length > 0:
            ray_dir_x = line_dx / line_length
            ray_dir_y = line_dy / line_length
        else:
            return debug_frame

        # Find all potential intersections
        intersections = []

        for obj in objects:
            if obj is None or 'bbox' not in obj:
                continue

            bbox = obj.get('bbox')
            label = obj.get('label', 'unknown')

            # Get object center and depth
            obj_x, obj_y = calculate_box_midpoint(bbox)
            obj_depth = get_object_depth(obj, current_depth_map)

            # Check ray-box intersection
            intersects, ray_distance = ray_bbox_intersect(x1, y1, ray_dir_x, ray_dir_y, bbox)

            if intersects:
                # Calculate pixel distance from line start
                pixel_distance = math.sqrt((obj_x - x1)**2 + (obj_y - y1)**2)

                # Check if within reasonable interaction distance
                if pixel_distance <= extended_distance_threshold:
                    # Calculate priority score
                    priority = priority_labels.index(label) if label in priority_labels else 999

                    # Check reach constraint if hand depth is available
                    within_reach = True
                    if hand_z is not None and obj_depth is not None:
                        depth_diff = abs(hand_z - obj_depth)
                        within_reach = depth_diff <= max_reach

                    intersections.append({
                        'object': obj,
                        'distance': pixel_distance,
                        'ray_distance': ray_distance,
                        'priority': priority,
                        'within_reach': within_reach,
                        'depth': obj_depth
                    })

                    if debug_vis:
                        color = (0, 255, 255) if within_reach else (0, 0, 255)
                        debug_vis.draw_box(bbox, color, 2, f"{label} ({priority})")

        # Sort by priority, then by distance
        intersections.sort(key=lambda x: (x['priority'], x['distance']))

        # Process the best intersection
        if intersections:
            best = intersections[0]
            obj = best['object']
            label = obj.get('label', 'unknown')

            if debug_vis:
                debug_vis.add_text(f"Target: {label}", 10, 60)
                debug_vis.add_text(f"Distance: {best['distance']:.1f}px", 10, 90)
                debug_vis.add_text(f"Reach: {'Yes' if best['within_reach'] else 'No'}", 10, 120)

            # Provide audio feedback if within reach
            if best['within_reach'] and audio_server:
                try:
                    obj_x, obj_y = calculate_box_midpoint(obj.get('bbox'))
                    obj_depth = best['depth'] or 0
                    broadcast_immediate_audio(
                        f"Pointing at {label}",
                        obj_x, obj_y, obj_depth,
                        f"interaction_{label}",
                        audio_server
                    )
                except Exception as e:
                    print(f"Error providing audio feedback: {e}")

        return debug_frame if debug_vis else None

    except Exception as e:
        print(f"Error in check_line_location: {e}")
        return debug_frame


def enhance_check_line_location(objects, lines, frame_copy, current_depth_map, audio_server, debug_frame=None):
    """
    Enhanced line interaction detection that processes all detected lines
    to identify objects being pointed at, prioritizing from hands/controllers.

    Args:
        objects: List of detected objects
        lines: List of detected lines
        frame_copy: Current frame
        current_depth_map: Current depth map
        audio_server: Audio server for feedback
        debug_frame: Optional debug visualization frame

    Returns:
        Debug frame if provided, otherwise None
    """
    if not objects or not lines:
        return debug_frame

    try:
        # Find hands and controllers
        hand_objects = filter_hand_objects(objects)

        # Find interaction-relevant objects (targets)
        interaction_objects = filter_interactive_objects(objects)

        # Return early if no interaction objects
        if not interaction_objects:
            return debug_frame

        # Add depth information to all objects
        for obj in objects:
            if obj is None or 'bbox' not in obj:
                continue
            obj['depth'] = get_object_depth(obj, current_depth_map)

        # Process all lines with the objects
        results = []

        # First, process lines from hands/controllers
        hand_lines = []
        other_lines = []

        for line in lines:
            if line is None:
                continue

            x1, y1, x2, y2 = line[0]
            is_hand_line = False

            # Check if line originates from a hand/controller
            for hand in hand_objects:
                hand_bbox = hand.get('bbox')
                if hand_bbox and (check_location_near_bbox(hand_bbox, x1, y1, 20) or
                                check_location_near_bbox(hand_bbox, x2, y2, 20)):
                    hand_lines.append((line, hand))
                    is_hand_line = True
                    break

            if not is_hand_line:
                other_lines.append(line)

        # Process hand-originated lines first (higher priority)
        for line, hand in hand_lines:
            x1, y1, x2, y2 = line[0]
            hand_bbox = hand.get('bbox')
            hand_label = hand.get('label', 'hand')

            result = check_line_location(
                interaction_objects, x1, y1, x2, y2,
                hand_bbox=hand_bbox, hand_label=hand_label,
                audio_server=audio_server, debug_frame=debug_frame,
                current_depth_map=current_depth_map
            )

            if result:
                results.append(result)

        # Process other lines if no hand interactions found
        if not results:
            for line in other_lines[:2]:  # Limit to first 2 non-hand lines
                x1, y1, x2, y2 = line[0]

                result = check_line_location(
                    interaction_objects, x1, y1, x2, y2,
                    audio_server=audio_server, debug_frame=debug_frame,
                    current_depth_map=current_depth_map
                )

                if result:
                    results.append(result)

        return debug_frame

    except Exception as e:
        print(f"Error in enhance_check_line_location: {e}")
        return debug_frame


def check_hand_proximity(hand_obj, objects, audio_server, debug_frame=None, current_depth_map=None):
    """
    Check for objects in proximity to a hand/controller and provide audio feedback.

    Args:
        hand_obj: Hand/controller object
        objects: List of all objects
        audio_server: Audio server for feedback
        debug_frame: Optional debug frame
        current_depth_map: Optional depth map

    Returns:
        Debug frame if provided, otherwise None
    """
    try:
        if not hand_obj or not objects:
            return debug_frame

        hand_bbox = hand_obj.get('bbox')
        if not hand_bbox:
            return debug_frame

        hand_x, hand_y = calculate_box_midpoint(hand_bbox)
        hand_depth = get_object_depth(hand_obj, current_depth_map)

        # Find nearby interactive objects
        nearby_objects = []
        proximity_threshold = 100  # pixels

        for obj in objects:
            if obj is None or obj == hand_obj:
                continue

            obj_bbox = obj.get('bbox')
            obj_label = obj.get('label')

            if not obj_bbox or not obj_label:
                continue

            # Skip non-interactive objects
            if obj_label not in INTERACTION_LABELS:
                continue

            obj_x, obj_y = calculate_box_midpoint(obj_bbox)
            distance = math.sqrt((hand_x - obj_x)**2 + (hand_y - obj_y)**2)

            if distance <= proximity_threshold:
                obj_depth = get_object_depth(obj, current_depth_map)
                nearby_objects.append({
                    'object': obj,
                    'distance': distance,
                    'depth': obj_depth
                })

        # Sort by distance and provide feedback for closest object
        if nearby_objects:
            nearby_objects.sort(key=lambda x: x['distance'])
            closest = nearby_objects[0]
            obj = closest['object']
            label = obj.get('label')

            if audio_server:
                try:
                    obj_bbox = obj.get('bbox')
                    obj_x, obj_y = calculate_box_midpoint(obj_bbox)
                    obj_depth = closest['depth'] or 0

                    broadcast_immediate_audio(
                        f"Near {label}",
                        obj_x, obj_y, obj_depth,
                        f"proximity_{label}",
                        audio_server
                    )
                except Exception as e:
                    print(f"Error providing proximity audio feedback: {e}")

        return debug_frame

    except Exception as e:
        print(f"Error in check_hand_proximity: {e}")
        return debug_frame