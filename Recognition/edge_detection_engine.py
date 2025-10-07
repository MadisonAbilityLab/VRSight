"""
Edge Detection Engine for VR-AI Scene Recognition.
Handles detection of VR pointer lines using color masking and Hough transforms.
"""

import time
import cv2
import numpy as np
import math
import threading

from settings import (
    EDGE_TOLERANCE, HOUGH_THRESHOLD, MIN_LINE_LENGTH, MAX_LINE_GAP,
    LINE_STRAIGHTNESS_THRESHOLD, EDGE_LABELS, POINTER_COLORS
)


def check_location_near_bbox(bbox, x, y, threshold=10):
    """
    Check if a point is near or inside a bounding box.

    Args:
        bbox (list): Bounding box coordinates [x1, y1, x2, y2]
        x (int): X coordinate to check
        y (int): Y coordinate to check
        threshold (int): Proximity threshold in pixels

    Returns:
        bool: True if point is near or inside the box
    """
    try:
        if bbox is None or len(bbox) != 4:
            return False

        x1, y1, x2, y2 = bbox
        return ((x1 - threshold <= x <= x2 + threshold) and
                (y1 - threshold <= y <= y2 + threshold))
    except Exception as e:
        print(f"Error in check_location_near_bbox: {e}")
        return False


class EdgeDetectionEngine:
    """Handles detection of VR pointer lines using color-based edge detection."""

    def __init__(self):
        """Initialize the edge detection engine."""
        self.edge_tolerance = EDGE_TOLERANCE
        self.pointer_colors = POINTER_COLORS
        self.morphology_kernel = np.ones((2, 2), np.uint8)

    def process_frame(self, frame, hand_objects=None):
        """
        Process a frame to detect pointer lines.

        Args:
            frame: Input BGR frame
            hand_objects: List of detected hand/controller objects

        Returns:
            tuple: (detected_lines, visualization_frame, fps)
        """
        if frame is None or frame.shape[0] <= 0 or frame.shape[1] <= 0:
            return None, frame, 0

        start_time = time.perf_counter()
        vis_frame = frame.copy()
        frame_height, frame_width = frame.shape[:2]

        try:
            # Filter for hand objects if provided
            valid_hands = []
            if hand_objects:
                valid_hands = [obj for obj in hand_objects
                             if obj is not None and
                             obj.get('label') in EDGE_LABELS and
                             'bbox' in obj]

            # Convert to HSV for color detection
            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # Create color masks for different pointer colors
            masks = []
            for color_name, color_range in self.pointer_colors.items():
                lower = np.array(color_range['lower'])
                upper = np.array(color_range['upper'])
                mask = cv2.inRange(hsv_frame, lower, upper)
                masks.append(mask)

            # Combine all color masks
            combined_mask = masks[0]
            for mask in masks[1:]:
                combined_mask = cv2.bitwise_or(combined_mask, mask)

            # Apply morphological operations to clean up the mask
            processed_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, self.morphology_kernel)

            # Time the core Hough line detection
            detection_start = time.perf_counter()
            pointer_lines = cv2.HoughLinesP(
                processed_mask, 1, np.pi/180,
                threshold=HOUGH_THRESHOLD,
                minLineLength=MIN_LINE_LENGTH,
                maxLineGap=MAX_LINE_GAP
            )
            detection_time = time.perf_counter() - detection_start
            detection_fps = 1 / detection_time if detection_time > 0 else 0

            if pointer_lines is None:
                # No lines found
                pipeline_fps = 1 / (time.perf_counter() - start_time)
                cv2.putText(vis_frame, f'Detection: {detection_fps:.1f} FPS', (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(vis_frame, f'Pipeline: {pipeline_fps:.1f} FPS', (10, 55),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                return None, vis_frame, detection_fps

            # Validate and score detected lines
            candidate_lines = self._validate_lines(pointer_lines, frame_width, frame_height, valid_hands)

            # Select best lines
            final_lines = self._select_best_lines(candidate_lines)

            # Visualize selected lines
            self._visualize_lines(vis_frame, final_lines)

            # Calculate performance metrics
            process_time = time.perf_counter() - start_time
            fps = 1 / process_time if process_time > 0 else 0

            # Calculate pipeline performance
            pipeline_fps = 1 / (time.perf_counter() - start_time)

            # Add performance info to visualization
            cv2.putText(vis_frame, f'Detection: {detection_fps:.1f} FPS', (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(vis_frame, f'Pipeline: {pipeline_fps:.1f} FPS', (10, 55),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(vis_frame, f'Lines: {len(final_lines)}', (10, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            return final_lines if final_lines else None, vis_frame, detection_fps

        except Exception as e:
            print(f"Warning in edge detection: {e}")
            pipeline_fps = 1 / (time.perf_counter() - start_time)
            cv2.putText(vis_frame, f'Pipeline: {pipeline_fps:.1f} FPS', (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            return None, vis_frame, 0  # No detection fps available on error

    def _validate_lines(self, lines, frame_width, frame_height, hand_objects):
        """
        Validate detected lines based on geometric and contextual criteria.

        Args:
            lines: Raw lines from Hough transform
            frame_width: Frame width
            frame_height: Frame height
            hand_objects: List of hand/controller objects

        Returns:
            list: List of (line, score) tuples for valid lines
        """
        candidate_lines = []

        for line in lines:
            x1, y1, x2, y2 = line[0]

            # Calculate line length
            line_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

            # Skip short lines
            if line_length < MIN_LINE_LENGTH:
                continue

            # Check if line originates from frame edge
            from_edge = self._is_line_from_edge(x1, y1, x2, y2, frame_width, frame_height)

            # Check if line is near a hand/controller
            near_hand = self._is_line_near_hand(x1, y1, x2, y2, hand_objects)

            # Check if line is straight (not curved)
            is_straight = self._is_straight_line(x1, y1, x2, y2)

            # Only accept lines that meet our criteria
            if (from_edge or near_hand) and is_straight:
                score = self._calculate_line_score(
                    x1, y1, x2, y2, line_length, frame_height, from_edge, near_hand
                )
                candidate_lines.append((line, score))

        return candidate_lines

    def _is_line_from_edge(self, x1, y1, x2, y2, frame_width, frame_height):
        """Check if a line originates from any frame edge."""
        edge_tol = self.edge_tolerance

        # Check if either endpoint is near an edge with proper direction
        checks = [
            # Left edge
            (x1 < edge_tol and x2 > x1 + 20) or (x2 < edge_tol and x1 > x2 + 20),
            # Right edge
            (x1 > frame_width - edge_tol and x2 < x1 - 20) or (x2 > frame_width - edge_tol and x1 < x2 - 20),
            # Top edge
            (y1 < edge_tol and y2 > y1 + 20) or (y2 < edge_tol and y1 > y2 + 20),
            # Bottom edge
            (y1 > frame_height - edge_tol and y2 < y1 - 20) or (y2 > frame_height - edge_tol and y1 < y2 - 20)
        ]

        return any(checks)

    def _is_line_near_hand(self, x1, y1, x2, y2, hand_objects):
        """Check if a line is near any hand/controller object."""
        if not hand_objects:
            return False

        for hand in hand_objects:
            hand_bbox = hand.get('bbox')
            if hand_bbox and (check_location_near_bbox(hand_bbox, x1, y1, 20) or
                             check_location_near_bbox(hand_bbox, x2, y2, 20)):
                return True

        return False

    def _is_straight_line(self, x1, y1, x2, y2):
        """
        Check if a line is straight by sampling points along it.
        Returns True if the line maintains straightness within tolerance.
        """
        line_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        num_samples = min(int(line_length / 10) + 2, 10)  # Sample every ~10px, max 10

        if num_samples < 3:
            return True  # Too short to check curvature

        # Generate sample points along the line
        line_points = []
        for i in range(num_samples):
            t = i / (num_samples - 1)
            x = int(x1 + t * (x2 - x1))
            y = int(y1 + t * (y2 - y1))
            line_points.append((x, y))

        # Calculate line equation: ax + by + c = 0
        a = y1 - y2
        b = x2 - x1
        c = x1*y2 - x2*y1
        norm = np.sqrt(a*a + b*b)

        if norm < 1e-6:
            return True  # Avoid division by zero

        # Count points that are close to the ideal straight line
        straight_count = 0
        for px, py in line_points[1:-1]:  # Skip endpoints
            # Distance from point to line
            dist = abs(a*px + b*py + c) / norm
            if dist < 3.0:  # Allow small deviation (3 pixels)
                straight_count += 1

        # Calculate straightness ratio
        straightness_ratio = straight_count / (num_samples - 2) if num_samples > 2 else 1.0

        return straightness_ratio >= LINE_STRAIGHTNESS_THRESHOLD

    def _calculate_line_score(self, x1, y1, x2, y2, line_length, frame_height, from_edge, near_hand):
        """Calculate a score for line quality and relevance."""
        score = line_length * 2  # Longer lines get priority

        if from_edge:
            score += 200  # Bonus for edge lines
        if near_hand:
            score += 100  # Bonus for hand-connected lines

        # Penalize lines from bottom of frame (often platform edges)
        bottom_y_threshold = frame_height - 50
        if y1 > bottom_y_threshold and y2 > bottom_y_threshold:
            score -= 150  # Heavy penalty for lines near bottom edge

        return score

    def _select_best_lines(self, candidate_lines, max_lines=3):
        """Select the best lines from candidates, removing duplicates."""
        if not candidate_lines:
            return []

        # Sort by score (higher is better)
        candidate_lines.sort(key=lambda x: -x[1])

        # Take the highest scoring line first
        final_lines = [candidate_lines[0][0]]

        # Add remaining lines if they're different enough
        for line, _ in candidate_lines[1:]:
            if not self._is_line_too_similar(line, final_lines) and len(final_lines) < max_lines:
                final_lines.append(line)

        return final_lines

    def _is_line_too_similar(self, new_line, existing_lines):
        """Check if a new line is too similar to existing lines."""
        x1, y1, x2, y2 = new_line[0]
        new_angle = math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180

        for line in existing_lines:
            ex1, ey1, ex2, ey2 = line[0]
            existing_angle = math.degrees(math.atan2(ey2 - ey1, ex2 - ex1)) % 180

            # Check angle similarity
            angle_diff = min(abs(new_angle - existing_angle), 180 - abs(new_angle - existing_angle))
            if angle_diff < 20:  # Similar angles
                # Check spatial proximity
                mid_x1 = (x1 + x2) / 2
                mid_y1 = (y1 + y2) / 2
                mid_x2 = (ex1 + ex2) / 2
                mid_y2 = (ey1 + ey2) / 2

                # Distance between midpoints
                dist = math.sqrt((mid_x1 - mid_x2)**2 + (mid_y1 - mid_y2)**2)
                if dist < 100:  # Close to each other
                    return True

        return False

    def _visualize_lines(self, vis_frame, lines):
        """Draw detected lines on the visualization frame."""
        colors = [(0, 255, 0), (255, 255, 0), (255, 0, 255)]  # Green, Yellow, Magenta

        for i, line in enumerate(lines):
            x1, y1, x2, y2 = line[0]
            color = colors[i % len(colors)]
            cv2.line(vis_frame, (x1, y1), (x2, y2), color, 2)


def run_edge_detection_thread(engine, frame_source, objects_source, result_callback,
                             thread_coordinator, running_flag):
    """
    Run edge detection in a dedicated thread.

    Args:
        engine: EdgeDetectionEngine instance
        frame_source: Function that returns the current frame
        objects_source: Function that returns current object detections
        result_callback: Function called with (detected_lines, visualization_frame)
        thread_coordinator: ThreadCoordinator instance for monitoring
        running_flag: Shared flag indicating if thread should continue
    """
    thread_name = "edge_detection"
    thread_coordinator.register_thread(thread_name)
    print("Edge detection thread started.")

    while running_flag[0]:  # Use list to allow modification from other threads
        try:
            thread_coordinator.heartbeat(thread_name)

            # Get current frame and objects
            frame = frame_source()
            if frame is None:
                continue

            objects = objects_source()

            # Extract hand objects for line validation
            hand_objects = []
            if objects:
                hand_objects = [obj for obj in objects
                              if obj is not None and
                              obj.get('label') in EDGE_LABELS and
                              'bbox' in obj]

            # Process frame for edge detection
            detected_lines, vis_frame, fps = engine.process_frame(frame, hand_objects)

            # Deliver results via callback
            result_callback(detected_lines, vis_frame)

        except Exception as e:
            thread_coordinator.increment_error(thread_name)
            print(f"ERROR in edge detection thread: {e}")
            # Continue running despite errors