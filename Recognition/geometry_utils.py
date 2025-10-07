"""
Geometry utilities handling coordinate transformations, bounding box operations, and spatial calculations.
"""


def calculate_box_midpoint(bbox):
    """
    Calculate the midpoint of a bounding box.

    Args:
        bbox (list): Bounding box coordinates [x1, y1, x2, y2]

    Returns:
        tuple: (midpoint_x, midpoint_y) coordinates
    """
    midpoint_x = int((bbox[0] + bbox[2]) / 2.0)
    midpoint_y = int((bbox[1] + bbox[3]) / 2.0)

    return midpoint_x, midpoint_y


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


def check_bbox_overlap(bbox1, bbox2, threshold=0.2):
    """
    Check if two bounding boxes overlap.

    Args:
        bbox1 (list): First bounding box [x1, y1, x2, y2]
        bbox2 (list): Second bounding box [x1, y1, x2, y2]
        threshold (float): Overlap ratio threshold (0.0 to 1.0)

    Returns:
        bool: True if boxes overlap above threshold
    """
    try:
        if bbox1 is None or bbox2 is None:
            return False

        left = max(bbox1[0], bbox2[0])
        right = min(bbox1[2], bbox2[2])
        top = max(bbox1[1], bbox2[1])
        bottom = min(bbox1[3], bbox2[3])

        if right < left or bottom < top:
            return False

        intersect = (right - left) * (bottom - top)

        box1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        box2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

        min_area = min(box1_area, box2_area)
        if min_area <= 0:
            return False

        return (intersect / min_area) > threshold
    except Exception as e:
        print(f"Error in check_bbox_overlap: {e}")
        return False


def calculate_distance(bbox1, bbox2):
    """
    Calculate the Euclidean distance between the centers of two bounding boxes.

    Args:
        bbox1 (list): First bounding box [x1, y1, x2, y2]
        bbox2 (list): Second bounding box [x1, y1, x2, y2]

    Returns:
        float: Distance between box centers, or inf on error
    """
    try:
        if bbox1 is None or bbox2 is None:
            return float('inf')

        x1, y1 = calculate_box_midpoint(bbox1)
        x2, y2 = calculate_box_midpoint(bbox2)
        return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    except Exception as e:
        print(f"Error in calculate_distance: {e}")
        return float('inf')


def expand_bbox(bbox, expansion_pixels):
    """
    Expand a bounding box by a specified number of pixels.

    Args:
        bbox (list): Bounding box coordinates [x1, y1, x2, y2]
        expansion_pixels (int): Number of pixels to expand in each direction

    Returns:
        list: Expanded bounding box [x1, y1, x2, y2]
    """
    if bbox is None or len(bbox) != 4:
        return bbox

    x1, y1, x2, y2 = bbox
    return [
        x1 - expansion_pixels,
        y1 - expansion_pixels,
        x2 + expansion_pixels,
        y2 + expansion_pixels
    ]


def clamp_bbox_to_frame(bbox, frame_width, frame_height):
    """
    Clamp bounding box coordinates to frame boundaries.

    Args:
        bbox (list): Bounding box coordinates [x1, y1, x2, y2]
        frame_width (int): Frame width
        frame_height (int): Frame height

    Returns:
        list: Clamped bounding box [x1, y1, x2, y2]
    """
    if bbox is None or len(bbox) != 4:
        return bbox

    x1, y1, x2, y2 = bbox
    return [
        max(0, min(x1, frame_width - 1)),
        max(0, min(y1, frame_height - 1)),
        max(0, min(x2, frame_width - 1)),
        max(0, min(y2, frame_height - 1))
    ]


def calculate_bbox_area(bbox):
    """
    Calculate the area of a bounding box.

    Args:
        bbox (list): Bounding box coordinates [x1, y1, x2, y2]

    Returns:
        int: Area in pixels, or 0 on error
    """
    try:
        if bbox is None or len(bbox) != 4:
            return 0

        x1, y1, x2, y2 = bbox
        width = max(0, x2 - x1)
        height = max(0, y2 - y1)
        return width * height
    except Exception as e:
        print(f"Error in calculate_bbox_area: {e}")
        return 0


def get_bbox_corners(bbox):
    """
    Get all four corners of a bounding box.

    Args:
        bbox (list): Bounding box coordinates [x1, y1, x2, y2]

    Returns:
        list: List of (x, y) corner coordinates
    """
    if bbox is None or len(bbox) != 4:
        return []

    x1, y1, x2, y2 = bbox
    return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]


def point_to_line_distance(px, py, x1, y1, x2, y2):
    """
    Calculate the perpendicular distance from a point to a line segment.

    Args:
        px, py (float): Point coordinates
        x1, y1, x2, y2 (float): Line segment endpoints

    Returns:
        float: Distance from point to line
    """
    try:
        # Calculate line equation: ax + by + c = 0
        a = y1 - y2
        b = x2 - x1
        c = x1*y2 - x2*y1

        # Calculate norm to avoid division by zero
        norm = np.sqrt(a*a + b*b)
        if norm < 1e-6:
            return float('inf')

        # Distance formula: |ax + by + c| / sqrt(a² + b²)
        return abs(a*px + b*py + c) / norm
    except Exception as e:
        print(f"Error in point_to_line_distance: {e}")
        return float('inf')


def line_intersects_bbox(x1, y1, x2, y2, bbox):
    """
    Check if a line segment intersects with a bounding box.

    Args:
        x1, y1, x2, y2 (float): Line segment endpoints
        bbox (list): Bounding box coordinates [bx1, by1, bx2, by2]

    Returns:
        bool: True if line intersects the bounding box
    """
    try:
        if bbox is None or len(bbox) != 4:
            return False

        bx1, by1, bx2, by2 = bbox

        # Check if either endpoint is inside the box
        if ((bx1 <= x1 <= bx2 and by1 <= y1 <= by2) or
            (bx1 <= x2 <= bx2 and by1 <= y2 <= by2)):
            return True

        # Check line-rectangle intersection using line-line intersection
        # for each edge of the bounding box
        def lines_intersect(x1, y1, x2, y2, x3, y3, x4, y4):
            denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
            if abs(denom) < 1e-10:
                return False

            t = ((x1-x3)*(y3-y4) - (y1-y3)*(x3-x4)) / denom
            u = -((x1-x2)*(y1-y3) - (y1-y2)*(x1-x3)) / denom

            return 0 <= t <= 1 and 0 <= u <= 1

        # Check intersection with each edge of the rectangle
        edges = [
            (bx1, by1, bx2, by1),  # Top edge
            (bx2, by1, bx2, by2),  # Right edge
            (bx2, by2, bx1, by2),  # Bottom edge
            (bx1, by2, bx1, by1)   # Left edge
        ]

        for ex1, ey1, ex2, ey2 in edges:
            if lines_intersect(x1, y1, x2, y2, ex1, ey1, ex2, ey2):
                return True

        return False
    except Exception as e:
        print(f"Error in line_intersects_bbox: {e}")
        return False