import cv2
import numpy as np
import threading
import queue
import time
import argparse

class DebugVisualizer:
    """
    Helper class to create debug visualizations for interaction detection.
    Provides methods to visualize ray casting, object selection, and depth measurements.
    """
    def __init__(self, frame):
        """Initialize with a frame to draw on"""
        if frame is not None:
            self.frame = frame.copy()
        else:
            self.frame = None
            
    def draw_box(self, bbox, color=(0, 255, 0), thickness=2, label=None):
        """Draw a bounding box on the frame with optional label"""
        if self.frame is None or bbox is None:
            return
            
        try:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(self.frame, (x1, y1), (x2, y2), color, thickness)
            
            if label:
                # Draw label background
                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                cv2.rectangle(self.frame, (x1, y1-20), (x1 + text_size[0], y1), color, -1)
                # Draw label text
                cv2.putText(self.frame, label, (x1, y1-5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        except Exception as e:
            print(f"Error drawing box: {e}")
            
    def draw_line(self, x1, y1, x2, y2, color=(0, 255, 0), thickness=2, label=None):
        """Draw a line on the frame with optional label"""
        if self.frame is None:
            return
            
        try:
            cv2.line(self.frame, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
            
            if label:
                # Calculate midpoint for label
                mid_x = int((x1 + x2) / 2)
                mid_y = int((y1 + y2) / 2)
                
                # Draw label background
                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                cv2.rectangle(self.frame, 
                              (mid_x, mid_y-20), 
                              (mid_x + text_size[0], mid_y), 
                              color, -1)
                              
                # Draw label text
                cv2.putText(self.frame, label, (mid_x, mid_y-5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        except Exception as e:
            print(f"Error drawing line: {e}")
            
    def add_circle(self, x, y, radius, color=(0, 255, 0), thickness=2):
        """Draw a circle on the frame"""
        if self.frame is None:
            return
            
        try:
            cv2.circle(self.frame, (int(x), int(y)), radius, color, thickness)
        except Exception as e:
            print(f"Error drawing circle: {e}")
            
    def add_text(self, text, x, y, color=(255, 255, 255), font_scale=0.6):
        """Add text to the frame with background for better visibility"""
        if self.frame is None or text is None:
            return
            
        try:
            # Draw background rectangle for better visibility
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)[0]
            cv2.rectangle(self.frame, 
                          (x-5, y-text_size[1]-5), 
                          (x + text_size[0]+5, y+5), 
                          (0, 0, 0), -1)
                          
            # Draw text
            cv2.putText(self.frame, text, (x, y), 
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1)
        except Exception as e:
            print(f"Error adding text: {e}")
            
    def add_decision_marker(self, x, y, is_selected, reason=None):
        """Add a check mark or X mark to indicate selection decision"""
        if self.frame is None:
            return
            
        try:
            if is_selected:
                # Draw a check mark (green)
                mark = "✓"
                color = (0, 255, 0)
            else:
                # Draw an X mark (red)
                mark = "✗"
                color = (0, 0, 255)
            
            # Draw the mark
            cv2.putText(self.frame, mark, (int(x), int(y)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
                        
            # Draw reason if provided
            if reason:
                cv2.putText(self.frame, reason, (int(x) + 20, int(y)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        except Exception as e:
            print(f"Error adding decision marker: {e}")
            
    def highlight_object(self, bbox, is_selected=False):
        """Highlight an object with a thick outline to show it's being processed"""
        if self.frame is None or bbox is None:
            return
            
        try:
            # Cyan color for object being processed
            color = (0, 255, 255) if not is_selected else (0, 255, 0)
            thickness = 3
            
            x1, y1, x2, y2 = bbox
            # Draw a thick outline
            cv2.rectangle(self.frame, (x1, y1), (x2, y2), color, thickness)
            
            # Draw diagonal lines in corners to make it more visible
            corner_length = 15
            # Top-left corner
            cv2.line(self.frame, (x1, y1), (x1 + corner_length, y1), color, thickness)
            cv2.line(self.frame, (x1, y1), (x1, y1 + corner_length), color, thickness)
            # Top-right corner
            cv2.line(self.frame, (x2, y1), (x2 - corner_length, y1), color, thickness)
            cv2.line(self.frame, (x2, y1), (x2, y1 + corner_length), color, thickness)
            # Bottom-left corner
            cv2.line(self.frame, (x1, y2), (x1 + corner_length, y2), color, thickness)
            cv2.line(self.frame, (x1, y2), (x1, y2 - corner_length), color, thickness)
            # Bottom-right corner
            cv2.line(self.frame, (x2, y2), (x2 - corner_length, y2), color, thickness)
            cv2.line(self.frame, (x2, y2), (x2, y2 - corner_length), color, thickness)
        except Exception as e:
            print(f"Error highlighting object: {e}")
    
    def draw_depth_indicator(self, x1, y1, x2, y2, depth1, depth2, max_depth_diff, within_range=True):
        """Draw a visual indicator of depth difference between two points"""
        if self.frame is None:
            return
            
        try:
            # Draw a line connecting the points
            line_color = (0, 255, 0) if within_range else (0, 0, 255)
            cv2.line(self.frame, (int(x1), int(y1)), (int(x2), int(y2)), line_color, 1)
            
            # Calculate midpoint for depth difference display
            mid_x = int((x1 + x2) / 2)
            mid_y = int((y1 + y2) / 2)
            
            # Calculate depth difference
            depth_diff = abs(depth1 - depth2)
            
            # Display depth difference
            text = f"Δz: {depth_diff:.1f}/{max_depth_diff:.1f}"
            
            # Add background for better visibility
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            bg_color = (0, 100, 0) if within_range else (100, 0, 0)
            cv2.rectangle(self.frame,
                         (mid_x - 5, mid_y - text_size[1] - 5),
                         (mid_x + text_size[0] + 5, mid_y + 5),
                         bg_color, -1)
            
            # Add text
            text_color = (255, 255, 255)
            cv2.putText(self.frame, text, (mid_x, mid_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
        except Exception as e:
            print(f"Error drawing depth indicator: {e}")
    
    def add_title(self, title, mode="Interaction Debug"):
        """Add a title banner at the top of the frame"""
        if self.frame is None or title is None:
            return
            
        try:
            # Create a dark banner across the top
            banner_height = 30
            cv2.rectangle(self.frame, 
                         (0, 0), 
                         (self.frame.shape[1], banner_height), 
                         (40, 40, 40), -1)
                         
            # Add title text
            cv2.putText(self.frame, f"{mode}: {title}", (10, 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                       
            # Add date/time stamp if needed
            import datetime
            timestamp = datetime.datetime.now().strftime("%H:%M:%S")
            time_size = cv2.getTextSize(timestamp, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            time_x = self.frame.shape[1] - time_size[0] - 10
            cv2.putText(self.frame, timestamp, (time_x, 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        except Exception as e:
            print(f"Error adding title: {e}")
    
    def visualize_ray_hit(self, ray_origin_x, ray_origin_y, hit_x, hit_y, object_bbox=None, hit_distance=None):
        """Visualize a ray hit with the object"""
        if self.frame is None:
            return
            
        try:
            # Draw the ray
            cv2.line(self.frame, 
                    (int(ray_origin_x), int(ray_origin_y)), 
                    (int(hit_x), int(hit_y)), 
                    (255, 0, 255), 2)
                    
            # Draw hit point
            cv2.circle(self.frame, (int(hit_x), int(hit_y)), 5, (255, 0, 255), -1)
            
            # Add hit distance label if provided
            if hit_distance is not None:
                distance_text = f"Dist: {hit_distance:.1f}px"
                mid_x = int((ray_origin_x + hit_x) / 2)
                mid_y = int((ray_origin_y + hit_y) / 2)
                
                # Add text with background
                text_size = cv2.getTextSize(distance_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                cv2.rectangle(self.frame,
                             (mid_x - 5, mid_y - text_size[1] - 5),
                             (mid_x + text_size[0] + 5, mid_y + 5),
                             (100, 0, 100), -1)
                cv2.putText(self.frame, distance_text, (mid_x, mid_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
            # Highlight hit object if provided
            if object_bbox is not None:
                self.highlight_object(object_bbox, True)
        except Exception as e:
            print(f"Error visualizing ray hit: {e}")
    
    def get_frame(self):
        """Return the debug visualization frame"""
        return self.frame


class InteractionDebugVisualizer:
    """
    Comprehensive debugging visualization system for VR interaction detection.
    
    This class provides detailed visual feedback for:
    1. Line interactions
    2. Hand proximity checks
    3. Object selection process
    4. Depth and spatial relationship visualization
    """
    
    def __init__(self, enable_debug=False):
        """
        Initialize the debug visualization system.
        
        Args:
            enable_debug (bool): Flag to enable/disable debug visualization
        """
        self.debug_mode = enable_debug
        self.debug_windows = {
            'interaction': {
                'name': 'Interaction Debug',
                'frame': None,
                'enabled': False
            },
            'proximity': {
                'name': 'Hand Proximity Debug', 
                'frame': None,
                'enabled': False
            }
        }
        
        # Thread-safe queue for debug frames
        self.debug_frame_queue = queue.Queue(maxsize=10)
        
        # Debug configuration
        self.debug_config = {
            'ray_color': (0, 0, 255),  # Red for ray casting
            'extended_ray_color': (255, 0, 0),  # Blue for extended ray
            'proximity_color': (0, 255, 255),  # Yellow for proximity
            'object_colors': {
                'within_reach': (0, 255, 0),  # Green
                'out_of_reach': (0, 0, 255),  # Red
                'near_threshold': (0, 255, 255),  # Yellow
                'selected': (0, 255, 0)  # Green
            },
            'line_thickness': 2,
            'text_scale': 0.5,
            'text_color': (255, 255, 255)  # White
        }
        
        # Status indicators
        self.status_indicators = {
            'debug_enabled': False,
            'interaction_enabled': False,
            'proximity_enabled': False
        }
        
        # Window size and positioning
        self.window_positions = {
            'interaction': (50, 50),
            'proximity': (700, 50)
        }
        
        # Track if windows have been created
        self.windows_created = {
            'interaction': False,
            'proximity': False
        }
        
        # Start debug visualization thread
        if self.debug_mode:
            self.debug_thread = threading.Thread(target=self._debug_window_manager, daemon=True)
            self.debug_thread.start()

    def toggle_debug_mode(self, mode='all'):
        """
        Toggle debug visualization for a specific mode.
        
        Args:
            mode (str): Debug mode to toggle ('all', 'interaction', or 'proximity')
        """
        if mode == 'all':
            # Toggle overall debug mode
            self.debug_mode = not self.debug_mode
            self.status_indicators['debug_enabled'] = self.debug_mode
            print(f"Debug mode: {'Enabled' if self.debug_mode else 'Disabled'}")
            
            # If turning off debug mode, disable all windows
            if not self.debug_mode:
                for key in self.debug_windows:
                    self.debug_windows[key]['enabled'] = False
                    self.status_indicators[f'{key}_enabled'] = False
                
                # Close all windows
                cv2.destroyAllWindows()
                for key in self.windows_created:
                    self.windows_created[key] = False
            
            # If turning on debug mode and thread not running, start it
            elif not hasattr(self, 'debug_thread') or not self.debug_thread.is_alive():
                self.debug_thread = threading.Thread(target=self._debug_window_manager, daemon=True)
                self.debug_thread.start()
                
            return
            
        # Toggle specific mode
        if mode not in self.debug_windows:
            print(f"Invalid debug mode: {mode}")
            return
        
        # Only toggle if overall debug mode is enabled
        if self.debug_mode:
            self.debug_windows[mode]['enabled'] = not self.debug_windows[mode]['enabled']
            self.status_indicators[f'{mode}_enabled'] = self.debug_windows[mode]['enabled']
            print(f"{mode.capitalize()} debug mode: {'Enabled' if self.debug_windows[mode]['enabled'] else 'Disabled'}")
            
            # Create window if enabled
            if self.debug_windows[mode]['enabled'] and not self.windows_created[mode]:
                # Create the window in the main thread
                try:
                    win_name = self.debug_windows[mode]['name']
                    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
                    cv2.moveWindow(win_name, *self.window_positions[mode])
                    cv2.resizeWindow(win_name, 640, 480)
                    self.windows_created[mode] = True
                    
                    # Initialize with a blank frame if needed
                    blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(blank_frame, f"Waiting for {mode} debug data...", 
                                (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    self.debug_windows[mode]['frame'] = blank_frame
                    cv2.imshow(win_name, blank_frame)
                    cv2.waitKey(1)  # This is critical to actually show the window
                    
                    print(f"Created debug window: {win_name}")
                except Exception as window_error:
                    print(f"ERROR creating {mode} window: {window_error}")
                    self.debug_windows[mode]['enabled'] = False
                    self.status_indicators[f'{mode}_enabled'] = False
            
            # Destroy window if disabled
            elif not self.debug_windows[mode]['enabled'] and self.windows_created[mode]:
                try:
                    cv2.destroyWindow(self.debug_windows[mode]['name'])
                    self.windows_created[mode] = False
                    print(f"Closed debug window: {self.debug_windows[mode]['name']}")
                except Exception as e:
                    print(f"Error closing window: {e}")
                    
            # Force a redraw to make windows appear immediately
            if self.debug_windows[mode]['enabled'] and self.debug_windows[mode]['frame'] is not None:
                try:
                    cv2.imshow(self.debug_windows[mode]['name'], self.debug_windows[mode]['frame'])
                    cv2.waitKey(1)
                except Exception as e:
                    print(f"Error in immediate redraw: {e}")
        else:
            print("Overall debug mode is disabled. Enable it first.")

    def _debug_window_manager(self):
        """
        Dedicated thread to manage debug window display and updates.
        """
        print("Debug window manager thread started.")
        
        # Use a separate thread for window creation to avoid blocking
        def create_window(mode, name, position, size):
            try:
                cv2.namedWindow(name, cv2.WINDOW_NORMAL)
                cv2.moveWindow(name, position[0], position[1])
                cv2.resizeWindow(name, size[0], size[1])
                blank_frame = np.zeros((size[1], size[0], 3), dtype=np.uint8)
                cv2.putText(blank_frame, f"Waiting for {mode} debug data...", 
                            (50, size[1]//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.imshow(name, blank_frame)
                cv2.waitKey(1)  # Critical to get the window to show
                print(f"Created window for {mode}")
                return True
            except Exception as e:
                print(f"Failed to create window for {mode}: {e}")
                return False
        
        while True:
            try:
                # Sleep briefly to avoid CPU hogging
                time.sleep(0.05)
                
                # Skip if debug mode is disabled
                if not self.debug_mode:
                    time.sleep(0.5)
                    continue
                
                # Process enabled windows
                for mode, window in self.debug_windows.items():
                    if not window['enabled']:
                        continue
                        
                    # Create window if needed
                    if window['enabled'] and not self.windows_created.get(mode, False):
                        # Use timeout to avoid hanging
                        window_creation = threading.Thread(
                            target=create_window,
                            args=(mode, window['name'], 
                                self.window_positions[mode], (640, 480))
                        )
                        window_creation.daemon = True
                        window_creation.start()
                        window_creation.join(timeout=1.0)
                        
                        self.windows_created[mode] = True
                    
                    # Skip if no frame is available
                    if window['frame'] is None:
                        continue
                        
                    # Show current frame with timeout
                    try:
                        if self.windows_created.get(mode, False):
                            cv2.imshow(window['name'], window['frame'])
                            cv2.waitKey(1)  # Non-blocking call
                    except Exception as e:
                        print(f"Error displaying {mode} frame: {e}")
                
                # Process debug frame queue
                try:
                    debug_data = self.debug_frame_queue.get(timeout=0.1)
                    if debug_data is not None:
                        mode = debug_data.get('mode')
                        frame = debug_data.get('frame')
                        
                        if mode in self.debug_windows and frame is not None:
                            # Update stored frame
                            self.debug_windows[mode]['frame'] = frame
                except queue.Empty:
                    pass  # No new frames, that's fine
                except Exception as e:
                    print(f"Error processing debug queue: {e}")
                    
            except Exception as e:
                print(f"Error in debug window manager: {e}")
                time.sleep(0.5)
    
    def queue_debug_frame(self, mode, frame):
        """
        Queue a debug frame for display with enhanced error handling.
        
        Args:
            mode (str): Debug mode ('interaction' or 'proximity')
            frame (np.ndarray): Frame to display
        """
        if not self.debug_mode or frame is None:
            print(f"Debug mode disabled or frame is None. debug_mode={self.debug_mode}")
            return
            
        try:
            # Check for valid mode
            if mode not in self.debug_windows:
                print(f"Invalid debug mode: {mode}")
                return
                
            # Check if window is enabled
            if not self.debug_windows[mode]['enabled']:
                print(f"Debug window {mode} is not enabled")
                return
                
            # Check frame dimensions
            if frame.shape[0] <= 0 or frame.shape[1] <= 0 or len(frame.shape) != 3:
                print(f"Invalid frame dimensions: {frame.shape}")
                return
                
            # Create debug data package
            debug_data = {
                'mode': mode,
                'frame': frame.copy()  # Ensure we have a clean copy
            }
            
            # Ensure the window exists - with more detailed error handling
            if not self.windows_created[mode]:
                print(f"Creating {mode} debug window...")
                try:
                    win_name = self.debug_windows[mode]['name']
                    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
                    cv2.moveWindow(win_name, *self.window_positions[mode])
                    cv2.resizeWindow(win_name, 640, 480)
                    self.windows_created[mode] = True
                    
                    # Show a temporary frame
                    temp_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(temp_frame, f"Loading {mode} debug view...", 
                                (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    self.debug_windows[mode]['frame'] = temp_frame
                    
                    # Try to show initial frame
                    try:
                        cv2.imshow(win_name, temp_frame)
                        cv2.waitKey(1)  # This is critical to actually display the window
                        print(f"Successfully displayed initial {mode} debug frame")
                    except Exception as show_error:
                        print(f"WARNING: Could not display initial frame: {show_error}")
                    
                    print(f"Created debug window: {win_name}")
                except Exception as window_error:
                    print(f"ERROR creating {mode} window: {window_error}")
                    print("This could indicate an OpenCV or display configuration issue")
                    import traceback
                    traceback.print_exc()
                    return
            
            # Put frame in queue with detailed error handling
            try:
                if self.debug_frame_queue.full():
                    print(f"Debug frame queue is full, removing oldest frame")
                    try:
                        old_frame = self.debug_frame_queue.get_nowait()
                        print(f"Removed old {old_frame.get('mode', 'unknown')} frame")
                    except Exception as remove_error:
                        print(f"Error removing old frame: {remove_error}")
                        
                # Now try to put the new frame in the queue
                self.debug_frame_queue.put_nowait(debug_data)
                print(f"Successfully queued {mode} debug frame")
                
            except Exception as queue_error:
                print(f"ERROR adding frame to queue: {queue_error}")
                import traceback
                traceback.print_exc()
                
        except Exception as e:
            print(f"CRITICAL ERROR in queue_debug_frame: {e}")
            import traceback
            traceback.print_exc()
                
    def parse_debug_arguments(self, args):
        """
        Parse debug arguments from command line arguments.
        
        Args:
            args: Parsed command-line arguments
        """
        # Check for main debug flag
        if hasattr(args, 'debug') and args.debug:
            self.debug_mode = True
            
        # Check for specific debug modes
        if hasattr(args, 'debug_interaction') and args.debug_interaction:
            self.debug_windows['interaction']['enabled'] = True
            
        if hasattr(args, 'debug_proximity') and args.debug_proximity:
            self.debug_windows['proximity']['enabled'] = True
        
        # If any debug mode is enabled, start debug thread if not already running
        if self.debug_mode and not hasattr(self, 'debug_thread'):
            self.debug_thread = threading.Thread(target=self._debug_window_manager, daemon=True)
            self.debug_thread.start()
            
        # Update status indicators
        self.status_indicators['debug_enabled'] = self.debug_mode
        self.status_indicators['interaction_enabled'] = self.debug_windows['interaction']['enabled']
        self.status_indicators['proximity_enabled'] = self.debug_windows['proximity']['enabled']
        
        # Force create windows for active modes
        for mode, window in self.debug_windows.items():
            if window['enabled'] and not self.windows_created[mode]:
                win_name = window['name']
                cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
                cv2.moveWindow(win_name, *self.window_positions[mode])
                cv2.resizeWindow(win_name, 640, 480)
                self.windows_created[mode] = True
                
                # Show initial blank frame
                blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(blank_frame, f"Initializing {mode} debug view...", 
                            (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                window['frame'] = blank_frame
                cv2.imshow(win_name, blank_frame)
                cv2.waitKey(1)
                
                print(f"Created initial {mode} debug window from args")
    
    def toggle_stats_display(self):
        """
        Toggle display of performance statistics
        """
        if not hasattr(self, 'show_stats'):
            self.show_stats = False
            
        self.show_stats = not self.show_stats
        print(f"Performance stats display: {'Enabled' if self.show_stats else 'Disabled'}")
        
        # Create an initial stats display when enabled
        if self.show_stats:
            # Create a basic stats frame for each active window
            for mode, window in self.debug_windows.items():
                if window['enabled'] and window['frame'] is not None:
                    # Create a copy of the current frame
                    stats_frame = window['frame'].copy()
                    
                    # Add stats overlay
                    height, width = stats_frame.shape[:2]
                    
                    # Stats panel background
                    stat_height = 120
                    cv2.rectangle(stats_frame, 
                                (10, height - stat_height - 10), 
                                (250, height - 10), 
                                (0, 0, 0), -1)
                    
                    # Add transparency
                    alpha = 0.7
                    overlay = stats_frame.copy()
                    cv2.rectangle(overlay, 
                                (10, height - stat_height - 10), 
                                (250, height - 10), 
                                (0, 0, 0), -1)
                    cv2.addWeighted(overlay, alpha, stats_frame, 1 - alpha, 0, stats_frame)
                    
                    # Add sample stats
                    text_y = height - stat_height
                    text_y += 20
                    cv2.putText(stats_frame, "Stats Display Enabled", (20, text_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                    text_y += 20
                    cv2.putText(stats_frame, "FPS: Computing...", (20, text_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                    text_y += 20
                    cv2.putText(stats_frame, "Objects: Computing...", (20, text_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                    # Queue the stats frame
                    self.queue_debug_frame(mode, stats_frame)

    def update_performance_stats(self, frame, stats):
        """
        Update performance statistics overlay on debug frames
        
        Args:
            frame: Frame to add stats overlay to
            stats: Dictionary of stats to display
        
        Returns:
            Modified frame with stats overlay
        """
        if not hasattr(self, 'show_stats') or not self.show_stats or frame is None:
            return frame
            
        try:
            # Create a copy
            stats_frame = frame.copy()
            height, width = stats_frame.shape[:2]
            
            # Stats panel background
            stat_height = 20 + (len(stats) * 20) if stats else 80
            cv2.rectangle(stats_frame, 
                        (10, height - stat_height - 10), 
                        (250, height - 10), 
                        (0, 0, 0), -1)
            
            # Add transparency
            alpha = 0.7
            overlay = stats_frame.copy()
            cv2.rectangle(overlay, 
                        (10, height - stat_height - 10), 
                        (250, height - 10), 
                        (0, 0, 0), -1)
            cv2.addWeighted(overlay, alpha, stats_frame, 1 - alpha, 0, stats_frame)
            
            # Add stats
            text_y = height - stat_height
            if stats:
                for stat_name, stat_value in stats.items():
                    text_y += 20
                    text = f"{stat_name}: {stat_value}"
                    cv2.putText(stats_frame, text, (20, text_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            else:
                text_y += 20
                cv2.putText(stats_frame, "No stats available", (20, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
            return stats_frame
        except Exception as e:
            print(f"Error adding stats overlay: {e}")
            return frame
    
    def add_status_overlay(self, frame):
        """
        Add debug status indicators to the main frame.
        
        Args:
            frame (np.ndarray): Frame to add status overlay to
            
        Returns:
            np.ndarray: Frame with status overlay
        """
        if frame is None:
            return frame
            
        try:
            # Create copy of frame
            overlay_frame = frame.copy()
            
            # Add small status indicator in corner
            status_width = 150
            status_height = 80
            margin = 10
            
            # Create semi-transparent rectangle in bottom right
            x1 = overlay_frame.shape[1] - status_width - margin
            y1 = overlay_frame.shape[0] - status_height - margin
            x2 = overlay_frame.shape[1] - margin
            y2 = overlay_frame.shape[0] - margin
            
            # Create overlay
            overlay = overlay_frame.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
            
            # Add transparency
            alpha = 0.7
            cv2.addWeighted(overlay, alpha, overlay_frame, 1 - alpha, 0, overlay_frame)
            
            # Add text
            text_x = x1 + 10
            text_y = y1 + 20
            
            # Debug status
            status_color = (0, 255, 0) if self.status_indicators['debug_enabled'] else (0, 0, 255)
            cv2.putText(overlay_frame, "Debug Mode", (text_x, text_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
            
            # Interaction status
            text_y += 20
            status_color = (0, 255, 0) if self.status_indicators['interaction_enabled'] else (100, 100, 100)
            cv2.putText(overlay_frame, "Interaction (i)", (text_x, text_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
            
            # Proximity status
            text_y += 20
            status_color = (0, 255, 0) if self.status_indicators['proximity_enabled'] else (100, 100, 100)
            cv2.putText(overlay_frame, "Proximity (h)", (text_x, text_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
            
            return overlay_frame
        except Exception as e:
            print(f"Error adding status overlay: {e}")
            return frame
        
    def toggle_live_stats(self, enabled=None):
        """
        Toggle display of live performance statistics
        
        Args:
            enabled: Explicitly set state if provided, otherwise toggle
        """
        # Initialize the show_stats attribute if it doesn't exist
        if not hasattr(self, 'show_stats'):
            self.show_stats = False
            
        if enabled is not None:
            self.show_stats = enabled
        else:
            self.show_stats = not self.show_stats
        
        print(f"Live stats display: {'Enabled' if self.show_stats else 'Disabled'}")
        
    def add_performance_overlay(self, frame, stats=None):
        """
        Add performance statistics to the frame
        
        Args:
            frame: The frame to add overlay to
            stats: Dictionary of statistics to display
            
        Returns:
            frame: The frame with overlay added
        """
        if not hasattr(self, 'show_stats'):
            self.show_stats = False
            
        if not self.show_stats or frame is None:
            return frame
            
        # Create overlay
        overlay_frame = frame.copy()
        height, width = overlay_frame.shape[:2]
        
        # Stats panel background
        stat_height = 120  # Adjust based on number of stats
        cv2.rectangle(overlay_frame, 
                    (10, height - stat_height - 10), 
                    (250, height - 10), 
                    (0, 0, 0), -1)
        
        # Add transparency
        alpha = 0.7
        try:
            frame_region = frame[height - stat_height - 10:height - 10, 10:250]
            overlay_region = overlay_frame[height - stat_height - 10:height - 10, 10:250]
            cv2.addWeighted(overlay_region, alpha, frame_region, 1 - alpha, 0, frame_region)
        except Exception as e:
            # In case of regions mismatch or other issues, continue without transparency
            print(f"Warning in performance overlay: {e}")
        
        # Add stats
        text_y = height - stat_height
        if stats:
            for stat_name, stat_value in stats.items():
                text_y += 20
                text = f"{stat_name}: {stat_value}"
                cv2.putText(overlay_frame, text, (20, text_y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        else:
            # Show default message if no stats provided
            text_y += 20
            cv2.putText(overlay_frame, "No performance stats available", (20, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return overlay_frame
    
    def visualize_hand_proximity(self, frame, hand_obj, objects, nearby_objects=None, selected_obj=None):
        """
        Create and queue a visualization for hand proximity detection.
        
        Args:
            frame: The frame to visualize on
            hand_obj: The hand or controller object
            objects: All detected objects
            nearby_objects: List of objects near the hand (optional)
            selected_obj: The selected object if any (optional)
        """
        if not self.debug_mode or not self.debug_windows['proximity']['enabled']:
            return
            
        try:
            # Define local helper function for midpoint calculation (to avoid circular imports)
            def calculate_midpoint(bbox):
                """Calculate midpoint of a bounding box [x1,y1,x2,y2]"""
                if bbox is None or len(bbox) != 4:
                    return None, None
                return int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2)
            
            # Create a copy of the frame for visualization
            debug_frame = frame.copy() if frame is not None else None
            if debug_frame is None:
                return
                
            # Create debug visualizer
            debug_vis = DebugVisualizer(debug_frame)
            
            # Add title
            debug_vis.add_title("Hand Proximity Detection", "Proximity Debug")
            
            # Draw hand/controller
            if hand_obj and 'bbox' in hand_obj:
                h_bbox = hand_obj.get('bbox')
                h_label = hand_obj.get('label', 'hand')
                h_depth = hand_obj.get('depth')
                
                # Calculate midpoint using local function
                h_mx, h_my = calculate_midpoint(h_bbox)
                
                # Draw hand with green box
                debug_vis.draw_box(h_bbox, (0, 255, 0), 2, f"{h_label} d:{h_depth}")
                
                # Draw proximity radius if midpoint is valid
                if h_mx is not None and h_my is not None:
                    debug_vis.add_circle(h_mx, h_my, 100, (0, 255, 255), 1)  # 100px proximity circle
            else:
                debug_vis.add_text("No hand/controller detected", 10, 70, (0, 0, 255))
            
            # Draw all objects with appropriate colors
            if objects:
                count = 0
                for obj in objects:
                    if obj is None or 'bbox' not in obj or obj == hand_obj:
                        continue
                        
                    bbox = obj.get('bbox')
                    label = obj.get('label', 'unknown')
                    depth = obj.get('depth')
                    
                    # Default color (white)
                    color = (255, 255, 255)
                    
                    # Check if this object is in nearby objects
                    is_nearby = False
                    if nearby_objects:
                        for i, nearby_tuple in enumerate(nearby_objects):
                            # Handle different tuple formats
                            if len(nearby_tuple) >= 1:
                                nearby_obj = nearby_tuple[0]
                                if nearby_obj == obj:
                                    is_nearby = True
                                    # Use yellow for nearby objects
                                    color = (0, 255, 255)
                                    # Add ranking number for top objects
                                    debug_vis.add_text(f"#{i+1}", bbox[0]-20, bbox[1]-5, (255, 255, 0))
                                    break
                    
                    # Use green for selected object
                    if selected_obj and obj == selected_obj:
                        color = (0, 255, 0)
                        debug_vis.highlight_object(bbox, True)
                        
                        # If selected and we have hand position, draw a connection line
                        if hand_obj and 'bbox' in hand_obj:
                            h_bbox = hand_obj.get('bbox')
                            h_mx, h_my = calculate_midpoint(h_bbox)
                            obj_mx, obj_my = calculate_midpoint(bbox)
                            
                            if h_mx is not None and obj_mx is not None:
                                debug_vis.draw_line(h_mx, h_my, obj_mx, obj_my, (0, 255, 0), 1, "Selected")
                    
                    # Draw the object
                    debug_vis.draw_box(bbox, color, 1, f"{label} d:{depth}")
                    count += 1
                
                debug_vis.add_text(f"Total objects: {count}", 10, 90)
            else:
                debug_vis.add_text("No objects detected", 10, 90, (0, 0, 255))
            
            # Add explanatory text
            debug_vis.add_text("Proximity detection: Green = selected, Yellow = nearby", 10, 50)
            
            # Queue the frame for display
            self.queue_debug_frame('proximity', debug_vis.get_frame())
            print(f"Proximity visualization queued with {len(nearby_objects) if nearby_objects else 0} nearby objects")
        except Exception as e:
            print(f"Error creating proximity visualization: {e}")
            import traceback
            traceback.print_exc()

    def visualize_line_interaction(self, frame, line_start, line_end, extended_point=None, 
                                objects=None, hand_bbox=None, hand_depth=None, hit_object=None):
        """
        Create and queue a visualization for line/ray-based interaction detection.
        
        Args:
            frame: The frame to visualize on
            line_start: Starting point (x,y) of the line/ray
            line_end: Ending point (x,y) of the detected line
            extended_point: Extended ray endpoint for collision detection (optional)
            objects: All detected objects
            hand_bbox: Bounding box of the hand/controller
            hand_depth: Depth value of the hand/controller
            hit_object: The object hit by the ray (if any)
        """
        if not self.debug_mode or not self.debug_windows['interaction']['enabled']:
            return
            
        try:
            # Create a copy of the frame for visualization
            debug_frame = frame.copy() if frame is not None else None
            if debug_frame is None:
                return
                
            # Create debug visualizer
            debug_vis = DebugVisualizer(debug_frame)
            
            # Add title
            debug_vis.add_title("Ray Interaction Detection", "Interaction Debug")
            
            # Define helper function for midpoint calculation (to avoid circular imports)
            def calculate_midpoint(bbox):
                """Calculate midpoint of a bounding box [x1,y1,x2,y2]"""
                if bbox is None or len(bbox) != 4:
                    return None, None
                return int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2)
            
            # Draw the detected line
            if line_start and line_end:
                debug_vis.draw_line(
                    line_start[0], line_start[1], 
                    line_end[0], line_end[1], 
                    (0, 255, 0), 2, "Detected Line"
                )
            
            # Draw extended ray if provided
            if line_end and extended_point:
                debug_vis.draw_line(
                    line_end[0], line_end[1], 
                    extended_point[0], extended_point[1], 
                    (255, 0, 0), 2, "Extended Ray"
                )
                debug_vis.add_circle(
                    extended_point[0], extended_point[1], 
                    5, (255, 0, 0), -1
                )
            
            # Draw hand/controller if provided
            if hand_bbox:
                debug_vis.draw_box(
                    hand_bbox, 
                    (0, 255, 0), 2, 
                    f"Hand/Controller Depth:{hand_depth}"
                )
            
            # Draw all objects with appropriate colors
            if objects:
                for obj in objects:
                    if obj is None or 'bbox' not in obj:
                        continue
                    
                    bbox = obj.get('bbox')
                    label = obj.get('label', 'unknown')
                    depth = obj.get('depth')
                    
                    # Default color (white)
                    color = (255, 255, 255)
                    thickness = 1
                    
                    # Use green for the hit object
                    if hit_object and obj == hit_object:
                        color = (0, 255, 0)
                        thickness = 2
                        debug_vis.highlight_object(bbox, True)
                        
                        # Draw a line connecting the ray to the hit object
                        if line_end:
                            obj_midx, obj_midy = calculate_midpoint(bbox)
                            if obj_midx is not None:
                                debug_vis.draw_line(
                                    line_end[0], line_end[1], 
                                    obj_midx, obj_midy, 
                                    (255, 0, 255), 2, "Hit"
                                )
                    
                    # Draw the object
                    debug_vis.draw_box(bbox, color, thickness, f"{label} d:{depth}")
            
            # Add explanatory text
            debug_vis.add_text("Ray interaction: Green = selected object", 10, 50)
            
            # Queue the frame for display
            self.queue_debug_frame('interaction', debug_vis.get_frame())
        except Exception as e:
            print(f"Error creating line interaction visualization: {e}")
            import traceback
            traceback.print_exc()

# Command-line argument parsing for debug mode
def add_debug_arguments(parser):
    """
    Add debug-related command-line arguments.
    
    Args:
        parser (argparse.ArgumentParser): Argument parser to modify
    """
    parser.add_argument('--debug', action='store_true', 
                        help='Enable debug visualization mode')
    parser.add_argument('--debug-interaction', action='store_true', 
                        help='Enable interaction debug visualization')
    parser.add_argument('--debug-proximity', action='store_true', 
                        help='Enable hand proximity debug visualization')
