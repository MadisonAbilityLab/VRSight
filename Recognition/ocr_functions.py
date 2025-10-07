import time
import cv2
import io
import numpy as np
import json
from api_key_rotator import APIKeyRotator
import gpt_functions as gpt
import threading
import os
from dotenv import load_dotenv
import concurrent.futures
import hashlib

load_dotenv()

# Additional imports for Azure OCR
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from msrest.authentication import CognitiveServicesCredentials

from difflib import SequenceMatcher

ocr_key_rotator = APIKeyRotator(
    service_name="Azure OCR",
    free_key_env_var="DW_FREE_AZURE_OCR_KEY",
    paid_key_env_var="AZURE_COMPUTERVISION_KEY",
    free_endpoint_env_var="DW_FREE_AZURE_OCR_ENDPOINT",
    paid_endpoint_env_var="AZURE_COMPUTERVISION_ENDPOINT"
)

def get_ocr_client():
    """Get a Computer Vision client with the current API key and endpoint"""
    current_key = ocr_key_rotator.get_key()
    current_endpoint = ocr_key_rotator.get_endpoint()
    
    return ComputerVisionClient(current_endpoint, CognitiveServicesCredentials(current_key))

# Initialize the OCR label array
ocr_label_list = [
    "button",
    "chat box",
    "chat bubble",
    "hud",
    "interactable",
    "menu",
    "nametag",
    "progress bar",
    "portal",
    "sign-text",
    "ui-text",
]

automatic_ocr_label_list = [
    "chat box", 
    "chat bubble", 
    "hud", 
    "nametag", 
    "progress bar"
]

# OCR result cache to prevent duplicate requests
class OCRCache:
    def __init__(self, max_size=100, ttl=30):
        self.cache = {}
        self.access_times = {}
        self.max_size = max_size
        self.ttl = ttl  # Time-to-live in seconds
        self.lock = threading.Lock()
        
    def get(self, key):
        """Get a cached OCR result if it exists and is not expired"""
        with self.lock:
            if key not in self.cache:
                return None
                
            # Check if entry has expired
            access_time = self.access_times.get(key, 0)
            if time.time() - access_time > self.ttl:
                # Remove expired entry
                del self.cache[key]
                del self.access_times[key]
                return None
                
            # Update access time
            self.access_times[key] = time.time()
            return self.cache[key]
    
    def set(self, key, value):
        """Store a result in the cache"""
        with self.lock:
            # Check if we need to evict an old entry
            if len(self.cache) >= self.max_size:
                # Find oldest entry
                oldest_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
                del self.cache[oldest_key]
                del self.access_times[oldest_key]
                
            # Add new entry
            self.cache[key] = value
            self.access_times[key] = time.time()
            
    def clear_expired(self):
        """Clear all expired entries from the cache"""
        with self.lock:
            current_time = time.time()
            expired_keys = [
                k for k, access_time in self.access_times.items() 
                if current_time - access_time > self.ttl
            ]
            
            for key in expired_keys:
                del self.cache[key]
                del self.access_times[key]
                
            return len(expired_keys)

# Initialize OCR cache
ocr_cache = OCRCache()

def prepare_image_for_ocr(frame, label=None):
    """
    Unified function to prepare images for OCR with improved preprocessing
    
    Args:
        frame: The image frame to process
        label: Optional label for context (not used for special handling)
        
    Returns:
        bytes: Processed image as JPEG byte buffer, or None if processing fails
    """
    try:
        if frame is None:
            return None
            
        # Ensure color image for consistent processing
        if len(frame.shape) == 2:  # Grayscale
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            
        # Get image dimensions
        h, w = frame.shape[:2]
        
        # Resize image if needed to meet Azure OCR requirements
        # Azure requires images between 50px and 4200px
        modified = False
        
        # Upscale small images
        if w < 50 or h < 50:
            scale = max(50.0/w, 50.0/h)
            new_w = max(50, int(w * scale))
            new_h = max(50, int(h * scale))
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            print(f"Upscaled from {w}x{h} to {new_w}x{new_h}")
            modified = True
            
        # Downscale large images
        elif w > 4000 or h > 4000:
            scale = min(4000.0/w, 4000.0/h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
            print(f"Downscaled from {w}x{h} to {new_w}x{new_h}")
            modified = True
            
        # Apply image enhancement for better OCR
        # Enhance contrast and sharpness
        enhanced = frame.copy()
        
        # Apply CLAHE for better contrast
        if len(frame.shape) == 3:  # Color image
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            enhanced_lab = cv2.merge((cl, a, b))
            enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
            
        # Encode as high-quality JPEG for Azure
        _, buffer = cv2.imencode('.jpg', enhanced, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        return buffer
        
    except Exception as e:
        print(f"Error preparing image for OCR: {e}")
        # Try simple encoding as fallback
        try:
            _, buffer = cv2.imencode('.jpg', frame)
            return buffer
        except:
            return None

#OCR method for cropping and multiple detections: 
# Feed in the frame, detection (you can feed in just one detection object)
#pass in auto as 0 integer (1 currently used for special object list)
def crop_and_ocr(raw_frame, detections, auto):
    """
    Process frame crops for OCR with improved error handling and caching
    """
    if raw_frame is None:
        print("ERROR: Null frame provided to crop_and_ocr")
        return None
        
    # Check if a single detection is passed (dictionary) or a list of detections
    if isinstance(detections, dict):  # Single detection
        detections = [detections]  # Convert to a list with one element
    
    results = []
    for detection in detections:
        try:
            if detection is None:
                continue
                
            bbox = detection.get("bbox")
            label = detection.get("label")
            
            if bbox is None or label is None:
                continue
                
            print(f"OCR: '{label}'")

            if label in ocr_label_list:
                if auto:  # perform if 1 for automatic query
                    if label in automatic_ocr_label_list:
                        try:
                            x1, y1, x2, y2 = bbox
                            
                            # Ensure coordinates are within frame boundaries
                            max_h, max_w = raw_frame.shape[:2]
                            x1 = max(0, min(x1, max_w-1))
                            y1 = max(0, min(y1, max_h-1))
                            x2 = max(0, min(x2, max_w-1))
                            y2 = max(0, min(y2, max_h-1))
                            
                            # Skip invalid boxes
                            if x1 >= x2 or y1 >= y2:
                                continue
                                
                            cropped_frame = raw_frame[y1:y2, x1:x2]
                            text = perform_ocr_on_frame(cropped_frame, label)
                            results.append({"label": label, "text": text})
                        except Exception as e:
                            print(f"Error cropping frame for OCR: {e}")
                    else:
                        print(f"'{label}' NOT in automatic_ocr_label_list")
                else:  # else perform normal OCR query when auto = 0
                    try:
                        x1, y1, x2, y2 = bbox
                        
                        # Ensure coordinates are within frame boundaries
                        max_h, max_w = raw_frame.shape[:2]
                        x1 = max(0, min(x1, max_w-1))
                        y1 = max(0, min(y1, max_h-1))
                        x2 = max(0, min(x2, max_w-1))
                        y2 = max(0, min(y2, max_h-1))
                        
                        # Skip invalid boxes
                        if x1 >= x2 or y1 >= y2:
                            continue
                            
                        cropped_frame = raw_frame[y1:y2, x1:x2]
                        text = perform_ocr_on_frame(cropped_frame, label)
                        results.append({"label": label, "text": text})
                    except Exception as e:
                        print(f"Error cropping frame for OCR: {e}")
            else:
                print(f"'{label}' NOT in ocr_label_list.")
        except Exception as e:
            print(f"Error processing detection for OCR: {e}")
            
    return results
            
def preprocess(text):
    """Preprocess text for comparison"""
    if text is None:
        return ""
    return "".join(e for e in text if e.isalnum()).lower()

# Function to calculate similarity between two texts
def is_similar(text1, text2, threshold=0.50):
    """
    Calculate similarity between two texts using SequenceMatcher
    
    Args:
        text1 (str): First text string
        text2 (str): Second text string
        threshold (float): Similarity threshold (0.0 to 1.0)
        
    Returns:
        bool: True if texts are similar, False otherwise
    """
    if text1 is None or text2 is None:
        return False
        
    try:
        similarity = SequenceMatcher(None, preprocess(text1), preprocess(text2)).ratio()
        return similarity >= threshold
    except Exception as e:
        print(f"Error calculating text similarity: {e}")
        return False

def azure_ocr_attempt_with_timeout(image_stream, timeout=5):
    """
    Attempt Azure OCR with improved error handling and timeout
    
    Args:
        image_stream: BytesIO stream containing the image
        timeout: Maximum time to wait for OCR to complete
        
    Returns:
        str: OCR text result, or None if OCR fails
    """
    def ocr_worker():
        try:
            # Reset stream position and verify it has data
            image_stream.seek(0)
            image_data = image_stream.read()
            if len(image_data) < 50:  # Very small image data
                return None, "Image data too small for OCR"
            
            # Reset stream position
            image_stream.seek(0)
            
            # Get fresh client with current API key
            client = get_ocr_client()
            
            # Make the API call
            ocr_result = client.read_in_stream(image_stream, raw=True)
            
            # Extract operation ID
            operation_location = ocr_result.headers.get("Operation-Location")
            if not operation_location:
                return None, "No operation location returned"
                
            operation_id = operation_location.split("/")[-1]
            
            # Poll for results with timeout
            max_retries = 5
            for retry in range(max_retries):
                result = client.get_read_result(operation_id)
                if result.status not in ["notStarted", "running"]:
                    break
                time.sleep(0.5)
            
            # Check for success and extract text
            if result.status == OperationStatusCodes.succeeded:
                text_result = " ".join(
                    line.text
                    for read_result in result.analyze_result.read_results
                    for line in read_result.lines
                ).strip()
                
                if text_result:  # Only return if text was found
                    return text_result, None
                else:
                    return None, "No text found in image"
            else:
                return None, f"OCR failed with status: {result.status}"
                
        except Exception as e:
            error_code = None
            if "429" in str(e):
                error_code = 429
            
            # Record error and try to switch keys
            switched = ocr_key_rotator.record_error(error_code)
            
            error_message = f"Azure OCR error: {e}"
            return None, error_message
    
    try:
        # Use concurrent.futures for clean timeout handling
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(ocr_worker)
            try:
                result, error = future.result(timeout=timeout)
                if result:
                    return result
                else:
                    print(f"OCR failed: {error}")
                    return None
            except concurrent.futures.TimeoutError:
                print(f"OCR timed out after {timeout} seconds")
                return None
    except Exception as e:
        print(f"Unexpected error in OCR: {e}")
        return None

def process_ui_with_gpt(frame, label=None):
    """
    Process UI elements using GPT when OCR fails
    
    Args:
        frame: Image frame to process
        label: Optional label of the UI element
        
    Returns:
        str: Extracted text or description
    """
    try:
        # Skip if we're being rate limited
        if not gpt.gpt_rate_limiter.wait_if_needed():
            return f"{label if label else 'UI element'} detected" 
            
        base64_frame = gpt.encode_frame_to_base64(frame)
        if base64_frame is None:
            return f"{label if label else 'UI element'} detected"
            
        # Construct prompt based on whether we have a label
        if label:
            prompt = (
                "You are performing OCR on a UI element from a VR application. "
                "Extract ONLY the visible text, numbers, or symbols exactly as they appear. "
                "Return ONLY the text with no additional explanation, description, or context. "
                f"This element is labeled as '{label}'. "
                "If no clear text is visible, respond with exactly: "
                f"'{label} detected but I can't tell what it says'"
            )
        else:
            prompt = (
                "You are performing OCR on a UI element from a VR application. "
                "Extract ONLY the visible text, numbers, or symbols exactly as they appear. "
                "Return ONLY the text with no additional explanation, description, or context. "
                "If no clear text is visible, respond with exactly: "
                "'UI element detected but I can't tell what it says'"
            )
        
        result = gpt.query_gpt(prompt, base64_frame)
        if not result:
            return f"{label if label else 'UI element'} detected"
            
        # Clean up the result
        result = clean_gpt_result(result)
        
        print(f"GPT fallback result: {result}")
        return result
        
    except Exception as e:
        print(f"GPT processing failed: {e}")
        return f"{label if label else 'UI element'} detected"

def clean_gpt_result(result):
    """
    Cleans GPT results to extract just the relevant text
    
    Args:
        result: The raw text from GPT
        
    Returns:
        str: Cleaned text with explanations removed
    """
    if result is None:
        return ""
        
    try:
        # Remove explanatory phrases GPT tends to add
        common_prefixes = [
            "The text shows", "The image shows", "The UI displays", 
            "The display shows", "This shows", "It shows", "I can see",
            "The interface shows", "The screen displays", "I see"
        ]
        
        cleaned = result
        for prefix in common_prefixes:
            if cleaned.lower().startswith(prefix.lower()):
                cleaned = cleaned[len(prefix):].strip()
                # Remove leading punctuation if present
                cleaned = cleaned.lstrip(":")
                cleaned = cleaned.lstrip(" ")
        
        # Remove quotes if GPT added them
        cleaned = cleaned.strip('"\'')
        
        return cleaned
    except Exception as e:
        print(f"Error cleaning GPT result: {e}")
        return result

def perform_ocr_on_frame(frame, label=None):
    """
    Enhanced OCR function with robust fallback chain
    
    Args:
        frame: The frame to process
        label: Optional label of the UI element (for better fallback responses)
    
    Returns:
        str: OCR result text
    """
    # Basic validation
    if frame is None or not isinstance(frame, np.ndarray):
        print("Error: Invalid frame provided to OCR.")
        return f"{label if label else 'UI element'} detected but image invalid"
        
    # Check dimensions
    if len(frame.shape) < 2:
        print("Error: Invalid frame dimensions for OCR.")
        return f"{label if label else 'UI element'} detected but image invalid"
        
    height, width = frame.shape[:2]
    
    # Check cache first with improved hashing
    cache_key = None
    try:
        # Create a more robust hash from frame content
        # Use a combination of frame shape and content hash of downsampled frame
        small_frame = cv2.resize(frame, (16, 16))  # Smaller for faster hashing
        frame_bytes = small_frame.tobytes()
        
        # Use hashlib for more reliable hashing
        hasher = hashlib.md5()
        hasher.update(frame_bytes)
        
        # Include shape in the key for additional uniqueness
        shape_str = f"{width}x{height}"
        combined_key = f"{shape_str}_{hasher.hexdigest()}"
        
        cache_key = hash(combined_key)
        
        cached_result = ocr_cache.get(cache_key)
        if cached_result is not None:
            print("Using cached OCR result")
            return cached_result
    except Exception as e:
        print(f"Error checking OCR cache: {e}")
        cache_key = None
    
    # Two-stage OCR approach: try Azure first, then GPT
    result = None
    
    # 1. Try Azure OCR first with image preparation
    try:
        # Prepare image for Azure OCR
        prepared_buffer = prepare_image_for_ocr(frame, label)
        
        # Explicit check if buffer is not None
        if prepared_buffer is not None and len(prepared_buffer) > 0:
            # Create stream for Azure OCR
            image_stream = io.BytesIO(prepared_buffer)
            
            # Run Azure OCR with timeout
            azure_result = azure_ocr_attempt_with_timeout(image_stream)
            
            if azure_result:
                result = azure_result
    except Exception as e:
        print(f"Error in Azure OCR processing: {e}")
    
    # 2. If Azure failed, try GPT
    if not result:
        try:
            result = process_ui_with_gpt(frame, label)
        except Exception as e:
            print(f"Error in GPT fallback: {e}")
            # Use simple detection message as last resort
            result = f"{label if label else 'UI element'} detected but I can't tell what it says"
    
    # Cache the result if we have a valid key
    if cache_key is not None and result:
        ocr_cache.set(cache_key, result)
    
    return result