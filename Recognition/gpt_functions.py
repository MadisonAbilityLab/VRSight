import time
from openai import AzureOpenAI
import os
import cv2
import base64
from dotenv import load_dotenv
from threading import Lock
from label_handler_list import get_semantic_label
from settings import CAM_WIDTH, CAM_HEIGHT, GPT_MIN_REQUEST_INTERVAL

# global variable for rate limiting
last_request_time = 0
MIN_REQUEST_INTERVAL = GPT_MIN_REQUEST_INTERVAL
current_emotion = None
emotion_lock = Lock()

load_dotenv()
# Initialize OpenAI GPT API
client = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2024-02-01"
)

# Enhanced rate limiter with adaptive backoff
class AdaptiveRateLimiter:
    def __init__(self):
        self.last_request_time = 0
        self.base_interval = 3.0  # Reduced from 5.0 to 3.0 seconds
        self.current_interval = 3.0
        self.max_interval = 30.0  # Reduced from 60.0 to 30.0 seconds (more aggressive but faster recovery)
        self.success_count = 0
        self.failure_count = 0
        self.lock = Lock()
        
    def wait_if_needed(self, label="None"):
        """Wait if needed based on rate limiting"""
        if label == "menu":
            return True

        with self.lock:
            current_time = time.time()
            time_since_last = current_time - self.last_request_time
            
            # If we need to wait
            if time_since_last < self.current_interval:
                wait_time = self.current_interval - time_since_last
                # print(f"Rate limiting: waiting {wait_time:.2f} seconds")
                # time.sleep(wait_time)
            
            self.last_request_time = time.time()
            return True
            
    def record_success(self):
        """Record a successful API call with faster recovery"""
        with self.lock:
            self.success_count += 1
            self.failure_count = 0
            
            # Gradually decrease interval after consecutive successes
            # More aggressive decrease to recover faster
            if self.success_count > 3 and self.current_interval > self.base_interval:  # Reduced from 5 to 3 successes
                self.current_interval = max(self.base_interval, 
                                          self.current_interval * 0.8)  # More aggressive 0.8 vs 0.9
                # print(f"Rate limit success: decreasing interval to {self.current_interval:.2f} seconds")
            
    def record_failure(self, error_code=None):
        """Record a failed API call with adaptive backoff"""
        with self.lock:
            self.failure_count += 1
            self.success_count = 0
            
            # Exponential backoff based on consecutive failures
            backoff_factor = min(2.0, 1.0 + (self.failure_count * 0.2))
            
            # If we got a 429, use a more aggressive backoff
            if error_code == 429:
                backoff_factor = min(3.0, 1.5 + (self.failure_count * 0.5))
            
            # Calculate new interval with bounds check
            new_interval = self.current_interval * backoff_factor
            self.current_interval = min(self.max_interval, new_interval)
            
            # print(f"Rate limit backoff: new interval is {self.current_interval:.2f} seconds")
            
            # Shorter immediate wait after a failure
            time.sleep(self.current_interval * 0.5)  # Only wait half the backoff time

# Create global instance of adaptive rate limiter
gpt_rate_limiter = AdaptiveRateLimiter()

# function to encode frame to Base64, getting a string with the 
# Base64 encoded binary data of a frame. max size of 320px or 1/2 original input size.
def encode_frame_to_base64(frame, max_dimension=320):
    """
    Encodes a frame to Base64 with proper error handling and size constraints.
    """
    try:
        if frame is None or frame.size == 0:
            # print("ERROR: Empty frame provided to Base64 encoder")
            return None
            
        # Calculate aspect ratio and new dimensions
        height, width = frame.shape[:2]
        aspect_ratio = width / height
        
        # Only resize if the image is larger than max_dimension
        if width > max_dimension or height > max_dimension:
            try:
                if width > height:
                    new_width = max_dimension
                    new_height = int(max_dimension / aspect_ratio)
                else:
                    new_height = max_dimension
                    new_width = int(max_dimension * aspect_ratio)
                    
                # Resize frame while maintaining aspect ratio
                resized_frame = cv2.resize(frame, (new_width, new_height))
            except Exception as e:
                # print(f"Error resizing frame: {e}")
                resized_frame = frame  # Use original if resize fails
        else:
            # Keep original size if image is already smaller
            resized_frame = frame
        
        # Encode to base64
        try:
            _, buffer = cv2.imencode('.jpg', resized_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
            base64_string = base64.b64encode(buffer).decode('utf-8')
            return base64_string
        except Exception as e:
            # print(f"Error encoding frame to Base64: {e}")
            return None
    except Exception as e:
        # print(f"Unexpected error in Base64 encoding: {e}")
        return None

# Rate-limiting logic for GPT requests
def can_make_request():
    """
    Determines if a GPT request can be made based on the rate-limiting interval.

    Returns:
        bool: True if enough time has passed since the last request, False otherwise.
    """
    global last_request_time
    current_time = time.time()
    if current_time - last_request_time > MIN_REQUEST_INTERVAL:
        last_request_time = current_time
        return True
    return False

def query_gpt(prompt, base64_frame, label=None):
    """Rate-limited GPT query function with improved error handling and classification"""
    if not gpt_rate_limiter.wait_if_needed(label):
        return "Rate limited. Try again later."
        
    if base64_frame is None:
        return "ERROR: Invalid image data"
        
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant that helps people analyze images in VR applications."},
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_frame}"}}
                ]}
            ]
        )
        gpt_rate_limiter.record_success()
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        error_message = str(e)
        
        # Classify error for better handling
        if "429" in error_message or "rate limit" in error_message.lower():
            error_code = 429
            error_type = "rate_limit"
        elif "401" in error_message or "unauthorized" in error_message.lower():
            error_code = 401
            error_type = "authentication"
        elif "5" in error_message[:3]:  # 5xx errors
            error_code = 500
            error_type = "server_error"
        elif "timeout" in error_message.lower() or "timed out" in error_message.lower():
            error_code = 408
            error_type = "timeout"
        else:
            error_code = 400
            error_type = "request_error"
            
        # Log with error type for better diagnostics
        print(f"GPT query failed ({error_type}): {error_message}")
        
        # Record failure with specific error code
        gpt_rate_limiter.record_failure(error_code)
        
        # Return informative message based on error type
        if error_type == "rate_limit":
            return "Rate limit exceeded. Try again later."
        elif error_type == "authentication":
            return "Authentication error. Check API credentials."
        elif error_type == "server_error":
            return "GPT service unavailable. Try again later."
        elif error_type == "timeout":
            return "Request timed out. Try again later."
        else:
            return "Error querying image"

def ask_gpt_about_object(bbox_frame, label = None, original_label=None):
    """
    Optimized version that analyzes object inside its bounding box frame 
    and queries GPT for a concise description.
    
    Args: 
        bbox_frame: the frame that only includes x1,y1 to x2,y2 of the frame
        label: optional object label for context

    Returns: A concise GPT response
    """
    base64_frame = encode_frame_to_base64(bbox_frame)
    if base64_frame is None:
        return "Unable to process image"
    
    semantic_label = label
    if label is not None:
        semantic_label = get_semantic_label(label)
        
    # Specific prompt for command_1 usage - requesting VERY concise descriptions
    image_prompt = (
        f"Don't describe anything about hand. Describe this {semantic_label} VR object in 5-10 words maximum. Be EXTREMELY concise. "
        f"Don't describe anything about hand. Focus only on essential visual characteristics. No introduction, just description."
    )

    # Use rate limiter
    if not gpt_rate_limiter.wait_if_needed():
        return f"a {semantic_label}"  # Simple fallback if rate limited
        
    # Request with short timeout to avoid blocking
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You provide extremely concise visual descriptions in 5-10 words."},
                {"role": "user", "content": [
                    {"type": "text", "text": image_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_frame}"}}
                ]}
            ],
            max_tokens=50,  # Limit to very short responses
            timeout=3.0  # 3 second timeout to prevent hanging
        )
        
        # Record success
        gpt_rate_limiter.record_success()
        
        # Clean up response if needed
        raw_response = response.choices[0].message.content.strip()
        
        # Post-process to ensure brevity
        words = raw_response.split()
        if len(words) > 12:  # If still too verbose
            return " ".join(words[:10]) + "..."  # Truncate
            
        return raw_response
        
    except Exception as e:
        error_message = str(e)
        
        # Classify error
        if "429" in error_message or "rate limit" in error_message.lower():
            error_code = 429
            error_type = "rate_limit"
        elif "timeout" in error_message.lower():
            error_code = 408
            error_type = "timeout"
        else:
            error_code = 400
            error_type = "request_error"
            
        # Log error
        print(f"GPT query failed ({error_type}): {error_message}")
        
        # Record failure
        gpt_rate_limiter.record_failure(error_code)
        
        # Return simple fallback
        return f"a {semantic_label}"
    
def ask_gpt_about_sign_object(bbox_frame, label = None, original_label=None):
    """
    Optimized version that analyzes object inside its bounding box frame 
    and queries GPT for a concise description.
    
    Args: 
        bbox_frame: the frame that only includes x1,y1 to x2,y2 of the frame
        label: optional object label for context

    Returns: A concise GPT response
    """
    base64_frame = encode_frame_to_base64(bbox_frame)
    if base64_frame is None:
        return "Unable to process image"
    
    semantic_label = label
    if label is not None:
        semantic_label = get_semantic_label(label)
        
    # Specific prompt for command_1 usage - requesting VERY concise descriptions
    image_prompt = (
        f"Don't describe anything about hand. Describe this {semantic_label} VR object, reading out the text on it in complete sentences, if applicable."
        f"If you cannot read any text in complete words or sentences, instead describe in 5-10 words, focusing only on essential visual characteristics. No introduction, just description."
    )

    # Use rate limiter
    if not gpt_rate_limiter.wait_if_needed():
        return f"a {semantic_label}"  # Simple fallback if rate limited
        
    # Request with short timeout to avoid blocking
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You read sign objects or provide extremely concise visual descriptions in 5-10 words."},
                {"role": "user", "content": [
                    {"type": "text", "text": image_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_frame}"}}
                ]}
            ],
            max_tokens=50,  # Limit to very short responses
            timeout=3.0  # 3 second timeout to prevent hanging
        )
        
        # Record success
        gpt_rate_limiter.record_success()
        
        # Clean up response if needed
        raw_response = response.choices[0].message.content.strip()
        
        # Post-process to ensure brevity
        words = raw_response.split()
        if len(words) > 20:  # If still too verbose
            return " ".join(words[:20]) + "..."  # Truncate
            
        return raw_response
        
    except Exception as e:
        error_message = str(e)
        
        # Classify error
        if "429" in error_message or "rate limit" in error_message.lower():
            error_code = 429
            error_type = "rate_limit"
        elif "timeout" in error_message.lower():
            error_code = 408
            error_type = "timeout"
        else:
            error_code = 400
            error_type = "request_error"
            
        # Log error
        print(f"GPT query failed ({error_type}): {error_message}")
        
        # Record failure
        gpt_rate_limiter.record_failure(error_code)
        
        # Return simple fallback
        return f"a {semantic_label}"

def ask_gpt_about_hand(bbox_frame, label = None, original_label=None):
    """
    Optimized version that analyzes object inside its bounding box frame 
    and queries GPT for a concise description.
    
    Args: 
        bbox_frame: the frame that only includes x1,y1 to x2,y2 of the frame
        label: optional object label for context

    Returns: A concise GPT response
    """
    base64_frame = encode_frame_to_base64(bbox_frame)
    if base64_frame is None:
        return "Unable to process image"
    
    semantic_label = label
    if label is not None:
        semantic_label = get_semantic_label(label)
        
    # Specific prompt for command_3 fallback usage - requesting VERY concise descriptions
    image_prompt = (
        f"The middle of this frame contains a polygonal, cartoonish hand from the virtual reality game 'Rec Room'. Describe the objects near it in 10 words maximum using the extra data included around the hand. Be EXTREMELY concise. Focus only on essential visual characteristics. "
        f"Do NOT mention the hand or anything about what the hand looks like, only the things around it. Read out any text you see if they are real words, but do not infer any letters. No introduction, just description."
    )

    # Use rate limiter
    if not gpt_rate_limiter.wait_if_needed():
        return f"a {semantic_label}"  # Simple fallback if rate limited
        
    # Request with short timeout to avoid blocking
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You provide extremely concise visual descriptions in 5-10 words."},
                {"role": "user", "content": [
                    {"type": "text", "text": image_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_frame}"}}
                ]}
            ],
            max_tokens=50,  # Limit to very short responses
            timeout=3.0  # 3 second timeout to prevent hanging
        )
        
        # Record success
        gpt_rate_limiter.record_success()
        
        # Clean up response if needed
        raw_response = response.choices[0].message.content.strip()
        
        # Post-process to ensure brevity
        words = raw_response.split()
        if len(words) > 20:  # If still too verbose
            return " ".join(words[:20]) + "..."  # Truncate
            
        return raw_response
        
    except Exception as e:
        error_message = str(e)
        
        # Classify error
        if "429" in error_message or "rate limit" in error_message.lower():
            error_code = 429
            error_type = "rate_limit"
        elif "timeout" in error_message.lower():
            error_code = 408
            error_type = "timeout"
        else:
            error_code = 400
            error_type = "request_error"
            
        # Log error
        print(f"GPT query failed ({error_type}): {error_message}")
        
        # Record failure
        gpt_rate_limiter.record_failure(error_code)
        
        # Return simple fallback
        return f"a {semantic_label}"

# Alternative GPT function with caching for frequently seen objects
_object_description_cache = {}

def ask_gpt_about_object_cached(bbox_frame, label = None):
    """
    Version with memory caching for faster responses to similar objects.
    
    Args: 
        bbox_frame: the frame that only includes x1,y1 to x2,y2 of the frame
        label: optional object label for context

    Returns: A concise GPT response with caching
    """
    if bbox_frame is None:
        return "Unable to process image"
        
    # Get semantic label
    semantic_label = label
    if label is not None:
        semantic_label = get_semantic_label(label)
        
    # Check cache based on label
    if semantic_label in _object_description_cache:
        return _object_description_cache[semantic_label]
    
    # Get base64 encoding
    base64_frame = encode_frame_to_base64(bbox_frame)
    if base64_frame is None:
        return "Unable to process image"
    
    # Same concise prompt as before
    image_prompt = (
        f"Describe this {semantic_label} VR object in 5-10 words maximum. Be EXTREMELY concise. "
        f"Focus only on essential visual characteristics. No introduction, just description."
    )

    # Check rate limiter
    if not gpt_rate_limiter.wait_if_needed():
        fallback = f"a {semantic_label}"
        _object_description_cache[semantic_label] = fallback
        return fallback
        
    # Request with short timeout
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You provide extremely concise visual descriptions in 5-10 words."},
                {"role": "user", "content": [
                    {"type": "text", "text": image_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_frame}"}}
                ]}
            ],
            max_tokens=50,
            timeout=3.0
        )
        
        gpt_rate_limiter.record_success()
        
        # Process response
        raw_response = response.choices[0].message.content.strip()
        
        # Ensure brevity
        words = raw_response.split()
        if len(words) > 12:
            result = " ".join(words[:10]) + "..."
        else:
            result = raw_response
            
        # Cache result
        _object_description_cache[semantic_label] = result
        return result
        
    except Exception as e:
        # Error handling
        fallback = f"a {semantic_label}"
        _object_description_cache[semantic_label] = fallback
        
        # Record error
        error_message = str(e)
        if "429" in error_message:
            gpt_rate_limiter.record_failure(429)
        else:
            gpt_rate_limiter.record_failure(400)
            
        return fallback


def ask_gpt_about_menu(bbox_frame):
    """
    Analyzes menu inside of its bounding box frame and queries GPT for description

    Args: 
        bbox_frame: the frame that only includes x1,y1 to x2,y2 of the frame

    Returns: A GPT response
    """
    base64_frame = encode_frame_to_base64(bbox_frame)
    if base64_frame is None:
        return "Unable to process menu image"

    # Attached is a base64 encoded image of a {label} VR object. Please describe what it looks like.
    image_prompt = (
        f"Attached is a base64 encoded image of a Virtual Reality menu. Describe what it looks like with the structure: 'A menu with [your response]'. If you see a specific object highlighted, via 1. raycast to the shape; 2. unique outline around the shape; and/or 3. a white dot with black outline on the shape, just read out the text or icon on the object: 'A menu with [your response] highlighted'. Your entire response should be VERY concise, using only a few words."
    )

    gpt_response = query_gpt(image_prompt, base64_frame, label="menu")

    return gpt_response

def ask_gpt_about_contents(bbox_frame, x, y):
    """
    Analyzes menu inside of its bounding box frame and queries GPT for description

    Args: 
        bbox_frame: the frame that only includes x1,y1 to x2,y2 of the frame
        x: x coordinate of the cursor
        y: y coordinate of the cursor

    Returns: A GPT response
    """
    base64_frame = encode_frame_to_base64(bbox_frame)
    if base64_frame is None:
        return "Unable to process menu content"

    # Attached is a base64 encoded image of a {label} VR object. Please describe what it looks like.
    image_prompt = (
        f"Attached is a base64 encoded image of a Virtual Reality menu and a cursor at the points {x},{y}. Describe what the cursor is hovering over with the structure: 'A menu with [your response]'. Your entire response should be VERY concise, using only a few words."
    )

    gpt_response = query_gpt(image_prompt, base64_frame, label="menu")

    return gpt_response

def ask_gpt_about_tone(raw_frame):
    """
    Ask GPT about the tone of the frame

    Args: 
        raw_frame: the raw frame. 

    Returns: A GPT response
    """
    print("Asking GPT about tone...")
    base64_frame = encode_frame_to_base64(raw_frame)
    global current_emotion
    if base64_frame is None:
        with emotion_lock:
            current_emotion = "neutral"
        return "neutral"

    # tones = ['neutral', 'sad', 'happy', 'surprising', 'fearful', 'angry', 'disgusting', 'contemptuous']
    tones = ['neutral', 'sad', 'cheerful', 'fearful']

    image_prompt = (
            f"Choose the best tone to describe this frame from: {tones}, keep your response as one word; for example --> 'neutral', without quotation marks"
    )
    gpt_response = query_gpt(image_prompt, base64_frame)
    
    if gpt_response:
        gpt_response = gpt_response.lower().strip()
        # print("Gpt response:", gpt_response)

        # Validate response is one of our expected tones
        if gpt_response not in tones:
            # print(f"WARNING: GPT tone response '{gpt_response}' not in the valid tones list. Defaulting to neutral")
            gpt_response = "neutral"
    else:
        # Handle case where GPT returned nothing
        # print("WARNING: GPT returned empty response for tone detection. Defaulting to neutral")
        gpt_response = "neutral"

    with emotion_lock:
        current_emotion = gpt_response
    
    # print("emotion set as:", current_emotion)

    return current_emotion

# Capture image from frame and ask gpt
def ask_gpt_about_frame(detections, raw_frame):
    """
    Analyzes objects detected in a frame and queries GPT for a description and tone.

    Args:
        detections (list): List of detected objects in the frame.
        raw_frame: The complete frame image

    Returns:
        str: Description of the frame content
    """
    try:
        # add descriptions iff any objects were detected
        if detections:
            # Combine all object data into a single descriptive prompt
            object_descriptions = []
            for det in detections:
                if det is None:
                    continue
                    
                label = det.get('label')
                bbox = det.get('bbox')
                
                if label is None or bbox is None:
                    continue
                
                if label is not None:
                    label = get_semantic_label(label)
                
                object_descriptions.append(f"'{label}' in the bounding box {bbox}")
                
            if not object_descriptions:
                return "ERROR: No valid objects detected."

            # convert the entire frame to base 64
            base64_frame = encode_frame_to_base64(raw_frame)
            if base64_frame is None:
                return "ERROR: Unable to process image"

            combined_description = "; ".join(object_descriptions)
            image_prompt = (
                    f"Describe what's going on in the frame in a single concise sentence. It is a virtual reality environment, but the user already knows that, so don't mention it. Some objects that have been pre-identified include: {combined_description}."
                    f"Use exactly this type of format, for example: 'There is an avatar sitting on a seat reading.'"
            )
            
            # query gpt4o
            gpt_response = query_gpt(image_prompt, base64_frame)
            return gpt_response
        
        # convert the entire frame to base 64
        base64_frame = encode_frame_to_base64(raw_frame)
        if base64_frame is None:
            return "ERROR: Unable to process image"

        image_prompt = (
                f"Describe what's going on in the frame in a single concise sentence. It is a virtual reality environment, but the user already knows that, so don't mention it."
                f"Use exactly this type of format, for example: 'There is an avatar sitting on a seat reading.'"
        )
        
        # query gpt4o
        gpt_response = query_gpt(image_prompt, base64_frame)
        return gpt_response
        
    except Exception as e:
        print(f"Error in ask_gpt_about_frame: {e}")
        return "Error analyzing frame"

def get_current_emotion():
    """Thread-safe getter for current emotion"""
    with emotion_lock:
        emotion = current_emotion if current_emotion is not None else 'neutral'
    return emotion  # tested and working on non-neutral scenes