import time
import threading
import queue
from typing import Dict, Optional, Callable, List

class CategoryCooldownManager:
    """
    Enhanced cooldown manager that queues speech instead of skipping.
    - Maintains a speech queue to ensure sequential playback
    - Allows pre-processing while waiting for speech to finish
    - Prioritizes user commands and special labels
    """
    def __init__(self):
        # Category-specific cooldown times (in seconds)
        # Only keep cooldowns for special_labels and user_command
        self.category_cooldowns = {
            "special_labels": 5,  # 5 seconds for all special labels
            "user_command": 1,    # 1 second cooldown after a command finishes
        }
        
        # Minimum silence period between utterances (seconds)
        self.min_silence_period = 1
        
        # Timestamps of last execution per label
        self.last_execution = {}
        
        # Text history for similarity checking
        self.text_history = {}
        
        # Speech state tracking
        self.currently_speaking = False
        self.speech_end_time = 0
        self.last_speech_end_time = 0  # To track silence periods
        
        # User command status
        self.user_command_active = False
        self.user_command_start_time = 0
        self.user_command_end_time = 0
        self.user_command_max_duration = 120  # Maximum duration in seconds (2 minutes)
        
        # Special label execution tracking
        self.special_label_cooldowns = {}  # Individual cooldowns for special labels
        self.special_label_cooldown_times = {}  # Cooldown times for special labels
        
        # Speech queue system
        self.speech_queue = queue.Queue()
        self.speech_worker_active = False
        self.speech_worker_thread = None
        
        # Failed speech tracking
        self.failed_speech_attempts = 0
        self.max_failed_attempts = 5

        self.debug_mode = False
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Start the speech worker thread
        self.start_speech_worker()
        
    def start_speech_worker(self):
        """Start the background worker thread that processes queued speech"""
        if not self.speech_worker_active:
            self.speech_worker_active = True
            self.speech_worker_thread = threading.Thread(
                target=self._speech_worker_thread_func, 
                daemon=True
            )
            self.speech_worker_thread.start()
    
    def _speech_worker_thread_func(self):
        """Background thread that processes queued speech sequentially with improved error handling"""
        while self.speech_worker_active:
            try:
                # Wait for next item in queue with timeout to allow clean shutdown
                try:
                    speech_item = self.speech_queue.get(timeout=0.5)
                except queue.Empty:
                    continue
                
                # Process the speech item
                try:
                    # Extract item details
                    category = speech_item.get('category')
                    label = speech_item.get('label')
                    text = speech_item.get('text')
                    speech_func = speech_item.get('speech_func')
                    callback = speech_item.get('callback')
                    timestamp = speech_item.get('timestamp', time.time())
                    
                    # Check if item is too old and should be skipped
                    current_time = time.time()
                    if current_time - timestamp > 60:  # Skip items older than 30 seconds
                        if self.debug_mode:
                            print(f"Skipping stale speech item: '{text}' (age: {current_time - timestamp:.1f}s)")
                        self.speech_queue.task_done()
                        continue
                    
                    # Wait for minimum silence period if needed
                    time_since_last_speech = current_time - self.last_speech_end_time

                    # Determine if this is a short text that needs less silence
                    is_short_text = text and len(text.split()) <= 2 and len(text) < 15

                    # Use shorter silence period for short texts
                    if is_short_text:
                        min_silence = 0.2  # Much shorter silence for short texts
                    else:
                        min_silence = self.min_silence_period

                    if time_since_last_speech < min_silence:
                        # Wait to respect the silence period
                        wait_time = min_silence - time_since_last_speech
                        if self.debug_mode:
                            print(f"Waiting {wait_time:.2f}s for silence period before speaking '{label}'")
                        time.sleep(wait_time)
                    
                    # Mark as speaking and estimate duration
                    with self.lock:
                        self.currently_speaking = True
                        duration = self.estimate_speech_duration(text)
                        # Add extra padding to duration to ensure complete playback
                        duration += 0.2  # Add 0.2 seconds padding
                        self.speech_end_time = time.time() + duration
                    
                    # Execute the speech function with error tracking
                    success = False
                    try:
                        if speech_func and callable(speech_func):
                            speech_func()
                            success = True
                        else:
                            print(f"WARNING: Speech function not callable for '{label}'")
                    except Exception as speech_error:
                        print(f"ERROR executing speech for '{label}': {speech_error}")
                        self.failed_speech_attempts += 1
                        
                        # If we've had too many failures, clear the queue
                        if self.failed_speech_attempts >= self.max_failed_attempts:
                            print(f"WARNING: Too many failed speech attempts ({self.failed_speech_attempts}). Clearing queue.")
                            self.clear_queue()
                            self.failed_speech_attempts = 0
                    
                    # If successful, reset failed attempts counter
                    if success:
                        self.failed_speech_attempts = 0
                    
                    # Wait for estimated speech duration or reduced time if speech failed
                    wait_duration = duration if success else min(duration, 0.5)
                    time.sleep(wait_duration)
                    
                    # Add additional gap after speech completes
                    time.sleep(0.5)  # 0.5 second gap after speech finishes
                    
                    # Update state after speech completes
                    with self.lock:
                        self.currently_speaking = False
                        self.last_speech_end_time = time.time()
                    
                    # Call the callback function if provided
                    if callback and callable(callback):
                        try:
                            callback()
                        except Exception as callback_error:
                            print(f"ERROR in speech callback: {callback_error}")
                    
                except Exception as e:
                    print(f"Error processing queued speech: {e}")
                    self.failed_speech_attempts += 1
                
                finally:
                    # Mark task as done
                    self.speech_queue.task_done()
            
            except Exception as e:
                print(f"Critical error in speech worker thread: {e}")
                time.sleep(1)  # Brief pause before continuing
    
    def stop_speech_worker(self):
        """Stop the speech worker thread gracefully"""
        self.speech_worker_active = False
        if self.speech_worker_thread and self.speech_worker_thread.is_alive():
            self.speech_worker_thread.join(timeout=1.0)
            
    def queue_speech(self, 
             category: str, 
             label: str, 
             text: Optional[str], 
             speech_func: Callable, 
             callback: Optional[Callable] = None,
             priority: int = 0) -> bool:
        """
        Queue speech for sequential execution
        
        Args:
            category: Category of the speech
            label: Specific label for the speech (ORIGINAL LABEL, not semantic)
            text: Text content (for duration estimation)
            speech_func: Function to execute to generate speech
            callback: Optional callback after speech completes
            priority: Priority level (higher = more urgent)
                
        Returns:
            bool: True if speech was queued, False if rejected
        """
        # Generate a key that combines category and label - using original label
        key = f"{category}:{label}"
        
        if text is None or text == "":
            text = " "  # Default text for empty cases
        
        with self.lock:
            current_time = time.time()
            
            # Automatically bypass cooldown for user commands
            bypass_cooldown = category == "user_command"
            
            # Check for individual cooldown on this exact label
            if not bypass_cooldown and key in self.last_execution:
                cooldown_time = self.category_cooldowns.get(category, 5)
                time_since_last = current_time - self.last_execution[key]
                
                if time_since_last < cooldown_time:
                    if self.debug_mode:
                        print(f"Label '{label}' is on cooldown, skipping (time since last: {time_since_last:.1f}s, cooldown: {cooldown_time}s)")
                    return False
            
            # Check for similar recent text to avoid repetition (also bypass for user commands)
            if not bypass_cooldown and text and key in self.text_history:
                last_text = self.text_history[key]
                if self._is_similar(text, last_text):
                    if self.debug_mode:
                        print(f"Text '{text}' is too similar to recent utterance, skipping")
                    return False
            
            # Update execution record and text history only if we're actually queuing the speech
            self.last_execution[key] = current_time
            if text:
                self.text_history[key] = text
                
            # Add to queue
            speech_item = {
                'category': category,
                'label': label,  # Original label
                'text': text,
                'speech_func': speech_func,
                'callback': callback,
                'priority': priority,
                'timestamp': current_time
            }
            
            # Add to the queue
            self.speech_queue.put(speech_item)
            if self.debug_mode:
                print(f"Queued speech: {category}:{label} - '{text[:30]}{'...' if text and len(text) > 30 else ''}' (priority: {priority})")
            return True
            
    def estimate_speech_duration(self, text: Optional[str]) -> float:
        """
        Estimate the duration of speech based on text length with improved handling for short texts.
        
        Args:
            text: The text to be spoken
            
        Returns:
            Estimated duration in seconds
        """
        if text is None or text == "":
            return 0.3  # Default short duration for empty text based on 'each word' calculation below
        
        # Rough estimate: average speaking rate is about 150 words per minute
        # So each word takes about 0.4 seconds
        words = text.split()
        word_count = len(words)
        char_count = len(text)
        
        # Very short text (1-2 words) gets much shorter duration
        if word_count <= 2 and char_count < 15:
            # Shorter minimum time for very short texts
            base_duration = max(0.4, word_count * 0.3)
            # Smaller processing delay for short texts
            processing_delay = 0.3
        else:
            # Base duration with minimum of 1 second
            base_duration = max(1, word_count * 0.3)
            # Longer sentences need more time on average
            # if word_count > 10:
                # base_duration += 1.2  # Add extra second for longer sentences
            # Standard processing delay
            processing_delay = 0.7
        
        # Total duration
        duration = base_duration + processing_delay
        
        return duration
        
    def is_speaking(self) -> bool:
        """
        Check if speech is currently ongoing.
        
        Returns:
            True if currently speaking, False otherwise
        """
        with self.lock:
            current_time = time.time()
            
            # If we're past the estimated end time, consider speech done
            if self.currently_speaking and current_time > self.speech_end_time:
                self.currently_speaking = False
                self.last_speech_end_time = current_time
                
            return self.currently_speaking
            
    def clear_queue(self):
        """Clear all pending speech from the queue"""
        try:
            # Empty the queue
            items_cleared = 0
            while not self.speech_queue.empty():
                try:
                    self.speech_queue.get_nowait()
                    self.speech_queue.task_done()
                    items_cleared += 1
                except queue.Empty:
                    break
                    
            if items_cleared > 0 and self.debug_mode:
                print(f"Cleared {items_cleared} items from speech queue")
                
        except Exception as e:
            print(f"Error clearing speech queue: {e}")
            
    def is_queue_empty(self) -> bool:
        """Check if the speech queue is empty"""
        return self.speech_queue.empty()
        
    def start_user_command(self, command_type: str) -> None:
        """
        Mark the beginning of a user command execution with timeout tracking
        
        Args:
            command_type: Type of command ('1' or '2')
        """
        with self.lock:
            # Clear pending non-critical speech
            self.clear_queue()
            
            self.user_command_active = True
            self.user_command_start_time = time.time()
            
            # User commands have longer cooldowns due to more complex outputs
            duration = self.category_cooldowns.get("user_command", 10)
            self.user_command_end_time = time.time() + duration
            
            if self.debug_mode:
                print(f"Starting user command: {command_type} (max duration: {self.user_command_max_duration}s)")
            
    def end_user_command(self) -> None:
        """Mark the end of user command execution."""
        with self.lock:
            command_duration = time.time() - self.user_command_start_time
            self.user_command_active = False
            
            if self.debug_mode:
                print(f"Ending user command (duration: {command_duration:.1f}s)")
            
    def is_user_command_active(self) -> bool:
        """
        Check if a user command is currently being processed with timeout safety
        
        Returns:
            True if a user command is active, False otherwise
        """
        with self.lock:
            current_time = time.time()
            
            # Check for command timeout (safety measure)
            if self.user_command_active and self.user_command_start_time > 0:
                command_duration = current_time - self.user_command_start_time
                if command_duration > self.user_command_max_duration:
                    print(f"WARNING: User command timed out after {command_duration:.1f}s (max: {self.user_command_max_duration}s)")
                    self.user_command_active = False
                    return False
            
            # If we're past the estimated end time, consider command done
            if self.user_command_active and current_time > self.user_command_end_time:
                self.user_command_active = False
                
            return self.user_command_active
    
    def register_special_label(self, label, cooldown_time):
        """
        Register a special label with a specific cooldown time.
        
        Args:
            label: The special label to register
            cooldown_time: Cooldown time in seconds
        """
        with self.lock:
            self.special_label_cooldowns[label] = 0  # Initialize with 0 (no cooldown yet)
            # Store the cooldown time separately
            self.special_label_cooldown_times[label] = cooldown_time
            print(f"Registered special label '{label}' with cooldown {cooldown_time}s")
            
    def can_execute_special_label(self, label: str) -> bool:
        """
        Check if a special label can be executed based on its individual cooldown.
        
        Args:
            label: The special label to check
            
        Returns:
            True if the label can be executed, False otherwise
        """
        with self.lock:
            current_time = time.time()
            
            # Check if the label is on cooldown
            if label in self.special_label_cooldowns:
                time_since_last = current_time - self.special_label_cooldowns[label]
                # Use the specific cooldown time for this label if available
                cooldown_time = self.special_label_cooldown_times.get(label, 5)  # Default 5s cooldown
                
                if time_since_last < cooldown_time:
                    if self.debug_mode:
                        print(f"Special label {label} on cooldown ({time_since_last:.1f}s < {cooldown_time}s)")
                    return False
                    
            return True
    
    def can_execute_category(self, category: str, label: str, text: str = "") -> bool:
        """
        Check if a label in a specific category can be executed.
        Only applies cooldowns to special_labels and user_command categories.
        
        Args:
            category: The category of the label
            label: The label to check
            text: Optional text to check for similarity
            
        Returns:
            True if the label can be executed, False otherwise
        """
        if label == 'menu':
            return True
        

        # Only apply cooldowns to special_labels and user_command categories
        if category not in ["special_labels", "user_command"]:
            return True  # Always allow regular categories
            
        with self.lock:
            current_time = time.time()
            key = f"{category}:{label}"
            
            # Check cooldown for this category+label
            if key in self.last_execution:
                cooldown_time = self.category_cooldowns.get(category, 5)
                time_since_last = current_time - self.last_execution[key]
                
                if time_since_last < cooldown_time:
                    if self.debug_mode:
                        print(f"Cooldown active for {category}:{label} ({time_since_last:.1f}s < {cooldown_time}s)")
                    return False
            
            # If text is provided, check for similarity with previous text
            if text and key in self.text_history:
                last_text = self.text_history[key]
                if self._is_similar(text, last_text):
                    if self.debug_mode:
                        print(f"Similar text detected for {category}:{label}")
                    return False
                    
            return True
            
    def record_special_label_execution(self, label: str) -> None:
        """
        Record the execution of a special label.
        
        Args:
            label: The special label that was executed
        """
        with self.lock:
            self.special_label_cooldowns[label] = time.time()
    
    def record_execution(self, category: str, label: str, text: str = "") -> None:
        """
        Record the execution of a label in a specific category.
        
        Args:
            category: The category of the label
            label: The label that was executed
            text: Optional text to record
        """
        with self.lock:
            key = f"{category}:{label}"
            self.last_execution[key] = time.time()
            
            if text:
                self.text_history[key] = text
    
    def can_trigger(self, label: str) -> bool:
        """
        Check if a label can be triggered (compatible with old CooldownManager interface)
        
        Args:
            label: The label to check
            
        Returns:
            True if the label can be triggered, False otherwise
        """
        if label == 'menu':
            return True
        
        # For special labels, use their specific cooldown logic
        if label in self.special_label_cooldowns:
            return self.can_execute_special_label(label)
            
        # For regular labels, check category cooldown
        with self.lock:
            current_time = time.time()
            time_since_last = current_time - self.special_label_cooldowns.get(label, 0)
            cooldown_time = 5  # Always use 5 seconds for special labels
            
            if time_since_last < cooldown_time:
                if self.debug_mode:
                    print(f"Special label {label} on cooldown ({time_since_last:.1f}s < {cooldown_time}s)")
                return False
        
        # For all other labels, no cooldown
        return True
    
    def trigger(self, label: str) -> None:
        """
        Mark a label as triggered (compatible with old CooldownManager interface)
        
        Args:
            label: The label that was triggered
        """
        # For special labels, use record_special_label_execution
        if label in self.special_label_cooldowns:
            self.record_special_label_execution(label)
        
        # For regular labels, record in last_execution
        with self.lock:
            key = f"trigger:{label}"
            self.last_execution[key] = time.time()
            
    def _is_similar(self, text1: str, text2: str, threshold: float = 0.7) -> bool:
        """
        Simple placeholder for text similarity check.

        Args:
            text1: First text
            text2: Second text
            threshold: Similarity threshold
            
        Returns:
            True if texts are similar, False otherwise
        """
        if text1 is None or text2 is None:
            return False
            
        # Very simple similarity check - would be replaced with proper implementation
        return text1.lower() == text2.lower()
    
    def toggle_debug(self):
        """Toggle debug mode to show detailed speech processing information"""
        self.debug_mode = not self.debug_mode
        print(f"Speech cooldown manager debug mode: {'ON' if self.debug_mode else 'OFF'}")
   