import os
import time
import threading

class APIKeyRotator:
    """
    Manages API key rotation for services like Azure Cognitive Services.
    Starts with free keys and falls back to paid keys when rate limits are reached.
    """
    def __init__(self, service_name, free_key_env_var, paid_key_env_var, 
                 free_endpoint_env_var=None, paid_endpoint_env_var=None,
                 region_env_var=None):
        self.service_name = service_name
        
        # Keys
        self.free_key = os.getenv(free_key_env_var)
        self.paid_key = os.getenv(paid_key_env_var)
        self.current_key = self.free_key
        
        # Endpoints (optional, used for OCR)
        self.free_endpoint = os.getenv(free_endpoint_env_var) if free_endpoint_env_var else None
        self.paid_endpoint = os.getenv(paid_endpoint_env_var) if paid_endpoint_env_var else None
        self.current_endpoint = self.free_endpoint
        
        # Region (optional, used for TTS)
        self.region = os.getenv(region_env_var) if region_env_var else None
        
        # State
        self.using_free = True
        self.error_count = 0
        self.last_error_time = 0
        self.lock = threading.Lock()
        
        # Track successful calls to measure health
        self.success_count = 0
        self.consecutive_successes = 0
        self.auto_recovery_threshold = 50  # Number of successful calls before trying free key again
        
        print(f"Initialized {service_name} key rotator. Free key available: {bool(self.free_key)}, Paid key available: {bool(self.paid_key)}")
        
    def get_key(self):
        """Get the current API key"""
        with self.lock:
            return self.current_key
            
    def get_endpoint(self):
        """Get the current endpoint URL (for OCR)"""
        with self.lock:
            return self.current_endpoint
            
    def get_region(self):
        """Get the region (for TTS)"""
        return self.region
            
    def record_error(self, error_code=None):
        """
        Record an API error and potentially switch keys
        
        Args:
            error_code: Optional error code for better decision making
        
        Returns:
            bool: True if switched to a new key, False otherwise
        """
        with self.lock:
            current_time = time.time()
            self.error_count += 1
            self.consecutive_successes = 0
            self.last_error_time = current_time
            
            # Check if we need to switch keys
            switch_threshold = 3  # Number of errors before switching
            
            # Rate limit errors trigger immediate switch
            if error_code == 429 or "429" in str(error_code):
                switch_threshold = 1
                
            if self.error_count >= switch_threshold:
                if self.using_free and self.paid_key:
                    print(f"API key rate limited, switching keys...")
                    self.current_key = self.paid_key
                    self.current_endpoint = self.paid_endpoint
                    self.using_free = False
                    self.error_count = 0
                    return True
                    
            return False
            
    def record_success(self):
        """
        Record a successful API call. After many successes, may try reverting to free key.
        
        Returns:
            bool: True if the key was switched back to free, False otherwise
        """
        with self.lock:
            self.success_count += 1
            self.consecutive_successes += 1
            self.error_count = 0
            
            # If using paid key but free key is available and we've had many successes,
            # try switching back to free key
            if not self.using_free and self.free_key and self.consecutive_successes >= self.auto_recovery_threshold:
                # print(f"Attempting to switch back to free {self.service_name} API key after {self.consecutive_successes} successful calls")
                self.current_key = self.free_key
                self.current_endpoint = self.free_endpoint
                self.using_free = True
                self.consecutive_successes = 0
                return True
                
            return False