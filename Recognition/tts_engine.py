import azure.cognitiveservices.speech as speechsdk
import os
import requests
import base64
from server import AudioPacket
import threading
import time
import traceback
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os
from api_key_rotator import APIKeyRotator
from label_handler_list import get_semantic_label

load_dotenv()

# WebSocket server endpoint
HOST = "localhost"
PORT = 8765

tts_key_rotator = APIKeyRotator(
    service_name="Azure TTS",
    free_key_env_var="ZX_FREE_AZURE_TTS_KEY",
    paid_key_env_var="AZURE_COGNITIVESERVICES_KEY",
    region_env_var="AZURE_TTS_REGION"
)

## subscription key & region needed
PROJECT_ID = "1233172"
FOLDER_ID = "206527486"

# BRANCH_ID = "0bbd8783-7d73-45a4-9281-5ca04b61a4d0"

_tts_service_health = {
    'last_success': None,
    'failure_count': 0,
    'in_degraded_mode': False,
    'degraded_until': None,
    'backoff_factor': 1
}

def prepare_tts_text(text, label=None, context=None):
    """
    Prepare text for TTS by applying semantic mappings.
    ONLY converts labels for TTS, not for processing.
    
    Args:
        text: The text to be synthesized
        label: The technical label associated with this text (optional)
        context: Additional context (optional)
        
    Returns:
        str: Processed text ready for TTS
    """
    # If no manipulation needed, return original
    if not label:
        return text
    
    # Get the semantic label
    semantic_label = get_semantic_label(label, context)
    
    # For simple label announcements (like when just reading out the label name)
    if text == label:
        return semantic_label
    
    # For longer texts that might include descriptions, we should avoid
    # doing simple search-and-replace to prevent false positives
    
    # IMPORTANT: We return the original text, NOT performing replacement
    # This prevents issues with replacing technical terms within longer text
    return text

def synthesize_speech(text, emotion="neutral", name="invalid_category"):
    """
    Synthesize speech with improved error handling and diagnostics
    
    Args:
        text (str): The text to synthesize
        emotion (str): Emotion to apply to the voice
        name (str): Category name for logging
        
    Returns:
        str: Base64 encoded audio data, or None if synthesis fails
    """
    
    print(f"Speaking: '{text[:80]}{'...' if len(text) > 80 else ''}' with emotion: {emotion}")
    
    global _tts_service_health
    
    if not text:
        return None
    
    # Check if we're in degraded mode
    if name != "menu":
        current_time = datetime.now()
        if _tts_service_health['in_degraded_mode']:
            if _tts_service_health['degraded_until'] and current_time < _tts_service_health['degraded_until']:
                # We're in a backoff period - return None
                print(f"TTS service in degraded mode until {_tts_service_health['degraded_until'].strftime('%H:%M:%S')}")
                return None
            else:
                # Try the service again
                _tts_service_health['in_degraded_mode'] = False
    
    # Get the current subscription keys
    current_subscription_key = tts_key_rotator.get_key()
    current_region = tts_key_rotator.get_region() or "eastus"
    
    # print('current key: {}'.format(current_subscription_key))
    # Limit text length to prevent timeouts/errors
    max_text_length = 300
    if len(text) > max_text_length:
        text = text[:max_text_length - 3] + "..."
    
    # Retry parameters
    max_retries = 3
    retry_count = 0
    last_error = None
    
    while retry_count < max_retries:
        try:
            # Detailed logging for diagnostics
            # print(f"TTS attempt {retry_count+1}/{max_retries} for text: '{text[:30]}{'...' if len(text) > 30 else ''}'")
            
            voice_name, style = get_voice_params(emotion)
            speech_config = speechsdk.SpeechConfig(subscription=current_subscription_key, region=current_region)
            speech_config.speech_synthesis_voice_name = voice_name
            
            # Add detailed telemetry for better diagnostics
            # speech_config.set_property(speechsdk.PropertyId.SpeechServiceConnection_InitialSilenceTimeoutMs, "3000")
            # speech_config.set_property(speechsdk.PropertyId.SpeechServiceConnection_EndSilenceTimeoutMs, "1000")
            # speech_config.set_property(speechsdk.PropertyId.Speech_LogFilename, "tts_log.txt")
            
            synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=None)
            ssml = f"""
            <speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis'
                   xmlns:mstts='http://www.w3.org/2001/mstts'
                   xml:lang='en-US'>
                <voice name='{voice_name}'>
                    <mstts:express-as style='{style}' styledegree="2">
                        {text}
                    </mstts:express-as>
                </voice>
            </speak>
            """
            
            # Use a timer to prevent blocking indefinitely
            result = None
            
            # Create a timer to abort the request if it takes too long
            abort_flag = [False]
            def timeout_callback():
                abort_flag[0] = True
                print("TTS request timed out - setting abort flag")
            
            abort_timer = threading.Timer(5.0, timeout_callback)
            abort_timer.daemon = True  # Mark as daemon so it doesn't block shutdown
            abort_timer.start()
            
            try:
                # Check abort flag before making request
                if abort_flag[0]:
                    raise Exception("TTS request aborted before starting - timeout")
                    
                # Make the request
                result = synthesizer.speak_ssml_async(ssml).get()
                
                # Cancel the timer if we got a result
                abort_timer.cancel()
                
                # Check for timeout that happened during API call
                if abort_flag[0]:
                    raise Exception("TTS request timed out during API call")
            finally:
                # Ensure timer is canceled
                abort_timer.cancel()
            
            if result is None:
                raise Exception("No result returned from TTS synthesis.")
                
            if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                # Record successful call
                _tts_service_health['last_success'] = datetime.now()
                _tts_service_health['failure_count'] = 0
                _tts_service_health['backoff_factor'] = 1
                _tts_service_health['in_degraded_mode'] = False
                
                # output_filename = f"audio_sample_{name}_{emotion}.mp3"
                # with open(os.path.join("/Users/zhengxue03/eclipse-workspace/UW-Madison/CS1207/VR-AI_SceneRecognition/Recognition/audio samples/DavisNeural",output_filename), "wb") as audio_file:
                #     audio_file.write(result.audio_data)
                
                tts_key_rotator.record_success()
                
                return base64.b64encode(result.audio_data).decode('utf-8')
            else:
                detailed_error = f"TTS generation failed: {result.reason}"
                if hasattr(result, 'cancellation_details') and result.cancellation_details:
                    detailed_error += f" - Details: {result.cancellation_details.reason}, {result.cancellation_details.error_details}"
                raise Exception(detailed_error)
                
        except Exception as e:
            last_error = str(e)
            retry_count += 1
            
            # Get traceback for debugging
            tb = traceback.format_exc()
            print(f"TTS error ({retry_count}/{max_retries}): {last_error}")
            print(f"Traceback: {tb}")
            
            # Analyze the error
            error_type, action, backoff_time = analyze_tts_error(e)
            
            # Record error and check if we switch keys
            should_switch = tts_key_rotator.record_error(error_type)
            
            if should_switch:
                # Get the new key and retry immediately with the new key
                print(f"Switched TTS API key, retrying request")
                current_subscription_key = tts_key_rotator.get_key()
                # Reset retry count to give the new key a fresh start
                retry_count -= 1
                continue
            
            if name == "menu":
                backoff_time = 0.1  # Minimal backoff for menus
                
            # Record failure
            _tts_service_health['failure_count'] += 1
            
            # If we've had multiple failures, increase backoff
            if _tts_service_health['failure_count'] > 5:
                # Enter degraded mode
                _tts_service_health['in_degraded_mode'] = True
                backoff_minutes = min(30, _tts_service_health['backoff_factor'] * 5)  # Maximum 30 minute backoff
                _tts_service_health['degraded_until'] = current_time + timedelta(minutes=backoff_minutes)
                _tts_service_health['backoff_factor'] *= 2  # Exponential backoff
                
                print(f"TTS service entering degraded mode for {backoff_minutes} minutes due to repeated failures")
                return None
            
            # Take action based on error analysis
            if action == "backoff":
                sleep_time = backoff_time * retry_count
                print(f"TTS backing off for {sleep_time} seconds due to {error_type}")
                time.sleep(sleep_time)
            elif action == "check_credentials":
                print("TTS authentication error - check your API credentials")
                time.sleep(2)
            elif action == "check_parameters":
                print("TTS canceled - may be due to invalid parameters or content")
                time.sleep(1)
            else:  # retry
                print(f"TTS error {error_type} - retrying after short delay")
                time.sleep(retry_count)
    
    # If we've exhausted retries, log the error
    print(f"TTS synthesis failed after {max_retries} attempts. Last error: {last_error}")
    
    # Return None to indicate failure
    return None

def notify_playcanvas(asset_id, api_url):
    payload = {"asset_id": asset_id}
    response = requests.post(api_url, json=payload)
    if response.status_code == 200:
        print(f"PlayCanvas notified about asset ID: {asset_id}")
    else:
        print(f"Error notifying PlayCanvas: {response.status_code}, {response.text}")
        
def get_voice_params(emotion):
    emotions = {
        "neutral": ("en-US-SaraNeural", "default"),
        "sad": ("en-US-SaraNeural", "sad"),
        "cheerful": ("en-US-SaraNeural", "cheerful"),
        "fearful": ("en-US-SaraNeural", "terrified"),
        "urgent": ("en-US-SaraNeural", "shouting")
        # "surprising": ("en-US-SaraNeural", "excited"),
        # "angry": ("en-US-SaraNeural", "angry"),
        # "disgusted": ("en-US-SaraNeural", "unfriendly"),
        # "contempt": ("en-US-SaraNeural", "disgruntled"),
    }
    return emotions.get(emotion, ("en-US-SaraNeural", "default"))

def get_tts_diagnostic_info():
    """Get detailed diagnostic information about the TTS service health"""
    return {
        'last_success': _tts_service_health['last_success'],
        'failure_count': _tts_service_health['failure_count'],
        'in_degraded_mode': _tts_service_health['in_degraded_mode'],
        'degraded_until': _tts_service_health['degraded_until'],
        'backoff_factor': _tts_service_health['backoff_factor']
    }

def analyze_tts_error(error):
    """
    Analyze TTS error to determine cause and appropriate action
    
    Args:
        error: Exception or error message
        
    Returns:
        tuple: (error_type, action, backoff_time)
    """
    error_str = str(error)
    
    # Authorization/credential issues
    if "401" in error_str or "Unauthorized" in error_str:
        return "auth_error", "check_credentials", 120
        
    # Rate limiting
    if "429" in error_str or "Too Many Requests" in error_str:
        return "rate_limit", "backoff", 30
        
    # Service unavailable
    if "503" in error_str or "Service Unavailable" in error_str:
        return "service_unavailable", "backoff", 60
        
    # Timeout
    if "timeout" in error_str.lower() or "timed out" in error_str.lower():
        return "timeout", "retry", 15
        
    # Canceled operation
    if "ResultReason.Canceled" in error_str:
        return "canceled", "check_parameters", 5
        
    return "unknown", "retry", 10