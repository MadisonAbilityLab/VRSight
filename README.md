<p align="center">
  <h1 align="center">VRSight: An AI-Driven Scene Description System to Improve Virtual Reality Accessibility for Blind People</h1>
  <p align="center">
    Daniel Killough¹, Justin Feng¹*, Zheng Xue "ZX" Ching¹*, Daniel Wang¹*, Rithvik Dyava¹*, Yapeng Tian², Yuhang Zhao¹
    <br><br>
    <sup>1</sup>University of Wisconsin-Madison,
    <sup>2</sup>University of Texas at Dallas<br>
    *Authors 2-5 contributed equally to this work.
    <br>
    <sub>Presented at UIST 2025 in Busan, Republic of Korea</sub>
  </p>
  <h3 align="center">
    <a href="https://dl.acm.org/doi/full/10.1145/3746059.3747641">Paper</a> |
    <a href="https://github.com/MadisonAbilityLab/VRSight">Code</a> |
    <a href="https://github.com/MadisonAbilityLab/VRSight/releases">VRSight System</a> | 
    <a href="https://huggingface.co/datasets/UWMadAbility/DISCOVR">DISCOVR Dataset</a> |
    <a href="https://huggingface.co/UWMadAbility/VRSight">Fine-Tuned Model Weights</a>
  </h3>
</p>

<p align="center">
<strong>VRSight</strong> provides spatial audio feedback for blind and low vision users in virtual reality (VR) environments by leveraging AI systems like real-time object detection, zero-shot depth estimation, and multimodal large language models. VRSight provides real-time audio descriptions and spatial interaction assistance without per-app developer integration, creating the first post hoc "3D screen reading" system for VR.
    <h3 align="center">
        <a href="https://x.com/i/status/1969153746337665262">Video Preview</a>
  </h3>
</p>

## ⚡ Quick*start
\* Setup is not so quick and likely will require a sighted aide experienced with installing systems via command line. Expect 1 hour. Opportunity for future improvement.

### Hardware Requirements
- **GPU**: NVIDIA GPU with CUDA support (recommended) or Apple Silicon (MPS)
- **RAM**: Minimum 8GB, recommended 16GB+
- **Storage**: 2GB+ for models and dependencies
- **VR Headset**: Any VR headset should work as long as you can cast the output. We've tested with Quest line of headsets (2, 3, Pro) using Meta Quest Developer Hub.
- **Corresponding casting utility for your VR Headset**: e.g., Meta Quest Developer Hub, SteamVR mirror, etc.

### Recommended Additional Hardware:
- **3-key keyboard**: e.g., [this one](https://www.amazon.com/dp/B09TMVXNGG) (non-affiliate link)
- **Long USB cables** (3m+)

### Software Requirements
- **PyTorch**: Install version compatible with your CUDA version.
- **Azure Models**: Requires valid subscriptions to Microsoft Azure for OpenAI, Cognitive Services, and SpeechSynthesizer. Estimated cost $25/year.
- **WebVR Utility**: We use PlayCanvas (free), which you can clone this repo: https://playcanvas.com/project/1233172/overview/vr-scene
- **Websocket Backend Utility**: We use Render (free), which you can clone this repo: https://vrsight-backend.onrender.com/

Opportunity for future work: Using on-device VLMs, OCR, TTS instead of querying Azure. Would increase hardware requirements but reduce monetary cost.

## Code Setup

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/MadisonAbilityLab/VRSight.git
cd VRSight/Recognition

# Create conda environment
conda create --name vrsight python=3.9
conda activate vrsight

# Install PyTorch (adjust for your CUDA version). Example:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install -r requirements.txt
```

### 2. Model Setup

```bash
# Create weights directory
mkdir -p weights

# Download VRSight model weights from HuggingFace
# Option 1: Using wget
wget -O weights/best.pt https://huggingface.co/UWMadAbility/VRSight/resolve/main/best.pt

# Option 2: Using curl
curl -L -o weights/best.pt https://huggingface.co/UWMadAbility/VRSight/resolve/main/best.pt

# Option 3: Manual download
# Visit: https://huggingface.co/UWMadAbility/VRSight/blob/main/best.pt
# Click "Download" and place in weights/best.pt

# Verify model download
python -c "
import torch
try:
    model = torch.load('weights/best.pt', map_location='cpu')
    print('✅ Model loaded successfully')
    print(f'Model size: {os.path.getsize(\"weights/best.pt\") / (1024*1024):.1f} MB')
except Exception as e:
    print(f'❌ Model loading failed: {e}')
"
```

### 3. Configuration

```bash
# Copy and edit environment configuration
cp .env.example .env

# Add your API keys to your .env
export AZURE_OPENAI_API_KEY="your_openai_key"
export AZURE_COGNITIVESERVICES_KEY="your_azure_tts_key"
export AZURE_TTS_REGION="your_region"
```



### 4. Launch Companion Apps
On your computer, launch your cloned webVR utility (e.g., https://playcanvas.com/project/1233172/overview/vr-scene) -> Launch.

If you're using a standalone headset like Meta Quest 3, open the built-in browser and navigate to the same Launch window. Then press the VR button to enter scene. 

Once loaded in, you can safely return to menu and continue using your VR headset as normal, but <strong>do not quit the VR scene -- keep it running in the background.</strong>

### 5. Run VRSight

Basic run with default settings:
```bash
python main.py
```

Run with specific camera index:
```bash
# Check available cameras
python -c "
import cv2
for i in range(5):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f'Camera {i}: Available')
        cap.release()
    else:
        print(f'Camera {i}: Not available')
"

# Run with specific index:
python main.py --camera-index [index]
```

### System Controls
Once VRSight is running:
- **1**: Trigger ContextCompass (general scene descriptions using GPT)
- **2**: Trigger SceneSweep (left-to-right spatial audio descriptions)
- **3**: Trigger AimAssist (specific, targeted spatial audio descriptions near user's hand or pointer end)
- **ESC** or **q**: Exit the application

### Expected Behavior
When running successfully, you should see:
- Real-time object detection, depth estimation, and edge detection preview windows
- Console output showing detected objects
- Audio feedback through configured TTS system
- WebSocket server running on localhost:8765

<!-- 
## 🎯 Overview

VRSight seeks virtual reality accessibility through a sophisticated multi-modal AI pipeline without obstructing base VR interaction functionality:

- **🔍 Real-time Object Detection**: YOLO-based detection engine with custom VR object recognition
- **📏 Depth Estimation**: DepthAnything V2 for accurate spatial understanding
- **🎧 Spatial Audio**: 3D positional audio feedback via Azure TTS integration
- **🎮 VR Interaction**: Hand/controller tracking with precise interaction detection
- **🧠 AI Scene Description**: Natural language scene understanding 
-->

## 🔧 Configuration

### Files
```python
# config_manager.py - Environment-specific settings
from config_manager import get_config

config = get_config()
print(f"Running in {config.environment.value} mode")
print(f"Using device: {config.models.device}")
```

### Key Configuration Options (Defaults Below):
```yaml
camera:
  width: 640
  height: 640
  webcam_index: 1

models:
  yolo_model_path: "weights/best.pt"
  depth_encoder: "vits"  # vits, vitb, vitl, vitg
  device: "auto"  # auto, cuda, mps, cpu

performance:
  memory_cleanup_threshold_mb: 1000
  thread_heartbeat_timeout: 10
  queue_max_size: 10

rate_limiting:
  gpt_min_request_interval: 10
  cooldown_interactables: 30
```

## 📊 Performance Metrics

VRSight achieves real-time performance:

### Processing Performance
- **Frame Rate**: 30+ FPS real-time processing
- **Latency**: Feedback as low as 2ms
- **Memory Usage**: 30% reduction through optimized buffering
- **High Accuracy**: Custom YOLO model trained on DISCOVR dataset: mAP50. Base model on COCO rarely detected; see paper for more details.

## 🎯 DISCOVR Dataset

VRSight is powered by the **DISCOVR** dataset, the first comprehensive VR object detection dataset.

### Dataset Overview
- **30 Object Classes** across 6 categories, including 
- **Training Images**
- **Validation Images**: [PLACEHOLDER: Number of validation images]
- **Test Images**: [PLACEHOLDER: Number of test images]
- YOLO Format for fine-tuning YOLOv8 model. 
- **Weights** available on [HuggingFace](https://huggingface.co/UWMadAbility/VRSight/blob/main/best.pt)

### VR-Specific Object Classes
[PLACEHOLDER: List of specific VR object categories, e.g.]
- Virtual UI elements (buttons, menus, panels)
- VR controllers and hand representations
- Interactive objects (grabbable items)
- Environmental elements
- [PLACEHOLDER: Add other specific categories]

### Dataset Features
- **Resolution**: 640x640
- **Environments**: [PLACEHOLDER: Types of VR environments covered]
- **Annotation Quality**: [PLACEHOLDER: Annotation methodology and quality metrics]

### Model Training
- **Base Model**: YOLOv8
- **Training Strategy**: [PLACEHOLDER: Fine-tuning approach, transfer learning details]
- **Performance Metrics**: [PLACEHOLDER: mAP scores, precision, recall on DISCOVR dataset]
- **Weights**: Available at [HuggingFace](https://huggingface.co/UWMadAbility/VRSight/blob/main/best.pt)

### Dataset Access
[PLACEHOLDER: How to access DISCOVR dataset]
- **Download Link**: [PLACEHOLDER: Direct download link if available]
- **License**: [PLACEHOLDER: Dataset license information]
- **Citation Requirements**: [PLACEHOLDER: How to cite the dataset]

## 🔍 API Reference

### Core Engines
```python
# Object Detection
from object_detection_engine import ObjectDetectionEngine
engine = ObjectDetectionEngine()

# Depth Estimation
from depth_detection_engine import DepthDetectionEngine
depth_engine = DepthDetectionEngine(encoder='vits')

# Thread Management
from thread_manager import get_thread_manager
thread_mgr = get_thread_manager()

# Memory Management
from memory_manager import get_memory_manager
memory_mgr = get_memory_manager()
```

### Configuration Management
```python
from config_manager import get_config, Environment

# Get environment-specific config
config = get_config()

# Manual environment setup
from config_manager import ConfigManager
manager = ConfigManager(Environment.PRODUCTION)
config = manager.load_config()
```

## 🚨 Troubleshooting

### Common Issues

**GPU Memory Errors**
```bash
# Reduce memory usage
export VR_AI_ENV=development  # Uses higher cleanup thresholds
```

**Model Loading Failures**
```bash
# Verify model file
python -c "import torch; torch.load('weights/best.pt')"

# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"
```

**Audio Issues**
```bash
# Check Azure TTS configuration
python -c "import azure.cognitiveservices.speech as speechsdk; print('Azure TTS available')"
```

### Performance Optimization

**For CPU-only systems:**
```python
# Force CPU mode in configuration
config.models.device = "cpu"
config.performance.queue_max_size = 5
```

**For high-performance systems:**
```python
# Enable optimizations
config.models.model_precision = "fp16"
config.performance.memory_cleanup_threshold_mb = 2000
```



## Contributing




### 🏗️ System Architecture
VRSight consists of a modular architecture as follows:

#### Core Detection Engines
- **`object_detection_engine.py`**: YOLO-based object detection with error recovery
- **`depth_detection_engine.py`**: DepthAnythingV2 depth estimation
- **`edge_detection_engine.py`**: VR pointer/line detection for interactions
  
<strong>Object detection models can be easily swapped</strong> by updating the `yolo_model_path` constant in `config.py` to a different `.pt` weight file.

#### Advanced Processing Systems
- **`scene_sweep_processor.py`**: Comprehensive scene reading and description
- **`aim_assist_processor.py`**: Hand/controller pointing command processing
- **`aim_assist_menu_pilot_processor.py`**: Additional handling for VR menu interaction (opportunity for future improvement)
- **`interaction_detection.py`**: Ray-casting and spatial interaction analysis

#### Infrastructure & Optimization
- **`thread_manager.py`**: Unified thread coordination and resource management
- **`memory_manager.py`**: Advanced memory optimization with leak detection
- **`config_manager.py`**: Environment-specific configuration management
- **`unified_rate_limiter.py`**: Intelligent rate limiting across all services

#### Utilities & Support
- **`geometry_utils.py`**: Spatial calculations and coordinate operations
- **`audio_utils.py`**: TTS synthesis and spatial audio management

## 🚀 Key Features

- **🎯 Multi-Modal Recognition**: Object detection, depth estimation, edge detection, and OCR
- **🔄 Real-time Processing**: Optimized pipeline achieving 30+ FPS with automatic quality scaling
- **🎧 Spatial Audio Feedback**: 3D positional audio with Azure TTS integration
- **🎮 VR Interaction Support**: Hand/controller tracking with precise targeting assistance
- **📊 Advanced Analytics**: Performance monitoring, memory management, and error recovery
- **🌍 Multi-Environment Support**: Development, production, and testing configurations
- **🧪 Enterprise Quality**: Comprehensive testing, benchmarking, and validation suite
- **⚙️ Modular Design**: 11 specialized modules following SOLID principles

## 🤝 Contributing

We welcome open-source contributions to improve VRSight!

1. Fork the repository
2. Create a feature branch
3. Push your code to the feature branch
4. Submit a pull request with clear description of the changes

## 📜 Citation

If you use VRSight or DISCOVR in your research, please cite our work:

```bibtex
@article{killough2024vrsight,
  title={VRSight: An AI-Driven Scene Description System to Improve Virtual Reality Accessibility for Blind People},
  author={Killough, Daniel and Feng, Justin and Ching, Zheng Xue and Wang, Daniel and Dyava, Rithvik and Tian, Yapeng and Zhao, Yuhang},
  journal={arXiv preprint arXiv:2508.02958},
  year={2024}
}
```

## 📄 License

This project is licensed under the Creative Commons Attribution 4.0 International License (CC-BY-4.0) - see the [LICENSE](LICENSE) file for details.

You are free to share and adapt this work for any purpose as long as you provide appropriate attribution to the original authors.

## Acknowledgments

We thank the University of Wisconsin-Madison Ability Lab, the University of Texas at Dallas, and all contributors to the DISCOVR dataset. Special thanks to the accessibility community for their invaluable feedback and testing.

For questions and support, please open a GitHub Issue or contact Daniel Killough at the MadAbility Lab at UW-Madison.
