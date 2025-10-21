# Stable Corner Detector

A real-time computer vision system for detecting and tracking the four corners of rectangular objects with visual stability and coordinate display.

[![Python Version](https://img.shields.io/badge/python-3.7%2B-blue)](https://www.python.org/downloads/)
[![OpenCV](https://img.shields.io/badge/opencv-4.0%2B-green)](https://opencv.org/)
[![License](https://img.shields.io/badge/license-MIT-orange)](LICENSE)

![Corner Detection Demo](assets/demo.gif)
*Real-time corner detection with coordinate display*

---

## 📋 Overview

This system implements a real-time computer vision solution for detecting and tracking the four corners of rectangular objects in video streams, with a focus on **visual stability** rather than raw precision. 

### Key Features

- ✅ **Robust Segmentation**: Adaptive thresholding handles varied lighting conditions
- ✅ **Stable Tracking**: 20-frame temporal smoothing eliminates position jitter
- ✅ **Real-time Coordinates**: On-screen display of exact pixel positions
- ✅ **Debug Visualization**: Toggle-able segmentation mask view
- ✅ **Simple Implementation**: Single-file, minimal dependencies

### Architecture Pipeline

```
Video Frame → Grayscale → Gaussian Blur → Adaptive Threshold 
→ Morphological Closing → Contour Detection → Convex Hull 
→ Polygonal Approximation → Rectangularity Validation 
→ Corner Ordering → Temporal Smoothing → Visualization
```

---

## 🚀 Quick Start

### Prerequisites

```bash
pip install opencv-python numpy
```

### Installation

```bash
git clone https://github.com/yourusername/stable-corner-detector.git
cd stable-corner-detector
```

### Usage

```bash
python corner_detector.py
```

**Controls:**
- `d` - Toggle debug view (shows segmentation mask)
- `q` - Quit application

### Camera Configuration

By default, the system uses camera index `1`. To change the camera:

```python
cap = cv2.VideoCapture(0)  # Change to 0 for default webcam
```

---

## ⚙️ Configuration

All detection parameters are configurable in the `__init__` method:

```python
# Segmentation Parameters
self.blur_size = 7                    # Gaussian blur kernel size
self.adaptive_block_size = 55         # Adaptive threshold block size
self.adaptive_c = 7                   # Threshold constant
self.morph_close_size = 15            # Morphological closing kernel

# Filtering Parameters
self.min_area = 2000                  # Minimum object area (px²)
self.max_area_ratio = 0.80            # Max area as frame fraction
self.epsilon = 0.02                   # Contour approximation accuracy
self.rectangularity_threshold = 0.80  # Minimum solidity ratio

# Stability Parameters
self.smoothing_factor = 20            # Frames to average (higher = more stable)
```

---

## 🔬 Technical Details

### Processing Pipeline

#### 1. Image Segmentation
**Method:** Adaptive Gaussian thresholding with morphological closing

Converts frames to grayscale, applies 7×7 Gaussian blur for noise reduction, then performs adaptive thresholding. This computes local thresholds based on neighborhood statistics, effectively handling non-uniform lighting. Morphological closing (dilation + erosion) fills gaps and merges fragmented regions.

**Output:** Binary mask with white objects on black background

#### 2. Rectangle Detection
**Method:** Contour analysis with convex hull and polygonal approximation

Extracts external contours and filters by area (2000px² to 80% of frame). Computes convex hull to eliminate concavities, applies Douglas-Peucker approximation to simplify polygons. Only 4-vertex shapes with solidity ≥ 0.80 are retained.

**Solidity Formula:** `contour_area / bounding_box_area`

#### 3. Corner Ordering
**Method:** Geometric analysis

Orders corners using coordinate properties:
- **Top-Left:** Minimum sum (x + y)
- **Bottom-Right:** Maximum sum (x + y)
- **Top-Right:** Minimum difference (y - x)
- **Bottom-Left:** Maximum difference (y - x)

#### 4. Temporal Smoothing
**Method:** 20-frame moving average filter

Maintains a deque of last 20 corner positions, computes element-wise mean. New corners contribute only 5% weight, creating stable output with 0.33-0.67s lag at 30 FPS.

### Algorithmic Rationale

| Technique | Purpose |
|-----------|---------|
| **Adaptive Thresholding** | Handles uneven illumination by computing local thresholds |
| **Convex Hull** | Repairs incomplete edges and provides occlusion robustness |
| **Temporal Smoothing** | Eliminates visual jitter for stable AR applications |
| **Coordinate Display** | Bridges visual and quantitative feedback (1-3% overhead) |

---

## 📊 Performance

| Metric | Value |
|--------|-------|
| **Frame Rate** | 28-55 FPS (640×480 on modern CPU) |
| **Latency** | 0.33-0.67 seconds (due to smoothing) |
| **Memory** | ~640 bytes (corner history buffer) |
| **Processing** | 1,800-3,300 frames/minute |

---

## ⚠️ Limitations

### Detection Constraints
- ❌ **Responsiveness:** 20-frame lag unsuitable for fast-moving objects
- ❌ **Occlusion:** Requires all 4 corners visible
- ❌ **Single Object:** Tracks only the largest rectangle
- ❌ **Shape Restriction:** Cannot detect non-rectangular objects
- ❌ **Lighting:** Fails in extreme brightness/darkness

### Implementation Constraints
- ❌ **Fixed Parameters:** Requires code modification for different environments
- ❌ **No GPU Acceleration:** CPU-bound processing
- ❌ **No Data Export:** Coordinates visible on-screen only
- ❌ **Integer Precision:** No sub-pixel accuracy

---

## 💡 Use Cases

### ✅ Suitable For
- Educational computer vision demonstrations
- Stationary or slow-moving object tracking
- Controlled lighting environments
- Calibration and measurement tasks
- Prototyping tracking systems
- Applications prioritizing stability over precision

### ❌ Not Suitable For
- Real-time robotics requiring instant response
- Heavy occlusion scenarios
- Multi-object tracking
- Precision measurement applications
- Uncontrolled outdoor environments
- Transparent or highly reflective objects

---

## 🛠️ Potential Improvements

### Detection Enhancements
- [ ] Implement Kalman filtering for predictive tracking
- [ ] Add optical flow for feature-based tracking
- [ ] Deploy ML models for robust detection
- [ ] Enable multi-object tracking with persistent IDs

### Display Enhancements
- [ ] Smart text positioning with overlap detection
- [ ] CSV/file logging for coordinate export
- [ ] Sub-pixel precision display mode
- [ ] Motion trail visualization
- [ ] Confidence metrics display

### System Improvements
- [ ] Adaptive parameter tuning
- [ ] GPU acceleration for morphological operations
- [ ] Region-of-interest tracking
- [ ] Configuration file support

---

## 📁 Project Structure

```
stable-corner-detector/
│
├── corner_detector.py      # Main application file
├── README.md               # This file
├── LICENSE                 # MIT License
├── requirements.txt        # Python dependencies
└── assets/
    └── demo.gif           # Demo animation
```

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Adithya Satheesh**

- GitHub: [@adhi062002]([https://github.com/adhi062002])
- Email: adithya062002@gmail.com

---

## 🙏 Acknowledgments

- OpenCV community for excellent documentation
- Classical computer vision algorithms that remain effective
- All contributors and users of this project

---

## 📚 References

- [OpenCV Documentation](https://docs.opencv.org/)
- [Adaptive Thresholding](https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html)
- [Contour Detection](https://docs.opencv.org/4.x/d4/d73/tutorial_py_contours_begin.html)
- [Douglas-Peucker Algorithm](https://en.wikipedia.org/wiki/Ramer%E2%80%93Douglas%E2%80%93Peucker_algorithm)
