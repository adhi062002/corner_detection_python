"""
Stable Corner Detector (Improved)
Author: Adithya Satheesh (Modified by AI)
Date: July 10, 2026
Description:
    A robust, non-interactive script for stable corner detection of rectangular objects.
    Features:
    - Dynamic parameter scaling with frame size
    - Subpixel corner refinement
    - Configurable via JSON
    - Better contour filtering (aspect ratio, convexity)
    - Logging and error handling
    - Debug visualizations
"""

import cv2
import numpy as np
import json
import logging
import time
from collections import deque
from typing import Optional, Tuple, List, Dict, Any

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class CornerDetector:
    def __init__(self, config_path: str = "config.json"):
        """
        Initialize the corner detector with parameters from a config file.
        """
        self.load_config(config_path)
        self.corner_history = deque(maxlen=self.smoothing_factor)
        self.fps_history = deque(maxlen=30)
        logger.info("CornerDetector initialized with config: %s", config_path)

    def load_config(self, config_path: str) -> None:
        """Load parameters from a JSON config file."""
        try:
            with open(config_path, "r") as f:
                config = json.load(f)

            # Core detection parameters
            core = config["core_detection"]
            self.blur_size = core["blur_size"]
            self.adaptive_block_size_ratio = core["adaptive_block_size_ratio"]
            self.adaptive_c = core["adaptive_c"]
            self.invert_threshold = core["invert_threshold"]
            self.morph_close_size = core["morph_close_size"]
            self.min_area = core["min_area"]
            self.max_area_ratio = core["max_area_ratio"]
            self.epsilon = core["epsilon"]
            self.rectangularity_threshold = core["rectangularity_threshold"]
            self.aspect_ratio_threshold = core["aspect_ratio_threshold"]
            self.convexity_threshold = core["convexity_threshold"]

            # Stability parameters
            stability = config["stability"]
            self.smoothing_factor = stability["smoothing_factor"]
            self.subpixel_refinement = stability["subpixel_refinement"]

            # Debug parameters
            debug = config["debug"]
            self.show_debug = debug["show_debug"]
            self.save_results = debug["save_results"]

            logger.info("Config loaded successfully.")
        except Exception as e:
            logger.error("Failed to load config: %s. Using defaults.", e)
            self._set_defaults()

    def _set_defaults(self) -> None:
        """Set default parameters if config loading fails."""
        self.blur_size = 7
        self.adaptive_block_size_ratio = 0.1
        self.adaptive_c = 7
        self.invert_threshold = 1
        self.morph_close_size = 15
        self.min_area = 2000
        self.max_area_ratio = 0.80
        self.epsilon = 0.02
        self.rectangularity_threshold = 0.80
        self.aspect_ratio_threshold = 2.0
        self.convexity_threshold = 0.95
        self.smoothing_factor = 20
        self.subpixel_refinement = True
        self.show_debug = False
        self.save_results = False

    def segment_image(self, frame: np.ndarray) -> np.ndarray:
        """
        Segment the image using adaptive thresholding and morphology.
        Dynamically scales block size with frame dimensions.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Dynamic blur size (must be odd)
        kernel_size = max(1, self.blur_size if self.blur_size % 2 == 1 else self.blur_size + 1)
        blurred = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)

        # Dynamic adaptive block size (must be odd and > 1)
        height, width = gray.shape
        block_size = max(3, int(min(height, width) * self.adaptive_block_size_ratio))
        block_size = block_size if block_size % 2 == 1 else block_size + 1

        threshold_type = cv2.THRESH_BINARY_INV if self.invert_threshold == 1 else cv2.THRESH_BINARY
        segmented_mask = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, threshold_type,
            block_size, self.adaptive_c
        )

        if self.morph_close_size > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.morph_close_size, self.morph_close_size))
            segmented_mask = cv2.morphologyEx(segmented_mask, cv2.MORPH_CLOSE, kernel)

        return segmented_mask

    def is_rectangular(self, contour: np.ndarray) -> bool:
        """
        Check if a contour is sufficiently rectangular using:
        - Solidity (contour area / bounding box area)
        - Aspect ratio
        - Convexity (contour area / convex hull area)
        """
        area = cv2.contourArea(contour)
        if area == 0:
            return False

        # Bounding box and aspect ratio
        _, (w, h), _ = cv2.minAreaRect(contour)
        bounding_box_area = w * h
        if bounding_box_area == 0:
            return False
        aspect_ratio = max(w, h) / min(w, h)
        if aspect_ratio > self.aspect_ratio_threshold:
            return False

        # Solidity
        solidity = area / bounding_box_area
        if solidity < self.rectangularity_threshold:
            return False

        # Convexity
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0:
            return False
        convexity = area / hull_area
        if convexity < self.convexity_threshold:
            return False

        return True

    def find_rectangles(self, segmented_mask: np.ndarray, frame_shape: Tuple[int, int]) -> List[Dict[str, Any]]:
        """Find all valid rectangles and return them sorted by area."""
        contours, _ = cv2.findContours(segmented_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_area = frame_shape[0] * frame_shape[1] * self.max_area_ratio

        rectangles = []
        for c in contours:
            area = cv2.contourArea(c)
            if not (self.min_area < area < max_area):
                continue

            hull = cv2.convexHull(c)
            peri = cv2.arcLength(hull, True)
            approx = cv2.approxPolyDP(hull, self.epsilon * peri, True)

            if len(approx) == 4 and self.is_rectangular(hull):
                rectangles.append({"corners": approx, "area": area})

        rectangles.sort(key=lambda x: x["area"], reverse=True)
        return rectangles

    def order_corners(self, corners: np.ndarray) -> np.ndarray:
        """Order corners: Top-Left, Top-Right, Bottom-Right, Bottom-Left."""
        pts = corners.reshape(4, 2)
        ordered = np.zeros((4, 2), dtype=np.float32)
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)
        ordered[0] = pts[np.argmin(s)]  # Top-Left
        ordered[2] = pts[np.argmax(s)]  # Bottom-Right
        ordered[1] = pts[np.argmin(diff)]  # Top-Right
        ordered[3] = pts[np.argmax(diff)]  # Bottom-Left
        return ordered

    def refine_corners(self, corners: np.ndarray, gray: np.ndarray) -> np.ndarray:
        """Refine corner positions using subpixel accuracy."""
        if not self.subpixel_refinement:
            return corners

        # Prepare corners for cornerSubPix (requires float32 and specific shape)
        corners_float = corners.reshape(-1, 1, 2).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
        try:
            refined_corners = cv2.cornerSubPix(gray, corners_float, winSize=(3, 3), zeroZone=(-1, -1), criteria=criteria)
            return refined_corners.reshape(4, 2)
        except:
            logger.warning("Subpixel refinement failed. Using original corners.")
            return corners

    def smooth_corners(self, corners: np.ndarray) -> np.ndarray:
        """Apply temporal smoothing to stabilize corner positions."""
        self.corner_history.append(corners)
        smoothed = np.mean(self.corner_history, axis=0)
        return smoothed.astype(np.float32)

    def draw_corners(self, frame: np.ndarray, corners: np.ndarray) -> np.ndarray:
        """Draw the 4 corners with labels and coordinates on the frame."""
        output = frame.copy()
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0)]  # TL, TR, BR, BL
        labels = ["TL", "TR", "BR", "BL"]

        for i, (corner, color, label) in enumerate(zip(corners, colors, labels)):
            x, y = int(corner[0]), int(corner[1])

            # Draw corner circle and outline
            cv2.circle(output, (x, y), 8, color, -1)
            cv2.circle(output, (x, y), 10, (255, 255, 255), 2)

            # Draw label and coordinates
            label_text = f"{label}"
            coord_text = f"({x},{y})"
            text_pos_x, text_pos_y = x + 15, y + 5
            cv2.putText(output, label_text, (text_pos_x, text_pos_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(output, coord_text, (text_pos_x, text_pos_y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Draw connecting lines
        for i in range(4):
            pt1 = tuple(corners[i].astype(int))
            pt2 = tuple(corners[(i + 1) % 4].astype(int))
            cv2.line(output, pt1, pt2, (0, 255, 255), 2)

        return output

    def add_info_overlay(self, frame: np.ndarray, has_corners: bool, fps: float) -> np.ndarray:
        """Draw an info overlay on the top-left of the frame."""
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (300, 90), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        status = "DETECTED" if has_corners else "SEARCHING..."
        status_color = (0, 255, 0) if has_corners else (0, 0, 255)

        cv2.putText(frame, "Stable Corner Detector", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, status, (180, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        return frame

    def process_frame(self, frame: np.ndarray, show_debug: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Main processing pipeline for a single frame.
        Returns:
            - Output frame with corners drawn
            - Debug frame (if show_debug=True)
        """
        if frame is None or frame.size == 0:
            logger.error("Invalid frame input.")
            return frame, None

        start_time = time.time()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Segment and find rectangles
        segmented_mask = self.segment_image(frame)
        rectangles = self.find_rectangles(segmented_mask, frame.shape)

        output = frame.copy()
        corners = None

        if rectangles:
            best_rectangle = rectangles[0]
            ordered_corners = self.order_corners(best_rectangle["corners"])
            ordered_corners = self.refine_corners(ordered_corners, gray)
            corners = self.smooth_corners(ordered_corners)
            output = self.draw_corners(output, corners)

            if self.save_results:
                self._save_corners(corners)
        else:
            if self.corner_history:
                self.corner_history.popleft()

        # Calculate FPS
        elapsed_time = time.time() - start_time
        fps = 1.0 / elapsed_time if elapsed_time > 0 else 999.0
        self.fps_history.append(fps)
        avg_fps = np.mean(self.fps_history)
        output = self.add_info_overlay(output, corners is not None, avg_fps)

        # Debug visualization
        debug = None
        if show_debug:
            debug = cv2.cvtColor(segmented_mask, cv2.COLOR_GRAY2BGR)
            for i, rect in enumerate(rectangles[:5]):
                color = (0, 255, 0) if i == 0 else (100, 100, 255)
                cv2.drawContours(debug, [rect["corners"]], -1, color, 2)

        return output, debug

    def _save_corners(self, corners: np.ndarray) -> None:
        """Save corner coordinates to a CSV file."""
        try:
            with open("corners.csv", "a") as f:
                flat_corners = corners.flatten()
                f.write(f"{time.time()},{','.join(map(str, flat_corners))}\n")
        except Exception as e:
            logger.error("Failed to save corners: %s", e)

def main():
    # Load config
    detector = CornerDetector()

    # Initialize camera
    cap = cv2.VideoCapture(detector.camera_device_id)
    if not cap.isOpened():
        logger.error("ERROR: Cannot open webcam!")
        return

    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, detector.camera_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, detector.camera_height)

    # Print instructions
    print("\n" + "=" * 60)
    print(" Stable Corner Detector Running...")
    print("  - Press 'd' to toggle debug view.")
    print("  - Press 's' to start/stop saving corner coordinates.")
    print("  - Press 'q' to quit.")
    print("=" * 60)

    show_debug = detector.show_debug
    debug_window_name = "Debug View"

    while True:
        ret, frame = cap.read()
        if not ret:
            logger.error("Failed to read frame.")
            break

        output, debug = detector.process_frame(frame, show_debug)
        cv2.imshow("Corner Detection", output)

        if show_debug and debug is not None:
            cv2.imshow(debug_window_name, debug)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("d"):
            show_debug = not show_debug
            if not show_debug:
                cv2.destroyWindow(debug_window_name)
        elif key == ord("s"):
            detector.save_results = not detector.save_results
            logger.info("Saving corners: %s", detector.save_results)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Add camera settings to CornerDetector class
    try:
        with open("config.json", "r") as f:
            config = json.load(f)
            CornerDetector.camera_device_id = config["camera"]["device_id"]
            CornerDetector.camera_width = config["camera"]["width"]
            CornerDetector.camera_height = config["camera"]["height"]
    except:
        CornerDetector.camera_device_id = 1
        CornerDetector.camera_width = 640
        CornerDetector.camera_height = 480

    main()
