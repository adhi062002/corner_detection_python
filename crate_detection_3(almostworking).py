"""
Grayscale Segmentation for Rectangular Corner Detection
Author: Adithya Satheesh (Modified by AI)
Date: October 19, 2025
Description: Segments a grayscale image and uses morphological closing to merge
             details. It then finds the convex hull of contours to robustly
             detect the corners of solid rectangular shapes, ignoring internal holes.
"""

import cv2
import numpy as np
from collections import deque
import time
import json
import os


class CornerDetector:
    def __init__(self, settings_file="settings.json"):
        """Initialize the corner detector with tunable parameters for segmentation."""
        
        self.settings_file = settings_file
        
        # --- Set Default Parameters ---
        self.blur_size = 7
        self.epsilon = 0.02
        self.min_area = 1000
        self.smoothing = 5
        self.max_area_ratio = 0.80
        self.rectangularity_threshold = 0.80
        self.adaptive_block_size = 41 
        self.adaptive_c = 5
        self.invert_threshold = 1
        self.morph_close_size = 10

        self.corner_history = deque(maxlen=self.smoothing)
        self.fps_history = deque(maxlen=30)
        
        self.setup_controls()
        self.load_settings()

    def save_settings(self):
        """Save the current control parameters to a JSON file."""
        settings = {
            'blur_size': self.blur_size, 'epsilon': self.epsilon, 'min_area': self.min_area,
            'smoothing': self.smoothing, 'max_area_ratio': self.max_area_ratio,
            'rectangularity_threshold': self.rectangularity_threshold,
            'adaptive_block_size': self.adaptive_block_size, 'adaptive_c': self.adaptive_c,
            'invert_threshold': self.invert_threshold, 'morph_close_size': self.morph_close_size
        }
        try:
            with open(self.settings_file, 'w') as f:
                json.dump(settings, f, indent=4)
            print(f"\n[INFO] Settings saved to {self.settings_file}")
        except IOError as e:
            print(f"\n[ERROR] Could not save settings: {e}")

    def load_settings(self):
        """Load control parameters from a JSON file if it exists."""
        if not os.path.exists(self.settings_file):
            print("[INFO] No settings file found. Using default values.")
            return
        try:
            with open(self.settings_file, 'r') as f:
                settings = json.load(f)
                for key, value in settings.items():
                    setattr(self, key, value) # Set all attributes from the file
            print(f"[INFO] Settings loaded from {self.settings_file}")
            self.update_controls()
        except (IOError, json.JSONDecodeError) as e:
            print(f"\n[ERROR] Could not load or parse settings file: {e}")

    def update_controls(self):
        """Update the trackbar positions to match the current parameter values."""
        cv2.setTrackbarPos('Blur', 'Controls', self.blur_size)
        cv2.setTrackbarPos('Min Area', 'Controls', self.min_area // 10)
        cv2.setTrackbarPos('Epsilon x100', 'Controls', int(self.epsilon * 100))
        cv2.setTrackbarPos('Smoothing', 'Controls', self.smoothing)
        cv2.setTrackbarPos('Adaptive Block Size', 'Controls', self.adaptive_block_size)
        cv2.setTrackbarPos('Adaptive C', 'Controls', self.adaptive_c)
        cv2.setTrackbarPos('Invert Threshold', 'Controls', self.invert_threshold)
        cv2.setTrackbarPos('Max Area %', 'Controls', int(self.max_area_ratio * 100))
        cv2.setTrackbarPos('Rectangularity %', 'Controls', int(self.rectangularity_threshold * 100))
        cv2.setTrackbarPos('Morph Close Size', 'Controls', self.morph_close_size)
    
    def setup_controls(self):
        """Create control panel for parameter tuning."""
        cv2.namedWindow('Controls')
        
        cv2.createTrackbar('Blur', 'Controls', self.blur_size, 25, 
                          lambda x: setattr(self, 'blur_size', x if x % 2 == 1 and x > 0 else x + 1))
        cv2.createTrackbar('Min Area', 'Controls', self.min_area // 10, 2000, 
                          lambda x: setattr(self, 'min_area', x * 10))
        cv2.createTrackbar('Epsilon x100', 'Controls', int(self.epsilon * 100), 10, 
                          lambda x: setattr(self, 'epsilon', x / 100))
        cv2.createTrackbar('Smoothing', 'Controls', self.smoothing, 20, 
                          self.update_smoothing)
        cv2.createTrackbar('Adaptive Block Size', 'Controls', self.adaptive_block_size, 255,
                           lambda x: setattr(self, 'adaptive_block_size', x if x % 2 == 1 and x > 2 else 3))
        cv2.createTrackbar('Adaptive C', 'Controls', self.adaptive_c, 50,
                           lambda x: setattr(self, 'adaptive_c', x))
        cv2.createTrackbar('Invert Threshold', 'Controls', self.invert_threshold, 1,
                           lambda x: setattr(self, 'invert_threshold', x))
        cv2.createTrackbar('Max Area %', 'Controls', int(self.max_area_ratio * 100), 100,
                           lambda x: setattr(self, 'max_area_ratio', x / 100.0))
        cv2.createTrackbar('Rectangularity %', 'Controls', int(self.rectangularity_threshold * 100), 100,
                           lambda x: setattr(self, 'rectangularity_threshold', x / 100.0))
        cv2.createTrackbar('Morph Close Size', 'Controls', self.morph_close_size, 50,
                           lambda x: setattr(self, 'morph_close_size', x))

    def update_smoothing(self, value):
        if value > 0:
            self.smoothing = value
            self.corner_history = deque(self.corner_history, maxlen=value)
            
    def segment_image(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kernel_size = self.blur_size if self.blur_size % 2 == 1 else self.blur_size + 1
        blurred = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
        
        block_size = self.adaptive_block_size if self.adaptive_block_size % 2 == 1 and self.adaptive_block_size > 1 else 3
        threshold_type = cv2.THRESH_BINARY_INV if self.invert_threshold == 1 else cv2.THRESH_BINARY
        
        segmented_mask = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, threshold_type,
            block_size, self.adaptive_c
        )
        
        if self.morph_close_size > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.morph_close_size, self.morph_close_size))
            segmented_mask = cv2.morphologyEx(segmented_mask, cv2.MORPH_CLOSE, kernel)
        
        return segmented_mask

    def is_rectangular(self, contour):
        _, (width, height), _ = cv2.minAreaRect(contour)
        contour_area = cv2.contourArea(contour)
        bounding_box_area = width * height
        if bounding_box_area == 0: return False
        ratio = contour_area / bounding_box_area
        return ratio >= self.rectangularity_threshold

    def find_rectangles(self, segmented_mask, frame_shape):
        """Find rectangular contours using their convex hull for robustness."""
        contours, _ = cv2.findContours(segmented_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_area = frame_shape[0] * frame_shape[1] * self.max_area_ratio
        
        rectangles = []
        for c in contours:
            area = cv2.contourArea(c)
            if not (self.min_area < area < max_area):
                continue
            
            # --- NEW STEP: Use the Convex Hull ---
            # This creates a solid shape around the contour, ignoring holes.
            hull = cv2.convexHull(c)
            
            # Now, perform all checks on the simplified hull
            hull_area = cv2.contourArea(hull)
            peri = cv2.arcLength(hull, True)
            approx = cv2.approxPolyDP(hull, self.epsilon * peri, True)
            
            # Check if the solid hull shape is a 4-sided polygon
            if len(approx) == 4 and self.is_rectangular(hull):
                rectangles.append({'corners': approx, 'area': hull_area})

        rectangles.sort(key=lambda x: x['area'], reverse=True)
        return rectangles

    def order_corners(self, corners):
        pts = corners.reshape(4, 2)
        ordered = np.zeros((4, 2), dtype=np.float32)
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)
        ordered[0] = pts[np.argmin(s)]
        ordered[2] = pts[np.argmax(s)]
        ordered[1] = pts[np.argmin(diff)]
        ordered[3] = pts[np.argmax(diff)]
        return ordered

    def smooth_corners(self, corners):
        self.corner_history.append(corners)
        if len(self.corner_history) == 1: return corners
        return np.mean(self.corner_history, axis=0).astype(np.float32)

    def draw_corners(self, frame, corners):
        output = frame.copy()
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0)]
        labels = ['TL', 'TR', 'BR', 'BL']
        for i, (corner, color, label) in enumerate(zip(corners, colors, labels)):
            x, y = int(corner[0]), int(corner[1])
            cv2.circle(output, (x, y), 8, color, -1)
            cv2.circle(output, (x, y), 10, (255, 255, 255), 2)
            cv2.putText(output, f'{label}', (x + 15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        for i in range(4):
            pt1 = tuple(corners[i].astype(int))
            pt2 = tuple(corners[(i + 1) % 4].astype(int))
            cv2.line(output, pt1, pt2, (0, 255, 255), 2)
        return output

    def add_info_overlay(self, frame, num_shapes, has_corners, fps):
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (300, 100), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        title = "Segmentation Corner Detector"
        status = "DETECTED" if has_corners else "No shape found"
        status_color = (0, 255, 0) if has_corners else (0, 0, 255)
        
        cv2.putText(frame, title, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f'FPS: {fps:.1f}', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f'Shapes Found: {num_shapes}', (20, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, status, (220, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 2)
        return frame

    def process_frame(self, frame, show_debug=False):
        start_time = time.time()
        
        segmented_mask = self.segment_image(frame)
        rectangles = self.find_rectangles(segmented_mask, frame.shape)

        corners = None
        output = frame.copy()
        
        if rectangles:
            best_rect = rectangles[0]
            ordered = self.order_corners(best_rect['corners'])
            corners = self.smooth_corners(ordered)
            output = self.draw_corners(frame, corners)
        
        elapsed_time = time.time() - start_time
        fps = 1.0 / elapsed_time if elapsed_time > 0 else 999.0
        
        self.fps_history.append(fps)
        avg_fps = np.mean(self.fps_history)
        
        output = self.add_info_overlay(output, len(rectangles), corners is not None, avg_fps)
        
        debug = None
        if show_debug:
            debug = cv2.cvtColor(segmented_mask, cv2.COLOR_GRAY2BGR)
            for i, rect in enumerate(rectangles[:5]):
                color = (0, 255, 0) if i == 0 else (100, 100, 255)
                # Draw the approximated corners on the debug view
                cv2.drawContours(debug, [rect['corners']], -1, color, 2)
        
        return output, corners, debug


def main():
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam!")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    detector = CornerDetector()
    
    print("\n" + "=" * 60)
    print(" ROBUST CORNER DETECTION (V5 - with Convex Hull)")
    print("=" * 60)
    print("\n HOW TO TUNE:")
    print("  - The system now uses a 'convex hull' to ignore holes in objects,")
    print("    making detection much more stable.")
    print("  - Use 'Morph Close Size' to merge separate parts of an object.")
    print("\n CONTROLS:")
    print("  'q' - Quit (and save settings)")
    print("  'c' - Manually Save Current Settings")
    print("  'd' - Toggle Debug View")
    print("  'r' - Reset Smoothing")
    print("=" * 60)
    
    show_debug = False
    debug_window_name = "Debug View (Segmentation Mask)"
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        output, corners, debug = detector.process_frame(frame, show_debug)
        cv2.imshow('Corner Detection', output)
        
        if show_debug and debug is not None:
            cv2.imshow(debug_window_name, debug)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): 
            print("\n[INFO] Exiting and saving settings...")
            detector.save_settings()
            break
        elif key == ord('c'):
            detector.save_settings()
        elif key == ord('d'):
            show_debug = not show_debug
            if not show_debug:
                cv2.destroyWindow(debug_window_name)
        elif key == ord('r'):
            detector.corner_history.clear()
            print("\n[RESET] Smoothing history cleared")
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()