"""
Grayscale Segmentation for Rectangular Corner Detection (No GUI)
Author: Adithya Satheesh (Modified by AI)
Date: October 19, 2025
Description: Segments a grayscale image and uses morphological closing to merge
             details. It then finds the convex hull of contours to robustly
             detect the corners of solid rectangular shapes, ignoring internal holes.
             This version is non-interactive with hard-coded parameters.
"""

import cv2
import numpy as np
from collections import deque
import time


class CornerDetector:
    def __init__(self):
        """
        Initialize the corner detector with hard-coded parameters.
        All tuning happens by changing the values in this section.
        """
        
        # --- Core Detection Parameters ---
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

        # --- Stability Control ---
        self.corner_history = deque(maxlen=self.smoothing)
        self.fps_history = deque(maxlen=30)
            
    def segment_image(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kernel_size = self.blur_size if self.blur_size % 2 == 1 and self.blur_size > 0 else 1
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
            
            hull = cv2.convexHull(c)
            hull_area = cv2.contourArea(hull)
            peri = cv2.arcLength(hull, True)
            approx = cv2.approxPolyDP(hull, self.epsilon * peri, True)
            
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
            # Draw all found rectangles on the debug view
            for i, rect in enumerate(rectangles[:5]):
                color = (0, 255, 0) if i == 0 else (100, 100, 255)
                cv2.drawContours(debug, [rect['corners']], -1, color, 2)
        
        return output, debug


def main():
    cap = cv2.VideoCapture(1) # Change to 1 if 0 is not your camera
    if not cap.isOpened():
        print("ERROR: Cannot open webcam!")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    detector = CornerDetector()
    
    print("\n" + "=" * 60)
    print(" Corner Detector Running (No GUI)")
    print("  - Parameters are hard-coded in the CornerDetector class.")
    print("\n CONTROLS:")
    print("  'd' - Toggle Debug View")
    print("  'q' - Quit")
    print("=" * 60)
    
    show_debug = False
    debug_window_name = "Debug View (Segmentation Mask)"
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        output, debug = detector.process_frame(frame, show_debug)
        cv2.imshow('Corner Detection', output)
        
        if show_debug and debug is not None:
            cv2.imshow(debug_window_name, debug)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): 
            break
        elif key == ord('d'):
            show_debug = not show_doc
            if not show_debug:
                cv2.destroyWindow(debug_window_name)
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()