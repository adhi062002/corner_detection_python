"""
Simple Corner Detection for Rectangular Shapes
Author: Adithya Satheesh
Date: October 19, 2025
Description: Detects corners of any rectangular object and labels them.
"""

import cv2
import numpy as np
from collections import deque
import time


class CornerDetector:
    def __init__(self):
        """Initialize the corner detector with tunable parameters."""
        
        # Detection parameters
        self.blur_size = 5
        self.canny_low = 50
        self.canny_high = 150
        self.epsilon = 0.02
        self.min_area = 500
        self.smoothing = 5
        
        # History for smoothing
        self.corner_history = deque(maxlen=self.smoothing)
        self.fps_history = deque(maxlen=30)
        
        # Create control window
        self.setup_controls()
    
    def setup_controls(self):
        """Create control panel for parameter tuning."""
        cv2.namedWindow('Controls')
        cv2.createTrackbar('Blur', 'Controls', self.blur_size, 25, 
                          lambda x: setattr(self, 'blur_size', x if x % 2 == 1 and x > 0 else x + 1))
        cv2.createTrackbar('Canny Low', 'Controls', self.canny_low, 255, 
                          lambda x: setattr(self, 'canny_low', x))
        cv2.createTrackbar('Canny High', 'Controls', self.canny_high, 255, 
                          lambda x: setattr(self, 'canny_high', x))
        cv2.createTrackbar('Epsilon x100', 'Controls', int(self.epsilon * 100), 10, 
                          lambda x: setattr(self, 'epsilon', x / 100))
        cv2.createTrackbar('Min Area', 'Controls', self.min_area // 10, 1000, 
                          lambda x: setattr(self, 'min_area', x * 10))
        cv2.createTrackbar('Smoothing', 'Controls', self.smoothing, 20, 
                          self.update_smoothing)
    
    def update_smoothing(self, value):
        """Update smoothing window."""
        if value > 0:
            self.smoothing = value
            self.corner_history = deque(self.corner_history, maxlen=value)
    
    def detect_edges(self, frame):
        """Detect edges in the frame."""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        kernel = self.blur_size if self.blur_size % 2 == 1 else self.blur_size + 1
        blurred = cv2.GaussianBlur(gray, (kernel, kernel), 0)
        
        # Apply Canny edge detection
        edges = cv2.Canny(blurred, self.canny_low, self.canny_high)
        
        # Dilate to connect broken edges
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        
        return edges
    
    def find_rectangles(self, edges):
        """Find all rectangular contours."""
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        rectangles = []
        
        for contour in contours:
            # Calculate area
            area = cv2.contourArea(contour)
            
            # Filter by minimum area
            if area < self.min_area:
                continue
            
            # Approximate the contour
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, self.epsilon * perimeter, True)
            
            # Check if it's a quadrilateral (4 corners)
            if len(approx) == 4:
                rectangles.append({
                    'corners': approx,
                    'area': area
                })
        
        # Sort by area (largest first)
        rectangles.sort(key=lambda x: x['area'], reverse=True)
        
        return rectangles
    
    def order_corners(self, corners):
        """
        Order corners consistently: Top-Left, Top-Right, Bottom-Right, Bottom-Left.
        
        Args:
            corners: Array of 4 corner points
            
        Returns:
            Ordered corners
        """
        # Reshape to (4, 2)
        pts = corners.reshape(4, 2)
        ordered = np.zeros((4, 2), dtype=np.float32)
        
        # Calculate sum and difference
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)
        
        # Top-left: smallest sum
        ordered[0] = pts[np.argmin(s)]
        # Bottom-right: largest sum
        ordered[2] = pts[np.argmax(s)]
        # Top-right: smallest difference
        ordered[1] = pts[np.argmin(diff)]
        # Bottom-left: largest difference
        ordered[3] = pts[np.argmax(diff)]
        
        return ordered
    
    def smooth_corners(self, corners):
        """Apply temporal smoothing to reduce jitter."""
        self.corner_history.append(corners)
        
        if len(self.corner_history) == 1:
            return corners
        
        # Average over history
        smoothed = np.mean(self.corner_history, axis=0)
        return smoothed.astype(np.float32)
    
    def draw_corners(self, frame, corners):
        """Draw the 4 corners with labels and coordinates."""
        output = frame.copy()
        
        # Define colors and labels for each corner
        colors = [
            (0, 255, 0),      # Green - Top-Left
            (255, 0, 0),      # Blue - Top-Right  
            (0, 0, 255),      # Red - Bottom-Right
            (255, 255, 0)     # Cyan - Bottom-Left
        ]
        
        labels = [
            'Top-Left',
            'Top-Right',
            'Bottom-Right',
            'Bottom-Left'
        ]
        
        short_labels = ['TL', 'TR', 'BR', 'BL']
        
        # Draw each corner
        for i, (corner, color, label, short) in enumerate(zip(corners, colors, labels, short_labels)):
            x, y = int(corner[0]), int(corner[1])
            
            # Draw filled circle
            cv2.circle(output, (x, y), 8, color, -1)
            # Draw outline
            cv2.circle(output, (x, y), 10, (255, 255, 255), 2)
            
            # Draw label background
            label_text = short
            (tw, th), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            
            # Position label to avoid going off screen
            label_x = x + 15
            label_y = y - 10
            
            if label_x + tw > output.shape[1] - 10:
                label_x = x - tw - 15
            if label_y - th < 10:
                label_y = y + th + 10
            
            # Draw background rectangle
            cv2.rectangle(output, 
                         (label_x - 3, label_y - th - 3), 
                         (label_x + tw + 3, label_y + baseline + 3), 
                         (0, 0, 0), -1)
            
            # Draw label text
            cv2.putText(output, label_text, (label_x, label_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Draw coordinates
            coord_text = f'({x}, {y})'
            cv2.putText(output, coord_text, (label_x, label_y + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Draw lines connecting corners
        for i in range(4):
            pt1 = tuple(corners[i].astype(int))
            pt2 = tuple(corners[(i + 1) % 4].astype(int))
            cv2.line(output, pt1, pt2, (0, 255, 255), 2)
        
        # Draw center point
        center = np.mean(corners, axis=0).astype(int)
        cv2.circle(output, tuple(center), 5, (255, 0, 255), -1)
        cv2.putText(output, 'Center', (center[0] + 8, center[1] - 8), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
        
        return output
    
    def add_info_overlay(self, frame, num_shapes, has_corners, fps):
        """Add information overlay."""
        output = frame.copy()
        
        # Semi-transparent background
        overlay = output.copy()
        cv2.rectangle(overlay, (10, 10), (300, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, output, 0.3, 0, output)
        
        # Title
        cv2.putText(output, 'Corner Detector', (20, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # FPS
        cv2.putText(output, f'FPS: {fps:.1f}', (20, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Shapes detected
        cv2.putText(output, f'Rectangles: {num_shapes}', (20, 82), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Status
        status = 'CORNERS DETECTED' if has_corners else 'No rectangles'
        color = (0, 255, 0) if has_corners else (0, 0, 255)
        cv2.putText(output, status, (20, 105), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return output
    
    def process_frame(self, frame, show_debug=False):
        """Main processing pipeline."""
        start_time = time.time()
        
        # Detect edges
        edges = self.detect_edges(frame)
        
        # Find rectangles
        rectangles = self.find_rectangles(edges)
        
        corners = None
        
        if rectangles:
            # Use the largest rectangle
            best_rect = rectangles[0]
            
            # Order corners
            ordered = self.order_corners(best_rect['corners'])
            
            # Smooth corners
            corners = self.smooth_corners(ordered)
            
            # Draw corners
            output = self.draw_corners(frame, corners)
        else:
            output = frame.copy()
        
        # Calculate FPS
        elapsed = time.time() - start_time
        fps = 1.0 / elapsed if elapsed > 0 else 30.0
        self.fps_history.append(fps)
        avg_fps = np.mean(self.fps_history)
        
        # Add overlay
        output = self.add_info_overlay(output, len(rectangles), corners is not None, avg_fps)
        
        # Create debug view
        debug = None
        if show_debug:
            debug = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            # Draw all detected rectangles in debug view
            for i, rect in enumerate(rectangles[:5]):  # Show top 5
                color = (0, 255, 0) if i == 0 else (100, 100, 255)
                cv2.drawContours(debug, [rect['corners']], -1, color, 2)
        
        return output, corners, debug


def main():
    """Main function."""
    
    # Open webcam
    cap = cv2.VideoCapture(1)
    
    if not cap.isOpened():
        print("ERROR: Cannot open webcam!")
        print("Try changing camera index: cv2.VideoCapture(1)")
        return
    
    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Initialize detector
    detector = CornerDetector()
    
    print("=" * 60)
    print(" RECTANGULAR CORNER DETECTION SYSTEM")
    print("=" * 60)
    print("\n CONTROLS:")
    print("  'q'     - Quit")
    print("  's'     - Save screenshot with corner coordinates")
    print("  'd'     - Toggle debug view (edge detection)")
    print("  'r'     - Reset smoothing")
    print("  'h'     - Print help")
    print("\n TUNING:")
    print("  Use the 'Controls' window trackbars to adjust detection")
    print("  - Increase Canny thresholds if too many edges")
    print("  - Decrease if missing edges")
    print("  - Adjust Min Area to filter small objects")
    print("=" * 60)
    
    show_debug = False
    screenshot_count = 0
    
    while True:
        # Read frame
        ret, frame = cap.read()
        
        if not ret:
            print("Failed to grab frame")
            break
        
        # Process frame
        output, corners, debug = detector.process_frame(frame, show_debug)
        
        # Show output
        cv2.imshow('Corner Detection', output)
        
        # Show debug if enabled
        if show_debug and debug is not None:
            cv2.imshow('Edge Detection (Debug)', debug)
        
        # Handle keyboard
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print("\nExiting...")
            break
            
        elif key == ord('s'):
            # Save screenshot
            filename = f'corners_{screenshot_count:03d}.png'
            cv2.imwrite(filename, output)
            print(f"\n[SAVED] {filename}")
            
            if corners is not None:
                print("\nCorner Coordinates (pixels):")
                labels = ['Top-Left', 'Top-Right', 'Bottom-Right', 'Bottom-Left']
                for i, (corner, label) in enumerate(zip(corners, labels)):
                    print(f"  {label:15s}: ({corner[0]:7.2f}, {corner[1]:7.2f})")
                
                # Calculate dimensions
                width = np.linalg.norm(corners[0] - corners[1])
                height = np.linalg.norm(corners[1] - corners[2])
                print(f"\nDimensions:")
                print(f"  Width:  {width:.2f} pixels")
                print(f"  Height: {height:.2f} pixels")
            else:
                print("  No corners detected in this frame")
            
            screenshot_count += 1
            
        elif key == ord('d'):
            show_debug = not show_debug
            if not show_debug:
                cv2.destroyWindow('Edge Detection (Debug)')
            print(f"\n[DEBUG] Edge view: {'ON' if show_debug else 'OFF'}")
            
        elif key == ord('r'):
            detector.corner_history.clear()
            print("\n[RESET] Smoothing cleared")
            
        elif key == ord('h'):
            print("\n" + "=" * 60)
            print(" HELP")
            print("=" * 60)
            print(" Point the camera at any rectangular object")
            print(" The system will automatically detect and label 4 corners")
            print("\n If detection fails:")
            print("  1. Adjust 'Canny Low' and 'Canny High' trackbars")
            print("  2. Press 'd' to see edge detection view")
            print("  3. Increase 'Min Area' if detecting small noise")
            print("  4. Ensure good lighting and contrast")
            print("=" * 60)
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("\nCleaned up. Goodbye!")


if __name__ == "__main__":
    main()