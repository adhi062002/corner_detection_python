"""
Crate Corner Detection and 2D Localization using Web Camera
Author: Adithya Satheesh
Date: October 19, 2025
Description: Detects and identifies the top surface corner points of a crate
             using computer vision with a standard webcam.
"""

import cv2
import numpy as np
from collections import deque
import time


class CrateCornerDetector:
    def __init__(self, smoothing_window=5):
        """
        Initialize the crate corner detector.
        
        Args:
            smoothing_window (int): Number of frames for temporal smoothing
        """
        self.smoothing_window = smoothing_window
        self.corner_history = deque(maxlen=smoothing_window)
        self.fps_history = deque(maxlen=30)
        
    def preprocess_frame(self, frame):
        """
        Preprocess the frame for better corner detection.
        
        Args:
            frame: Input BGR frame
            
        Returns:
            Preprocessed binary image
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply adaptive thresholding for better edge detection
        binary = cv2.adaptiveThreshold(
            blurred, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 
            11, 2
        )
        
        return binary
    
    def detect_crate_contour(self, binary_image):
        """
        Detect the crate contour from binary image.
        
        Args:
            binary_image: Binary preprocessed image
            
        Returns:
            Largest rectangular contour or None
        """
        # Find contours
        contours, _ = cv2.findContours(
            binary_image, 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            return None
        
        # Filter contours by area and find the largest one
        valid_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            # Filter small contours (adjust threshold as needed)
            if area > 1000:
                # Approximate contour to polygon
                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
                
                # Look for quadrilaterals (4 corners)
                if len(approx) == 4:
                    valid_contours.append((area, approx))
        
        if not valid_contours:
            return None
        
        # Return the largest valid contour
        valid_contours.sort(key=lambda x: x[0], reverse=True)
        return valid_contours[0][1]
    
    def order_corners(self, corners):
        """
        Order corners in consistent manner: top-left, top-right, bottom-right, bottom-left.
        
        Args:
            corners: Array of 4 corner points
            
        Returns:
            Ordered array of corners
        """
        # Reshape to (4, 2)
        pts = corners.reshape(4, 2)
        
        # Initialize ordered array
        rect = np.zeros((4, 2), dtype=np.float32)
        
        # Sum and difference method
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)
        
        # Top-left has smallest sum
        rect[0] = pts[np.argmin(s)]
        # Bottom-right has largest sum
        rect[2] = pts[np.argmax(s)]
        # Top-right has smallest difference
        rect[1] = pts[np.argmin(diff)]
        # Bottom-left has largest difference
        rect[3] = pts[np.argmax(diff)]
        
        return rect
    
    def smooth_corners(self, corners):
        """
        Apply temporal smoothing to reduce jitter.
        
        Args:
            corners: Current frame corners
            
        Returns:
            Smoothed corners
        """
        self.corner_history.append(corners)
        
        if len(self.corner_history) < 2:
            return corners
        
        # Average corners over history
        smoothed = np.mean(self.corner_history, axis=0)
        return smoothed.astype(np.float32)
    
    def draw_corners(self, frame, corners):
        """
        Draw corner points and labels on the frame.
        
        Args:
            frame: Input frame
            corners: Array of 4 corner points
            
        Returns:
            Frame with drawn corners
        """
        output = frame.copy()
        
        # Define colors and labels
        colors = [
            (0, 255, 0),    # Green - Top Left
            (255, 0, 0),    # Blue - Top Right
            (0, 0, 255),    # Red - Bottom Right
            (255, 255, 0)   # Cyan - Bottom Left
        ]
        labels = ['TL', 'TR', 'BR', 'BL']
        
        # Draw corners
        for i, (corner, color, label) in enumerate(zip(corners, colors, labels)):
            x, y = int(corner[0]), int(corner[1])
            
            # Draw circle at corner
            cv2.circle(output, (x, y), 8, color, -1)
            cv2.circle(output, (x, y), 10, (255, 255, 255), 2)
            
            # Draw label
            cv2.putText(
                output, 
                f'{label}', 
                (x + 15, y - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, 
                color, 
                2
            )
            
            # Draw coordinates
            cv2.putText(
                output, 
                f'({x}, {y})', 
                (x + 15, y + 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.4, 
                (255, 255, 255), 
                1
            )
        
        # Draw connecting lines
        for i in range(4):
            pt1 = tuple(corners[i].astype(int))
            pt2 = tuple(corners[(i + 1) % 4].astype(int))
            cv2.line(output, pt1, pt2, (0, 255, 255), 2)
        
        return output
    
    def add_info_overlay(self, frame, corners, fps):
        """
        Add information overlay to the frame.
        
        Args:
            frame: Input frame
            corners: Detected corners
            fps: Current FPS
            
        Returns:
            Frame with overlay
        """
        output = frame.copy()
        h, w = output.shape[:2]
        
        # Create semi-transparent overlay
        overlay = output.copy()
        cv2.rectangle(overlay, (10, 10), (300, 100), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, output, 0.4, 0, output)
        
        # Add text information
        cv2.putText(
            output, 
            'Crate Corner Detection', 
            (20, 35), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.6, 
            (0, 255, 0), 
            2
        )
        cv2.putText(
            output, 
            f'FPS: {fps:.1f}', 
            (20, 60), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.5, 
            (255, 255, 255), 
            1
        )
        cv2.putText(
            output, 
            f'Corners: {len(corners) if corners is not None else 0}/4', 
            (20, 85), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.5, 
            (255, 255, 255), 
            1
        )
        
        return output
    
    def process_frame(self, frame):
        """
        Main processing pipeline for corner detection.
        
        Args:
            frame: Input BGR frame
            
        Returns:
            Processed frame with detected corners
        """
        start_time = time.time()
        
        # Preprocess frame
        binary = self.preprocess_frame(frame)
        
        # Detect crate contour
        contour = self.detect_crate_contour(binary)
        
        if contour is not None:
            # Order corners
            ordered_corners = self.order_corners(contour)
            
            # Apply temporal smoothing
            smoothed_corners = self.smooth_corners(ordered_corners)
            
            # Draw corners on frame
            output = self.draw_corners(frame, smoothed_corners)
        else:
            output = frame.copy()
            cv2.putText(
                output, 
                'No crate detected', 
                (20, frame.shape[0] - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                (0, 0, 255), 
                2
            )
            smoothed_corners = None
        
        # Calculate FPS
        fps = 1.0 / (time.time() - start_time)
        self.fps_history.append(fps)
        avg_fps = np.mean(self.fps_history)
        
        # Add info overlay
        output = self.add_info_overlay(output, smoothed_corners, avg_fps)
        
        return output, smoothed_corners


def main():
    """Main function to run the corner detection system."""
    
    # Initialize webcam
    cap = cv2.VideoCapture(1)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    # Set camera properties for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # Initialize detector
    detector = CrateCornerDetector(smoothing_window=5)
    
    print("=" * 50)
    print("Crate Corner Detection System")
    print("=" * 50)
    print("Controls:")
    print("  'q' - Quit")
    print("  's' - Save screenshot")
    print("  'r' - Reset smoothing")
    print("=" * 50)
    
    screenshot_count = 0
    
    while True:
        # Capture frame
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Failed to capture frame")
            break
        
        # Process frame
        output, corners = detector.process_frame(frame)
        
        # Display result
        cv2.imshow('Crate Corner Detection', output)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('s'):
            filename = f'corner_detection_{screenshot_count:03d}.png'
            cv2.imwrite(filename, output)
            print(f"Screenshot saved: {filename}")
            if corners is not None:
                print(f"Corner coordinates:")
                for i, corner in enumerate(corners):
                    print(f"  Corner {i}: ({corner[0]:.2f}, {corner[1]:.2f})")
            screenshot_count += 1
        elif key == ord('r'):
            detector.corner_history.clear()
            print("Smoothing reset")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("\nSystem terminated successfully")


if __name__ == "__main__":
    main()