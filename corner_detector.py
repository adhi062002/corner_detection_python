"""
Stable Corner Detector with Depth Camera Support
Author: Adithya Satheesh (Modified by AI)
Date: July 10, 2026
Description:
    A script for stable corner detection of rectangular objects using RGB + Depth data.
    Features:
    - 2D corner detection (RGB)
    - 3D corner localization (Depth)
    - Depth-based filtering
    - Pose estimation (PnP)
    - 3D visualization
"""

import cv2
import numpy as np
import json
import logging
import time
from collections import deque
from typing import Optional, Tuple, List, Dict, Any

# --- PyRealSense2 Imports ---
import pyrealsense2 as rs

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class DepthCornerDetector:
    def __init__(self, config_path: str = "config.json"):
        """
        Initialize the corner detector with RGB + Depth support.
        """
        self.load_config(config_path)
        self.corner_history = deque(maxlen=self.smoothing_factor)
        self.depth_corner_history = deque(maxlen=self.depth_smoothing_factor)
        self.fps_history = deque(maxlen=30)
        self.pipeline = None
        self.config = rs.config()
        self._init_realsense()
        logger.info("DepthCornerDetector initialized with config: %s", config_path)

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
            self.depth_smoothing_factor = stability["depth_smoothing_factor"]

            # Depth parameters
            depth = config["depth"]
            self.min_depth = depth["min_depth"]
            self.max_depth = depth["max_depth"]
            self.depth_consistency_threshold = depth["depth_consistency_threshold"]
            self.use_depth_segmentation = depth["use_depth_segmentation"]
            self.depth_segmentation_threshold = depth["depth_segmentation_threshold"]

            # Camera parameters
            camera = config["camera"]
            self.rgb_device_id = camera["rgb_device_id"]
            self.depth_device_id = camera["depth_device_id"]
            self.width = camera["width"]
            self.height = camera["height"]
            self.fps = camera["fps"]

            # Debug parameters
            debug = config["debug"]
            self.show_debug = debug["show_debug"]
            self.show_3d = debug["show_3d"]
            self.save_results = debug["save_results"]

            logger.info("Config loaded successfully.")
        except Exception as e:
            logger.error("Failed to load config: %s. Using defaults.", e)
            self._set_defaults()

    def _set_defaults(self) -> None:
        """Set default parameters if config loading fails."""
        # Core detection
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

        # Stability
        self.smoothing_factor = 20
        self.subpixel_refinement = True
        self.depth_smoothing_factor = 10

        # Depth
        self.min_depth = 0.1
        self.max_depth = 5.0
        self.depth_consistency_threshold = 0.05
        self.use_depth_segmentation = True
        self.depth_segmentation_threshold = 0.5

        # Camera
        self.rgb_device_id = 0
        self.depth_device_id = 0
        self.width = 640
        self.height = 480
        self.fps = 30

        # Debug
        self.show_debug = False
        self.show_3d = False
        self.save_results = False

    def _init_realsense(self) -> None:
        """Initialize the RealSense pipeline."""
        try:
            self.pipeline = rs.pipeline()
            self.config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
            self.config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
            self.config.enable_record_to_file("output.bag")  # Optional: Record data
            self.pipeline.start(self.config)

            # Get camera intrinsics for RGB and Depth
            self.rgb_intrinsics = self._get_intrinsics(rs.stream.color)
            self.depth_intrinsics = self._get_intrinsics(rs.stream.depth)
            self.depth_to_rgb_extrinsics = self._get_extrinsics(rs.stream.depth, rs.stream.color)
            self.rgb_to_depth_extrinsics = self._get_extrinsics(rs.stream.color, rs.stream.depth)

            logger.info("RealSense pipeline initialized.")
        except Exception as e:
            logger.error("Failed to initialize RealSense: %s", e)
            raise

    def _get_intrinsics(self, stream: rs.stream) -> rs.intrinsics:
        """Get camera intrinsics for a given stream."""
        profile = self.pipeline.get_active_profile()
        stream_profile = profile.get_stream(stream)
        intrinsics = stream_profile.as_video_stream_profile().get_intrinsics()
        return intrinsics

    def _get_extrinsics(self, from_stream: rs.stream, to_stream: rs.stream) -> rs.extrinsics:
        """Get extrinsics (rotation/translation) between two streams."""
        profile = self.pipeline.get_active_profile()
        from_profile = profile.get_stream(from_stream)
        to_profile = profile.get_stream(to_stream)
        extrinsics = from_profile.get_extrinsics_to(to_profile)
        return extrinsics

    def get_frames(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Capture synchronized RGB + Depth frames.
        Returns:
            - RGB frame (BGR format)
            - Depth frame (16-bit, meters)
        """
        try:
            frames = self.pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()

            if not color_frame or not depth_frame:
                return None, None

            # Convert to numpy arrays
            rgb_frame = np.asanyarray(color_frame.get_data())
            depth_frame = np.asanyarray(depth_frame.get_data()).astype(np.float32)
            depth_frame = depth_frame / 1000.0  # Convert mm to meters

            return rgb_frame, depth_frame
        except Exception as e:
            logger.error("Failed to get frames: %s", e)
            return None, None

    def segment_image(self, frame: np.ndarray, depth_frame: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Segment the image using adaptive thresholding and morphology.
        Optionally uses depth for additional segmentation.
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

        # Depth-based segmentation (optional)
        if depth_frame is not None and self.use_depth_segmentation:
            depth_mask = np.logical_and(
                depth_frame >= self.depth_segmentation_threshold - 0.1,
                depth_frame <= self.depth_segmentation_threshold + 0.1
            ).astype(np.uint8) * 255
            segmented_mask = cv2.bitwise_and(segmented_mask, depth_mask)

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

    def get_3d_corners(
        self,
        corners_2d: np.ndarray,
        depth_frame: np.ndarray,
        rgb_intrinsics: rs.intrinsics,
        depth_intrinsics: rs.intrinsics,
        depth_to_rgb_extrinsics: rs.extrinsics
    ) -> np.ndarray:
        """
        Convert 2D corners to 3D world coordinates using depth.
        Args:
            corners_2d: (4, 2) array of 2D corner coordinates (x, y)
            depth_frame: Depth frame (meters)
            rgb_intrinsics: RGB camera intrinsics
            depth_intrinsics: Depth camera intrinsics
            depth_to_rgb_extrinsics: Extrinsics from depth to RGB
        Returns:
            (4, 3) array of 3D corners (X, Y, Z) in meters
        """
        corners_3d = np.zeros((4, 3), dtype=np.float32)

        # Get depth sensor intrinsics
        depth_intr = depth_intrinsics
        rgb_intr = rgb_intrinsics

        # Get extrinsics (depth to RGB)
        extr = depth_to_rgb_extrinsics
        rotation = np.array(extr.rotation).reshape(3, 3)
        translation = np.array(extr.translation)

        for i, (x, y) in enumerate(corners_2d):
            # Get depth at (x, y) in RGB frame
            # First, undistort the RGB pixel
            rgb_pixel = rs.rs2_deproject_pixel_to_point(
                rgb_intr, [x, y], 1.0  # We assume depth=1 for deprojection (we'll scale later)
            )
            rgb_pixel = np.array(rgb_pixel)[:2]  # Only need (x, y)

            # Map RGB pixel to depth frame
            depth_pixel = rs.rs2_transform_point_to_point(
                extr, rgb_intr, rgb_pixel
            )
            depth_x, depth_y = int(depth_pixel[0]), int(depth_pixel[1])

            # Check if depth pixel is within bounds
            if 0 <= depth_x < depth_frame.shape[1] and 0 <= depth_y < depth_frame.shape[0]:
                depth = depth_frame[depth_y, depth_x]
                if self.min_depth <= depth <= self.max_depth:
                    # Deproject depth pixel to 3D
                    point_3d = rs.rs2_deproject_pixel_to_point(
                        depth_intr, [depth_x, depth_y], depth
                    )
                    corners_3d[i] = point_3d
                else:
                    logger.warning(f"Corner {i} has invalid depth: {depth}")
                    corners_3d[i] = [0, 0, 0]
            else:
                logger.warning(f"Corner {i} maps outside depth frame: ({depth_x}, {depth_y})")
                corners_3d[i] = [0, 0, 0]

        return corners_3d

    def smooth_3d_corners(self, corners_3d: np.ndarray) -> np.ndarray:
        """Apply temporal smoothing to 3D corner positions."""
        self.depth_corner_history.append(corners_3d)
        smoothed = np.mean(self.depth_corner_history, axis=0)
        return smoothed.astype(np.float32)

    def estimate_pose(self, corners_2d: np.ndarray, corners_3d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate the 6DOF pose (rotation and translation) of the rectangle.
        Uses PnP (Perspective-n-Point) with known 3D corners.
        Assumes the rectangle is axis-aligned in 3D space (e.g., lying flat on a table).
        """
        # Define the 3D model of the rectangle (assume it's 1x1 meters for simplicity)
        # You can replace this with the actual dimensions if known
        model_3d = np.array([
            [0, 0, 0],      # Top-Left
            [1, 0, 0],      # Top-Right
            [1, 1, 0],      # Bottom-Right
            [0, 1, 0]       # Bottom-Left
        ], dtype=np.float32)

        # Use the first 3 corners for PnP (4th is redundant)
        image_points = corners_2d[:3].reshape(3, 1, 2)
        object_points = model_3d[:3].reshape(3, 1, 3)

        # Camera matrix (from RGB intrinsics)
        camera_matrix = np.array([
            [self.rgb_intrinsics.fx, 0, self.rgb_intrinsics.ppx],
            [0, self.rgb_intrinsics.fy, self.rgb_intrinsics.ppy],
            [0, 0, 1]
        ], dtype=np.float32)

        # Distortion coefficients (assume zero for simplicity)
        dist_coeffs = np.zeros((5, 1), dtype=np.float32)

        # Solve PnP
        success, rvec, tvec = cv2.solvePnP(
            object_points, image_points, camera_matrix, dist_coeffs
        )

        if success:
            return rvec, tvec
        else:
            logger.warning("PnP failed. Returning zeros.")
            return np.zeros(3), np.zeros(3)

    def draw_corners(self, frame: np.ndarray, corners: np.ndarray, corners_3d: Optional[np.ndarray] = None) -> np.ndarray:
        """Draw the 4 corners with labels, coordinates, and 3D info on the frame."""
        output = frame.copy()
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0)]  # TL, TR, BR, BL
        labels = ["TL", "TR", "BR", "BL"]

        for i, (corner, color, label) in enumerate(zip(corners, colors, labels)):
            x, y = int(corner[0]), int(corner[1])

            # Draw corner circle and outline
            cv2.circle(output, (x, y), 8, color, -1)
            cv2.circle(output, (x, y), 10, (255, 255, 255), 2)

            # Draw label and 2D coordinates
            label_text = f"{label}"
            coord_text = f"({x},{y})"
            text_pos_x, text_pos_y = x + 15, y + 5
            cv2.putText(output, label_text, (text_pos_x, text_pos_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(output, coord_text, (text_pos_x, text_pos_y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            # Draw 3D coordinates if available
            if corners_3d is not None:
                x_3d, y_3d, z_3d = corners_3d[i]
                coord_3d_text = f"3D:({x_3d:.2f},{y_3d:.2f},{z_3d:.2f})"
                cv2.putText(output, coord_3d_text, (text_pos_x, text_pos_y + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

        # Draw connecting lines
        for i in range(4):
            pt1 = tuple(corners[i].astype(int))
            pt2 = tuple(corners[(i + 1) % 4].astype(int))
            cv2.line(output, pt1, pt2, (0, 255, 255), 2)

        return output

    def add_info_overlay(self, frame: np.ndarray, has_corners: bool, fps: float, pose: Optional[Tuple] = None) -> np.ndarray:
        """Draw an info overlay on the top-left of the frame."""
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (350, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        status = "DETECTED" if has_corners else "SEARCHING..."
        status_color = (0, 255, 0) if has_corners else (0, 0, 255)

        cv2.putText(frame, "Stable Corner Detector (RGB+Depth)", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, status, (200, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

        if pose is not None:
            rvec, tvec = pose
            pose_text = f"Pose: T=({tvec[0]:.2f},{tvec[1]:.2f},{tvec[2]:.2f})"
            cv2.putText(frame, pose_text, (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        return frame

    def process_frame(self, show_debug: bool = False, show_3d: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Main processing pipeline for a single RGB + Depth frame.
        Returns:
            - Output RGB frame with corners drawn
            - Debug frame (if show_debug=True)
            - 3D point cloud (if show_3d=True)
        """
        # Get frames
        rgb_frame, depth_frame = self.get_frames()
        if rgb_frame is None or depth_frame is None:
            logger.error("No frames captured.")
            return None, None, None

        start_time = time.time()
        gray = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2GRAY)

        # Segment and find rectangles
        segmented_mask = self.segment_image(rgb_frame, depth_frame)
        rectangles = self.find_rectangles(segmented_mask, rgb_frame.shape)

        output = rgb_frame.copy()
        corners_2d = None
        corners_3d = None
        pose = None

        if rectangles:
            best_rectangle = rectangles[0]
            ordered_corners = self.order_corners(best_rectangle["corners"])
            ordered_corners = self.refine_corners(ordered_corners, gray)
            corners_2d = self.smooth_corners(ordered_corners)

            # Get 3D corners
            corners_3d = self.get_3d_corners(
                corners_2d,
                depth_frame,
                self.rgb_intrinsics,
                self.depth_intrinsics,
                self.depth_to_rgb_extrinsics
            )
            corners_3d = self.smooth_3d_corners(corners_3d)

            # Estimate pose
            pose = self.estimate_pose(corners_2d, corners_3d)

            # Draw corners
            output = self.draw_corners(output, corners_2d, corners_3d)

            if self.save_results:
                self._save_corners(corners_2d, corners_3d)
        else:
            if self.corner_history:
                self.corner_history.popleft()
            if self.depth_corner_history:
                self.depth_corner_history.popleft()

        # Calculate FPS
        elapsed_time = time.time() - start_time
        fps = 1.0 / elapsed_time if elapsed_time > 0 else 999.0
        self.fps_history.append(fps)
        avg_fps = np.mean(self.fps_history)
        output = self.add_info_overlay(output, corners_2d is not None, avg_fps, pose)

        # Debug visualization
        debug = None
        if show_debug:
            debug = cv2.cvtColor(segmented_mask, cv2.COLOR_GRAY2BGR)
            for i, rect in enumerate(rectangles[:5]):
                color = (0, 255, 0) if i == 0 else (100, 100, 255)
                cv2.drawContours(debug, [rect["corners"]], -1, color, 2)

        # 3D visualization
        point_cloud = None
        if show_3d and corners_3d is not None:
            point_cloud = self._create_point_cloud(depth_frame, rgb_frame)

        return output, debug, point_cloud

    def _save_corners(self, corners_2d: np.ndarray, corners_3d: np.ndarray) -> None:
        """Save 2D and 3D corner coordinates to a CSV file."""
        try:
            with open("corners_3d.csv", "a") as f:
                flat_2d = corners_2d.flatten()
                flat_3d = corners_3d.flatten()
                f.write(f"{time.time()},{','.join(map(str, flat_2d))},{','.join(map(str, flat_3d))}\n")
        except Exception as e:
            logger.error("Failed to save corners: %s", e)

    def _create_point_cloud(self, depth_frame: np.ndarray, rgb_frame: np.ndarray) -> np.ndarray:
        """
        Create a colored point cloud from depth and RGB frames.
        Returns:
            (N, 3) array of 3D points (X, Y, Z) with colors (R, G, B)
        """
        # Get depth intrinsics
        depth_intr = self.depth_intrinsics

        # Create point cloud
        points = []
        colors = []
        for y in range(depth_frame.shape[0]):
            for x in range(depth_frame.shape[1]):
                depth = depth_frame[y, x]
                if self.min_depth <= depth <= self.max_depth:
                    point = rs.rs2_deproject_pixel_to_point(depth_intr, [x, y], depth)
                    points.append(point)
                    colors.append(rgb_frame[y, x])

        points = np.array(points, dtype=np.float32)
        colors = np.array(colors, dtype=np.uint8)
        return points, colors

    def visualize_3d(self, point_cloud: Tuple[np.ndarray, np.ndarray]) -> None:
        """
        Visualize the 3D point cloud using Open3D.
        Requires: pip install open3d
        """
        try:
            import open3d as o3d

            points, colors = point_cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)

            o3d.visualization.draw_geometries([pcd])
        except ImportError:
            logger.error("Open3D not installed. Cannot visualize 3D point cloud.")
        except Exception as e:
            logger.error("Failed to visualize 3D point cloud: %s", e)

def main():
    # Initialize detector
    detector = DepthCornerDetector()

    # Print instructions
    print("\n" + "=" * 60)
    print(" Stable Corner Detector (RGB + Depth) Running...")
    print("  - Press 'd' to toggle debug view.")
    print("  - Press '3' to toggle 3D visualization.")
    print("  - Press 's' to start/stop saving corner coordinates.")
    print("  - Press 'q' to quit.")
    print("=" * 60)

    show_debug = detector.show_debug
    show_3d = detector.show_3d
    debug_window_name = "Debug View"
    point_cloud_window_name = "3D Point Cloud"

    try:
        while True:
            output, debug, point_cloud = detector.process_frame(show_debug, show_3d)

            if output is not None:
                cv2.imshow("Corner Detection (RGB + Depth)", output)

            if show_debug and debug is not None:
                cv2.imshow(debug_window_name, debug)

            if show_3d and point_cloud is not None:
                detector.visualize_3d(point_cloud)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("d"):
                show_debug = not show_debug
                if not show_debug:
                    cv2.destroyWindow(debug_window_name)
            elif key == ord("3"):
                show_3d = not show_3d
            elif key == ord("s"):
                detector.save_results = not detector.save_results
                logger.info("Saving corners: %s", detector.save_results)

    finally:
        detector.pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
