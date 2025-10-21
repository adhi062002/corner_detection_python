import cv2
import numpy as np
import os

def find_crate_corners_optimized(image_path):
    """
    Detects the four main corners of a crate using optimized image processing.

    Parameters:
        image_path (str): The file path to the image containing the crate.
    """
    # --- 1. Load and Pre-process Image ---
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}. Please check the path.")
        return
        
    output_image = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Increased blur kernel (9, 9) to heavily suppress internal texture and noise
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)

    # --- 2. Thresholding ---
    
    # Use global thresholding (80) with inversion. This is robust for dark object on a light background.
    # Note: If the lighting changes drastically, this value (80) may need tuning.
    threshold_value = 80
    _, thresh = cv2.threshold(blurred, threshold_value, 255, cv2.THRESH_BINARY_INV) 

    # --- 3. Morphological Operations (Crucial for a clean, solid contour) ---
    
    # CLOSING: Fills holes (like the crate's internal slots) to create a solid external shape
    kernel_close = np.ones((5, 5), np.uint8) 
    morphed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_close, iterations=4) 
    
    # OPENING: Removes external noise (floor specs) to isolate the crate
    kernel_open = np.ones((7, 7), np.uint8) 
    morphed = cv2.morphologyEx(morphed, cv2.MORPH_OPEN, kernel_open, iterations=2)
    
    # --- 4. Find Contours ---
    # Retrieve only the external contours
    contours, _ = cv2.findContours(
        morphed,          
        cv2.RETR_EXTERNAL, 
        cv2.CHAIN_APPROX_SIMPLE 
    )

    if not contours:
        print("No contours found after processing.")
        return

    # --- 5. Filter and Approximate ---

    # Get the largest contour (assumed to be the crate)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    crate_contour = contours[0] 

    perimeter = cv2.arcLength(crate_contour, True)
    
    # Epsilon (0.04) is increased to force a simplification to a 4-sided polygon
    epsilon = 0.04 * perimeter 
    approx_corners = cv2.approxPolyDP(crate_contour, epsilon, True)

    # --- 6. Verify and Draw ---

    if len(approx_corners) == 4:
        print("✅ Found crate with 4 corners! Applying sub-pixel refinement...")
        
        # Flatten corners array
        corner_points = approx_corners.reshape(-1, 2)
        
        # --- Sub-Pixel Accuracy (Optional but recommended for 'high accuracy') ---
        # Refine corners on the blurred grayscale image
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        float_corners = np.float32(approx_corners)
        refined_corners = cv2.cornerSubPix(
            gray, 
            float_corners, 
            (11, 11), 
            (-1, -1), 
            criteria
        )

        # Draw the detected contour (Green)
        cv2.drawContours(output_image, [approx_corners], -1, (0, 255, 0), 3)
        
        # Draw the rough corners (Red)
        for (x, y) in corner_points:
             cv2.circle(output_image, (x, y), 8, (0, 0, 255), 2)
             
        # Draw the final, refined corners (Blue, filled circle)
        for (x, y) in refined_corners.reshape(-1, 2):
            # Print the final corner coordinates
            print(f"   Refined Corner: ({int(x)}, {int(y)})")
            cv2.circle(output_image, (int(x), int(y)), 8, (255, 0, 0), -1)
            
    else:
        print(f"⚠️ Found shape with {len(approx_corners)} corners, not 4. Check tuning parameters (epsilon/threshold).")
        # Draw the best contour found (Yellow) for debugging
        cv2.drawContours(output_image, [crate_contour], -1, (0, 255, 255), 3)


    # --- 7. Display Results ---
    
    # Resize output for display if the image is too large
    h, w = output_image.shape[:2]
    if h > 1000:
        # Calculate new dimensions to keep aspect ratio
        new_w = 800
        new_h = int(h * (new_w / w))
        output_image = cv2.resize(output_image, (new_w, new_h))
        morphed_display = cv2.resize(morphed, (new_w, new_h))
    else:
        morphed_display = morphed
        
    cv2.imshow("1. Optimized Corner Detection", output_image)
    cv2.imshow("2. Final Cleaned Mask (Contour Source)", morphed_display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ==============================================================================
# === EXECUTION BLOCK ===
# ==============================================================================

# IMPORTANT:
# 1. Place your image file (e.g., 'image_65c82c.jpg') in the same directory as this script.
# 2. Update the variable below with your actual image file name.
IMAGE_FILENAME = r"D:\Work\corner_detection\crate.jpg" # <--- **CHANGE THIS**

# Construct the full path
current_dir = os.path.dirname(os.path.abspath(__file__))
image_full_path = os.path.join(current_dir, IMAGE_FILENAME)

if os.path.exists(image_full_path):
    print(f"Attempting to process image: {image_full_path}")
    find_crate_corners_optimized(image_full_path)
else:
    print(f"🚨 File not found at: {image_full_path}")
    print("Please ensure the image file name is correct and in the same directory.")