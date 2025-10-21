import cv2
import numpy as np

# Load image
img = cv2.imread("crate.jpg")
if img is None:
    raise FileNotFoundError("crate.jpg not found in directory")

# Resize for easier visualization (optional)
img = cv2.resize(img, (800, 600))

# --- Preprocessing ---
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# Adaptive threshold for better segmentation under varying light
thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 15, 3)

# Morphological operations to clean noise
kernel = np.ones((5, 5), np.uint8)
morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

# Edge detection
edges = cv2.Canny(morph, 50, 150)

# Combine edges with threshold for better contour definition
combined = cv2.bitwise_or(edges, morph)

# --- Contour detection ---
contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
if not contours:
    raise RuntimeError("No contours detected. Try adjusting threshold parameters.")

# Find the largest contour (likely the crate)
contour = max(contours, key=cv2.contourArea)
peri = cv2.arcLength(contour, True)
approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

# Draw results
output = img.copy()
cv2.drawContours(output, [contour], -1, (0, 255, 0), 2)

# Check if we have 4 corners (quadrilateral)
if len(approx) == 4:
    for i, pt in enumerate(approx):
        x, y = pt[0]
        cv2.circle(output, (x, y), 10, (0, 0, 255), -1)
        cv2.putText(output, f"P{i+1}", (x + 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
else:
    print(f"Detected {len(approx)} corners — not exactly 4. Try tuning parameters.")

# --- Visualization ---
cv2.imshow("Original", img)
cv2.imshow("Edges + Mask", combined)
cv2.imshow("Detected Corners", output)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Optionally save result
cv2.imwrite("crate_corners_detected.jpg", output)
print("Result saved as crate_corners_detected.jpg")
