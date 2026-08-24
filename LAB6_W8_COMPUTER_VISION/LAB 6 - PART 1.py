import cv2

# Load the image
image_path = "mandalorian & grogu.jpg"  # Replace this with your image file path
img = cv2.imread(image_path)

# Check if the image is loaded correctly
if img is None:
    print("Error: Could not load the image. Check the file path.")
    exit()

# Display the original image
cv2.imshow("Original Image", img)
cv2.waitKey(0)

# 1. Resize the image
resized = cv2.resize(img, (200, 200))
cv2.imshow("Resized Image", resized)
cv2.imwrite("resized.jpg", resized)  # Save the resized image
cv2.waitKey(0)

# 2. Rotate the image 90 degrees clockwise
rotated = cv2.rotate(resized, cv2.ROTATE_90_CLOCKWISE)
cv2.imshow("Rotated Image", rotated)
cv2.imwrite("rotated.jpg", rotated)  # Save the rotated image
cv2.waitKey(0)

# 3. Apply Gaussian blur with default settings
blurred = cv2.GaussianBlur(img, (5, 5), 0)
cv2.imshow("Blurred Image (ksize=5x5, sigmaX=0)", blurred)
cv2.imwrite("blurred.jpg", blurred)  # Save the blurred image
cv2.waitKey(0)

# 4. Experiment with different kernel sizes and sigmaX
# a) Small kernel size
blur_small = cv2.GaussianBlur(img, (3, 3), 0)
cv2.imshow("Small Blur (ksize=3x3, sigmaX=0)", blur_small)
cv2.imwrite("blur_small.jpg", blur_small)
cv2.waitKey(0)

# b) Large kernel size
blur_large = cv2.GaussianBlur(img, (9, 9), 0)
cv2.imshow("Large Blur (ksize=9x9, sigmaX=0)", blur_large)
cv2.imwrite("blur_large.jpg", blur_large)
cv2.waitKey(0)

# c) High sigmaX value
blur_high_sigma = cv2.GaussianBlur(img, (5, 5), 10)
cv2.imshow("High Sigma Blur (ksize=5x5, sigmaX=10)", blur_high_sigma)
cv2.imwrite("blur_high_sigma.jpg", blur_high_sigma)
cv2.waitKey(0)

# d) Combination of large kernel and high sigmaX
blur_combined = cv2.GaussianBlur(img, (9, 9), 15)
cv2.imshow("Combined Blur (ksize=9x9, sigmaX=15)", blur_combined)
cv2.imwrite("blur_combined.jpg", blur_combined)
cv2.waitKey(0)

# Close all OpenCV windows
cv2.destroyAllWindows()
