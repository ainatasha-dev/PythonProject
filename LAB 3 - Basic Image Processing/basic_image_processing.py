
# 2.	TASK 1: Load and Display Image
import cv2
import matplotlib.pyplot as plt
image = cv2.imread("sample_2.jpg")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(image_rgb)
plt.title("Original Image")
plt.axis("off")
plt.show()

# 3.	TASK 2: Check Image Size and Pixel Value
print("Image shape:", image.shape)
pixel = image[50, 50]
print("Pixel value at (50,50):", pixel)


# 4.	TASK 3: Convert to Grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
plt.imshow(gray, cmap="gray")
plt.title("Grayscale Image")
plt.axis("off")
plt.show()
print("Grayscale shape:", gray.shape)

# 5.	TASK 4: Brightness Adjustment

bright = cv2.convertScaleAbs(image, alpha=1, beta=50)
bright_rgb = cv2.cvtColor(bright, cv2.COLOR_BGR2RGB)
plt.imshow(bright_rgb)
plt.title("Brighter Image")
plt.axis("off")
plt.show()

# 6.	TASK 5: Contrast Adjustment
contrast = cv2.convertScaleAbs(image, alpha=1.8, beta=0)
contrast_rgb = cv2.cvtColor(contrast, cv2.COLOR_BGR2RGB)
plt.imshow(contrast_rgb)
plt.title("High Contrast Image")
plt.axis("off")
plt.show()

# 7.	TASK 6: Thresholding

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
plt.imshow(threshold, cmap="gray")
plt.title("Threshold Image")
plt.axis("off")
plt.show()

# 8.	TASK 7: Filtering

blur = cv2.GaussianBlur(image, (7, 7), 0)
blur_rgb = cv2.cvtColor(blur, cv2.COLOR_BGR2RGB)
plt.imshow(blur_rgb)
plt.title("Blurred Image")
plt.axis("off")
plt.show()
