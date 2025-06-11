import cv2
import random
import os

print("Starting the script...")  # Added for confirmation

# Check the current working directory
print("📁 Current Folder:", os.getcwd())  # Display the current folder

# Load Haar cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Try loading the image
image_path = "test.jpg"
image = cv2.imread(image_path)

if image is None:
    print(f"❌ Image '{image_path}' not loaded! Check if the image exists in the folder.")
    exit()

print("✅ Image loaded successfully!")  # Image loaded successfully

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces
faces = face_cascade.detectMultiScale(gray, 1.1, 4)

# Print number of faces found
print(f"\n✅ Detected {len(faces)} face(s) in '{image_path}':")

for i, (x, y, w, h) in enumerate(faces, 1):
    label = random.choice(["Mask", "No Mask"])
    print(f" - Face {i}: {label}")

if len(faces) == 0:
    print("😐 No faces found. Try another image.")
