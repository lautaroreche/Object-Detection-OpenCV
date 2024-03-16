import cv2


# Loads image variables and indicates the name of the popup screen
image_path = "image.png"
title = f"Detected Objects in {image_path}"
original_image = cv2.imread(image_path)

# Transforms the image to grayscale to improve the match
gray_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)

# Load the Haar Cascade classifier that has positive and negative images of the object to be detected
haar_cascade_classifier_eye = cv2.CascadeClassifier(f"{cv2.data.haarcascades}haarcascade_eye.xml")
haar_cascade_classifier_face = cv2.CascadeClassifier(f"{cv2.data.haarcascades}haarcascade_frontalface_default.xml")

# Generates a list that contains within a list for each element detected according to the Haar cascade classifier
# Each list has 4 parameters: position in x, position in y, width and height of the detected element.
detected_eye = haar_cascade_classifier_eye.detectMultiScale(gray_image, minSize = (55, 55), maxSize = (150, 150))
detected_face = haar_cascade_classifier_face.detectMultiScale(gray_image, minSize = (300, 300), maxSize = (450, 450))

# Draw a rectangle on each detected element
if (len(detected_eye) > 0) or (len(detected_face) > 0):
    for variable in detected_eye, detected_face:
        for (x, y, width, height) in variable:
            # Rectangle parameters: image, start coordinates, end coordinates, color, thickness
            cv2.rectangle(original_image, (x, y), (x + height, y + width), (0, 255, 0), 1)

# Create a screen with the given title and automatic size
cv2.namedWindow(title, cv2.WINDOW_NORMAL)
# Show the image
cv2.imshow(title, original_image)
# Change the size of the popup window
cv2.resizeWindow(title, 600, 500)
# Wait until any key is pressed
cv2.waitKey(0)
cv2.destroyAllWindows()
