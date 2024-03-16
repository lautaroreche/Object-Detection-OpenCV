import cv2


# Loads image variables and indicates the name of the popup screen
title = "Detected Objects in webcam"
webcam = cv2.VideoCapture(0)
keep_open = True

while webcam.isOpened() and keep_open:
    # Returns a boolean if the video read is successfull, and quits if it is not
    auxiliar, frame = webcam.read()
    if not auxiliar:
        print("No frame can be read from the webcam")
        break

    # Create a screen with the given title and automatic size 
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)

    # Transforms the image to grayscale to improve the match
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    # Load the Haar Cascade classifier that has positive and negative images of the object to be detected
    haar_cascade_classifier_eye = cv2.CascadeClassifier(f"{cv2.data.haarcascades}haarcascade_eye.xml")
    haar_cascade_classifier_face = cv2.CascadeClassifier(f"{cv2.data.haarcascades}haarcascade_frontalface_default.xml")
    
    # Generates a list that contains within a list for each element detected according to the Haar cascade classifier
    # Each list has 4 parameters: position in x, position in y, width and height of the detected element.
    detected_eye = haar_cascade_classifier_eye.detectMultiScale(gray_frame, minSize = (55, 55), maxSize = (150, 150))
    detected_face = haar_cascade_classifier_face.detectMultiScale(gray_frame, minSize = (260, 260), maxSize = (450, 450))
    
    # Draw a rectangle on each detected element
    if (len(detected_eye) > 0) or (len(detected_face) > 0):
        for variable in detected_eye, detected_face:
            for (x, y, width, height) in variable:
                # Rectangle parameters: image, start coordinates, end coordinates, color, thickness
                cv2.rectangle(frame, (x, y), (x + height, y + width), (0, 255, 0), 1)

    # Show the image
    cv2.imshow(title, frame)
    
    # Quit if any key is pressed
    if cv2.waitKey(1) & 0xFF != 255:
        keep_open = False


webcam.release()
# Once any key is pressed the popup screen closes
cv2.destroyAllWindows()
