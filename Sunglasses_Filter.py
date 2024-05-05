import cv2

# Load the Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the sunglasses image
sunglasses_img = cv2.imread('sunglasses.png', cv2.IMREAD_UNCHANGED)

def put_sunglasses_on_face(frame, face):
    # Extract the dimensions of the face region
    x, y, w, h = face

    # Resize the sunglasses to fit the face
    sunglasses_resized = cv2.resize(sunglasses_img, (w, h))

    # Ensure the sunglasses image has an alpha channel
    if sunglasses_resized.shape[2] == 3:
        sunglasses_resized = cv2.cvtColor(sunglasses_resized, cv2.COLOR_BGR2BGRA)

    # Calculate the coordinates for placing the sunglasses
    x1 = x
    y1 = y + int(h / 8)
    x2 = x1 + w
    y2 = y1 + int(h / 2)

    # Add the sunglasses to the frame with alpha blending
    alpha_s = sunglasses_resized[:, :, 3] / 255.0
    alpha_l = 1.0 - alpha_s

    for c in range(0, 3):
        sunglasses_region = sunglasses_resized[:, :, c][:frame[y1:y2, x1:x2, c].shape[0], :frame[y1:y2, x1:x2, c].shape[1]]
        frame_region = frame[y1:y2, x1:x2, c]
        frame[y1:y1 + sunglasses_region.shape[0], x1:x1 + sunglasses_region.shape[1], c] = (
                alpha_s[:frame_region.shape[0], :frame_region.shape[1]] * sunglasses_region +
                alpha_l[:frame_region.shape[0], :frame_region.shape[1]] * frame_region)

def detect_faces(frame):
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Iterate over each detected face
    for (x, y, w, h) in faces:
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Put sunglasses on the face
        put_sunglasses_on_face(frame, (x, y, w, h))

    return frame

# Initialize the webcam
cap = cv2.VideoCapture(0)
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Detect faces and put sunglasses on them
    frame_with_sunglasses = detect_faces(frame)

    # Display the resulting frame
    cv2.imshow('Sunglasses Filter', frame_with_sunglasses)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()