#import cv2

# video_cap = cv2.VideoCapture(0)
# while True:    # Capture frame-by-frame
#     ret, video_data = video_cap.read()

#     # Display the resulting frame
#     cv2.imshow("Video_live", video_data)

#     # Hit 'q' on the keyboard to quit!
#     if cv2.waitKey(1) & 0xFF == ord("a"):
#         break
# # Release handle to the webcam
# video_cap.release()

import cv2

# Load cascade
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

if face_cascade.empty():
    print("Error loading cascade")
    exit()

# Start camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)

    cv2.imshow("Face Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("a"):
         break
cap.release()
cv2.destroyAllWindows()