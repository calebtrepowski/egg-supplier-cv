import cv2
from camera import capture
# capture = cv2.VideoCapture(index=2)
while True:
    ret, frame = capture.read()
    if not ret:
        print("Error reading image from camera")
        continue

    cv2.imshow('Show camera', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
