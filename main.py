import cv2
from predicter import EggPredicter
from conveyer_belt import ConveyerBelt

egg_predicter = EggPredicter(confidence_threshold=0.10)
belt = ConveyerBelt("/dev/ttyUSB0", 9600)
capture = cv2.VideoCapture(index=2)


ret, frame = capture.read()
cv2.imshow('Egg Supplier with Robotic Arm', frame)

belt.move()
while True:
    ret, frame = capture.read()
    if not ret:
        print("Error reading image from camera")
        continue

    try:
        predictions = egg_predicter.predict(frame)
    except Exception as e:
        print(e)
        continue
    if len(predictions) >= 6:
        belt.stop()
        egg_predicter.set_confidence(0.50)

    cv2.imshow('Egg Supplier with Robotic Arm', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        belt.stop()
        break

capture.release()
cv2.destroyAllWindows()
