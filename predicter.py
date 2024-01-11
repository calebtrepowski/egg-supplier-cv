from inference_sdk import InferenceHTTPClient, InferenceConfiguration
import requests
from dotenv import dotenv_values
import cv2
from roi import ROI
from prediction import Prediction

config = dotenv_values(".env")
PROJECT_ID = config["ROBOFLOW_PROJECT_ID"]
MODEL_VERSION = config["ROBOFLOW_MODEL_VERSION"]
CONFIDENCE_THRESHOLD = 0.75

client = InferenceHTTPClient(
    api_url="http://localhost:8080",
    api_key=config["ROBOFLOW_API_KEY"]
)
client.configure(InferenceConfiguration(
    confidence_threshold=CONFIDENCE_THRESHOLD))
client.select_api_v0()

eggs_roi = ROI("eggs")
eggs_roi.load()

capture = cv2.VideoCapture(2)

while True:
    ret, frame = capture.read()
    if not ret:
        print("Error reading image from camera")
        exit()

    eggs_roi_frame = eggs_roi.get_frame(frame)
    eggs_roi.draw(frame)

    try:
        results = client.infer(
            eggs_roi_frame,
            model_id=f"{PROJECT_ID}/{MODEL_VERSION}")
    except requests.exceptions.ConnectionError:
        print("********\nRoboflow server not connected. See README file for setup commands.\n********")
        exit()

    for prediction in results["predictions"]:
        cx = int(prediction["x"]) + eggs_roi.x
        cy = int(prediction["y"]) + eggs_roi.y
        width = int(prediction["width"])
        height = int(prediction["height"])
        confidence = prediction["confidence"]
        label = prediction["class"]

        prediction = Prediction(cx, cy, width, height, confidence, label)
        prediction.draw(frame)

    cv2.imshow('Eggs/Holes Prediction', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
