from inference_sdk import InferenceHTTPClient, InferenceConfiguration
from dotenv import dotenv_values
import cv2

config = dotenv_values(".env")
PROJECT_ID = config["ROBOFLOW_PROJECT_ID"]
MODEL_VERSION = config["ROBOFLOW_MODEL_VERSION"]
CONFIDENCE_THRESHOLD = 0.80
custom_configuration = InferenceConfiguration(
    confidence_threshold=CONFIDENCE_THRESHOLD)

client = InferenceHTTPClient(
    api_url="http://localhost:8080",
    api_key=config["ROBOFLOW_API_KEY"]
)
client.select_api_v0()
capture = cv2.VideoCapture(2)

while True:
    ret, image = capture.read()
    if not ret:
        print("Error reading image from camera")
        exit()

    results = client.infer(image, model_id=f"{PROJECT_ID}/{MODEL_VERSION}")

    for prediction in results["predictions"]:
        x = int(prediction["x"])
        y = int(prediction["y"])
        w = int(prediction["width"])
        h = int(prediction["height"])

        confidence = prediction["confidence"]
        label = prediction["class"]
        cv2.rectangle(image, (x-int(w/2), y-int(h/2)),
                      (x+int(w/2), y+int(h/2)), (0, 255, 0), 1)

    cv2.imshow('Original', image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
