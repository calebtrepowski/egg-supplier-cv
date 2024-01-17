from inference_sdk import InferenceHTTPClient, InferenceConfiguration
from dotenv import dotenv_values
import cv2
from roi import ROI
from prediction import Prediction

config = dotenv_values(".env")


class EggPredicter:
    PROJECT_ID: str = config["ROBOFLOW_PROJECT_ID"]
    MODEL_VERSION: str = config["ROBOFLOW_MODEL_VERSION"]
    inference_client: InferenceHTTPClient
    eggs_roi: ROI

    def __init__(self, confidence_threshold: float, capture: cv2.VideoCapture):

        self.inference_client = InferenceHTTPClient(
            api_url="http://localhost:8080",
            api_key=config["ROBOFLOW_API_KEY"]
        )

        self.inference_client.configure(InferenceConfiguration(
            confidence_threshold=confidence_threshold))
        self.inference_client.select_api_v0()

        self.eggs_roi = ROI("eggs")
        self.eggs_roi.load(capture)

    def calibrate_roi(self, capture: cv2.VideoCapture):
        self.eggs_roi.save(capture)

    def set_confidence(self, new_confidence: float):
        self.inference_client.configure(InferenceConfiguration(
            confidence_threshold=new_confidence))

    def predict(self, frame: cv2.typing.MatLike) -> list[Prediction]:
        eggs_roi_frame = self.eggs_roi.get_frame(frame)
        self.eggs_roi.draw(frame)
        results = self.inference_client.infer(
            eggs_roi_frame,
            model_id=f"{EggPredicter.PROJECT_ID}/{EggPredicter.MODEL_VERSION}")["predictions"]
        predictions = []
        for prediction in results:
            cx = int(prediction["x"]) + self.eggs_roi.x
            cy = int(prediction["y"]) + self.eggs_roi.y
            width = int(prediction["width"])
            height = int(prediction["height"])
            confidence = prediction["confidence"]
            label = prediction["class"]

            prediction = Prediction(cx, cy, width, height, confidence, label)
            predictions.append(prediction)
            prediction.draw(frame)

        return predictions


if __name__ == "__main__":
    import cv2
    capture = cv2.VideoCapture(2)
    egg_predicter = EggPredicter(0.1, capture)


    while True:
        ret, frame = capture.read()
        if not ret:
            print("Error reading image from camera")
            exit()
        try:
            predictions = egg_predicter.predict(frame)
        except Exception as e:
            print(e)
            continue
        cv2.imshow('Predicter', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    capture.release()
    cv2.destroyAllWindows()
