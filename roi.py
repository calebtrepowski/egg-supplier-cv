import cv2
from colors import ColorsBGR
from camera import capture


class ROI:
    x: int
    y: int
    width: int
    height: int
    label: str

    def __init__(self, label: str):
        self.x = None
        self.y = None
        self.width = None
        self.height = None
        self.label = label

    def _capture(self):
        # capture = cv2.VideoCapture(index=2)
        for i in range(10):
            ret, frame = capture.read()
        ret, frame = capture.read()
        WINDOW_NAME = f'Select {self.label} ROI'
        roi = cv2.selectROI(WINDOW_NAME,
                            frame,
                            fromCenter=False,
                            showCrosshair=True)
        cv2.destroyWindow(WINDOW_NAME)
        x, y, w, h = roi
        # capture.release()
        return x, y, w, h

    def save(self):
        x, y, w, h = self._capture()
        with open(f"roi_{self.label}.txt", "w") as f:
            f.write(f"{x},{y},{w},{h}\n")

    def load(self):
        try:
            with open(f"roi_{self.label}.txt") as f:
                roi_str = f.read().rstrip("\n").split(",")
                self.x, self.y, self.width, self.height = [
                    int(i) for i in roi_str]

        except Exception:
            self.save()
            with open(f"roi_{self.label}.txt") as f:
                roi_str = f.read().rstrip("\n").split(",")
                self.x, self.y, self.width, self.height = [
                    int(i) for i in roi_str]

    def draw(self, frame: cv2.typing.MatLike):
        cv2.rectangle(frame,
                      (self.x, self.y),
                      (self.x+self.width, self.y+self.height),
                      ColorsBGR.BLUE,
                      thickness=2)
        cv2.putText(frame,
                    f"{self.label} ROI",
                    (self.x, self.y-10),
                    cv2.FONT_ITALIC,
                    fontScale=1,
                    color=ColorsBGR.BLUE,
                    thickness=2)

    def get_frame(self, frame: cv2.typing.MatLike):
        return frame[self.y:self.y+self.height,
                     self.x:self.x+self.width]
