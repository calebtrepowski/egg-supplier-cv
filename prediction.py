import cv2
from typing import Literal
from colors import ColorsBGR


class Prediction:
    cx: int
    cy: int
    width: int
    height: int
    confidence: float
    class_label: Literal["egg", "hole"]

    def __init__(self, cx: int, cy: int, width: int, height: int, confidence: float, class_label: str):
        self.cx = cx
        self.cy = cy
        self.width = width
        self.height = height
        self.confidence = confidence
        self.class_label = "egg" if class_label == "eggs" else "hole"

    def draw(self, frame: cv2.typing.MatLike):
        color = ColorsBGR.EGG_PREDICTION if self.class_label == "egg" else ColorsBGR.HOLE_PREDICTION
        cv2.rectangle(frame,
                      (self.cx-int(self.width/2), self.cy-int(self.height/2)),
                      (self.cx+int(self.width/2), self.cy+int(self.height/2)),
                      color=color,
                      thickness=1)
        cv2.putText(frame,
                    f"{self.confidence:.2f}",
                    (self.cx-int(self.width/2), self.cy - int(self.height/2)-10),
                    cv2.FONT_ITALIC,
                    fontScale=0.5,
                    color=color,
                    thickness=1)
