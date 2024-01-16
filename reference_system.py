import cv2
import numpy as np
from roi import ROI


class ReferenceCircle:
    position_camera: np.array
    radius_camera: int
    position_robot: np.array
    roi: ROI

    def __init__(self, label: str, position_robot: tuple | list, capture: cv2.VideoCapture):
        self.position_robot = np.array(position_robot, dtype=np.float32)
        self.roi = ROI(label)
        self.roi.load(capture)
        self.position_camera = None

    def update_roi(self, capture: cv2.VideoCapture):
        self.roi.save(capture)

    def draw(self, frame: cv2.typing.MatLike):
        self.roi.draw(frame)
        if self.position_camera is not None:
            cv2.circle(frame,
                       (self.position_camera[0],
                        self.position_camera[1]),
                       self.radius_camera,
                       (0, 140, 255),
                       thickness=1)
            cv2.circle(frame,
                       (self.position_camera[0],
                        self.position_camera[1]),
                       2,
                       (0, 140, 255),
                       thickness=4)
        else:
            print("position camera is None")

    def update_position_camera(self, capture: cv2.VideoCapture):
        self.roi.load(capture)

        max_circle = None

        for i in range(10):
            ret, frame = capture.read()
            roi_frame = self.roi.get_frame(frame)

            gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
            gray_blurred = cv2.GaussianBlur(gray, (9, 9), 2)
            circles = cv2.HoughCircles(
                gray_blurred,
                method=cv2.HOUGH_GRADIENT,
                dp=1,
                minDist=50,
                param1=50,
                param2=32,
                minRadius=5,
                maxRadius=50
            )
            # minRadius=int(max(self.roi.width,
            #               self.roi.height)/5),
            # maxRadius=int(max(self.roi.width,
            #               self.roi.height)/2)

            if circles is not None:
                circles = np.uint16(np.around(circles))
                for i in circles[0, :]:
                    if max_circle is None or i[2] > max_circle[2]:
                        print("new max circle!")
                        max_circle = i

        if max_circle is not None:
            cv2.circle(frame,
                       (max_circle[0] + self.roi.x,
                        max_circle[1] + self.roi.y),
                       max_circle[2],
                       (0, 140, 255),
                       thickness=1)
            cv2.circle(frame,
                       (max_circle[0] + self.roi.x, max_circle[1]+self.roi.y),
                       2,
                       (0, 140, 255),
                       thickness=2)

            self.position_camera = np.array(
                (max_circle[0]+self.roi.x,
                 max_circle[1]+self.roi.y))
            self.radius_camera = max_circle[2]

        self.roi.draw(frame)
        cv2.imshow("Reference Circle", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # capture.release()


class ReferenceSystem:
    reference_1: ReferenceCircle
    reference_2: ReferenceCircle
    reference_3: ReferenceCircle
    reference_4: ReferenceCircle

    TRANSFORMATION_MATRIX = np.array(((-1, 0), (0, 1)))

    def __init__(self, capture: cv2.VideoCapture):
        self.reference_1 = ReferenceCircle("reference_1", (132, 79), capture)
        self.reference_2 = ReferenceCircle("reference_2", (-138, 91), capture)
        self.reference_3 = ReferenceCircle("reference_3", (-138, 337), capture)
        self.reference_4 = ReferenceCircle("reference_4", (132, 335), capture)


TRANSFORMATION_MATRIX = np.array(((-1, 0), (0, 1)))


def get_new_reference(new_zero: ReferenceCircle, current_reference: ReferenceCircle) -> np.array:
    new_reference = current_reference.position_robot-new_zero.position_robot
    new_reference = TRANSFORMATION_MATRIX @ new_reference
    return new_reference


if __name__ == "__main__":
    from predicter import EggPredicter
    egg_predicter = EggPredicter(confidence_threshold=0.15)

    # image = cv2.imread(
    #     "homography-2.png")

    reference_1 = ReferenceCircle("reference_1", (132, 79))
    reference_2 = ReferenceCircle("reference_2", (-138, 91))
    reference_3 = ReferenceCircle("reference_3", (-138, 337))
    reference_4 = ReferenceCircle("reference_4", (132, 335))

    reference_1.update_roi()
    while reference_1.position_camera is None:
        reference_1.update_position_camera()

    reference_2.update_roi()
    while reference_2.position_camera is None:
        reference_2.update_position_camera()

    # reference_3.update_roi()
    while reference_3.position_camera is None:
        reference_3.update_position_camera()

    # reference_4.update_roi()
    while reference_4.position_camera is None:
        reference_4.update_position_camera()

    # exit()

    reference_1_new = np.array((0, 0))
    reference_2_new = get_new_reference(reference_1, reference_2)
    reference_3_new = get_new_reference(reference_1, reference_3)
    reference_4_new = get_new_reference(reference_1, reference_4)

    # reference_1.position_camera = np.array((219, 87))
    # reference_2.position_camera = np.array((468, 101))
    # reference_3.position_camera = np.array((513, 428))
    # reference_4.position_camera = np.array((207, 413))

    # reference_1.radius_camera = 12
    # reference_2.radius_camera = 12
    # reference_3.radius_camera = 12
    # reference_4.radius_camera = 12

    src_pts = np.array(
        (
            reference_1.position_camera,
            reference_2.position_camera,
            reference_3.position_camera,
            reference_4.position_camera
        )).reshape(-1, 1, 2)
    dst_pts = np.array(
        (
            reference_1_new,
            reference_2_new,
            reference_3_new,
            reference_4_new
        )).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    while True:
        ret, frame = capture.read()
        if not ret:
            print("Could not read camera")
            continue
        # reference_1.draw(frame)
        # reference_2.draw(frame)
        # reference_3.draw(frame)
        # reference_4.draw(frame)

        try:
            predictions = egg_predicter.predict(frame)
            for i, p in enumerate(predictions):
                target_point_camera = np.array((p.cx, p.cy, 1))
                target_point_robot_new = H @ target_point_camera
                target_point_camera = target_point_camera[:2]
                target_point_robot = np.array((0, 0))
                target_point_robot[0] = reference_1.position_robot[0] - \
                    target_point_robot_new[0]
                target_point_robot[1] = reference_1.position_robot[1] + \
                    target_point_robot_new[1]
                cv2.circle(frame, target_point_camera, 2, (0, 255, 0), 2)
                cv2.putText(frame,
                            f"({target_point_robot[0]:.2f},{target_point_robot[1]:.2f})",
                            (target_point_camera[0],
                             target_point_camera[1]+20),
                            cv2.FONT_ITALIC,
                            fontScale=0.40,
                            color=(0, 255, 0),
                            thickness=1)

        except Exception as e:
            print(e)
            continue

        # print(reference_1.position_robot)
        # print(target_point_robot_new)

        cv2.imshow('Reference System', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    capture.release()
    cv2.destroyAllWindows()

    exit()

    # reference_1.update_roi()
    # while reference_1.position_camera is None:
    #     reference_1.update_position_camera()

    # reference_2.update_roi()
    # while reference_2.position_camera is None:
    #     reference_2.update_position_camera()

    # reference_3.update_roi()
    # while reference_3.position_camera is None:
    #     reference_3.update_position_camera()

    # while True:
    #     ret, frame = capture.read()
    #     if not ret:
    #         print("Error reading image from camera")
    #         continue

    #     reference_1.draw(frame)
    #     reference_2.draw(frame)
    #     reference_3.draw(frame)

    #     cv2.imshow('Reference System', frame)

    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break

    # capture.release()
    # cv2.destroyAllWindows()
    # exit()
# capture = cv2.VideoCapture(index=2)


#     red_circle_frame = red_circle_roi.get_frame(frame)
#     red_circle_roi.draw(frame)

#     gray = cv2.cvtColor(red_circle_frame, cv2.COLOR_BGR2GRAY)
#     gray_blurred = cv2.GaussianBlur(gray, (9, 9), 2)
#     circles = cv2.HoughCircles(
#         gray_blurred,
#         cv2.HOUGH_GRADIENT,
#         dp=1, minDist=50, param1=50, param2=32, minRadius=int(max(red_circle_roi.width, red_circle_roi.height)/5),
#         maxRadius=int(max(red_circle_roi.width, red_circle_roi.height)/2)
#     )

#     if circles is not None:
#         circles = np.uint16(np.around(circles))
#         for i in circles[0, :]:

#             cv2.circle(frame,
#                        (i[0] + red_circle_roi.x, i[1] + red_circle_roi.y),
#                        i[2],
#                        (0, 140, 255),
#                        thickness=1)
#             cv2.circle(frame,
#                        (i[0] + red_circle_roi.x, i[1]+red_circle_roi.y),
#                        2,
#                        (0, 140, 255),
#                        thickness=2)

#     cv2.imshow('Reference System', frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# capture.release()
# cv2.destroyAllWindows()
