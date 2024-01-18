import tkinter as tk
from tkinter import ttk
import cv2
import threading
from PIL import Image, ImageTk
import re
from reference_system import ReferenceSystem, ReferenceCircle
from predicter import EggPredicter
from conveyer_belt import ConveyerBelt
from scara_robot import ScaraRobot
from supply_egg import SupplySystem


class EggSupplierCV:
    window: tk.Tk
    video_capture: cv2.VideoCapture
    camera_thread: threading.Thread
    reference_system: ReferenceSystem
    egg_predicter: EggPredicter
    confidence_threshold: tk.DoubleVar
    belt: ConveyerBelt
    scara_robot: ScaraRobot
    supply_system = SupplySystem
    is_robot_busy: bool

    is_routine_running: bool
    continue_routine: bool

    robot_status: int

    robot_x: tk.DoubleVar
    robot_y: tk.DoubleVar
    robot_z: tk.DoubleVar
    robot_j1: tk.DoubleVar
    robot_j2: tk.DoubleVar
    robot_j3: tk.DoubleVar
    robot_j4: tk.DoubleVar

    def __init__(self, /, video_source=0, belt_port: str = "", robot_port: str = ""):

        self.window = tk.Tk()
        self.window.title("Envasador de Huevos - GUI")
        self.window.bind('<Escape>', lambda e: self.window.quit())
        self.window.bind('q', lambda e: self.window.quit())
        self.window.bind('w', lambda e: self.stop_thread()
                         if self.is_predicter_running else self.start_thread())
        self.window.minsize(1100, 580)

        self.video_capture = cv2.VideoCapture(video_source)

        self.root_frame = ttk.Frame(self.window, padding=10)
        self.root_frame.grid(sticky=tk.NSEW)

        self.confidence_threshold = tk.DoubleVar()
        self.confidence_threshold.set(0.50)
        self.egg_predicter = EggPredicter(
            self.confidence_threshold.get(), self.video_capture)

        self.robot_status = 0
        self.robot_x = tk.DoubleVar()
        self.robot_y = tk.DoubleVar()
        self.robot_z = tk.DoubleVar()
        self.robot_j1 = tk.DoubleVar()
        self.robot_j2 = tk.DoubleVar()
        self.robot_j3 = tk.DoubleVar()
        self.robot_j4 = tk.DoubleVar()

        self._create_camera_frame()
        self._create_robot_frame()

        self.thread = None
        self.is_predicter_running = False
        self.is_routine_running = False
        self.continue_routine = False

        self.reference_system = ReferenceSystem(self.video_capture)

        for circle in (self.reference_system.reference_1,
                       self.reference_system.reference_2,
                       self.reference_system.reference_3,
                       self.reference_system.reference_4):
            while circle.position_camera is None:
                circle.update_roi(self.video_capture)
                circle.update_position_camera(self.video_capture)
                print(circle.position_camera)

        # self.belt = ConveyerBelt(belt_port, 115200)
        self.scara_robot = ScaraRobot(robot_port, 115200)
        self.supply_system = SupplySystem()
        self.is_robot_busy = False

        self.callbacks_ids = []

        self.window.protocol("WM_DELETE_WINDOW", self.on_close)

    def _create_camera_frame(self):
        self.camera_frame = ttk.Frame(self.root_frame, padding=10)
        self.camera_frame.grid(row=0, column=0, sticky=tk.NS)

        self.camera_canvas = tk.Canvas(
            self.camera_frame,
            width=self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH),
            height=self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.camera_canvas.grid(row=0, column=0, columnspan=5)

        ttk.Label(self.camera_frame, text="Actualizar ROIs").grid(
            row=1, columnspan=5, pady=10, sticky=tk.NS)

        self.update_eggs_roi_button = ttk.Button(
            self.camera_frame,
            text="Huevos",
            command=self.update_eggs_roi)
        self.update_eggs_roi_button.grid(
            row=2, column=0, padx=2, sticky=tk.NSEW)

        self.update_reference_1_roi_button = ttk.Button(
            self.camera_frame,
            text="Referencia 1",
            command=lambda: self.update_circle_roi(self.reference_system.reference_1))
        self.update_reference_1_roi_button.grid(
            row=2, column=1, padx=2, sticky=tk.NSEW)

        self.update_reference_2_roi_button = ttk.Button(
            self.camera_frame,
            text="Referencia 2",
            command=lambda: self.update_circle_roi(self.reference_system.reference_2))
        self.update_reference_2_roi_button.grid(
            row=2, column=2, padx=2, sticky=tk.NSEW)

        self.update_reference_3_roi_button = ttk.Button(
            self.camera_frame,
            text="Referencia 3",
            command=lambda: self.update_circle_roi(self.reference_system.reference_3))
        self.update_reference_3_roi_button.grid(
            row=2, column=3, padx=2, sticky=tk.NSEW)

        self.update_reference_4_roi_button = ttk.Button(
            self.camera_frame,
            text="Referencia 4",
            command=lambda: self.update_circle_roi(self.reference_system.reference_4))
        self.update_reference_4_roi_button.grid(
            row=2, column=4, padx=2, sticky=tk.NSEW)

        self.confidence_threshold_string = tk.StringVar()
        self.confidence_threshold_string.set(
            f"Umbral de certeza ({self.confidence_threshold.get():.2f})")
        self.confidence_threshold_label = ttk.Label(
            self.camera_frame, textvariable=self.confidence_threshold_string)
        self.confidence_threshold_label.grid(
            row=3, column=0, columnspan=2, pady=20, sticky=tk.W)

        self.confidence_threshold_scale = ttk.Scale(
            self.camera_frame, from_=0.01, to=0.99, variable=self.confidence_threshold, command=self.update_confidence_threshold)
        self.confidence_threshold_scale.grid(
            row=3, column=2, columnspan=3, sticky=tk.EW, padx=4)

    def _create_robot_frame(self):
        self.robot_frame = ttk.Frame(self.root_frame, padding=10)
        self.robot_frame.grid(row=0, column=1, sticky=tk.NS)

        self.robot_calibrate_button = ttk.Button(
            self.robot_frame, text="Calibrar limites", command=lambda: self.send_command(self.scara_robot.go_to_limit()))
        self.robot_calibrate_button.grid(
            row=0, column=0, columnspan=2, sticky=tk.EW, pady=2)

        self.robot_home_button = ttk.Button(
            self.robot_frame, text="Ir a Home",
            command=lambda: self.scara_robot.go_to_articular_coordinate(j1=0, j2=SupplySystem.SAFE_Z_POSITION_MM, j3=0))
        self.robot_home_button.grid(
            row=1, column=0, columnspan=2, sticky=tk.EW, pady=2)

        self.robot_start_routine_button = ttk.Button(
            self.robot_frame, text="Iniciar rutina", command=self.run_supply_egg)
        self.robot_start_routine_button.grid(
            row=2, column=0, columnspan=2, sticky=tk.EW, pady=2)

        self.g_code_text = tk.Text(
            self.robot_frame, wrap=tk.WORD, height=10, width=40)
        self.g_code_scrollbar = ttk.Scrollbar(
            self.robot_frame, command=self.g_code_text.yview)
        self.g_code_text.config(
            yscrollcommand=self.g_code_scrollbar.set(first=0, last=0.25))
        self.g_code_text.grid(row=3, column=0, sticky=tk.NSEW)
        self.g_code_scrollbar.grid(row=3, column=1, sticky=tk.NS)

        self.g_code_text.config(state=tk.DISABLED)

        ttk.Separator(self.robot_frame, orient=tk.HORIZONTAL).grid(
            row=4, columnspan=2, sticky=tk.EW, pady=10)

        self.terminal_input_entry = ttk.Entry(self.robot_frame)
        self.terminal_input_entry.grid(
            row=5, columnspan=2, sticky=tk.EW, pady=2)
        self.terminal_input_entry.bind(
            "<Return>", lambda e: self.send_command())
        self.terminal_input_entry.bind(
            "<KP_Enter>", lambda e: self.send_command())
        self.terminal_send_button = ttk.Button(
            self.robot_frame, text="Enviar comando", command=self.send_command)
        self.terminal_send_button.grid(
            row=6, columnspan=2, sticky=tk.EW, pady=2)

        ttk.Separator(self.robot_frame, orient=tk.HORIZONTAL).grid(
            row=7, columnspan=2, sticky=tk.EW, pady=10)
        ttk.Label(self.robot_frame, text="XYZ").grid(
            row=8, columnspan=2, pady=2, sticky=tk.EW)
        ttk.Label(self.robot_frame, textvariable=self.robot_x).grid(
            row=9, columnspan=2, pady=2, sticky=tk.EW)
        ttk.Label(self.robot_frame, textvariable=self.robot_y).grid(
            row=10, columnspan=2, pady=2, sticky=tk.EW)
        ttk.Label(self.robot_frame, textvariable=self.robot_z).grid(
            row=11, columnspan=2, pady=2, sticky=tk.EW)
        ttk.Label(self.robot_frame, text="J1 J2 J3 J4").grid(
            row=12, columnspan=2, pady=2, sticky=tk.EW)
        ttk.Label(self.robot_frame, textvariable=self.robot_j1).grid(
            row=13, columnspan=2, pady=2, sticky=tk.EW)
        ttk.Label(self.robot_frame, textvariable=self.robot_j2).grid(
            row=14, columnspan=2, pady=2, sticky=tk.EW)
        ttk.Label(self.robot_frame, textvariable=self.robot_j3).grid(
            row=15, columnspan=2, pady=2, sticky=tk.EW)
        ttk.Label(self.robot_frame, textvariable=self.robot_j4).grid(
            row=16, columnspan=2, pady=2, sticky=tk.EW)

    def update(self):
        _, frame = self.video_capture.read()

        if frame is not None:
            self.reference_system.reference_1.draw(frame)
            self.reference_system.reference_2.draw(frame)
            self.reference_system.reference_3.draw(frame)
            self.reference_system.reference_4.draw(frame)

            self.egg_predicter.eggs_roi.draw(frame)

            try:
                self.predictions = self.egg_predicter.predict(frame)
                for i, p in enumerate(self.predictions):
                    target_point_robot = self.reference_system.get_robot_coordinates(
                        p.cx, p.cy)
                    p.draw(frame, number=i)
                    cv2.putText(frame,
                                f"{i}: ({target_point_robot[0]:.2f},{target_point_robot[1]:.2f})",
                                (450, 470-20*i),
                                cv2.FONT_ITALIC,
                                fontScale=0.55,
                                color=(0, 0, 255),
                                thickness=2)
            except Exception as e:
                print(e)

            self.tkinter_image = self.convert_to_tkinter_image(frame)
            self.camera_canvas.create_image(
                0, 0, anchor=tk.NW, image=self.tkinter_image)
        if self.is_predicter_running:
            self.window.after(10, self.update)

    def start_thread(self):
        print("starting thread")
        if not self.is_predicter_running:
            self.is_predicter_running = True
            self.thread = threading.Thread(target=self.update)
            self.thread.start()

    def stop_thread(self):
        print("stopping thread")
        if self.is_predicter_running:
            self.is_predicter_running = False
            self.thread.join()

    def on_close(self):
        self.stop_thread()
        self.scara_robot.serial_port.close()
        self.video_capture.release()
        self.window.destroy()

    def convert_to_tkinter_image(self, frame):
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        photo = ImageTk.PhotoImage(image=image)
        return photo

    def run(self):
        self.update()
        self.start_thread()
        self.window.mainloop()

    def update_circle_roi(self, circle: ReferenceCircle):
        self.stop_thread()
        circle.position_camera = None
        circle.radius_camera = None
        while circle.position_camera is None:
            circle.update_roi(self.video_capture)
            circle.update_position_camera(self.video_capture)
        self.reference_system.update_homography_matrix()
        self.window.after(500, self.start_thread)

    def update_eggs_roi(self):
        self.stop_thread()
        self.egg_predicter.calibrate_roi(self.video_capture)
        self.egg_predicter.eggs_roi.load(self.video_capture)
        self.window.after(500, self.start_thread)

    def update_confidence_threshold(self, value: float):
        self.confidence_threshold_string.set(
            f"Umbral de certeza ({float(value):.2f})")
        self.confidence_threshold.set(value)
        self.egg_predicter.set_confidence(value)

    def update_g_code_text(self, command_sent: str):
        command_sent += "\n"
        self.g_code_text.config(state="normal")
        self.g_code_text.insert(tk.END, command_sent)
        self.g_code_text.config(state="disabled")

    def send_command(self, command=None):
        self.scara_robot.serial_port.read_all()

        if command is not None:
            self.terminal_input_entry.delete(0, tk.END)
            self.terminal_input_entry.insert(0, command)

        command = self.terminal_input_entry.get().upper()

        # self.scara_robot.serial_port.write((command + "\n").encode())
        self.scara_robot.send_command(command)
        self.update_g_code_text(command)
        self.terminal_input_entry.delete(0, tk.END)
        self.terminal_input_entry.focus()

        self.callbacks_ids.append(self.window.after(
            500, self.update_robot_response))

    def run_supply_egg(self):
        self.is_routine_running = True
        print("iniciando rutina!")
        self.window.after(500, self.predict_holes)

    def predict_holes(self):
        print("predict holes 1")
        if not self.is_routine_running:
            return
        print("predict holes 2")

        if self.is_robot_busy:
            self.window.after(1000, self.predict_holes)

        for i in range(2500):
            # if len(self.predictions) >= 6:
            if True:
                self.target_hole = None
                for prediction in self.predictions:
                    if prediction.class_label == "hole":
                        self.target_hole = prediction
                        print("target hole!")
                        self.stop_thread()
                        break
                if self.target_hole is not None:
                    if self.supply_system.next_egg_index < len(self.supply_system.supply_eggs):
                        supply_egg = self.supply_system.supply_eggs[self.supply_system.next_egg_index]
                        target_point_robot = self.reference_system.get_robot_coordinates(
                            self.target_hole.cx, self.target_hole.cy)

                        self.current_command = 0

                        self.commands_list = (
                            self.scara_robot.go_to_articular_coordinate(
                                j2=SupplySystem.SAFE_Z_POSITION_MM),
                            self.scara_robot.go_to_cartesian_coordinate(
                                x=supply_egg.grab_x_position_mm,
                                y=supply_egg.grab_y_position_mm),
                            self.scara_robot.go_to_articular_coordinate(
                                j2=SupplySystem.APPROACH_Z_POSITION_MM),
                            self.scara_robot.go_to_articular_coordinate(
                                j4=supply_egg.grab_j4_angle_degrees),
                            self.scara_robot.go_to_articular_coordinate(
                                j2=SupplySystem.GRAB_Z_POSITION_MM),
                            self.scara_robot.act_tool(
                                value=SupplySystem.GRAB_GRIPPER_VALUE),
                            self.scara_robot.go_to_articular_coordinate(
                                j2=SupplySystem.SAFE_Z_POSITION_MM),
                            self.scara_robot.go_to_cartesian_coordinate(
                                x=target_point_robot[0], y=target_point_robot[1]),
                            self.scara_robot.go_to_articular_coordinate(
                                j2=SupplySystem.RELEASE_Z_POSITION_MM),
                            self.scara_robot.release_tool(),
                            self.scara_robot.go_to_articular_coordinate(
                                j2=SupplySystem.SAFE_Z_POSITION_MM),
                            self.scara_robot.go_to_articular_coordinate(
                                j1=0,
                                j3=0)
                        )
                    self.callbacks_ids.append(
                        self.window.after(500, self.execute_next_command))
                    self.is_robot_busy = True
                    break
        if self.target_hole is None:
            print("no detectado")
            self.is_routine_running = False

    def execute_next_command(self):
        print("execute next command")
        print(self.robot_status)
        if self.robot_status == 0:
            if self.current_command < len(self.commands_list):
                print(
                    f"next_command!: {self.commands_list[self.current_command]}")
                self.send_command(self.commands_list[self.current_command])
                self.current_command += 1
                self.callbacks_ids.append(self.window.after(
                    1000, self.execute_next_command))
            else:
                print("huevo entregado")
                self.commands_list = tuple()
                self.start_thread()
                self.window.after(5000, self.predict_holes)
                self.is_robot_busy = False
                self.supply_system.next_egg_index += 1
        elif self.robot_status is None:
            self.callbacks_ids.append(self.window.after(
                1000, self.execute_next_command))
        else:
            print("cancelling tasks")
            for task_id in self.callbacks_ids:
                self.window.after_cancel(task_id)
            self.callbacks_ids.clear()

    def update_robot_response(self):
        print("update_robot_response")
        self.robot_response = self.scara_robot.serial_port.readline(
        ).decode().rstrip("&\r\n").lstrip("$")
        print(self.robot_response)
        if not self.robot_response.startswith("["):
            # self.robot_response = ""
            self.callbacks_ids.append(self.window.after(
                500, self.update_robot_response))
        else:
            self.is_robot_busy = False
            self.parse_robot_params()

    def parse_robot_params(self):
        pattern = r"\[(\d+)\]XYZ:\(([\d.-]+),([\d.-]+),([\d.-]+)\);J1J2J3J4:\(([\d.-]+),([\d.-]+),([\d.-]+),([\d.-]+)\)"
        match = re.match(pattern, self.robot_response)

        if match:
            status_number = int(match.group(1))
            x_coord, y_coord, z_coord = float(match.group(2)), float(
                match.group(3)), float(match.group(4))
            j1, j2, j3, j4 = float(match.group(5)), float(
                match.group(6)), float(match.group(7)), float(match.group(8))

            self.robot_status = status_number
            self.robot_x.set(x_coord)
            self.robot_y.set(y_coord)
            self.robot_z.set(z_coord)
            self.robot_j1.set(j1)
            self.robot_j2.set(j2)
            self.robot_j3.set(j3)
            self.robot_j4.set(j4)
            self.robot_response = ""
        else:
            self.robot_status = None


if __name__ == "__main__":
    EggSupplierCV(video_source=2, robot_port="/dev/ttyUSB0").run()
