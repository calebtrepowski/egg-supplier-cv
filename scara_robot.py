from serial import Serial
import time


class ScaraRobot:
    serial_port: Serial

    def __init__(self, port: str, baudrate: int):
        self.serial_port = Serial(port, baudrate, timeout=1.0)
        time.sleep(2)

    def go_to_cartesian_coordinate(self, /,
                                   x: float | None = None,
                                   y: float | None = None,
                                   z: float | None = None,
                                   phi: float | None = None):
        g_code_command = "G0"

        if x is not None:
            g_code_command += f" X{x}"
        if y is not None:
            g_code_command += f" Y{y}"
        if z is not None:
            g_code_command += f" Z{z}"

        self.send_command(g_code_command)
        return g_code_command

    def go_to_articular_coordinate(self, /,
                                   j1: float | None = None,
                                   j2: float | None = None,
                                   j3: float | None = None,
                                   j4: float | None = None):
        g_code_command = "G11"

        if j1 is not None:
            g_code_command += f" H{j1}"
        if j2 is not None:
            g_code_command += f" J{j2}"
        if j3 is not None:
            g_code_command += f" K{j3}"
        if j4 is not None:
            g_code_command += f" L{j4}"

        self.send_command(g_code_command)
        return g_code_command

    def set_articular_coordinate(self, /,
                                 j1: float | None = None,
                                 j2: float | None = None,
                                 j3: float | None = None,
                                 j4: float | None = None):
        g_code_command = "G12"

        if j1 is not None:
            g_code_command += f" H{j1}"
        if j2 is not None:
            g_code_command += f" J{j2}"
        if j3 is not None:
            g_code_command += f" K{j3}"
        if j4 is not None:
            g_code_command += f" L{j4}"

        self.send_command(g_code_command)
        return g_code_command

    def go_to_limit(self):
        g_code_command = "G10"
        self.send_command(g_code_command)
        return g_code_command

    def send_command(self, command: str):
        self.serial_port.write((command + "\n").encode())

    def read_response(self):
        return self.serial_port.read_all().decode()


if __name__ == "__main__":
    robot = ScaraRobot("/dev/ttyUSB0", 115200)
    robot.set_articular_coordinate(j1=0)
