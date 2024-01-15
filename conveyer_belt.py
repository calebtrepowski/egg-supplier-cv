import serial
import time


class ConveyerBelt:
    serial_port = serial.Serial

    def __init__(self, port: str, baudrate: int):
        self.serial_port = serial.Serial(port, baudrate, timeout=1.0)
        time.sleep(2)

    def move(self):
        self.serial_port.write(b"1")

    def stop(self):
        self.serial_port.write(b"0")


if __name__ == "__main__":
    import time
    belt = ConveyerBelt("/dev/ttyUSB0", 9600)
    if not belt.serial_port.is_open:
        exit()
    belt.move()
    time.sleep(2)
    belt.stop()
