import cv2
import numpy as np
import serial
from threading import Thread
import argparse
import time

class WebcamVideoStream:
    def __init__(self, src, width, height):
        self.width = width
        self.height = height
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while True:
            if self.stopped:
                return
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True


class HumanTracking:
    def __init__(self, width, height, serial_port=None, baud_rate=115200, show_frame=True, save_video=False):
        self.width = width
        self.height = height
        self.show_frame = show_frame
        self.save_video = save_video
        self.serial_port = serial.Serial(serial_port, baud_rate, timeout=1) if serial_port else None

        print("Human Tracking Started")

        self.net = cv2.dnn.readNetFromCaffe(
            '/home/rpiauv-server/auv_code/deploy.prototxt', 
            '/home/rpiauv-server/auv_code/mobilenet_iter_73000.caffemodel'
        )

        self.CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

        self.vs = WebcamVideoStream(src=0, width=self.width, height=self.height).start()

        # Initialize video writer if saving video
        if self.save_video:
            current_time = str(int(time.time()))
            filename = f'output_{current_time}.avi'
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # Codec for video
            self.video_writer = cv2.VideoWriter(filename, fourcc, 10.0, (self.width, self.height))

    def send_to_esp32(self, error_x, error_y):
        if self.serial_port:
            data = f"{error_x},{error_y}\n"
            try:
                self.serial_port.write(data.encode())
            except serial.SerialException:
                print("Error: Could not send data to ESP32")

    def track_humans(self):
        start_time = time.time() 

        frame = self.vs.read()
        if frame is None:
            return

        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
        self.net.setInput(blob)
        detections = self.net.forward()

        frame_center_x = self.width // 2
        frame_center_y = self.height // 2
        cv2.circle(frame, (frame_center_x, frame_center_y), 3, (0, 255, 0), -1)

        person_detected = False
        for i in np.arange(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.2:
                idx = int(detections[0, 0, i, 1])
                if self.CLASSES[idx] == "person":
                    person_detected = True
                    box = detections[0, 0, i, 3:7] * np.array([self.width, self.height, self.width, self.height])
                    (startX, startY, endX, endY) = box.astype("int")

                    pos_x = (startX + endX) // 2
                    pos_y = startY
                    error_x = pos_x - frame_center_x
                    error_y = pos_y - frame_center_y

                    self.send_to_esp32(error_x, error_y)

                    label = f"{self.CLASSES[idx]}: {confidence:.2f}"
                    cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                    cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

                    # Draw error lines
                    cv2.line(frame, (frame_center_x, frame_center_y), (pos_x, frame_center_y), (0, 0, 255), 1)  # X-axis (red)
                    cv2.line(frame, (pos_x, pos_y), (pos_x, frame_center_y), (255, 0, 0), 1)  # Y-axis (blue)

                    # Display error values on frame
                    cv2.putText(frame, f"Error X: {error_x}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    cv2.putText(frame, f"Error Y: {error_y}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        # Display FPS in terminal
        end_time = time.time()
        fps = 1 / (end_time - start_time)

        if  person_detected:
            print(f"w:{self.width} - h:{self.height} | FPS: {fps:.2f} | pos_x:{pos_x} - pos_y:{pos_y} | error_x:{error_x} - error_y:{error_y}")
        else:
            print(f"w:{self.width} - h:{self.height} | FPS: {fps:.2f} | No person detected ! |")

        if self.show_frame:
            cv2.imshow('Frame', frame)
            if cv2.waitKey(20) & 0xFF == ord('q'):
                self.vs.stop()
                if self.serial_port:
                    self.serial_port.close()
                cv2.destroyAllWindows()
                exit()

        # If we are saving video, write the frame to the video file
        if self.save_video:
            self.video_writer.write(frame)

    def __del__(self):
        if self.save_video:
            self.video_writer.release()  # Release the video writer


def main():
    parser = argparse.ArgumentParser(description="Human Tracking Program")
    parser.add_argument('--hide', action='store_true', help="Hide the frame display")
    parser.add_argument('--vdo', action='store_true', help="Save and show the video")
    parser.add_argument('--port', type=str, default=None, help="Serial port for ESP32 communication (e.g., /dev/ttyUSB0 or COM3)")
    args = parser.parse_args()

    show_frame = not args.hide  # If --hide, will not display frame
    save_video = args.vdo  # If --vdo, will save and show video
    serial_port = args.port

    print(f"\n\nshow_frame:{show_frame} | save_video:{save_video} | serial_port:{serial_port}\n\n")

    width, height = 320, 240  # Set width and height
    tracker = HumanTracking(width, height, serial_port=serial_port,show_frame=show_frame, save_video=save_video)

    while True:
        tracker.track_humans()

if __name__ == '__main__':
    main()
