import cv2
import numpy as np
import serial
import time
import argparse
from threading import Thread
from pupil_apriltags import Detector
from queue import Queue

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

class TrackingSystem:
    def __init__(self, width, height, serial_port=None, baud_rate=115200, show_frame=True, detect_tag=False, save_video=False):
        self.width = width
        self.height = height
        self.frame_center_x = self.width // 2
        self.frame_center_y = self.height // 2
        self.show_frame = show_frame
        self.serial_port = serial.Serial(serial_port, baud_rate, timeout=1) if serial_port else None
        self.detect_tag = detect_tag
        self.save_video = save_video

        print("Tracking System Started")

        self.net = cv2.dnn.readNetFromCaffe(
            'deploy.prototxt', 
            'mobilenet_iter_73000.caffemodel'
        )

        self.CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

        self.apriltag_detector = Detector(families="tag36h11")
        self.vs = WebcamVideoStream(src=0, width=self.width, height=self.height).start()

        if self.save_video:
            current_time = str(int(time.time()))
            filename = f'output_{current_time}.avi'
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            self.video_writer = cv2.VideoWriter(filename, fourcc, 30.0, (self.width, self.height))

        self.prev_frame_time = 0
        self.new_frame_time = 0
        self.last_error_x = 0
        self.last_error_y = 0
        self.no_target_count = 0
        self.max_no_target_count = 8
        self.EN = -1

        self.serial_queue = Queue()
        self.serial_thread = Thread(target=self.serial_worker, daemon=True)
        self.serial_thread.start()

    def serial_worker(self):
        while True:
            data = self.serial_queue.get()
            if self.serial_port:
                try:
                    self.serial_port.write(f"{data}\n".encode())
                except serial.SerialException:
                    print("Error: Could not send data to ESP32")

    def send_to_esp32(self, data):
        self.EN *= -1
        data += f',{self.EN}'
        if self.serial_queue.qsize() < 5:
            self.serial_queue.put(data)
        #print(f"{self.EN}\t{data}")

    def detect_apriltag(self, gray, frame):
        tags = self.apriltag_detector.detect(gray)
        target_tag = None
        other_tags = []
        
        for tag in tags:
            tag_id = tag.tag_id
            if tag_id == 3:  # ID 3 เป็นเป้าหมายหลักแทนคน
                target_tag = tag
            elif tag_id in [1, 2]:  # ID 1 และ 2 จัดเก็บแยก
                other_tags.append(tag)
            
            if self.show_frame or self.save_video:
                for i in range(4):
                    p1 = tuple(tag.corners[i].astype(int))
                    p2 = tuple(tag.corners[(i+1) % 4].astype(int))
                    cv2.line(frame, p1, p2, (255, 0, 0), 2)
                cv2.putText(frame, f"ID: {tag_id}", (tag.center[0].astype(int), tag.center[1].astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        return target_tag, other_tags

    def track(self):
        self.new_frame_time = time.time()
        fps = 1 / (self.new_frame_time - self.prev_frame_time)
        self.prev_frame_time = self.new_frame_time
        
        frame = self.vs.read()
        if frame is None:
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.show_frame or self.save_video:
            cv2.circle(frame, (self.frame_center_x, self.frame_center_y), 2, (0, 255, 0), -1)

        # ตรวจจับ AprilTag ID 3 แทนการตรวจจับคน
        target_tag, other_tags = self.detect_apriltag(gray, frame)

        if target_tag is not None:  # ตรวจพบ ID 3
            pos_x = int(target_tag.center[0])
            pos_y = int(target_tag.center[1])
            error_x = pos_x - self.frame_center_x
            error_y = pos_y - self.frame_center_y
            error_y = int(error_y * 50 / 120)
            
            self.last_error_x = error_x
            self.last_error_y = error_y
            self.no_target_count = 0

            if self.show_frame or self.save_video:
                cv2.circle(frame, (pos_x, pos_y), 5, (0, 255, 0), -1)
                cv2.putText(frame, f"Error X: {error_x}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                cv2.putText(frame, f"Error Y: {error_y}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                cv2.line(frame, (self.frame_center_x, self.frame_center_y), (pos_x, self.frame_center_y), (0, 0, 255), 1)
                cv2.line(frame, (pos_x, pos_y), (pos_x, self.frame_center_y), (255, 0, 0), 1)

            self.send_to_esp32(f"{error_x},{error_y}")
            print(f"AprilTag ID 3 Detected | Error X: {error_x}, Error Y: {error_y} | send_to_esp32: {error_x},{error_y}")

        else:
            # ตรวจสอบ AprilTags อื่นๆ (ID 1, ID 2) เมื่อไม่พบ ID 3
            if other_tags:
                for tag in other_tags:
                    tag_id = tag.tag_id
                    self.no_target_count = 0
                    self.send_to_esp32(f"999,{tag_id}")
                    print(f"AprilTag Detected | ID: {tag_id} | send_to_esp32: 999,{tag_id}")
                    break  # ส่งแค่ tag แรกที่พบ
            else:
                if self.no_target_count < self.max_no_target_count:
                    self.no_target_count += 1
                    self.send_to_esp32(f"{self.last_error_x},{self.last_error_y}")
                    print(f"No Target Detected ({self.no_target_count}/{self.max_no_target_count}) | Sending last errors: {self.last_error_x},{self.last_error_y}")
                else:
                    self.last_error_x = 0
                    self.last_error_y = 0
                    self.send_to_esp32("777,777")
                    print(f"No Target Detected ({self.no_target_count}/{self.max_no_target_count}) | send_to_esp32: 777,777")

        if self.show_frame:
            cv2.imshow('Frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.vs.stop()
                if self.serial_port:
                    self.serial_port.close()
                cv2.destroyAllWindows()
                exit()

        if self.save_video:
            self.video_writer.write(frame)

        print(f"\tFPS: {fps:.2f}")

    def __del__(self):
        if hasattr(self, 'vs'):
            self.vs.stop()
        if hasattr(self, 'video_writer'):
            self.video_writer.release()

def main():
    parser = argparse.ArgumentParser(description="AprilTag Tracking")
    parser.add_argument('--hide', action='store_true', help="Hide the frame display")
    parser.add_argument('--port', type=str, default=None, help="Serial port for ESP32 (e.g., /dev/ttyUSB0)")
    parser.add_argument('--tag', action='store_true', help="Enable AprilTag detection when ID 3 is not detected")
    parser.add_argument('--vdo', action='store_true', help="Save and show the video")
    args = parser.parse_args()

    show_frame = not args.hide
    serial_port = args.port
    detect_tag = args.tag
    save_video = args.vdo

    tracker = TrackingSystem(320, 240, serial_port=serial_port, show_frame=show_frame, detect_tag=detect_tag, save_video=save_video)

    while True:
        tracker.track()

if __name__ == '__main__':
    main()