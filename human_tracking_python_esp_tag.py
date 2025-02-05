import cv2
import numpy as np
import serial
import time
import argparse
from threading import Thread
from pupil_apriltags import Detector

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
        self.show_frame = show_frame
        self.serial_port = serial.Serial(serial_port, baud_rate, timeout=1) if serial_port else None
        self.detect_tag = detect_tag
        self.save_video = save_video

        print("Tracking System Started")

        # โหลดโมเดลสำหรับ Human Tracking
        self.net = cv2.dnn.readNetFromCaffe(
            '/home/q/auv_code/deploy.prototxt', 
            '/home/q/auv_code/mobilenet_iter_73000.caffemodel'
        )

        self.CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

        # ตัวตรวจจับ AprilTag
        self.apriltag_detector = Detector(families="tag36h11")

        # กล้อง
        self.vs = WebcamVideoStream(src=0, width=self.width, height=self.height).start()

        # ถ้าเลือกบันทึกวิดีโอ
        if self.save_video:
            current_time = str(int(time.time()))
            filename = f'output_{current_time}.avi'
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # Codec for video
            self.video_writer = cv2.VideoWriter(filename, fourcc, 10.0, (self.width, self.height))

        # ตัวแปรสำหรับการคำนวณ FPS
        self.prev_frame_time = 0
        self.new_frame_time = 0

    def send_to_esp32(self, data):
        """ ส่งข้อมูลไปยัง ESP32 """
        if self.serial_port:
            try:
                self.serial_port.write(f"{data}\n".encode())
            except serial.SerialException:
                print("Error: Could not send data to ESP32")

    def detect_human(self, frame):
        """ ตรวจจับมนุษย์และคำนวณตำแหน่ง """
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
        self.net.setInput(blob)
        detections = self.net.forward()

        for i in np.arange(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.3:
                idx = int(detections[0, 0, i, 1])
                if self.CLASSES[idx] == "person":
                    box = detections[0, 0, i, 3:7] * np.array([self.width, self.height, self.width, self.height])
                    (startX, startY, endX, endY) = box.astype("int")
                    pos_x = (startX + endX) // 2
                    pos_y = startY
                    return True, pos_x, pos_y, (startX, startY, endX, endY)
        return False, 0, 0, None

    def detect_apriltag(self, frame):
        """ ตรวจจับ AprilTag และคืนค่าหมายเลขแท็ก """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        tags = self.apriltag_detector.detect(gray)

        for tag in tags:
            tag_id = tag.tag_id
            #print(f"พบแท็ก ID: {tag_id}")

            # วาดกรอบ
            for i in range(4):
                p1 = tuple(tag.corners[i].astype(int))
                p2 = tuple(tag.corners[(i+1) % 4].astype(int))
                cv2.line(frame, p1, p2, (255, 0, 0), 2)

            # แสดงค่า ID
            cv2.putText(frame, f"ID: {tag_id}", (tag.center[0].astype(int), tag.center[1].astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            return True, tag_id
        return False, None

    def track(self):
        """ ตรวจจับมนุษย์ก่อน ถ้าไม่พบ ค่อยตรวจจับ AprilTag """
        self.new_frame_time = time.time()
        
        # คำนวณ FPS
        fps = 1 / (self.new_frame_time - self.prev_frame_time)
        self.prev_frame_time = self.new_frame_time
        
        frame = self.vs.read()
        if frame is None:
            return

        frame_center_x = self.width // 2
        frame_center_y = self.height // 2
        cv2.circle(frame, (frame_center_x, frame_center_y), 3, (0, 255, 0), -1)

        # ตรวจจับมนุษย์ก่อน
        person_detected, pos_x, pos_y, bbox = self.detect_human(frame)

#ตรวจ tag ก่อน  งานส่วนใหญ่คือแทกคน  ถ้าจะให้ตรวจtag ก่อนแล้วค่อยตรวจคนมันน่าจะช้ากว่าเพราะตอนที่จะหาคนต้องรู้ก่อนว่าไม่มีtag   ต่างกันถ้าตรวจคนก่อน  ไม่จำเป็นต้องหาว่ามีtagอยู่มั้ย
        if person_detected:
            error_x = pos_x - frame_center_x
            error_y = pos_y - frame_center_y

            # วาดกรอบรอบมนุษย์
            startX, startY, endX, endY = bbox
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            cv2.putText(frame, f"Error X: {error_x}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.putText(frame, f"Error Y: {error_y}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

            # Draw error lines
            cv2.line(frame, (frame_center_x, frame_center_y), (pos_x, frame_center_y), (0, 0, 255), 1)  # X-axis (red)
            cv2.line(frame, (pos_x, pos_y), (pos_x, frame_center_y), (255, 0, 0), 1)  # Y-axis (blue)

            self.send_to_esp32(f"{error_x},{error_y}")
            print(f"Human Detected | Error X: {error_x}, Error Y: {error_y}")

        else:
            # ถ้าไม่เจอมนุษย์ ตรวจจับ AprilTag
            tag_detected, tag_id = self.detect_apriltag(frame) if self.detect_tag else (False , None)

            if tag_detected:
                print(f"AprilTag Detected | ID: {tag_id}")
                self.send_to_esp32(f"999,{tag_id}")
            else:
                self.send_to_esp32("0,0")
                print("No Target Detected")

        if self.show_frame:
            cv2.imshow('Frame', frame)
            if cv2.waitKey(20) & 0xFF == ord('q'):
                self.vs.stop()
                if self.serial_port:
                    self.serial_port.close()
                cv2.destroyAllWindows()
                exit()

        if self.save_video:
            self.video_writer.write(frame)

        # แสดง FPS ใน terminal
        print(f"FPS: {fps:.2f}")

    def __del__(self):
        if hasattr(self, 'vs'):
            self.vs.stop()
        if hasattr(self, 'video_writer'):
            self.video_writer.release()


def main():
    parser = argparse.ArgumentParser(description="Human & AprilTag Tracking")
    parser.add_argument('--hide', action='store_true', help="Hide the frame display")
    parser.add_argument('--port', type=str, default=None, help="Serial port for ESP32 (e.g., /dev/ttyUSB0 )")
    parser.add_argument('--tag', action='store_true', help="Enable AprilTag detection when human is not detected")
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
