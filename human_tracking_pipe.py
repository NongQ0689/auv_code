import cv2
import numpy as np
import serial
import time
import argparse
from threading import Thread
import mediapipe as mp
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

        print("Tracking System Started with MediaPipe Pose")

        # MediaPipe Pose setup
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)

        # ตัวตรวจจับ AprilTag
        self.apriltag_detector = Detector(families="tag36h11")

        # กล้อง
        self.vs = WebcamVideoStream(src=0, width=self.width, height=self.height).start()

        # ถ้าเลือกบันทึกวิดีโอ
        if self.save_video:
            current_time = str(int(time.time()))
            filename = f'output_{current_time}.avi'
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            self.video_writer = cv2.VideoWriter(filename, fourcc, 16.0, (self.width, self.height))

        # ตัวแปรสำหรับการคำนวณ FPS
        self.prev_frame_time = 0
        self.new_frame_time = 0

        self.last_error_x = 0
        self.last_error_y = 0
        self.no_target_count = 0
        self.max_no_target_count = 24


        self.EN = -1  # สำหรับเช็คการส่งข้อมูล

    def send_to_esp32(self, data):
        self.EN *= -1
        data += f',{self.EN}'
        print(f"{self.EN}\t", end='')

        """ ส่งข้อมูลไปยัง ESP32 """
        if self.serial_port:
            try:
                self.serial_port.write(f"{data}\n".encode())
            except serial.SerialException:
                print("Error: Could not send data to ESP32")

    def detect_human(self, frame):
        """ ตรวจจับมนุษย์ด้วย MediaPipe Pose และคำนวณตำแหน่ง """
        # แปลงสีเฟรมเพื่อใช้กับ MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)

        if results.pose_landmarks:
            # วาดโครงร่างใบหน้าและร่างกาย
            self.mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style())

            # ใช้จุดไหล่เป็นตัวกำหนดตำแหน่ง
            landmarks = results.pose_landmarks.landmark
            
            # ใช้จุดไหล่ขวาและไหล่ซ้ายเพื่อคำนวณศูนย์กลาง
            left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
            
            # คำนวณจุดกึ่งกลางระหว่างไหล่
            center_x = int((left_shoulder.x + right_shoulder.x) / 2 * self.width)
            center_y = int((left_shoulder.y + right_shoulder.y) / 2 * self.height)
            
            # สร้างกรอบสี่เหลี่ยมรอบร่างกาย
            x_coordinates = [landmark.x for landmark in landmarks]
            y_coordinates = [landmark.y for landmark in landmarks]
            
            x_min = int(min(x_coordinates) * self.width)
            x_max = int(max(x_coordinates) * self.width)
            y_min = int(min(y_coordinates) * self.height)
            y_max = int(max(y_coordinates) * self.height)
            
            # รับค่าความมั่นใจจากเฉลี่ยของ visibility ของจุดสำคัญ
            confidence = sum([landmark.visibility for landmark in landmarks]) / len(landmarks)
            
            # แสดงความมั่นใจบนเฟรม
            cv2.putText(frame, f"Conf: {confidence:.2f}", (x_min, y_min - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
            return True, center_x, center_y, (x_min, y_min, x_max, y_max)
        
        return False, 0, 0, None

    def detect_apriltag(self, frame):
        """ ตรวจจับ AprilTag และคืนค่าหมายเลขแท็ก """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        tags = self.apriltag_detector.detect(gray)

        for tag in tags:
            tag_id = tag.tag_id
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
        frame_center_y = self.height #// 2


        # ตรวจจับมนุษย์ด้วย MediaPipe Pose
        person_detected, pos_x, pos_y, bbox = self.detect_human(frame)

        if person_detected:
            error_x = pos_x - frame_center_x
            error_y = pos_y - frame_center_y
            error_y = int(error_y * 50 / 120)  # ปรับสเกลตามที่ระบุ
            
            # กำหนดน้ำหนักสำหรับค่าใหม่ (alpha) - ปรับตามความต้องการ
            alpha = 0.3  # น้ำหนักมากขึ้นจะทำให้ค่าเปลี่ยนแปลงเร็วขึ้น
            
            # คำนวณค่า error แบบเฉลี่ยเคลื่อนที่
            if self.last_error_x is None:  # กรณีแรก
                self.last_error_x = error_x
                self.last_error_y = error_y
            else:
                self.last_error_x = int(alpha * error_x + (1 - alpha) * self.last_error_x)
                self.last_error_y = int(alpha * error_y + (1 - alpha) * self.last_error_y)
            
            # รีเซ็ตตัวนับ
            self.no_target_count = 0

            if self.show_frame or self.save_video:

                # วาดกรอบรอบมนุษย์
                startX, startY, endX, endY = bbox
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                cv2.putText(frame, f"Raw Error X: {error_x}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                cv2.putText(frame, f"Raw Error Y: {error_y}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                cv2.putText(frame, f"Smooth Error X: {self.last_error_x}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                cv2.putText(frame, f"Smooth Error Y: {self.last_error_y}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

                # วาดเส้น error
                cv2.line(frame, (frame_center_x, frame_center_y), (pos_x, frame_center_y), (0, 0, 255), 1)  # X-axis (red)
                cv2.line(frame, (pos_x, pos_y), (pos_x, frame_center_y), (255, 0, 0), 1)  # Y-axis (blue)

            # ส่งค่า error แบบ smooth
            self.send_to_esp32(f"{self.last_error_x},{self.last_error_y}")
            print(f"Human Detected | Raw Error X: {error_x}, Y: {error_y} | Smooth Error X: {self.last_error_x}, Y: {self.last_error_y}")

        else:
            # ถ้าไม่เจอมนุษย์ ตรวจจับ AprilTag
            tag_detected, tag_id = self.detect_apriltag(frame) if self.detect_tag else (False, None)

            if tag_detected:
                self.no_target_count = 0
                self.send_to_esp32(f"999,{tag_id}")
                print(f"AprilTag Detected | ID: {tag_id} | send_to_esp32: 999,{tag_id}")
            else:
                if self.no_target_count < self.max_no_target_count:
                    self.no_target_count += 1

                if self.no_target_count < self.max_no_target_count:
                    # ค่อยๆ ลดค่า error ลงเมื่อไม่พบเป้าหมาย (decay factor)
                    decay_factor = 0.8  # ค่ามากกว่าจะลดช้าลง
                    if self.last_error_x is not None:
                        self.last_error_x = int(self.last_error_x * decay_factor)
                        self.last_error_y = int(self.last_error_y * decay_factor)
                    
                    self.send_to_esp32(f"{self.last_error_x},{self.last_error_y}")
                    print(f"No Target Detected ({self.no_target_count}/{self.max_no_target_count}) | Decaying errors: {self.last_error_x},{self.last_error_y}")
                else:
                    self.last_error_x = 0
                    self.last_error_y = 0
                    self.send_to_esp32("777,777")
                    print(f"No Target Detected ({self.no_target_count}/{self.max_no_target_count}) | send_to_esp32: 777,777")

        # แสดง FPS บนเฟรม
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

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

        print(f"\tFPS: {fps:.2f}")

    def __del__(self):
        if hasattr(self, 'vs'):
            self.vs.stop()
        if hasattr(self, 'pose'):
            self.pose.close()
        if hasattr(self, 'video_writer'):
            self.video_writer.release()


def main():
    parser = argparse.ArgumentParser(description="Human & AprilTag Tracking with MediaPipe")
    parser.add_argument('--hide', action='store_true', help="Hide the frame display")
    parser.add_argument('--port', type=str, default=None, help="Serial port for ESP32 (e.g., /dev/ttyUSB0)")
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