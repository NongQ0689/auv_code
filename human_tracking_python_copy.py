import cv2
import numpy as np
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
    def __init__(self, width, height, show_frame=True, save_video=False):
        self.width = width
        self.height = height
        self.show_frame = show_frame
        self.save_video = save_video

        print("Human Tracking Started")

        # Load the pre-trained MobileNet SSD model
        self.net = cv2.dnn.readNetFromCaffe(
            '/home/rpiauv-server/auv_code/deploy.prototxt', 
            '/home/rpiauv-server/auv_code/mobilenet_iter_73000.caffemodel'
        )

        # Define the class labels MobileNet SSD was trained on
        self.CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
                        "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                        "dog", "horse", "motorbike", "person", "pottedplant",
                        "sheep", "sofa", "train", "tvmonitor"]

        # Initialize the video stream
        self.vs = WebcamVideoStream(src=0, width=self.width, height=self.height).start()

        # Initialize video writer if saving video
        if self.save_video:
            current_time = str(int(time.time()))
            filename = f'output_{current_time}.avi'
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # Codec for video
            self.video_writer = cv2.VideoWriter(filename, fourcc, 10.0, (self.width, self.height))


    def track_humans(self):
        start_time = time.time()
        frame = self.vs.read()

        if frame is None:
            return

        # Prepare the frame for the neural network
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
        self.net.setInput(blob)

        # Perform forward pass to get the detections
        detections = self.net.forward()

        # Frame center coordinates
        frame_center_x = self.width // 2
        frame_center_y = self.height // 2

        # Draw the center point
        cv2.circle(frame, (frame_center_x, frame_center_y), 3, (0, 255, 0), -1)  # Green dot

        person_detected = False  # Initialize person detection flag

        # Loop over the detections
        for i in np.arange(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            # Filter out weak detections by ensuring the confidence is greater than a threshold
            if confidence > 0.2:
                idx = int(detections[0, 0, i, 1])
                if self.CLASSES[idx] == "person":
                    person_detected = True

                    box = detections[0, 0, i, 3:7] * np.array([self.width, self.height, self.width, self.height])
                    (startX, startY, endX, endY) = box.astype("int")

                    # Calculate position and errors
                    pos_x = (startX + endX) // 2
                    pos_y = startY
                    error_x = pos_x - frame_center_x
                    error_y = pos_y - frame_center_y

        # Display FPS in terminal
        end_time = time.time()
        fps = 1 / (end_time - start_time)

        if  person_detected:
            print(f"w:{self.width} - h:{self.height} | FPS: {fps:.2f} | pos_x:{pos_x} - pos_y:{pos_y} | error_x:{error_x} - error_y:{error_y}")
        else:
            print(f"w:{self.width} - h:{self.height} | FPS: {fps:.2f} | No person detected ! |")

        # Display the resulting frame
        if self.show_frame:
            cv2.imshow('Frame', frame)

            # If the 'q' key is pressed, stop the video stream and close the window
            if cv2.waitKey(20) & 0xFF == ord('q'):
                print("Shutting down...")
                self.vs.stop()  # Stop video stream
                cv2.destroyAllWindows()  # Close the window
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
    args = parser.parse_args()

    show_frame = not args.hide  # If --hide, will not display frame
    save_video = args.vdo  # If --vdo, will save and show video

    print(f"show_frame:{show_frame} | save_video:{save_video}")

    width, height = 320, 240  # Set width and height
    tracker = HumanTracking(width, height, show_frame=show_frame, save_video=save_video)

    while True:
        tracker.track_humans()


if __name__ == '__main__':
    main()

