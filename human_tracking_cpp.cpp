#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <thread>
#include <iostream>
#include <chrono>

using namespace cv;
using namespace cv::dnn;
using namespace std;

class WebcamVideoStream {
public:
    WebcamVideoStream(int src, int width, int height) : width(width), height(height) {
        stream.open(src);
        stream.set(CAP_PROP_FRAME_WIDTH, width);
        stream.set(CAP_PROP_FRAME_HEIGHT, height);
        stopped = false;
        stream >> frame;
    }

    void start() {
        thread(&WebcamVideoStream::update, this).detach();
    }

    void update() {
        while (true) {
            if (stopped) return;
            stream >> frame;
        }
    }

    Mat read() {
        return frame;
    }

    void stop() {
        stopped = true;
    }

private:
    VideoCapture stream;
    Mat frame;
    int width, height;
    bool stopped;
};

class HumanTracking {
public:
    HumanTracking(int width, int height, bool show_frame = true, bool save_video = false) 
        : width(width), height(height), show_frame(show_frame), save_video(save_video) {

        // Load the pre-trained MobileNet SSD model
        net = readNetFromCaffe("deploy.prototxt", "mobilenet_iter_73000.caffemodel");

        // Define the class labels MobileNet SSD was trained on
        CLASSES = { "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", 
                    "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor" };

        // Initialize the video stream
        vs = new WebcamVideoStream(0, width, height);
        vs->start();

        // Initialize video writer if saving video
        if (save_video) {
            string filename = "output_" + to_string(chrono::system_clock::to_time_t(chrono::system_clock::now())) + ".avi";
            video_writer.open(filename, VideoWriter::fourcc('M', 'J', 'P', 'G'), 10.0, Size(width, height));
        }
    }

    void track_humans() {
        auto start_time = chrono::high_resolution_clock::now();
        Mat frame = vs->read();

        if (frame.empty()) {
            return;
        }

        // Prepare the frame for the neural network
        Mat blob;
        blobFromImage(frame, blob, 0.007843, Size(300, 300), Scalar(127.5, 127.5, 127.5));
        net.setInput(blob);

        // Perform forward pass to get the detections
        Mat detections = net.forward();

        // Frame center coordinates
        int frame_center_x = width / 2;
        int frame_center_y = height / 2;

        // Draw the center point
        circle(frame, Point(frame_center_x, frame_center_y), 3, Scalar(0, 255, 0), -1);  // Green dot

        bool person_detected = false;  // Initialize person detection flag

        int pos_x = 0;
        int pos_y = 0;
        int error_x = 0;
        int error_y = 0;

        // Loop over the detections
        for (int i = 0; i < detections.size[2]; i++) {
            float confidence = detections.ptr<float>(0, 0)[i * detections.size[3] + 2];

            // Filter out weak detections by ensuring the confidence is greater than a threshold
            if (confidence > 0.2) {
                int idx = static_cast<int>(detections.ptr<float>(0, 0)[i * detections.size[3] + 1]);
                if (CLASSES[idx] == "person") {
                    person_detected = true;

                    Rect box = Rect(
                        static_cast<int>(detections.ptr<float>(0, 0)[i * detections.size[3] + 3] * width),
                        static_cast<int>(detections.ptr<float>(0, 0)[i * detections.size[3] + 4] * height),
                        static_cast<int>(detections.ptr<float>(0, 0)[i * detections.size[3] + 5] * width) - static_cast<int>(detections.ptr<float>(0, 0)[i * detections.size[3] + 3] * width),
                        static_cast<int>(detections.ptr<float>(0, 0)[i * detections.size[3] + 6] * height) - static_cast<int>(detections.ptr<float>(0, 0)[i * detections.size[3] + 4] * height)
                    );

                    // Calculate position and errors
                    pos_x = (box.x + box.x + box.width) / 2;
                    pos_y = box.y;
                    error_x = pos_x - frame_center_x;
                    error_y = pos_y - frame_center_y;


                    // Draw the bounding box
                    string label = format("%s: %.2f", CLASSES[idx].c_str(), confidence);
                    rectangle(frame, box, Scalar(0, 255, 0), 2);
                    putText(frame, label, Point(box.x, max(box.y - 15, 15)), FONT_HERSHEY_SIMPLEX, 0.4, Scalar(0, 255, 0), 1);

                    // Draw error lines
                    line(frame, Point(frame_center_x, frame_center_y), Point(pos_x, frame_center_y), Scalar(0, 0, 255), 1);  // X-axis (red)
                    line(frame, Point(pos_x, pos_y), Point(pos_x, frame_center_y), Scalar(255, 0, 0), 1);  // Y-axis (blue)

                    // Display error values on the frame
                    putText(frame, format("Error X: %d", error_x), Point(10, 20), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255), 1);
                    putText(frame, format("Error Y: %d", error_y), Point(10, 40), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 0, 0), 1);
                }
            }
        }

        // Display FPS in terminal
        auto end_time = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::milliseconds>(end_time - start_time).count();
        double fps = 1000.0 / duration;

        if (person_detected) {
            cout << "w:" << width << " - h:" << height << " | FPS: " << fps << " | pos_x:" << pos_x << " - pos_y:" << pos_y << " | error_x:" << error_x << " - error_y:" << error_y << endl;

        }else
        {
            cout << "w:" << width << " - h:" << height << " | FPS: " << fps << " | No person detected!" << endl;
        }

        // Display the resulting frame
        if (show_frame) {
            imshow("Frame", frame);

            // If the 'q' key is pressed, stop the video stream and close the window
            if (waitKey(20) == 'q') {
                cout << "Shutting down..." << endl;
                vs->stop();
                destroyAllWindows();
                exit(0);
            }
        }

        // If we are saving video, write the frame to the video file
        if (save_video) {
            video_writer.write(frame);
        }
    }

    ~HumanTracking() {
        if (save_video) {
            video_writer.release();  // Release the video writer
        }
    }

private:
    int width, height;
    bool show_frame, save_video;
    vector<string> CLASSES;
    Net net;
    WebcamVideoStream* vs;
    VideoWriter video_writer;
};

int main(int argc, char** argv) {
    bool show_frame = true;
    bool save_video = false;

    // Parse command line arguments
    if (argc > 1) {
        for (int i = 1; i < argc; i++) {
            string arg = argv[i];
            if (arg == "--hide") {
                show_frame = false;
            } else if (arg == "--vdo") {
                save_video = true;
            }
        }
    }

    int width = 320, height = 240;  // Set width and height
    HumanTracking tracker(width, height, show_frame, save_video);

    while (true) {
        tracker.track_humans();
    }

    return 0;
}
