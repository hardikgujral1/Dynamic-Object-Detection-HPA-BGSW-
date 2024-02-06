#include <opencv2/opencv.hpp>
#include <fstream>

using namespace cv;
using namespace std;
using namespace cv::dnn;

const float INPUT_WIDTH = 640.0;
const float INPUT_HEIGHT = 640.0;
const float SCORE_THRESHOLD = 0.5;
const float NMS_THRESHOLD = 0.45;
const float CONFIDENCE_THRESHOLD = 0.45;

const float FONT_SCALE = 0.7;
const int FONT_FACE = FONT_HERSHEY_SIMPLEX;
const int THICKNESS = 1;

Scalar BLACK(0, 0, 0);
Scalar BLUE(255, 178, 50);
Scalar YELLOW(0, 255, 255);
Scalar RED(0, 0, 255);

void draw_label(Mat& input_image, const string& label, int left, int top)
{
    int baseLine;
    Size label_size = getTextSize(label, FONT_FACE, FONT_SCALE, THICKNESS, &baseLine);
    top = max(top, label_size.height);

    Point tlc = Point(left, top);
    Point brc = Point(left + label_size.width, top + label_size.height + baseLine);

    rectangle(input_image, tlc, brc, BLACK, FILLED);
    putText(input_image, label, Point(left, top + label_size.height), FONT_FACE, FONT_SCALE, YELLOW, THICKNESS);
}

vector<Mat> pre_process(const Mat& input_image, Net& net)
{
    Mat blob;
    blobFromImage(input_image, blob, 1.0 / 255.0, Size(INPUT_WIDTH, INPUT_HEIGHT), Scalar(), true, false);

    net.setInput(blob);

    vector<Mat> outputs;
    net.forward(outputs, net.getUnconnectedOutLayersNames());

    return outputs;
}

// Function to post-process the image
Mat post_process(const Mat& input_image, const vector<Mat>& outputs, const vector<string>& class_names)
{
    vector<int> class_ids;
    vector<float> confidences;
    vector<Rect> boxes;

    float x_factor = input_image.cols / INPUT_WIDTH;
    float y_factor = input_image.rows / INPUT_HEIGHT;

    const int dimensions = 85;
    const int rows = 25200;

    const float* data = outputs[0].ptr<float>();

    for (int i = 0; i < rows; ++i)
    {
        float confidence = data[4];

        if (confidence >= CONFIDENCE_THRESHOLD)
        {
            const float* classes_scores = data + 5;

            Mat scores(1, class_names.size(), CV_32FC1, const_cast<float*>(classes_scores));

            Point class_id;
            double max_class_score;
            minMaxLoc(scores, 0, &max_class_score, 0, &class_id);

            if (max_class_score > SCORE_THRESHOLD)
            {
                // Filter classes based on names (car, truck, person)
                string detected_class = class_names[class_id.x];
                if (detected_class == "car" || detected_class == "truck" || detected_class == "person")
                {
                    confidences.push_back(confidence);
                    class_ids.push_back(class_id.x);

                    float cx = data[0];
                    float cy = data[1];
                    float w = data[2];
                    float h = data[3];

                    int left = int((cx - 0.5 * w) * x_factor);
                    int top = int((cy - 0.5 * h) * y_factor);
                    int width = int(w * x_factor);
                    int height = int(h * y_factor);

                    boxes.push_back(Rect(left, top, width, height));
                }
            }
        }

        data += dimensions;
    }

    vector<int> indices;
    NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD, indices);

    for (int i = 0; i < indices.size(); i++)
    {
        int idx = indices[i];
        Rect box = boxes[idx];

        int left = box.x;
        int top = box.y;
        int width = box.width;
        int height = box.height;

        // Update the memory buffer with detected objects
        // Assuming input_image is the allocated buffer
        rectangle(const_cast<Mat&>(input_image), Point(left, top), Point(left + width, top + height), BLUE, 3 * THICKNESS);

        string label = format("%.2f", confidences[idx]);
        label = class_names[class_ids[idx]] + ":" + label;

        draw_label(const_cast<Mat&>(input_image), label, left, top);
    }

    // Returning the post-processed result
    return input_image.clone();
}

int main(int argc, char* argv[])
{
    if (argc != 3)
    {
        cout << "Usage: " << argv[0] << " <video_file_path> <base_address>" << endl;
        return -1;
    }

    // Get the base address from the command line
    void* baseAddress = (void*)strtoul(argv[2], NULL, 16);

    vector<string> class_list;
    ifstream ifs("Model/coco.names");
    string line;

    while (getline(ifs, line))
    {
        class_list.push_back(line);
    }

    VideoCapture cap(argv[1]);

    if (!cap.isOpened())
    {
        cout << "Error: Could not open video file or capture device." << endl;
        return -1;
    }

    int frame_width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int frame_height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    int frames_per_second = cap.get(cv::CAP_PROP_FPS);

    VideoWriter video_writer("Output File/output.avi", VideoWriter::fourcc('M', 'J', 'P', 'G'), frames_per_second, Size(frame_width * 2, frame_height));

    Net net;
    net = readNet("Model/yolov5n.onnx");

    Mat frame, processed_frame;
    while (cap.read(frame))
    {
        // Allocate memory using the provided base address
        size_t bufferSize = frame.total() * frame.elemSize();
        void* memoryBuffer = malloc(bufferSize);

        if (memoryBuffer == NULL)
        {
            cerr << "Memory allocation failed" << endl;
            return -1;
        }

        // Copy the image data to the allocated buffer
        memcpy(memoryBuffer, frame.data, bufferSize);

        // Create a cv::Mat from the allocated memory buffer
        Mat img(frame.size(), frame.type(), memoryBuffer);

        // Use the allocated memory buffer for object detection
        processed_frame = post_process(img, pre_process(frame, net), class_list);

        // Concatenate raw video and processed video frames horizontally
        Mat output_frame(frame.rows, frame.cols * 2, frame.type());
        frame.copyTo(output_frame(Rect(0, 0, frame.cols, frame.rows)));
        processed_frame.copyTo(output_frame(Rect(frame.cols, 0, frame.cols, frame.rows)));

        // Print the allocated memory address and buffer name
        cout << "Allocated memory address: " << memoryBuffer << " Buffer Name: " << "buffer_name_here" << endl;

        video_writer.write(output_frame);
        imshow("Object Detection", output_frame);

        // Don't forget to free the memory when done
        free(memoryBuffer);

        if (waitKey(1) == 27)
            break;
    }

    cap.release();
    video_writer.release();
    return 0;
}
