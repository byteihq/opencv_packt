#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

#ifdef IMAGE_READ
void image_read(const char *path)
{
    if (!path)
        return;

    cv::Mat color_img = cv::imread(path, cv::IMREAD_COLOR);
    cv::Mat gray_img = cv::imread(path, cv::IMREAD_GRAYSCALE);

    cv::imshow("Color BGR", color_img);
    cv::imshow("Gray", gray_img);

    if (!cv::imwrite("samples/gray_img.jpg", gray_img))
        throw std::runtime_error("Failed to save gray img");

    cv::waitKey(0);
}
#endif

#ifdef VIDEO_READ
inline void print_help_msg()
{
    std::cout << "\thelp h usage ?\t\tDisplay the help text\n\tvideo\t\t\tPath to video file. If empty try to use webcam" << std::endl;
}

inline void print_err_msg()
{
    std::cerr << "Invalid program args" << std::endl;;
}

void read_video(const cv::String &path)
{
    cv::VideoCapture cap;
    if (path.empty())
        cap.open(0);
    else
        cap.open(path);
    
    if (!cap.isOpened())
        throw std::runtime_error("Failed to open video");
    
    constexpr const char *window_name = "video";
    cv::namedWindow(window_name);
    cv::Mat frame;
    cv::FileStorage fs("info.yml", cv::FileStorage::FORMAT_YAML | cv::FileStorage::WRITE);
    cv::Mat mean;
    cv::Mat stddev;
    int counter = 0;
    for(;; ++counter, cv::imshow(window_name, frame))
    {
        cap >> frame;
        if (frame.empty() || cv::waitKey(30) >= 0)
            break;
        cv::meanStdDev(frame, mean, stddev);
        fs << "i" << counter;
        fs << "Mean" << mean;
        fs << "StdDev" << stddev;
    }

    fs.release();
}
#endif

int main(int argc, char **argv)
{
#ifdef IMAGE_READ
    constexpr const char *path = "samples/image.jpg";
    image_read(path);
#endif

#ifdef VIDEO_READ
    constexpr const char *keys = {"{help h usage ? | | Display the help text}"
                                  "{video | | Path to video file. If empty try to use webcam}"};

    cv::CommandLineParser parser(argc, argv, keys);
    if (parser.has("help"))
    {
        print_help_msg();
        return 0;
    }

    cv::String video_path = parser.get<cv::String>("video");

    if (!parser.check())
    {
        print_err_msg();
        return -1;
    }

    read_video(video_path);
#endif
    return 0;
}
