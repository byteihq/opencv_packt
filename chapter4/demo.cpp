#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>
#include <cmath>

class Window
{
private:
    cv::String _name;

public:
    explicit Window(const cv::String &winname, int flags = 1) : _name(winname)
    {
        cv::namedWindow(winname, flags);
    }

    cv::String &get_name()
    {
        return _name;
    }

    const cv::String &get_name() const
    {
        return _name;
    }

    int show(const cv::Mat &img, int delay = 0)
    {
        cv::imshow(_name, img);
        return cv::waitKey(delay);
    }

    ~Window()
    {
        cv::destroyWindow(_name);
    }
};

class Button
{
private:
    cv::String _name;

public:
    Button(const cv::String &name, cv::ButtonCallback callback, void *userdata = nullptr, int type = 0, bool init_state = false) : _name(name)
    {
        cv::createButton(name, callback, userdata, type, init_state);
    }

    virtual ~Button() = default;
};

class Histogram
{
private:
    Button _show_btn;
    Button _equalize_btn;
    Button _lomograhy_button;
    const cv::Mat &_img;
    Window &_win;

    static void cv_show_callback(int state, void *userdata)
    {
        if (!userdata)
            return;

        const Histogram *hist = reinterpret_cast<Histogram *>(userdata);
        std::vector<cv::Mat> bgr;
        cv::split(hist->_img, bgr);

        constexpr int numbins = 256;
        const float range[] = {0, 256};
        const float *histRange = {range};

        cv::Mat b_hist, g_hist, r_hist;
        cv::calcHist(&bgr[0], 1, 0, cv::Mat(), b_hist, 1, &numbins, &histRange);
        cv::calcHist(&bgr[1], 1, 0, cv::Mat(), g_hist, 1, &numbins, &histRange);
        cv::calcHist(&bgr[2], 1, 0, cv::Mat(), r_hist, 1, &numbins, &histRange);

        constexpr int w = 512;
        constexpr int h = 300;
        cv::Mat out(h, w, CV_8UC3, cv::Scalar(20, 20, 20));

        cv::normalize(b_hist, b_hist, 0, h, cv::NORM_MINMAX);
        cv::normalize(g_hist, g_hist, 0, h, cv::NORM_MINMAX);
        cv::normalize(r_hist, r_hist, 0, h, cv::NORM_MINMAX);

        int step = cvRound((float)w / h);
        for (size_t i = 1; i < numbins; ++i)
        {
            cv::line(out, cv::Point(step * (i - 1), h - cvRound(b_hist.at<float>(i - 1))),
                     cv::Point(step * i, h - cvRound(b_hist.at<float>(i))), cv::Scalar(255, 0, 0));

            cv::line(out, cv::Point(step * (i - 1), h - cvRound(g_hist.at<float>(i - 1))),
                     cv::Point(step * i, h - cvRound(g_hist.at<float>(i))), cv::Scalar(0, 255, 0));

            cv::line(out, cv::Point(step * (i - 1), h - cvRound(r_hist.at<float>(i - 1))),
                     cv::Point(step * i, h - cvRound(r_hist.at<float>(i))), cv::Scalar(0, 0, 255));
        }

        hist->_win.show(out);
    }

    static void cv_equalize_callback(int state, void *userdata)
    {
        if (!userdata)
            return;

        const Histogram *hist = reinterpret_cast<Histogram *>(userdata);
        cv::Mat ycrcb;
        cv::cvtColor(hist->_img, ycrcb, cv::COLOR_BGR2YCrCb);

        std::vector<cv::Mat> channels;
        cv::split(ycrcb, channels);
        if (channels.empty())
            return;

        cv::equalizeHist(channels[0], channels[0]);
        cv::merge(channels, ycrcb);

        cv::Mat result;
        cv::cvtColor(ycrcb, result, cv::COLOR_YCrCb2BGR);

        hist->_win.show(result);
    }

    static void cv_lomography_callback(int state, void *userdata)
    {
        if (!userdata)
            return;

        const Histogram *hist = reinterpret_cast<Histogram *>(userdata);

        cv::Mat lut(1, 256, CV_8UC1);
        const double exp__1 = std::exp(-1.0);
        for (size_t  i = 0; i < 256; ++i)
            lut.at<uchar>(i) = cvRound(1 / (1 + std::pow(exp__1, ((i / 256.f) - 0.5) / 0.1)));
        
        std::vector<cv::Mat> bgr;
        cv::split(hist->_img, bgr);

        cv::LUT(bgr[0], lut, bgr[0]);

        cv::Mat result;
        cv::merge(bgr, result);

        cv::Mat halo(hist->_img.rows, hist->_img.cols, CV_32FC3, cv::Scalar(0.3, 0.3, 0.3));
        cv::circle(halo, cv::Point(hist->_img.cols / 2, hist->_img.rows / 2), hist->_img.cols / 3, cv::Scalar(1, 1, 1), -1);
        cv::blur(halo, halo, cv::Size(hist->_img.cols / 3, hist->_img.rows / 3));

        cv::Mat resultf;
        result.convertTo(resultf, CV_32FC3);
        cv::multiply(resultf, halo, resultf);
        resultf.convertTo(result, CV_8UC3);

        hist->_win.show(result);
    }

public:
    Histogram(const cv::Mat &img, Window &win) : _show_btn("Show histogram", Histogram::cv_show_callback, this, cv::QT_PUSH_BUTTON), _equalize_btn("Equalize histogram", Histogram::cv_equalize_callback, this, cv::QT_PUSH_BUTTON), _lomograhy_button("Lomography", Histogram::cv_lomography_callback, this, cv::QT_PUSH_BUTTON), _img(img), _win(win) {}
};

inline void print_help()
{
    std::cout << "\thelp h usage ?\tDisplay the help text\n\timage\t\tPath to img" << std::endl;
}

cv::Mat open_img(const cv::String &path)
{
    if (path.empty())
        return {};

    return cv::imread(path, cv::IMREAD_UNCHANGED);
}

int main(int argc, char **argv)
{
    try
    {
        cv::Mat img;
        {
            constexpr const char *keys =
                {
                    "{help h usage ? | | Display the help text}"
                    "{image | | Path to img }"};
            cv::CommandLineParser parser(argc, argv, keys);

            if (parser.has("help"))
            {
                print_help();
                return 0;
            }

            auto path = parser.get<cv::String>("image");
            if (!parser.check())
                throw std::runtime_error("Invalid path to img");

            img = open_img(path);
            if (!img.data)
                throw std::runtime_error("Failed to open img");
        }
        Window w("Input");
        Window hist_w("Histogram");
        Histogram h(img, hist_w);
        w.show(img);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what();
    }

    return 0;
}