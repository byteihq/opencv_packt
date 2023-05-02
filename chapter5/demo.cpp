#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>

class Image final
{
private:
    cv::Mat _img;

public:
    Image() = default;

    Image(const Image &) = default;

    Image &operator=(const Image &) = default;

    explicit Image(const cv::String &path, int flags = 1)
    {
        if (!open(path, flags))
            throw std::runtime_error("Failed to open img");
    }

    explicit Image(const cv::Mat &mat): _img(mat) {}

    bool open(const cv::String &path, int flags = 1)
    {
        _img = cv::imread(path, flags);
        return (_img.data != nullptr);
    }

    cv::Mat &operator()()
    {
        return _img;
    }

    const cv::Mat &operator()() const
    {
        return _img;
    }

    ~Image() = default;
};

class BaseFilter
{

protected:
    enum class Tag
    {
        UNSPECIFIED,
        MEDIAN,
        LIGHT,
        BINARY
    };

private:
    Tag _tag;

public:
    BaseFilter() = delete;

    BaseFilter(Tag tag) : _tag(tag) {}

    Tag tag() const
    {
        return _tag;
    }

    std::string name() const
    {
        switch (_tag)
        {
        case Tag::MEDIAN:
            return "median";
        case Tag::LIGHT:
            return "light";
        case Tag::BINARY:
            return "binary";
        case Tag::UNSPECIFIED:
        default:
            return "unspecified";
        }
    }

    virtual Image operator()() = 0;

    virtual ~BaseFilter() = default;
};

class MedianFilter final : public BaseFilter
{
private:
    Image _img;
    int _kernel_size;

public:
    MedianFilter(Image img, int kernel_size = 3) : BaseFilter(BaseFilter::Tag::MEDIAN), _img(img), _kernel_size(kernel_size) {}

    Image operator()() override
    {
        Image out;
        cv::medianBlur(_img(), out(), _kernel_size);

        return out;
    }

    ~MedianFilter() = default;
};

class LightFilter final : public BaseFilter
{
public:
    enum class Method
    {
        DIV,
        SUB
    };

private:
    Image _img;
    Image _pattern_img;
    Method _method;

public:
    LightFilter(Image img, Image pattern_img, Method method = Method::DIV) : BaseFilter(BaseFilter::Tag::LIGHT), _img(img), _pattern_img(pattern_img), _method(method) {}

    Image operator()() override
    {
        Image result;
        switch (_method)
        {
        case Method::DIV:
        {
            cv::Mat img32;
            cv::Mat pattern32;
            _img().convertTo(img32, CV_32F);
            _pattern_img().convertTo(pattern32, CV_32F);

            result() = 1 - (img32 / pattern32);
            result().convertTo(result(), CV_8U, 255);
            break;
        }
        case Method::SUB:
        {
            result() = _pattern_img() - _img();
            break;
        }
        default:
            throw std::runtime_error("Usupported method");
        }

        return result;
    }

    ~LightFilter() = default;
};

class BinaryFilter final : public BaseFilter
{
    Image _img;
    double _threshold;
    double _max_value;
    int _type;
public:
    BinaryFilter(Image img, double threshold, double max_value, int type): BaseFilter(BaseFilter::Tag::BINARY), _img(img), _threshold(threshold), _max_value(max_value), _type(type) {}

    Image operator()() override
    {
        Image res;
        cv::threshold(_img(), res(), _threshold, _max_value, _type);

        return res;
    }

    ~BinaryFilter() = default;
};

void connected_components(Image img, int area_threshold = 1000)
{
    Image gray;
    cv::cvtColor(img(), gray(), cv::COLOR_BGR2GRAY);
    cv::Mat labels;
    cv::Mat stats;
    cv::Mat centroids;
    auto num_objects = cv::connectedComponentsWithStats(gray(), labels, stats, centroids);
    if (num_objects < 2)
        return;
    Image res(cv::Mat::zeros(img().rows, img().cols, CV_8UC3));
    cv::RNG rng(0xFFFFFFFF);
    for (size_t i = 1; i < num_objects; ++i)
    {
        auto area = stats.at<int>(i, cv::CC_STAT_AREA);
        if (area < area_threshold)
            continue;
        res().setTo(cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255)), cv::Mat(labels == i));
        cv::putText(res(), std::to_string(area), centroids.at<cv::Point2d>(i), cv::FONT_HERSHEY_COMPLEX, 0.4, cv::Scalar(255, 255, 255));
    }
    cv::imshow("Result", res());
    cv::waitKey(0);
}

inline void print_help()
{
    std::cout << "\thelp h usage ?\tDisplay the help text\n\timage\t\tPath to img" << std::endl;
}

int main(int argc, char **argv)
{
    try
    {
        Image img;
        Image pattern_img;
        {
            constexpr const char *keys =
                {
                    "{help h usage ? | | Display the help text}"
                    "{image | | Path to img }"
                    "{pattern | | Path to pattern img}"};
            cv::CommandLineParser parser(argc, argv, keys);
            if (parser.has("help"))
            {
                print_help();
                return 0;
            }

            const cv::String img_path = parser.get<cv::String>("image");
            const cv::String pattern_img_path = parser.get<cv::String>("pattern");
            if (!parser.check())
                throw std::runtime_error("Invalid params");

            if (!img.open(img_path) || !pattern_img.open(pattern_img_path))
                throw std::runtime_error("Failed to open img");
        }

        MedianFilter m(img);
        auto filtered_img = m();

        LightFilter ls(filtered_img, pattern_img, LightFilter::Method::SUB);
        auto light_sub = ls();

        BinaryFilter b(light_sub, 30, 255, cv::THRESH_BINARY);
        auto bin = b();

        connected_components(bin);
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what();
    }
    return 0;
}
