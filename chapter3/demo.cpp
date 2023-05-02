#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

#include <GL/gl.h>

bool loadTexture(const cv::Mat &frame, GLuint texture)
{
    if (frame.empty())
        return false;

    glBindTexture(GL_TEXTURE_2D, texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, frame.cols, frame.rows, 0, GL_BGR, GL_UNSIGNED_BYTE, frame.data);
    return true;
}

class OpenGL
{
private:
    GLuint _texture;
    GLfloat _angle;

    static void cv_callback(void *userdata)
    {
        if (!userdata)
            return;
        OpenGL *data = reinterpret_cast<OpenGL *>(userdata);

        glLoadIdentity();
        glBindTexture(GL_TEXTURE_2D, data->_texture);
        glRotatef(data->_angle, 1.0f, 1.0f, 1.0f);
        glBegin(GL_QUADS);

        glTexCoord2d(0.0, 0.0);
        glVertex2d(-1.0, -1.0);

        glTexCoord2d(1.0, 0.0);
        glVertex2d(1.0, -1.0);

        glTexCoord2d(1.0, 1.0);
        glVertex2d(1.0, 1.0);

        glTexCoord2d(0.0, 1.0);
        glVertex2d(-1.0, 1.0);

        glEnd();
    }

public:
    OpenGL(const cv::String &winname, GLuint texture, GLuint angle) : _texture(texture), _angle(angle)
    {
        cv::setOpenGlDrawCallback(winname, OpenGL::cv_callback, this);
    }

    GLuint get_texture() const noexcept
    {
        return _texture;
    }

    GLfloat get_angle() const noexcept
    {
        return _angle;
    }

    void set_angle(GLfloat angle) noexcept
    {
        if (angle > 360)
            angle -= 360;
        _angle = angle;
    }
};

int main()
{
    cv::VideoCapture cap(0);

    constexpr const char *winname = "OpenGL Camera";
    cv::namedWindow(winname, cv::WINDOW_OPENGL);

    GLuint texture;
    GLfloat angle = 0;

    glEnable(GL_TEXTURE_2D);
    glGenTextures(1, &texture);
    OpenGL gl(winname, texture, angle);

    cv::Mat frame;
    while (cv::waitKey(10) != 'q')
    {
        cap >> frame;
        loadTexture(frame, gl.get_texture());
        cv::updateWindow(winname);

        gl.set_angle(gl.get_angle() + 4);
    }

    cv::destroyAllWindows();
    return 0;
}
