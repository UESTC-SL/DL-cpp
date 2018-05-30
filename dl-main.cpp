#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;


class ParallelConv : public cv::ParallelLoopBody {
public:
    ParallelConv (vector<cv::Mat> &img, vector<cv::Mat> &kernel, vector<cv::Mat> &result)
        : m_img(img), m_ker(kernel), m_res(result) {
    }

    virtual void operator ()(const cv::Range& range) const {
        for (int r = range.start; r < range.end; r++) {
            cv::filter2D(m_img[int(r/16)], m_res[int(r/16)], -1, m_ker[r]);
        }
    }

    ParallelConv& operator=(const ParallelConv &) {
        return *this;
    };

private:
    vector<cv::Mat> &m_img;
    vector<cv::Mat> &m_ker;
    vector<cv::Mat> &m_res;
};


int main() {

    vector<cv::Mat> img(16, cv::Mat(8, 8, CV_64F));
    for (int i = 0; i < 16; i++) {
        cv::randu(img[i], 0., 1.);
    }
    vector<cv::Mat> result(16, cv::Mat(8, 8, CV_64F));
    vector<cv::Mat> kernel(256, cv::Mat(5, 5, CV_64F));
    for (int i = 0; i < 256; i++) {
        cv::randu(kernel[i], 0., 0.1);
    }

    double t1 = (double) cv::getTickCount();

    ParallelConv parallelConv(img, kernel, result);
    for (int i = 0; i < 5000; i++) {
        cv::parallel_for_(cv::Range(0, 256), parallelConv);
    }

    t1 = ((double) cv::getTickCount() - t1) / cv::getTickFrequency();
    cout << "Elapsed time: " << t1 << " s" << endl;

    // cv::imshow("IMG", img);
    // cv::waitKey(0);

    return EXIT_SUCCESS;
}

