#ifndef CLASSIFICATION_H_
#define CLASSIFICATION_H_

#include <caffe/caffe.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iosfwd>
#include <memory>
#include <utility>
#include <vector>
#include <iostream>
#include <string>
#include <time.h>

using namespace caffe;
using std::string;
//typedef std::pair<int, float> Prediction;
typedef std::pair<string, float> Prediction;


class  ClassifierImpl {
public:
	ClassifierImpl(const string& model_file,
		const string& trained_file,
		const string& mean_file,
		const string& label_file
		);

	std::vector<Prediction>  Classify(const cv::Mat& img, int N = 2);
private:
	void SetMean(const string& mean_file);

	std::vector<float> Predict(const cv::Mat& img);

	void WrapInputLayer(std::vector<cv::Mat>* input_channels);

	void Preprocess(const cv::Mat& img,
		std::vector<cv::Mat>* input_channels);

private:
	shared_ptr<Net<float> > net_;
	cv::Size input_geometry_;
	int num_channels_;
	cv::Mat mean_;
	std::vector<string> labels_;
};
#endif