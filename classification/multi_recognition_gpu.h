#ifndef MULTI_RECOGNITION_GPU_H_
#define MULTI_RECOGNITION_GPU_H_

#ifdef MULTI_RECOGNITION_API_EXPORTS
#define MULTI_RECOGNITION_API __declspec(dllexport)
#else
#define MULTI_RECOGNITION_API __declspec(dllimport)
#endif
#define  _AFXDLL
#endif
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <string>
#include <vector>
#include <iostream>
#include <io.h>
class ClassifierImpl;
using std::string;
using std::vector;
//typedef std::pair<int, float> Prediction;
typedef std::pair<string, float> Prediction;

class MULTI_RECOGNITION_API MultiClassifier
{
public:
	MultiClassifier(const string& model_file,
		const string& trained_file,
		const string& mean_file,
		const string& label_file);

	~MultiClassifier();
	vector<Prediction> Classify(const cv::Mat& img, int N = 2);
	void getFiles(std::string path, std::vector<std::string>& files);
private:
	ClassifierImpl *Impl;
};
