#include "multi_recognition_gpu.h"
#include "classification.h"


MultiClassifier::MultiClassifier(const string& model_file, const string& trained_file, const string& mean_file,const string& label_file)
{
	Impl = new ClassifierImpl(model_file, trained_file, mean_file, label_file);
}
MultiClassifier::~MultiClassifier()
{
	//delete Impl;//���������ͷź����ַ��ʿ�ָ��
}
std::vector<Prediction>  MultiClassifier::Classify(const cv::Mat& img, int N /* = 2 */)
{
	return Impl->Classify(img, N);
}