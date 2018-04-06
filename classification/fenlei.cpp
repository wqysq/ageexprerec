// calssify.cpp : �������̨Ӧ�ó������ڵ㡣
//
#include <iostream>
#include <string>
//#include <opencv2/imgproc/imgproc.hpp>
//#include <opencv2/core/core.hpp>
//#include <opencv2/highgui/highgui.hpp>
#include "multi_recognition_gpu.h"
#include <dlib/opencv.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing.h>


#ifdef _DEBUG
#define new DEBUG_NEW
#endif

using namespace cv;
using namespace std;
using namespace dlib;

string model_file_shape = "./data/shape_predictor_68_face_landmarks.dat";
std::string model_file("./Model/deploy_gender.prototxt");
std::string trained_file("./Model/gender_net.caffemodel");
std::string mean_file("./Model/mean.binaryproto");
std::string label_file("./Model/gender.txt");

std::string model_file_age("./Model/deploy_age.prototxt"); 
std::string trained_file_age("./Model/age_net.caffemodel"); 
std::string label_file_age("./Model/age.txt"); 

std::string model_file_emo("./Model/VGG_S_rgb/deploy.prototxt");
std::string trained_file_emo("./Model/VGG_S_rgb/EmotiW_VGG_S.caffemodel");
std::string label_file_emo("./Model/VGG_S_rgb/emotion.txt");
std::string mean_file_emo("./Model/VGG_S_rgb/mean.binaryproto");

//string trained_file_face = "./Model/vgg_face_caffe/VGG_FACE.caffemodel";
//string model_file_face = "./Model/vgg_face_caffe/VGG_FACE_deploy.prototxt ";
//string label_file_face = "./Model/vgg_face_caffe/names.txt ";
//string mean_file_face = mean_file_emo;
char*  strLogDir = "./log/";

bool DeleteDirectory(char* strDirName);
void forcerectinimg(Mat& I, Rect& rect);

int main(int argc, char** argv)
{

	//MultiClassifier myclassifier(model_file, trained_file, mean_file);//, label_file);//, label_file);

	unsigned int nLastCropWnds = 0; //���浱ǰ��ʾcropͼ���С���ڸ���(�����жϹرն���Ĵ���)

	//-----------------------------------------------------------------------------
	//�����������͹ؼ�����ѵ������
	printf("Loading face detector training data ..... ");
	frontal_face_detector detector = get_frontal_face_detector();
	printf("Done!\n");
	printf("Loading face shape detector training data ..... ");
	shape_predictor pose_model;
	deserialize(model_file_shape) >> pose_model;
	printf("Done!\n");

	//������������ͱ�ǩʶ��ѵ������
	//caffe::GlobalInit(&argc, &argv);
	printf("Loading Gender recog training data ..... ");
	MultiClassifier myclassifier(model_file, trained_file, mean_file,label_file);
	printf("Done!\n");
	printf("Loading Age recog training data ..... ");
	MultiClassifier myclassifier_age(model_file_age, trained_file_age, mean_file,label_file_age);
	printf("Done!\n");
	printf("Loading Age recog training data ..... ");
	MultiClassifier myclassifier_emo(model_file_emo, trained_file_emo, mean_file_emo, label_file_emo);
	printf("Done!\n");
	//��������ʶ��ѵ������
	//printf("Loading Face recog training data ..... ");
	//MultiClassifier myclassifier_face(model_file_face, trained_file_face, mean_file_face, label_file_face);
	//string layer_name = "fc7";
	////Mat img1 = imread("E:\\something\\data\\lfw_funneled\\Adrian_Nastase\\Adrian_Nastase_0001.jpg");
	//Mat img1 = imread("D:\\caffe-windows-master\\one1.jpg");
	//Mat img2 = imread("D:\\caffe-windows-master\\two1.jpg");
	///*Mat img2 = imread("E:\\something\\data\\lfw_funneled\\Aaron_Eckhart\\Aaron_Eckhart_0001.jpg");*/
	//std::vector<float>feature1 = myclassifier_face.get_layer_feature(img1, layer_name);
	//std::vector<float>feature2 = myclassifier_face.get_layer_feature(img2, layer_name);

	//-----------------------------------------------------------------------------
	printf("Initialize camera and to recog..... \n");
	int camindex = 0;
	VideoCapture capture(camindex);
	if (!capture.isOpened())
	{
		printf("ERROR: Capture from CAM [%d] fail!\n", camindex);
		return false;
	}
	Mat frame, I;
	unsigned long nframe = 0;
	while (1)
	{

		//ץȡ��Ƶ֡
		nframe++;
		if (!capture.read(frame))//capture >> frame; 
		{
			printf("ERROR: Unable to read next frame!\n");
			return false;
		}
		I = frame.clone();

		//-----------------------------------------------------
		//������⼰�沿�����㶨λ
		std::vector<Rect> vecFaceRects;
		std::vector<std::vector<Point>> vec2dFacePts;
		{
			//cv_image<bgr_pixel> cimg(I);//��opencv��matͼ��ṹ��װΪdlibͼ��ṹ(����������)
			cv_image<bgr_pixel> cimg(I);
			//�������
			std::vector<dlib::rectangle> vec_faces = detector(cimg);
			//�沿�����㶨λ
			std::vector<full_object_detection> shapes;
			for (unsigned long i = 0; i < vec_faces.size(); ++i)
				shapes.push_back(pose_model(cimg, vec_faces[i]));

			//�������ת��Ϊopencv��ʽ
			for (unsigned int i = 0; i < vec_faces.size(); i++)
			{
				Rect faceRect(vec_faces[i].left(), vec_faces[i].top(), vec_faces[i].width(), vec_faces[i].height());
				vecFaceRects.push_back(faceRect);

				std::vector<Point> vecFacePts;
				const full_object_detection& d = shapes[i];
				for (unsigned int j = 0; j < 68; j++)
					vecFacePts.push_back(Point(d.part(j)(0), d.part(j)(1)));
				vec2dFacePts.push_back(vecFacePts);
			}
		}

		//�����������μ�������
		for (unsigned int i = 0; i < vecFaceRects.size(); i++)
		{
			//������������
			cv::rectangle(I, vecFaceRects[i], CV_RGB(255, 0, 0), 1);
			//�����沿������
			//for (unsigned int j = 0; j<vec2dFacePts.size(); j++)
			//{
			//	circle(I, vec2dFacePts[i][j], 2, CV_RGB(0, 0, 255));
			//	//��ʾ����������
			//	char buffer[20];
			//	_itoa_s(j,buffer,10);
			//	string text(buffer);
			//	putText(I,text,pt,FONT_HERSHEY_SIMPLEX,0.3,CV_RGB(0,0,0));
			//}
			//�����沿����
			std::vector<Point> vecFacePts = vec2dFacePts[i];
			for (unsigned long j = 1; j <= 16; ++j)
				line(I, vecFacePts[j], vecFacePts[j - 1], CV_RGB(0, 255, 0));
			for (unsigned long j = 28; j <= 30; ++j)
				line(I, vecFacePts[j], vecFacePts[j - 1], CV_RGB(0, 255, 0));
			for (unsigned long j = 18; j <= 21; ++j)
				line(I, vecFacePts[j], vecFacePts[j - 1], CV_RGB(0, 255, 0));
			for (unsigned long j = 23; j <= 26; ++j)
				line(I, vecFacePts[j], vecFacePts[j - 1], CV_RGB(0, 255, 0));
			for (unsigned long j = 31; j <= 35; ++j)
				line(I, vecFacePts[j], vecFacePts[j - 1], CV_RGB(0, 255, 0));
			line(I, vecFacePts[30], vecFacePts[35], CV_RGB(0, 255, 0));
			for (unsigned long j = 37; j <= 41; ++j)
				line(I, vecFacePts[j], vecFacePts[j - 1], CV_RGB(0, 255, 0));
			line(I, vecFacePts[36], vecFacePts[41], CV_RGB(0, 255, 0));
			for (unsigned long j = 43; j <= 47; ++j)
				line(I, vecFacePts[j], vecFacePts[j - 1], CV_RGB(0, 255, 0));
			line(I, vecFacePts[42], vecFacePts[47], CV_RGB(0, 255, 0));
			for (unsigned long j = 49; j <= 59; ++j)
				line(I, vecFacePts[j], vecFacePts[j - 1], CV_RGB(0, 255, 0));
			line(I, vecFacePts[48], vecFacePts[59], CV_RGB(0, 255, 0));
			for (unsigned long j = 61; j <= 67; ++j)
				line(I, vecFacePts[j], vecFacePts[j - 1], CV_RGB(0, 255, 0));
			line(I, vecFacePts[60], vecFacePts[67], CV_RGB(0, 255, 0));
		}


		//-----------------------------------------------------------------------------
		//������һ����ü�
		std::vector<Mat> vecIcrop;
		for (unsigned int i = 0; i < vecFaceRects.size(); i++)
		{
			Mat I = frame;
			Mat I_crop;
			std::vector<Point> vecFacePts = vec2dFacePts[i];

			Point2d pt_eye_left(0, 0), pt_eye_right(0, 0);
			for (unsigned int i = 42; i <= 47; i++)
			{
				pt_eye_left.x += vecFacePts[i].x;
				pt_eye_left.y += vecFacePts[i].y;
			}
			pt_eye_left.x = cvRound(pt_eye_left.x / 6.0);
			pt_eye_left.y = cvRound(pt_eye_left.y / 6.0);
			for (unsigned int i = 36; i <= 41; i++)
			{
				pt_eye_right.x += vecFacePts[i].x;
				pt_eye_right.y += vecFacePts[i].y;
			}
			pt_eye_right.x = cvRound(pt_eye_right.x / 6.0);
			pt_eye_right.y = cvRound(pt_eye_right.y / 6.0);

			//�������۵����۵ķ���ʸ��
			Point2d pt_eyedir = pt_eye_left - pt_eye_right;
			//�������۵����۵ľ��벢������ų߶�
			float f_dis = (float)norm(pt_eyedir);
			float f_scale = float(70.0 / f_dis);//70
			//������ת�Ƕ�
			float f_rotangle = (float)atan2(pt_eyedir.y, pt_eyedir.x);//����y������,pt_eyedir.yǰӦȡ-
			f_rotangle = float(f_rotangle / CV_PI * 180);
			//��������ת����ͼ��,ʹ˫��ˮƽ��˫�۾���Ϊ70
			//1.�������Ʊ任����(angle, rotation center-��ʱ��, scale)
			Mat mat_rotscale = getRotationMatrix2D(pt_eye_right, f_rotangle, f_scale);
			//2.��תͼ��
			Mat I_rotscale;
			warpAffine(I, I_rotscale, mat_rotscale, I.size(), INTER_LINEAR, BORDER_CONSTANT, Scalar::all(0));
			//�ü�ͼ��
			Rect rec_crop;
			rec_crop.x = cvRound(pt_eye_right.x - 79);	rec_crop.y = cvRound(pt_eye_right.y - 79);
			rec_crop.width = 227;						rec_crop.height = 227;
			forcerectinimg(I_rotscale, rec_crop);
			I_crop = I_rotscale(rec_crop).clone();
			if (I_crop.rows != 227 || I_crop.cols != 227) resize(I_crop, I_crop, Size(227, 227));

			//ͼ��Ԥ�����todo

			vecIcrop.push_back(I_crop);
		}

		//-----------------------------------------------------------------------------
		//ʶ���ж�
		for (unsigned int i = 0; i < vecIcrop.size(); i++)
		{
			Mat I_crop = vecIcrop[i];

			//�Ա�ʶ��
			double t1 = (double)cvGetTickCount();
			std::vector<Prediction> predictions_gender = myclassifier.Classify(I_crop, 1);
			t1 = (double)cvGetTickCount() - t1;
			//����ʶ��
			double t2 = (double)cvGetTickCount();
			std::vector<Prediction> predictions_age = myclassifier_age.Classify(I_crop, 8);
			t2 = (double)cvGetTickCount() - t2;
			//����ʶ��
			double t3 = (double)cvGetTickCount();
			std::vector<Prediction> predictions_emo = myclassifier_emo.Classify(I_crop, 10);
			t3 = (double)cvGetTickCount() - t3;
			////����ʶ��
			//double t4 = (double)cvGetTickCount();
			//std::vector<Prediction> predictions_face = myclassifier_face.Classify(I_crop, 10);
			//t4 = (double)cvGetTickCount() - t3;

			//��ӡʶ����
			printf("[%ld] - %d:\n", nframe, i);
			printf("\t>>Gender: %8s --> %.2f  (%.4f s)\n", predictions_gender[0].first, predictions_gender[0].second, t1 / ((double)cvGetTickFrequency()*1e+6));
			
			printf("\t>>Age   : %8s --> %.2f  (%.4f s)\n", predictions_age[0].first, predictions_age[0].second, t2 / ((double)cvGetTickFrequency()*1e+6));
			printf("\t        : %8s --> %.2f \n", predictions_age[1].first, predictions_age[1].second);

			printf("\t>>Emo   : %8s --> %.2f  (%.4f s)\n", predictions_emo[0].first, predictions_emo[0].second, t3 / ((double)cvGetTickFrequency()*1e+6));
			printf("\t        : %8s --> %.2f \n", predictions_emo[1].first, predictions_emo[1].second);

			//printf("\t>>Face   : %8s --> %.2f  (%.4f s)\n", predictions_face[0].first, predictions_face[0].second, t3 / ((double)cvGetTickFrequency()*1e+6));
			//printf("\t        : %8s --> %.2f \n", predictions_face[1].first, predictions_face[1].second);

			//��ʾʶ����
			char buffer1[20];
			sprintf(buffer1, "%d", i);
			if (!I_crop.empty()) imshow(buffer1, I_crop);

			char buffer2[100];
			/*sprintf(buffer2, "Face: %s (%.2f),%s (.2f)", predictions_face[0].first, predictions_face[0].second);
			putText(I, buffer2, Point(vecFaceRects[i].x, vecFaceRects[i].y - 15), FONT_HERSHEY_TRIPLEX, 0.4, CV_RGB(255, 0, 0));*/
			sprintf(buffer2, "Emo: %s (%.2f), %s (%.2f)", predictions_emo[0].first, predictions_emo[0].second, predictions_emo[1].first, predictions_emo[1].second);
			putText(I, buffer2, Point(vecFaceRects[i].x, vecFaceRects[i].y - 10), FONT_HERSHEY_TRIPLEX, 0.4, CV_RGB(255, 0, 0));
			sprintf(buffer2, "Age: %s (%.2f), %s (%.2f)", predictions_age[0].first, predictions_age[0].second, predictions_age[1].first, predictions_age[1].second);
			putText(I, buffer2, Point(vecFaceRects[i].x, vecFaceRects[i].y - 30), FONT_HERSHEY_TRIPLEX, 0.4, CV_RGB(255, 0, 0));
			sprintf(buffer2, "Gender: %s (%.2f)", predictions_gender[0].first, predictions_gender[0].second);
			putText(I, buffer2, Point(vecFaceRects[i].x, vecFaceRects[i].y - 50), FONT_HERSHEY_TRIPLEX, 0.4, CV_RGB(255, 0, 0));


		}
		printf("-----------------------------------------------------\n");

		imshow("output", I);
		if (waitKey(1) >= 0) break;

		//�رն���Ĵ���
		for (unsigned int i = nLastCropWnds; i > vecIcrop.size(); i--)
		{
			char buffer[20];
			_itoa_s(i - 1, buffer, 10);
			string text(buffer);
			destroyWindow(text);
		}
		nLastCropWnds = vecIcrop.size();

	}				
		////-----------------------------------------------------------------------------
		////���log�ļ���
		//printf("Removing all prevoious log data ..... ");
		//if (DeleteDirectory(strLogDir))
		//{
		//	printf("ERROR: DeleteDirectory() return false!\n");
		//}
		//printf("Done!\n");
		//printf("Program exit success!\n");

		return true;
}

////ɾ��ָ��Ŀ¼�����е������ļ�
//bool DeleteDirectory(char* strDirName)
//{
//	CFileFind tempFind;
//	char strTempFileFind[MAX_PATH];
//	sprintf(strTempFileFind, "%s//*.*", strDirName);
//
//	BOOL IsFinded = tempFind.FindFile(strTempFileFind);
//	while (IsFinded)
//	{
//		IsFinded = tempFind.FindNextFile();
//		if (!tempFind.IsDots())
//		{
//			char strFoundFileName[MAX_PATH];
//			strcpy(strFoundFileName, tempFind.GetFileName().GetBuffer(MAX_PATH));
//			if (tempFind.IsDirectory())
//			{
//				char strTempDir[MAX_PATH];
//				sprintf(strTempDir, "%s//%s", strDirName, strFoundFileName);
//				DeleteDirectory(strTempDir);
//			}
//			else
//			{
//				char strTempFileName[MAX_PATH];
//				sprintf(strTempFileName, "%s//%s", strDirName, strFoundFileName);
//				DeleteFile(strTempFileName);
//			}
//		}
//	}
//	tempFind.Close();
//
//	if (!RemoveDirectory(strDirName)) return FALSE;
//
//	return true;
//}

//����������������ͼ���ڲ�
//���ڱ���ü�ͼ��ʱ����Խ�����
void forcerectinimg(Mat& I, Rect& rect)
{
	if (rect.x < 0) rect.x = 0;	else if (rect.x + rect.width >= I.cols) rect.width = I.cols - rect.x - 1;
	if (rect.y < 0) rect.y = 0;	else if (rect.y + rect.height >= I.rows) rect.height = I.rows - rect.y - 1;
}