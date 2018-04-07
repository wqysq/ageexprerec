# ageexprerec
## Caffe

License

Caffe is a deep learning framework made with expression, speed, and modularity in mind. It is developed by Berkeley AI Research ([BAIR](http://bair.berkeley.edu/))/The Berkeley Vision and Learning Center (BVLC) and community contributors.

Check out the [project site ](http://caffe.berkeleyvision.org/)for all the details like

[DIY Deep Learning for Vision with Caffe](https://docs.google.com/presentation/d/1UeKXVgRvvxg9OUdh_UiC5G71UMscNPlvArsWER41PsU/edit#slide=id.p)

[Tutorial Documentation](http://caffe.berkeleyvision.org/tutorial/)

[BAIR reference models](http://caffe.berkeleyvision.org/model_zoo.html) and the [community model zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo)

[Installation instructions](http://caffe.berkeleyvision.org/installation.html)

and step-by-step examples.
## Windows Setup
Requirements: Visual Studio 2013,Dlib-18.17，OpenCV-2.4.10，CUDA 7.5.This caffe is from MS：[MicrosoftCaffe](https://github.com/Microsoft/caffe)

## Dlib
Download dlib-18.17 [from Baidu Yun](https://pan.baidu.com/s/1gey9Wd1),then you should cmake dlib generate dlib.lib.
In this project,I use dlib for face detection.

## OpenCV
Download OpenCV-2.4.10 [from OpenCV website](https://opencv.org/),in this project,I use OpenCV Mat read and write data.

## Build
 Open CommonSettings.props,channge: CpuOnlyBuild true or false,UseCuDnn true or false,CudaVersion.
 Now, you should be able to build .\windows\libcaffe,generate  libcaffe.lib.
## Test
modify classification project，add classification.h multi_recognition_gpu.h multi_recognition_gpu.cpp and multifenlei.cpp in classification project

## Result
![Result](https://github.com/wqysq/ageexprerec/classification/result.png)
