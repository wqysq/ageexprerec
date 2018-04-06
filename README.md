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

## Build
 Open CommonSettings.props,channge: CpuOnlyBuild true or false,UseCuDnn true or false,CudaVersion.
 build libcaffe project,generate dll.
## Test
modify classification project，add .cpp and .h
