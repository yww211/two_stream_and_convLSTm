Two-stream and convLSTM
=====
* 'Description' :This is the sorce code  associated with the paper 《Two-Stream Convolutional Network for Improving Activity Recognition Using Convolutional Long Short-Term Memory Networks》
* 'File Description'：
  *CNN-GPUs：this is a floder contains implement of cnn used as Two-stram network, suporting to fine-tune on specific dataset with  multi-GPU
  *CNN-Pred-Feat:use the networks defined in the CNN-GPUs to extract both RGB feature description and optical feature description
  *Optical-flow:use opencv to compute optical  information and map the optical  information into  optical video
  *RNN-convLSTM: the implement of convLSTM and the tarining and testing process of  convLSTM on the feature maps from two-stram CNN
* 'Dependencies':
  * torch
  * cudnn8.0 and CUDA
  * [extracunn](https://github.com/viorik/extracunn)
  * [RNN library](https://github.com/Element-Research/rnn)
  * ffmpeg
  * [Torch Video Decoder Library](https://github.com/e-lab/torch-toolbox/tree/master/Video-decoder)
