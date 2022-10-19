## Image Captioning : 
Image Captioning is the process of generating textual description of an image. It uses both Natural Language Processing and Computer Vision to generate the captions. The dataset will be in the form [image → captions]. The dataset consists of input images and their corresponding output captions.

## Encoder:
The Convolutional Neural Network(CNN) can be thought of as an encoder. The input image is given to CNN to extract the features. The last hidden state of the CNN is connected to the Decoder.

## Decoder:
The Decoder is a Recurrent Neural Network(RNN) which does language modelling up to the word level. The first time step receives the encoded output from the encoder and also the <START> vector.

## Model Building
For this, Pretrained Resnet-50 model which is trained on ImageNet dataset (available publicly on google) is used for image feature extraction. 5 layered LSTM layer model is used for image caption generation.
