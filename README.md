# Bus-Driver-Behavior-Detection
CNN+Bi-directional-LSTM; Video classification; Four categories（Normal; Smoking; Using mobile; Off seat）</br>
code -> optic.py[-d The directory of dataset]: This script USES the optical flow  to process dataset images. </br>
        vgg_lstm.py: Model definition.</br>
        log_IBRD.py: Log format definition.</br>
dataset -> img -> Train-Image & Test-Image</br>
h5 -> The weights of the VGG model and the Bi-LSTM model.</br>
npy -> The feature maps of data.

The dataset you can download on [Driver Behavior Dataset](https://drive.google.com/open?id=1yFrP9yoFDqG5rfyD94IkiZWfs2YK_uYg)</br>
Package details:</br>
Python 2.7</br>
h5py 2.7.0 </br>
Keras 1.2.2 </br>
numpy 1.13.1 </br>
opencv 2.4.11 </br>
Theano 0.8.2 </br>
