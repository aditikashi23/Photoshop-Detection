# RvF - Real vs Fake Face Detection

Deepfakes are everywhere now, and like us, you're probably wondering ["wow how am I going to figure out if that video of Morgan Freeman telling me he isn't real, is actually real?"](https://youtu.be/oxXpB9pSETo). While a true deepfake detector is too complicated for a single semester, we're going to do something a little bit easier - try and determine whether a picture of a face is fake or not. You'll definitely enjoy this project if you want to learn about neural networks, computer vision, and figuring out if your friend keeps Photoshopping their Instagram posts.

**Skills Learned**: Machine Learning, Deep Learning, Computer Vision, PyTorch, TensorFlow

## Model Architecture Overview

This convolutional neural network (CNN) is designed for Photoshop manipulation detection. It starts with a `BatchNorm2d` layer to normalize input across channels, enhancing training stability. `ZeroPad2d` ensures consistent input size before each convolutional block. The model uses two convolutional layers: the first expands from 4 to 16 channels, and the second from 16 to 128 channels, extracting progressively complex features. A `Dropout` layer with a small probability (0.05) is included after each convolution to reduce overfitting. Each convolutional block also has a `ReLU` activation for non-linearity and a `MaxPool2d` layer for spatial downsampling. Finally, a `Flatten` layer prepares the features for a fully connected layer that outputs predictions for 10 classes.

