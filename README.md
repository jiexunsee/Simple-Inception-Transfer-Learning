# A classifier for cats and dogs

This is a response to Siraj's challenge of the week.

This is a classifier made with basic TensorFlow, using Transfer Learning from the Inception-V3 model. Data is taken from [this](https://www.kaggle.com/c/dogs-vs-cats) Kaggle competition, as recommended by Siraj. 

## How to use

### Training
First, train the model, run:

`python trainclassifier.py train`

This script will call the getvector.py script, which uses the Inception-V3 model to produce 2048-dimensional vector representations of images. These 2048-dimensional vectors have already been saved in the data_inputs.txt and data_labels.txt files, for convenience. (It takes quite long on a CPU to run a few hundred images through the Inception network, let alone the full 25,000 image dataset.)

After getting these vector representations, I use a fully connected one layer neural network to output a prediction vector, using TensorFlow.

For my training, I only used 300 images of cats and dogs combined. However, the network performs remarkably well for such few training examples.

### Testing
To test the model, run:

`python trainclassifier.py test cat.jpg`

This runs the cat.jpg image through the Inception-V3 network to get the 2048-dimensional vector. Then it loads the saved TensorFlow one-layer neural network, and feeds the cat.jpg image vector into it. Do try other pictures too, the predictions are quite accurate.


## Notes:

I got a lot of the code from the TF Slim tutorial. It's taken from on their [github](https://github.com/tensorflow/models/tree/master/slim)


P.S. made a grammatical error that I realised too late to correct - my variables should be named 'input' instead of 'inputs'
