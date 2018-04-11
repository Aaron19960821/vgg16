# VGG16

This a implementation of vgg16 with tensorflow and python, study-oriented.

## Overview of vgg16

**vgg16** is an important convolutional neural network posted by Karen Simonyan. The link of their essay is here below:  
**https://arxiv.org/abs/1409.1556**  

## Usage

### Train

To train vgg16 with this project, you only need to provide a path. The path should contain a file named "model.json" and this JSON
file contains all information to train a vgg16 network with this project.  

A sample **model.json** is shown below:  

```
{
	"learning_rate":0.01,
	"momentum":0.9,
	"batchsize":8,
	"batches":70,
	"channel":3,
	"classes":2,
	"classnamelist":"labels.txt",
	"trainlist":"train.txt",
	"labellist":"labellist.txt"
}
```

The **classnamelist** is a file that contains all names of classes. The number of classes should be equal to the number of classes that
provides in "model.json".  

The **trainlist** file contains the absolute path to all pictures in the training set.  
The **labellist** file contains the index of class of a picture, corresponding to the index of pictures in the **trainlist**.  

To train a vgg16 using this project:  

```python
#!coding=utf-8

from src.vgg16 import Vgg16

def main():
		#init a new Vgg16 instance
    net = Vgg16()
		#load json file
    net.loadWithUntrainedJson(srcDir="./example")
		#train CNN, the model will be deployed in tgtDir
    net.train(tgtDir='./example')

if __name__ == '__main__':
    main()
```
