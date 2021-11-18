# TUC-dml-semantic-hierarchy

This repository supplies the code relating to the TU Chemnitz modeling seminar research project "Image Classification using Deep Metric Learning and a Semantic Hierarchy".

The example dataset provided with this repository is a reduced version of the CIFAR-100 dataset (2000 images from 100 classes) and may be used to test the code.
The specific dataset used for the project may not be made available to the general public.


## How to use:

To validate the results of the project use the two Jupyter notebooks contained within the repository.

The notebook "Get_Results.ipynb" is used to train the model based on parameters specified in the first cell. The model and history are then saved under a name that is based on the parameters used.
If wou want to change the optimizer or use a scheduler please do so by manually altering the eighth cell of the notebook.
The parameter perform_aug is used to specify whether data augmentation needs to be performed for classes that have only one image provided with the dataset.
The parameter perform_split signifies whether the dataset needs to be split into train and test set. In case that a test set needs to be used that is different in structure from the one used for the project (one from every class), the dataloaders need to be created manually.

The notebook "Test_Embedding.ipynb" may then be used to visualize the embedding space by loading a model obtained via the first notebook. Specify the same parameters used for training the model to load the correct model.


## Information:

The method implemented here is based on the one by Barz and Denzler from the following paper:

> [**Hierarchy-based Image Embeddings for Semantic Image Retrieval.**][1]  
> Björn Barz and Joachim Denzler.  
> IEEE Winter Conference on Applications of Computer Vision (WACV), 2019.

In case of any issues or questions, please contact marcel.scherenberg@s2020.tu-chemnitz.de.