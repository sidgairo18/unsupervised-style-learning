# Unsupervised Style Learning (2020)

This repository contains the source code, models and data files for the work titled: [Unsupervised Image Style Embeddings for Retrieval and Recognition Tasks](https://openaccess.thecvf.com/content_WACV_2020/html/Gairola_Unsupervised_Image_Style_Embeddings_for_Retrieval_and_Recognition_Tasks_WACV_2020_paper.html) (accepted at WACV 2020).

Please visit our project page for more details: https://sidgairo18.github.io/style

## Dependencies

```
* Python3
* PyTorch (and other dependencies for PyTorch)
* Numpy
* OpenCV 3.3.0
* Visdom Line Plotter
* tqdm
* cudnn (CUDA for training on GPU)

```

These are all easily installable via, e.g., `pip install numpy`. Any reasonably recent version of these packages should work. It is recommended to use a python `virtual` environment to setup the dependencies and the code.


## Running the code

### Training Dataset Construction

**Feature Extraction**

For feature extraction using pre-trained VGG network and PCA reduction use the following [repo](https://github.com/sidgairo18/Retrieval-Using-LSH-and-KD-Tree-Plus-Feature-Visualization).

**Clustering**
Feature extraction is followed by KMeans clustering. The optimal number of clusters for each dataset are determined using the *elbow method*.

### Training with Classification Loss (Stage 1)

* We train a CNN with augmented by a 256-dimensional bottleneck layer
* The training proceeds for 30 epochs and minimize cross-entropy loss for multi-class classification.
* We stop this after 30 epochs and the weights are saved which are used later in Stage 2.
* During this stage, we simply use the cluster ID for each image as its class label.
* Hyperparameters: lr = 0.001, Adam optimizer, Categorical Cross-Entropy (These are emperically chosen).
* Python script for this part is `classification_net.py`

### Training with Triplet Loss (Stage 2)

* Stage 2 of the pipeline, requries training a Triplet Convnet with Triplet Loss (MarginRanking loss)
* For this we require an anchor image, a positive sample and a negative sample. (How these images are sampled is explained in section 3.1.2 of the Paper)
* We train this triplet network for 50 epochs (Hyperparameters: lr = 0.001, SGD optimizer, MarginRanking Loss).
* The model weights from Stage 1 are loaded before the training for Stage 2 is started.
* Python script for this part is `train.py`
* For more information on the Triplet Network and embedding networks, take a look at `networks.py` and `triplet_network.py` files

**Note 1:** The bottle neck layer chosen has 256 dimensions (from experiments it was seen 256 dimensions instead of 128 makes not much difference in performance).

**Note 2:** The code may be slightly different from the parameters mentioned in the paper, but is sufficient to reproduce the results given in the paper (this is based on my unofficial implmentation of the [work](https://www.cvssp.org/data/Flickr25K/index_files/iccv17.pdf) and it's [public code](https://github.com/sidgairo18/Sketching-with-Style-ICCV-2017-PyTorch-).

### Running the Training Procedure

* For Stage 1, run `python classification.py`
* For Stage 2, run `python train.py`
* For details on the data-loader and data text files see next section.

### Data files in /data folder

* the `classification_dataloader` expects 2 files: `filenames_filename` and `labels_filename`.
* `filenames_filename` => A text file with each line containing a path to an image, e.g., `images/class1/sample.jpg`
* `labels_filename` => A text file with each line containing 1 integer, label index of the image.
* Similarly the `triplet_dataloader` expects 2 files: `filenames_filename` and `triplets_filename`.
* `filenames_filename` => A text file with each line containing a path to an image, e.g., `images/class1/sample.jpg`
* `triplets_filename` => A text file with each line containing 3 integers, where integer i refers to the i-th image in `filenames_filename`. For a line with integers "a b c", a triplet is defined such that image a is more similar to image c than it is to image b.

## Dataset

For the information on the dataset and splits used please go over **Sec 4** of the [paper](https://openaccess.thecvf.com/content_WACV_2020/papers/Gairola_Unsupervised_Image_Style_Embeddings_for_Retrieval_and_Recognition_Tasks_WACV_2020_paper.pdf), and [supplementary material](https://sidgairo18.github.io/style_supp.pdf).

The datasets used are:

1. **[BAM](https://bam-dataset.org/)**: Behance Artistic Media dataset, we use a subset of BAM dataset with 121K images (sampled similar to Behance-Net-TT 110K as in [work](https://www.cvssp.org/data/Flickr25K/index_files/iccv17.pdf)) balanced across media and emotional styles, and with a Train:Val:Test split as 80:5:15.
2. **[AVA Style Dataset](http://vislab.berkeleyvision.org/datasets.html)** Train:Val:Test split 85:5:10
3. **[Flickr](http://vislab.berkeleyvision.org/datasets.html)**: Train:Val:Test split 60:20:20
4. **[Wikipaintings](http://vislab.berkeleyvision.org/datasets.html)**: Train:Val:Test split 85:5:10
5. **[DeviantArt](https://www.deviantart.com/)**: Train:Val:Test split 85:5:10
6. **[WallArt](https://www.juniqe.com/wall-art/prints)**

### Accessing the Datasets

* 1 to request access to the dataset please visit the BAM website [here](https://bam-dataset.org/)
* 2, 3, 4 can be downloaded from [here](https://github.com/sergeyk/vislab/tree/master/vislab).
* 5, 6 have not been released publically yet due to licensing issues. But can be easily recreated as described in the paper and can be accessed at the respective websites.

## Feature Visualization

t-sne for feature visualizations

* [Link1](https://github.com/sidgairo18/tsne-for-Feature-Visualizations)
* [Link2](https://github.com/sidgairo18/Retrieval-Using-LSH-and-KD-Tree-Plus-Feature-Visualization/tree/master/Feature-Visualizations)


## End Notes and Disclaimer:

* The different dataset images have not been included.
* The text files in the data folder are just for reference. They may vary according to your own data files.
* To request access to the Dataset please visit the BAM website and refer to the notes under **Dataset** section.
* Feel free to use this code for your own work, but please cite the work if using this work officially.
* In case of any bugs or errors, please be gracious enough to report an issue on this repo.

## To cite this work: 

```
@InProceedings{Gairola_2020_WACV,
author = {Gairola, Siddhartha and Shah, Rajvi and Narayanan, P. J.},
title = {Unsupervised Image Style Embeddings for Retrieval and Recognition Tasks},
booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
month = {March},
year = {2020}
}
```

## License

We distriute the source code under the [MIT License](https://opensource.org/licenses/mit-license.php).

