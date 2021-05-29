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

