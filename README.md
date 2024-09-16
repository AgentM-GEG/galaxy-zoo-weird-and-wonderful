# Galaxy Zoo Weird & Wonderful Analysis Pipeline

This repository encompasses the processed "consensus" data from the Galaxy Zoo: Weird & Wonderful project along with corresponding notebooks that illustrate the processing and analysis plot generation. These results have been included in the Mantha et al., 2024 Citizen Science Theory & Practice submission.

Very briefly, the Galaxy Zoo: Weird & Wonderful project (https://www.zooniverse.org/projects/zookeeper/galaxy-zoo-weird-and-wonderful) is a citizen science project that we launched on the Zooniverse citizen science platform (www.zooniverse.org) where we asked the volunteer community to identify any astronomical images (containing galaxies) that they thought were interesting and/or odd. The goal of this project was to understand the relationship between what us humans percieve as being unusual or interesting in images vs. what a machine model would. We trained a Wasserstein Generative Adversarial Network (wGAN) along with a latent feature vector localizer (a convolutional encoder) to learn generalized image-level features amongst a large sample of ~2 Million images from the Subaru Hyper Suprime Cam Survey and computed an anomaly score for each image as a metric to guage their unusuality compared to a "general/normal" ensemble. We then asked a subset of these images to be visually insepcted.

In this repo, there are a couple of resources:

* `data` folder contains the following sub-folders
  - `ww_data` folder contains various tables corresponding to the data downloaded from the Zooniverse (information of images, talk discussion board information).
  -  `anomaly_scores` contains csv files that are tables corresponding to images processed by the wGAN and each csv table contains **[image_identifier, image scores, feature scores, anomaly scores]** info for each image.
  -  `latent_space` is currently empty, but it is expected to contain the `npy` files where each file is a (N, 128) vector, where N is the total number of image processed by wGAN in a run. These are the 128 dimensional latent space vectors. Please download them using the following google drive link: https://drive.google.com/drive/folders/1UJ6trtwn2vq8ZZqGGsBiEf6jWWUnHLN_?usp=drive_link
  -  `model_checkpoints` contains three files corresponding to the saved wGAN model components: Generator, Discriminator, and the Encoder.

* `scripts` folder contains the following files:
  - `Analyze_Talk_Consensus.ipynb` is a explainatory notebook which contains detailed steps to load the aforementioned data and generate important plots that were shown the paper (including UMAP latent space exploration).
  - `compute_anomaly_scores.py` is a standalone python file that illustrates how to load the provided model checkpoints to apply the wGAN and generate anomaly scores for any number of images within a desired folder.

**Coming Soon**

Scripts that outline the wGAN architecture alongside the training and inference examples.
