# CycleGAN Project
This project implements CycleGAN, a deep learning model for image-to-image translation, using PyTorch. CycleGAN is capable of learning mappings between two domains without paired data. It can be used for various tasks such as style transfer, object transfiguration, and domain adaptation.

## CycleGAN Publication
To understand the concepts and architecture of CycleGAN, please refer to the original paper:

"Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks" by Jun-Yan Zhu, Taesung Park, Phillip Isola, and Alexei A. Efros. Link to the [paper](https://arxiv.org/abs/1703.10593)

## Project Files
config.py: This file contains the configuration settings for the CycleGAN training. It includes parameters such as device selection (CPU or CUDA), dataset directories, batch size, learning rate, lambda values, number of workers, and checkpoint filenames.

**dataset.py:** This file defines the custom dataset class, HorseZebraDataset. It loads horse and zebra images from their respective directories and applies transformations using the albumentations library. The dataset is created by combining images from both domains, ensuring equal representation during training.

**discriminator_model.py:** This file contains the discriminator model architecture implemented as a neural network. The Discriminator class defines the structure of the discriminator network, consisting of multiple convolutional blocks. The discriminator is responsible for distinguishing between real and fake images.

**generator_model.py:** This file contains the generator model architecture implemented as a neural network. The Generator class defines the structure of the generator network, which learns to translate images from one domain to another. It consists of an encoder-decoder architecture with skip connections and residual blocks.

**train.py:** This file is the main script for training the CycleGAN model. It sets up the discriminator and generator models, optimizers, loss functions, and data loaders. The training loop iterates over the dataset, performs forward and backward passes, and updates the model parameters. Checkpoints can be saved and loaded for resuming training.

**utils.py:** This file provides utility functions for saving and loading model checkpoints, setting random seeds, and other helper functions. It includes functions like save_checkpoint, load_checkpoint, and seed_everything.

**Getting Started**
To run this project, follow these steps:

1. Set up your Python environment with PyTorch and other required dependencies.
2. Download the Horse-to-Zebra dataset or prepare your own dataset in a similar format.
3. Modify the configuration settings in the config.py file to match your dataset and desired training parameters.
4. Run the train.py script to start training the CycleGAN model.
5. Monitor the training progress and generated images.
6. Adjust the hyperparameters, network architectures, or loss functions as needed.
7. Use the trained model for image-to-image translation on unseen data.


Please note that the success of the training process and the quality of the generated images heavily depend on the dataset, hyperparameters, and the complexity of the desired image translation task.

Feel free to explore and experiment with different options and techniques to enhance the performance of the CycleGAN model.

