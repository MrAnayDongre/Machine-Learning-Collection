# WGAN Project
This project implements the Wasserstein GAN (WGAN) algorithm for generative adversarial network training, using PyTorch. WGAN is a variation of GANs that introduces the Wasserstein distance as a loss function, providing more stable training and better gradient flow.

## WGAN Publication
To understand the concepts and theory behind WGAN, please refer to the original paper:

"Wasserstein GAN" by Martin Arjovsky, Soumith Chintala, and Léon Bottou. Link to the [paper](https://arxiv.org/abs/1701.07875)

## Project Files
**model.py:** This file contains the implementation of the Discriminator and Generator models. The Discriminator is referred to as a "critic" in the WGAN paper, as it learns to estimate the Wasserstein distance. The Generator follows a typical generator architecture with transpose convolutions. The models are initialized with appropriate weights, and a test function is included to verify their functionality.

**train.py:** This file is responsible for training the WGAN model. It sets up the data loaders, initializes the generator and critic models, and defines the optimization process. The training loop involves iteratively updating the critic and generator networks based on the WGAN loss. The losses and generated images are monitored and visualized using TensorBoard.

**Getting Started**
To run this project, follow these steps:

1. Set up your Python environment with PyTorch and other required dependencies.
2. Prepare your training dataset. In this example, the MNIST dataset is used, but you can use other datasets as well. Ensure that the dataset is in the appropriate format and structure.
3. Modify the hyperparameters and configuration settings in the train.py file to match your dataset and desired training parameters.
4. Run the train.py script to start training the WGAN model.
5. Monitor the training progress and generated images using TensorBoard or other visualization tools.
6. Adjust the hyperparameters, network architectures, or loss functions as needed to achieve better results.
7. Utilize the trained generator model for generating new samples or performing other tasks.


Please note that the success of the training process and the quality of the generated images heavily depend on factors such as the dataset, hyperparameters, and the complexity of the task.

Feel free to explore and experiment with different options and techniques to enhance the performance of the WGAN model.