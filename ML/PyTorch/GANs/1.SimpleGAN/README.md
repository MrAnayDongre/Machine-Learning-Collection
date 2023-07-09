# SimpleGAN Project
This project implements a simple Generative Adversarial Network (GAN) architecture, referred to as SimpleGAN, using PyTorch. The SimpleGAN consists of a Generator and Discriminator model trained on the MNIST dataset to generate realistic hand-drawn digits.

## SimpleGAN Publication
The SimpleGAN architecture is a simplified version of the original GAN introduced by Ian Goodfellow and his team. To gain a deeper understanding of GANs, you may refer to the original paper:

"Generative Adversarial Networks" by Ian J. Goodfellow, et al. Link to the [paper](https://arxiv.org/abs/1406.2661)

# Project File
**fc_gan.py:** This file contains the implementation of the Discriminator and Generator models for SimpleGAN. The Discriminator model learns to distinguish between real and fake images, while the Generator model generates fake images to fool the Discriminator. The models are implemented using fully connected layers and are initialized with appropriate weights. The forward method is defined for both models.

**Getting Started**
To run this project, follow these steps:

1. Set up your Python environment with PyTorch and other required dependencies.
2. Prepare your training dataset. In this example, the MNIST dataset is used, which consists of hand-drawn digits. 3. Ensure that the dataset is in the appropriate format and structure.
4. Modify the hyperparameters and configuration settings in the fc_gan.py file to match your dataset and desired training parameters.
5. Run the fc_gan.py script to start training the SimpleGAN model.
6. Monitor the training progress and generated images using TensorBoard or other visualization tools.
7. Adjust the hyperparameters, network architectures, or loss functions as needed to achieve better results.
8. Utilize the trained Generator model for generating new digit samples or performing other tasks.


Please note that the success of the training process and the quality of the generated images heavily depend on factors such as the dataset, hyperparameters, and the complexity of the task.

Feel free to explore and experiment with different options and techniques to enhance the performance of the SimpleGAN model.

