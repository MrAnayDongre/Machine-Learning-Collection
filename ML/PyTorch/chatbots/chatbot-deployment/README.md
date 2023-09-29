# Chatbot for E-commerce Website

This is a chatbot for an e-commerce website that sells tea and coffee. The chatbot is designed to assist customers by answering their questions, providing product information, and offering recommendations.

## Table of Contents

- [Project Overview](#project-overview)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Training the Chatbot](#training-the-chatbot)
- [Usage](#usage)
- [Customization](#customization)
- [Technologies Used](#technologies-used)
- [Author](#author)

## Project Overview

The chatbot is a conversational AI designed to engage with customers visiting the e-commerce website. It can answer common questions, provide product details, offer recommendations, and handle customer inquiries related to products, payments, delivery, and more. The chatbot is built using Python and relies on natural language processing techniques to understand and respond to user queries.

## Project Structure

The project consists of the following files and folders:

- `train.py`: This script is used to train the chatbot's machine learning model using PyTorch. It reads the intents and responses from `intents.json`, preprocesses the data, and trains the neural network.

- `nltk_utils.py`: This file contains utility functions for tokenization, stemming, and creating a bag of words from sentences.

- `model.py`: It defines the neural network architecture used for the chatbot.

- `intents.json`: This JSON file contains the intents, patterns, and responses used for training the chatbot.

- `chat.py`: This script loads the trained model and provides a function to generate responses based on user input.

- `app.py`: This is a Flask web application that serves as the interface for the chatbot. It handles user input, sends it to the chatbot, and displays the responses.

- `templates`: This folder contains an HTML template (`base.html`) used for rendering the chat interface.

- `static`: This folder contains static assets such as CSS styles and JavaScript (`app.js`) for the chat interface.

## Getting Started

### Prerequisites

Before you begin, ensure you have met the following requirements:

- Python (>=3.6) is installed on your system.
- You have access to a terminal or command prompt.

### Installation

1. Clone the repository to your local machine:

   ```bash
   git clone repo_name
   ```

2. Change to the project's directory:

    ```bash
    cd chatbot-for-ecommerce
    ```

3. Install the required dependencies by running:

    ```bash
    pip install -r requirements.txt
    ```

### Training the Chatbot
To train the chatbot model, follow these steps:

1. Open a terminal and navigate to the project directory if you're not already there.

2. Run the training script:

    ```bash
    python train.py
    ```

## Usage

1. Start the Flask web application by running:

    ```bash
    python app.py
    ```
2. Access the chatbot interface by opening a web browser and navigating to http://localhost:5000.

3. Use the chat interface to interact with the chatbot. Type a message, and the chatbot will provide responses based on the trained model and the content of intents.json.

4. You can ask questions about products, payments, delivery, and more.

## Customization
To customize the chatbot for your specific e-commerce website, you can modify the intents.json file. Add new intents, patterns, and responses to make the chatbot more informative and engaging.

## Technologies Used
* Python
* PyTorch
* Flask
* NLTK (Natural Language Toolkit)

## Contributing

Contributions are welcome! If you'd like to improve this chatbot or add new features, please feel free to submit a pull request.