# Tea and Coffee E-commerce Chatbot Application


This is a chatbot application for an e-commerce website that specializes in selling tea and coffee. It allows users to interact with the chatbot through a graphical user interface (GUI) built with `tkinter`. Users can inquire about available items, payment methods, delivery options, and more.

## Table of Contents

- [Dependencies](#dependencies)
- [Setup and Usage](#setup-and-usage)
- [Project Structure](#project-structure)
- [Customization](#customization)
- [Contributing](#contributing)


## Dependencies

This chatbot application relies on the following Python libraries. You can install them using `pip`:

- **Tkinter**: A standard Python interface to the Tk GUI toolkit. You may need to install it separately based on your operating system.

    ```bash
    # On Debian/Ubuntu-based systems
    sudo apt-get install python3-tk

    # On Red Hat/Fedora-based systems
    sudo dnf install python3-tkinter

    # On macOS
    brew install python-tk
    ```

- **torch**: PyTorch is used for natural language understanding (NLU) and machine learning. You can install it using:

    ```bash
    pip install torch
    ```

- **nltk**: The Natural Language Toolkit is used for text processing. Install it with:

    ```bash
    pip install nltk
    ```

- **numpy**: NumPy is used for numerical operations. Install it with:

    ```bash
    pip install numpy
    ```

- **json**: The `json` module is used for parsing intent data. It's included in the Python standard library.

- **random**: The `random` module is used for generating random responses. It's included in the Python standard library.

## Setup and Usage

1. Clone this repository to your local machine:

    ```bash
    git clone repo_name
    ```

2. Navigate to the project directory:

    ```bash
    cd repo_name
    ```

3. Ensure you have the necessary dependencies installed.
    ```bash
    pip install -r requirements.txt
    ```

4. Run the chatbot application:

    ```bash
    python app.py
    ```

5. The chatbot GUI will open, allowing you to interact with it.

## Project Structure

The project structure is organized as follows:

- `app.py`: The main application that handles the GUI and user interactions.
- `model.py`: Defines the neural network model used for intent classification.
- `nltk_utils.py`: Contains utility functions for text processing and tokenization.
- `intents.json`: Stores predefined intents, patterns, and responses for the chatbot.
- `data.pth`: Saved model data, including the trained neural network model and word embeddings.
- `requirements.txt`: Lists the Python dependencies for the project.
- `screenshots/`: Directory containing screenshots of the chatbot GUI.

## Customization

You can customize the chatbot's behavior by modifying the `intents.json` file. Add new intents, patterns, and responses to enhance the chatbot's capabilities and make it more relevant to your e-commerce website.

## Contributing

Contributions are welcome! If you'd like to improve this chatbot or add new features, please feel free to submit a pull request.


