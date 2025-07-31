# DEM-WEB App

This project implements a density-equalizing mapping application using a neural network model. It consists of a backend for processing and generating domain images, a model definition for the neural network architecture, and a web application for user interaction.

## Project Structure

- **backend.py**: Implements the backend logic of the application, handling command-line arguments, loading the model, generating original and mapped domain images, and calculating quality metrics.
  
- **model_def.py**: Defines the neural network model (UNet) used for density mapping, including its architecture, initialization, loss computation, and optimization functions.
  
- **web_app.py**: Implements the frontend using Streamlit, allowing users to upload checkpoint files, select function expressions, and generate original and mapped domains.
  
- **requirements.txt**: Lists all the required Python packages and their versions for the project.

## Installation

To install the required packages, run:

```
pip install -r requirements.txt
```

## Usage

1. Start the web application by running:
   ```
   streamlit run web_app.py
   ```

2. Upload a checkpoint file and select a function expression to generate the original and mapped domains.

3. The backend will process the inputs and generate the corresponding images and metrics.

## License

This project is licensed under the MIT License.
