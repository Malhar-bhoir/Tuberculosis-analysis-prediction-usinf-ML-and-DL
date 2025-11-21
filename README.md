# Tuberculosis Detection using Chest X-rays

This project aims to classify chest X-ray images as either 'Normal' or indicating 'Tuberculosis' using various machine learning models.

## Project Overview

The project utilizes two datasets of chest X-ray images: the Shenzhen Hospital X-ray Set and the Montgomery County X-ray Set. The images are preprocessed, resized, and augmented to create a balanced dataset for training and validation. Several models, including a Convolutional Neural Network (CNN), Support Vector Machine (SVM), and Random Forest, are trained and evaluated for their performance in classifying the X-rays.

## Dataset

The project uses the following datasets:

*   **Shenzhen Hospital X-ray Set:** Available on kaggle.
*   **Montgomery County X-ray Set:** Available on kaggle.

**Note:** Ensure the datasets are downloaded and placed in the specified directory structure or update the code to reflect the correct paths.

## Project Structure

The main script contains the following sections:

1.  **Setup and Imports:** Importing necessary libraries for image processing, model building, and evaluation.
2.  **Data Loading and Preprocessing:** Loading image lists, creating dataframes, and assigning labels based on filenames.
3.  **Data Analysis:** Checking class distribution and image characteristics (shape, pixel values).
4.  **Train/Validation Split:** Splitting the data into training and validation sets.
5.  **Directory Structure Creation:** Creating directories to organize the images for training and validation.
6.  **Image Transfer:** Copying and resizing images into the created directories.
7.  **Data Augmentation:** Augmenting the training data to balance the classes and improve model robustness.
8.  **Data Generators:** Setting up image data generators for efficient training.
9.  **Model Architecture:** Defining and compiling the CNN model.
10. **Model Training:** Training the CNN model with callbacks for checkpointing and learning rate reduction.
11. **Model Evaluation:** Evaluating the CNN model on the validation set and generating a confusion matrix and classification report.
12. **Other Models:** Implementing and evaluating SVM and Random Forest models.
13. **Model Comparison:** Visualizing the accuracy comparison of the different models.

## Dependencies

The project requires the following libraries. You can install them using pip:
Use code with caution
bash pip install tensorflow==2.* # or the version you used pip install keras==2.* # or the version you used pip install pandas pip install numpy pip install scikit-learn pip install matplotlib pip install opencv-python pip install imageio pip install skimage

## Usage

1.  **Download the datasets:** Obtain the Shenzhen and Montgomery County X-ray datasets and place them in the specified directory paths.
2.  **Run the code:** Execute the Python script in a Colab environment or a Jupyter Notebook.
3.  **Analyze the results:** Review the printed accuracy scores, classification reports, and the generated confusion matrix to assess model performance.

## Model Performance

The accuracy of the implemented models is compared in the project. The CNN model generally shows promising results. The comparison is visualized using bar plots.

## Future Work

*   Explore other CNN architectures and hyperparameters.
*   Implement transfer learning with pre-trained models.
*   Investigate different data augmentation techniques.
*   Incorporate additional evaluation metrics.
*   Develop a user interface for prediction.

## Contributing

Feel free to contribute to this project by submitting pull requests.

# Web App



-----

# 🩺 Tuberculosis Detection Web App

A Deep Learning-powered web application that detects Tuberculosis (TB) from chest X-ray images. Built with **Python**, **TensorFlow/Keras**, and **Streamlit**.

## 📋 Table of Contents

  - [Overview](https://www.google.com/search?q=%23overview)
  - [Features](https://www.google.com/search?q=%23features)
  - [Project Structure](https://www.google.com/search?q=%23project-structure)
  - [Prerequisites](https://www.google.com/search?q=%23prerequisites)
  - [Installation](https://www.google.com/search?q=%23installation)
  - [Usage](https://www.google.com/search?q=%23usage)
  - [Model Details](https://www.google.com/search?q=%23model-details)
  - [Disclaimer](https://www.google.com/search?q=%23disclaimer)

## 🧐 Overview

This project uses a Convolutional Neural Network (CNN) trained on chest X-ray datasets to classify images as either **Normal** or **Tuberculosis**. The user interface is built using Streamlit, allowing for easy image uploading and instant prediction results.

## ✨ Features

  * **Simple Interface:** User-friendly drag-and-drop file uploader.
  * **Real-time Prediction:** Instant classification using a pre-trained Keras model (`.h5`).
  * **Visual Feedback:** Displays the uploaded X-ray and the prediction confidence score.
  * **Confidence Scoring:** Shows the probability percentage of the detection.

## 📂 Project Structure

```bash
tb_project/
├── models/
│   ├── 3-conv-CNN.h5          # The trained CNN model
│   └── sgd_val_fmed_model.h5  # (Optional) Alternative model
├── app.py                     # Main Streamlit application script
├── requirements.txt           # List of dependencies
└── README.md                  # Project documentation
```

## ⚙️ Prerequisites

  * **Python 3.9** or **3.10** (Recommended for stability with TensorFlow).
  * Basic understanding of terminal/command prompt usage.

## 🛠️ Installation

**⚠️ IMPORTANT NOTE FOR WINDOWS USERS:**
To avoid `OSError: [Errno 2] No such file or directory` due to long file paths, place this project in a folder close to the root drive (e.g., `D:\tb_project` or `C:\projects\tb_app`). **Do not** nest it deep inside many subfolders.

1.  **Clone or Download the Project**
    Download the files and place them in a short directory path (e.g., `D:\tb_project`).

2.  **Create a Virtual Environment**
    Open your terminal/command prompt in the project folder and run:

    ```bash
    # Windows
    python -m venv myenv

    # Mac/Linux
    python3 -m venv myenv
    ```

3.  **Activate the Environment**

    ```bash
    # Windows
    .\myenv\Scripts\Activate

    # Mac/Linux
    source myenv/bin/activate
    ```

4.  **Install Dependencies**

    ```bash
    pip install -r requirements.txt
    ```

## 🚀 Usage

1.  Ensure your virtual environment is active (`(myenv)` should appear in your terminal).
2.  Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```
3.  A web browser will automatically open (usually at `http://localhost:8501`).
4.  Upload a Chest X-ray image (`.jpg`, `.png`, or `.jpeg`) to see the result.

## 🧠 Model Details

The application uses a **Convolutional Neural Network (CNN)** saved as `3-conv-CNN.h5`.

  * **Input Shape:** 150x150 pixels (RGB).
  * **Output:** Binary classification (Normal vs. Tuberculosis).
  * **Preprocessing:** Images are resized to 150x150 and pixel values are normalized to the [0, 1] range.

## ⚠️ Disclaimer

**This tool is for educational and research purposes only.** It is **not** a substitute for professional medical diagnosis. Always consult a qualified doctor or radiologist for medical advice.
