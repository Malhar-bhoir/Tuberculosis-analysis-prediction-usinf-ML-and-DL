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

