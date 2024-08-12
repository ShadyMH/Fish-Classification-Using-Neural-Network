# Fish Species Classification using CNN

This project focuses on classifying different species of fish using a Convolutional Neural Network (CNN). The dataset consists of images of nine different species of fish, and the project demonstrates the end-to-end process of developing, training, and evaluating a CNN model to accurately classify fish species.

## Project Structure

- **cnn__3.py**: The main Python script that includes the data processing, model building, training, and evaluation steps.
- **פרויקט גמר - רשתות נוירונים.pdf**: The project report that provides an overview, methodology, and results of the project.
- **Outputs**: A folder containing all the generated graphs and results from the project, including accuracy and loss plots, confusion matrix, and model predictions.

## Dataset

The dataset was meticulously compiled and curated to include images of nine fish species:
- Black Sea Sprat
- Gilt-Head Bream
- Hourse Mackerel
- Red Mullet
- Red Sea Bream
- Sea Bass
- Shrimp
- Striped Red Mullet
- Trout

Each species in the dataset contains approximately 1000 high-quality images, totaling 9000 images. The images were carefully collected, ensuring a diverse range of poses and conditions to make the classification task challenging and realistic.

## Project Overview

### Problem Statement

The primary challenge was to classify images of fish into one of nine species. To achieve this, I designed and developed a Convolutional Neural Network (CNN) that could accurately distinguish between these species based on their visual features.

### Approach

1. **Data Preparation**: The images were preprocessed by resizing them to a uniform size of 244x244 pixels. Data augmentation techniques, such as rotation, flipping, and scaling, were applied to enhance the model's robustness and prevent overfitting.

2. **Model Architecture**: A custom CNN model was built using TensorFlow and Keras. The architecture includes multiple convolutional layers to extract features, followed by pooling layers to reduce dimensionality, and fully connected dense layers to perform the final classification. Dropout layers were included to regularize the model and improve its generalization capabilities.

3. **Training**: The model was trained using the Adam optimizer and categorical cross-entropy loss function. The training process was carefully monitored using accuracy and loss metrics, and adjustments were made to the model architecture and hyperparameters to optimize performance.

   - **Training and Validation Accuracy**: The plot below shows the accuracy of the model during training and validation across the epochs.
     ![Training and Validation Accuracy](outputs/training_validation_accuracy.png)

   - **Training and Validation Loss**: The plot below illustrates the model's loss during training and validation.
     ![Training and Validation Loss](outputs/training_validation_loss.png)

4. **Evaluation**: The model's performance was evaluated on a separate test set using a confusion matrix and classification report. These tools provided insights into the accuracy and precision of the model across the different fish species.

   - **Confusion Matrix**: The confusion matrix below displays the classification performance across the different species.
     ![Confusion Matrix](outputs/confusion_matrix.png)

   - **Model Predictions**: Below is a sample of the model's predictions. Correct predictions are highlighted in blue, while incorrect ones are shown in red.
     ![Model Predictions](outputs/model_predictions.png)

### Results

The final model achieved an accuracy of 98%, demonstrating its effectiveness in classifying fish species. The high accuracy is a testament to the careful design of the CNN architecture and the thoroughness of the data preparation process.

## How to Run the Project

1. Clone the repository:
   ```
   git clone https://github.com/your-username/fish-classification-cnn.git
   cd fish-classification-cnn
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Prepare the dataset by organizing the images into the appropriate directories as outlined in the script.

4. Run the script:
   ```
   python cnn__3.py
   ```

5. The script will process the data, build and train the model, and display the evaluation results.

## Conclusion

This project successfully demonstrates the use of Convolutional Neural Networks for image classification tasks, specifically for identifying different species of fish. The model's high accuracy highlights the effectiveness of the methods and techniques employed in its development.
