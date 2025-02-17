# X-RAY IMAGE CLASSIFIER FOR PNEUMONIA DETECTION 

 <p align="Left">
  <img src="https://github.com/user-attachments/assets/8cc406ca-6efe-41d0-96f2-ff1c6d09fe5f" width="800" height="500" alt="Alt text">
   </p>

## Table of Contents

1. [Introduction](#OVERVIEW)  
2. [Background & Problem Statement](#BACKGROUND-&-PROBLEM-STATEMENT)  
3. [Aim](#AIM)
4. [Tech Stack](#TECH-STACK)
5. [Data](#DATA)
6. [Model Building](#MODEL-BUILDING)
7. [Model Evaluation](#MODEL-EVALUATION)
8. [Project Outcomes & Insights](#PROJECTS-OUTCOMES-&-INSIGHTS)   
9. [Challenges](#challenges)
10. [Key Takeaways & Next Steps](#Key-Takeaways-&-Next-Steps)
13. [References](#references)  

## OVERVIEW
Pneumonia is a severe lung infection that affects millions of people globally, with young children and the elderly being the most vulnerable. Early detection of pneumonia is critical for timely medical intervention and reducing mortality rates. However, diagnosing pneumonia through chest X-ray (CXR) images can be challenging due to similarities with other lung conditions.

This project aims to develop an AI-powered pneumonia detection model using deep learning techniques to classify chest X-ray images as either "Normal" or "Pneumonia". By leveraging latest Deep learning algorithm , the model can assist radiologists and healthcare professionals in improving diagnostic accuracy, reducing human error, and accelerating decision-making.

## BACKGROUND & PROBLEM STATEMENT
Pneumonia is one of the leading causes of death worldwide, particularly among young children and the elderly. It is estimated to be responsible for millions of deaths annually.<br>
There were an estimated 880,000 deaths from pneumonia in children under the age of five in 2016. Most were less than 2 years of age.
2.5 million people died from pneumonia in 2019. Almost a third of all victims were children younger than 5 years, it is the leading cause of death for children under 5.<br>
Accurately diagnosing pneumonia is challenging and involves several steps.<br>
* It typically requires the evaluation of a chest radiograph (CXR) by specialized experts, along with confirmation through clinical history, vital signs, and laboratory tests.<br>
* Pneumonia often presents as increased opacity areas on CXR but diagnosing it can be complicated due to similar findings in other lung conditions like fluid overload, bleeding, volume loss, lung cancer, or post-radiation or surgical changes. Pleural effusion (fluid in the pleural space) can also create opacity on CXR. <br>
* Comparing CXRs taken at different times and considering clinical symptoms and history can be valuable for an accurate diagnosis. 
### Problem Statement
* Reduce the dependency on expert radiologists for initial screening.
* Enhance early detection to improve patient outcomes.

## AIM
The primary objective of this project is to develop a deep learning model that can:
* Accurately classify chest X-ray images as Normal or Pneumonia.

## TECH-STACK

- **Programming Language**:
  - Python – The core programming language used for data preprocessing, model building, and evaluation.

- **Libraries & Frameworks**:
  - TensorFlow & Keras – Used for building and training the Convolutional Neural Network (CNN) model.
  - NumPy – For handling arrays and performing mathematical operations on image data.
  - Matplotlib & Seaborn – For visualizing results, graphs, and training progress.
  - OpenCV – To handle image processing tasks like resizing and normalization.
 
- **Development Environment:**
  - Jupyter Notebooks – For an interactive development environment and documentation of experiments.
  - Google Colab – For running models on cloud GPUs, allowing faster training and validation.

- **Model Evaluation Metrics:**
  - Accuracy – To evaluate the proportion of correct predictions.
  - Recall – To measure the model’s ability to identify all pneumonia cases.
  - Loss Function:
  - Binary Crossentropy – Used to calculate the loss for binary classification (Normal vs. Pneumonia).

- **Image Processing & Data Augmentation:**
  - Keras ImageDataGenerator – Used for real-time data augmentation (e.g., flips, shifts, zooms) to increase the diversity of training data.
 
- **Version Control:**
  - Git & GitHub – For version control and collaborative development of the project.

## DATA 
**This project dataset can be downloaded using the Link:** https://tinyurl.com/pneumonia-dataset

We are given with chest X-Ray Images of Pneumonia and Normal Patients,<br>
We are given a separate file for X-ray of normal patients and Pneumonia patient each in training, testing set and Validation Set <br>

* TRAINING SET:<br>
We have 5216 entities in our Training data<br>
* TESTING SET:<br>
We have 624 entities in our Testing data<br>
* VALIDATION SET:<br>
We have 16 entities in our Validation data<br>

### Data Preparation and Pre-processing

  #### 1) Data Cleaning, Organization and Profiling
  
  * Removing inconsistencies (duplicate or corrupted images).
  * Organizing files into appropriate categories for training, testing, and validation.
  * Profiling the dataset to understand class distribution and detect potential biases.
  
  #### 2) Data Transformation
  
To enhance the model’s robustness, various transformation techniques are applied:
  * Augmenting Training Data – Random transformations such as flipping, shifting, and zooming are used to artificially expand the dataset.
  * Reducing Overfitting – These transformations introduce variability, preventing the model from memorizing training data.
  * Improving Generalization – Helps the model learn more distinctive and meaningful patterns, improving performance on unseen data.
    
  #### 3) Data Normalization
  
To ensure consistency, pixel values are re-scaled to a standard range, optimizing the input for deep learning models.

<br>**NOTE**:<br>
**Data Generators** were defined for  applying transformations in real-time, ensuring diverse training samples.

## MODEL BUILDING

After completing data preprocessing, the next step is to build a Convolutional Neural Network (CNN) for pneumonia classification using TensorFlow and Keras.

### Model Architecture

The CNN architecture consists of:
 - Convolutional Layers – Extracts important features from X-ray images.
 - Pooling Layers – Reduces dimensionality while retaining key information.
 - Fully Connected Layers – Classifies images as Normal or Pneumonia.
 - Activation Functions – ReLU for hidden layers, Sigmoid for output.

### Training the Model
We will train the model using binary cross-entropy loss, Adam optimizer, and track validation loss to save the best model.

cnn_model = cnn.fit(
    training_set,
    steps_per_epoch=163,
    epochs=1,
    validation_data=validation_set,
    validation_steps=624
)


## MODEL EVALUATION

**Validation set Evaluation**	<br>
* Validation loss: 0.6133 <br>
* Validation accuracy: 0.6875 <br>
* The Model achieves 68.75% Accuracy in Predicting Pneumonia Cases on the VALIDATION SET <br>


**Test set Evaluation**	<br>
* Test loss: 0.3352278769016266 <br>
* Test accuracy: 0.8846153616905212 <br>
* The model demonstrates 88.4% accuracy on the test dataset, indicating strong performance in classifying pneumonia cases.

**Confusion Matrix** <br>

<img src="https://github.com/user-attachments/assets/d609b752-8f27-4fd7-a52f-56fc574188e1" width="450" height="350" alt="Alt text"><br>


**Classification Report (1: Pneumonia 0: Normal)** <br>
<br>
<img src="https://github.com/princed145/Pneumonia-detection-tool-/assets/63622088/6f6a6474-b97f-424e-af41-8a18cfb2a2ac" width="650" height="300" alt="Alt text">


**ROC-AUC Curve** <br>
<br>
<img src="https://github.com/princed145/Pneumonia-detection-tool-/assets/63622088/30a83c2a-d9e3-43a3-ac2d-46450fe03591" width="450" height="350" alt="Alt text"><br>


## PROJECT OUTCOMES & INSIGHTS 

Conclusion & Key Findings
* The model achieves 50% accuracy in distinguishing pneumonia-positive cases, which is equivalent to random chance.
* Recall is the most critical metric in this scenario, as it determines how well the model detects pneumonia cases (True Positives).
* The model attains 50% recall on the test set, meaning:
  * It correctly identifies half of the pneumonia cases (True Positives).
  * The other half of pneumonia cases are misclassified as negative (False Negatives).
<br> **For example**, if there are 120 pneumonia cases in the test set, the model correctly identifies only 60 cases, while 60 cases go undetected, which is a major limitation in a real-world healthcare setting.

## CHALLENGES

- Data Quality & Quantity:

  Ensuring the X-ray images are of consistent quality and resolution.
  Handling missing, mislabeled, or corrupted images.

-  Class Imbalance:

  Balancing the number of normal versus pneumonia cases to prevent bias in the model.

- Overfitting:

  The model may learn the training data too well, failing to generalize on unseen data.
  Limited data availability could increase the risk of overfitting.

- Data Augmentation Complexity:

  Applying transformations (e.g., flips, shifts, zooms) without distorting critical medical features

## KEY TAKEAWAYS & NEXT STEPS
* Improving Recall: Since missing pneumonia cases can have severe health implications, the next step would be to tune the model (e.g., adjusting class weights, using different architectures, or increasing data augmentation) to reduce false negatives.
* Balancing Precision & Recall: While recall is crucial, optimizing precision is also necessary to minimize false alarms.
* Exploring Additional Features: Incorporating clinical metadata (e.g., patient symptoms, age, or medical history) alongside X-ray images may enhance prediction accuracy.


## REFERENCES

https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
