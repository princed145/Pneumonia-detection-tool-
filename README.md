# PROJECT REPORT 
# Title
X-ray Image Classifier for Pneumonia Detection
# Introduction/Description 
Pneumonia is one of the leading causes of death worldwide, particularly among young children and the elderly. It is estimated to be responsible for millions of deaths annually.<br>
There were an estimated 880,000 deaths from pneumonia in children under the age of five in 2016. Most were less than 2 years of age.
2.5 million people died from pneumonia in 2019. Almost a third of all victims were children younger than 5 years, it is the leading cause of death for children under 5.<br>
Accurately diagnosing pneumonia is challenging and involves several steps.<br>
•	It typically requires the evaluation of a chest radiograph (CXR) by specialized experts, along with confirmation through clinical history, vital signs, and laboratory tests.<br>
•	Pneumonia often presents as increased opacity areas on CXR but diagnosing it can be complicated due to similar findings in other lung conditions like fluid overload, bleeding, volume loss, lung cancer, or post-radiation or surgical changes. Pleural effusion (fluid in the pleural space) can also create opacity on CXR. <br>
•	Comparing CXRs taken at different times and considering clinical symptoms and history can be valuable for an accurate diagnosis. 

# Problem Statement 
Our Goal is to Build an Algorithm which Accurately Classifies the X-ray images(NORMAL VS PNEUMONIA)

# Dataset Description
We are given with chest X-Ray Images of Pneumonia and Normal Patients,<br>
We are given a separate file for X-ray of normal patients and Pneumonia patient each in training, testing set and Validation Set <br>

TRAINING SET:<br>
We have 5216 entities in our Training data<br>
TESTING SET:<br>
We have 624 entities in our Testing data<br>
VALIDATION SET:<br>
We have 16 entities in our Validation data<br>

# Methods
Data Organization and Profiling <br>

Data Preparation for Model Building<br>

**Augmenting the Training Data**<br>

To Increasing the Diversity of Traning Data - Applying Random Transformation (Like Flips, Shift, Zoom)<br>
To Reduce Overfitting<br>
To Improve Generalization<br>
For Better Feature Learning<br>

**Data Normalization**<br>

Re-scaling Pixels to a standard size <br>

**Data Generators**<br>

The above Two methods are been applied by defining data generators <br>

# Model Building

Using keras Custom CNN Model for Model Building <br>

# Model Evaluation

**Validation set Evaluation**	<br>
Validation loss: 0.6133 <br>
Validation accuracy: 0.6875 <br>
Our Model is 68.75% Accurate in Predicting Pneumonia Images on the VALIDATION SET <br>

**Test set Evaluation**	<br>
Test loss: 0.3352278769016266 <br>
Test accuracy: 0.8846153616905212 <br>
Our Model is 88.4% Accurate in Predicting Pneumonia Images on TEST SET

![image](https://github.com/princed145/Pneumonia-detection-tool-/assets/63622088/6f6a6474-b97f-424e-af41-8a18cfb2a2ac)




