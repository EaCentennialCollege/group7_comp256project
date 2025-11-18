SYNOPSIS
Your team has been approached by the law enforcement security company ABC, which is looking to integrate facial recognition and identification capabilities into its system. To support this effort, ABC has provided your team with the umist_cropped.mat dataset for training purposes.1

Download the Dataset
[📄 Download umist_cropped.mat](umist_cropped.mat)





1. Data Preparation [2 points]
Download the dataset (umist_cropped.mat) from Luminate and explore its structure to understand the image and label organization. Load the dataset and convert them into a Pandas DataFrame with corresponding labels.

2. Data Splitting [5 points]
Split the dataset into training, validation, and test sets using stratified sampling to ensure that each set contains a balanced number of images per person. In the project report, explain your choice of split ratio and discuss how stratified sampling helps maintain class balance and supports reliable model evaluation. After splitting, apply normalization on the training data, and use the fitted scaler to transform the validation and test sets.

Include visualization plots in the report showing the distribution of images per person in each subset after the split.

3. Dimensionality Reduction [8 points]
Try at least two different dimensionality reduction techniques to better understand how to represent high-dimensional image data in lower-dimensional spaces.

Below are some suggested methods, but you are encouraged to explore and select the techniques best suited for image data:

PCA (Principal Component Analysis): Try different numbers of components (e.g., 10, 20, 50, 100) and visualize the explained variance ratio to determine how many components capture most of the variance.
Autoencoder: Implement a simple neural network autoencoder (using frameworks like Keras) to learn a compressed representation (latent code) and compare its results with linear methods like PCA.
t-SNE & UMAP: These techniques are not covered in this course, but if you are interested, you can research and apply them.
Include visualizations for each technique you apply, and discuss the differences in how they represent the dataset in your analysis report.

4. Clustering [25 points]

After reducing the dimensionality of the dataset, each group should select at least two clustering techniques covered in this course, such as K-Means, Hierarchical Clustering, DBSCAN, etc., and apply them to the reduced dataset. 

In your report, clearly explain the following:

The clustering methods your team chose and the rationale behind their selection.
How you tuned the key parameters of each method (e.g., number of clusters for K-Means, linkage method for Hierarchical Clustering, or epsilon and min_samples for DBSCAN).
Any challenges encountered while clustering high-dimensional image data and how dimensionality reduction helped address them.
Include 2D visualizations of the clustering results using matplotlib, and provide an analysis of the purity or composition of each cluster. This analysis will help assess how effectively each method groups images of the same individual.

5. Image Recognition using Supervised Learning – Neural Network Classifier [30 points]

Build and train a neural network to classify the face images using the labeled data. Choose an appropriate architecture (e.g., a Convolutional Neural Network (CNN), or a general Artificial Neural Network (ANN), depending on how you represent the data). Use the training and validation sets to train and fine-tune your model.

Note: Optionally, you can enhance your feature set by incorporating the cluster assignments or the distances to cluster centroids (obtained from your previous clustering step) as additional numerical features.

In your project report:

Clearly illustrate and explain the architecture of your model (number of layers, type of layers, activation functions, etc.)
Discuss the rationale behind your choice of:
Activation functions
Loss function
Optimizer and training strategy
       3. Explain how you tuned hyperparameters such as learning rate, batch size, number of epochs, and regularization techniques.

       4. Describe how including cluster labels and/or centroid distances as additional features impacted model performance (if used).

       5. Include training curves and performance metrics (accuracy, precision, recall, F1-score) on validation and test sets.

      6. Display a few sample test images with both true and predicted labels. Provide a brief explanation of how accurately your neural network  was able to classify the individuals.

6. Project Demonstration: [30 Points]

Each team is required to present their completed project as a group. During the presentation, clearly explain the key design decisions made throughout the project, the challenges your team encountered, and the strategies used to overcome them. Conclude by presenting and discussing the final results of your project. The presentation should reflect the team's collaboration and a shared understanding of the entire workflow.

Deliverables to Submit:
Please submit the following files in your project submission folder:

Python source code file(s) (.py scripts) containing your complete implementation
A project report (PDF or Word) including explanations, analyses, and all required plots and visualizations as outlined in the project instructions
Any additional models, pipelines, or data files necessary to run your code and reproduce your results
Please name your project folder in this format: GroupX_comp257Project

where X is your group number (e.g., Group1_comp257Project). Ensure all submitted files are contained within this folder.
