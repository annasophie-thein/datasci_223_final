### DATASCI 223 Final Project:

#### Overview of the Problem
Optical Coherence Tomography (OCT) is a non-invasive diagnostic technique that provides an in vivo cross-sectional view of the retina. This project aims to classify common ocular diseases that can be diagnosed on OCT images; age-related macular degeneration (AMD), diabetic retinopathy (DR), macular hole (MH), and choroidal neovascularization (CNV). Using a machine learning model in clinic to classify the OCT images could provide support and time optimization for the ophthalmologist reviewing the images. 

This project involved organizing data into training and testing sets, preprocessing images, segmenting regions of interest (ROI), creating data structures for model training, training a convolutional neural network (CNN) model, and evaluating the model on unseen data.

#### Description of the Dataset
The dataset consists of OCT images from a publicly available dataset (https://www.kaggle.com/datasets/paultimothymooney/kermany2018). 
1) Folder creation: The images were categorized into five classes and placed in subfolder in the directory: AMD, DR, CNV, MH, and normal. Each  represented either one of the listed ocular diseases or "normal".
2) Segmentation: The goal of the segmentation was to simplify and/or change the representation of the images to make them easier to analyze. This included border fill-in, thresholding using Otsu's Binary Thresholding to minimize the intra-class variance of the pixel intensities in the two classes while maximizing the inter-class variance, image opening, image dilation, and cropping out ROIs.
3) Data creation: This included loading the segmented image data, preprocessing, shuffling, and saving it into a HDF5 file for later use in model training and testing. The HDF5 file is an open source file format that supports large, complex, heterogeneous data.

The final dimensions of the images were tailored for the input requirements of the CNN model, in this case 128x128 pixels. The X-value in the file described the pixels, while the y-value described the class out of the five possible classes with labels 0 = normal, 1 = CNV, 2 = DR, 3 = AMD, and 4 = MH. 

#### How to Run the Code
Dependencies include Python libraries `sklearn`, `os`, `shutil`, `multiprocessing`, `cv2` (OpenCV), `numpy`, `time`, `h5py`, `tqdm`, and `tensorflow`/`keras`. To run the code:
1. Download datasets to directory. Correct directories in scripts as needed. 
2. _1folder_creater.ipynb
3. _2image_segmenter_train.py
4. _2image_segmenter_test.py
5. _3data_creater_train.py
6. _3data_creater_test.py
7. _4data_checker.py
8. _5classifier_train.py
9. _6classifier_test.py

Note that the segmenter function used in _2image_segmenter is defined in "segmenter_function.py". 

#### Decisions Made Along the Way
- **Trade-offs**: For time efficiency, aggressive image segmentation and noise reduction techniques were used, possibly omitting subtle features relevant for some disease classifications.
- **Model Complexity**: A relatively simple CNN architecture was chosen to balance between accuracy and computational efficiency. This might limit the model's ability to capture highly intricate patterns within the OCT images.

#### Example Output
The final scripts perform disease classification on a test image, displaying the predicted disease class overlaid on the original OCT image. 

#### Next steps 
Due to limited time, we performed minimal testing and optimization of the model. Next steps would include thorough testing and optimization, obtaining an accuary table and confusion matrix, and testing the model on other datasets. 

#### Citations
Data, code, and conceptual frameworks are based on existing works in ocular disease classification from OCT images.
Sources: 
- https://www.nature.com/articles/s41598-023-30853-z
- https://www.kaggle.com/datasets/paultimothymooney/kermany2018
- https://www.cell.com/cell/fulltext/S0092-8674(18)30154-5
- https://github.com/sairamadithya/OCT-classification
- https://github.com/annasophie-thein/retinal-oct-classify/tree/master

### Short-form Presentation:

#### Problem Statement
We aim to develop a model for classifying common ocular diseases from OCT images to facilitate clinical decision making. 

#### Existing Work Pulled From
Our approach is inspired by recent advancements in medical image processing and deep learning, particularly in applying CNNs for image classification and segmentation tasks in clinical areas. 

#### Your Contribution
We contributed by creating a model for preprocessing OCT images, segmenting them into regions of interest, and classifying them into five distinct categories of either age-related macular degeneration, diabetic retinopathy, macular hole, and choroidal neovascularization, or normal.

#### Tools/Methods Used
- Python for scripting.
- OpenCV for image processing tasks.
- TensorFlow/Keras for building and training the CNN model.
- Multiprocessing to expedite the image segmentation process.
- HDF5 files for efficient storage and access to large volumes of image data.

#### Issues Overcome Along the Way
- Handling varied image qualities and sizes by standardizing the input dimensions for the CNN.
- Balancing model complexity to ensure adequate learning capacity without overfitting or excessive computational demands.
- Optimizing image segmentation to highlight relevant features for disease classification while minimizing noise.
