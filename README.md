### DATASCI 223 Final Project:

#### Overview of the Problem
Optical Coherence Tomography (OCT) is a non-invasive diagnostic technique that provides an in vivo cross-sectional view of the retina. This project aims to classify common ocular diseases that can be diagnosed on OCT images; age-related macular degeneration (AMD), diabetic retinopathy (DR), macular hole (MH), and choroidal neovascularization (CNV). Using a machine learning model in clinic to classify the OCT images could provide support and time optimization for the ophthalmologist reviewing the images. This process involved organizing data into training and testing sets, preprocessing images, segmenting regions of interest (ROI), creating data structures for model training, training a convolutional neural network (CNN) model, and evaluating the model on unseen data.

#### Description of the Dataset
The dataset consists of OCT images from a publicly available dataset (https://www.kaggle.com/datasets/paultimothymooney/kermany2018). 
1) Folder creation: The images were categorized into five classes and placed in subfolder in the directory: AMD, DR, CNV, MH, and normal. Each  represents either one of the listed ocular diseases or "normal".
2) Segmentation: The goal of the segmentation was to simplify and/or change the representation of the images to make them easier to analyze. This included border fill-in, thresholding using Otsu's Binary Thresholding to to minimize the intra-class variance of the pixel intensities in the two classes while maximizing the inter-class variance, image opening, image dilation, and cropping out ROIs.
3) Data creation: This included loading the segmented image data, preprocessing, shuffling , and saving it into a HDF5 file for later use in model training or testing. The HDF5 file is an open source file format that supports large, complex, heterogeneous data.

The final dimensions of the images were tailored for the input requirements of the CNN model, in this case 128x128 pixels.

#### How to Run the Code
Dependencies include Python libraries `sklearn`, `os`, `shutil`, `multiprocessing`, `cv2` (OpenCV), `numpy`, `time`, `h5py`, `tqdm`, and `tensorflow`/`keras`. To run the code:
Run code in order, i.e. 
1. _1folder_creater.ipynb
2. _2image_segmenter_test.py
1. Ensure all dependencies are installed using pip or conda.
2. Organize the OCT images into the specified directory structure.
3. Execute the scripts in the following order: data organization, image segmentation, HDF5 file creation for training and testing data, model training, and model testing.

#### Decisions Made Along the Way
- **Trade-offs**: For time efficiency, aggressive image segmentation and noise reduction techniques were used, possibly omitting subtle features relevant for some disease classifications.
- **Model Complexity**: A relatively simple CNN architecture was chosen to balance between accuracy and computational efficiency. This might limit the model's ability to capture highly intricate patterns within the OCT images.

#### Example Output
The final scripts perform disease classification on a test image, displaying the predicted disease class overlaid on the original OCT image. The output is a visual confirmation of the model's prediction capability.

#### Citations
- Data, code, and conceptual frameworks are based on existing works in ocular disease classification from OCT images. https://www.nature.com/articles/s41598-023-30853-z 

### Short-form Presentation:

#### Problem Statement
We aim to develop an automated system for classifying and segmenting ocular diseases from OCT images, facilitating early diagnosis and treatment planning.

#### Existing Work Pulled From
Our approach is inspired by recent advancements in medical image processing and deep learning, particularly in applying CNNs for image classification and segmentation tasks in clinical areas. 

#### Your Contribution
We contributed by implementing a streamlined pipeline for preprocessing OCT images, segmenting them into regions of interest, and classifying them into five distinct categories representing different ocular conditions, including a normal eye condition.

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
