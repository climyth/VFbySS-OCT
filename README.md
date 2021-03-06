# Visual field prediction from Topcon Swept-source OCT
![](https://github.com/climyth/VFbySS-OCT/blob/master/WebImages/Examplpes.jpg?raw=true)

### Features
- Inception V3, Inception V4, InceptionResnet V2 backboned deep learning model
- Predicts Humphrey's visual field 24-2 total threshold values from Topcon SS-OCT
- Predicts entire picture of visual field
- Uses combined OCT images (GCL+ thickness + RNFL thickness map);
- The best mean prediction error is 4.44 dB (InceptionResnet V2)


### Prerequisites
- python 3.6
- tensorflow >= 1.6.0
- keras 2.2.4

### How can I test OCT image?
1. Download all files from github
2. You can download weight file here (download and copy it to 'Weights' folder)
   <br/>(1) InceptionResnet V2: https://drive.google.com/open?id=14NsTiurh30967hkJDfbROvtz3LP-E43N
   <br/>(2) Inception V3: https://drive.google.com/open?id=143UIWLpgy7N_pwqBtmbgZgovVKtjTihr
   <br/>(3) Inception V4: https://drive.google.com/open?id=1ux-PmhrPk4xHfMhq_KEJxLzGLPX4mo8j
3. Open TestModel.py
4. Modify "Setup"
```python
# Setup ====================================================================================
image_file = "TestSet/OCT019.jpg"    # combined OCT image
vf_file = "TestSet/test_data.xlsm"   # ground truth visual field file
vf_sheet = "Test"  # data sheet in excel file
oct_filename_col = 0   # column number (starts from 0) of OCT filename
thv_start_col = 113    # column number (starts from 0) where THV values begin
weight_file1 = "Weights/InceptionResnet_SS_OCT.hdf5"   # InceptionResnet V2
weight_file2 = "Weights/InceptionV3_SS_OCT.hdf5"       # inception V3
weight_file3 = "Weights/InceptionV4_SS_OCT.hdf5"       # inception V4
# ===========================================================================================
```
5. Run TestModel.py;
6. You can see the popup window like below.<br/><br/>
![](https://github.com/climyth/VFbySS-OCT/blob/master/WebImages/TestOutput.JPG?raw=true)

### How can I make "combined OCT" image?
1. Download "SSOCT_Converter.exe" in "utils" folder
2. In utils folder, there are sample OCT images to generate combined OCT image.
3. Image file name must follow the rule:<br/>
   patientID_number_eye.jpg  (ex. 012345678_001_od.jpg)<br/>
   Note: _eye is '_od' for right eye, '_os' for left eye
4. Run "SSOCT_Converter.exe"<br/><br/>
![](https://github.com/climyth/VFbySS-OCT/blob/master/WebImages/OCTConverter.PNG?raw=true)
<br/><br/>
5. set source folder and output folder
6. press Run button. That's it!

### How can I train model with my own OCT images?
1. Prepare your own OCT iamges and visual field data (excel file)
2. Generate "combined OCT" images from your train set
3. In your visual field excel file, default column number is <br/>
   (1) your data must be in "Train" data sheet <br/>
   (2) 1st column must contain the list of image file names <br/> 
   (3) 2nd column must contain patients' id list <br/>
   (4) 3rd column must contain eye list (OD or OS) <br/>
   (5) visual field total threshold values must begin at 130th column by default <br/>
   (6) There must be 54 columns of total threshold values (includes two physiologic scotoma point).<br/>
4. If you want to set your own column number, modify 'DataLoad.py'.
```python
# Setup ======================================
# column number starts form 0
filename_col = 0   # image file column
pid_col = 1        # patients' id column.
eye_col = 2        # eye column ('OD' or 'OS')
thv_col = 129      # THV beginning column
# =============================================
```
5. Modify 'Setup' in 'Train.py'
```python
# Setup ====================================================================
base_model_name = "InceptionV3"   # InceptionV3, InceptionV4, InceptionResnet
base_folder = "Z:/PaperResearch/VFbySS-OCT"
vf_file = "/train_data.xlsm"
weight_save_folder = "/Weights/" + base_model_name
pretrained_weights = ""   # if no pretrained weight, just leave ""
tensorboard_log_folder = "/Logs"
# ==========================================================================
```
6. Run the Train.py
7. You can monitor loss trend in tensorboard. 
8. To prevent overfitting, we used "repeated random sub-sampling cross validation method". To do this, just repeat to run Train.py. In each run, you can set "pretrained_weights" to continue the training from last weight file.


### For Research Use Only
The performance characteristics of this product have not been evaluated by the Food and Drug Administration and is not intended for commercial use or purposes beyond research use only.
