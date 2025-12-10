# Overview

Here, we introduce vEMRec, a feature-based 3D rigid alignment and Gaussian-filter-based 3D elastic registration method aimed at eliminating rigid misalignment and nonlinear distortion to restore the true 3D structure of biological specimens. The method has two key stages: sequential rigid alignment to correct rotation and displacement, followed by Gaussian filtering to address nonlinear distortions. During rigid alignment, stable edge features are extracted and matched to compute transformation parameters. In the elastic registration phase, a 1D Gaussian filter is applied to decouple nonlinear distortions from natural deformations. This approach effectively corrects distortions while preserving the integrity of the biological structure, providing a strong foundation for further analysis.

# Installation

### Step 1: Create conda environment and activate
```bash
conda create -n vEMRec python=3.9
conda activate vEMRec
```

### Step 2: Install dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Pre-trained Models

For rigid alignment, you need to download `superpoint_v1.pth` from [this link](https://drive.google.com/file/d/1lwu904dn4c-7-iLwobQ5GSd0mX_8aohP/view?usp=drive_link) and place it in the `src/rigid` folder.
Similarly, download `epoch_70_checkpoint.pth.tar` from [this link](https://drive.google.com/file/d/1Y4QgWxEiAoMeBeURSmC54NgPM50R8J3o/view?usp=drive_link) and place it in the `src/rigid/edge` folder.

For elastic registration, we provide two pre-trained models, trained on the CREMI and OpenOrganelle datasets. These can be downloaded from [this folder](https://drive.google.com/drive/folders/1TaD_dh8WzLLheLXYvaRbLFOjDOovMUxi?usp=drive_link) and placed in the `pre_model` folder.


---

# Walkthrough

## 3D Rigid Alignment  
This section explains how to perform edge detection and 3D rigid alignment on a series of PNG images using vEMRec. The images should be correctly numbered (e.g., 0000.png, 0001.png, etc.) in the source folder.

### Edge Detection
```bash
cd src/rigid/edge
python evaluate_edge.py --input_path /path/to/img_folder --output_dir /path/to/mask_folder
```

### Alignment
Next, you can run the following code to perform 3D rigid alignment:
```bash
cd src/rigid
python main.py --iters 5 --input_dir /path/to/img_folder --input_mask /path/to/mask_folder --output_dir /path/to/output_folder --use_ransac 1
```

## 3D Elastic Registration
In this section, this guide will explain how to use vEMRec for 3D elastic registration.

### Testing
vEMRec supports two forms of 3D elastic registration. For small-sized images (around 1024 pixels), you can run the following code to achieve 3D elastic registration.

#### Small-sized
```bash
cd src/elastic 
python single_process.py --input_dir /path/to/img_folder --output_dir /path/to/output_folder --model_path /path/to/model 
```

For large-sized images, you can run the following code to perform 3D elastic registration.

#### Large-sized
```bash
cd src/elastic
python process_big.py --input_dir /path/to/img_folder --output_dir /path/to/output_folder --model_path /path/to/model --height large_image_height --width large_image_width --patch_sz 1024 --overlap 50
```

### Training
vEMRec estimates the displacement field between slices using an optical flow neural network and integrates the displacement field with a Gaussian filter. Here, we outline the preparation of training data and the training for the network.

#### Data Preparation
The optical flow network in vEMRec is trained on the CREMI[^1] dataset and fine-tuned on the dataset provided by OpenOrganelle[^2]. Download the training data on the CREMI website. Then, run the following code:
```bash
cd src/utils
python aug_data.py --input_file /path/to/sample_A_padded_20160501.hdf --output_dir /path/to/train_data/a_padded --size 1024 --border 80
python deform_serial.py --input_file /path/to/train_data/a_padded --output_dir /path/to/train_data/a_padded_warp --alpha 4.0 --sigma 0.08
```
> Apply the same procedure to the other two data files provided by CREMI, resulting in the training data.

For the data provided by OpenOrganelle, run the following code to generate the training dataset.
```bash
cd src/utils
sh download.sh
```

#### Train
Run the following code to train the model:
```bash
cd src/elastic
python train.py --dataset cremi --root_dataset /path/to/train_data --base_path /path/to/result
```

[^1]: [CREMI Dataset](https://cremi.org/)
[^2]: [OpenOrganelle Dataset](https://openorganelle.janelia.org/)

