# M6 Project: Video Surveillance for Road Traffic Monitoring 
## Group 06

### Team members:
* _Daniel Fuentes_ - daanyfuentes@gmail.com - [DanielFuentes16](https://github.com/DanielFuentes16)
* _Daniel Yuste_ - danielyustegalvez@gmail.com - [DanielYG](https://github.com/DanielYG)
* _Sergi García Sarroca_ - g.garcia.sarroca@gmail.com - [SoftonCV](https://github.com/SoftonCV)
* _Isaac Pérez Sanz_ - rv.antoni@hotmail.com - [ipesanz](https://github.com/ipesanz)

## Project Description: 

This project is related to basic concepts and techniques related to video sequences mainly for surveillance applications. It contains the work of the authors along 5 weeks achieving different objectives: 

- The use of statistical models to estimate the background information of the video sequence.
- Use of deep learning techniques to detect the foreground.
- Use optical flow estimations and compensations. 
- Track detections. 
- Analyze system performance evaluation. 

# Week 1: Evaluation Metrics: 

Main Tasks: 

(a) Detection metrics: 

 - [] IoU & mAP for (ground truth + noise)
 - [] mAP for provided object detections

(b) Detection metrics. Temporal analysis:

 - [] IoU vs time

(c) Optical flow evaluation metrics:

 - [x] MSEN: Mean Square Error in Non-occluded areas
 - [x] PEPN: Percentage of Erroneous Pixels in Non-occluded areas 
 - [x] Analysis & Visualizations

(d) Visual representation optical flow:

 - [x] Optical Flow Plot

## Visual Results:

**Optical Flow**
![Optical Flow](https://github.com/mcv-m6-video/mcv-m6-2021-team6/blob/main/img/week1.jpg)

**Optical Flow Plot**
![Optical Plot](https://github.com/mcv-m6-video/mcv-m6-2021-team6/blob/main/img/opticalFlow_plot.jpg)

# Week 2: Bakckground Substraction: 

Main Tasks: 

Update Missing Work from Week 1 : 

(a) Detection metrics: 

 - [x] IoU & mAP for (ground truth + noise)
 - [x] mAP for provided object detections

(b) Detection metrics. Temporal analysis:

 - [x] IoU vs time

Week 2 tasks: 

(a) Background Estimation: 

 - [x] Gaussian Modelling
 - [x] Evaluation

(b) Stauffer & Grimson:

 - [x] Adaptive Modelling
 - [x] Comparison with task (a)
 
 (c) Comparison with state-of-the-art
 
 (d) Adding Color Spaces
 
## Visual Results:
 
 **Background Estimation**
 
 ![Background Example](https://github.com/mcv-m6-video/mcv-m6-2021-team6/blob/main/W2/bg_removal_a6_p0.2_6.gif)
 
 **Denoised Test**

 ![Denoise](https://github.com/mcv-m6-video/mcv-m6-2021-team6/blob/main/W2/try_dnoise.gif)

# Week 3: Segmentation, Object Detection & Tracking: 

Main Tasks: 

(a) Object detection:

 - [x] Off-the-Shelf
 - [x] Fine-tune to your data

(b) Object tracking

## Visual Results:

**Fast R-CNN**

![Fast](https://github.com/mcv-m6-video/mcv-m6-2021-team6/blob/main/img/task1_1_fast.gif)

**Mask R-CNN**

![Mask](https://github.com/mcv-m6-video/mcv-m6-2021-team6/blob/main/img/task1_1_mask.gif)

# Week 4: OpticalFlow & Tracking:

(a) Optical Flow:

 - [x] Optical Flow with Block Matching
 - [x] Off-the-Shelf Optical Flow

(b) Video stabilization:

 - [x] Video stabilization with Block Matching
 - [x] Off-the-shelf Stabilization

(c) Object Tracking: 

 - [x] Object Tracking with Optical Flow

**Visual Results:**

**Optical Flow**

![Optical](https://github.com/mcv-m6-video/mcv-m6-2021-team6/blob/main/img/week4_1.jpg)

**Video Stabilization**

![Video](https://github.com/mcv-m6-video/mcv-m6-2021-team6/blob/main/img/unnamed.gif)

# Week 5: Single-Camera & Multi-Camera Tracking:

Main tasks:

(a) Multi-target single-camera (MTSC) tracking:

 - [x] Read the Data & Evaluation description for Track 3 (Multiple-camera tracking).
 - [x] Obtain results for SEQ 1 & SEQ 4


(b)  Multi-target multi-camera (MTMC) tracking

## Visual Results:

**S03 - c010 & c011**
![seq3_c010](https://github.com/mcv-m6-video/mcv-m6-2021-team6/blob/main/img/gif_w5_s3_c10_and_11.gif)

**S04 - c038 & c040**
![seq4_c038](https://github.com/mcv-m6-video/mcv-m6-2021-team6/blob/main/img/gif_w5_s4_c38_and_c40.gif)


- Slides for the project: [T06-Google Slides](https://docs.google.com/presentation/d/1aU-1_J8-TkcwG78auCCrVxdEHGYVX8jjovoIKQD9pC4/edit#slide=id.p)

- Link to the Overleaf article (non-editable): [Group06-Overleaf]()
