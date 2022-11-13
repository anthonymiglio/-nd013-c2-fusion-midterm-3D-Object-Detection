# Writeup: 3D Object Detection

In this project, a fuse measurements implementation from LiDAR and camera and track vehicles over time. It uses real-world data from the Waymo Open Dataset, detect objects in 3D point clouds and apply an extended Kalman filter for sensor fusion and tracking.

Object detection: In this part, a deep-learning approach is used to detect vehicles in LiDAR data based on a birds-eye view perspective of the 3D point-cloud. Also, a series of performance measures is used to evaluate the performance of the detection approach. The tasks in this part make up the mid-term project.

To run this project in the workspace folder: 
```
python loop_over_dataset.py
```
And in Set parameters and perform initializations, select an **exercise** to run: 'ID_S4_EX3' # 'ID_S1_EX1', 'ID_S1_EX2', 'ID_S2_EX1', 'ID_S2_EX2', 'ID_S2_EX3', 'ID_S3_EX1', 'ID_S3_EX2', 'ID_S4_EX1', 'ID_S4_EX2', 'ID_S4_EX3'
<img src="/img/exercise_selection.png"/>


## Step 1 Compute Lidar Point-Cloud from Range Image
Typically, the sensor data provided by a lidar scanner is represented as a 3d point cloud, where each point corresponds to the measurement of a single lidar beam. Each point is described by a coordinate in $(x, y, z)$ and additional attributes such as the intensity of the reflected laser pulse or even a secondary return caused by partial reflection at object boundaries.

An alternative form of representing lidar scans are range images. This data structure holds 3d points as a 360 degree "photo" of the scanning environment with the row dimension denoting the elevation angle of the laser beam and the column dimension denoting the azimuth angle. With each incremental rotation around the z-axis, the lidar sensor returns a number of range and intensity measurements, which are then stored in the corresponding cells of the range image.

Mapping of 3d points into range image cells
<img src="/img/c1-5-img2.png" width="700"/>
 
### 1.1 Visualize range image channels
In the Waymo Open dataset, lidar data is stored as a range image. Therefore, this task is about extracting two of the data channels within the range image, which are "range" and "intensity", and convert the floating-point data to an 8-bit integer value range.

**ID_S1_EX1**: A detailed description of all required steps can be found in the code: ``show_range_image`` function, on ``student/objdet_pcl.py`` file
- Extract lidar data and range image for the roof-mounted lidar
- Extract the range and the intensity channel from the range image
- Convert range image “range” channel to 8bit
- Convert range image “intensity” channel to 8bit
- Crop range image to +/- 90 deg. left and right of the forward-facing x-axis
- Stack cropped range and intensity image vertically and visualize the result using OpenCV

<img src="/img/ID_S1_EX1.png"/>

### 1.2 Visualize point-cloud
The goal of this task is to use the Open3D library to display the lidar point-cloud in a 3d viewer
in order to develop a feel for the nature of lidar point-clouds.
**ID_S1_EX2**: A detailed description of all required steps can be found in the code: ``show_pcl`` function, on ``student/objdet_pcl.py`` file
- Visualize the point-cloud using the open3d module

<img src="/img/ID_S1_EX2.png"/>

- Find 10 examples of vehicles with varying degrees of visibility in the point-cloud

| pcl 01                             | pcl 02                             | pcl 03                             | pcl 04                             | pcl 05                             |
|:----------------------------------:|:----------------------------------:|:----------------------------------:|:----------------------------------:|:----------------------------------:|
| <img src="/img/ID_S1_EX2_01.png"/> | <img src="/img/ID_S1_EX2_02.png"/> | <img src="/img/ID_S1_EX2_03.png"/> | <img src="/img/ID_S1_EX2_04.png"/> | <img src="/img/ID_S1_EX2_05.png"/> |
| pcl 06                             | pcl 07                             | pcl 08                             | pcl 09                             | pcl 10                             |
| <img src="/img/ID_S1_EX2_06.png"/> | <img src="/img/ID_S1_EX2_07.png"/> | <img src="/img/ID_S1_EX2_08.png"/> | <img src="/img/ID_S1_EX2_09.png"/> | <img src="/img/ID_S1_EX2_10.png"/> |

<!-- |                                    |                                    | -->
<!-- |:----------------------------------:|:----------------------------------:| -->
<!-- | <img src="/img/ID_S1_EX2_01.png"/> | <img src="/img/ID_S1_EX2_02.png"/> | -->
<!-- | <img src="/img/ID_S1_EX2_03.png"/> | <img src="/img/ID_S1_EX2_04.png"/> | -->
<!-- | <img src="/img/ID_S1_EX2_05.png"/> | <img src="/img/ID_S1_EX2_06.png"/> | -->
<!-- | <img src="/img/ID_S1_EX2_07.png"/> | <img src="/img/ID_S1_EX2_08.png"/> | -->
<!-- | <img src="/img/ID_S1_EX2_09.png"/> | <img src="/img/ID_S1_EX2_10.png"/> | -->


- Try to identify vehicle features that appear stable in most of the inspected examples and describe them:

| Vehicle features on pcl 02                  | Vehicle features on pcl 06                  |
|:-------------------------------------------:|:-------------------------------------------:|
| <img src="/img/ID_S1_EX2_02_features.png"/> | <img src="/img/ID_S1_EX2_06_features.png"/> |

## Step 2 Create Birds-Eye View from Lidar PCL
### 2.1 Convert sensor coordinates to bev-map coordinates
The goal of this task is to perform the first step in creating a birds-eye view (BEV) perspective of the lidar point-cloud. Based on the (x,y)-coordinates in sensor space, you must compute the respective coordinates within the BEV coordinate space so that in subsequent tasks, the actual BEV map can be filled with lidar data from the point-cloud.
**ID_S2_EX1**: A detailed description of all required steps can be found in the code: ``bev_from_pcl function``, on ``student/objdet_pcl.py`` file
- Convert coordinates in x,y [m] into x,y [pixel] based on width and height of the bev map

<img src="/img/ID_S2_EX1.png"/>

### 2.2 Compute intensity layer of bev-map
The goal of this task is to fill the "intensity" channel of the BEV map with data from the pointcloud. In order to do so, you will need to identify all points with the same (x,y)-coordinates within the BEV map and then assign the intensity value of the top-most lidar point to the respective BEV pixel.
**ID_S2_EX2**: A detailed description of all required steps can be found in the code: ``bev_from_pcl function``, on ``student/objdet_pcl.py`` file
- Assign lidar intensity values to the cells of the bird-eye view map
- Adjust the intensity in such a way that objects of interest (e.g. vehicles) are clearly visible

| Intensity layer of bev-map                    | Intensity Bev-map: Car in detail                     |
|:---------------------------------------------:|:----------------------------------------------------:|
| <img src="/img/ID_S2_EX2_intensity_map.png"/> | <img src="/img/ID_S2_EX2_intensity_map_detail.png"/> |

### 2.3 Compute height layer of bev-map
The goal of this task is to fill the "height" channel of the BEV map with data from the pointcloud. In order to do so, please make use of the sorted and pruned point-cloud
lidar_pcl_top from the previous task and normalize the height in each BEV map pixel by the difference between max. and min. height which is defined in the configs structure.
**ID_S2_EX3**: A detailed description of all required steps can be found in the code: ``bev_from_pcl function``, on ``student/objdet_pcl.py`` file
- Make use of the sorted and pruned point-cloud lidar_pcl_top from the previous task
- Normalize the height in each BEV map pixel by the difference between max. and min. height
- Fill the "height" channel of the BEV map with data from the point-cloud

| Height layer of bev-map                    | Height Bev-map: Car in detail                     |
|:------------------------------------------:|:-------------------------------------------------:|
| <img src="/img/ID_S2_EX3_height_map.png"/> | <img src="/img/ID_S2_EX3_height_map_detail.png"/> |

## Step 3 Model-based Object Detection in BEV Image 
### 3.1 Add a second model from a GitHub repo
The goal of this task is to illustrate how a new model can be integrated into an existing framework. Clone the repo [Super Fast and Accurate 3D Object Detection based on 3D LiDAR Point Clouds](https://github.com/maudzung/SFA3D).
Also note that in this project, we are only focussing on the detection of vehicles, even though the Waymo Open dataset contains labels for other road users as well.

**ID_S3_EX1**: A detailed description of all required steps can be found in the code: detect_objects , load_configs_model and create_model function, on student/objdet_detect.py file
- In addition to Complex YOLO, extract the code for output decoding and post-processing from the GitHub repo.

Result: on the local variable: "detections" in the function ``detect_objects`` on the code

### 3.2 Extract 3D bounding boxes from model response
As the model input is a three-channel BEV map, the detected objects will be returned with coordinates and properties in the BEV coordinate space. Thus, before the detections can move along in the processing pipeline, they need to be converted into metric coordinates in vehicle space. This way all detections have the format [1, x, y, z, h, w, l, yaw], where 1 denotes the class id for the object type vehicle.

**ID_S3_EX2**: A detailed description of all required steps can be found in the code: ``detect_objects`` function, on ``student/objdet_detect.py`` file
- Transform BEV coordinates in [pixels] into vehicle coordinates in [m]
- Convert model output to expected bounding box format [class-id, x, y, z, h, w, l, yaw]

<img src="/img/ID_S3_EX2_labels_vs_detected_objects.png"/>

## Step 4 Performance Evaluation for Object Detection
### 4.1 Compute intersection-over-union (IOU) between labels and detections

The goal of this task is to find pairings between ground-truth labels and detections, so that we can determine wether an object has been (a) missed (false negative), (b) successfully detected (true positive) or (c) has been falsely reported (false positive). Based on the labels within the Waymo Open Dataset, your task is to compute the geometrical overlap between the bounding boxes of labels and detected objects and determine the percentage of this overlap in relation to the area of the bounding boxes. A default method in the literature to arrive at this value is called intersection over union

**ID_S4_EX1**: A detailed description of all required steps can be found in the code: ``detect_objects`` function, on ``student/objdet_eval.py`` file
- For all pairings of ground-truth labels and detected objects, compute the degree of geometrical overlap
- The function tools.compute_box_corners returns the four corners of a bounding box which can be used with the Polygon structure of the Shapely toolbox
- Assign each detected object to a label only if the IOU exceeds a given threshold
- In case of multiple matches, keep the object/label pair with max. IOU
- Count all object/label-pairs and store them as “true positives”

Result: on the local variables: ``ious`` and ``center_devs`` on the code

### 4.2 Compute false-negatives and false-positives
Based on the pairings between ground-truth labels and detected objects, the goal of this task is to determine the number of false positives and false negatives for the current frame. After
all frames have been processed, an overall performance measure will be computed based on the results produced in this task.

**ID_S4_EX2**: A detailed description of all required steps can be found in the code: ``detect_objects`` function, on ``student/objdet_eval.py`` file
- Compute the number of false-negatives and false-positives based on the results from IOU and the number of ground-truth labels

Result: on the local variable: ``det_performance`` on the code

### 4.3 Compute precision and recall
After processing all the frames of a sequence, the performance of the object detection algorithm shall now be evaluated. To do so in a meaningful way, the two standard measures
"precision" and "recall" will be used, which are based on the accumulated number of positives and negatives from all frames.

**ID_S4_EX3**: A detailed description of all required steps can be found in the code: ``detect_objects`` function, on ``student/objdet_eval.py`` file
- Compute “precision” over all evaluated frames using true-positives and false-positives
- Compute “recall” over all evaluated frames using true-positives and false-negatives

Result: ``precision = 0.9724137931034482, recall = 0.9215686274509803``

<img src="/img/ID_S4_EX3.png"/>


## Do you see any benefits in camera-lidar fusion compared to lidar-only tracking (in theory and in your concrete results)?
In theory, camera-lidar fusion has concrete benefits compared to Lidar-only tracking. Using a camera only has shortcomings like in-depth error measurement, darkness, or weather conditions like fog and rain. Yet, it has the benefit of image classification. Furthermore, Lidar performs very well in darkness and bad weather with minimal in-depth error. Merging both signals eliminates disadvantages.

## Which challenges will a sensor fusion system face in real-life scenarios? Did you see any of these challenges in the project?
In real-life scenarios, even sensor fusion faces difficulty in eliminating all challenges. First, the budget will limit the sensor quality and capabilities. Moreover, Sensor calibration is a must to rely on a proper perception system, and occlusions can occur. Even though this is a simulation, one way or another, these challenges were faced while coding the project.

## Can you think of ways to improve your tracking results in the future?
- Increase the number of sensors in the dataset to improve the tracking results.
- Use a different pre-trained neural network and compare the tracking performances. Yolo_v7 or SSD.
