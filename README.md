# PlaneDetector
Framework for plane detection algorithms approbation

## About 
This framework introduces pipeline for plane segmentation in point clouds recorded by LiDaRs and RGBD cameras.
It also provides all necessary scripts to prepare plane segmentation and association dataset for both RGBD and LiDaR data.

[//]: # (What is already done:)
[//]: # (* Pipeline for mapping rgb images planes annotations to point cloud and their visualization)
[//]: # (* Plane outliers detection and removing using Open3D RANSAC implementation)
[//]: # (* Basic plane detector algorithm based on Open3D RANSAC)
[//]: # (* Metrics for plane detection: IoU, Dice, classic ones)

## Framework details
Commonly framework consists of three parts: 
1. Common code for RGBD and LiDaR data processing as point clouds (including annotations, associations, detections, metrics and dataset loaders)
2. Set of scripts and miniapps for dataset preparation and metrics evaluation
3. Set of Python notebooks with examples and MVPs

## Main features
There are 5 main features in the framework: different datasets format loading, annotation tools format parsing, point clouds association, planes detection and metrics evaluation

### 1. Dataset formats
Framework supports 4 main datasets: ICL NUIM (both TUM-based and raw formats) and TUM for RGBD data, KITTI and CARLA Simulator for LiDaR data. 
Special loaders were implemented for each dataset format --- they can be found at `src/loaders`.
Each loader implements `BaseLoader` interface.

### 2. Annotations
Two annotation tools were used to create plane segmentation dataset: CVAT for RGBD data and SSE for LiDaR data. Framework supports formats for both of them.
Implementation of `BaseAnnotator` which is an interface for point cloud annotations for both tools can be found in `src/annotations/cvat` and `src/annotations/sse` respectively. 
Also our own format of storing RGBD labels is supported by `src/annotations/image_custom/DiAnnotation` but it uses simple model of point cloud from Open3D, not the framework one.

Except loading annotations from different formats framework also can filter outlines from it using RANSAC algorithm from Open3D.
#### Example of annotations loading with outliers filtering:
| Original depth image     |  Annotated RGB image  |
|------------|-------------| 
|![](https://github.com/DmiitriyJarosh/PlaneDetector/blob/main/examples/original_depth.png?raw=true)|![](https://github.com/DmiitriyJarosh/PlaneDetector/blob/main/examples/annotations.png?raw=true) |

Framework build point cloud based on depth image and map annotations to it using different colors.

| Result of mapping |
|------------| 
| ![](https://github.com/DmiitriyJarosh/PlaneDetector/blob/main/examples/annotation_without_outliers_example.png?raw=true)|

As you can see, some objects aren't well annotated: lamp and plant on the left side, for example.
To fix such mistakes in annotation outlier detector can be used. This framework has one based on RANSAC algorithm.

| Result of mapping with outliers extraction |
|--------------|
| ![](https://github.com/DmiitriyJarosh/PlaneDetector/blob/main/examples/annotation_with_outliers_example.png?raw=true)|

As you can see now, plant and lamp are marked with black --- as not annotated objects, but not as a wall behind them!

### 3. Point clouds association
The problem of association planes in point clouds requires extra research, so this framework has only one implementation of naive IoU algorithm in `src/associators`.
More detailed review of such algorithms, implementation of the best ones and framework for their evaluation and comparison can be found at **//TODO: link to ivan repo**.

### 4. Plane detection
The problem of detection planes in point clouds requires extra research, so this framework has only one implementation of naive RANSAC algorithm based on Open3D library in `src/detectors`.
Most popular algorithms for plane detection were wrapped with Docker containers and can be found at **//TODO: link to dockers repo**. 
Framework for their evaluation with quality and perfomance measurements can be found at **//TODO: link to summer school repo**.

### 5. Metrics evaluation
There are a few metrics that can be used to calculate quality of plane detection. Mainly they are from spheres of instance and panoptic segmentation.
Framework contains first version of metrics set in `src/metrics`, but they are **deprecated** ones. For modern implementation of large metrics set look at EVOPS library at **//TODO: link to EVOPS**.

### Sample pipeline
To combine all this features in single pipeline you can use main driver script in `src/main.py`.
It loads point clouds from specified dataset, apply annotations in the selected format with filtered outliers to them, detect planes on depth image and then compare the results using metrics.

#### How to use
Run it with `python main.py [dataset_path] --frame_number=[frame_number] --loader=[loader_name] [--annotations_path=[annotations_path] [--disable_annotation_filter_outliers]] [--algo=[plane_detection_algo_name] [--metric=[metric_name_1] --metric=[metric_name_2] ...]]`, where `dataset_path` ---
path to the dataset folder, `frame_number` --- number of the frame in dataset (enumeration started from 0),
`loader_name` --- name of the dataset loader (tum and icl_tum are available), `annotations_path` --- path to `annotations.xml` file,
`plane_detection_algo_name` --- algorithm to use for detection of planes and `metric_name_X` --- name of the metric to benchmark the chosen algorithm.

**Example of usage**:
`python main.py C:\dataset --frame_num=0 --loader=icl_tum --annotations_path=C:\annotations.xml`


## Scripts and miniapps
This framework also contains a lot of useful scripts implemented during plane detection benchmark creation. To overview them take a look at `scripts` folder.
