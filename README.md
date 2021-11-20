# PlaneDetector
Framework for plane detection algorithms approbation

## About 
This framework has pipeline for plane segmentation in point clouds and some visualization tools


What is already done:
* Pipeline for mapping rgb images planes annotations to point cloud and their visualization
* Plane outliers detection and removing using Open3D RANSAC implementation
* Basic plane detector algorithm based on Open3D RANSAC
* Metrics for plane detection: IoU, Dice, classic ones

## RGB annotations and outliers
This framework can map annotated rgb images and their depth component to the point cloud.

### How to use
Run it with `python main.py [dataset_path] --frame_number=[frame_number] --loader=[loader_name] [--annotations_path=[annotations_path] [--disable_annotation_filter_outliers]] [--algo=[plane_detection_algo_name] [--metric=[metric_name_1] --metric=[metric_name_2] ...]]`, where `dataset_path` ---
path to the dataset folder, `frame_number` --- number of the frame in dataset (enumeration started from 0),
`loader_name` --- name of the dataset loader (tum and icl_tum are available), `annotations_path` --- path to `annotations.xml` file,
`plane_detection_algo_name` --- algorithm to use for detection of planes and `metric_name_X` --- name of the metric to benchmark the chosen algorithm.

#### Example of usage:
`python main.py C:\dataset --frame_num=0 --loader=icl_tum --annotations_path=C:\annotations.xml`

### Example:
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