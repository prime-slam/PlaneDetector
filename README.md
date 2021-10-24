# PlaneDetector
Framework for plane detection algorithms approbation

## About 
This framework has pipeline for plane segmentation in point clouds and some visualization tools


What is already done:
* Pipeline for mapping rgb images planes annotations to point cloud and their visualization
* Plane outliers detection and removing using Open3D RANSAC implementation

## RGB annotations and outliers
This framework can map annotated rgb images and their depth component to the point cloud.

### How to use
Run it with `python main.py [depth_path] [annotations_path] --annotation_frame_number=[frame_number]`, where `depth_path` ---
path to the depth image, `annotations_path` --- path to `annotations.xml` file and `frame_number` --- number of the frame in annotations for which point cloud is generated (enumeration started from 0)

#### Example of usage:
`python main.py C:\depth\0.png C:\Desktop\annotations.xml --annotation_frame_number=0`

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