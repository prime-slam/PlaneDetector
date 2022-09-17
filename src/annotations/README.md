# Annotations
Here you can find annotation parsers and point cloud annotators which supports formats of labeling tools used for dataset creation: CVAT and SSE.
Each annotator is divided into two classes: `Annotation` class which represents parser for annotation format file and `Annotator` class which apply annotation to point cloud instance.
Also our own format of storing RGBD labels is supported by `DiAnnotation` in `image_custom` folder, but it uses simple model of point cloud from Open3D, not the framework one.