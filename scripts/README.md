# Scripts
Here you can find a lot of useful scripts implemented during plane detection benchmark creation.

## Content
All scripts are divided into 5 groups:
* Scripts for RGBD-based sequences generation --- `rgbd_annotations` folder
* Scripts for CARLA simulator-based sequences generation --- `carla` folder
* Scripts for KITTI-based sequences generation --- `kitti` folder
* Scripts for metrics evaluation --- `metric_eval` folder
* Other useful scripts for file conversions and so on --- `utils` folder

## RGBD-based sequences
RGBD-based sequences were generated from TUM and ICL NUIM datasets annotated using CVAT tool for RGB images. In `rgbd_annotations` you can find next scripts:
* `rgbd_sequence_builder.py` --- script for generation sequnce annotation in our custom image format. It concats parts of CVAT annotations in one (overriding duplicated frames by the next loaded annotation part).
As different pices of CVAT annotations can't use same labels, this script provides format for hand-made association or can do it using `NaiveIoUAssociator`.
Moreover, during associations loading to point clouds outliers are filtered and for TUM dataset planes with huge amount of zero-depth points or situated too far from camera can be removed from annotated ones too.
* `outliers_remover.py` --- script for postprocessing of annotations in our custom image format. It uses connected components algorithm to fix labeler mistakes with miss clicked points. It was applied only to ICL NUIM Living room sequence.
* `labeler_assistance` folder --- contains scripts used to help labeler
  * `cvat_annotation_fixer.py` --- was used to fix z level of annotations with overlapped planes, labeler selected annotations which have to be on top by hand and this tool changes their z level just in `xml` file of CVAT anotation.
  * `image_separator.py` --- moves each n-th frame from sequence to sparate folder
  * `tum_fix_skipped_depth.py` --- remove pixels (fill with black color) from RGB image from TUM dataset in places where on depth image pixels have zero depth. It helps to reduce amount of annotations for labeler.

## CARLA Simulator
CARLA simulator was used as a source of synthetic LiDaR point clouds with ability of auto plane detection. Scripts for it can be found in `carla` folder.
* `modified_triangles.py` --- script for auto plane detection in CARLA meshes. It loads mesh, groups (uses DBScan for normals) triangls by normals (like planar groups), divides (uses DBScan for vertices) them by connectivity (like planar instances --- planes) and then converts mesh to point cloud with already prepared labels which are saved as `npy` file.
* `pointclouds_processor.py` --- script for frames annotation. It gathers frames into single map, annotates it using segmented meshes and then map points with annotations back to frames.

## KITTI
KITTI dataset was used as a source of real LiDaR point clouds and was annotated using SSE by map parts (map was built from frames using gt poses, and then it was divided into rectangles). Scripts for it can be found in `kitti` folder.
* `reprocess_annot.py` --- script for appending KITTI semantic labels to prepared annotations
* `map_builder.py` --- script for creating map parts for labeler. It concats frames using gt poses into single map, loads existing annotations to it (first few frames were annotated using ranges), splits map into rectangles and save them as segmented point clouds in SSE format (with normalized labels and description which frames are parts of this rectangle)
* `map_to_frame_annotator.py` --- script for annotating frames using prelabelled map parts. It concats previously annotated rectangle parts of map and then map annotations to each frame using gt poses. Also, labels can be filtered with DBScan algorithms --- it was used to remove miss clicks from annotations.
* `labeler_assistance` folder --- contains scripts used to help labeler
  * `label_mapper.py` --- helps to understand which piece of annotation provides this global label to map
  * `sse_labels_generator.py` --- helps to generate a lot of different labels for SSE in required format for this tool configuration

## Metrics evaluation
For convenient metrics evaluation on planes detected by different algorithms a few scripts were developed. They can be found in `metric_eval` folder.
* `with_detection` folder --- contains `eval.py` miniapp which can load frames from dataset, prepare them for plane detection, run Docker images with algorithms on this data and then evaluate metrics. **Warning**: it is _**DEPRECATED**_ solution! If you want to launch Docker files use modern solution at **//TODO: link to summer school repo** or prepare detection results yourself and use script from `without_detection` folder.
* `without_detection` folder --- contains `metric_eval.py` and `ml_eval.py` scripts which can calculate metrics from EVOPS package on annotations in our format and detection results in `npy` format or ML output format respectively. Results are stored in the csv file.

## Utils
Other useful scripts are gathered in `utils` folder.
* `img_merger.py` --- script for combining images in matrix, it was used to generate image in the paper
* `npy_to_img.py` --- script for converting annotations in npy format to our custom image format (use only for RGBD-based data)
* `pcd_to_bin.py` --- script for converting point clouds from `pcd` to `bin` format (used in KITTI and SSE)
* `pcd_to_npy.py` --- script for extracting annotations from colored point cloud in `pcd` format to separate `npy` file
* `pcd_viewer.py` --- script for pcd visualization where data is `pcd` + `npy` format
* `unique_planes_counter.py` --- script for counting total amount of unique planes in sequence and average count of planes in frame. Works with `npy` files.