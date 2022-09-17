# Dataset loaders
Framework supports 4 main datasets: ICL NUIM (both TUM-based and raw formats) and TUM for RGBD data, KITTI and CARLA Simulator for LiDaR data. 
Each loader implements interface declared in `BaseLoader`. Moreover, loaders for RGBD datasets use common base class `ImageLoader` which implements base logic connected with image loading and converting to point cloud.
For support of extra point clouds `O3DLoader` was implemented --- it helps to load point clouds from `.ply` and `.pcd` files. 