# ORB
> OpenCV compatable ORB feature detector used in ORB-SLAM1/2/3
>
> This repo is part of UC Brekeley OpenARK Team2 capstone project



## File structure

```
# Code header and implementation
include/orb
src/

# Example using customized orb feature
examples/

# Compare with opencv ORB and brisk
compare/
```



## Install Dependency

1. CMake
2. OpenCV 3.4.0 from github and build with CMake
3. (optional, for compare only) BRISK. You can download brisk from here [link](http://github.com/xiaosong9905/brisk) where we have modify the brisk to be compatable with OpenCV 3.4.0 



## Build library

```shell
cd orb # cd into current folder
mkdir build 
cd build

# Config & build package
cmake ..
make -j 10

# Install package
sudo make install 
```



## Reference

1. BRISK OpenCV compatable package [link](https://github.com/ILLIXR/BRISK)
2. ORB-SLAM [link](https://github.com/UZ-SLAMLab/ORB_SLAM3)
3. OpenCV [link](https://github.com/opencv/opencv)