# ORB
OpenCV compatable ORB feature detector used in ORB-SLAM1/2/3

## File structure
```
# Code header and implementation
include/orb
src/

# Example using customized orb feature
examples/
```

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