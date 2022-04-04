## TODO List

1. computeOrientation() & constructor setting
2. QuadTreeDistributePerPyramidLevel() : lots of work
3. Testing functionality
    1. compare with opencv orb on multiple image. Write a compare.cc under example/ and test detected keypoints with multiple images (in door, outdoor, low light, high light)
    2. compare with brisk (the uk package) on multiple image. same setting as above. 
    3. run our ORB inside okvis, validate algorithm work, compare with previous okvis result (feature detected, tracking accuracy in different condition)
    3. run our ORB inside OpenARK. Same comparision above.