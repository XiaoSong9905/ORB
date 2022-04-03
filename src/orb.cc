#include <cassert>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <cstdio>
#include <orb/orb.h>
#include <opencv2/core/core.hpp>

namespace orb
{

// some hyperparameter use to determine the size of image pyramid generation
const int PATCH_SIZE = 31;
const int HALF_PATCH_SIZE = 15;
const int EDGE_SIZE = 19;

// From OpenCV ORB
static int bit_pattern_31_[256*4] =
{
    printf("ORBDetectorDescriptor is constructed\n"); fflush(stdout);
}

void ORBDetectorDescriptor::detectAndCompute( cv::InputArray _image, \
                                              cv::InputArray _mask, \
                                              std::vector<cv::KeyPoint>& _keypoints, \
                                              cv::OutputArray _descriptors, \
                                              bool _useProvidedKeypoints )
{
    printf("ORBDetectorDescriptor::detectAndCompute is called\n"); fflush(stdout);
    orb_opencv->detectAndCompute( _image, _mask, _keypoints, _descriptors, _useProvidedKeypoints );
}


std::string ORBDetectorDescriptor::getDefaultName() const
{
    return (cv::FeatureDetector::getDefaultName() + ".ORBDetectorDescriptor");
}


void ORBDetectorDescriptor::computePyramid( const cv::Mat& image )
{

}

/**
 * @brief Compute key point on every pyramid level using quad tree approach
 *
 * @param keypointsPyramid: a vector of vectors. The first dimension represents the layers of 
 *                          the computed image pyramid, while the inner vectors contains the 
 *                          actual key points located at that layer.
 */
void ORBDetectorDescriptor::computeKeyPointQuadTree( std::vector<std::vector<cv::KeyPoint>> &  keypointsPyramid )
{
    keypointsPyramid.resize(nPyramidLayer);
    const float W = 30;

    // itr through the whole pyramid and process each layer.
    for (int i = 0; i < nPyramidLayer; i++) 
    {
        // get the valid image area of current layer
        const int minX = EDGE_SIZE - 3;     // 3 is the radius set for the FAST calculation 
        const int minY = EDGE_SIZE - 3;
        const int maxX = imagePyramid[i].cols - minX;
        const int maxY = imagePyramid[i].rows - minX;

        const float width = maxX - minX;
        const float height = maxY - minY;

        // get the amount of grids we have within current layer
        const int gridCols = width / W;
        const int gridRows = height / W;

        // calculate the size of each grid
        const int gridWidth = std::ceil(width / gridCols);
        const int gridHeight = std::ceil(height / gridRows);

        // reserve extra space for keypoints
        std::vector<cv::KeyPoint> keypointsToDistribute;
        keypointsToDistribute.reserve(featureAmountTarget * 10);

        // traverse all the grids
        for (int r = 0; r < gridRows; r++)
        {
            const float initialRCoord = r * gridHeight + minY;
            float terminateRCoord = initialRCoord + gridHeight + 6;

            if (initialRCoord > maxY - 3)
                continue;
            if (terminateRCoord > maxY)
                terminateRCoord = maxY;

            for (int c = 0; c < gridCols; c++)
            {
                const float initialCCoord = c * gridWidth + minX;
                float terminateCCoord = initialCCoord + gridWidth + 6;

                if (initialCCoord > maxX - 3)
                    continue;
                if (terminateCCoord > maxX)
                    terminateCCoord = maxX;

                // a vector, holds all of the keypoints within this grid.
                std::vector<cv::KeyPoint> gridKeypoints;

                // OpenCV's FAST detector, first try, with higher threshold.
                FAST(imagePyramid[i].rowRange(initialRCoord, terminateRCoord).colRange(initialCCoord, terminateCCoord),
                    gridKeypoints,
                    defaultFASTThreshold,
                    true
                );

                if (gridKeypoints.empty())
				{
					FAST(imagePyramid[i].rowRange(initialRCoord, terminateRCoord).colRange(initialCCoord, terminateCCoord),
						gridKeypoints,
						minFASTThreshold,
						true
					);
				}

                if (!gridKeypoints.empty())
                {
                    for (auto vit = gridKeypoints.begin(); vit != gridKeypoints.end(); vit++)
                    {
                        vit->pt.x += c * gridWidth;
                        vit->pt.y += r * gridHeight;

                        keypointsToDistribute.push_back(*vit);
                    }
                }
            }
        }

        // store a reference to all the keypoints that belongs to the current layer
        std::vector<cv::KeyPoint>& currentKeypoints = keypointsPyramid[i];
        currentKeypoints.reserve(featureAmountTarget);

        // TODO:: a function here to re-distribute the points into oct tree
        currentKeypoints = QuadTreeDistribute(keypointsToDistribute,
                                              minX, maxX,
                                              minY, maxY,
                                              targetFeaturePerLevel[i],
                                              i);

        // traverse all feature points and restore their coordinates under current layer
        for (int k = 0; k < currentKeypoints.size(); k++)
        {
            currentKeypoints[k].pt.x += minX;
            currentKeypoints[k].pt.y += minY;

            currentKeypoints[k].octave = i;
            currentKeypoints[k].size = PATCH_SIZE * layerScaleFactors[i];
        }
    }

    // TODO:: compute orientations for the key points of each level;
    // this step must execute AFTER the QUAD tree distribution finished so it cannot be included within 
    // previous step.
    for (int i = 0; i < nPyramidLayer; i++)
    {
        findOrientation(imagePyramid[i], keypointsPyramid[i], pyramidUBoundaries);
    }

}

int getValue(int i, float cosine, float sine){
    int xPrime = cvRound(pattern[i].x * cosine - pattern[i].y * sine)
    int yPrime = cvRound(pattern[i].x * sine + pattern[i].y * cosine);
    return center[yPrime * step + xPrime];    
}

void ORBDetectorDescriptor::computeDescriptors( const cv::Mat& image, \
                                                std::vector<cv::KeyPoint>& keypoints, \
                                                cv::Mat& descriptors, \
                                                const vector<Point>& pattern )
{
    // NOTE: when implementing, you should merge `ORBextractor.cc computeDescriptors` \
    //  and `ORBextractor.cc computeOrbDescriptor` to this one single function
    descriptors = Mat::zeros(keypoints.size(), 32, CV_8UC1);

    const float factorPi = (float)(CV_PI/180.0);
    for (int i=0; i<keypoints.size(); ++i){

        // get the angle of the keypoint (and get the cos and sin value)
        float angle = (float)keypoints[i].angle * factorPi;
        float cosine = (float)cos(angle), sine = (float)sin(angle);

        // get the center of the image
        const uchar* center = &image.at<uchar>(cvRound(kpt.pt.y), cvRound(kpt.pt.x));
        const int step = (int)image.step;

        // brief descriptors are 32 x 8bit
        // need 16 random points for 8 bit comparison
        for (int j=0; j<32; ++j, pattern+=16)
        {
            
            int t0, t1, val;
            
            t0 = getValue(0, cosine, sine); 
            t1 = getValue(1, cosine, sine);
            val = t0 < t1;							
            t0 = getValue(2, cosine, sine); 
            t1 = getValue(3, cosine, sine);
            val |= (t0 < t1) << 1;					
            t0 = getValue(4, cosine, sine); 
            t1 = getValue(5, cosine, sine);
            val |= (t0 < t1) << 2;					
            t0 = getValue(6, cosine, sine); 
            t1 = getValue(7, cosine, sine);
            val |= (t0 < t1) << 3;					
            t0 = getValue(8, cosine, sine); 
            t1 = getValue(9, cosine, sine);
            val |= (t0 < t1) << 4;					
            t0 = getValue(10, cosine, sine); 
            t1 = getValue(11, cosine, sine);
            val |= (t0 < t1) << 5;					
            t0 = getValue(12, cosine, sine); 
            t1 = getValue(13, cosine, sine);
            val |= (t0 < t1) << 6;					
            t0 = getValue(14, cosine, sine); 
            t1 = getValue(15, cosine, sine);
            val |= (t0 < t1) << 7;				

            descriptors[i][j] = (uchar)val;
        }
    }
}

// TODO:: finish this, find orientation helper
void ORBDetectorDescriptor::findOrientation(std::vector<cv::Mat> imagePyramid, std::vector<std::vector<cv::KeyPoint>>& keypointsPyramid, std::vector<int> pyramidUBoundaries)
{

}

// TODO:: finish this
std::vector<cv::KeyPoint> ORBDetectorDescriptor::QuadTreeDistribute(const std::vector<cv::KeyPoint>& keypointsToDistribute, const int& minX, const int& maxX, const int& minY, const int& maxY, const int& nFeatures, const int& level)
{
    const int nIni = std::round(static_cast<float>(maxX - minX) / (maxY - minY));
    const float hX = static_cast<float>(maxX - minX) / nIni;

    std::list<QuadTreeNode> nodesList;
    std::vector<QuadTreeNode*> initialNodesPtrs;

    initialNodesPtrs.resize(nIni);

    // initialize all quad tree nodes and push them into container
    for (int i = 0; i < nIni; i++)
    {
        QuadTreeNode qNode;

        qNode.UL = cv::Point2i(hX * static_cast<float>(i), 0);
        qNode.UR = cv::Point2i(hX * static_cast<float>(i + 1), 0);
        qNode.BL = cv::Point2i(qNode.UL.x, maxY - minY);
        qNode.BR = cv::Point2i(qNode.UR.x, maxY - minY);

        qNode.keypoints.reserve(keypointsToDistribute.size());

        nodesList.push_back(qNode);
        initialNodesPtrs[i] = &nodesList.back();
    }

	// link points to child nodes
    for (int i = 0; i < keypointsToDistribute.size(); i++)
    {
        const cv::KeyPoint& kp = keypointsToDistribute[i];
        initialNodesPtrs[kp.pt.x / hX]->keypoints.push_back(kp);
    }
    
    // traverse the quad tree nodes list, mark the nodes that no longer needs to be split, delete the nodes that did not 
    // have a keypoint associated with it.
    auto lit = nodesList.begin();
    while (lit != nodesList.end())
    {

    }

    bool bFinish = false;

    std::vector<std::pair<int, QuadTreeNode*>> sizeAndPtr2Node;
    sizeAndPtr2Node.reserve(nodesList.size() * 4);

    while (!bFinish)
    {

    }

    std::vector<cv::KeyPoint> resultKeyPoints;
    resultKeyPoints.reserve(nFeatures);

    for (auto lit = nodesList.begin(); lit != nodesList.end(); lit++)
    {

    }

    return resultKeyPoints;
}

// TODO:: finish this
void QuadTreeNode::divide(QuadTreeNode& n1, QuadTreeNode& n2, QuadTreeNode& n3, QuadTreeNode& n4)
{

}

} // namespace orb
