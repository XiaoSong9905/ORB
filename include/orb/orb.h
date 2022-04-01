#pragma once

#include <string>
#include <vector>
#include <opencv2/features2d.hpp>

namespace orb
{

class QuadTreeNode
{
public:
    // a list of all keypoints belongs to this node
    std::vector<cv::KeyPoint> keypoints;

    // boundaries of this node
    cv::Point2i UL, UR, BL, BR;

    // iterator over all quad tree nodes.
    std::list<QuadTreeNode>::iterator lit;

    // a flag used to determine if a node already entered its final status.
    bool isFinal;

    /**
     * @brief The QuadNode constructor
     */
    QuadTreeNode() : isFinal(false){}

    /**
     * @brief Divide the current node into four sub areas.
     *
     * @param[in & out] n1   divided node 1
     * @param[in & out] n2   divided node 2
     * @param[in & out] n3   divided node 3
     * @param[in & out] n4   divided node 4
     */
    void divide(QuadTreeNode& n1, QuadTreeNode& n2, QuadTreeNode& n3, QuadTreeNode& n4);
};

class ORBDetectorDescriptor : public cv::Feature2D
{
    public:
        /**
         * @brief The ORB constructor
         * 
         * @param param-name param-content
         */
        ORBDetectorDescriptor( /* TODO: add parameter as needed */ );

        /**
         * @brief Destroy the ORBDetectorDescriptor object
         * 
         */
        virtual ~ORBDetectorDescriptor();

        /**
         * @brief Detect keypoint and compute the descriptor
         * 
         * @note detect() and compute() internally call detectAndCompute() in cv::Feature2D
         * @note when use the ORBDetectorDescriptor, we should call detectAndCompute() function directely
         * 
         * @param image Image
         * @param mask Mask specifying where to look for keypoints (optional). 
         *   It must be a 8-bit integer matrix with non-zero values in the region of interest.
         *   In our ORBDetectorDescriptor, mask is currently not supported
         * @param keypoints Detected keypoints. Should be empty container when input
         * @param descriptors Computed descriptors. Should be empty when input
         * @param useProvidedKeypoints use provided keypoints to compute descriptor
         *   In our ORBDetectorDescriptor, this argument should always be false since we compute keypoint
         *   inside the detectAndCompute() function
         * 
         */
        virtual void detectAndCompute( cv::InputArray image, \
                                       cv::InputArray mask, \
                                       std::vector<cv::KeyPoint>& keypoints, \
                                       cv::OutputArray descriptors, \
                                       bool useProvidedKeypoints=false ) override;

        /**
         * @brief Getname of this extractor, used in some opencv funciton
         * 
         * @return std::string 
         */
        virtual std::string getDefaultName() const override;
    
    protected:
        // TODO add helper function & helper member varaible here
        
        // Number of pyramid layer
        int nPyramidLayer;

        // default OpenCV FAST detection threshold
        int defaultFASTThreshold;

        // min OpenCV FAST detection threshold
        int minFASTThreshold;

        // target amount of features we want to extract from one image 
        int featureAmountTarget;

        // the scaleFactor applied on different layers within the image pyramid.
        double scaleFactor;

        // Image at each pyramid level
        std::vector<cv::Mat> imagePyramid;

        // BRISK random pattern
        std::vector<cv::point> briskPattern;

        // a look-up table for the scale factors of each layer within the pyramid.
        std::vector<float> layerScaleFactors;

        std::vector<int> targetFeaturePerLevel;

        // a look-up table for U axis boundary size on each level 
        std::vector<int> pyramidUBoundaries;

        /**
         * @brief Build image pyramic
         * 
         * @param image Source image of original size
         */
        void computePyramid( const cv::Mat& image );

        /**
         * @brief Compute key point on every pyramid level using quad tree approach
         * 
         * @param keypointsPyramid keypoints per each pyramid level
         */
        void computeKeyPointQuadTree( std::vector<std::vector<cv::KeyPoint>& keypointsPyramid ); 

        /**
         * @brief Compute descriptor for each pyramic layer
         * 
         * @param image Image at specific pyramid layer
         * @param keypoints Keypoints detected at that layer
         * @param descriptors output descriptors
         */
        void computeDescriptors( const cv::Mat& image, \
                                 std::vector<cv::KeyPoint>& keypoints, \
                                 cv::Mat& descriptors );

        void findOrientation(std::vector<cv::Mat> imagePyramid, std::vector<std::vector<cv::KeyPoint>>& keypointsPyramid, std::vector<int> pyramidUBoundaries);

        std::vector<cv::KeyPoint> QuadTreeDistribute(const std::vector<cv::KeyPoint>& keypointsToDistribute, const int &minX, const int &maxX, const int &minY, const int &maxY, const int &nFeatures, const int &level);
};

};