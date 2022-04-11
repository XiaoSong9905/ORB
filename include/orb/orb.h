#pragma once

#include <string>
#include <vector>
#include <list>
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
    bool is_final;

    /**
     * @brief The QuadNode constructor
     */
    QuadTreeNode() : is_final(false){}

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


/**
 * @brief ORB feature detector & descriptor
 * 
 * @note This class contain internal state information. Should avoid using multithread to call detectAndCompute()
 * 
 */
class ORBDetectorDescriptor : public cv::Feature2D
{
    public:

        /**
         * @brief The ORB constructor
         * 
         * @param _num_features : total number of features to extract
         * @param pyramid_scale_factor : scale factor between pyramid layer
         * @param _pyramid_num_level : number of pyramid layer
         * @param _fast_default_threshold : default threshold used by fast
         * @param _fast_min_threshold : adaptive min threshold used by fast
         */
        ORBDetectorDescriptor( int _num_features, \
                               float pyramid_scale_factor, \
                               int _pyramid_num_level, \
                               int _fast_default_threshold, \
                               int _fast_min_threshold );

        /**
         * @brief Destroy the ORBDetectorDescriptor object
         * 
         */
        virtual ~ORBDetectorDescriptor() = default;

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
         * @brief Get name of this extractor, used in some opencv funciton
         * 
         * @return cv::String 
         */
        virtual cv::String getDefaultName() const override;
    
    protected:
        /* Feature extraction related settings */
        
        // total number of features extracted
        int num_features;

        /* Image pyramid related settings */

        // Number of pyramid layers
        int pyramid_num_level;

        // scale factor between each image pyramid layer 
        // double pyramid_scale_factor;

        // number of features for each pyramid layer
        std::vector<int> pyramid_num_features_per_level;

        // Image at each pyramid level
        std::vector<cv::Mat> pyramid_scaled_image;

        // Scaling factor (and inverse) for each pyramic layer
        std::vector<float> pyramid_scale_factors;
        std::vector<float> pyramid_inv_scale_factors;

        /* BRISK related setting */
        std::vector<cv::Point> brisk_random_pattern;
    
        /* FAST related setting */
        // default OpenCV FAST detection threshold
        int fast_default_threshold;

        // min OpenCV FAST detection threshold
        // this is used in adaptive fast
        int fast_min_threshold;

        /* Rotation related setting */
        // a look-up table for U axis boundary size on each level 
        std::vector<int> patch_umax;

    protected:

        /**
         * @brief Build image pyramic
         * 
         * @param image Source image of original size
         */
        void computePyramid( const cv::Mat& image );

        /**
         * @brief Compute key point on every pyramid level using quad tree approach
         * 
         * @param pyramid_keypoints_per_level keypoints per each pyramid level
         */
        void computeFASTKeyPointQuadTree( \
            std::vector<std::vector<cv::KeyPoint>> &pyramid_keypoints_per_level ); 

        /**
         * @brief Find IC_Angle for every keypoints in a given pyramid layer
         * 
         * @param pyramid_keypoints_per_level keypoints per each pyramid level
         */
        void computeOrientation( \
            std::vector<std::vector<cv::KeyPoint>>& pyramid_keypoints_per_level );

        /**
         * @brief Compute descriptor for each pyramic layer
         * 
         * @param image Image at specific pyramid layer
         * @param keypoints Keypoints detected at that layer
         * @param descriptors output descriptors
         */
        void computeBRISKDescriptorsPerPyramidLevel( \
            const cv::Mat& image, \
            std::vector<cv::KeyPoint>& keypoints, \
            cv::Mat& descriptors );

        /**
         * @brief Use quad tree to distribute all keypoints uniformly across current layer
         * 
         * @param keypoints_to_distribute_level_i: keypoint wait to distribute
         * @param keypoints_level_i destination container
         * @param roi_minmax_xy : ROI of image
         * @param num_feature_level_i : desired number of feature at this layer
         * @param level_i : pyramid level
         */
        void QuadTreeDistributePerPyramidLevel( \
            const std::vector<cv::KeyPoint>& keypoints_to_distribute_level_i, \
                  std::vector<cv::KeyPoint>& keypoints_level_i, \
            int roi_min_x, int roi_max_x, \
            int roi_min_y, int roi_max_y, \
            int num_feature_level_i, \
            int level_i );
};

}