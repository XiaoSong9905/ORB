#include <cassert>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <orb/orb.h>

namespace orb
{

// some hyperparameter use to determine the size of image pyramid generation
constexpr int EXTRACTOR_PATCH_SIZE = 31;
constexpr int HALF_EXTRACTOR_PATCH_SIZE = 15;

// Edge for compute image pyramid
constexpr int PYRAMID_EDGE_SIZE = 19;

// From OpenCV ORB
// static, so that only visible within this file
static int bit_pattern_31_[256*4] =
{
    8,-3, 9,5/*mean (0), correlation (0)*/,
    4,2, 7,-12/*mean (1.12461e-05), correlation (0.0437584)*/,
    -11,9, -8,2/*mean (3.37382e-05), correlation (0.0617409)*/,
    7,-12, 12,-13/*mean (5.62303e-05), correlation (0.0636977)*/,
    2,-13, 2,12/*mean (0.000134953), correlation (0.085099)*/,
    1,-7, 1,6/*mean (0.000528565), correlation (0.0857175)*/,
    -2,-10, -2,-4/*mean (0.0188821), correlation (0.0985774)*/,
    -13,-13, -11,-8/*mean (0.0363135), correlation (0.0899616)*/,
    -13,-3, -12,-9/*mean (0.121806), correlation (0.099849)*/,
    10,4, 11,9/*mean (0.122065), correlation (0.093285)*/,
    -13,-8, -8,-9/*mean (0.162787), correlation (0.0942748)*/,
    -11,7, -9,12/*mean (0.21561), correlation (0.0974438)*/,
    7,7, 12,6/*mean (0.160583), correlation (0.130064)*/,
    -4,-5, -3,0/*mean (0.228171), correlation (0.132998)*/,
    -13,2, -12,-3/*mean (0.00997526), correlation (0.145926)*/,
    -9,0, -7,5/*mean (0.198234), correlation (0.143636)*/,
    12,-6, 12,-1/*mean (0.0676226), correlation (0.16689)*/,
    -3,6, -2,12/*mean (0.166847), correlation (0.171682)*/,
    -6,-13, -4,-8/*mean (0.101215), correlation (0.179716)*/,
    11,-13, 12,-8/*mean (0.200641), correlation (0.192279)*/,
    4,7, 5,1/*mean (0.205106), correlation (0.186848)*/,
    5,-3, 10,-3/*mean (0.234908), correlation (0.192319)*/,
    3,-7, 6,12/*mean (0.0709964), correlation (0.210872)*/,
    -8,-7, -6,-2/*mean (0.0939834), correlation (0.212589)*/,
    -2,11, -1,-10/*mean (0.127778), correlation (0.20866)*/,
    -13,12, -8,10/*mean (0.14783), correlation (0.206356)*/,
    -7,3, -5,-3/*mean (0.182141), correlation (0.198942)*/,
    -4,2, -3,7/*mean (0.188237), correlation (0.21384)*/,
    -10,-12, -6,11/*mean (0.14865), correlation (0.23571)*/,
    5,-12, 6,-7/*mean (0.222312), correlation (0.23324)*/,
    5,-6, 7,-1/*mean (0.229082), correlation (0.23389)*/,
    1,0, 4,-5/*mean (0.241577), correlation (0.215286)*/,
    9,11, 11,-13/*mean (0.00338507), correlation (0.251373)*/,
    4,7, 4,12/*mean (0.131005), correlation (0.257622)*/,
    2,-1, 4,4/*mean (0.152755), correlation (0.255205)*/,
    -4,-12, -2,7/*mean (0.182771), correlation (0.244867)*/,
    -8,-5, -7,-10/*mean (0.186898), correlation (0.23901)*/,
    4,11, 9,12/*mean (0.226226), correlation (0.258255)*/,
    0,-8, 1,-13/*mean (0.0897886), correlation (0.274827)*/,
    -13,-2, -8,2/*mean (0.148774), correlation (0.28065)*/,
    -3,-2, -2,3/*mean (0.153048), correlation (0.283063)*/,
    -6,9, -4,-9/*mean (0.169523), correlation (0.278248)*/,
    8,12, 10,7/*mean (0.225337), correlation (0.282851)*/,
    0,9, 1,3/*mean (0.226687), correlation (0.278734)*/,
    7,-5, 11,-10/*mean (0.00693882), correlation (0.305161)*/,
    -13,-6, -11,0/*mean (0.0227283), correlation (0.300181)*/,
    10,7, 12,1/*mean (0.125517), correlation (0.31089)*/,
    -6,-3, -6,12/*mean (0.131748), correlation (0.312779)*/,
    10,-9, 12,-4/*mean (0.144827), correlation (0.292797)*/,
    -13,8, -8,-12/*mean (0.149202), correlation (0.308918)*/,
    -13,0, -8,-4/*mean (0.160909), correlation (0.310013)*/,
    3,3, 7,8/*mean (0.177755), correlation (0.309394)*/,
    5,7, 10,-7/*mean (0.212337), correlation (0.310315)*/,
    -1,7, 1,-12/*mean (0.214429), correlation (0.311933)*/,
    3,-10, 5,6/*mean (0.235807), correlation (0.313104)*/,
    2,-4, 3,-10/*mean (0.00494827), correlation (0.344948)*/,
    -13,0, -13,5/*mean (0.0549145), correlation (0.344675)*/,
    -13,-7, -12,12/*mean (0.103385), correlation (0.342715)*/,
    -13,3, -11,8/*mean (0.134222), correlation (0.322922)*/,
    -7,12, -4,7/*mean (0.153284), correlation (0.337061)*/,
    6,-10, 12,8/*mean (0.154881), correlation (0.329257)*/,
    -9,-1, -7,-6/*mean (0.200967), correlation (0.33312)*/,
    -2,-5, 0,12/*mean (0.201518), correlation (0.340635)*/,
    -12,5, -7,5/*mean (0.207805), correlation (0.335631)*/,
    3,-10, 8,-13/*mean (0.224438), correlation (0.34504)*/,
    -7,-7, -4,5/*mean (0.239361), correlation (0.338053)*/,
    -3,-2, -1,-7/*mean (0.240744), correlation (0.344322)*/,
    2,9, 5,-11/*mean (0.242949), correlation (0.34145)*/,
    -11,-13, -5,-13/*mean (0.244028), correlation (0.336861)*/,
    -1,6, 0,-1/*mean (0.247571), correlation (0.343684)*/,
    5,-3, 5,2/*mean (0.000697256), correlation (0.357265)*/,
    -4,-13, -4,12/*mean (0.00213675), correlation (0.373827)*/,
    -9,-6, -9,6/*mean (0.0126856), correlation (0.373938)*/,
    -12,-10, -8,-4/*mean (0.0152497), correlation (0.364237)*/,
    10,2, 12,-3/*mean (0.0299933), correlation (0.345292)*/,
    7,12, 12,12/*mean (0.0307242), correlation (0.366299)*/,
    -7,-13, -6,5/*mean (0.0534975), correlation (0.368357)*/,
    -4,9, -3,4/*mean (0.099865), correlation (0.372276)*/,
    7,-1, 12,2/*mean (0.117083), correlation (0.364529)*/,
    -7,6, -5,1/*mean (0.126125), correlation (0.369606)*/,
    -13,11, -12,5/*mean (0.130364), correlation (0.358502)*/,
    -3,7, -2,-6/*mean (0.131691), correlation (0.375531)*/,
    7,-8, 12,-7/*mean (0.160166), correlation (0.379508)*/,
    -13,-7, -11,-12/*mean (0.167848), correlation (0.353343)*/,
    1,-3, 12,12/*mean (0.183378), correlation (0.371916)*/,
    2,-6, 3,0/*mean (0.228711), correlation (0.371761)*/,
    -4,3, -2,-13/*mean (0.247211), correlation (0.364063)*/,
    -1,-13, 1,9/*mean (0.249325), correlation (0.378139)*/,
    7,1, 8,-6/*mean (0.000652272), correlation (0.411682)*/,
    1,-1, 3,12/*mean (0.00248538), correlation (0.392988)*/,
    9,1, 12,6/*mean (0.0206815), correlation (0.386106)*/,
    -1,-9, -1,3/*mean (0.0364485), correlation (0.410752)*/,
    -13,-13, -10,5/*mean (0.0376068), correlation (0.398374)*/,
    7,7, 10,12/*mean (0.0424202), correlation (0.405663)*/,
    12,-5, 12,9/*mean (0.0942645), correlation (0.410422)*/,
    6,3, 7,11/*mean (0.1074), correlation (0.413224)*/,
    5,-13, 6,10/*mean (0.109256), correlation (0.408646)*/,
    2,-12, 2,3/*mean (0.131691), correlation (0.416076)*/,
    3,8, 4,-6/*mean (0.165081), correlation (0.417569)*/,
    2,6, 12,-13/*mean (0.171874), correlation (0.408471)*/,
    9,-12, 10,3/*mean (0.175146), correlation (0.41296)*/,
    -8,4, -7,9/*mean (0.183682), correlation (0.402956)*/,
    -11,12, -4,-6/*mean (0.184672), correlation (0.416125)*/,
    1,12, 2,-8/*mean (0.191487), correlation (0.386696)*/,
    6,-9, 7,-4/*mean (0.192668), correlation (0.394771)*/,
    2,3, 3,-2/*mean (0.200157), correlation (0.408303)*/,
    6,3, 11,0/*mean (0.204588), correlation (0.411762)*/,
    3,-3, 8,-8/*mean (0.205904), correlation (0.416294)*/,
    7,8, 9,3/*mean (0.213237), correlation (0.409306)*/,
    -11,-5, -6,-4/*mean (0.243444), correlation (0.395069)*/,
    -10,11, -5,10/*mean (0.247672), correlation (0.413392)*/,
    -5,-8, -3,12/*mean (0.24774), correlation (0.411416)*/,
    -10,5, -9,0/*mean (0.00213675), correlation (0.454003)*/,
    8,-1, 12,-6/*mean (0.0293635), correlation (0.455368)*/,
    4,-6, 6,-11/*mean (0.0404971), correlation (0.457393)*/,
    -10,12, -8,7/*mean (0.0481107), correlation (0.448364)*/,
    4,-2, 6,7/*mean (0.050641), correlation (0.455019)*/,
    -2,0, -2,12/*mean (0.0525978), correlation (0.44338)*/,
    -5,-8, -5,2/*mean (0.0629667), correlation (0.457096)*/,
    7,-6, 10,12/*mean (0.0653846), correlation (0.445623)*/,
    -9,-13, -8,-8/*mean (0.0858749), correlation (0.449789)*/,
    -5,-13, -5,-2/*mean (0.122402), correlation (0.450201)*/,
    8,-8, 9,-13/*mean (0.125416), correlation (0.453224)*/,
    -9,-11, -9,0/*mean (0.130128), correlation (0.458724)*/,
    1,-8, 1,-2/*mean (0.132467), correlation (0.440133)*/,
    7,-4, 9,1/*mean (0.132692), correlation (0.454)*/,
    -2,1, -1,-4/*mean (0.135695), correlation (0.455739)*/,
    11,-6, 12,-11/*mean (0.142904), correlation (0.446114)*/,
    -12,-9, -6,4/*mean (0.146165), correlation (0.451473)*/,
    3,7, 7,12/*mean (0.147627), correlation (0.456643)*/,
    5,5, 10,8/*mean (0.152901), correlation (0.455036)*/,
    0,-4, 2,8/*mean (0.167083), correlation (0.459315)*/,
    -9,12, -5,-13/*mean (0.173234), correlation (0.454706)*/,
    0,7, 2,12/*mean (0.18312), correlation (0.433855)*/,
    -1,2, 1,7/*mean (0.185504), correlation (0.443838)*/,
    5,11, 7,-9/*mean (0.185706), correlation (0.451123)*/,
    3,5, 6,-8/*mean (0.188968), correlation (0.455808)*/,
    -13,-4, -8,9/*mean (0.191667), correlation (0.459128)*/,
    -5,9, -3,-3/*mean (0.193196), correlation (0.458364)*/,
    -4,-7, -3,-12/*mean (0.196536), correlation (0.455782)*/,
    6,5, 8,0/*mean (0.1972), correlation (0.450481)*/,
    -7,6, -6,12/*mean (0.199438), correlation (0.458156)*/,
    -13,6, -5,-2/*mean (0.211224), correlation (0.449548)*/,
    1,-10, 3,10/*mean (0.211718), correlation (0.440606)*/,
    4,1, 8,-4/*mean (0.213034), correlation (0.443177)*/,
    -2,-2, 2,-13/*mean (0.234334), correlation (0.455304)*/,
    2,-12, 12,12/*mean (0.235684), correlation (0.443436)*/,
    -2,-13, 0,-6/*mean (0.237674), correlation (0.452525)*/,
    4,1, 9,3/*mean (0.23962), correlation (0.444824)*/,
    -6,-10, -3,-5/*mean (0.248459), correlation (0.439621)*/,
    -3,-13, -1,1/*mean (0.249505), correlation (0.456666)*/,
    7,5, 12,-11/*mean (0.00119208), correlation (0.495466)*/,
    4,-2, 5,-7/*mean (0.00372245), correlation (0.484214)*/,
    -13,9, -9,-5/*mean (0.00741116), correlation (0.499854)*/,
    7,1, 8,6/*mean (0.0208952), correlation (0.499773)*/,
    7,-8, 7,6/*mean (0.0220085), correlation (0.501609)*/,
    -7,-4, -7,1/*mean (0.0233806), correlation (0.496568)*/,
    -8,11, -7,-8/*mean (0.0236505), correlation (0.489719)*/,
    -13,6, -12,-8/*mean (0.0268781), correlation (0.503487)*/,
    2,4, 3,9/*mean (0.0323324), correlation (0.501938)*/,
    10,-5, 12,3/*mean (0.0399235), correlation (0.494029)*/,
    -6,-5, -6,7/*mean (0.0420153), correlation (0.486579)*/,
    8,-3, 9,-8/*mean (0.0548021), correlation (0.484237)*/,
    2,-12, 2,8/*mean (0.0616622), correlation (0.496642)*/,
    -11,-2, -10,3/*mean (0.0627755), correlation (0.498563)*/,
    -12,-13, -7,-9/*mean (0.0829622), correlation (0.495491)*/,
    -11,0, -10,-5/*mean (0.0843342), correlation (0.487146)*/,
    5,-3, 11,8/*mean (0.0929937), correlation (0.502315)*/,
    -2,-13, -1,12/*mean (0.113327), correlation (0.48941)*/,
    -1,-8, 0,9/*mean (0.132119), correlation (0.467268)*/,
    -13,-11, -12,-5/*mean (0.136269), correlation (0.498771)*/,
    -10,-2, -10,11/*mean (0.142173), correlation (0.498714)*/,
    -3,9, -2,-13/*mean (0.144141), correlation (0.491973)*/,
    2,-3, 3,2/*mean (0.14892), correlation (0.500782)*/,
    -9,-13, -4,0/*mean (0.150371), correlation (0.498211)*/,
    -4,6, -3,-10/*mean (0.152159), correlation (0.495547)*/,
    -4,12, -2,-7/*mean (0.156152), correlation (0.496925)*/,
    -6,-11, -4,9/*mean (0.15749), correlation (0.499222)*/,
    6,-3, 6,11/*mean (0.159211), correlation (0.503821)*/,
    -13,11, -5,5/*mean (0.162427), correlation (0.501907)*/,
    11,11, 12,6/*mean (0.16652), correlation (0.497632)*/,
    7,-5, 12,-2/*mean (0.169141), correlation (0.484474)*/,
    -1,12, 0,7/*mean (0.169456), correlation (0.495339)*/,
    -4,-8, -3,-2/*mean (0.171457), correlation (0.487251)*/,
    -7,1, -6,7/*mean (0.175), correlation (0.500024)*/,
    -13,-12, -8,-13/*mean (0.175866), correlation (0.497523)*/,
    -7,-2, -6,-8/*mean (0.178273), correlation (0.501854)*/,
    -8,5, -6,-9/*mean (0.181107), correlation (0.494888)*/,
    -5,-1, -4,5/*mean (0.190227), correlation (0.482557)*/,
    -13,7, -8,10/*mean (0.196739), correlation (0.496503)*/,
    1,5, 5,-13/*mean (0.19973), correlation (0.499759)*/,
    1,0, 10,-13/*mean (0.204465), correlation (0.49873)*/,
    9,12, 10,-1/*mean (0.209334), correlation (0.49063)*/,
    5,-8, 10,-9/*mean (0.211134), correlation (0.503011)*/,
    -1,11, 1,-13/*mean (0.212), correlation (0.499414)*/,
    -9,-3, -6,2/*mean (0.212168), correlation (0.480739)*/,
    -1,-10, 1,12/*mean (0.212731), correlation (0.502523)*/,
    -13,1, -8,-10/*mean (0.21327), correlation (0.489786)*/,
    8,-11, 10,-6/*mean (0.214159), correlation (0.488246)*/,
    2,-13, 3,-6/*mean (0.216993), correlation (0.50287)*/,
    7,-13, 12,-9/*mean (0.223639), correlation (0.470502)*/,
    -10,-10, -5,-7/*mean (0.224089), correlation (0.500852)*/,
    -10,-8, -8,-13/*mean (0.228666), correlation (0.502629)*/,
    4,-6, 8,5/*mean (0.22906), correlation (0.498305)*/,
    3,12, 8,-13/*mean (0.233378), correlation (0.503825)*/,
    -4,2, -3,-3/*mean (0.234323), correlation (0.476692)*/,
    5,-13, 10,-12/*mean (0.236392), correlation (0.475462)*/,
    4,-13, 5,-1/*mean (0.236842), correlation (0.504132)*/,
    -9,9, -4,3/*mean (0.236977), correlation (0.497739)*/,
    0,3, 3,-9/*mean (0.24314), correlation (0.499398)*/,
    -12,1, -6,1/*mean (0.243297), correlation (0.489447)*/,
    3,2, 4,-8/*mean (0.00155196), correlation (0.553496)*/,
    -10,-10, -10,9/*mean (0.00239541), correlation (0.54297)*/,
    8,-13, 12,12/*mean (0.0034413), correlation (0.544361)*/,
    -8,-12, -6,-5/*mean (0.003565), correlation (0.551225)*/,
    2,2, 3,7/*mean (0.00835583), correlation (0.55285)*/,
    10,6, 11,-8/*mean (0.00885065), correlation (0.540913)*/,
    6,8, 8,-12/*mean (0.0101552), correlation (0.551085)*/,
    -7,10, -6,5/*mean (0.0102227), correlation (0.533635)*/,
    -3,-9, -3,9/*mean (0.0110211), correlation (0.543121)*/,
    -1,-13, -1,5/*mean (0.0113473), correlation (0.550173)*/,
    -3,-7, -3,4/*mean (0.0140913), correlation (0.554774)*/,
    -8,-2, -8,3/*mean (0.017049), correlation (0.55461)*/,
    4,2, 12,12/*mean (0.01778), correlation (0.546921)*/,
    2,-5, 3,11/*mean (0.0224022), correlation (0.549667)*/,
    6,-9, 11,-13/*mean (0.029161), correlation (0.546295)*/,
    3,-1, 7,12/*mean (0.0303081), correlation (0.548599)*/,
    11,-1, 12,4/*mean (0.0355151), correlation (0.523943)*/,
    -3,0, -3,6/*mean (0.0417904), correlation (0.543395)*/,
    4,-11, 4,12/*mean (0.0487292), correlation (0.542818)*/,
    2,-4, 2,1/*mean (0.0575124), correlation (0.554888)*/,
    -10,-6, -8,1/*mean (0.0594242), correlation (0.544026)*/,
    -13,7, -11,1/*mean (0.0597391), correlation (0.550524)*/,
    -13,12, -11,-13/*mean (0.0608974), correlation (0.55383)*/,
    6,0, 11,-13/*mean (0.065126), correlation (0.552006)*/,
    0,-1, 1,4/*mean (0.074224), correlation (0.546372)*/,
    -13,3, -9,-2/*mean (0.0808592), correlation (0.554875)*/,
    -9,8, -6,-3/*mean (0.0883378), correlation (0.551178)*/,
    -13,-6, -8,-2/*mean (0.0901035), correlation (0.548446)*/,
    5,-9, 8,10/*mean (0.0949843), correlation (0.554694)*/,
    2,7, 3,-9/*mean (0.0994152), correlation (0.550979)*/,
    -1,-6, -1,-1/*mean (0.10045), correlation (0.552714)*/,
    9,5, 11,-2/*mean (0.100686), correlation (0.552594)*/,
    11,-3, 12,-8/*mean (0.101091), correlation (0.532394)*/,
    3,0, 3,5/*mean (0.101147), correlation (0.525576)*/,
    -1,4, 0,10/*mean (0.105263), correlation (0.531498)*/,
    3,-6, 4,5/*mean (0.110785), correlation (0.540491)*/,
    -13,0, -10,5/*mean (0.112798), correlation (0.536582)*/,
    5,8, 12,11/*mean (0.114181), correlation (0.555793)*/,
    8,9, 9,-6/*mean (0.117431), correlation (0.553763)*/,
    7,-4, 8,-12/*mean (0.118522), correlation (0.553452)*/,
    -10,4, -10,9/*mean (0.12094), correlation (0.554785)*/,
    7,3, 12,4/*mean (0.122582), correlation (0.555825)*/,
    9,-7, 10,-2/*mean (0.124978), correlation (0.549846)*/,
    7,0, 12,-2/*mean (0.127002), correlation (0.537452)*/,
    -1,-6, 0,-11/*mean (0.127148), correlation (0.547401)*/
};


/**
 * @brief The ORB constructor
 * 
 * @param _num_features : total number of features to extract
 * @param pyramid_scale_factor : scale factor between pyramid layer
 * @param _pyramid_num_level : number of pyramid layer
 * @param _fast_default_threshold : default threshold used by fast
 * @param _fast_min_threshold : adaptive min threshold used by fast
 */
ORBDetectorDescriptor::ORBDetectorDescriptor( int _num_features, \
                                              float pyramid_scale_factor, \
                                              int _pyramid_num_level, \
                                              int _fast_default_threshold, \
                                              int _fast_min_threshold ):
    num_features( _num_features ), \
    pyramid_num_level( _pyramid_num_level ), \
    fast_default_threshold( _fast_default_threshold ), \
    fast_min_threshold( _fast_min_threshold )
{
    /* Image pyramid setting */
    float pyramid_inv_scale_factor = 1.0f / pyramid_scale_factor;

    // Scaling factor for each layer
    pyramid_scale_factors.resize( pyramid_num_level );
    pyramid_inv_scale_factors.resize( pyramid_num_level );

    pyramid_scale_factors[ 0 ] = 1.0f;
    pyramid_inv_scale_factors[ 0 ] = 1.0f;
    for ( int level_i = 1; level_i < pyramid_num_level; ++level_i )
    {
        pyramid_scale_factors[ level_i ] = pyramid_scale_factors[ level_i - 1 ] * pyramid_scale_factor;
        pyramid_inv_scale_factors[ level_i ] = 1.0f / pyramid_scale_factors[ level_i ];
    }

    pyramid_scaled_image.resize( pyramid_num_level );

    // Number of features per level
    // Here, the total number of features is distributed per each layer based on the scale factor
    // NOTE: one can also choose other methods to distributed total features across layers
    pyramid_num_features_per_level.resize( pyramid_num_level );

    float num_features_per_level = num_features * \
        ( 1 - pyramid_inv_scale_factor ) / \
        ( 1 - (float)pow( (double)pyramid_inv_scale_factor, (double)pyramid_num_level ) );

    int num_features_cnt = 0;
    for ( int level_i = 0; level_i < pyramid_num_level - 1; level_i++ )
    {
        pyramid_num_features_per_level[ level_i ] = cvRound( num_features_per_level );
        num_features_cnt += pyramid_num_features_per_level[ level_i ];
        num_features_per_level *= pyramid_inv_scale_factor;
    }
    // Set remaining feature points to last level
    pyramid_num_features_per_level[ pyramid_num_level - 1 ] = std::max( num_features - num_features_cnt, 0 );
    
    /* BRISK setting */

    // Number of random pair to use in brisk
    const cv::Point *brisk_random_pattern_src_ptr = (const cv::Point*)bit_pattern_31_;
    std::vector<cv::Point> tmp_brisk_random_pattern( brisk_random_pattern_src_ptr, \
                                                     brisk_random_pattern_src_ptr + 512 );
    tmp_brisk_random_pattern.swap( brisk_random_pattern );

    /* Rotation setting */
    int v;
    int v0;
    int vmax = cvFloor( HALF_EXTRACTOR_PATCH_SIZE * sqrt(2.f) / 2 + 1 );
    int vmin = cvCeil( HALF_EXTRACTOR_PATCH_SIZE * sqrt(2.f) / 2);
    const double HALF_EXTRACTOR_PATCH_SIZE_SQR = HALF_EXTRACTOR_PATCH_SIZE * HALF_EXTRACTOR_PATCH_SIZE ;

    for (v = 0; v <= vmax; ++v)
    {
        patch_umax[ v ] = cvRound( sqrt(HALF_EXTRACTOR_PATCH_SIZE_SQR - v * v) );
    }
    
	for (v = HALF_EXTRACTOR_PATCH_SIZE, v0 = 0; v >= vmin; --v)
    {
        while ( patch_umax[ v0 ] == patch_umax[ v0 + 1 ])
            ++v0;
        patch_umax[ v ] = v0;
        ++v0;
    }
}


void ORBDetectorDescriptor::detectAndCompute( cv::InputArray _image, \
                                              cv::InputArray _mask, \
                                              std::vector<cv::KeyPoint>& _keypoints, \
                                              cv::OutputArray _descriptors, \
                                              bool _useProvidedKeypoints )
{
    // Currently don't support provided keypoint
    if ( _useProvidedKeypoints || _keypoints.size() )
    {
        throw std::runtime_error("ORBDetectorDescriptor::detectAndCompute do not support provided keypoint\n" );
    }

    // Currently doesn't support customize map
    if ( !_mask.empty() )
    {
        throw std::runtime_error("ORBDetectorDescriptor::detectAndCompute do not support customize mask\n");
    }

    // No input image, return
    if ( _image.empty() )
    {
        return;
    }

    // Require grayscale uint8
    cv::Mat _gray_image_u8 = _image.getMat();
    if ( _gray_image_u8.type() != CV_8UC1 )
    {
        throw std::runtime_error("ORBDetectorDescriptor::detectAndCompute input require uint 8, invalid input type\n");
    }

    // Build image pyramic
    // pyramid data is stored inside `pyramid_scaled_image`
    computePyramid( _gray_image_u8 );

    // Per pyramid layer keypoints
    std::vector<std::vector<cv::KeyPoint>> pyramid_keypoints_per_level( pyramid_num_level );

    // Compute KeyPoint per each pyramid level using quad tree
    // This is where the "uniform" feature distribution come from
    // Apply FAST as underlying algorithm
    computeFASTKeyPointQuadTree( pyramid_keypoints_per_level );

    // Compute rotation for every keypoints
    computeOrientation( pyramid_keypoints_per_level );

    // Count total number of feature points
    int total_num_features = 0;
    for ( int level_i = 0; level_i < pyramid_num_level; ++level_i )
    {
        total_num_features += pyramid_keypoints_per_level.size();
    }

    // No keypoints is detected across all layer of pyramid
    cv::Mat descriptors;
    if ( total_num_features == 0 )
    {
        _descriptors.release();
    }
    // Allocate space for descriptors
    else
    {
        _descriptors.create( total_num_features, 32, CV_8U );

        // `descriptors` and `_descriptors` points to same underlying data in memory
        // we modift `descriptors` in our code and its underlying data will change
        descriptors = _descriptors.getMat();
    }

    //std::vector<cv::KeyPoint> tmp_keypoints; 
    //tmp_keypoints.reserve( total_num_features );
    _keypoints.clear();
    _keypoints.reserve( total_num_features );

    // For every pyramid layer, Gaussian blur & compute descriptor
    int keypoints_cnt = 0;
    for ( int level_i = 0; level_i < pyramid_num_level; ++level_i )
    {
        std::vector<cv::KeyPoint>& keypoints_level_i = pyramid_keypoints_per_level[ level_i ];
        int num_keypoints_level_i = keypoints_level_i.size();
        if ( num_keypoints_level_i == 0 )
            continue;

        // NOTE: unlinke ORBextractor.cc that clone an image mat and gaussian blur on top
        // we gaussian blur on pyramid_scaled_image content directely to reduce memory footprint.
        cv::GaussianBlur( pyramid_scaled_image[ level_i ], \
                          pyramid_scaled_image[ level_i ], \
                          cv::Size( 7, 7 ), 2, 2, cv::BORDER_REFLECT_101 );

        cv::Mat descriptors_level_i( num_keypoints_level_i, 32, CV_8U );
        
        // NOTE: this function is now a member function and can access briskPattern directely.
        computeBRISKDescriptorsPerPyramidLevel( \
            pyramid_scaled_image[ level_i ], \
            keypoints_level_i, \
            descriptors_level_i );

        const float scale_factor_level_i = pyramid_scale_factors[ level_i ];

        // Add keypoints & descriptor to container
        for ( int keypoint_level_i_index = 0; \
                  keypoint_level_i_index < num_keypoints_level_i; \
                  keypoint_level_i_index++ )
        {
            // Scale level_i keypoints to original
            keypoints_level_i[ keypoint_level_i_index ].pt *= scale_factor_level_i;

            // Add to keypoints container
            _keypoints.emplace_back( keypoints_level_i[ keypoint_level_i_index ] );

            // Add to descriptor matrix
            descriptors_level_i.row( keypoint_level_i_index ).copyTo( descriptors.row( keypoints_cnt ) );
            keypoints_cnt++;
        }
    }

    // Swap keypoints container
    //tmp_keypoints.swap( _keypoints );
}


/**
 * @brief Get name of this extractor, used in some opencv funciton
 * 
 * @return cv::String
 */
cv::String ORBDetectorDescriptor::getDefaultName() const
{
    return (cv::FeatureDetector::getDefaultName() + ".ORBDetectorDescriptor");
}


/**
 * @brief Build image pyramic
 * 
 * @param image Source image of original size under uint8 gray scale
 */
void ORBDetectorDescriptor::computePyramid( const cv::Mat& image )
{
    for (int level_i = 0; level_i < pyramid_num_level; ++level_i) 
    {
        float level_i_inv_scale = pyramid_inv_scale_factors[ level_i ];

        cv::Size level_i_scale_size( cvRound((float) image.cols * level_i_inv_scale ), \
                                     cvRound((float) image.rows * level_i_inv_scale ));

        cv::Size level_i_whole_size( level_i_scale_size.width + PYRAMID_EDGE_SIZE * 2, \
                                     level_i_scale_size.height + PYRAMID_EDGE_SIZE * 2 );
        
        // Tmp image that contain border on both side
        // We only need the center part of this tmp image
        cv::Mat tmp_image( level_i_whole_size, image.type() );

        // NOTE: cv::Mat operator ( cv::Rect ) is shallow copy
        // when below line run, `pyramid_scaled_image[ level_i ]` and `tmp_image` all refer to the same underlying data
        // `pyramid_scaled_image[ level_i ]` point to the region of interest specified by cv::Rect
        // when the for loop finish, `tmp_image` out of scope, the reference count to that particular data - 1
        // but the `pyramid_scaled_image[ level_i ]` still point to that data
        pyramid_scaled_image[ level_i ] = tmp_image( cv::Rect( PYRAMID_EDGE_SIZE, \
                                                               PYRAMID_EDGE_SIZE, \
                                                               level_i_scale_size.width, \
                                                               level_i_scale_size.height ) );

        // Compute the resized image
        // For non first layer image, resize + padding border
        if ( level_i != 0) 
        {
            // Use previous layer + resize to generate current layer
            cv::resize( pyramid_scaled_image[ level_i -1 ], \
                        pyramid_scaled_image[ level_i ], \
                        level_i_scale_size, 0, 0, cv::INTER_LINEAR );

            // Padding the resized to tmp
            // https://docs.opencv.org/3.4/dc/da3/tutorial_copyMakeBorder.html
            // Below line change data in tmp_image, and thus change the `pyramid_scaled_image[ level_i ]` that share the same data
            cv::copyMakeBorder( pyramid_scaled_image[ level_i ], tmp_image, \
                PYRAMID_EDGE_SIZE, PYRAMID_EDGE_SIZE, PYRAMID_EDGE_SIZE, PYRAMID_EDGE_SIZE, \
                cv::BorderTypes::BORDER_REFLECT_101 + cv::BorderTypes::BORDER_ISOLATED);
        }
        // For first layer image, padding border
        else 
        {
            cv::copyMakeBorder( image, tmp_image, \
                PYRAMID_EDGE_SIZE, PYRAMID_EDGE_SIZE, PYRAMID_EDGE_SIZE, PYRAMID_EDGE_SIZE, \
                cv::BorderTypes::BORDER_REFLECT_101 );
        }
    }
}


/**
 * @brief Compute key point on every pyramid level using quad tree approach
 *
 * @param pyramid_keypoints_per_level: a vector of vectors. The first dimension represents the layers of 
 *                          the computed image pyramid, while the inner vectors contains the 
 *                          actual key points located at that layer.
 */
void ORBDetectorDescriptor::computeFASTKeyPointQuadTree( \
    std::vector<std::vector<cv::KeyPoint>> &pyramid_keypoints_per_level )
{
    // To ensure relative even/uniform keypoint detected across images
    // image is divided into multiple grid
    // an adaptive threshold is used to extract FAST keypoint from every grid

    // cell size
    const int cell_size = 30;

    // itr through the whole pyramid and process each layer.
    for (int level_i = 0; level_i < pyramid_num_level; level_i++) 
    {
        // get the ROI image area of current layer
        // FAST run in radious of 3, thus we can't run fast starting from the (0,0) boundary
        const int roi_min_x = PYRAMID_EDGE_SIZE - 3; // 3 is the radius set for the FAST calculation 
        const int roi_min_y = PYRAMID_EDGE_SIZE - 3;
        const int roi_max_x = pyramid_scaled_image[ level_i ].cols - roi_min_x;
        const int roi_max_y = pyramid_scaled_image[ level_i ].rows - roi_min_x;

        const int roi_range_x = roi_max_x - roi_min_x;
        const int roi_range_y = roi_max_y - roi_min_y;

        // get the amount of grids we have within current layer
        // this is equivalent of taking floor
        const int num_cell_x = roi_range_x / cell_size;
        const int num_cell_y = roi_range_y / cell_size;

        // calculate the size of each grid
        // cell_size_x/y might be diff from original cell_size due to boundary
        const int cell_size_x = std::ceil( roi_range_x * 1.0f / num_cell_x );
        const int cell_size_y = std::ceil( roi_range_y * 1.0f / num_cell_y );

        // Container that tmp store keypoints
        // Those keypoints will be "distributed" uniformly across images using octree
        std::vector<cv::KeyPoint> keypoints_to_distribute_level_i;
        keypoints_to_distribute_level_i.reserve( num_features * 10 );

        // traverse all the cell to compute fast on each cell
        for ( int cell_y_i = 0; cell_y_i < num_cell_y; cell_y_i++ )
        {
            // current cell start index ( y )
            const int cell_y_i_start = cell_y_i * cell_size_y + roi_min_y;

            // current cell end index ( y )
            // +6 account for FAST 3 pixel border
            // consider potential out of bound issue
            const int cell_y_i_end = ( cell_y_i_start + cell_size_y + 6 > roi_max_x ? roi_max_x : cell_y_i_start + cell_size_y + 6 );

            // out of bound
            if ( cell_y_i_start > roi_max_y - 3 )
                continue;

            for ( int cell_x_j = 0; cell_x_j < num_cell_x; cell_x_j++ )
            {
                const int cell_x_j_start = cell_x_j * cell_size_x + roi_min_x;
                const int cell_x_j_end = ( cell_x_j_start + cell_size_x + 6 > roi_max_x ? roi_max_x : cell_x_j_start + cell_size_x + 6 );

                // out of bound
                if  (cell_x_j_start > roi_max_x - 3 )
                    continue;

                // Current grid is defined as
                // [ cell_y_i_start, cell_y_i_end ) x [ cell_x_j_start, cell_x_j_end )

                // all of the keypoints within this grid.
                std::vector<cv::KeyPoint> curr_cell_keypoints;

                // OpenCV's FAST detector, first try, with higher threshold.
                // use rowRange colRange to specify only run on the current grid part
                cv::FAST( pyramid_scaled_image[ level_i ].rowRange( cell_y_i_start, cell_y_i_end ).colRange( cell_x_j_start, cell_x_j_end ), \
                    curr_cell_keypoints, fast_default_threshold, true );

                // adaptive methods, if default threshold do not detect any feature, try lower threshold
                // If still unable to detect any keypoint, give up
                if ( curr_cell_keypoints.empty() )
                    cv::FAST( pyramid_scaled_image[ level_i ].rowRange( cell_y_i_start, cell_y_i_end).colRange( cell_x_j_start, cell_x_j_end ), \
                        curr_cell_keypoints, fast_min_threshold, true );

                if ( !curr_cell_keypoints.empty() )
                {
                    for ( auto& cell_keypoints_i : curr_cell_keypoints )
                    {
                        // Convert FAST location back to original image coordinate
                        cell_keypoints_i.pt.x += cell_x_j * cell_size_x;
                        cell_keypoints_i.pt.y += cell_y_i * cell_size_y;

                        // Save to tmp container
                        keypoints_to_distribute_level_i.emplace_back( cell_keypoints_i );
                    }
                }
            } // end iterate cell x
        } // end iterate cell y

        // store a reference to all the keypoints that belongs to the current layer
        std::vector<cv::KeyPoint>& keypoints_level_i = pyramid_keypoints_per_level[ level_i ];
        keypoints_level_i.reserve( pyramid_num_features_per_level[ level_i ] );

        // NOTE: number of keypoints in `keypoints_to_distribute_level_i` may exceed the expected keypoints at current layer
        //      during the quad tree distribution process, we will eliminate those keypoints.
        QuadTreeDistributePerPyramidLevel( \
            keypoints_to_distribute_level_i, \
            keypoints_level_i, \
            roi_min_x, roi_max_x, \
            roi_min_y, roi_max_y, \
            pyramid_num_features_per_level[ level_i ], \
            level_i );
        
        const int patch_size_level_i = EXTRACTOR_PATCH_SIZE * pyramid_scale_factors[ level_i ];

        // traverse all feature points and restore their coordinates under current layer
        // NOTE: coordinate is set to current layer, not the gloabl layer (layer 0)
        // This is mainly because we need to compute orientation based on per image level keypoints location
        for ( auto& keypoints_level_i_j : keypoints_level_i )
        {
            keypoints_level_i_j.pt.x += roi_min_x;
            keypoints_level_i_j.pt.y += roi_min_y;
            keypoints_level_i_j.octave = level_i;
            keypoints_level_i_j.size = patch_size_level_i;
        }
    } // end of level i
}


/**
 * @brief Compute descriptor for each pyramic layer
 * 
 * @param image Image at specific pyramid layer
 * @param keypoints Keypoints detected at that layer
 * @param descriptors output descriptors
 */
void ORBDetectorDescriptor::computeBRISKDescriptorsPerPyramidLevel( \
    const cv::Mat& image, \
    std::vector<cv::KeyPoint>& keypoints, \
    cv::Mat& descriptors )
{
    // Zero out container
    descriptors = cv::Mat::zeros( keypoints.size(), 32, CV_8UC1 );

    const float factorPi = (float)(CV_PI/180.0);
    
    // step that opencv used to save image
    const int img_step = int(image.step);

    // Compute BRISK descriptor for every keypoint
    for ( size_t keypoint_idx = 0; keypoint_idx < keypoints.size(); ++keypoint_idx )
    {
        // Ptr to brisk random pair pattern
        const cv::Point* brisk_random_pattern_ptr = brisk_random_pattern.data();

        // Get reference to keypoint
        const cv::KeyPoint& keypoint_i = keypoints[ keypoint_idx ];

        // get the angle of the keypoint (and get the cos and sin value)
        float keypoint_i_angle = keypoint_i.angle * factorPi;
        float keypoint_i_cos = cos( keypoint_i_angle );
        float keypoint_i_sin = sin( keypoint_i_angle );

        // get the center of the keypoint in image
        const uchar* keypoint_i_img_center = &image.at<uchar>( \
            cvRound( keypoint_i.pt.y ), \
            cvRound( keypoint_i.pt.x ) );

        #define ORB_BRISK_GET_PATTERN( idx ) \
            keypoint_i_img_center[ \
                cvRound( brisk_random_pattern_ptr[ idx ].x * keypoint_i_sin + \
                         brisk_random_pattern_ptr[ idx ].y * keypoint_i_cos ) * img_step + \
                cvRound( brisk_random_pattern_ptr[ idx ].x * keypoint_i_cos - \
                         brisk_random_pattern_ptr[ idx ].y * keypoint_i_sin ) ]

        // Random sample 8 pixel pair around the the keypoint's image location center
        // NOTE: below for loop run for each keypoint
        for ( int i = 0; i < 32; ++i, brisk_random_pattern_ptr += 16 )
        {
            // t0, t1 : two random pixel define by brisk_random_pattern
            // val : use every bit to represent the descriptor result
            //      after the 8 pair finish, val contain 8 binary represent result
            int t0, t1, val;

            t0 = ORB_BRISK_GET_PATTERN( 0 ); 
            t1 = ORB_BRISK_GET_PATTERN( 1 );
            val = t0 < t1;

            t0 = ORB_BRISK_GET_PATTERN( 2 ); 
            t1 = ORB_BRISK_GET_PATTERN( 3 );
            val |= (t0 < t1) << 1;

            t0 = ORB_BRISK_GET_PATTERN( 4 ); 
            t1 = ORB_BRISK_GET_PATTERN( 5 );
            val |= (t0 < t1) << 2;

            t0 = ORB_BRISK_GET_PATTERN( 6 ); 
            t1 = ORB_BRISK_GET_PATTERN( 7 );
            val |= (t0 < t1) << 3;

            t0 = ORB_BRISK_GET_PATTERN( 8 ); 
            t1 = ORB_BRISK_GET_PATTERN( 9 );
            val |= (t0 < t1) << 4;

            t0 = ORB_BRISK_GET_PATTERN( 10 ); 
            t1 = ORB_BRISK_GET_PATTERN( 11 );
            val |= (t0 < t1) << 5;

            t0 = ORB_BRISK_GET_PATTERN( 12 ); 
            t1 = ORB_BRISK_GET_PATTERN( 13 );
            val |= (t0 < t1) << 6;

            t0 = ORB_BRISK_GET_PATTERN( 14 ); 
            t1 = ORB_BRISK_GET_PATTERN( 15 );
            val |= (t0 < t1) << 7;                

            descriptors.ptr(keypoint_idx)[ i ] = (uchar)val;
        }
    }
}


/**
 * @brief Find IC_Angle for every keypoints in a given pyramid layer
 * 
 * @param pyramid_keypoints_per_level keypoints per each pyramid level
 */
void ORBDetectorDescriptor::computeOrientation( \
    std::vector<std::vector<cv::KeyPoint>>& pyramid_keypoints_per_level )
{
    // Compute orientation for every level keypoints
    for ( int level_j = 0; level_j < pyramid_num_level; ++level_j )
    {
        std::vector<cv::KeyPoint>& keypoints_level_j = pyramid_keypoints_per_level[ level_j ];
        cv::Mat image_level_j = pyramid_scaled_image[ level_j ]; // reference to same underlying data, no copy

        // Compute rotation for every keypoints
        for ( auto& keypoint_i : keypoints_level_j )
        {
            const uchar* keypoint_i_img_center = &image_level_j.at<uchar>( \
                cvRound( keypoint_i.pt.y ), cvRound( keypoint_i.pt.x ) );
            
            int m_01 = 0, m_10 = 0;

            for (int u = -HALF_EXTRACTOR_PATCH_SIZE; u <= HALF_EXTRACTOR_PATCH_SIZE; ++u)
            {
                m_10 += u * keypoint_i_img_center[u];
            }

            int img_step = (int)image_level_j.step1();

            for (int v = 1; v <= HALF_EXTRACTOR_PATCH_SIZE; ++v)
            {
                int v_sum = 0;
                int d = patch_umax[v];
                for (int u = -d; u <= d; ++u)
                {
                    int val_plus  = keypoint_i_img_center[u + v*img_step];
                    int val_minus = keypoint_i_img_center[u - v*img_step];

                    v_sum += (val_plus - val_minus);

                    m_10 += u * (val_plus + val_minus);
                }
                m_01 += v * v_sum;
            }

            // Compute angle
            keypoint_i.angle = cv::fastAtan2( float(m_01), float(m_10) );

        } // end for every keypoint
    } // end for every layer
}

/**
 * @brief Use quad tree to distribute all keypoints uniformly across current layer
 * 
 * @param keypoints_to_distribute_level_i: keypoint wait to distribute
 * @param keypoints_level_i destination container
 * @param roi_minmax_xy : ROI of image
 * @param num_feature_level_i : desired number of feature at this layer
 * @param level_i : pyramid level
 */
void ORBDetectorDescriptor::QuadTreeDistributePerPyramidLevel( \
    const std::vector<cv::KeyPoint>& keypoints_to_distribute_level_i, \
          std::vector<cv::KeyPoint>& keypoints_level_i, \
    int roi_min_x, int roi_max_x, \
    int roi_min_y, int roi_max_y, \
    int num_feature_level_i, \
    int level_i )
{
    const int num_init_node = std::round( float(roi_max_x - roi_min_x) / float(roi_max_y - roi_min_y));
    const float num_node_pixel_x = float(roi_max_x - roi_min_x) / num_init_node;

    // a list use to hold all quad tree nodes of this level
    std::list<QuadTreeNode> quad_tree_node_list;

    std::vector<QuadTreeNode*> quad_tree_roots_ptrs;
    quad_tree_roots_ptrs.resize( num_init_node );

    // initialize all quad tree nodes and push them into container
    for (int i = 0; i < num_init_node; i++)
    {
        QuadTreeNode quad_tree_node;

        quad_tree_node.UL = cv::Point2i(num_node_pixel_x * static_cast<float>(i), 0);
        quad_tree_node.UR = cv::Point2i(num_node_pixel_x * static_cast<float>(i + 1), 0);
        quad_tree_node.BL = cv::Point2i(quad_tree_node.UL.x, roi_max_y - roi_min_y);
        quad_tree_node.BR = cv::Point2i(quad_tree_node.UR.x, roi_max_y - roi_min_y);

        quad_tree_node.keypoints.reserve( keypoints_to_distribute_level_i.size() );

        quad_tree_node_list.push_back(quad_tree_node);
        quad_tree_roots_ptrs[i] = &quad_tree_node_list.back();
    }

    // link points to child nodes
    for ( const auto& keypoints_to_distribute_j : keypoints_to_distribute_level_i )
    {
        quad_tree_roots_ptrs[ keypoints_to_distribute_j.pt.x / num_node_pixel_x ]->keypoints.emplace_back( keypoints_to_distribute_j );
    }

    // traverse the quad tree nodes list, mark the nodes that no longer needs to be split, delete the nodes that did not 
    // have a key points associated with it.
    auto lit = quad_tree_node_list.begin();
    while (lit != quad_tree_node_list.end())
    {
        // mark the node as final if the initial node only have one key point associate with it
        if (lit->keypoints.size() == 1)
        {
            lit->is_final = true;
            lit++;
        }
        else if (lit->keypoints.empty())
            // erase the node if no key point is associate with it
            lit = quad_tree_node_list.erase(lit);
        else
            // update the iterator
            lit++;
    }

    // a flag use to mark if the distribution process is finished
    bool is_finished = false;

    // a vector use to hold the <number of key points belong to one node, quad tree node ptr> pair.
    // we use this to discover which nodes can endure further divide
    std::vector<std::pair<int, QuadTreeNode*>> node_keypoint_size_vs_node_ptr_pair;
    node_keypoint_size_vs_node_ptr_pair.reserve(quad_tree_node_list.size() * 4);

    while (!is_finished)
    {
        int previous_size = quad_tree_node_list.size();

        lit = quad_tree_node_list.begin();

        int expanded_nodes_counter = 0;

        node_keypoint_size_vs_node_ptr_pair.clear();

        while (lit != quad_tree_node_list.end())
        {
            if (lit->is_final)
            {
                lit++;
                continue;
            }
            else
            {
                QuadTreeNode n1, n2, n3, n4;
                lit->divide(n1, n2, n3, n4);

                if (n1.keypoints.size() > 0)
                {
                    quad_tree_node_list.push_front(n1);
                    if (n1.keypoints.size() > 1)
                    {
                        expanded_nodes_counter++;

                        node_keypoint_size_vs_node_ptr_pair.push_back(std::make_pair(n1.keypoints.size(), &quad_tree_node_list.front()));

                        quad_tree_node_list.front().lit = quad_tree_node_list.begin();
                    }
                }
                if (n2.keypoints.size() > 0)
                {
                    quad_tree_node_list.push_front(n2);
                    if (n2.keypoints.size() > 1)
                    {
                        expanded_nodes_counter++;
                        node_keypoint_size_vs_node_ptr_pair.push_back(std::make_pair(n2.keypoints.size(), &quad_tree_node_list.front()));
                        quad_tree_node_list.front().lit = quad_tree_node_list.begin();
                    }
                }
                if (n3.keypoints.size() > 0)
                {
                    quad_tree_node_list.push_front(n3);
                    if (n3.keypoints.size() > 1)
                    {
                        expanded_nodes_counter++;
                        node_keypoint_size_vs_node_ptr_pair.push_back(std::make_pair(n3.keypoints.size(), &quad_tree_node_list.front()));
                        quad_tree_node_list.front().lit = quad_tree_node_list.begin();
                    }
                }
                if (n4.keypoints.size() > 0)
                {
                    quad_tree_node_list.push_front(n4);
                    if (n4.keypoints.size() > 1)
                    {
                        expanded_nodes_counter++;
                        node_keypoint_size_vs_node_ptr_pair.push_back(std::make_pair(n4.keypoints.size(), &quad_tree_node_list.front()));
                        quad_tree_node_list.front().lit = quad_tree_node_list.begin();
                    }
                }

                lit = quad_tree_node_list.erase(lit);
            }
        }

        if ((int)quad_tree_node_list.size() >= num_feature_level_i || (int)quad_tree_node_list.size() == previous_size)
        {
            is_finished = true;
        }

        else if (((int)quad_tree_node_list.size() + expanded_nodes_counter * 3) > num_feature_level_i)
        {
            while (!is_finished)
            {
                previous_size = quad_tree_node_list.size();

                std::vector<std::pair<int, QuadTreeNode*> > prev_node_keypoint_size_vs_node_ptr_pair = node_keypoint_size_vs_node_ptr_pair;
                node_keypoint_size_vs_node_ptr_pair.clear();

                sort(prev_node_keypoint_size_vs_node_ptr_pair.begin(), prev_node_keypoint_size_vs_node_ptr_pair.end());

                for (int j = prev_node_keypoint_size_vs_node_ptr_pair.size() - 1; j >= 0; j--)
                {
                    QuadTreeNode n1, n2, n3, n4;
                    prev_node_keypoint_size_vs_node_ptr_pair[j].second->divide(n1, n2, n3, n4);

                    if (n1.keypoints.size() > 0)
                    {
                        quad_tree_node_list.push_front(n1);
                        if (n1.keypoints.size() > 1)
                        {
                            node_keypoint_size_vs_node_ptr_pair.push_back(std::make_pair(n1.keypoints.size(), &quad_tree_node_list.front()));
                            quad_tree_node_list.front().lit = quad_tree_node_list.begin();
                        }
                    }
                    if (n2.keypoints.size() > 0)
                    {
                        quad_tree_node_list.push_front(n2);
                        if (n2.keypoints.size() > 1)
                        {
                            node_keypoint_size_vs_node_ptr_pair.push_back(std::make_pair(n2.keypoints.size(), &quad_tree_node_list.front()));
                            quad_tree_node_list.front().lit = quad_tree_node_list.begin();
                        }
                    }
                    if (n3.keypoints.size() > 0)
                    {
                        quad_tree_node_list.push_front(n3);
                        if (n3.keypoints.size() > 1)
                        {
                            node_keypoint_size_vs_node_ptr_pair.push_back(std::make_pair(n3.keypoints.size(), &quad_tree_node_list.front()));
                            quad_tree_node_list.front().lit = quad_tree_node_list.begin();
                        }
                    }
                    if (n4.keypoints.size() > 0)
                    {
                        quad_tree_node_list.push_front(n4);
                        if (n4.keypoints.size() > 1)
                        {
                            node_keypoint_size_vs_node_ptr_pair.push_back(std::make_pair(n4.keypoints.size(), &quad_tree_node_list.front()));
                            quad_tree_node_list.front().lit = quad_tree_node_list.begin();
                        }
                    }

                    quad_tree_node_list.erase(prev_node_keypoint_size_vs_node_ptr_pair[j].second->lit);

                    if ((int)quad_tree_node_list.size() >= num_feature_level_i)
                        break;
                }

                if ((int)quad_tree_node_list.size() >= num_feature_level_i || (int)quad_tree_node_list.size() == previous_size)
                    is_finished = true;
            }
        }
    }

    for (auto lit = quad_tree_node_list.begin(); lit != quad_tree_node_list.end(); lit++)
    {
        // NOTE: add to `keypoints_level_i` directly
        std::vector<cv::KeyPoint>& keypoints_of_this_node = lit->keypoints;
        cv::KeyPoint* best_keypoint = &keypoints_of_this_node[0];

        float max_response = best_keypoint->response;

        for (size_t k = 1; k < keypoints_of_this_node.size(); k++)
        {
            if (keypoints_of_this_node[k].response > max_response)
            {
                best_keypoint = &keypoints_of_this_node[k];
                max_response = keypoints_of_this_node[k].response;
            }
        }

        keypoints_level_i.push_back(*best_keypoint);
    }
}

/**
 * @brief Divide the current node into four sub areas.
 *
 * @param[in & out] n1   divided node 1
 * @param[in & out] n2   divided node 2
 * @param[in & out] n3   divided node 3
 * @param[in & out] n4   divided node 4
 */
void QuadTreeNode::divide(QuadTreeNode& n1, QuadTreeNode& n2, QuadTreeNode& n3, QuadTreeNode& n4)
{
    // get the half width and half height value of the image area represented by current node
    const int half_width_x = std::ceil(static_cast<float>(this->UR.x - this->UL.y) / 2);
    const int half_height_y = std::ceil(static_cast<float>(this->BR.y - this->UL.y) / 2);

    // update the boundaries value for the child nodes
    // node one, stores the upper left area
    n1.UL = this->UL;
    n1.UR = cv::Point2i(UL.x + half_width_x, UL.y);
    n1.BL = cv::Point2i(UL.x, UL.y + half_height_y);
    n1.BR = cv::Point2i(UL.x + half_width_x, UL.y + half_height_y);

    // node two, stores the upper right area
    n2.UL = n1.UR;
    n2.UR = this->UR;
    n2.BL = n1.BR;
    n2.BR = cv::Point2i(UR.x, UL.y + half_height_y);
    
    // node three, stores the bottom left area
    n3.UL = n1.BL;
    n3.UR = n1.BR;
    n3.BL = this->BL;
    n3.BR = cv::Point2i(n1.BR.x, BL.y);

    // node four, stores the bottom right area
    n4.UL = n3.UR;
    n4.UR = n2.BR;
    n4.BL = n3.BR;
    n4.BR = this->BR;

    // reserve the space for storing key points
    n1.keypoints.reserve(this->keypoints.size());
    n2.keypoints.reserve(this->keypoints.size());
    n3.keypoints.reserve(this->keypoints.size());
    n4.keypoints.reserve(this->keypoints.size());

    // redistribute the key points to corresponding new nodes
    for (size_t i = 0; i < this->keypoints.size(); i++)
    {
        // get a reference of the key point we are processing
        const cv::KeyPoint& keypoint_reference = this->keypoints[i];

        // test which sub area does the current key point belongs to
        // assign it to the correct new nodes
        if (keypoint_reference.pt.x < n1.UR.x)
        {
            if (keypoint_reference.pt.y < n1.BR.y)
                n1.keypoints.push_back(keypoint_reference);
            else
                n3.keypoints.push_back(keypoint_reference);
        }
        else if (keypoint_reference.pt.y < n1.BR.y)
        {
            n2.keypoints.push_back(keypoint_reference);
        }
        else
        {
            n4.keypoints.push_back(keypoint_reference);
        }
    }

    // check the final number of key points been assigned to each nodes
    // if the amount == 1, we mark the node as final and stop further divide it
    if (n1.keypoints.size() == 1)
        n1.is_final = true;
    if (n2.keypoints.size() == 1)
        n2.is_final = true;
    if (n3.keypoints.size() == 1)
        n3.is_final = true;
    if (n4.keypoints.size() == 1)
        n4.is_final = true;
}

} // namespace orb
