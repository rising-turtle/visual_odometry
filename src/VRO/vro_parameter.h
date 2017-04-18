/*  
 * Oct. 2, 2016, David Z 
 *
 * parameters for VRO 
 *
 * */

#ifndef VRO_PARAMETERS
#define VRO_PARAMETERS

#include <string>

class CParams
{
  public:
    // CParams();
    ~CParams(); 
  
    float m_nn_distance_ratio;  // feature correspondence distance threshold 
    float m_max_dist_for_inliers;     // max distance to be considered as inliers by RANSAC 
    int m_ransac_iterations;          // ransac iteration times 
    int m_min_matches;                // minimal matched features before RANSAC 

    /* feature extraction parameters  */
    int m_max_num_features;     // max_number of features to be extracted 
    std::string m_feature_detector_type;    // SIFTGPU, SIFT, SURF, ORB 
    std::string m_feature_descriptor_type;  // SIFTGPU, SIFT, SURF, ORB 
    std::string m_feature_match_type; // BRUTE FORTH, KDTREE, SIFTGPU 
    double m_siftgpu_edge_threshold;        // siftgpu edge threshold 
    double m_siftgpu_contrast_threshold;    // siftgpu contrast threshold 
    int m_sift_num_features;                // 0 return every detected features, >0 return specific number features
    int m_sift_octave_layers;               // number of octave layers 
    double m_sift_contrast_threshold;      // sift contrast threshold
    

  static CParams* Instance(); 

  private:
    CParams();
    static CParams* mp_instance;
    // TODO: add siftgpu_edge|contrast|_threshold, sift_|octave|contrast_threshold 
};

#endif
