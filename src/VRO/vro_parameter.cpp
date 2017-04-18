
#include "vro_parameter.h"

CParams* CParams::mp_instance = NULL; 
CParams* CParams::Instance()
{
  if(mp_instance == NULL)
  {
    mp_instance = new CParams(); 
  }
  return mp_instance;
}

CParams::CParams(): 
m_nn_distance_ratio(0.5), 
m_max_num_features(500), 
m_feature_detector_type("SIFT"),
  m_feature_descriptor_type("SIFT"),
  m_feature_match_type("FLANN"), 
  m_max_dist_for_inliers(0.05),
  m_min_matches(12), 
  m_ransac_iterations(200),
  m_siftgpu_edge_threshold(10.), 
  m_siftgpu_contrast_threshold(0.0008), 
  m_sift_num_features(0), 
  m_sift_octave_layers(5), 
  m_sift_contrast_threshold(0.02) // 0.04
{}
CParams::~CParams(){}
