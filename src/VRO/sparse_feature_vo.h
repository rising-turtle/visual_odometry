/*
 * Sep. 29, 2016, David Z 
 * 
 * A interface for VRO 
 *
 * */

#ifndef SPARSE_FEATURE_VO_H
#define SPARSE_FEATURE_VO_H

#include "camera_node.h"
#include "cam_model.h"
#include "feature_detector.h"
#include "vro_parameter.h"
#include <tf/tf.h>
#include <string>
#include <vector>

class CSparseFeatureVO 
{
  public:
    // CSparseFeatureVO(CamModel cm, std::string detector_type, std::string extractor_type, std::string match_type); 
    // CSparseFeatureVO(CamModel cm, CParams p = CParams());
    CSparseFeatureVO(CamModel cm); 
    virtual ~CSparseFeatureVO(); 
    
    virtual void featureExtraction(cv::Mat& visual, cv::Mat& depth, float depth_scale, CCameraNode& cn, cv::Mat mask = cv::Mat()); 
    void projectTo3D( cv::Mat& depth, float depth_scale, CCameraNode& cn);

    virtual tf::Transform VRO(CCameraNode& tar, CCameraNode& src, _F = 0,  Eigen::Matrix<double, 6, 6>* cov=0 );
    virtual tf::Transform VRO(cv::Mat& tar_i, cv::Mat& tar_d, cv::Mat& src_i, cv::Mat& src_d, float depth_scale, _F = 0, Eigen::Matrix<double, 6, 6>* cov=0 );
    virtual tf::Transform VRO(cv::Mat& tar_i, cv::Mat& tar_d, cv::Mat& tar_mask, cv::Mat& src_i, cv::Mat& src_d, cv::Mat& src_mask, float depth_scale);

    // std::string m_detector_type;
    // std::string m_extractor_type; 
    // std::string m_match_type; 
    // CParams m_params;

void generatePointCloud(cv::Mat& rgb, cv::Mat& depth, int skip, float depth_scale, std::vector<float>& pts, std::vector<unsigned char>& color, int * rect = 0);

    cv::Ptr<cv::FeatureDetector> mp_detector; 
    cv::Ptr<cv::DescriptorExtractor> mp_extractor; 
    CamModel m_cam_model; 

#ifdef USE_SIFT_GPU
    // project siftGPU 
    void projectTo3DSiftGPU(cv::Mat& depth, float depth_scale, 
        std::vector<float>& descriptors_in, CCameraNode& cn);
#endif

};


#endif
