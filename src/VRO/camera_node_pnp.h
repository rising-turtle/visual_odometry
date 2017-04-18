/*
 *  
 *  Feb.28 2017, David Z 
 *
 *  use PnP methods inplemented in opengv 
 * 
 *
 * */


#ifndef CAMERA_NODE_PNP_H
#define CAMERA_NODE_PNP_H

#include "camera_node_ba.h"
#include <map>

class CCameraNodePnP : public CCameraNodeBA
{
  public:
    CCameraNodePnP(); 
    virtual ~CCameraNodePnP(); 

    virtual bool getRelativeTransformationTo(CCameraNode* older_node, std::vector<cv::DMatch>* initial_matches, 
        Eigen::Matrix4f& resulting_transformation, float& rmse, std::vector<cv::DMatch>& matches); 
    virtual MatchingResult matchNodePair(CCameraNode* );  
    virtual void computeCov(CCameraNode* , std::vector<cv::DMatch>& matches, _F F,  Eigen::Matrix<double, 6, 6>& cov);  
        // std::vector<Eigen::Vector3d, Eigen::aligned_allocator< Eigen::Vector3d> >& , std::vector<Eigen::Vector3d, Eigen::aligned_allocator< Eigen::Vector3d> >&); 
    int m_min_inliers; // minimal number of inliers 
    int m_ransac_num;  // number of ransac iterations
    double m_inlier_threshold; // to search for inliers 
};


#endif
