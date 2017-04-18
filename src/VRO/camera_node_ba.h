/*
 * Feb. 8, 2017 David Z 
 *
 * camera node, match feature in BA mode
 *
 * */


#ifndef CAMERA_NODE_BA_H
#define CAMERA_NODE_BA_H

#include "camera_node.h"
#include <map>

class CamModel; 

class CCameraNodeBA :  public CCameraNode
{
  public:
    CCameraNodeBA();
    virtual ~CCameraNodeBA(); 
    
    int projectionMatching(CCameraNode* other, std::vector<cv::DMatch>* matches, Eigen::Matrix4f& T, CamModel*); 
    std::map<int, int> matchNodePairBA(CCameraNodeBA*, Eigen::Matrix4f& T, CamModel *); 
    std::map<int, int> matchNodePairInliers(CCameraNodeBA* old_node, Eigen::Matrix4f& Tji, CamModel* pcam);
    MatchingResult matchNodePairVRO(CCameraNodeBA*, CamModel*);       // match points through Identity projection 
    MatchingResult matchNodePairVRO(CCameraNodeBA*, Eigen::Matrix4f& T,  CamModel*); 
    bool matchByMatches(CCameraNodeBA* , MatchingResult& , std::vector<cv::DMatch>& matches); 
    std::vector<int> mv_feature_qid;  // if -1, this feature not in the graph, else means this feature match 
    bool mb_checked ; 
    void check(); // whether mv_feature_qid has been initialized 
  
    double siftDescriptorDis(cv::Mat& m1, cv::Mat& m2); 

  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};



#endif
