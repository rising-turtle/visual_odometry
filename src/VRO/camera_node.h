/*
 * Sep. 29, 2016 David Z 
 *
 * camera node, store the feature information, match features, compute transformation 
 *
 * */

#ifndef CAMERA_NODE_H
#define CAMERA_NODE_H

#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <vector>
#include <string>
#include <QMutex>
#include "matching_result.h"
#include "vro_parameter.h"

class CamCov; 
class CamModel;

typedef std::vector<Eigen::Vector4f, Eigen::aligned_allocator<Eigen::Vector4f> > std_vector_of_eigen_vector4f;
typedef Eigen::Matrix<double, 6, 1> (*_F)(Eigen::Matrix4f&); 
typedef Eigen::Matrix<double, 6, 1> (*_Fd)(Eigen::Matrix4d&);

class CCameraNode
{
  public:
    CCameraNode();
    virtual ~CCameraNode();

    unsigned int featureMatching(CCameraNode* , std::vector<cv::DMatch>* matches);  // feature match 
    cv::flann::Index* getFlannIndex();  // build KDtree 
    void knnSearch(cv::Mat& query, cv::Mat& indices, cv::Mat& dists, int knn, const cv::flann::SearchParams& params); 
    
    virtual MatchingResult matchNodePair(CCameraNode* ); 

    virtual bool getRelativeTransformationTo(CCameraNode* , std::vector<cv::DMatch>* initial_matches, 
        Eigen::Matrix4f& resulting_transformation, float& rmse, std::vector<cv::DMatch>& matches); // compute transformation 
 
    void getRANSACInliers(CCameraNode* , std::vector<cv::DMatch>* initial_matches, 
        Eigen::Matrix4f& resulting_transformation, float& rmse, std::vector<cv::DMatch>& matches); // compute transformation 
 
    void computeInliersAndError(std::vector<cv::DMatch> & all_matches, Eigen::Matrix4f& transformation4f, 
        std::vector<Eigen::Vector4f, Eigen::aligned_allocator<Eigen::Vector4f> >& origins, 
        std::vector<Eigen::Vector4f, Eigen::aligned_allocator<Eigen::Vector4f> >& earlier,
        size_t min_inliers, std::vector<cv::DMatch>& inliers, double& return_mean_error, double squaredMaxInlierDistInM); 

    double errorFunction(Eigen::Vector3f& x1, Eigen::Vector3f& x2, Eigen::Matrix4d& transformation);

    // compute covariance 
    virtual void computeCov(CCameraNode* , std::vector<cv::DMatch>& matches, _F F, Eigen::Matrix<double, 6, 6>& cov); 
    static CamCov* mp_camCov_;  // 
    static void set_cam_cov(CamModel& model);  // set camera model for the comCov class

    cv::flann::Index* m_pflannIndex ; 
    // std::string m_match_type; // match type 
    // std::string m_feature_type; // feature type
    static QMutex siftgpu_mutex; // mutex for gpu 

    cv::Mat m_img; // original intensity image 
    static bool gb_dis_match_point; 
    int m_id; // number of camera node in the graph when this node was added
    int m_seq_id; // number of images that have been processed (even if they were not added)
    int m_vertex_id;  // id of the vertex in the g2o/isam graph 
    
    cv::Mat m_feature_descriptors; // feature discriptor
    std::vector<float> m_siftgpu_descriptors; // siftgpu discriptors 
    std::vector<cv::KeyPoint> m_feature_loc_2d; // 2d location in image of feature points 
    std_vector_of_eigen_vector4f m_feature_loc_3d; // 3d location in space of feature points 

  public:
      EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};



#endif
