#ifndef TRANSFORMATION_ESTIMATION_EUCLIDEAN_H
#define TRANSFORMATION_ESTIMATION_EUCLIDEAN_H
#include <Eigen/Core>
#include <opencv2/features2d/features2d.hpp>
#include <vector>
class CCameraNode;

// Compute the transformation from matches using pcl::TransformationFromCorrespondences
extern Eigen::Matrix4f getTransformFromMatches(const CCameraNode* newer_node,
                                        const CCameraNode* older_node, 
                                        const std::vector<cv::DMatch> & matches);
                                       
// Compute the transformation from matches using pcl::TransformationFromCorrespondences
extern Eigen::Matrix4f getTransformFromMatches(const CCameraNode* newer_node,
                                        const CCameraNode* older_node, 
                                        const std::vector<cv::DMatch> & matches,
                                        bool& valid, 
                                        float max_dist_m = -1);

// Compute the transformation from matches using Eigen::umeyama
extern Eigen::Matrix4f getTransformFromMatchesUmeyama(const CCameraNode* newer_node,
                                               const CCameraNode* older_node,
                                               std::vector<cv::DMatch> matches,
                                               bool& valid);

inline bool containsNaN(const Eigen::Matrix4f& mat){
    return (mat.array() != mat.array()).any(); //No NaNs
}
#endif
