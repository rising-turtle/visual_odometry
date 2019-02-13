#include "transformation_estimation_euclidean.h"
#include "camera_node.h"
#include <pcl/common/transformation_from_correspondences.h>
#include <Eigen/Geometry>

Eigen::Matrix4f getTransformFromMatches(const CCameraNode* newer_node,
                                        const CCameraNode* earlier_node,
                                        const std::vector<cv::DMatch>& matches)
{
  pcl::TransformationFromCorrespondences tfc;
  float weight = 1.0;

  for(int i=0; i<matches.size(); i++)
  {
    const cv::DMatch& m = matches[i];
    Eigen::Vector3f from = newer_node->m_feature_loc_3d[m.queryIdx].head<3>();
    Eigen::Vector3f to = earlier_node->m_feature_loc_3d[m.trainIdx].head<3>();
    if(pcl_isnan(from(2)) || pcl_isnan(to(2)))
      continue;
    tfc.add(from, to, weight);// 1.0/(to(2)*to(2)));//the further, the less weight b/c of quadratic accuracy decay
  }
  // get relative movement from samples
  return tfc.getTransformation().matrix();
}

namespace{
inline double HuberWeight(double dis)
{
  double k = 1.2; 
  double w;
  dis = fabs(dis);
  if(dis <= k)
  {
    w = 1.;
  }else{
    w = k/dis;
  }
  return w; 
}
}

Eigen::Matrix4f getTransformFromMatches(const CCameraNode* newer_node,
                                        const CCameraNode* earlier_node,
                                        const std::vector<cv::DMatch>& matches,
                                        bool& valid, 
                                        const float max_dist_m) 
{
  pcl::TransformationFromCorrespondences tfc;
  valid = true;
  std::vector<Eigen::Vector3f> t, f;

  // BOOST_FOREACH(const cv::DMatch& m, matches)
  for(int i=0; i<matches.size(); i++)
  {
    const cv::DMatch& m = matches[i];
    Eigen::Vector3f from = newer_node->m_feature_loc_3d[m.queryIdx].head<3>();
    Eigen::Vector3f to = earlier_node->m_feature_loc_3d[m.trainIdx].head<3>();
    if(pcl_isnan(from(2)) || pcl_isnan(to(2)))
      continue;
    float weight = 1.0;
    // double dis = (from.norm() + to.norm())/2.;
    // weight = HuberWeight(dis); 
    // ParameterServer* ps = ParameterServer::instance();

    //Create point cloud inf necessary
    // if(ps->get<int>("segment_to_optimize") > 0){
    //  weight =1/( earlier_node->feature_locations_3d_[m.trainIdx][3] \
    //            + newer_node->feature_locations_3d_[m.queryIdx][3]);
    // } else {
    //  weight =1/( earlier_node->feature_locations_3d_[m.trainIdx][2] \
    //            + newer_node->feature_locations_3d_[m.queryIdx][2]);
    // }
    //Validate that 3D distances are corresponding
    if (max_dist_m > 0) 
    {  //storing is only necessary, if max_dist is given
      if(f.size() >= 1)
      {
        float delta_f = (from - f.back()).squaredNorm();//distance to the previous query point
        float delta_t = (to   - t.back()).squaredNorm();//distance from one to the next train point

        if ( fabs(delta_f-delta_t) >  max_dist_m ) {
          valid = false;
          return Eigen::Matrix4f();
        }
      }
      f.push_back(from);
      t.push_back(to);    
    }

    tfc.add(from, to, weight);// 1.0/(to(2)*to(2)));//the further, the less weight b/c of quadratic accuracy decay
  }

  // get relative movement from samples
  return tfc.getTransformation().matrix();
}

Eigen::Matrix4f getTransformFromMatchesUmeyama(const CCameraNode* newer_node,
                                               const CCameraNode* earlier_node,
                                               std::vector<cv::DMatch> matches,
                                               bool& valid) 
{
  Eigen::Matrix<float, 3, Eigen::Dynamic> tos(3,matches.size()), froms(3,matches.size());
  std::vector<cv::DMatch>::const_iterator it = matches.begin();
  for (int i = 0 ;it!=matches.end(); it++, i++) {
    Eigen::Vector3f f = newer_node->m_feature_loc_3d[it->queryIdx].head<3>(); //Oh my god, c++
    Eigen::Vector3f t = earlier_node->m_feature_loc_3d[it->trainIdx].head<3>();
    if(pcl_isnan(f(2)) || pcl_isnan(t(2)))
      continue;
    froms.col(i) = f;
    tos.col(i) = t;
  }
  Eigen::Matrix4f res = Eigen::umeyama(froms, tos, false);
  valid = !containsNaN(res);
  return res;
}
