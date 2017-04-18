#include "camera_node_ba.h"
#include "cam_model.h"
#include "transformation_estimation_euclidean.h"

using namespace std; 

CCameraNodeBA::CCameraNodeBA():
mb_checked(false)
{}
CCameraNodeBA::~CCameraNodeBA(){}

void CCameraNodeBA::check()
{
  if(!mb_checked)
  {
    cout<<"check() at node "<< m_id<<endl; 
  if(mv_feature_qid.size() != m_feature_loc_2d.size())
    mv_feature_qid.resize(m_feature_loc_2d.size(), -1); 
    mb_checked = true; 
  }
}

MatchingResult CCameraNodeBA::matchNodePairVRO(CCameraNodeBA* old_node, Eigen::Matrix4f& T, CamModel* pcam)
{
  check();
  old_node->check();
  
  MatchingResult mr; 
  mr.succeed_match = false; 

  vector<cv::DMatch> matches; //  = mr.all_matches; 
  Eigen::Matrix4f Tji = T; 
  projectionMatching(old_node, &matches,  Tji, pcam); 

  mr.succeed_match = matchByMatches(old_node, mr, matches);
  return mr; 


}

bool CCameraNodeBA::matchByMatches(CCameraNodeBA* old_node, MatchingResult& mr, vector<cv::DMatch>& matches)
{
  mr.all_matches = matches; 
  if(matches.size() < 5)
  {
    // ROS_WARN("%s at %d projected matches have %d features, too few!", __FILE__, __LINE__, matches.size()); 
    // return mr;
    return false; 
  }

  // compute transformation 
  // mr.inlier_matches = matches; 
  // bool valid = true; 
  // mr.final_trafo = getTransformFromMatches(this, old_node, mr.inlier_matches, valid, 0); 
  // mr.succeed_match = true; 

  bool found_transformation = CCameraNode::getRelativeTransformationTo(old_node, &mr.all_matches, mr.final_trafo, mr.rmse, mr.inlier_matches); 

  mr.succeed_match = found_transformation; 
  return found_transformation; 
}

MatchingResult CCameraNodeBA::matchNodePairVRO(CCameraNodeBA* old_node, CamModel* pcam)
{
  check();
  old_node->check();
  
  Eigen::Matrix4f Tji = Eigen::Matrix4f::Identity(); 

  return matchNodePairVRO(old_node, Tji, pcam); 
}

map<int, int> CCameraNodeBA::matchNodePairInliers(CCameraNodeBA* old_node, Eigen::Matrix4f& Tji, CamModel* pcam)
{
    // first spatial search, then RANSAC find inliers 
    map<int, int> ret; 
    check();
    old_node->check(); 
 
    // spatial search 
    vector<cv::DMatch> matches; 
    projectionMatching(old_node, &matches, Tji, pcam);

    // RANSAC find inliers  
    std::vector<cv::DMatch> inliers; 
    float rmse; 
    Eigen::Matrix4f T; 
    CCameraNode::getRANSACInliers(old_node, &matches, T, rmse, inliers); 
    for(int i=0; i<inliers.size(); i++)
    {
      cv::DMatch& m = inliers[i];
      {
        ret.insert(pair<int, int>(m.trainIdx, m.queryIdx)); 
      }
    }
  return ret; 

}

map<int,int> CCameraNodeBA::matchNodePairBA(CCameraNodeBA* old_node, Eigen::Matrix4f& Tji, CamModel* pcam)
{
  map<int, int> ret; 
  check();
  old_node->check(); 
  
  vector<cv::DMatch> matches; 
  // featureMatching(old_node, &matches); 
  projectionMatching(old_node, &matches, Tji, pcam);

  // CamModel * pcam = CamModel::gCamModel(); 

  float ou, ov, tu, tv;
  for(int i=0; i<matches.size(); i++)
  {
    cv::DMatch& m = matches[i];
/*  Eigen::Vector4f& pi = old_node->m_feature_loc_3d[m.trainIdx]; 
    Eigen::Vector4f pj = Tji * pi; 
    
    pcam->convertXYZ2UV(pj(0), pj(1), pj(2), ou, ov); 
    
    // cout <<" point pi: "<<endl<<pi<<endl;
    // cout <<" transform to pj: "<<endl<<pj<<endl;
    // cout <<" dis between fi and fj: "<<m.distance<<endl;

    if(ou < 2 || ou >= pcam->m_cols-2 || ov < 2 || ov >= pcam->m_rows -2)
    {
      // cout <<" ou = "<<ou <<" ov = "<<ov<<" outside img range"<<endl;
      continue; 
    }
    cv::KeyPoint& kp = m_feature_loc_2d[m.queryIdx]; 
    tu = kp.pt.x; tv = kp.pt.y; 

    double dis = SQ(tu-ou) + SQ(tv-ov); 
    if(dis <= 49)
    */
    {
      ret.insert(pair<int, int>(m.trainIdx, m.queryIdx)); 
    }
  }
  return ret; 
}


int CCameraNodeBA::projectionMatching(CCameraNode* other, vector<cv::DMatch>* matches, Eigen::Matrix4f& Tji, CamModel* pcam)
{
  if(matches->size() > 0) 
    matches->clear(); 
  
  double Thresh_dis = 90000;
  float ou, ov, tu, tv; 
  for(int i=0 ; i < other->m_feature_loc_2d.size(); i++)
  {
    Eigen::Vector4f& pi = other->m_feature_loc_3d[i]; 
    Eigen::Vector4f pj = Tji * pi; 
    
    pcam->convertXYZ2UV(pj(0), pj(1), pj(2), ou, ov); 
    
    if(ou < 2 || ou >= pcam->m_cols-2 || ov < 2 || ov >= pcam->m_rows -2)
    {
      // cout <<" ou = "<<ou <<" ov = "<<ov<<" outside img range"<<endl;
      continue; 
    }
    
    cv::Mat fi_des = other->m_feature_descriptors.row(i); 
    
    double best_dis = -1; 
    int pj_id = -1;

    for(int j=0; j< m_feature_loc_2d.size(); j++)
    {
      cv::KeyPoint& kp = m_feature_loc_2d[j]; 
      tu = kp.pt.x; tv = kp.pt.y; 
      
      // close enough, find descriptor distance 
      if(fabs(tu - ou) <= 3 && fabs(tv - ov) <= 3)
      {
        cv::Mat fj_des = m_feature_descriptors.row(j); 
        
        double dis = siftDescriptorDis(fi_des, fj_des); 
        if(best_dis == -1 || dis < best_dis)
        {
          best_dis = dis; 
          pj_id = j; 
        }
      }
    }
    if(pj_id != -1 && best_dis <= Thresh_dis)
    {
      cv::DMatch match; 
      match.queryIdx = pj_id; 
      match.trainIdx = i; 
      match.distance = best_dis; 
      matches->push_back(match); 
    }
  }
  return matches->size(); 
}

double CCameraNodeBA::siftDescriptorDis(cv::Mat& m1, cv::Mat& m2)
{
  double ret = 0;
  for(int i=0; i<m1.rows; i++)
    for(int j=0; j<m2.cols; j++)
  {
    float d = m1.at<float>(i,j) - m2.at<float>(i,j);
    ret += d*d; 
  }
  return ret; 
}
