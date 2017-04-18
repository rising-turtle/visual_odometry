#include "camera_node_pnp.h"
#include "cam_cov.h"
#include "cam_model.h"
#include "covariance_check.h"
#include <ros/ros.h>
#include <fstream>

#include <opengv/absolute_pose/methods.hpp>
#include <opengv/sac/Ransac.hpp>
#include <opengv/absolute_pose/CentralAbsoluteAdapter.hpp>
#include <opengv/sac_problems/absolute_pose/AbsolutePoseSacProblem.hpp>

// #include <gtsam/geometry/pose3.h>

using namespace std; 
using namespace opengv; 

// this is akward, since it will need to compile with gtsam 
/*Eigen::Matrix<double, 6, 1> cov_Helper(Eigen::Matrix4d& m)
{
  // Eigen::Matrix4d md = m.cast<double>(); 
  Pose3 p(m); 
  gtsam::Vector6 r = Pose3::ChartAtOrigin::Local(p); 
  return r; 
} */

CCameraNodePnP::CCameraNodePnP(): 
m_min_inliers(12), 
m_ransac_num(2000)
{}
CCameraNodePnP::~CCameraNodePnP(){}

void CCameraNodePnP::computeCov(CCameraNode* older_node, std::vector<cv::DMatch>& inliers, _F F, Eigen::Matrix<double, 6, 6>& cov )
{
  // 1, compute dF1/dpt1, dF2/dpt2  
  int N = inliers.size(); 
  // Eigen::Matrix<double, 6, 3*N> dFdtt = Eigen::Matrix<double, 6, 3*N>::Zero(); 
  // Eigen::Matrix<double, 6, 3*N> dFdtf = Eigen::Matrix<double, 6, 3*N>::Zero();
  Eigen::MatrixXd dFdtt(6, 3*N); 
  Eigen::MatrixXd dFdtf(6, 3*N); 
  double d_err = 1e-3; 
  double de2 = 2*d_err; 
  int fi, ti; 
  float tmp_f, tmp_t; 
  Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
  Eigen::Matrix4f fT = T.cast<float>(); 
  transformation_t T_epnp; 
  Eigen::Matrix<double, 6, 1> x1, x2; // depends on what you define [euler, t] or [q, t] or [t q]
  Eigen::Vector3d pt; 
  
  // construct data structure 
  bearingVectors_t bearingVs; 
  points_t pts; 
  for(int i=0; i<N; i++)
  {
    cv::DMatch& m = inliers[i]; 

    Eigen::Vector4f pi = older_node->m_feature_loc_3d[m.trainIdx]; 
    Eigen::Vector4f pj = this->m_feature_loc_3d[m.queryIdx]; 

    point_t bodyPoint(pi(0), pi(1), pi(2)); 
    pts.push_back(bodyPoint); 
    pt << pj(0), pj(1), pj(2); 
    pt *= (1./pt.norm()); 
    bearingVs.push_back(pt); 
  }

  for(int i=0; i<N; i++)
  {
    fi = inliers[i].queryIdx;  // node j 
    ti = inliers[i].trainIdx;  // node i 
    for(int m=0; m<3; m++)
    {
      // first dF1/dtt
      // tmp_t = older_node->m_feature_loc_3d[ti][m]; 
      tmp_t = pts[i][m]; 
      // older_node->m_feature_loc_3d[ti][m] = tmp_t + d_err; 
      pts[i][m] = tmp_t + d_err; 
      // T = getTransformFromMatches(this, older_node, matches); 
      {
        absolute_pose::CentralAbsoluteAdapter adapter(bearingVs, pts); 
        T_epnp = absolute_pose::epnp(adapter); 
        T.block<3,4>(0,0) = T_epnp; 
      }
      fT = T.cast<float>(); 
      x1 = F(fT); 
      // older_node->m_feature_loc_3d[ti][m] = tmp_t - d_err; 
      pts[i][m] = tmp_t - d_err; 
      {
        absolute_pose::CentralAbsoluteAdapter adapter(bearingVs, pts); 
        T_epnp = absolute_pose::epnp(adapter); 
        T.block<3,4>(0,0) = T_epnp; 
      }
      // T = getTransformFromMatches(this, older_node, matches);
      fT = T.cast<float>(); 
      x2 = F(fT); 
      dFdtt.col(i*3+m) = (x1-x2)/de2; 
      // recover 
      // older_node->m_feature_loc_3d[ti][m] = tmp_t; 
      pts[i][m] = tmp_t; 
      
      // second dF2/dtf
      tmp_f = this->m_feature_loc_3d[fi][m]; 
      this->m_feature_loc_3d[fi][m] = tmp_f + d_err; 
      Eigen::Vector4f& ptf = this->m_feature_loc_3d[fi];  
      pt << ptf(0), ptf(1), ptf(2);
      pt *= (1./pt.norm()); 
      bearingVs[i] = pt; 
    
      {
        absolute_pose::CentralAbsoluteAdapter adapter(bearingVs, pts); 
        T_epnp = absolute_pose::epnp(adapter); 
        T.block<3,4>(0,0) = T_epnp; 
      }

      // T = getTransformFromMatches(this, older_node, matches); 
      fT = T.cast<float>(); 
      x1 = F(fT); 

      this->m_feature_loc_3d[fi][m] = tmp_f - d_err; 
      // T = getTransformFromMatches(this, older_node, matches); 
      pt << ptf(0), ptf(1), ptf(2);
      pt *= (1./pt.norm()); 
      bearingVs[i] = pt; 

      {
        absolute_pose::CentralAbsoluteAdapter adapter(bearingVs, pts); 
        T_epnp = absolute_pose::epnp(adapter); 
        T.block<3,4>(0,0) = T_epnp; 
      }

      fT = T.cast<float>(); 
      x2 = F(fT); 
      dFdtf.col(i*3+m) = (x1-x2)/de2; 
      // recover
      this->m_feature_loc_3d[fi][m] = tmp_f; 
      pt << ptf(0), ptf(1), ptf(2);
      pt *= (1./pt.norm()); 
      bearingVs[i] = pt; 
    }
  }

  // 2. compute Stt and Stf
  if(CCameraNode::mp_camCov_ == 0)
  {
    cout <<__FILE__<<" has not set camCov, use constant instead \n"; 
    CCameraNode::mp_camCov_ =  new CamCov(); 
  }
  // Eigen::Matrix<double, 3*N, 3*N> Stt = Eigen::Matrix<double, 3*N, 3*N>::Zero();
  // Eigen::Matrix<double, 3*N, 3*N> Stf = Eigen::Matrix<double, 3*N, 3*N>::Zero();
  Eigen::MatrixXd Stt  = Eigen::MatrixXd::Zero(3*N, 3*N); 
  Eigen::MatrixXd Stf  = Eigen::MatrixXd::Zero(3*N, 3*N); 
  double Sxx, Syy, Szz; 
  float u, v; 
  double z; 
  
  for(int i=0; i<N; i++)
  {
    fi = inliers[i].queryIdx; 
    ti = inliers[i].trainIdx; 
 
   // 2.1 pt_to 
   z = older_node->m_feature_loc_3d[ti][2]; 
   u = older_node->m_feature_loc_2d[ti].pt.x; 
   v = older_node->m_feature_loc_2d[ti].pt.y; 
   CCameraNode::mp_camCov_->getSigma(u, v, z, Sxx, Syy, Szz); 
   Stt(i*3,i*3) = Sxx; 
   Stt(i*3+1, i*3+1) = Syy; 
   Stt(i*3+2, i*3+2) = Szz; 
    
   if(std::isnan(Sxx) || std::isnan(Syy) || std::isnan(Szz))
   {
      cout<<__FILE__<<" u v z: "<<u<<" "<<v<<" "<<z<<endl;
      cout<<__FILE__<<" nan Sxx, Syy or Szz"<< Sxx <<" "<<Syy<<" "<<Szz<<endl; 
   }

   // 2.2 pt_from 
   z = this->m_feature_loc_3d[fi][2]; 
   u = this->m_feature_loc_2d[fi].pt.x; 
   v = this->m_feature_loc_2d[fi].pt.y; 
   CCameraNode::mp_camCov_->getSigma(u, v, z, Sxx, Syy, Szz); 
   Stf(i*3, i*3) = Sxx; 
   Stf(i*3+1, i*3+1) = Syy; 
   Stf(i*3+2, i*3+2) = Szz; 
   
   if(std::isnan(Sxx) || std::isnan(Syy) || std::isnan(Szz))
   {
      cout<<__FILE__<<" u v z: "<<u<<" "<<v<<" "<<z<<endl;
      cout<<__FILE__<<" nan Sxx, Syy or Szz"<< Sxx <<" "<<Syy<<" "<<Szz<<endl; 
   }

  }
  
  // 3. compute Covariance 
  cov = dFdtt*Stt*dFdtt.transpose() + dFdtf*Stf*dFdtf.transpose(); 
  // ouf<<std::fixed<<"F1: \n"<<dFdtt<<endl<<"F2: \n"<<dFdtf<<endl<<"Stt: \n"<<Stt<<endl    <<"Stf: \n"<<Stf<<endl    <<"cov: \n"<<cov<<endl;

  if(std::isnan(cov(0,0)))
  {
    ofstream ouf("debug_cov.log"); 
    ouf<<std::fixed<<"F1: \n"<<dFdtt<<endl<<"F2: \n"<<dFdtf<<endl<<"Stt: \n"<<Stt<<endl    <<"Stf: \n"<<Stf<<endl    <<"cov: \n"<<cov<<endl;
  }

  double precision = 1e8;
  double precision_inv = 1e-8; 
  Eigen::Matrix<double, 6,6 > Inf = cov.inverse(); 
  Eigen::Matrix<double, 6,6> I6 = Eigen::Matrix<double, 6,6>::Identity(); 
  Inf = Inf * cov; 
  if(!VRO::MatrixEqual(Inf, I6, 1e-5))
  {
    cout <<" Inf*COV is not Identity, try to reduce precision "<<endl<<Inf<<endl; 
    VRO::reducePrecision(cov, precision, precision_inv); 
    Inf = cov.inverse(); 
    Inf = Inf * cov; 
    if(!VRO::MatrixEqual(Inf, I6, 1e-5))
    {
      ROS_ERROR("failed to make it invertable!"); 
    }else
    {
      ROS_WARN("succeed to make it invertable by reducing precision!"); 
    }
  }

  return ;
}

bool CCameraNodePnP::getRelativeTransformationTo(CCameraNode* older_node, std::vector<cv::DMatch>* initial_matches, Eigen::Matrix4f& resulting_transformation, float& rmse, std::vector<cv::DMatch>& matches)
{
    assert(initial_matches != NULL); 

    // RANSAC 3D-2D PnP methods in opengv 
    bearingVectors_t bearingVectors; 
    points_t points; 
    
    for(int i=0; i< initial_matches->size(); i++)
    {
      cv::DMatch& m = (*initial_matches)[i]; 
      
      Eigen::Vector4f pi = older_node->m_feature_loc_3d[m.trainIdx]; 
      Eigen::Vector4f pj = this->m_feature_loc_3d[m.queryIdx]; 

      point_t bodyPoint(pi(0), pi(1), pi(2)); 
      points.push_back(bodyPoint); 
      Eigen::Vector3d pt; pt << pj(0), pj(1), pj(2); 
      pt *= (1./pt.norm()); 
      bearingVectors.push_back(pt); 
    }

    absolute_pose::CentralAbsoluteAdapter adapter(bearingVectors, points); 
    
    // ransac
    sac::Ransac<sac_problems::absolute_pose::AbsolutePoseSacProblem> ransac; 
    std::shared_ptr<
      sac_problems::absolute_pose::AbsolutePoseSacProblem> absposeproblem_ptr(
          new sac_problems::absolute_pose::AbsolutePoseSacProblem(
            adapter, 
            sac_problems::absolute_pose::AbsolutePoseSacProblem::KNEIP)); 
    ransac.sac_model_ = absposeproblem_ptr; 
    ransac.threshold_ = 1.0 - cos(2.5*atan(sqrt(2.)*0.5/800.)); 
    ransac.max_iterations_ = m_ransac_num; 

    bool succeed = ransac.computeModel(); 

    // print result 
    // cout << "3D-2D ransac result is: "<<endl<<ransac.model_coefficients_<<endl; 
    // cout << "the number of inliers: "<<ransac.inliers_.size()<<endl; 
    
    if(succeed) succeed = ransac.inliers_.size() >= m_min_inliers;  // check whether the number of inliers can be larger than this 

    if(succeed )
    {
      transformation_t T_epnp = absolute_pose::epnp(adapter, ransac.inliers_);
      
      cout <<"T_epnp = "<<endl<<T_epnp<<endl; 
      cout <<"T_kneip = "<<endl<<ransac.model_coefficients_<<endl;

      resulting_transformation.block<3,4>(0,0) =  T_epnp.cast<float>(); // ransac.model_coefficients_.cast<float>(); 
      matches.resize(ransac.inliers_.size()); 
      for(int i=0; i<ransac.inliers_.size(); i++)
      {
        cv::DMatch& m = matches[i]; 
        m = (*initial_matches)[ransac.inliers_[i]]; 
      }
      
      // compute covariance 
      // computeCovPnP(older_node, matches, ransac.inliers_, cov_Helper,  bearingVectors, points); 

    }
  return succeed; 
}


MatchingResult CCameraNodePnP::matchNodePair(CCameraNode* older_node)
{
  MatchingResult mr; 
  mr.succeed_match = false;
  bool found_transformation = false; 

  this->featureMatching(older_node, &mr.all_matches); 
  
  if(mr.all_matches.size() < m_min_inliers)
  {
     ROS_WARN("%s at %d Too few inliers between %i and %i for RANSAC Threshold = %d", __FILE__, __LINE__, this->m_id, older_node->m_id, m_min_inliers); 
  }else
  {
     found_transformation = getRelativeTransformationTo(older_node, &mr.all_matches, mr.ransac_trafo, mr.rmse, mr.inlier_matches); 
     if(found_transformation)
     {
        mr.final_trafo = mr.ransac_trafo; 
        // mr.edge.informationMatrix = Eigen::Matrix<double, 6, 6>::Identity()*(mr.inlier_matches.size()/(mr.rmse*mr.rmse)); 
        mr.edge.id1 = older_node->m_id; 
        mr.edge.id2 = this->m_id; 
        mr.edge.transform = mr.final_trafo.cast<double>(); 
        mr.succeed_match = true;
        ROS_INFO("RANSAC found a %s transformation with %d inliers matches, initial matches %d", found_transformation? "valid" : "invalid", (int) mr.inlier_matches.size(), (int)mr.all_matches.size());
     }else{
        mr.edge.id1 = -1; 
        mr.edge.id2 = -1;
        mr.succeed_match = false; 
        ROS_INFO("RANSAC found no valid trafo, but had initially %d feature matches",(int) mr.all_matches.size());
     }
    
      // display the inlier matching 
      if(CCameraNode::gb_dis_match_point)
      {
        ROS_WARN("%s at Line %d to show matched points!", __FILE__, __LINE__);
        cv::Mat img1 = m_img; 
        cv::Mat img2 = older_node->m_img; 
        cv::Mat outImg; 
        cv::drawMatches(img1, m_feature_loc_2d, img2, older_node->m_feature_loc_2d, mr.inlier_matches, outImg); 
        cv::imshow("debug_matches", outImg); 
        cv::waitKey(0); 
      } 
  }

  return mr; 
}
