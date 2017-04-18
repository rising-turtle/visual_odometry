#include "camera_node.h"
#include <ros/ros.h>
#include "transformation_estimation_euclidean.h"
#include <tf/tf.h>
#include <pcl_ros/transforms.h>

#include <iostream>
#include <fstream>
#include "cam_cov.h"
#include "cam_model.h" 
#include "covariance_check.h"

#ifdef USE_SIFT_GPU
#include "sift_gpu_wrapper.h"
#endif

// default set it as zero 
static CamCov* CCameraNode::mp_camCov_ = 0; 
static bool CCameraNode::gb_dis_match_point = false; 

void CCameraNode::set_cam_cov(CamModel& m)
{
  if(CCameraNode::mp_camCov_ != 0)
  {
    delete CCameraNode::mp_camCov_; 
    CCameraNode::mp_camCov_ = 0; 
  }
  CCameraNode::mp_camCov_ = new CamCov(m); 
}


using namespace std ; 
namespace {

#define D2R(d) ((d)*M_PI/180.)
#define COST(number, mean_error) ((mean_error)/(double(number)))
  tf::Transform getTranRPYt(double r, double p, double y, double tx, double ty, double tz)
  {
    tf::Transform tt; 
    tf::Quaternion q; 
    q.setRPY(r, p, y); 
    tf::Matrix3x3 R(q); 
    tt.setBasis(R); 
    tt.setOrigin(tf::Vector3(tx, ty, tz)); 
    return tt;
  }
}

std::vector<cv::DMatch> sample_matches_prefer_by_distance(unsigned int sample_size, 
    std::vector<cv::DMatch>& matches_with_depth);

QMutex CCameraNode::siftgpu_mutex;

// CCameraNode::CCameraNode(std::string match_type): m_match_type(match_type), m_pflannIndex(NULL){}
CCameraNode::CCameraNode(): m_pflannIndex(NULL), m_id(-1){}
CCameraNode::~CCameraNode(){}

MatchingResult CCameraNode::matchNodePair(CCameraNode* older_node)
{
  MatchingResult mr; 
  mr.succeed_match = false;
  try{
    bool found_transformation = false; 
    this->featureMatching(older_node, &mr.all_matches); 
    // TODO: parameterize this variable 
    // unsigned int min_initial_matches = 20; 
    unsigned int min_initial_matches = CParams::Instance()->m_min_matches;

    if(mr.all_matches.size() < min_initial_matches)
    {
      ROS_WARN("%s at %d Too few inliers between %i and %i for RANSAC Threshold = %d", __FILE__, __LINE__, this->m_id, older_node->m_id, min_initial_matches); 
    }else
    {
      found_transformation = getRelativeTransformationTo(older_node, &mr.all_matches, mr.ransac_trafo, mr.rmse, mr.inlier_matches); 
      if(found_transformation) // find a good transformation 
      {
        mr.final_trafo = mr.ransac_trafo; 
        mr.edge.informationMatrix = Eigen::Matrix<double, 6, 6>::Identity()*(mr.inlier_matches.size()/(mr.rmse*mr.rmse)); 
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
  }catch(std::exception e)
  {
    ROS_ERROR("Caught Exception in comparison of cam_node %i and %i : %s", this->m_id, older_node->m_id, e.what()); 
  }
  return mr; 
}

void CCameraNode::computeCov(CCameraNode* older_node, std::vector<cv::DMatch>& matches, _F F, Eigen::Matrix<double, 6, 6>& cov)
{
  // for debug 
  // ofstream ouf("debug_cov.log"); 

  // 1, compute dF1/dpt1, dF2/dpt2  
  int N = matches.size(); 
  // Eigen::Matrix<double, 6, 3*N> dFdtt = Eigen::Matrix<double, 6, 3*N>::Zero(); 
  // Eigen::Matrix<double, 6, 3*N> dFdtf = Eigen::Matrix<double, 6, 3*N>::Zero();
  Eigen::MatrixXd dFdtt(6, 3*N); 
  Eigen::MatrixXd dFdtf(6, 3*N); 
  double d_err = 1e-3; 
  double de2 = 2*d_err; 
  int fi, ti; 
  float tmp_f, tmp_t; 
  Eigen::Matrix4f T; 
  Eigen::Matrix<double, 6, 1> x1, x2; // depends on what you define [euler, t] or [q, t] or [t q]
  for(int i=0; i<N; i++)
  {
    fi = matches[i].queryIdx; 
    ti = matches[i].trainIdx; 
    for(int m=0; m<3; m++)
    {
      // first dF1/dtt
      tmp_t = older_node->m_feature_loc_3d[ti][m]; 
      older_node->m_feature_loc_3d[ti][m] = tmp_t + d_err; 
      T = getTransformFromMatches(this, older_node, matches); 
      x1 = F(T); 
      older_node->m_feature_loc_3d[ti][m] = tmp_t - d_err; 
      T = getTransformFromMatches(this, older_node, matches);
      x2 = F(T); 
      dFdtt.col(i*3+m) = (x1-x2)/de2; 
      // recover 
      older_node->m_feature_loc_3d[ti][m] = tmp_t; 
      
      // second dF2/dtf
      tmp_f = this->m_feature_loc_3d[fi][m]; 
      this->m_feature_loc_3d[fi][m] = tmp_f + d_err; 
      T = getTransformFromMatches(this, older_node, matches); 
      x1 = F(T); 
      this->m_feature_loc_3d[fi][m] = tmp_f - d_err; 
      T = getTransformFromMatches(this, older_node, matches); 
      x2 = F(T); 
      dFdtf.col(i*3+m) = (x1-x2)/de2; 
      // recover
      this->m_feature_loc_3d[fi][m] = tmp_f; 
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
    fi = matches[i].queryIdx; 
    ti = matches[i].trainIdx; 
 
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

void CCameraNode::getRANSACInliers(CCameraNode* other, std::vector<cv::DMatch>* initial_matches, 
    Eigen::Matrix4f& resulting_transformation, float& rmse, std::vector<cv::DMatch>& matches)
{
  unsigned int min_inlier_matches = 5;
  if(initial_matches->size() < min_inlier_matches)
  {
    ROS_INFO("only %d initial features between %d cam_node and %d cam_node", initial_matches->size(), this->m_id, other->m_id); 
    return false; 
  }
  float max_dist_m = CParams::Instance()->m_max_dist_for_inliers; 
  int ransac_iterations = CParams::Instance()->m_ransac_iterations; 

  int N = initial_matches->size();
  double one_over_indices = 1./(double)(N);
  double probability_ = 0.99;
  double log_probability  = log (1.0 - probability_);

  // initialization 
  matches.clear(); 
  resulting_transformation = Eigen::Matrix4f::Identity(); 
  rmse = 1e6;
  double final_cost = -1; 
  unsigned int valid_iteration = 0;  // best inlier cnt 
  const unsigned int sample_size = 4;  // chose this many randomly from the correspondences 
  bool valid_tf = false; 
  
  std::vector<cv::DMatch> * matches_with_depth = initial_matches; 
  std::sort(matches_with_depth->begin(), matches_with_depth->end()); 

  int real_iterations = 0; 
  for(int n=0; (n < ransac_iterations && matches_with_depth->size() >= sample_size); n++)
  {
    // Initialize Results of refinement 
    double refined_error = 1e6; 
    double refined_cost = -1;
    double inlier_error ; 
    std::vector<cv::DMatch> refined_matches; 
    std::vector<cv::DMatch> inlier = sample_matches_prefer_by_distance(sample_size, * matches_with_depth); 
    Eigen::Matrix4f refined_transformation = Eigen::Matrix4f::Identity(); 
    
    real_iterations ++ ; 
    Eigen::Matrix4f transformation = getTransformFromMatches(this, other, inlier, valid_tf, max_dist_m*2.5); 

    for(int refinements = 1; refinements < 20; refinements ++)
    {
        if(!valid_tf || transformation != transformation) break;  // not a good inlier guess 
      
        computeInliersAndError(*initial_matches, transformation, this->m_feature_loc_3d, 
              other->m_feature_loc_3d, refined_matches.size(), inlier, inlier_error, max_dist_m * max_dist_m); 

        if(inlier.size() < min_inlier_matches || inlier_error > max_dist_m)
        {
          if(inlier.size() > 0)
          {
          }
          break;  // hopeless case 
        }
        if(refined_cost == -1 ||  inlier.size() >= refined_matches.size() && inlier_error <= refined_error )
        {
          // refined_cost = COST(inlier.size(), inlier_error); 
          size_t prev_num_inliers = refined_matches.size(); 
          assert(inlier_error >= 0); 
          refined_transformation = transformation; 
          refined_matches = inlier; 
          refined_error = inlier_error; 
          if(inlier.size() == prev_num_inliers) break;  // only error changed, no change next iteration 
        }else
        {
          ROS_INFO("at loop n = %d inlier.size() = %d , inlier_error = %f", n, inlier.size(), inlier_error); 
          break;
        }
        transformation = getTransformFromMatches(this, other, inlier, valid_tf, 0); 
    }
    
    if(refined_matches.size() > 0)   // valid iteration 
    {
      if(refined_error <= rmse && refined_matches.size() >= matches.size() && 
           refined_matches.size() >= min_inlier_matches)
      // if(final_cost == -1 || (COST(refined_matches.size(), refined_error) < final_cost && refined_matches.size() > matches.size()*0.5))
      {
        ROS_WARN("valid match found at loop n = %d with matches size %d and rmse %f ", n, refined_matches.size(), refined_error);

        // final_cost = COST(refined_matches.size(), refined_error); 
        valid_iteration ++ ; 
        rmse = refined_error ; 
        resulting_transformation = refined_transformation; 
        matches.assign(refined_matches.begin(), refined_matches.end());

        if(refined_matches.size() > initial_matches->size()*0.75) n += 10;
        if(refined_matches.size() > initial_matches->size()*0.85) n += 20; 
        if(refined_matches.size() > initial_matches->size()*0.9) 
        {
          // ROS_WARN("refined_matches.size() = %d initial_matches.size() = %d break", refined_matches.size(), initial_matches->size());
          break; 
        }

        double w = refined_matches.size()*one_over_indices; 
        double p_at_least_one_outlier = 1. - pow(w, (double)(sample_size)); 
        p_at_least_one_outlier = (std::max) (std::numeric_limits<double>::epsilon (), 
            p_at_least_one_outlier);       // Avoid division by -Inf
        p_at_least_one_outlier = (std::min) (1.0 - std::numeric_limits<double>::epsilon (),
            p_at_least_one_outlier);   // Avoid division by 0.
        int tmp = log_probability / log(p_at_least_one_outlier);
        if(tmp < 50) tmp = 50;  // at least no less than 50 times 
        if(tmp < ransac_iterations) ransac_iterations = tmp;
        // ROS_ERROR("iteration at n = %d max_iteration = %d, final.size = %d final_error %f", n, ransac_iterations, matches.size(), rmse);
      }else{
        // ROS_INFO("at loop n = %d refined.size() = %d matches.size() = %d , refined_error = %f rmse = %f ", n, refined_matches.size(), matches.size(), refined_error, rmse); 
      }
    }
    
  }
  ROS_INFO("%i good iterations (from %i), inlier pct %i, inlier cnt: %i, error (MHD): %.2f",valid_iteration, ransac_iterations, (int) (matches.size()*1.0/initial_matches->size()*100),(int) matches.size(),rmse);

  // return matches.size() >= min_inlier_matches; 
 
  return ;
}


bool CCameraNode::getRelativeTransformationTo(CCameraNode* other, std::vector<cv::DMatch>* initial_matches, 
    Eigen::Matrix4f& resulting_transformation, float& rmse, std::vector<cv::DMatch>& matches)
{
  // ROS_INFO("in %s at line %d function getRelativeTransformationTo", __FILE__, __LINE__);
  assert(initial_matches != NULL); 
  
  // TODO: parameterize this variable 
  // unsigned int min_initial_matches = 20; 
  unsigned int min_initial_matches = CParams::Instance()->m_min_matches;
  unsigned int min_inlier_matches = min_initial_matches ; // 12; // 12  5
  if(initial_matches->size() < min_initial_matches)
  {
    ROS_INFO("only %d initial features between %d cam_node and %d cam_node", initial_matches->size(), this->m_id, other->m_id); 
    return false; 
  }
  
  if(min_inlier_matches > 0.65 * initial_matches->size()) 
    min_inlier_matches = initial_matches->size()*0.65; 
  
  if(min_inlier_matches < 12) min_inlier_matches = 12; 

  float max_dist_m = CParams::Instance()->m_max_dist_for_inliers; 
  int ransac_iterations = CParams::Instance()->m_ransac_iterations; 
  // ransac_iterations = 100;  // for debug

  int N = initial_matches->size();
  double one_over_indices = 1./(double)(N);
  double probability_ = 0.99;
  double log_probability  = log (1.0 - probability_);

  // initialization 
  matches.clear(); 
  resulting_transformation = Eigen::Matrix4f::Identity(); 
  rmse = 1e6;
  double final_cost = -1; 
  unsigned int valid_iteration = 0;  // best inlier cnt 
  const unsigned int sample_size = 4;  // chose this many randomly from the correspondences 
  bool valid_tf = false; 
  
  std::vector<cv::DMatch> * matches_with_depth = initial_matches; 
  std::sort(matches_with_depth->begin(), matches_with_depth->end()); 

  int real_iterations = 0; 
  for(int n=0; (n < ransac_iterations && matches_with_depth->size() >= sample_size); n++)
  {
    // Initialize Results of refinement 
    double refined_error = 1e6; 
    double refined_cost = -1;
    double inlier_error ; 
    std::vector<cv::DMatch> refined_matches; 
    std::vector<cv::DMatch> inlier = sample_matches_prefer_by_distance(sample_size, * matches_with_depth); 
    Eigen::Matrix4f refined_transformation = Eigen::Matrix4f::Identity(); 
    
    real_iterations ++ ; 
    Eigen::Matrix4f transformation = getTransformFromMatches(this, other, inlier, valid_tf, max_dist_m*2.5); 

    // for debug 
    // if(valid_iteration == 0)
    {
      // tf::Transform T = getTranRPYt(D2R(0.12), D2R(-0.968), D2R(0.399), -0.002, -0.01, 0.0434);
      // pcl_ros::transformAsMatrix(T, transformation);
    }

    for(int refinements = 1; refinements < 20; refinements ++)
    {
        if(!valid_tf || transformation != transformation) break;  // not a good inlier guess 
      
        computeInliersAndError(*initial_matches, transformation, this->m_feature_loc_3d, 
              other->m_feature_loc_3d, refined_matches.size(), inlier, inlier_error, max_dist_m * max_dist_m); 

        /*
        if(valid_iteration == 0 && refinements == 1)
        {
          cout <<" query 280 pt: "<<this->m_feature_loc_2d[280].pt<<" train 259 pt: "<<other->m_feature_loc_2d[259].pt<<endl;

          // cout << "refinements: "<< refinements<< " first transformation : "<<transformation<<endl;
          ofstream ouf("inlier_matches.log"); 
          for(int i=0; i<inlier.size(); i++)
          {
            cv::DMatch& m = inlier[i];
            ouf << m.queryIdx <<"\t"<<m.trainIdx <<"\t"<<endl; // <<this->m_feature_loc_3d[m.queryIdx]<<"\t"<<other->m_feature_loc_3d[m.trainIdx]<<endl;
          }
          ouf.close();
        }*/

        if(inlier.size() < min_inlier_matches || inlier_error > max_dist_m)
        {
          if(inlier.size() > 0)
          {
            // ROS_INFO("%s inlier number: %d min_inlier_matches: %d inlier_error: %f max_dist_m %f", __FILE__, inlier.size(), min_inlier_matches, inlier_error, max_dist_m);
          }
          break;  // hopeless case 
        }
        // if(refined_cost == -1 || (COST(inlier.size(), inlier_error) < refined_cost && inlier.size() > refined_matches.size()*0.5))
        if(refined_cost == -1 ||  inlier.size() >= refined_matches.size() && inlier_error <= refined_error )
        {
          // ROS_WARN("at loop n = %d inlier.size() = %d , inlier_error = %f refined.size() = %d, refined_error = %f", n, inlier.size(), inlier_error, refined_matches.size(), refined_error); 
          // refined_cost = COST(inlier.size(), inlier_error); 
          size_t prev_num_inliers = refined_matches.size(); 
          assert(inlier_error >= 0); 
          refined_transformation = transformation; 
          refined_matches = inlier; 
          refined_error = inlier_error; 
          if(inlier.size() == prev_num_inliers) break;  // only error changed, no change next iteration 
        }else
        {
          ROS_INFO("at loop n = %d inlier.size() = %d , inlier_error = %f", n, inlier.size(), inlier_error); 
          break;
        }
        transformation = getTransformFromMatches(this, other, inlier, valid_tf, 0); 
    }
    
    if(refined_matches.size() > 0)   // valid iteration 
    {
      if(refined_error <= rmse && refined_matches.size() >= matches.size() && 
           refined_matches.size() >= min_inlier_matches)
      // if(final_cost == -1 || (COST(refined_matches.size(), refined_error) < final_cost && refined_matches.size() > matches.size()*0.5))
      {
        ROS_WARN("valid match found at loop n = %d with matches size %d and rmse %f ", n, refined_matches.size(), refined_error);

        // final_cost = COST(refined_matches.size(), refined_error); 
        valid_iteration ++ ; 
        rmse = refined_error ; 
        resulting_transformation = refined_transformation; 
        matches.assign(refined_matches.begin(), refined_matches.end());

        if(refined_matches.size() > initial_matches->size()*0.75) n += 10;
        if(refined_matches.size() > initial_matches->size()*0.85) n += 20; 
        if(refined_matches.size() > initial_matches->size()*0.9) 
        {
          // ROS_WARN("refined_matches.size() = %d initial_matches.size() = %d break", refined_matches.size(), initial_matches->size());
          break; 
        }

        double w = refined_matches.size()*one_over_indices; 
        double p_at_least_one_outlier = 1. - pow(w, (double)(sample_size)); 
        p_at_least_one_outlier = (std::max) (std::numeric_limits<double>::epsilon (), 
            p_at_least_one_outlier);       // Avoid division by -Inf
        p_at_least_one_outlier = (std::min) (1.0 - std::numeric_limits<double>::epsilon (),
            p_at_least_one_outlier);   // Avoid division by 0.
        int tmp = log_probability / log(p_at_least_one_outlier);
        if(tmp < 50) tmp = 50;  // at least no less than 50 times 
        if(tmp < ransac_iterations) ransac_iterations = tmp;
        // ROS_ERROR("iteration at n = %d max_iteration = %d, final.size = %d final_error %f", n, ransac_iterations, matches.size(), rmse);
      }else{
        // ROS_INFO("at loop n = %d refined.size() = %d matches.size() = %d , refined_error = %f rmse = %f ", n, refined_matches.size(), matches.size(), refined_error, rmse); 
      }
    }
    
  }
  ROS_INFO("%i good iterations (from %i), inlier pct %i, inlier cnt: %i, error (MHD): %.2f",valid_iteration, ransac_iterations, (int) (matches.size()*1.0/initial_matches->size()*100),(int) matches.size(),rmse);

  return matches.size() >= min_inlier_matches; 
}

void CCameraNode::computeInliersAndError(std::vector<cv::DMatch> & all_matches, 
    Eigen::Matrix4f& transformation4f, 
    std::vector<Eigen::Vector4f, Eigen::aligned_allocator<Eigen::Vector4f> >& origins, 
    std::vector<Eigen::Vector4f, Eigen::aligned_allocator<Eigen::Vector4f> >& earlier,
    size_t min_inliers, 
    std::vector<cv::DMatch>& inliers, 
    double& return_mean_error, 
    double squaredMaxInlierDistInM)
{
    inliers.clear(); 
    assert(all_matches.size() > 0); 
    inliers.reserve(all_matches.size()); 
    const size_t all_matches_all = all_matches.size(); 
    double mean_error = 0.0; 
    Eigen::Matrix4d transformation4d = transformation4f.cast<double>();

    for(int i=0; i<all_matches.size(); i++)
    {
      cv::DMatch & m = all_matches[i]; 
      Eigen::Vector4f& ori = origins[m.queryIdx]; 
      Eigen::Vector4f& tar = earlier[m.trainIdx]; 
      if(ori(2) == 0. || tar(2) == 0.) continue;  // bad points 
      Eigen::Vector3f v_ori = ori.head<3>(); 
      Eigen::Vector3f v_tar = tar.head<3>();
      double euclidean_dis = errorFunction(v_ori, v_tar, transformation4d);
      /*
      static bool once = true;
      if(m.queryIdx == 280 && once)
      {
        ROS_ERROR("queryIdx = %d trainIdx = %d error: %f dis_thresold = %f", m.queryIdx, m.trainIdx, euclidean_dis, squaredMaxInlierDistInM);
        cout << "v_ori: "<<v_ori<< " v_tar: "<<v_tar<<endl;
        once = false; 
      }*/

      if(euclidean_dis > squaredMaxInlierDistInM) continue; // outlier 
     
      mean_error += euclidean_dis; 
      inliers.push_back(m); 
    }
    
    if(inliers.size() < 3)
    {
      // ROS_WARN("number of inliers is only %d ", inliers.size()); 
      return_mean_error = 1e9; 
    }else
    {
      mean_error /= inliers.size(); 
      return_mean_error = sqrt(mean_error); 
    }
    return ;
}

double CCameraNode::errorFunction(Eigen::Vector3f& x1, Eigen::Vector3f& x2, Eigen::Matrix4d& transformation)
{
  // NAN CHECK 
  bool nan1 = isnan(x1(2));
  bool nan2 = isnan(x2(2));
  if(nan1 || nan2)
  {
    //FIXME: Handle Features with NaN, by reporting the reprojection error
    return std::numeric_limits<double>::max();
  }

  // transformation 
  Eigen::Matrix4d tf_12 = transformation;
  Eigen::Vector4d x_1, x_2; 
  x_1.head<3>() = x1.cast<double>(); x_1(3) = 1.; 
  x_2.head<3>() = x2.cast<double>(); x_2(3) = 1.;
  Eigen::Vector3d mu_1 = x_1.head<3>();
  Eigen::Vector3d mu_2 = x_2.head<3>();
  Eigen::Vector3d mu_1_in_frame_2 = (tf_12 * x_1).head<3>(); // 
  double delta_sq_norm = (mu_1_in_frame_2 - mu_2).squaredNorm();
  return delta_sq_norm;
}

unsigned int CCameraNode::featureMatching(CCameraNode* other, std::vector<cv::DMatch>* matches)
{
  if(matches->size() > 0) 
    matches->clear(); 
  
  // neighbors to search 
  const int k = 2; 
  double sum_distances = 0.0;

  // double max_dist_ratio_fac = ps->get<double>("nn_distance_ratio");
  double max_dist_ratio_fac = CParams::Instance()->m_nn_distance_ratio; // TODO: parameterize this variable, siftgpu = 0.95, for sift, 0.5 or 0.7 
#ifdef USE_SIFT_GPU
  if(CParams::Instance()->m_feature_match_type == "SIFTGPU")
  {
    siftgpu_mutex.lock(); 
      sum_distances = SiftGPUWrapper::getInstance()->match(m_siftgpu_descriptors, m_feature_descriptors.rows, other->m_siftgpu_descriptors, other->m_feature_descriptors.rows, matches);
    siftgpu_mutex.unlock(); 
  }
#endif
  if(CParams::Instance()->m_feature_match_type == "BRUTEFORCE" || CParams::Instance()->m_feature_descriptor_type == "ORB")
  { 
    cv::Ptr<cv::DescriptorMatcher> matcher;
    std::string brute_force_type("BruteForce"); //L2 per default
    if(CParams::Instance()->m_feature_descriptor_type == "ORB")
      brute_force_type.append("-HammingLUT");
    matcher = cv::DescriptorMatcher::create(brute_force_type);
    std::vector< std::vector<cv::DMatch> > bruteForceMatches;
    matcher->knnMatch(m_feature_descriptors, other->m_feature_descriptors, bruteForceMatches, k);
    srand((long)std::clock());
    std::set<int> train_indices;
    matches->reserve(bruteForceMatches.size());
    for(unsigned int i = 0; i < bruteForceMatches.size(); i++) 
    {
      cv::DMatch m1 = bruteForceMatches[i][0];
      cv::DMatch m2 = bruteForceMatches[i][1];
      float dist_ratio_fac = m1.distance / m2.distance;
      if (dist_ratio_fac < max_dist_ratio_fac) 
      {//this check seems crucial to matching quality
        int train_idx = m1.trainIdx;
        if(train_indices.count(train_idx) > 0)
          continue; //FIXME: Keep better
        train_indices.insert(train_idx);
        sum_distances += m1.distance;
        m1.distance = dist_ratio_fac + (float)rand()/(1000.0*RAND_MAX); //add a small random offset to the distance, since later the dmatches are inserted to a set, which omits duplicates and the duplicates are found via the less-than function, which works on the distance. Therefore we need to avoid equal distances, which happens very often for ORB
        matches->push_back(m1);
      } 
    }
  }else if(CParams::Instance()->m_feature_match_type == "FLANN" && CParams::Instance()->m_feature_descriptor_type != "ORB")
  {
    // ROS_INFO("here I am, m_feature_match_type = %s m_feature_type = %s", m_feature_match_type.c_str(), m_feature_type.c_str());
    if(other->getFlannIndex() == NULL)
    {
      ROS_FATAL("CamNode %i in featureMatching: flann Index of CamNode %i was not initialized", this->m_id, other->m_id); 
      return -1; 
    }

    // TODO: use sufficient features to reduce search time  
    // now, just search all the detected features 
    // double max_dist_ratio_fac = 0.5; // TODO: parameterize this variable 
 
    // compare 
    // http://opencv-cocoa.googlecode.com/svn/trunk/samples/c/find_obj.cpp
    int start_feature = 0; 
    int num_features = m_feature_descriptors.rows; 
    cv::Mat indices(num_features, k, CV_32S); 
    cv::Mat dists(num_features, k, CV_32F); 
    cv::Mat relevantDescriptors = m_feature_descriptors.rowRange(start_feature, start_feature + num_features); 
    {
      other->knnSearch(relevantDescriptors, indices, dists, k, cv::flann::SearchParams(16)); 
    }
    
    int * indices_ptr = indices.ptr<int>(0); 
    float * dists_ptr = dists.ptr<float>(0); 
    cv::DMatch match; 
    double avg_ratio = 0.0; 
    std::set<int> train_indices; 
    matches->reserve(indices.rows); 
    for(int i=0; i<indices.rows; i++)
    {
      float dist_ratio_fac = static_cast<float>(dists_ptr[2*i])/static_cast<float>(dists_ptr[2*i+1]);
      avg_ratio += dist_ratio_fac; 
      if(max_dist_ratio_fac > dist_ratio_fac)
      {
        int train_idx = indices_ptr[2*i]; 
        if(train_indices.count(train_idx) > 0)  // this feature has been used 
          continue; 
        train_indices.insert(train_idx); 
        match.queryIdx = i; 
        match.trainIdx = train_idx; 
        match.distance = dist_ratio_fac; 
        // match.distance = static_cast<float>(dists_ptr[2*i]);
        sum_distances += match.distance; 
        
        assert(match.trainIdx < other->m_feature_descriptors.rows);
        assert(match.queryIdx < m_feature_descriptors.rows); 
        matches->push_back(match);
      }
    }
    ROS_INFO("Feature Matches between Nodes %3d (%4d features) and %3d (%4d features) (features %d to %d of first node):\t%4d. Percentage: %f%%, Avg NN Ratio: %f", this->m_id, (int)this->m_feature_loc_2d.size(), other->m_id, (int)other->m_feature_loc_2d.size(), start_feature, start_feature+num_features, (int)matches->size(), (100.0*matches->size())/((float)start_feature+num_features), avg_ratio / (start_feature+num_features));
  }

   return matches->size(); 
}

cv::flann::Index * CCameraNode::getFlannIndex()
{
  if(m_pflannIndex == NULL && CParams::Instance()->m_feature_match_type == "FLANN" && CParams::Instance()->m_feature_descriptor_type != "ORB")
  {
    m_pflannIndex = new cv::flann::Index(m_feature_descriptors, cv::flann::KDTreeIndexParams());
  }
  return m_pflannIndex;
}

std::vector<cv::DMatch> sample_matches_prefer_by_distance(unsigned int sample_size, 
    std::vector<cv::DMatch>& matches_with_depth)
{
  std::set<std::vector<cv::DMatch>::size_type> sampled_ids;
  int safety_net = 0;
  while(sampled_ids.size() < sample_size && matches_with_depth.size() >= sample_size){
    //generate a set of samples. Using a set solves the problem of drawing a sample more than once
    int id1 = rand() % matches_with_depth.size();
    int id2 = rand() % matches_with_depth.size();
    if(id1 > id2) id1 = id2; //use smaller one => increases chance for lower id
    sampled_ids.insert(id1);
    if(++safety_net > 10000){ ROS_ERROR("Infinite Sampling"); break; } 
  }

  std::vector<cv::DMatch> sampled_matches;
  sampled_matches.reserve(sampled_ids.size());
  // BOOST_FOREACH(std::vector<cv::DMatch>::size_type id, sampled_ids){
  for(std::set<std::vector<cv::DMatch>::size_type>::iterator it = sampled_ids.begin(); 
      it != sampled_ids.end(); ++it)
  {
     // sampled_matches.push_back(matches_with_depth[id]);
     sampled_matches.push_back(matches_with_depth[*it]);
  }
  return sampled_matches;
}


void CCameraNode::knnSearch(cv::Mat& query,
    cv::Mat& indices,
    cv::Mat& dists,
    int knn, 
    const cv::flann::SearchParams& params)
{
  (this->getFlannIndex())->knnSearch(query, indices, dists, knn, params);
}



