
/*
 * Feb. 27, 2017 David Z 
 * 
 * test for PnP method using SwissRanger 4000 
 *
 * */

#include <ros/ros.h>
#include <string> 
#include <tf/tf.h>
#include <iostream>
#include <map>
#include <pcl/common/transforms.h>
#include "sparse_feature_vo.h"
#include "SR_reader_cv.h"
#include "camera_node_ba.h"
#include "pc_from_image.h"
#include "vtk_viewer.h"

#include <opengv/absolute_pose/methods.hpp>
#include <opengv/sac/Ransac.hpp>
#include <opengv/absolute_pose/CentralAbsoluteAdapter.hpp>
#include <opengv/sac_problems/absolute_pose/AbsolutePoseSacProblem.hpp>

using namespace cv; 
using namespace std; 
using namespace opengv; 

#define R2D(r) (((r)*180.)/M_PI)

void init_parameters(); 
void testTwoFrameMatch(); 
void print_tf(ostream& out, tf::Transform tT); 

int main(int argc, char* argv[])
{
  ros::init(argc, argv, "test_PnP"); 
  ros::NodeHandle n;
  init_parameters(); 
  testTwoFrameMatch(); 
  return 0; 
}

void testTwoFrameMatch()
{
  // read sr4k data from disk 
  string src_file = "../../../src/visual_odometry/data/d1_0726.bdat"; 
  string tar_file = "../../../src/visual_odometry/data/d1_0724.bdat"; 
  
  ros::NodeHandle nh("~"); 
  nh.param("src_filename", src_file, src_file); 
  nh.param("tar_filename", tar_file, tar_file); 

  // CSReader r4k; 
  CSReadCV r4k; 

  // generate imgs and dpts 
  cv::Mat tar_cv_d_img(SR_HEIGHT, SR_WIDTH, CV_16UC1); 
  cv::Mat src_cv_d_img(SR_HEIGHT, SR_WIDTH, CV_16UC1); 
  cv::Mat src_cv_i_img; 
  cv::Mat tar_cv_i_img; 

  if(!r4k.readOneFrameCV(src_file, src_cv_i_img, src_cv_d_img))
  {
    ROS_INFO("%s failed to read file %s", __FILE__, src_file.c_str()); 
    return ;
  }
  if(!r4k.readOneFrameCV(tar_file, tar_cv_i_img, tar_cv_d_img))
  {
    ROS_INFO("%s failed to read file %s", __FILE__, tar_file.c_str()); 
    return ;
  }

  // VRO 
  CamModel sr4k(250.5773, 250.5773, 90, 70, -0.8466, 0.5370); 
  sr4k.z_offset = 0.015;  // this is only for sr4k 
  sr4k.setDepthScale(0.001); 
  CCameraNode::set_cam_cov(sr4k); 
  CCameraNode::gb_dis_match_point = true; // whether to see the result of 

  // generate node 
  CSparseFeatureVO vo(sr4k); 
  CCameraNodeBA* ni = new CCameraNodeBA(); 
  CCameraNodeBA* nj = new CCameraNodeBA(); 
  vo.featureExtraction(tar_cv_i_img, tar_cv_d_img, 0.001, *ni); 
  vo.featureExtraction(src_cv_i_img, src_cv_d_img, 0.001, *nj); 
  
  MatchingResult mr = nj->matchNodePair(ni); 

  Eigen::Matrix4f Tij = mr.final_trafo;  

  // print_tf(std::cout, tran); 
  cout <<"VRO result: "<<endl<<Tij<<endl;

  // try PnP method in opengv 
  {
    bearingVectors_t bearingVectors; 
    points_t points; 
    
    for(int i=0; i< mr.all_matches.size(); i++)
    {
      cv::DMatch& m = mr.all_matches[i]; 
      
      Eigen::Vector4f pi = ni->m_feature_loc_3d[m.trainIdx]; 
      Eigen::Vector4f pj = nj->m_feature_loc_3d[m.queryIdx]; 

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
    ransac.threshold_ = 1.0 - cos(atan(sqrt(2.)*0.5/800.)); 
    ransac.max_iterations_ = 50; 

    ransac.computeModel(); 

    // print result 
    cout << "3D-2D ransac result is: "<<endl<<ransac.model_coefficients_<<endl; 
    cout << "the number of inliers: "<<ransac.inliers_.size()<<endl; 
    Tij.block<3,4>(0,0) = ransac.model_coefficients_.cast<float>(); 
  }


  CloudPtr pci(new Cloud); 
  CloudPtr pcj(new Cloud); 
  CloudPtr pcj_ni(new Cloud); 
  generatePointCloud(tar_cv_i_img, tar_cv_d_img, 0.001, sr4k, *pci);
  generatePointCloud(src_cv_i_img, src_cv_d_img, 0.001, sr4k, *pcj); 
  
  pcl::transformPointCloud(*pcj, *pcj_ni,  Tij); // mr.final_trafo); 
  
  markColor(*pci, GREEN); 
  markColor(*pcj_ni, RED); 
  *pci += *pcj_ni; 
  
  CVTKViewer<pcl::PointXYZRGBA> v;
  // v.getViewer()->addCoordinateSystem(0.2, 0, 0); 
  v.addPointCloud(pci, "pci + pcj"); 
  while(!v.stopped())
  {
    v.runOnce(); 
    usleep(100*1000); 
  }

  return ;

}


void print_tf(ostream& out, tf::Transform tT)
{
  tfScalar r, p, y, tx, ty, tz;
  tT.getBasis().getEulerYPR(y, p, r); 
  tf::Vector3 t = tT.getOrigin(); 
  tx = t.getX(); ty = t.getY(); tz = t.getZ();
  out<<"test_vro: yaw: "<<R2D(y)<<" pitch: "<<R2D(p)<<" roll: "<<R2D(r)<<" tx: "<<tx<<" ty: "<<ty<<" tz: "<<tz<<" qx = "<<
    tT.getRotation().x()<<" qy = "<<tT.getRotation().y()<<" qz= "<<tT.getRotation().z()<<" qw = "<<tT.getRotation().w()<<endl;
}

void init_parameters()
{
  // parameters configure 
  CParams* pP = CParams::Instance(); 
  pP->m_feature_detector_type = "SIFT";  // SIFTGPU 
  pP->m_feature_descriptor_type = "SIFT";  // SIFTGPU
  pP->m_feature_match_type = "FLANN"; 
  pP->m_nn_distance_ratio = 0.5; // 0.95 for SIFTGPU, 0.5-0.7 for SIFT 
  pP->m_max_dist_for_inliers = 0.05;  
  pP->m_max_num_features = 500; 
  pP->m_ransac_iterations = 5000;
  pP->m_min_matches = 12; 
}
