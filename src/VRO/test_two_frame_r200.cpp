/*
 * Oct. 7, 2017 David Z 
 *
 *
 * test VRO between two frames using RealSense R200 
 *
 * */

#include <ros/ros.h>
#include <string> 
#include <tf/tf.h>
#include <iostream>
#include "sparse_feature_vo.h"
// #include "SR_reader_cv.h"
#include "rs_r200_wrapper.h"
#include "camera_node.h"

using namespace cv; 
using namespace std; 

#define R2D(r) (((r)*180.)/M_PI)

void testTwoFrameMatch(); 
// void map_raw_img_to_grey(unsigned short * pRaw, unsigned char* pGrey, int N);
void print_tf(ostream& out, tf::Transform tT); 
void init_parameters(); 

int main(int argc, char* argv[])
{
  ros::init(argc, argv, "vro_rs2h_two_frame"); 
  ros::NodeHandle n;
  testTwoFrameMatch(); 
  return 0; 
}

void testTwoFrameMatch()
{
  init_parameters(); 
  string file_dir = "/home/david/work/data/up/320_60";
  string match_from = "1506980525.731679244.png"; 
  string match_to = "1506980571.752547004.png"; 
  
  ros::NodeHandle nh("~"); 
  nh.param("rs_file_dir", file_dir, file_dir); 
  nh.param("match_from", match_from, match_from);
  nh.param("match_to", match_to, match_to); 

  // set camera model 
  float d = 1.; // 2.;
  // CamModel c2h(608.1673/d, 605.7/2,  323.717/d, 228.8423/d, 0.18159, -0.60858); 

  CamModel c2h(313.8326, 316.9582, 153.7794, 118.7975, -0.0845, 0.05186);
  c2h.width = 320; // 640 
  c2h.height = 240; // 480

  CCameraNode::set_cam_cov(c2h); 
  float dpt_scale = 0.001; 
  c2h.setDepthScale(dpt_scale); 

  // CSReader r4k; 
  CRSR200Wrapper rs2h; 

  // generate imgs and dpts 
  cv::Mat from_dpt; // (SR_HEIGHT, SR_WIDTH, CV_16UC1); 
  cv::Mat to_dpt;   // (SR_HEIGHT, SR_WIDTH, CV_16UC1); 
  cv::Mat from_grey; 
  cv::Mat to_grey; 

  CCameraNode::gb_dis_match_point = true; // whether to see the result of 
  CSparseFeatureVO spr_vo(c2h); 
  
  string f_from_rgb, f_from_dpt, f_to_rgb, f_to_dpt; 
  
  f_from_rgb = file_dir + "/color/" + match_from; 
  f_from_dpt = file_dir + "/depth/" + match_from; 

  if(!rs2h.readOneFrameCV(f_from_rgb, f_from_dpt, from_grey, from_dpt))
  {
    ROS_INFO("%s failed to read files: %s frame %s ", __FILE__, f_from_rgb.c_str(), f_from_dpt.c_str()); 
    return ;
  }

  // display imgs 
  // cv::imshow("from img", from_grey); 
  // cv::waitKey(10); 

  f_to_rgb = file_dir + "/color/" + match_to;
  f_to_dpt = file_dir + "/depth/" + match_to; 

  if(!rs2h.readOneFrameCV(f_to_rgb, f_to_dpt, to_grey, to_dpt )) 
  {
    ROS_INFO("%s failed to read files : %s frame %s ", __FILE__, f_to_rgb.c_str(), f_to_dpt.c_str()); 
    return ;
  }

  cv::imshow("to img", to_grey); 
  cv::waitKey(10); 

  tf::Transform tran = spr_vo.VRO(to_grey, to_dpt, from_grey, from_dpt, dpt_scale); 

  print_tf(std::cout, tran); 
  return ; 
}


void init_parameters()
{
  // parameters configure 
  CParams* pP = CParams::Instance(); 
  pP->m_feature_detector_type = "SIFT";  // SIFTGPU 
  pP->m_feature_descriptor_type = "SIFT";  // SIFTGPU
  pP->m_feature_match_type = "FLANN"; 
  pP->m_nn_distance_ratio = 0.4; // 0.5 0.95 for SIFTGPU, 0.5-0.7 for SIFT 
  pP->m_max_dist_for_inliers = 0.07; // 0.05  
  pP->m_max_num_features = 5000; 
  pP->m_ransac_iterations = 5000; 
  pP->m_sift_num_features = 0; 
  pP->m_sift_contrast_threshold = 0.02;  // 0.02 0.04
  pP->m_sift_octave_layers = 5;
  pP->m_min_matches = 4;
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


