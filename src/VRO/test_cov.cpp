/*
 * Nov.9, 2016, David Z
 *
 * test covariance 
 *
 * */

#include <iostream>
#include <tf/tf.h>
#include "camera_node.h"
#include "transformation_estimation_euclidean.h"

using namespace std; 
using namespace Eigen; 

Matrix<double, 6, 1> F(Matrix4f&); 

int main(int argc, char* argv[])
{
  // construct node 1 from 
  CCameraNode* pn1 = new CCameraNode;  // this node 
  
  {
    pn1->m_feature_loc_2d.resize(5); 

    // initialize five points 
    Vector4f p1(3.1427, 4.0579, -3.7301, 1.); pn1->m_feature_loc_3d.push_back(p1);     
    Vector4f p2(4.1338, 1.3236, -4.0246, 1); pn1->m_feature_loc_3d.push_back(p2); 
    Vector4f p3(-2.215, 0.4688, 4.5751, 1);  pn1->m_feature_loc_3d.push_back(p3); 
    Vector4f p4(4.6489, -3.4239, 4.7059, 1); pn1->m_feature_loc_3d.push_back(p4); 
    Vector4f p5(4.5717, -0.1462, 3.0028, 1); pn1->m_feature_loc_3d.push_back(p5); 
  }

  // construct node 2 to
  CCameraNode * pn2 = new CCameraNode; // older node
  
  {
    pn2->m_feature_loc_2d.resize(5); 

    // initialize five points 
    Vector4f p1(0.9209, 4.5365, -3.075, 1); pn2->m_feature_loc_3d.push_back(p1);     
    Vector4f p2(0.8639, 2.2714, -4.9196, 1); pn2->m_feature_loc_3d.push_back(p2); 
    Vector4f p3(2.2066, -2.6096, 4.5337, 1);  pn2->m_feature_loc_3d.push_back(p3); 
    Vector4f p4(6.3067, -5.3763, -1.616, 1); pn2->m_feature_loc_3d.push_back(p4); 
    Vector4f p5(5.7262, -1.7316, -1.4455, 1); pn2->m_feature_loc_3d.push_back(p5); 
  }
  
  // construct dmatch 
  std::vector<cv::DMatch> m; 
  for(int i=0;i<5; i++)
  {
    cv::DMatch mm; 
    mm.trainIdx = i; mm.queryIdx = i; 
    m.push_back(mm); 
  }

  // tg [0.5 -1 0], eg [0.5236, 0.7854, 0.1745] 
  Matrix4f T = getTransformFromMatches(pn1, pn2, m); 
  Matrix<double, 6, 1> r = F(T); 
  cout <<"result : \n"<<r<<endl; 

  // compute covariance 
  Matrix<double, 6, 6> C; 
  pn1->computeCov(pn2, m, F, C); 
  cout <<"cov: \n "<<C*1e3<<endl; 

  return 0; 
}


Matrix<double, 6, 1> F(Matrix4f& T)
{
  Matrix<double, 6, 1> r; 
  tf::Matrix3x3 R; 
  R.setValue(T(0,0), T(0,1), T(0,2), 
             T(1,0), T(1,1), T(1,2), 
             T(2,0), T(2,1), T(2,2)); 
  
  r(0) = T(0,3); r(1) = T(1,3); r(2) = T(2,3); 
  R.getEulerYPR(r(5), r(4), r(3)); 
  return r;
}



