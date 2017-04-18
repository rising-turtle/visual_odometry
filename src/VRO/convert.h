/*  
 *  Sep. 30, 2016 David Z 
 *  
 *  Convert between Eigen, tf::Transform, cv::Mat 
 *
 * */


#ifndef CONVERT_H
#define CONVERT_H

#include <tf/tf.h>
#include <Eigen/Geometry>

template <typename T >
tf::Transform eigenTransf2TF(T& transf) 
{
  tf::Transform result;
  tf::Vector3 translation;
  translation.setX(transf.translation().x());
  translation.setY(transf.translation().y());
  translation.setZ(transf.translation().z());

  tf::Quaternion rotation;
  Eigen::Quaterniond quat;
  quat = transf.rotation();
  rotation.setX(quat.x());
  rotation.setY(quat.y());
  rotation.setZ(quat.z());
  rotation.setW(quat.w());

  result.setOrigin(translation);
  result.setRotation(rotation);
  //printTransform("from conversion", result);
  return result;
}

template<>
tf::Transform eigenTransf2TF(Eigen::Matrix4f& transf)
{
  tf::Transform result ; 
  Eigen::Isometry3d t; 
  Eigen::Matrix4d trd = transf.cast<double>(); 
  t.matrix() = trd; 
  return eigenTransf2TF(t); 
}

template<>
tf::Transform eigenTransf2TF(Eigen::Matrix4d& transd)
{
  tf::Transform result ;
  Eigen::Isometry3d t; 
  t.matrix() = transd; 
  return eigenTransf2TF(t); 
}


#endif
