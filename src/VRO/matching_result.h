/* This file is part of RGBDSLAM.
 * 
 * RGBDSLAM is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * RGBDSLAM is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with RGBDSLAM.  If not, see <http://www.gnu.org/licenses/>.
 */


#ifndef MATCHING_RESULT_H
#define MATCHING_RESULT_H
#include <opencv2/features2d/features2d.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>

struct LoadedEdge3D
{
  int id1, id2;
  //enum { OTHER=0, RANSAC, ICP } edge_type; 
  //edge_type type;
  //g2o::SE3Quat mean;
  Eigen::Isometry3d transform;
  Eigen::Matrix<double, 6,6> informationMatrix;
};

class MatchingResult {
    public:
        MatchingResult(); 
        std::vector<cv::DMatch> inlier_matches;
        std::vector<cv::DMatch> all_matches;
        LoadedEdge3D edge;
        float rmse;
        Eigen::Matrix4f ransac_trafo;
        Eigen::Matrix4f final_trafo;
        Eigen::Matrix4f icp_trafo;
        unsigned int inlier_points, outlier_points, occluded_points, all_points;
        const char* toString();
        bool succeed_match;
    public:
      EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};
        
#endif
