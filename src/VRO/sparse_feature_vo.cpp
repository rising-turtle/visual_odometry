#include "sparse_feature_vo.h"
#include "feature_detector.h"
#include "convert.h"

#ifdef USE_SIFT_GPU
#include "sift_gpu_wrapper.h"
#endif


using namespace std;

/*CSparseFeatureVO::CSparseFeatureVO(CamModel cm, std::string detector_type, 
    std::string extractor_type, std::string match_type) : 
  m_cam_model(cm), m_detector_type(detector_type), m_extractor_type(extractor_type), m_match_type(match_type)
{
  mp_detector = myCreateDetector(m_detector_type); 
  mp_extractor = createDescriptorExtractor(m_extractor_type); 
}

CSparseFeatureVO::CSparseFeatureVO(CamModel cm, CParams p) : 
  m_cam_model(cm), m_params(p)
{
  mp_detector = myCreateDetector(m_params.m_detector_type); 
  mp_extractor = createDescriptorExtractor(m_params.m_descriptor_type); 
}*/

void removeDepthless(std::vector<cv::KeyPoint>& f_loc_2d, cv::Mat& dpt);
void depthToCV8UC1(cv::Mat& depth_img, cv::Mat& mono8_img);

CSparseFeatureVO::CSparseFeatureVO(CamModel cm) : 
  m_cam_model(cm)
{
  mp_detector = myCreateDetector(CParams::Instance()->m_feature_detector_type); 
  mp_extractor = createDescriptorExtractor(CParams::Instance()->m_feature_descriptor_type); 
}
CSparseFeatureVO::~CSparseFeatureVO(){}

void CSparseFeatureVO::featureExtraction(cv::Mat& visual, cv::Mat& depth, float depth_scale, CCameraNode& cn, cv::Mat mask)
{
  // convert visual img into grey img
  cv::Mat grey_img; 
  if(visual.type() == CV_8UC3){
    cvtColor(visual, grey_img, CV_RGB2GRAY);
  } else 
  {
    grey_img = visual;
  }
  if(CCameraNode::gb_dis_match_point)
  {
    cn.m_img = visual.clone(); // if try to display image 
  }

#ifdef USE_SIFT_GPU
  // detect features, and extract descriptors 
  if(CParams::Instance()->m_feature_detector_type == "SIFTGPU")
  {
    // SIFT GPU 
    std::vector<float> descriptors;
    SiftGPUWrapper* siftgpu = SiftGPUWrapper::getInstance();
    // TODO: add mask for siftgpu->detect()
    siftgpu->detect(grey_img, cn.m_feature_loc_2d, descriptors);
    // ROS_INFO("detect siftgpu 2d features %d ", cn.m_feature_loc_2d.size());
    projectTo3DSiftGPU(depth, depth_scale, descriptors, cn); 
    // cn.setFeatureType(CParams::Instance()->m_detector_type); 
    // ROS_INFO("Siftgpu Feature Descriptors size: %d x %d", cn.m_feature_descriptors.rows, cn.m_feature_descriptors.cols);
  }else
#endif
  {
    // cv::Mat mono8_img; 
    // depthToCV8UC1(depth, mono8_img);

    // Other features 
    mp_detector->detect( grey_img, cn.m_feature_loc_2d, mask);// mono8_img detection_mask //  fill 2d locations
    // ROS_INFO("%s at first detect sift 2d features %d ", __FILE__, cn.m_feature_loc_2d.size());

    // 1, removedepthless() 
    removeDepthless(cn.m_feature_loc_2d, depth); 
    // ROS_INFO("after remove Depth there are %d features left", cn.m_feature_loc_2d.size());

    // 2, max_keypoints cv::KeyPointsFilter::retainBest(feature_locations_2d_, max_keyp);  
    if(cn.m_feature_loc_2d.size() > CParams::Instance()->m_max_num_features)
    {
      cv::KeyPointsFilter::retainBest(cn.m_feature_loc_2d, CParams::Instance()->m_max_num_features); 
      cn.m_feature_loc_2d.resize(CParams::Instance()->m_max_num_features);
    }
    // 3, compute locations of feature points 
    projectTo3D(depth, depth_scale, cn); //takes less than 0.01 sec

    // 4, compute descriptors 
    mp_extractor->compute(grey_img, cn.m_feature_loc_2d, cn.m_feature_descriptors); //fill feature_descriptors with information 
    // cn.setFeatureType(CParams::Instance()->m_detector_type);
    // ROS_INFO("%s detect feature points: %zu", __FILE__, cn.m_feature_loc_2d.size());
    // ROS_INFO("Feature Descriptors size: %d x %d", cn.m_feature_descriptors.rows, cn.m_feature_descriptors.cols);
  }

  // 
  // ROS_INFO(""); 

  return ;
}

tf::Transform CSparseFeatureVO::VRO(CCameraNode& tar, CCameraNode& src, _F F, Eigen::Matrix<double, 6, 6>* cov)
{
  MatchingResult mr = src.matchNodePair(&tar); 
  tf::Transform ret = eigenTransf2TF(mr.final_trafo); 
  
  if(F != 0 && cov !=0)
  {
    src.computeCov(&tar, mr.inlier_matches, F, (*cov)); 
  }

  return ret; 
}

tf::Transform CSparseFeatureVO::VRO(cv::Mat& tar_i, cv::Mat& tar_d, cv::Mat& src_i, cv::Mat& src_d, float depth_scale, _F F, Eigen::Matrix<double, 6, 6>* cov )
{
  CCameraNode src_n; 
  featureExtraction(src_i, src_d, depth_scale, src_n); 
  
  CCameraNode tar_n; 
  featureExtraction(tar_i, tar_d, depth_scale, tar_n); 
 
  return VRO(tar_n, src_n, F, cov); 
}

tf::Transform CSparseFeatureVO::VRO(cv::Mat& tar_i, cv::Mat& tar_d, cv::Mat& tar_mask, cv::Mat& src_i, cv::Mat& src_d, cv::Mat& src_mask, float depth_scale)
{
  CCameraNode src_n; 
  featureExtraction(src_i, src_d, depth_scale, src_n, src_mask); 
  
  CCameraNode tar_n; 
  featureExtraction(tar_i, tar_d, depth_scale, tar_n, tar_mask); 
 
  return VRO(tar_n, src_n); 
}


void CSparseFeatureVO::generatePointCloud(cv::Mat& rgb, cv::Mat& depth, int skip, float depth_scale, vector<float>& pts, vector<unsigned char>& color, int * rect)
{
  double z; 
  double px, py, pz; 
  int N = (rgb.rows/skip)*(rgb.cols/skip); 
  pts.reserve(N*3); 
  color.reserve(N*3); 

  unsigned char r, g, b; 
  int pixel_data_size = 3; 
  if(rgb.type() == CV_8UC1)
  {
    pixel_data_size = 1; 
  }
  
  int color_idx; 
  char red_idx = 2, green_idx =1, blue_idx = 0;

  int sv, su, ev, eu; 
  if(rect == 0)
  {
    sv = su = 0; 
    ev = rgb.rows; 
    eu = rgb.cols; 
  }else{
    su = rect[0]; sv = rect[1]; eu = rect[2]; ev = rect[3]; 
    su = su < 0 ? 0 : su;   sv = sv < 0 ? 0 : sv; 
    eu = eu <= rgb.cols ? eu : rgb.cols; ev = ev <= rgb.rows ? ev : rgb.rows;
  }

  for(int v = sv; v<ev; v+=skip)
  for(int u = su; u<eu; u+=skip)
  {
    z = depth.at<unsigned short>((v), (u))*depth_scale;
    if(std::isnan(z) || z <= 0.0) continue; 
    m_cam_model.convertUVZ2XYZ(u, v, z, px, py, pz); 
    pts.push_back(px);  pts.push_back(py);  pts.push_back(pz); 
    color_idx = (v*rgb.cols + u)*pixel_data_size;
    if(pixel_data_size == 3)
    {
      r = rgb.at<uint8_t>(color_idx + red_idx);
      g = rgb.at<uint8_t>(color_idx + green_idx); 
      b = rgb.at<uint8_t>(color_idx + blue_idx);
    }else{
      r = g = b = rgb.at<uint8_t>(color_idx); 
    }
    color.push_back(r); color.push_back(g); color.push_back(b); 
  }
  return ;
}

void CSparseFeatureVO::projectTo3D( cv::Mat& depth, float depth_scale, CCameraNode& cn)
{
  // bool use_feature_min_depth = ParameterServer::instance()->get<bool>("use_feature_min_depth"); //TODO
  // size_t max_keyp = ParameterServer::instance()->get<int>("max_keypoints");
  std::vector<cv::KeyPoint>& feature_locations_2d = cn.m_feature_loc_2d; 
  std::vector<Eigen::Vector4f, Eigen::aligned_allocator<Eigen::Vector4f> >& feature_locations_3d = cn.m_feature_loc_3d; 

  cv::Point2f p2d;
  if(feature_locations_3d.size())
  {
    feature_locations_3d.clear();
  }

  float u, v; 
  double z, px, py, pz; 

  int d_out_range = 0, d_nanz = 0, d_nanpt = 0; 
  // ROS_INFO("%s before projectTo3D, 2d feature size = %d", __FILE__, feature_locations_2d.size());
  for(unsigned int i = 0; i < feature_locations_2d.size(); /*increment at end of loop*/)
  {
    p2d = feature_locations_2d[i].pt;
    u = p2d.x; v = p2d.y; 
    if (u >= depth.cols || u < 0 ||
        v >= depth.rows || v < 0 ||
        std::isnan(p2d.x) || std::isnan(p2d.y))
    { //TODO: Unclear why points should be outside the image or be NaN
      feature_locations_2d.erase(feature_locations_2d.begin()+i);
      ++d_out_range; 
      continue;
    }
    // z = depth.at<unsigned short>(int(v+0.5), int(u+0.5)) * depth_scale; 
    z = depth.at<unsigned short>((int)(v), (int)(u))*depth_scale;
    if(std::isnan(z) || z <= 0.0) 
    {
      feature_locations_2d.erase(feature_locations_2d.begin()+i);
      ++d_nanz; 
      continue;
    }
    m_cam_model.convertUVZ2XYZ(u, v, z, px, py, pz); 
    if(std::isnan (px) || std::isnan(py) || std::isnan(pz))
    {
      // ROS_DEBUG("Feature %d has been extracted at NaN depth. Omitting", i);
      //FIXME Use parameter here to choose whether to use

      feature_locations_2d.erase(feature_locations_2d.begin()+i);
      ++d_nanpt; 
      continue;
    }
    /*
    if(i== 259)
    {
      ROS_WARN("feature 259 at u %f v %f X Y Z : %lf %lf %lf", u, v, px, py, pz);
    }
    */
    feature_locations_3d.push_back(Eigen::Vector4f(px, py, pz, 1.0));
    i++; //Only increment if no element is removed from vector
    // if(feature_locations_3d.size() >= max_keyp) break;
  }

  // ROS_WARN("%s out_range %d nanz %d nanpt %d feature_pts has %d",__FILE__, d_out_range, d_nanz, d_nanpt, feature_locations_3d.size());
  feature_locations_2d.resize(feature_locations_3d.size());
  return ;
}

#ifdef USE_SIFT_GPU
// project 2d descriptors 
void CSparseFeatureVO::projectTo3DSiftGPU(cv::Mat& depth, float depth_scale, 
        std::vector<float>& descriptors_in, CCameraNode& cn)
{ 
  std::vector<cv::KeyPoint>& feature_locations_2d = cn.m_feature_loc_2d; 
  std::vector<Eigen::Vector4f, Eigen::aligned_allocator<Eigen::Vector4f> >& feature_locations_3d = 
    cn.m_feature_loc_3d; 
  cv::Mat& descriptors_out = cn.m_feature_descriptors; 
  std::vector<float>& siftgpu_descriptors = cn.m_siftgpu_descriptors; 

  //  clear feature 3d 
  if(feature_locations_3d.size())
  {
    feature_locations_3d.clear();
  }

  // parameters 
  std::list<int> featuresUsed;
  cv::Point2f p2d; 
  int index = -1;
  float u, v;
  double z, px, py, pz; 
  for(unsigned int i = 0; i < feature_locations_2d.size(); /*increment at end of loop*/)
  {
    ++index;   
    p2d = feature_locations_2d[i].pt;
    u = p2d.x;  v = p2d.y; 
    double z = depth.at<unsigned short>((int)(v+0.5), (int)(u+0.5)) * depth_scale ;
    if(std::isnan(z) || z <= 0.0)
    {
       // ROS_WARN("delete point at i = %d with z = %lf", i, z);
       feature_locations_2d.erase(feature_locations_2d.begin()+i);
       continue; 
    }

    m_cam_model.convertUVZ2XYZ(u, v, z, px, py, pz); 
    if(std::isnan (px) || std::isnan(py) || std::isnan(pz))
    {
      // ROS_DEBUG("Feature %d has been extracted at NaN depth. Omitting", i);
      //FIXME Use parameter here to choose whether to use
      feature_locations_2d.erase(feature_locations_2d.begin()+i);
      continue;
    }

    feature_locations_3d.push_back(Eigen::Vector4f(px, py, pz, 1));
    // if(i < 10)
    {
      // ROS_INFO("feature point at %d has %f %f %lf %lf %lf %lf", i, u, v, z, px, py, pz); 
    }
    i++; //Only increment if no element is removed from vector
    featuresUsed.push_back(index);  //save id for constructing the descriptor matrix
  }
  
  // create descriptor matrix
  int size = feature_locations_3d.size();
  descriptors_out = cv::Mat(size, 128, CV_32F);
  siftgpu_descriptors.resize(size * 128);
  for (int y = 0; y < size && featuresUsed.size() > 0; ++y) 
  {
    int id = featuresUsed.front();
    featuresUsed.pop_front();
    for (int x = 0; x < 128; ++x) 
    {
      descriptors_out.at<float>(y, x) = descriptors_in[id * 128 + x];
      siftgpu_descriptors[y * 128 + x] = descriptors_in[id * 128 + x];
    }
  }
  feature_locations_2d.resize(feature_locations_3d.size());
  return ;
}
#endif

void depthToCV8UC1(cv::Mat& depth_img, cv::Mat& mono8_img)
{
  if(depth_img.type() == CV_32FC1)
  {
    depth_img.convertTo(mono8_img, CV_8UC1, 100, 0); 
  }else if(depth_img.type() == CV_16UC1)
  {
    mono8_img = cv::Mat(depth_img.size(), CV_8UC1); 
    depth_img.convertTo(mono8_img, CV_8UC1, 0.05, -25); 
  }
}

void removeDepthless(std::vector<cv::KeyPoint>& f_loc_2d, cv::Mat& dpt)
{
  cv::Point2f p2d; 
  unsigned int i=0; 
  unsigned short depth; 
  while(i<f_loc_2d.size())
  {
    p2d = f_loc_2d[i].pt; 
    if(std::isnan(p2d.x) || std::isnan(p2d.y) || p2d.x >= dpt.cols || p2d.x<0 || p2d.y >= dpt.rows || p2d.y <0)
    {
      f_loc_2d.erase(f_loc_2d.begin() + i); 
      continue; 
    }
    depth = dpt.at<unsigned short>((int)(p2d.y+0.5), (int)(p2d.x+0.5)); 
    if(std::isnan(depth))
    {
      f_loc_2d.erase(f_loc_2d.begin() + i); 
      continue; 
    }
    i++;
  }
}

