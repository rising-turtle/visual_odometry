project(visual_odometry) 

include_directories(../../../cam_model/src ../../../plane/src)

set(VRO_srcs feature_detector.cpp feature_adjuster.cpp aorb.cpp camera_node.cpp camera_node_ba.cpp camera_node_pnp.cpp sparse_feature_vo.cpp 
   scoped_timer.cpp matching_result.cpp transformation_estimation_euclidean.cpp vro_parameter.cpp)  # cam_model.cpp cam_cov.cpp
if (${USE_SIFT_GPU})
  set(VRO_srcs ${VRO_srcs} sift_gpu_wrapper.cpp)
endif (${USE_SIFT_GPU})

add_library(vro ${VRO_srcs})
message("OpenCV_LIBS: ${OpenCV_LIBS}")
target_link_libraries(vro cam_model ${catkin_LIBRARIES} ${QT_LIBRARIES} ${LIBS_LINK} ${OpenCV_LIBS} opengv )

# add_library(sr4k_io SR_reader.cpp SR_reader_cv.cpp)

add_executable(testVROSR4k test_vro_sr4k.cpp)
target_link_libraries(testVROSR4k vro sr4k_io)

add_executable(testVRORS2h test_vro_r200.cpp)
target_link_libraries(testVRORS2h vro rs2h_io)

add_executable(testVRORS2h_2f test_two_frame_r200.cpp)
target_link_libraries(testVRORS2h_2f vro rs2h_io)

add_executable(test_cov test_cov.cpp)
target_link_libraries(test_cov vro ${catkin_LIBRARIES})


########### testPnP method #############
# add_executable(test_PnP test_PnP.cpp)
# target_link_libraries(test_PnP vro sr4k_io ${catkin_LIBRARIES} plane opengv)
