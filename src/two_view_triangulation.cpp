#include <iostream>
#include <vector>
#include <algorithm>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/core/eigen.hpp>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include <g2o/core/robust_kernel_impl.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
// #include <pcl/visualization/cloud_viewer.h>
#include <pcl/io/ply_io.h>



int main(int argc, char const *argv[])
{
    int fx = 1520.400000;
    int fy = 1525.900000;
    int cx = 302.320000;
    int cy = 246.870000;
    cv::Mat cameraMatrix = (cv::Mat1d(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);

    cv::Mat img1, img2;
    img1 = cv::imread("../sequence/templeR0013.png", cv::IMREAD_COLOR);
    img2 = cv::imread("../sequence/templeR0014.png", cv::IMREAD_COLOR);

    cv::Ptr<cv::ORB> orb_detector = cv::ORB::create();
    std::vector<cv::KeyPoint> kp1, kp2;
    cv::Mat des1, des2;
    orb_detector->detectAndCompute(img1, cv::noArray(), kp1, des1);
    orb_detector->detectAndCompute(img2, cv::noArray(), kp2, des2);
    
    cv::Ptr<cv::BFMatcher> matcher = cv::BFMatcher::create(cv::NORM_HAMMING, true);
    std::vector<cv::DMatch> matches;
    matcher->match(des1, des2, matches);

    // lambda to sort matches by distance between points
    std::sort(matches.begin(), matches.end(), [](const cv::DMatch& lhs, const cv::DMatch& rhs) {return lhs.distance < rhs.distance;});

    std::vector<cv::Point2i> pts1, pts2;
    for (auto match : matches) {
        pts1.push_back(kp1[match.queryIdx].pt);
        pts2.push_back(kp2[match.trainIdx].pt);
    }

    cv::Mat essentialMat, mask;
    essentialMat = cv::findEssentialMat(pts1, pts2, cameraMatrix, cv::RANSAC, 0.999, 1.0, 1000, mask);

    cv::Mat R, t;
    int numPoints = cv::recoverPose(essentialMat, pts1, pts2, cameraMatrix, R, t, mask);

    std::vector<cv::Point2f> maskedPts1, maskedPts2;
    for (int i = 0; i < mask.total(); i++)
    {
        uchar keepElement = mask.at<uchar>(i);
        if (keepElement > 0)
        {
            maskedPts1.push_back(pts1.at(i));
            maskedPts2.push_back(pts2.at(i));
        }
    }

    cv::Mat relativePose;
    cv::hconcat(R, t, relativePose);
    cv::Mat cam1ProjMat = cameraMatrix * cv::Mat::eye(3,4,CV_64F);
    cv::Mat cam2ProjMat = cameraMatrix * relativePose;
    
    cv::Mat pts4d;
    cv::triangulatePoints(cam1ProjMat, cam2ProjMat, maskedPts1, maskedPts2, pts4d);
    std::vector<cv::Vec3f> pts3d;
    for (int i = 0; i < pts4d.cols; i++)
    {
        cv::Vec4f pt4d = pts4d.col(i);
        cv::Vec3f pt3d;
        pt3d[0] = pt4d[0] / pt4d[3];
        pt3d[1] = pt4d[1] / pt4d[3];
        pt3d[2] = pt4d[2] / pt4d[3];
        pts3d.push_back(pt3d);
    }
    
    g2o::SparseOptimizer optimizer;
    optimizer.setVerbose(true);
    // sets the type of solver and initializes solver
    auto linearSolverType = std::make_unique<g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>>();
    auto solver = std::make_unique<g2o::BlockSolver_6_3>(std::move(linearSolverType));
    g2o::OptimizationAlgorithmLevenberg* algorithm = new g2o::OptimizationAlgorithmLevenberg(std::move(solver));
    optimizer.setAlgorithm(algorithm);

    int vertexId = 0;
    // add camera 1 pose as vertex to the graph
    Eigen::Quaterniond cam1RotQuat;
    cam1RotQuat.setIdentity();
    Eigen::Vector3d cam1TransVec;
    cam1TransVec.setZero();
    g2o::SE3Quat cam1Pose(cam1RotQuat, cam1TransVec);
    g2o::VertexSE3Expmap *cam1VertexSE3 = new g2o::VertexSE3Expmap();
    cam1VertexSE3->setEstimate(cam1Pose);
    cam1VertexSE3->setId(vertexId);
    cam1VertexSE3->setFixed(true);
    optimizer.addVertex(cam1VertexSE3);
    vertexId++;

    // add camera 2 pose as vertex to the graph
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> cam2RotMat;
    cv::cv2eigen(R, cam2RotMat);
    Eigen::Quaterniond cam2RotQuat((Eigen::Matrix<double, 3, 3>) cam2RotMat);
    Eigen::Vector3d cam2TransVec;
    cv::cv2eigen(t, cam2TransVec);
    g2o::SE3Quat cam2Pose(cam2RotQuat, cam2TransVec);
    g2o::VertexSE3Expmap *cam2VertexSE3 = new g2o::VertexSE3Expmap();
    cam2VertexSE3->setEstimate(cam2Pose);
    cam2VertexSE3->setId(vertexId);
    cam1VertexSE3->setFixed(false);
    optimizer.addVertex(cam2VertexSE3);
    vertexId++;
    
    for (int i = 0; i < numPoints; i++)
    {   
        // add point as vertex in the graph
        g2o::VertexPointXYZ *vertexPoint = new g2o::VertexPointXYZ();
        vertexPoint->setId(vertexId);
        vertexPoint->setMarginalized(true);
        cv::Vec3f pt3d = pts3d.at(i);
        Eigen::Vector3d estimate(pt3d[0], pt3d[1], pt3d[2]);   
        vertexPoint->setEstimate(estimate);
        vertexPoint->setFixed(false);
        optimizer.addVertex(vertexPoint);
        vertexId++;

        // add the edge between the point and cam1 pose
        g2o::EdgeSE3ProjectXYZ *edge1 =  new g2o::EdgeSE3ProjectXYZ();
        edge1->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(vertexPoint));
        edge1->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(cam1VertexSE3));
        cv::Point2f pt1 = maskedPts1.at(i);
        Eigen::Vector2d measurement1(pt1.x, pt1.y);
        edge1->setMeasurement(measurement1);
        edge1->setInformation(Eigen::Matrix2d::Identity());
        edge1->fx = fx;
        edge1->fy = fy;
        edge1->cx = cx;
        edge1->cy = cy;
        g2o::RobustKernelHuber* rk1 = new g2o::RobustKernelHuber;
        edge1->setRobustKernel(rk1);
        optimizer.addEdge(edge1);

        // add the edge between the point and cam2 pose
        g2o::EdgeSE3ProjectXYZ *edge2 =  new g2o::EdgeSE3ProjectXYZ();
        edge2->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(vertexPoint));
        edge2->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(cam2VertexSE3));
        cv::Point2f pt2 = maskedPts2.at(i);
        Eigen::Vector2d measurement2(pt2.x, pt2.y);
        edge2->setMeasurement(measurement2);
        edge2->setInformation(Eigen::Matrix2d::Identity());
        edge2->fx = fx;
        edge2->fy = fy;
        edge2->cx = cx;
        edge2->cy = cy;
        g2o::RobustKernelHuber* rk2 = new g2o::RobustKernelHuber;
        edge2->setRobustKernel(rk2);
        optimizer.addEdge(edge2);

        // Eigen::Vector2d reprojection1 = edge1->cam_project(cam1VertexSE3->estimate().map(vertexPoint->estimate()));
        // reprojectionErrorBeforeOptimization += (measurement1 - reprojection1).squaredNorm();
        // Eigen::Vector2d reprojection2 = edge2->cam_project(cam2VertexSE3->estimate().map(vertexPoint->estimate()));
        // reprojectionErrorBeforeOptimization += (measurement2 - reprojection2).squaredNorm();
    }

    optimizer.initializeOptimization();
    std::cout << "Performing Bundle Adjustment:" << std::endl;
    optimizer.optimize(50);

    // get the optimized points
    std::vector<Eigen::Vector3d> optimizedPoints;
    for (int i = 0; i < numPoints; i++)
    {
        int id = i + 2;
        g2o::VertexPointXYZ* vertexPoint = static_cast<g2o::VertexPointXYZ*>(optimizer.vertex(id));
        Eigen::Vector3d estimate = vertexPoint->estimate();
        optimizedPoints.push_back(estimate);
    }
    
    pcl::PointCloud<pcl::PointXYZRGB> cloud;
    cloud.width = numPoints * 2;
    cloud.height = 1;
    cloud.is_dense = true;
    cloud.resize(cloud.height * cloud.width);
    
    for (size_t i = 0; i < numPoints; i++)
    {
        // set unoptimized 3d point
        cv::Vec3f point = pts3d.at(i);
        cloud.points[i].x = point[0];
        cloud.points[i].y = point[1];
        cloud.points[i].z = point[2];

        // set color of unoptimized point
        cloud.points[i].r = 0;
        cloud.points[i].g = 0;
        cloud.points[i].b = 255;

        // set optimized 3d point
        Eigen::Vector3d optimizedPoint = optimizedPoints.at(i);
        cloud.points[i*2].x = optimizedPoint[0];
        cloud.points[i*2].y = optimizedPoint[1];
        cloud.points[i*2].z = optimizedPoint[2];
        
        // set color of optimized point
        cloud.points[i*2].r = 255;
        cloud.points[i*2].g = 0;
        cloud.points[i*2].b = 0;
    }


    // bug between PCL and VTK: https://github.com/PointCloudLibrary/pcl/issues/5237
    // have to build from master, which I don't care for
    // pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloudPtr = cloud.makeShared();
    // pcl::visualization::CloudViewer viewer ("Simple Cloud Viewer");
    // viewer.showCloud (cloudPtr);
    // while (!viewer.wasStopped ()){;}

    // save point cloud as file
    pcl::io::savePLYFileBinary("../two_view_triangulation_ba.ply", cloud);
    return 0;
}
