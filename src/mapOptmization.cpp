/************************************************* 
GitHub: https://github.com/smilefacehh/LIO-SAM-DetailedNote
Author: lutao2014@163.com
Date: 2021-02-21 
--------------------------------------------------
功能简介:
    1、scan-to-map匹配：提取当前激光帧特征点（角点、平面点），局部关键帧map的特征点，执行scan-to-map迭代优化，更新当前帧位姿；
    2、关键帧因子图优化：关键帧加入因子图，添加激光里程计因子、GPS因子、闭环因子，执行因子图优化，更新所有关键帧位姿；
    3、闭环检测：在历史关键帧中找距离相近，时间相隔较远的帧设为匹配帧，匹配帧周围提取局部关键帧map，同样执行scan-to-map匹配，得到位姿变换，构建闭环因子数据，加入因子图优化。

订阅：
    1、订阅当前激光帧点云信息，来自FeatureExtraction；
    2、订阅GPS里程计；
    3、订阅来自外部闭环检测程序提供的闭环数据，本程序没有提供，这里实际没用上。


发布：
    1、发布历史关键帧里程计；
    2、发布局部关键帧map的特征点云；
    3、发布激光里程计，rviz中表现为坐标轴；
    4、发布激光里程计；
    5、发布激光里程计路径，rviz中表现为载体的运行轨迹；
    6、发布地图保存服务；
    7、发布闭环匹配关键帧局部map；
    8、发布当前关键帧经过闭环优化后的位姿变换之后的特征点云；
    9、发布闭环边，rviz中表现为闭环帧之间的连线；
    10、发布局部map的降采样平面点集合；
    11、发布历史帧（累加的）的角点、平面点降采样集合；
    12、发布当前帧原始点云配准之后的点云。
**************************************************/ 
#include "utility.h"
#include "lio_sam/cloud_info.h"
#include "lio_sam/save_map.h"

#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/navigation/GPSFactor.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/inference/Symbol.h>

#include <gtsam/nonlinear/ISAM2.h>

// F：使用PCL库进行RANSAC平面拟合可能需要添加额外的头文件
#include <pcl/point_cloud.h>
#include <pcl/filters/plane_clipper3D.h>
#include <pcl/filters/extract_indices.h>
// 这个头文件包含了实现随机采样一致性（RANSAC）算法的类，用于拟合点云中的几何模型。
#include <pcl/sample_consensus/ransac.h> 
// 这个头文件包含了用于拟合平面模型的 RANSAC 模型类
#include <pcl/sample_consensus/sac_model_plane.h>
// 这个头文件包含了用于计算点云法向量的类
#include <pcl/features/normal_3d.h>

using namespace gtsam;

// GTSAM库中的符号缩写
using symbol_shorthand::X; // Pose3 (x,y,z,r,p,y)
using symbol_shorthand::V; // Vel   (xdot,ydot,zdot)
using symbol_shorthand::B; // Bias  (ax,ay,az,gx,gy,gz)
using symbol_shorthand::G; // GPS pose


/**
 * 6D位姿点云结构定义
*/
struct PointXYZIRPYT
{
    PCL_ADD_POINT4D     // 用于定义点的四维坐标的宏
    PCL_ADD_INTENSITY;  // 用于添加点云强度信息的宏
    float roll;         
    float pitch;
    float yaw;
    double time;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW  // 确保内存对齐的宏 
} EIGEN_ALIGN16;                     // 另一个宏，指定内存对齐方式


// 使用PCL的宏来注册 PointXYZIRPYT 结构体，便在PCL中使用这个结构体作为点云数据类型。
// 这个宏用于在pcl中注册自定义的点云数据结构。通过注册，PCL可以识别并使用这个结构体作为点云类型。
POINT_CLOUD_REGISTER_POINT_STRUCT (PointXYZIRPYT,
                                   (float, x, x) (float, y, y)
                                   (float, z, z) (float, intensity, intensity)
                                   (float, roll, roll) (float, pitch, pitch) (float, yaw, yaw)
                                   (double, time, time))

typedef PointXYZIRPYT  PointTypePose;//对已存在的数据类型起别名

// 公共继承
class mapOptimization : public ParamServer
{
    // F
private:
    // 地面点云数据队列
    std::deque<sensor_msgs::PointCloud2> groundcloudQueue;
    // 队列front帧，作为当前处理帧点云
    // typedef pcl::PointXYZI PointType;
    sensor_msgs::PointCloud2 currentGroundCloudMsg;
    pcl::PointCloud<PointGroundType>::Ptr   GroundCloudIn;

public:

    // gtsam
    NonlinearFactorGraph gtSAMgraph;  //因子图
    Values initialEstimate; //GTSAM初始位姿
    Values optimizedEstimate; //GTSAM优化后的状态估计
    ISAM2 *isam; // isam优化器
    Values isamCurrentEstimate;//Gtsam的优化结果，保存优化后所有的关键帧的位姿
    Eigen::MatrixXd poseCovariance; //位姿的先验协方差矩阵

    // 发布器
    ros::Publisher pubLaserCloudSurround;//发布全局地图点云（当前关键帧的周围关键帧对应的角点点云和面点点云，应用关键帧位姿变换到全局系下）
    ros::Publisher pubLaserOdometryGlobal; //发布里程计（优化后）
    ros::Publisher pubLaserOdometryIncremental;//发布里程计（优化前）
    ros::Publisher pubKeyPoses; //发布关键帧位姿
    ros::Publisher pubPath;//发布轨迹

    ros::Publisher pubHistoryKeyFrames;//发布历史关键帧（闭环匹配关键帧的局部map（拼接点云））
    ros::Publisher pubIcpKeyFrames;//发布ICP关键帧（即源点云应用ICP得到的变换矩阵后的点云）
    ros::Publisher pubRecentKeyFrames;// 发布局部map的降采样后的平面点集合
    ros::Publisher pubRecentKeyFrame;// 发布历史帧（累加的）的角点、平面点降采样集合
    ros::Publisher pubCloudRegisteredRaw;//发布当前帧的点云（点云的位姿经过因子图优化）
    ros::Publisher pubLoopConstraintEdge;//发布闭环边，rviz中表现为闭环帧之间的连线

    ros::Subscriber subCloud;//订阅当前激光帧点云信息，来自featureExtraction函数
    ros::Subscriber subGPS;//订阅GPS数据
    ros::Subscriber subLoop;//订阅回环数据（好像没用到，本程序未提供外部的回环数据）
    ros::Subscriber subground;// F1:订阅地面点云

    ros::ServiceServer srvSaveMap;//保存地图服务

    std::deque<nav_msgs::Odometry> gpsQueue;//	GPS数据队列
    lio_sam::cloud_info cloudInfo;//当前激光帧角点、平面点集合

    // 历史所有关键帧的角点集合（降采样）
    vector<pcl::PointCloud<PointType>::Ptr> cornerCloudKeyFrames;
    // 历史所有关键帧的平面点集合（降采样）
    vector<pcl::PointCloud<PointType>::Ptr> surfCloudKeyFrames;
    
    // 历史关键帧位姿（位置）  历史关键帧集合
    // 将关键帧的位姿以pcl::PointCloud格式来存，方便使用pcl自带的kd树寻找最近邻及降采样
    // pcl::PointCloud表明使用PCL库中的点云数据结构
    // <PointType>为模板参数，指定了点云中每个点的数据类型
    // Ptr为智能指针，用于指向PCL点云对象实例
    // cloudKeyPoses3D是一个指向PCL点云对象的指针，该对象中存储了类型为PointType的点的3D坐标
    // PointType：3D位姿的别名；PointTypePose：6D位姿的别名
    pcl::PointCloud<PointType>::Ptr cloudKeyPoses3D;
    // 历史关键帧位姿（位置+姿态）
    pcl::PointCloud<PointTypePose>::Ptr cloudKeyPoses6D;
    // 历史关键帧位姿的拷贝（位置），用于回环检测
    pcl::PointCloud<PointType>::Ptr copy_cloudKeyPoses3D;
    // 历史关键帧位姿的拷贝（位置+姿态），用于回环检测
    pcl::PointCloud<PointTypePose>::Ptr copy_cloudKeyPoses6D;

    // 当前（最新一帧）激光帧角点集合
    pcl::PointCloud<PointType>::Ptr laserCloudCornerLast; 
    // 当前激光帧平面点集合
    pcl::PointCloud<PointType>::Ptr laserCloudSurfLast; 
    // 当前激光帧角点集合，降采样，DS: DownSize
    pcl::PointCloud<PointType>::Ptr laserCloudCornerLastDS; 
    // 当前激光帧平面点集合，降采样
    pcl::PointCloud<PointType>::Ptr laserCloudSurfLastDS; 

    // 当前帧与局部map匹配上了的角点、平面点，加入同一集合；后面是对应点的参数
    pcl::PointCloud<PointType>::Ptr laserCloudOri;//存储当前帧和局部地图匹配上的角点和面点
    pcl::PointCloud<PointType>::Ptr coeffSel;//存储当前帧和局部地图匹配上的角点和面点的参数

    // 当前帧与局部map匹配上了的角点、参数、标记
    std::vector<PointType> laserCloudOriCornerVec; // 角点
    std::vector<PointType> coeffSelCornerVec; // 参数
    std::vector<bool> laserCloudOriCornerFlag; // 标记
    // 当前帧与局部map匹配上了的平面点、参数、标记
    std::vector<PointType> laserCloudOriSurfVec; 
    std::vector<PointType> coeffSelSurfVec;
    std::vector<bool> laserCloudOriSurfFlag;

    // 键是int类型，值是一对点云对象，每个点云对象都是pcl::PointCloud<PointType>类型的
    // 局部地图数据，存的是关键帧的角点和面点集合（降采样）
    map<int, pair<pcl::PointCloud<PointType>, pcl::PointCloud<PointType>>> laserCloudMapContainer;
    // 局部map的角点集合
    pcl::PointCloud<PointType>::Ptr laserCloudCornerFromMap;
    // 局部map的平面点集合
    pcl::PointCloud<PointType>::Ptr laserCloudSurfFromMap;
    // 局部map的角点集合，降采样
    pcl::PointCloud<PointType>::Ptr laserCloudCornerFromMapDS;
    // 局部map的平面点集合，降采样
    pcl::PointCloud<PointType>::Ptr laserCloudSurfFromMapDS;

    // 局部关键帧构建的map点云，对应kdtree，用于scan-to-map找相邻点 最近邻查找
    pcl::KdTreeFLANN<PointType>::Ptr kdtreeCornerFromMap;
    pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurfFromMap;

    pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurroundingKeyPoses;//存放周围关键帧（3D位姿）
    pcl::KdTreeFLANN<PointType>::Ptr kdtreeHistoryKeyPoses;//存放历史关键帧（3D位姿）

    // 降采样滤波器
    pcl::VoxelGrid<PointType> downSizeFilterCorner;
    pcl::VoxelGrid<PointType> downSizeFilterSurf;
    pcl::VoxelGrid<PointType> downSizeFilterICP;
    pcl::VoxelGrid<PointType> downSizeFilterSurroundingKeyPoses; // for surrounding key poses of scan-to-map optimization
    
    ros::Time timeLaserInfoStamp;//当前激光数据的时间戳（ros单位）
    double timeLaserInfoCur;//当前激光数据的时间戳（单位为秒）

    // 当前帧激光的位姿估计（优化前和优化后都是用这个保存）
    float transformTobeMapped[6]; // 包含旋转和平移信息的数据，存储位姿的数组

    std::mutex mtx; //全局-互斥锁
    std::mutex mtxLoopInfo;// 回环检测-互斥锁

    bool isDegenerate = false;//是否退化
    cv::Mat matP;//矩阵，退化时使用

    // 局部map角点降采样后的size
    int laserCloudCornerFromMapDSNum = 0;
    // 局部map平面点降采样后的size
    int laserCloudSurfFromMapDSNum = 0;
    // 最新帧激光角点降采样后的size
    int laserCloudCornerLastDSNum = 0;      
    // 最新帧激光面点降采样后的size
    int laserCloudSurfLastDSNum = 0;

    // 添加回环检测因子完成标志位（添加回环检测因子后置为true，优化后并完成位姿更新置为false）
    bool aLoopIsClosed = false;
    // 	已经回环的关键帧
    // 具有回环关系的关键帧以及对应的闭环匹配帧
    map<int, int> loopIndexContainer; // from new to old
    // 回环检测对（哪两帧发生闭环）
    vector<pair<int, int>> loopIndexQueue;
    // 回环检测的相对位姿（gtsam形式）
    vector<gtsam::Pose3> loopPoseQueue;
    // 回环检测的噪声
    // 用于存储指向 gtsam::noiseModel::Diagonal 类型共享指针（shared_ptr）的元素。
    // 每个对象由一个共享指针管理，以便进行动态内存分配和释放
    vector<gtsam::noiseModel::Diagonal::shared_ptr> loopNoiseQueue;
    // 回环检测队列（未使用） 存放外部回环数据，本程序没有提供
    deque<std_msgs::Float64MultiArray> loopInfoVec;

    nav_msgs::Path globalPath;//全局的路径

    // 当前帧位姿
    // transformTobeMapped的矩阵表示，将当前帧激光从激光坐标系转到地图坐标系
    Eigen::Affine3f transPointAssociateToMap;
    // 前一帧位姿
    // 上一帧关键帧的位姿
    Eigen::Affine3f incrementalOdometryAffineFront;
    // 当前帧位姿
    // 当前关键帧的位姿
    Eigen::Affine3f incrementalOdometryAffineBack;

    /**
     * 构造函数
    */
    mapOptimization()
    {
        // ISAM2参数
        ISAM2Params parameters; // 用于存储iSAM2的参数设置
        parameters.relinearizeThreshold = 0.1;
        parameters.relinearizeSkip = 1;
        isam = new ISAM2(parameters); // 创建了一个新的iSAM2优化器对象，将之前设置的参数传递给它

        // 发布历史关键帧里程计
        pubKeyPoses                 = nh.advertise<sensor_msgs::PointCloud2>("lio_sam/mapping/trajectory", 1);
        // 发布局部关键帧map的特征点云
        pubLaserCloudSurround       = nh.advertise<sensor_msgs::PointCloud2>("lio_sam/mapping/map_global", 1);
        // 发布激光里程计，rviz中表现为坐标轴（预积分模块中订阅的激光里程计）
        pubLaserOdometryGlobal      = nh.advertise<nav_msgs::Odometry> ("lio_sam/mapping/odometry", 1);
        // 发布激光里程计，它与上面的激光里程计基本一样，只是roll、pitch用imu数据加权平均了一下，z做了限制
        pubLaserOdometryIncremental = nh.advertise<nav_msgs::Odometry> ("lio_sam/mapping/odometry_incremental", 1);
        // 发布激光里程计路径，rviz中表现为载体的运行轨迹
        pubPath                     = nh.advertise<nav_msgs::Path>("lio_sam/mapping/path", 1);

        // 订阅当前激光帧点云信息，来自featureExtraction
        subCloud = nh.subscribe<lio_sam::cloud_info>("lio_sam/feature/cloud_info", 1, &mapOptimization::laserCloudInfoHandler, this, ros::TransportHints().tcpNoDelay());
        // 订阅GPS里程计
        subGPS   = nh.subscribe<nav_msgs::Odometry> (gpsTopic, 200, &mapOptimization::gpsHandler, this, ros::TransportHints().tcpNoDelay());
        // 订阅来自外部闭环检测程序提供的闭环数据，本程序没有提供，这里实际没用上
        subLoop  = nh.subscribe<std_msgs::Float64MultiArray>("lio_loop/loop_closure_detection", 1, &mapOptimization::loopInfoHandler, this, ros::TransportHints().tcpNoDelay());
        // F2：订阅linefit功能包发布的地面点云
        // subground = nh.subscribe<sensor_msgs::PointCloud2>("ground_segmentation/ground_cloud", 1, &mapOptimization::groundCloudInfoHandler, this, ros::TransportHints().tcpNoDelay());
        // 这里的回调函数的返回值必须是void类型，不能是bool
        subground = nh.subscribe<sensor_msgs::PointCloud2>("ground_segmentation/ground_cloud", 1, &mapOptimization::groundCloudInfoHandler, this, ros::TransportHints().tcpNoDelay());


        // 发布地图保存服务
        srvSaveMap  = nh.advertiseService("lio_sam/save_map", &mapOptimization::saveMapService, this);

        // 发布闭环匹配关键帧局部map
        pubHistoryKeyFrames   = nh.advertise<sensor_msgs::PointCloud2>("lio_sam/mapping/icp_loop_closure_history_cloud", 1);
        // 发布当前关键帧经过闭环优化后的位姿变换之后的特征点云
        pubIcpKeyFrames       = nh.advertise<sensor_msgs::PointCloud2>("lio_sam/mapping/icp_loop_closure_corrected_cloud", 1);
        // 发布闭环边，rviz中表现为闭环帧之间的连线
        pubLoopConstraintEdge = nh.advertise<visualization_msgs::MarkerArray>("/lio_sam/mapping/loop_closure_constraints", 1);

        // 发布局部map的降采样平面点集合
        pubRecentKeyFrames    = nh.advertise<sensor_msgs::PointCloud2>("lio_sam/mapping/map_local", 1);
        // 发布历史帧（累加的）的角点、平面点降采样集合
        pubRecentKeyFrame     = nh.advertise<sensor_msgs::PointCloud2>("lio_sam/mapping/cloud_registered", 1);
        // 发布当前帧原始点云配准之后的点云
        pubCloudRegisteredRaw = nh.advertise<sensor_msgs::PointCloud2>("lio_sam/mapping/cloud_registered_raw", 1);

        // 四个降采样滤波器
        downSizeFilterCorner.setLeafSize(mappingCornerLeafSize, mappingCornerLeafSize, mappingCornerLeafSize);
        downSizeFilterSurf.setLeafSize(mappingSurfLeafSize, mappingSurfLeafSize, mappingSurfLeafSize);
        downSizeFilterICP.setLeafSize(mappingSurfLeafSize, mappingSurfLeafSize, mappingSurfLeafSize);
        downSizeFilterSurroundingKeyPoses.setLeafSize(surroundingKeyframeDensity, surroundingKeyframeDensity, surroundingKeyframeDensity); // for surrounding key poses of scan-to-map optimization

        allocateMemory();//初始化分配内存
    }

    /**
     * 初始化
    */
    void allocateMemory()
    {
        // F
        // typedef pcl::PointXYZI PointType;
        // linefit发布的地面点云中不包含强度字段
        GroundCloudIn.reset(new pcl::PointCloud<PointGroundType>()); // 分配内存空间

        cloudKeyPoses3D.reset(new pcl::PointCloud<PointType>());
        cloudKeyPoses6D.reset(new pcl::PointCloud<PointTypePose>());
        copy_cloudKeyPoses3D.reset(new pcl::PointCloud<PointType>());
        copy_cloudKeyPoses6D.reset(new pcl::PointCloud<PointTypePose>());

        // 用于构建kd-tree并执行最近邻搜索，kd-tree用于加速各种空间查询任务
        kdtreeSurroundingKeyPoses.reset(new pcl::KdTreeFLANN<PointType>());
        kdtreeHistoryKeyPoses.reset(new pcl::KdTreeFLANN<PointType>());

        laserCloudCornerLast.reset(new pcl::PointCloud<PointType>()); // corner feature set from odoOptimization
        laserCloudSurfLast.reset(new pcl::PointCloud<PointType>()); // surf feature set from odoOptimization
        laserCloudCornerLastDS.reset(new pcl::PointCloud<PointType>()); // downsampled corner featuer set from odoOptimization
        laserCloudSurfLastDS.reset(new pcl::PointCloud<PointType>()); // downsampled surf featuer set from odoOptimization

        laserCloudOri.reset(new pcl::PointCloud<PointType>());
        coeffSel.reset(new pcl::PointCloud<PointType>());

        laserCloudOriCornerVec.resize(N_SCAN * Horizon_SCAN);
        coeffSelCornerVec.resize(N_SCAN * Horizon_SCAN);
        laserCloudOriCornerFlag.resize(N_SCAN * Horizon_SCAN);
        laserCloudOriSurfVec.resize(N_SCAN * Horizon_SCAN);
        coeffSelSurfVec.resize(N_SCAN * Horizon_SCAN);
        laserCloudOriSurfFlag.resize(N_SCAN * Horizon_SCAN);

        std::fill(laserCloudOriCornerFlag.begin(), laserCloudOriCornerFlag.end(), false);
        std::fill(laserCloudOriSurfFlag.begin(), laserCloudOriSurfFlag.end(), false);

        laserCloudCornerFromMap.reset(new pcl::PointCloud<PointType>());
        laserCloudSurfFromMap.reset(new pcl::PointCloud<PointType>());
        laserCloudCornerFromMapDS.reset(new pcl::PointCloud<PointType>());
        laserCloudSurfFromMapDS.reset(new pcl::PointCloud<PointType>());

        kdtreeCornerFromMap.reset(new pcl::KdTreeFLANN<PointType>());
        kdtreeSurfFromMap.reset(new pcl::KdTreeFLANN<PointType>());

        // 初始化 旋转和平移均为0。初始化位姿数组
        for (int i = 0; i < 6; ++i){
            transformTobeMapped[i] = 0;//存储当前激光帧位姿的数组
        }

        // 使用opencv库创建一个6x6的浮点型矩阵，并初始化所有元素为0
        // CV_32F表示矩阵的元素类型为单精度浮点数
        matP = cv::Mat(6, 6, CV_32F, cv::Scalar::all(0));
    }

    /**
     * 订阅当前激光帧点云信息，来自featureExtraction
     * 1、当前帧位姿初始化
     *   1) 如果是第一帧，用原始imu数据的RPY初始化当前帧位姿（旋转部分）
     *   2) 后续帧，用imu里程计计算两帧之间的增量位姿变换，作用于前一帧的激光位姿，得到当前帧激光位姿
     * 2、提取局部角点、平面点云集合，加入局部map
     *   1) 对最近的一帧关键帧，搜索时空维度上相邻的关键帧集合，降采样一下
     *   2) 对关键帧集合中的每一帧，提取对应的角点、平面点，加入局部map中
     * 3、当前激光帧角点、平面点集合降采样
     * 4、scan-to-map优化当前帧位姿
     *   (1) 要求当前帧特征点数量足够多，且匹配的点数够多，才执行优化
     *   (2) 迭代30次（上限）优化
     *      1) 当前激光帧角点寻找局部map匹配点
     *          a.更新当前帧位姿，将当前帧角点坐标变换到map系下，在局部map中查找5个最近点，距离小于1m，且5个点构成直线（用距离中心点的协方差矩阵，特征值进行判断），则认为匹配上了
     *          b.计算当前帧角点到直线的距离、垂线的单位向量，存储为角点参数
     *      2) 当前激光帧平面点寻找局部map匹配点
     *          a.更新当前帧位姿，将当前帧平面点坐标变换到map系下，在局部map中查找5个最近点，距离小于1m，且5个点构成平面（最小二乘拟合平面），则认为匹配上了
     *          b.计算当前帧平面点到平面的距离、垂线的单位向量，存储为平面点参数
     *      3) 提取当前帧中与局部map匹配上了的角点、平面点，加入同一集合
     *      4) 对匹配特征点计算Jacobian矩阵，观测值为特征点到直线、平面的距离，构建高斯牛顿方程，迭代优化当前位姿，存transformTobeMapped
     *   (3)用imu原始RPY数据与scan-to-map优化后的位姿进行加权融合，更新当前帧位姿的roll、pitch，约束z坐标
     * 5、设置当前帧为关键帧并执行因子图优化
     *   1) 计算当前帧与前一帧位姿变换，如果变化太小，不设为关键帧，反之设为关键帧
     *   2) 添加激光里程计因子、GPS因子、闭环因子
     *   3) 执行因子图优化
     *   4) 得到当前帧优化后位姿，位姿协方差
     *   5) 添加cloudKeyPoses3D，cloudKeyPoses6D，更新transformTobeMapped，添加当前关键帧的角点、平面点集合
     * 6、更新因子图中所有变量节点的位姿，也就是所有历史关键帧的位姿，更新里程计轨迹
     * 7、发布激光里程计
     * 8、发布里程计、点云、轨迹
    */
    // 形参为cloud_info的点云消息类型，订阅的是feature发布的经过特征提取的点云信息（包含点云数据）
    void laserCloudInfoHandler(const lio_sam::cloud_infoConstPtr& msgIn)
    {
        // 当前激光帧时间戳
        timeLaserInfoStamp = msgIn->header.stamp; //当前激光帧的时间戳（ros单位）
        timeLaserInfoCur = msgIn->header.stamp.toSec(); // 将时间戳从ros内部表示转换为以秒为单位的浮点数

        // 提取当前激光帧的角点、面点集合
        cloudInfo = *msgIn;//解引用，提取指针指向的点云信息
        //pcl::fromROSMsg pcl库中的函数，用于将ros消息转换为PCL点云对象

        // 从订阅的点云信息中提取当前激光帧的角点集合和面点集合
        pcl::fromROSMsg(msgIn->cloud_corner,  *laserCloudCornerLast);
        pcl::fromROSMsg(msgIn->cloud_surface, *laserCloudSurfLast);

        /**
        * 一共三个线程使用到这把锁
        * 1. 雷达里程计线程，也就是当前线程
        * 2. 发布全局地图线程，执行关键帧点云拷贝转换操作
        * 3. 回环检测线程，执行关键帧姿态拷贝操作
        */
        std::lock_guard<std::mutex> lock(mtx);//全局锁

        // mapping执行频率控制
        // 记录上一帧的时间戳，两帧之间时间间隔大于mappingProcessInterval（0.15）才会进行处理
        static double timeLastProcessing = -1;
        // 这里不是对每一帧lidar点云都进行处理，而是控制lidar里程计的频率
        // timeLaserInfoCur为当前lidar点云的时间戳
        if (timeLaserInfoCur - timeLastProcessing >= mappingProcessInterval)
        {
            timeLastProcessing = timeLaserInfoCur;//上一帧激光帧时间戳

            // 当前帧位姿初始化（位姿初始估计）
            // 1、如果是第一帧，用原始imu数据的RPY初始化当前帧位姿（旋转部分）
            // 2、后续帧，用imu里程计计算两帧之间的增量位姿变换，作用于前一帧的激光位姿，得到当前帧激光位姿
            updateInitialGuess();

            // 提取局部角点、平面点云集合，加入局部map
            // 1、对最近的一帧关键帧，搜索时空维度上相邻的关键帧集合，降采样一下
            // 2、对关键帧集合中的每一帧，提取对应的角点、平面点，加入局部map中
            extractSurroundingKeyFrames();//构建局部地图

            // 当前激光帧角点、平面点集合降采样
            downsampleCurrentScan();//对当前帧点云做降采样

            // scan-to-map优化当前帧位姿（位姿优化）
            // 1、要求当前帧特征点数量足够多，且匹配的点数够多，才执行优化
            // 2、迭代30次（上限）优化 （30次为上限，当满足收敛条件时即可退出迭代优化）
            //    1) 当前激光帧角点寻找局部map匹配点
            //       a.更新当前帧位姿，将当前帧角点坐标变换到map系下，在局部map中查找5个最近点，距离小于1m，且5个点构成直线（用距离中心点的协方差矩阵，特征值进行判断），则认为匹配上了
            //       b.计算当前帧角点到直线的距离、垂线的单位向量，存储为角点参数
            //    2) 当前激光帧平面点寻找局部map匹配点
            //       a.更新当前帧位姿，将当前帧平面点坐标变换到map系下，在局部map中查找5个最近点，距离小于1m，且5个点构成平面（最小二乘拟合平面），则认为匹配上了
            //       b.计算当前帧平面点到平面的距离、垂线的单位向量，存储为平面点参数
            //    3) 提取当前帧中与局部map匹配上了的角点、平面点，加入同一集合
            //    4) 对匹配特征点计算Jacobian矩阵，观测值为特征点到直线、平面的距离，构建高斯牛顿方程，迭代优化当前位姿，存transformTobeMapped
            // 3、用imu原始RPY数据与scan-to-map优化后的位姿进行加权融合，更新当前帧位姿的roll、pitch，约束z坐标
            scan2MapOptimization();//最终得到当前帧的位姿

            // 设置当前帧为关键帧并执行因子图优化
            // 1、计算当前帧与前一帧位姿变换，如果变化太小，不设为关键帧，反之设为关键帧
            // 2、添加激光里程计因子、GPS因子、闭环因子
            // 3、执行因子图优化
            // 4、得到当前帧优化后位姿，位姿协方差
            // 5、添加cloudKeyPoses3D，cloudKeyPoses6D，更新transformTobeMapped，添加当前关键帧的角点、平面点集合
            saveKeyFramesAndFactor();

            // 更新因子图中所有变量节点的位姿，也就是所有历史关键帧的位姿，更新里程计轨迹
            correctPoses();

            // 发布激光里程计
            // 这里发布了两个里程计数据分别是优化后的和没有经过优化的，以及一个odom to lidar的tf变换
            publishOdometry();

            // 发布里程计、点云、轨迹
            // 1、发布历史关键帧位姿集合
            // 2、发布局部map的降采样平面点集合
            // 3、发布历史帧（累加的）的角点、平面点降采样集合
            // 4、发布里程计轨迹
            publishFrames();
        }
    }

    /**
     * 订阅GPS里程计，添加到GPS里程计队列
    */
    void gpsHandler(const nav_msgs::Odometry::ConstPtr& gpsMsg)
    {
        gpsQueue.push_back(*gpsMsg);
    }


    /**
     * F3:订阅linefit发布的地面点云，对地面点云进行ransac拟合地面，获取平面参数
    */
    
    void groundCloudInfoHandler(const sensor_msgs::PointCloud2ConstPtr& groundCloudMsg)
    {
    	ROS_INFO("------1------");
    	if (!cachegroundCloud(groundCloudMsg))
            return;
    	
    
    }
    
    
    
    
    bool cachegroundCloud(const sensor_msgs::PointCloud2ConstPtr& groundCloudMsg) 
    {
        // ROS消息转换为PCL对象  GroundCloudIn
        
        ROS_INFO("------2------");
        groundcloudQueue.push_back(*groundCloudMsg);
        if (groundcloudQueue.size() <= 2)
            return false;

        currentGroundCloudMsg = std::move(groundcloudQueue.front());
        groundcloudQueue.pop_front();
        
        

        if (sensor == SensorType::VELODYNE)
        {
            // 转换成pcl点云格式
            pcl::moveFromROSMsg(currentGroundCloudMsg, *GroundCloudIn);
            ROS_INFO("------3------");
        }
        /*
        else if (sensor == SensorType::OUSTER)
        {
            // 转换成Velodyne格式
            pcl::moveFromROSMsg(currentGroundCloudMsg, *tmpOusterCloudIn);
            laserCloudIn->points.resize(tmpOusterCloudIn->size());
            laserCloudIn->is_dense = tmpOusterCloudIn->is_dense;
            for (size_t i = 0; i < tmpOusterCloudIn->size(); i++)
            {
                auto &src = tmpOusterCloudIn->points[i];
                auto &dst = laserCloudIn->points[i];
                dst.x = src.x;
                dst.y = src.y;
                dst.z = src.z;
                dst.intensity = src.intensity;
                dst.ring = src.ring;
                dst.time = src.t * 1e-9f;
            }
        }
        */
        else
        {
            ROS_ERROR_STREAM("Unknown sensor type: " << int(sensor));
            ros::shutdown();
        }

        // pcl::PointCloud<PointType>::Ptr filtered(new pcl::PointCloud<PointType>());

        // 用于存储地面参数
        Eigen::VectorXf GroundCoeffs;
        // ransac拟合平面得到平面参数
        if (!getGroundCoeff(GroundCloudIn,GroundCoeffs)){
            ROS_INFO("------5:getGroundCoeff failure------");
            return false;
        }   
        else{
            ROS_INFO("------6:getGroundCoeff success------");
        	std::cout << "GroundCoeffs前四个分量: ";
    		for (int i = 0; i < 4; ++i) {
        		std::cout << GroundCoeffs[i] << " ";
    		}
    		std::cout << std::endl;
        
        }

    	return true;

    }

    /**
     * F4:ransac拟合地面，提取地面参数
    */
    bool getGroundCoeff(const pcl::PointCloud<PointGroundType>::Ptr cloud_in,Eigen::VectorXf &coeffs) const
    {
        // floor_pts_thresh:构成地面的最小点数阈值（需要在utility.h和params.yaml中增加该参数）
        // floor_pts_thresh=50
        ROS_INFO("------4------");
        if (cloud_in->size() < floor_pts_thresh)
            return false;

        pcl::SampleConsensusModelPlane<PointGroundType>::Ptr model_p(new pcl::SampleConsensusModelPlane<PointGroundType>(cloud_in));
        pcl::RandomSampleConsensus<PointGroundType> ransac(model_p);
        ransac.setDistanceThreshold(0.1);
        ransac.computeModel();
        pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
        ransac.getInliers(inliers->indices);

        // 内点太少
        if (inliers->indices.size() < floor_pts_thresh)
        {
            return false;
        }

        Eigen::Matrix4f tilt_matrix = Eigen::Matrix4f::Identity();
        Eigen::Vector4f reference = tilt_matrix.inverse() * Eigen::Vector4f::UnitZ();
        // 通过RANSAC算法获取拟合模型的参数，这里的参数是通过迭代后得到的最佳参数
        ransac.getModelCoefficients(coeffs);
        double dot = coeffs.head<3>().dot(reference.head<3>());

        // 如果点积的绝对值小于这个阈值，说明法向量与参考向量的夹角过大 
        // floor_normal_thresh=10
        if (std::abs(dot) < std::cos(floor_normal_thresh * M_PI / 180.0))
        {
            // the normal is not vertical
            return false;
        }

        // 如果点积小于 0，表示法向量与 z 轴的方向相反
        // 确保法向量指向空间的上方
        if (coeffs.head<3>().dot(Eigen::Vector3f::UnitZ()) < 0.0f)
        {
            coeffs *= -1.0f;
        }

        return true;

    }


    /**
     * 激光坐标系下的激光点，通过激光帧位姿，变换到世界坐标系下
    */
    void pointAssociateToMap(PointType const * const pi, PointType * const po)
    {
        po->x = transPointAssociateToMap(0,0) * pi->x + transPointAssociateToMap(0,1) * pi->y + transPointAssociateToMap(0,2) * pi->z + transPointAssociateToMap(0,3);
        po->y = transPointAssociateToMap(1,0) * pi->x + transPointAssociateToMap(1,1) * pi->y + transPointAssociateToMap(1,2) * pi->z + transPointAssociateToMap(1,3);
        po->z = transPointAssociateToMap(2,0) * pi->x + transPointAssociateToMap(2,1) * pi->y + transPointAssociateToMap(2,2) * pi->z + transPointAssociateToMap(2,3);
        po->intensity = pi->intensity;
    }

    /**
     * 对点云cloudIn进行变换transformIn，返回结果点云
    */
    pcl::PointCloud<PointType>::Ptr transformPointCloud(pcl::PointCloud<PointType>::Ptr cloudIn, PointTypePose* transformIn)
    {
        pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());

        int cloudSize = cloudIn->size();
        cloudOut->resize(cloudSize);

        Eigen::Affine3f transCur = pcl::getTransformation(transformIn->x, transformIn->y, transformIn->z, transformIn->roll, transformIn->pitch, transformIn->yaw);
        
        #pragma omp parallel for num_threads(numberOfCores)
        for (int i = 0; i < cloudSize; ++i)
        {
            const auto &pointFrom = cloudIn->points[i];
            cloudOut->points[i].x = transCur(0,0) * pointFrom.x + transCur(0,1) * pointFrom.y + transCur(0,2) * pointFrom.z + transCur(0,3);
            cloudOut->points[i].y = transCur(1,0) * pointFrom.x + transCur(1,1) * pointFrom.y + transCur(1,2) * pointFrom.z + transCur(1,3);
            cloudOut->points[i].z = transCur(2,0) * pointFrom.x + transCur(2,1) * pointFrom.y + transCur(2,2) * pointFrom.z + transCur(2,3);
            cloudOut->points[i].intensity = pointFrom.intensity;
        }
        return cloudOut;
    }

    /**
     * 位姿格式变换
    */
    gtsam::Pose3 pclPointTogtsamPose3(PointTypePose thisPoint)
    {
        return gtsam::Pose3(gtsam::Rot3::RzRyRx(double(thisPoint.roll), double(thisPoint.pitch), double(thisPoint.yaw)),
                                  gtsam::Point3(double(thisPoint.x),    double(thisPoint.y),     double(thisPoint.z)));
    }

    /**
     * 位姿格式变换
    */
    gtsam::Pose3 trans2gtsamPose(float transformIn[])
    {
        return gtsam::Pose3(gtsam::Rot3::RzRyRx(transformIn[0], transformIn[1], transformIn[2]), 
                                  gtsam::Point3(transformIn[3], transformIn[4], transformIn[5]));
    }

    /**
     * Eigen格式的位姿变换
    */
    Eigen::Affine3f pclPointToAffine3f(PointTypePose thisPoint)
    { 
        return pcl::getTransformation(thisPoint.x, thisPoint.y, thisPoint.z, thisPoint.roll, thisPoint.pitch, thisPoint.yaw);
    }

    /**
     * Eigen格式的位姿变换
    */
    Eigen::Affine3f trans2Affine3f(float transformIn[])
    {
        return pcl::getTransformation(transformIn[3], transformIn[4], transformIn[5], transformIn[0], transformIn[1], transformIn[2]);
    }

    /**
     * 位姿格式变换
    */
    PointTypePose trans2PointTypePose(float transformIn[])
    {
        PointTypePose thisPose6D;
        thisPose6D.x = transformIn[3];
        thisPose6D.y = transformIn[4];
        thisPose6D.z = transformIn[5];
        thisPose6D.roll  = transformIn[0];
        thisPose6D.pitch = transformIn[1];
        thisPose6D.yaw   = transformIn[2];
        return thisPose6D;
    }

    












    /**
     * 保存全局关键帧特征点集合
    */
    bool saveMapService(lio_sam::save_mapRequest& req, lio_sam::save_mapResponse& res)
    {
      string saveMapDirectory;

      cout << "****************************************************" << endl;
      cout << "Saving map to pcd files ..." << endl;
      if(req.destination.empty()) saveMapDirectory = std::getenv("HOME") + savePCDDirectory;
      else saveMapDirectory = std::getenv("HOME") + req.destination;
      cout << "Save destination: " << saveMapDirectory << endl;
      // 这个代码太坑了！！注释掉
      // int unused = system((std::string("exec rm -r ") + saveMapDirectory).c_str());
      // unused = system((std::string("mkdir -p ") + saveMapDirectory).c_str());
      // 保存历史关键帧位姿
      pcl::io::savePCDFileBinary(saveMapDirectory + "/trajectory.pcd", *cloudKeyPoses3D);
      pcl::io::savePCDFileBinary(saveMapDirectory + "/transformations.pcd", *cloudKeyPoses6D);
      // 提取历史关键帧角点、平面点集合
      pcl::PointCloud<PointType>::Ptr globalCornerCloud(new pcl::PointCloud<PointType>());
      pcl::PointCloud<PointType>::Ptr globalCornerCloudDS(new pcl::PointCloud<PointType>());
      pcl::PointCloud<PointType>::Ptr globalSurfCloud(new pcl::PointCloud<PointType>());
      pcl::PointCloud<PointType>::Ptr globalSurfCloudDS(new pcl::PointCloud<PointType>());
      pcl::PointCloud<PointType>::Ptr globalMapCloud(new pcl::PointCloud<PointType>());
      for (int i = 0; i < (int)cloudKeyPoses3D->size(); i++) {
          *globalCornerCloud += *transformPointCloud(cornerCloudKeyFrames[i],  &cloudKeyPoses6D->points[i]);
          *globalSurfCloud   += *transformPointCloud(surfCloudKeyFrames[i],    &cloudKeyPoses6D->points[i]);
          cout << "\r" << std::flush << "Processing feature cloud " << i << " of " << cloudKeyPoses6D->size() << " ...";
      }

      if(req.resolution != 0)
      {
        cout << "\n\nSave resolution: " << req.resolution << endl;

        // 降采样
        downSizeFilterCorner.setInputCloud(globalCornerCloud);
        downSizeFilterCorner.setLeafSize(req.resolution, req.resolution, req.resolution);
        downSizeFilterCorner.filter(*globalCornerCloudDS);
        pcl::io::savePCDFileBinary(saveMapDirectory + "/CornerMap.pcd", *globalCornerCloudDS);
        // 降采样
        downSizeFilterSurf.setInputCloud(globalSurfCloud);
        downSizeFilterSurf.setLeafSize(req.resolution, req.resolution, req.resolution);
        downSizeFilterSurf.filter(*globalSurfCloudDS);
        pcl::io::savePCDFileBinary(saveMapDirectory + "/SurfMap.pcd", *globalSurfCloudDS);
      }
      else
      {
        pcl::io::savePCDFileBinary(saveMapDirectory + "/CornerMap.pcd", *globalCornerCloud);
        pcl::io::savePCDFileBinary(saveMapDirectory + "/SurfMap.pcd", *globalSurfCloud);
      }

      // 保存到一起，全局关键帧特征点集合
      *globalMapCloud += *globalCornerCloud;
      *globalMapCloud += *globalSurfCloud;

      int ret = pcl::io::savePCDFileBinary(saveMapDirectory + "/GlobalMap.pcd", *globalMapCloud);
      res.success = ret == 0;

      downSizeFilterCorner.setLeafSize(mappingCornerLeafSize, mappingCornerLeafSize, mappingCornerLeafSize);
      downSizeFilterSurf.setLeafSize(mappingSurfLeafSize, mappingSurfLeafSize, mappingSurfLeafSize);

      cout << "****************************************************" << endl;
      cout << "Saving map to pcd files completed\n" << endl;

      return true;
    }

    /**
     * 展示线程
     * 1、发布局部关键帧map的特征点云
     * 2、保存全局关键帧特征点集合
    */
    void visualizeGlobalMapThread()
    {
        ros::Rate rate(0.2);
        while (ros::ok()){
            rate.sleep();
            // 发布局部关键帧map的特征点云
            publishGlobalMap();
        }

        if (savePCD == false)
            return;

        lio_sam::save_mapRequest  req;
        lio_sam::save_mapResponse res;

        // 保存全局关键帧特征点集合
        if(!saveMapService(req, res)){
            cout << "Fail to save map" << endl;
        }
    }

    /**
     * 发布局部关键帧map的特征点云
     * 全局地图可视化线程
    */
    void publishGlobalMap()
    {
        if (pubLaserCloudSurround.getNumSubscribers() == 0)
            return;

        if (cloudKeyPoses3D->points.empty() == true)
            return;

        pcl::KdTreeFLANN<PointType>::Ptr kdtreeGlobalMap(new pcl::KdTreeFLANN<PointType>());;
        pcl::PointCloud<PointType>::Ptr globalMapKeyPoses(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr globalMapKeyPosesDS(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr globalMapKeyFrames(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr globalMapKeyFramesDS(new pcl::PointCloud<PointType>());

        // kdtree查找最近一帧关键帧相邻的关键帧集合
        std::vector<int> pointSearchIndGlobalMap;
        std::vector<float> pointSearchSqDisGlobalMap;
        mtx.lock();
        kdtreeGlobalMap->setInputCloud(cloudKeyPoses3D);
        kdtreeGlobalMap->radiusSearch(cloudKeyPoses3D->back(), globalMapVisualizationSearchRadius, pointSearchIndGlobalMap, pointSearchSqDisGlobalMap, 0);
        mtx.unlock();

        for (int i = 0; i < (int)pointSearchIndGlobalMap.size(); ++i)
            globalMapKeyPoses->push_back(cloudKeyPoses3D->points[pointSearchIndGlobalMap[i]]);
        // 降采样
        pcl::VoxelGrid<PointType> downSizeFilterGlobalMapKeyPoses;
        downSizeFilterGlobalMapKeyPoses.setLeafSize(globalMapVisualizationPoseDensity, globalMapVisualizationPoseDensity, globalMapVisualizationPoseDensity); // for global map visualization
        downSizeFilterGlobalMapKeyPoses.setInputCloud(globalMapKeyPoses);
        downSizeFilterGlobalMapKeyPoses.filter(*globalMapKeyPosesDS);

        // 提取局部相邻关键帧对应的特征点云
        for (int i = 0; i < (int)globalMapKeyPosesDS->size(); ++i){
            // 距离过大
            if (pointDistance(globalMapKeyPosesDS->points[i], cloudKeyPoses3D->back()) > globalMapVisualizationSearchRadius)
                continue;
            int thisKeyInd = (int)globalMapKeyPosesDS->points[i].intensity;
            // 将点云从lidar系变换到全局系下
            *globalMapKeyFrames += *transformPointCloud(cornerCloudKeyFrames[thisKeyInd],  &cloudKeyPoses6D->points[thisKeyInd]);
            *globalMapKeyFrames += *transformPointCloud(surfCloudKeyFrames[thisKeyInd],    &cloudKeyPoses6D->points[thisKeyInd]);
        }
        // 降采样，发布
        pcl::VoxelGrid<PointType> downSizeFilterGlobalMapKeyFrames; // for global map visualization
        downSizeFilterGlobalMapKeyFrames.setLeafSize(globalMapVisualizationLeafSize, globalMapVisualizationLeafSize, globalMapVisualizationLeafSize); // for global map visualization
        downSizeFilterGlobalMapKeyFrames.setInputCloud(globalMapKeyFrames);
        downSizeFilterGlobalMapKeyFrames.filter(*globalMapKeyFramesDS);
        publishCloud(pubLaserCloudSurround, globalMapKeyFramesDS, timeLaserInfoStamp, odometryFrame);
    }











    /**
     * 回环检测独立线程
    * 1. 由于回环检测中用到了点云匹配，较为耗时，所以独立为单独的线程运行
    * 2. 新的回环关系被检测出来时被主线程加入因子图中优化
    * 
     * 1、闭环scan-to-map，icp优化位姿
     *   1) 在历史关键帧中查找与当前关键帧距离最近的关键帧集合，选择时间相隔较远的一帧作为候选闭环帧
     *   2) 提取当前关键帧特征点集合，降采样；提取闭环匹配关键帧前后相邻若干帧的关键帧特征点集合，降采样
     *   3) 执行scan-to-map优化，调用icp方法，得到优化后位姿，构造闭环因子需要的数据，在因子图优化中一并加入更新位姿
     * 2、rviz展示闭环边
    */
    void loopClosureThread()
    {
        // 回环检测线程的执行前提：回环检测标志为true
        if (loopClosureEnableFlag == false)
            return;

        // 创建了一个ros::Rate对象，该对象的构造函数接受一个参数 loopClosureFrequency，该参数指定了循环频率，即每秒执行的次数
        ros::Rate rate(loopClosureFrequency);

        // ros::ok()函数：检查ROS节点是否仍在运行，只要ROS节点处于运行状态就返回true，执行循环
        while (ros::ok())
        {
            rate.sleep();// 等待一段时间，用来控制循环的频率
            // 闭环scan-to-map，icp优化位姿
            // 1、在历史关键帧中查找与当前关键帧距离最近的关键帧集合，选择时间相隔较远的一帧作为候选闭环帧
            // 2、提取当前关键帧特征点集合，降采样；提取闭环匹配关键帧前后相邻若干帧的关键帧特征点集合，降采样
            // 3、执行scan-to-map优化，调用icp方法，得到优化后位姿，构造闭环因子需要的数据，在因子图优化中一并加入更新位姿
            // 注：闭环的时候没有立即更新当前帧的位姿，而是添加闭环因子，让图优化去更新位姿
            performLoopClosure();// 执行回环检测
            // rviz展示闭环边
            visualizeLoopClosure();// 回环检测结果可视化
        }
    }

    /**
     * 订阅来自外部闭环检测程序提供的闭环数据，本程序没有提供，这里实际没用上
    */
    void loopInfoHandler(const std_msgs::Float64MultiArray::ConstPtr& loopMsg)
    {
        std::lock_guard<std::mutex> lock(mtxLoopInfo);
        if (loopMsg->data.size() != 2)
            return;

        loopInfoVec.push_back(*loopMsg);

        while (loopInfoVec.size() > 5)//保证队列中回环数据不超过5
            loopInfoVec.pop_front();
    }

    /**
     * 闭环scan-to-map，icp优化位姿
     * 1、在历史关键帧中查找与当前关键帧距离最近的关键帧集合，选择时间相隔较远的一帧作为候选闭环帧
     * 2、提取当前关键帧特征点集合，降采样；提取闭环匹配关键帧前后相邻若干帧的关键帧特征点集合，降采样
     * 3、执行scan-to-map优化，调用icp方法，得到优化后位姿，构造闭环因子需要的数据，在因子图优化中一并加入更新位姿
     * 注：闭环的时候没有立即更新当前帧的位姿，而是添加闭环因子，让图优化去更新位姿
    */
    void performLoopClosure()
    {
        if (cloudKeyPoses3D->points.empty() == true)
            return;

        // 加锁拷贝历史关键帧的3D和6D位姿，避免多线程干扰
        mtx.lock();
        *copy_cloudKeyPoses3D = *cloudKeyPoses3D;
        *copy_cloudKeyPoses6D = *cloudKeyPoses6D;
        mtx.unlock();

        // 当前关键帧索引，候选闭环匹配帧索引
        int loopKeyCur;
        int loopKeyPre;
        // not-used
        // 先利用外部提供的闭环数据检测回环，如果为false则进行内部数据的回环检测
        if (detectLoopClosureExternal(&loopKeyCur, &loopKeyPre) == false)
            // 在历史关键帧中查找与当前关键帧距离最近的关键帧集合，选择时间相隔较远的一帧作为候选闭环帧
            if (detectLoopClosureDistance(&loopKeyCur, &loopKeyPre) == false)
                return;

        // 提取
        // 创建两个pcl指针指向点云对象，分别存放当前关键帧和闭环匹配帧的拼接点云
        pcl::PointCloud<PointType>::Ptr cureKeyframeCloud(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr prevKeyframeCloud(new pcl::PointCloud<PointType>());
        {
            // 提取当前关键帧特征点集合，降采样
            // 这里设置索引范围为0，也就是仅把当前关键帧的角点点云和面点点云拼接在一起，无周围关键帧
            loopFindNearKeyframes(cureKeyframeCloud, loopKeyCur, 0);
            // 提取闭环匹配关键帧前后相邻若干帧的关键帧特征点集合，降采样
            loopFindNearKeyframes(prevKeyframeCloud, loopKeyPre, historyKeyframeSearchNum);
            // 如果当前关键帧或匹配帧的特征点较少，则直接返回，不进行回环检测
            if (cureKeyframeCloud->size() < 300 || prevKeyframeCloud->size() < 1000)
                return;
            // 发布闭环匹配关键帧局部map
            if (pubHistoryKeyFrames.getNumSubscribers() != 0)
                publishCloud(pubHistoryKeyFrames, prevKeyframeCloud, timeLaserInfoStamp, odometryFrame);
        }

        // 使用ICP匹配当前帧点云和拼接子图
        // ICP参数设置
        // ICP算法用于点云配准，寻找两个点云之间的最佳变换以最小化两帧点云之间的距离误差
        // 创建ICP算法对象，可以使用它来设置ICP算法的各种参数以及执行匹配
        static pcl::IterativeClosestPoint<PointType, PointType> icp;
        // 设置最大对应点的距离
        icp.setMaxCorrespondenceDistance(historyKeyframeSearchRadius*2);
        // 最大迭代次数
        icp.setMaximumIterations(100);
        // 终止条件
        icp.setTransformationEpsilon(1e-6);// 变换矩阵的收敛容差，当变换矩阵的变化小于阈值时，ICP算法停止
        icp.setEuclideanFitnessEpsilon(1e-6);// 欧几里得拟合度的容差
        // 设置RANSAC迭代次数，这里不使用
        icp.setRANSACIterations(0);

        // scan-to-map，调用icp匹配
        // 设置ICP算法的源点云和目标点云
        // ICP算法对初值敏感，因此ICP要配准的原始点云和目标点云应该已经大致对齐了
        icp.setInputSource(cureKeyframeCloud);
        icp.setInputTarget(prevKeyframeCloud);
        // 创建用于存储匹配结果的点云对象
        pcl::PointCloud<PointType>::Ptr unused_result(new pcl::PointCloud<PointType>());
        // 执行ICP匹配操作，将源点云与目标点云进行匹配
        // 通过调用 align 函数，ICP算法会尝试寻找最佳的变换矩阵，以最小化源点云和目标点云之间的距离误差。
        icp.align(*unused_result);//调用align函数启动icp迭代
        // unused_result为变换后的点云，此处与closed_cloud一致

        // 未收敛，或者匹配不够好
        // hasConverged()用于检查ICP是否收敛；getFitnessScore()的值越小表示匹配得越好
        if (icp.hasConverged() == false || icp.getFitnessScore() > historyKeyframeFitnessScore)
            return;

        // 发布当前关键帧经过闭环优化后的位姿变换之后的特征点云
        // 发布ICP配准后的点云
        if (pubIcpKeyFrames.getNumSubscribers() != 0)
        {
            // 创建点云对象
            pcl::PointCloud<PointType>::Ptr closed_cloud(new pcl::PointCloud<PointType>());
            // 调用上面ICP算法匹配得到的变换矩阵，对源点云进行变换，得到了匹配后的点云closed_cloud
            pcl::transformPointCloud(*cureKeyframeCloud, *closed_cloud, icp.getFinalTransformation());
            // 用发布器pubIcpKeyFrames发布匹配后的点云
            publishCloud(pubIcpKeyFrames, closed_cloud, timeLaserInfoStamp, odometryFrame);
        }

        // 闭环优化得到的当前关键帧与闭环关键帧之间的位姿变换
        float x, y, z, roll, pitch, yaw;
        Eigen::Affine3f correctionLidarFrame;
        correctionLidarFrame = icp.getFinalTransformation();// 获取ICP匹配得到的变换矩阵
        
        // 闭环优化前当前帧位姿
        Eigen::Affine3f tWrong = pclPointToAffine3f(copy_cloudKeyPoses6D->points[loopKeyCur]);
        // 闭环优化后当前帧位姿
        Eigen::Affine3f tCorrect = correctionLidarFrame * tWrong;
        pcl::getTranslationAndEulerAngles (tCorrect, x, y, z, roll, pitch, yaw);
        gtsam::Pose3 poseFrom = Pose3(Rot3::RzRyRx(roll, pitch, yaw), Point3(x, y, z));
        // 闭环匹配帧的位姿
        gtsam::Pose3 poseTo = pclPointTogtsamPose3(copy_cloudKeyPoses6D->points[loopKeyPre]);
        gtsam::Vector Vector6(6);
        float noiseScore = icp.getFitnessScore();//获取噪声分数，越小匹配得越好
        // 设置6个维度的噪声均相同
        Vector6 << noiseScore, noiseScore, noiseScore, noiseScore, noiseScore, noiseScore;
        // 创建噪声模型对象，用于构建约束
        noiseModel::Diagonal::shared_ptr constraintNoise = noiseModel::Diagonal::Variances(Vector6);

        // 添加闭环因子需要的数据，这些数据在addLoopFactor中会使用
        // 将回环检测结果加入队列中
        mtx.lock();
        loopIndexQueue.push_back(make_pair(loopKeyCur, loopKeyPre));
        // 计算闭环优化后的当前帧的位姿到闭环匹配帧位姿之间的相对位姿变换，并加入位姿队列中
        loopPoseQueue.push_back(poseFrom.between(poseTo));
        loopNoiseQueue.push_back(constraintNoise);
        mtx.unlock();

        // 将这对闭环匹配关键帧的索引加入容器中
        loopIndexContainer[loopKeyCur] = loopKeyPre;
    }

    /**
     * 在历史关键帧中查找与当前关键帧距离最近的关键帧集合，选择时间相隔较远的一帧作为候选闭环帧
    */
    bool detectLoopClosureDistance(int *latestID, int *closestID)
    {
        // 当前关键帧和闭环匹配帧索引
        int loopKeyCur = copy_cloudKeyPoses3D->size() - 1;//最后一帧也就是最新一帧关键帧
        int loopKeyPre = -1;

        // 当前帧已经添加过闭环对应关系，不再继续添加
        // 一帧点云只能存在一个匹配的回环关系
        auto it = loopIndexContainer.find(loopKeyCur);
        if (it != loopIndexContainer.end())
            return false;

        // 在历史关键帧中查找与当前关键帧距离最近的关键帧集合
        std::vector<int> pointSearchIndLoop; //搜索到的关键帧的索引
        std::vector<float> pointSearchSqDisLoop; // 距离给定关键帧的平方距离
        // 设置kd-tree的输入为历史关键帧3D位姿集合
        kdtreeHistoryKeyPoses->setInputCloud(copy_cloudKeyPoses3D);
        // 指定搜索半径，在kd-tree中查找距离当前关键帧最近的关键帧，并存储搜索到的关键帧的索引的平方距离
        kdtreeHistoryKeyPoses->radiusSearch(copy_cloudKeyPoses3D->back(), historyKeyframeSearchRadius, pointSearchIndLoop, pointSearchSqDisLoop, 0);
        
        // 在候选关键帧集合中，找到与当前帧时间相隔较远的帧，设为候选匹配帧
        for (int i = 0; i < (int)pointSearchIndLoop.size(); ++i)
        {
            int id = pointSearchIndLoop[i];
            // timeLaserInfoCur为当前激光帧的时间戳
            // 确保搜索到的候选关键帧是较久之前采集到的
            if (abs(copy_cloudKeyPoses6D->points[id].time - timeLaserInfoCur) > historyKeyframeSearchTimeDiff)
            {
                loopKeyPre = id;
                break; // 一旦找到时间上满足要求的候选关键帧则立即退出循环，不再遍历其余候选帧
            }
        }

        // 没有找到当前帧的闭环匹配帧
        if (loopKeyPre == -1 || loopKeyCur == loopKeyPre)
            return false;

        *latestID = loopKeyCur;
        *closestID = loopKeyPre;

        return true;
    }

    /**
     * not-used, 来自外部闭环检测程序提供的闭环匹配索引对
     * * 参数：当前关键帧索引，候选闭环匹配帧索引
    */
    bool detectLoopClosureExternal(int *latestID, int *closestID)
    {
        // this function is not used yet, please ignore it
        int loopKeyCur = -1; // 当前关键帧索引
        int loopKeyPre = -1; // 回环候选帧索引

        // mtxLoopInfo：回环检测互斥锁
        std::lock_guard<std::mutex> lock(mtxLoopInfo);
        if (loopInfoVec.empty())// 如果回环检测队列为空，该队列存放外部提供的回环数据
            return false;

        double loopTimeCur = loopInfoVec.front().data[0];
        double loopTimePre = loopInfoVec.front().data[1];
        loopInfoVec.pop_front();

        // 当前帧与回环候选帧在时间上不能过近
        if (abs(loopTimeCur - loopTimePre) < historyKeyframeSearchTimeDiff)
            return false;

        int cloudSize = copy_cloudKeyPoses6D->size();
        if (cloudSize < 2) // 历史关键帧过少
            return false;

        // latest key
        loopKeyCur = cloudSize - 1;
        // 从历史关键帧位姿集合中找到与给定时间loopTimeCur最近的关键帧的索引
        for (int i = cloudSize - 1; i >= 0; --i)
        {
            if (copy_cloudKeyPoses6D->points[i].time >= loopTimeCur)
            // round函数确保索引值为整数
                loopKeyCur = round(copy_cloudKeyPoses6D->points[i].intensity);
            else
                break;
        }

        // previous key
        loopKeyPre = 0;
        for (int i = 0; i < cloudSize; ++i)
        {
            if (copy_cloudKeyPoses6D->points[i].time <= loopTimePre)
                loopKeyPre = round(copy_cloudKeyPoses6D->points[i].intensity);
            else
                break;
        }

        if (loopKeyCur == loopKeyPre)
            return false;

        // loopIndexContainer：存放回环检测匹配对的容器
        auto it = loopIndexContainer.find(loopKeyCur);
        if (it != loopIndexContainer.end()) // 容器中已经存在了当前关键帧对应的匹配候选帧
            return false;

        *latestID = loopKeyCur;
        *closestID = loopKeyPre;

        return true;
    }

    /**
     * 提取key索引的关键帧前后相邻若干帧的关键帧特征点集合，降采样
    */
    void loopFindNearKeyframes(pcl::PointCloud<PointType>::Ptr& nearKeyframes, const int& key, const int& searchNum)
    {
        // 提取key索引的关键帧前后相邻若干帧的关键帧特征点集合
        nearKeyframes->clear(); //用于存放拼接点云
        int cloudSize = copy_cloudKeyPoses6D->size();
        // searchNum为索引搜索范围
        for (int i = -searchNum; i <= searchNum; ++i)
        {
            int keyNear = key + i; // 在当前关键帧索引的前后searchNum范围内进行搜索
            if (keyNear < 0 || keyNear >= cloudSize )
                continue;
            // transformPointCloud：对输入点云执行指定位姿变换并返回结果点云
            // 把当前关键帧前后索引范围内的角点点云和面点点云均拼接在一起
            *nearKeyframes += *transformPointCloud(cornerCloudKeyFrames[keyNear], &copy_cloudKeyPoses6D->points[keyNear]);
            *nearKeyframes += *transformPointCloud(surfCloudKeyFrames[keyNear],   &copy_cloudKeyPoses6D->points[keyNear]);
        }

        if (nearKeyframes->empty())
            return;

        // 降采样
        pcl::PointCloud<PointType>::Ptr cloud_temp(new pcl::PointCloud<PointType>());
        downSizeFilterICP.setInputCloud(nearKeyframes);
        downSizeFilterICP.filter(*cloud_temp);
        *nearKeyframes = *cloud_temp;
    }

    /**
     * rviz展示闭环边
    */
    void visualizeLoopClosure()
    {
        if (loopIndexContainer.empty())
            return;
        
        // 创建数组用于存储可视化元素
        visualization_msgs::MarkerArray markerArray;
        // 创建了一个Marker对象，表示闭环顶点
        visualization_msgs::Marker markerNode;
        markerNode.header.frame_id = odometryFrame;
        markerNode.header.stamp = timeLaserInfoStamp;//设置为当前激光数据的时间戳
        // 指定Marker的动作，这里设置为添加（ADD），表示将要添加新的Marker。
        markerNode.action = visualization_msgs::Marker::ADD;
        // 设置Marker的类型为球体列表，表示 markerNode 将显示为一系列球体。
        markerNode.type = visualization_msgs::Marker::SPHERE_LIST;
        // 设置Marker的名称空间
        markerNode.ns = "loop_nodes";
        // 为Marker分配一个唯一的ID
        markerNode.id = 0;
        // 设置Marker的姿态，这里将其姿态设置为无旋转，w 分量为1，表示单位四元数。
        markerNode.pose.orientation.w = 1;
        // 设置Marker的尺寸，这里将其尺寸设置为x、y、z方向都为0.3.
        markerNode.scale.x = 0.3; markerNode.scale.y = 0.3; markerNode.scale.z = 0.3; 
        // 设置颜色
        markerNode.color.r = 0; markerNode.color.g = 0.8; markerNode.color.b = 1;
        // 设置Marker的不透明度，这里将其不透明度设置为1，表示完全不透明。
        markerNode.color.a = 1;

        // 闭环边
        visualization_msgs::Marker markerEdge;
        markerEdge.header.frame_id = odometryFrame;
        markerEdge.header.stamp = timeLaserInfoStamp;
        markerEdge.action = visualization_msgs::Marker::ADD;
        markerEdge.type = visualization_msgs::Marker::LINE_LIST;
        markerEdge.ns = "loop_edges";
        markerEdge.id = 1;
        markerEdge.pose.orientation.w = 1;
        markerEdge.scale.x = 0.1;
        markerEdge.color.r = 0.9; markerEdge.color.g = 0.9; markerEdge.color.b = 0;
        markerEdge.color.a = 1;

        // 遍历存放闭环索引对的容器
        for (auto it = loopIndexContainer.begin(); it != loopIndexContainer.end(); ++it)
        {
            int key_cur = it->first;//当前关键帧索引
            int key_pre = it->second;//闭环匹配帧索引
            geometry_msgs::Point p;
            p.x = copy_cloudKeyPoses6D->points[key_cur].x;
            p.y = copy_cloudKeyPoses6D->points[key_cur].y;
            p.z = copy_cloudKeyPoses6D->points[key_cur].z;
            markerNode.points.push_back(p);// markerNode 表示用于可视化闭环检测结果的球体列表
            markerEdge.points.push_back(p);// markerEdge 用于表示与闭环关联的边缘或线段
            p.x = copy_cloudKeyPoses6D->points[key_pre].x;
            p.y = copy_cloudKeyPoses6D->points[key_pre].y;
            p.z = copy_cloudKeyPoses6D->points[key_pre].z;
            markerNode.points.push_back(p);
            markerEdge.points.push_back(p);
        }

        markerArray.markers.push_back(markerNode);
        markerArray.markers.push_back(markerEdge);
        pubLoopConstraintEdge.publish(markerArray); // 发布闭环边，rviz中表现为闭环帧之间的连线
    }







    


    /**
     * 当前帧位姿初始化
     * 1、如果是第一帧，用原始imu数据的RPY初始化当前帧位姿（旋转部分）
     * 2、后续帧，用imu里程计计算两帧之间的增量位姿变换，作用于前一帧的激光位姿，得到当前帧激光位姿
    */
    void updateInitialGuess()
    {
        // 前一帧的位姿，注：这里指lidar的位姿，后面都简写成位姿
        // trans2Affine3f用于将一个包含旋转和平移信息的数组转换为eigen库中的仿射变换矩阵
        incrementalOdometryAffineFront = trans2Affine3f(transformTobeMapped);//存储当前激光帧位姿的数组

        // 前一帧的初始化姿态角（来自原始imu数据），用于估计第一帧的位姿（旋转部分）
        static Eigen::Affine3f lastImuTransformation;
        
        // 如果关键帧集合为空，继续进行初始化
        // cloudKeyPoses3D：历史关键帧位姿集合（位置）
        // 初始化第一个关键帧，如果历史关键帧位置集合为空，说明当前激光帧为第一帧
        if (cloudKeyPoses3D->points.empty())
        {
            // 当前帧位姿的旋转部分，用激光帧信息中的RPY（来自imu原始数据）初始化
            // cloudInfo中为当前激光帧角点和面点集合
            transformTobeMapped[0] = cloudInfo.imuRollInit;
            transformTobeMapped[1] = cloudInfo.imuPitchInit;
            transformTobeMapped[2] = cloudInfo.imuYawInit;

            if (!useImuHeadingInitialization) // GPS的参数，偏航角未进行初始化
                transformTobeMapped[2] = 0;

            // 利用旋转和平移信息来创建一个仿射变换矩阵
            lastImuTransformation = pcl::getTransformation(0, 0, 0, cloudInfo.imuRollInit, cloudInfo.imuPitchInit, cloudInfo.imuYawInit); 
            return;
        }

        // 用当前帧和前一帧对应的imu里程计计算相对位姿变换，再用前一帧的位姿与相对变换，计算当前帧的位姿，存transformTobeMapped
        static bool lastImuPreTransAvailable = false; // 判断接收的消息中是否有设置好的预积分数据
        static Eigen::Affine3f lastImuPreTransformation;
        if (cloudInfo.odomAvailable == true) // 接收的点云信息中有imu里程计数据
        {
            // 当前帧的初始估计位姿（来自imu里程计），后面用来计算增量位姿变换
            Eigen::Affine3f transBack = pcl::getTransformation(cloudInfo.initialGuessX,    cloudInfo.initialGuessY,     cloudInfo.initialGuessZ, 
                                                               cloudInfo.initialGuessRoll, cloudInfo.initialGuessPitch, cloudInfo.initialGuessYaw);
            // 下面if部分代码的作用是将第一帧lidar帧计算得到的位姿矩阵transBack赋值给lastImuPreTransformation，作为上一帧的位姿矩阵
            if (lastImuPreTransAvailable == false)
            {
                // lastImuPreTransAvailable是一个静态变量，初始被设置为false,之后就变成了true 
                // 这段代码只调用一次，是初始时，把imu位姿赋值给lastImuPreTransformation
                // 赋值给前一帧
                lastImuPreTransformation = transBack;
                lastImuPreTransAvailable = true;
            } else {
                // 当前帧相对于前一帧的位姿变换，imu里程计计算得到
                Eigen::Affine3f transIncre = lastImuPreTransformation.inverse() * transBack;
                // 前一帧的位姿
                Eigen::Affine3f transTobe = trans2Affine3f(transformTobeMapped);
                // 当前帧的位姿
                Eigen::Affine3f transFinal = transTobe * transIncre;
                // 更新当前帧位姿transformTobeMapped
                pcl::getTranslationAndEulerAngles(transFinal, transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5], 
                                                              transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);
                // 赋值给前一帧
                lastImuPreTransformation = transBack;

                lastImuTransformation = pcl::getTransformation(0, 0, 0, cloudInfo.imuRollInit, cloudInfo.imuPitchInit, cloudInfo.imuYawInit); // save imu before return;
                return;
            }
        }

        // 只在第一帧调用（注意上面的return），用IMU的增量初始化位姿估计，仅初始化旋转部分
        // 没懂这部分？？
        if (cloudInfo.imuAvailable == true)
        {
            // 当前帧的姿态角（来自原始imu数据）
            Eigen::Affine3f transBack = pcl::getTransformation(0, 0, 0, cloudInfo.imuRollInit, cloudInfo.imuPitchInit, cloudInfo.imuYawInit);
            // 当前帧相对于前一帧的姿态变换
            Eigen::Affine3f transIncre = lastImuTransformation.inverse() * transBack;

            // 前一帧的位姿
            Eigen::Affine3f transTobe = trans2Affine3f(transformTobeMapped);
            // 当前帧的位姿
            Eigen::Affine3f transFinal = transTobe * transIncre;
            // 更新当前帧位姿transformTobeMapped
            pcl::getTranslationAndEulerAngles(transFinal, transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5], 
                                                          transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);

            // 用当前帧的transBack来更新上一帧位姿
            lastImuTransformation = pcl::getTransformation(0, 0, 0, cloudInfo.imuRollInit, cloudInfo.imuPitchInit, cloudInfo.imuYawInit); // save imu before return;
            return;
        }
    }

    /**
     * not-used
    */
    void extractForLoopClosure()
    {
        pcl::PointCloud<PointType>::Ptr cloudToExtract(new pcl::PointCloud<PointType>());
        // 
        int numPoses = cloudKeyPoses3D->size();
        for (int i = numPoses-1; i >= 0; --i)
        {
            if ((int)cloudToExtract->size() <= surroundingKeyframeSize)
                cloudToExtract->push_back(cloudKeyPoses3D->points[i]);
            else
                break;
        }

        // 将相邻关键帧集合对应的角点、平面点，加入到局部map中，作为scan-to-map匹配的局部点云地图
        extractCloud(cloudToExtract);
    }

    /**
     * 提取局部角点、平面点云集合，加入局部map
     * 1、对最近的一帧关键帧，搜索时空维度上相邻的关键帧集合，降采样一下
     * 2、对关键帧集合中的每一帧，提取对应的角点、平面点，加入局部map中
    */
    void extractNearby()
    {
        // 创建了两个指向点云对象的智能指针，均是存储3D位姿
        // 相邻关键帧集合
        pcl::PointCloud<PointType>::Ptr surroundingKeyPoses(new pcl::PointCloud<PointType>());
        // 降采样后的相邻关键帧集合
        pcl::PointCloud<PointType>::Ptr surroundingKeyPosesDS(new pcl::PointCloud<PointType>());
        // 存储kd-tree搜索结果的索引和距离
        std::vector<int> pointSearchInd;
        std::vector<float> pointSearchSqDis;

        // 设置kdtree的输入，全局关键帧位姿集合（历史所有关键帧集合的3D位姿）
        kdtreeSurroundingKeyPoses->setInputCloud(cloudKeyPoses3D); 
        // 对最近的一帧关键帧，在半径区域内搜索空间区域上相邻的关键帧集合
        // 使用kd-tree对最近的一帧关键帧进行半径搜索，该关键帧即进行搜索的中心点，(double)surroundingKeyframeSearchRadius为搜索半径
        // pointSearchInd, pointSearchSqDis 分别存储搜索到的关键帧的索引和搜索到的关键帧与中心关键帧之间的平方距离
        kdtreeSurroundingKeyPoses->radiusSearch(cloudKeyPoses3D->back(), (double)surroundingKeyframeSearchRadius, pointSearchInd, pointSearchSqDis);
        
        // 遍历搜索结果，pointSearchInd存的是结果在cloudKeyPoses3D下面的索引
        for (int i = 0; i < (int)pointSearchInd.size(); ++i)
        {
            int id = pointSearchInd[i];
            // 加入相邻关键帧位姿集合中
            surroundingKeyPoses->push_back(cloudKeyPoses3D->points[id]);
        }

        // 降采样一下
        // 相邻关键帧降采样滤波器
        downSizeFilterSurroundingKeyPoses.setInputCloud(surroundingKeyPoses);
        downSizeFilterSurroundingKeyPoses.filter(*surroundingKeyPosesDS);

        // 加入时间上相邻的一些关键帧，比如当载体在原地转圈，这些帧加进来是合理的
        int numPoses = cloudKeyPoses3D->size();
        for (int i = numPoses-1; i >= 0; --i)
        {
            // timeLaserInfoCur：当前激光帧的时间戳
            // 只有6D位姿的点云数据结构中存储了time
            if (timeLaserInfoCur - cloudKeyPoses6D->points[i].time < 10.0)
                surroundingKeyPosesDS->push_back(cloudKeyPoses3D->points[i]);
            else
                break;
        }

        // 将相邻关键帧集合对应的角点、平面点，加入到局部map中，作为scan-to-map匹配的局部点云地图
        extractCloud(surroundingKeyPosesDS);
    }

    /**
     * 将相邻关键帧集合对应的角点、平面点，加入到局部map中，作为scan-to-map匹配的局部点云地图
    */
    void extractCloud(pcl::PointCloud<PointType>::Ptr cloudToExtract)
    {
        // 相邻关键帧集合对应的角点、平面点，加入到局部map中；注：称之为局部map，后面进行scan-to-map匹配，所用到的map就是这里的相邻关键帧对应点云集合
        // 下面两个为局部map的角点和面点集合
        laserCloudCornerFromMap->clear();
        laserCloudSurfFromMap->clear(); 
        // 遍历当前帧（实际是取最近的一个关键帧来找它相邻的关键帧集合）时空维度上相邻的关键帧集合
        // cloudToExtract为降采样后的相邻关键帧集合
        for (int i = 0; i < (int)cloudToExtract->size(); ++i)
        {
            // 距离超过阈值，丢弃
            // pointDistance：计算两点之间的距离
            // 剔除掉相邻关键帧集合中距离给定最近关键帧超过阈值的关键帧
            if (pointDistance(cloudToExtract->points[i], cloudKeyPoses3D->back()) > surroundingKeyframeSearchRadius)
                continue;

            // 相邻关键帧索引
            int thisKeyInd = (int)cloudToExtract->points[i].intensity; // 从点云中当前点的强度值提取的索引
            // laserCloudMapContainer是一个map容器（存储局部地图数据），键为索引，值为一对点云对象（角点和面点）
            // 用于将索引与一对角点和平面点云关联起来
            // 检查该map容器中是否存在索引为 thisKeyInd 的条目。
            if (laserCloudMapContainer.find(thisKeyInd) != laserCloudMapContainer.end()) 
            {
                // 从容器中获取索引对应的角点点云和面点点云，并存入相应的局部map中
                *laserCloudCornerFromMap += laserCloudMapContainer[thisKeyInd].first;
                *laserCloudSurfFromMap   += laserCloudMapContainer[thisKeyInd].second;
            } else {
                // 相邻关键帧对应的角点、平面点云，通过6D位姿变换到世界坐标系下
                // transformPointCloud；对输入点云进行位姿变换，输入为历史所有关键帧的角点集合和面点集合，
                // cloudKeyPoses6D：历史关键帧位姿
                // cornerCloudKeyFrames历史所有关键帧的角点集合
                pcl::PointCloud<PointType> laserCloudCornerTemp = *transformPointCloud(cornerCloudKeyFrames[thisKeyInd],  &cloudKeyPoses6D->points[thisKeyInd]);
                pcl::PointCloud<PointType> laserCloudSurfTemp = *transformPointCloud(surfCloudKeyFrames[thisKeyInd],    &cloudKeyPoses6D->points[thisKeyInd]);
                // 加入局部map
                *laserCloudCornerFromMap += laserCloudCornerTemp;
                *laserCloudSurfFromMap   += laserCloudSurfTemp;
                laserCloudMapContainer[thisKeyInd] = make_pair(laserCloudCornerTemp, laserCloudSurfTemp);
            }
            
        }

        // 降采样局部角点map
        downSizeFilterCorner.setInputCloud(laserCloudCornerFromMap);
        downSizeFilterCorner.filter(*laserCloudCornerFromMapDS);
        laserCloudCornerFromMapDSNum = laserCloudCornerFromMapDS->size();
        // 降采样局部平面点map
        downSizeFilterSurf.setInputCloud(laserCloudSurfFromMap);
        downSizeFilterSurf.filter(*laserCloudSurfFromMapDS);
        laserCloudSurfFromMapDSNum = laserCloudSurfFromMapDS->size();

        // 太大了，清空一下内存
        if (laserCloudMapContainer.size() > 1000)
            laserCloudMapContainer.clear();
    }

    /**
     * 提取局部角点、平面点云集合，加入局部map
     * 1、对最近的一帧关键帧，搜索时空维度上相邻的关键帧集合，降采样一下
     * 2、对关键帧集合中的每一帧，提取对应的角点、平面点，加入局部map中
    */
    void extractSurroundingKeyFrames()
    {
        // 如果历史关键帧集合为空则跳过
        if (cloudKeyPoses3D->points.empty() == true)
            return; 
        
        // if (loopClosureEnableFlag == true)
        // {
        //     extractForLoopClosure();    
        // } else {
        //     extractNearby();
        // }

        // 提取局部角点、平面点云集合，加入局部map
        // 1、对最近的一帧关键帧，搜索时空维度上相邻的关键帧集合，降采样一下
        // 2、对关键帧集合中的每一帧，提取对应的角点、平面点，加入局部map中
        extractNearby();
    }

    /**
     * 当前激光帧角点、平面点集合降采样
    */
    void downsampleCurrentScan()
    {
        // 当前激光帧角点集合降采样
        laserCloudCornerLastDS->clear();
        downSizeFilterCorner.setInputCloud(laserCloudCornerLast);
        downSizeFilterCorner.filter(*laserCloudCornerLastDS);
        laserCloudCornerLastDSNum = laserCloudCornerLastDS->size();

        // 当前激光帧平面点集合降采样
        laserCloudSurfLastDS->clear();
        downSizeFilterSurf.setInputCloud(laserCloudSurfLast);
        downSizeFilterSurf.filter(*laserCloudSurfLastDS);
        laserCloudSurfLastDSNum = laserCloudSurfLastDS->size();
    }

    /**
     * 更新当前帧位姿
    */
    void updatePointAssociateToMap()
    {
        // trans2Affine3f：将位姿数组转换为矩阵
        // 该矩阵用于后续将point从lidar系变换到map系下
        transPointAssociateToMap = trans2Affine3f(transformTobeMapped);
    }

    /**
     * 当前激光帧角点寻找局部map匹配点
     * 1、更新当前帧位姿，将当前帧角点坐标变换到map系下，在局部map中查找5个最近点，距离小于1m，且5个点构成直线（用距离中心点的协方差矩阵，特征值进行判断），则认为匹配上了
     * 2、计算当前帧角点到直线的距离、垂线的单位向量，存储为角点参数
    */
    void cornerOptimization()
    {
        // 更新当前帧位姿
        // 计算将当前帧角点坐标变换到map(world)系下的矩阵
        updatePointAssociateToMap();

        // 使用OpenMP库的指令来并行化一个循环，用于实现多线程并行化
        // numberOfCores：指定并行化使用的线程数
        #pragma omp parallel for num_threads(numberOfCores)
        // 遍历当前帧角点集合
        for (int i = 0; i < laserCloudCornerLastDSNum; i++)
        {
            PointType pointOri, pointSel, coeff;
            // kd-tree搜索的索引和距离
            std::vector<int> pointSearchInd;
            std::vector<float> pointSearchSqDis;

            // 角点（坐标还是lidar系）
            pointOri = laserCloudCornerLastDS->points[i];
            // 根据当前帧位姿，从lidar系变换到世界坐标系（map系）下
            // 第一步计算的矩阵在这个函数中用到
            pointAssociateToMap(&pointOri, &pointSel);
            // 在局部角点map中查找当前角点相邻的5个角点
            kdtreeCornerFromMap->nearestKSearch(pointSel, 5, pointSearchInd, pointSearchSqDis);

            // 初始化协方差矩阵、特征值和特征向量
            cv::Mat matA1(3, 3, CV_32F, cv::Scalar::all(0));//协方差矩阵
            cv::Mat matD1(1, 3, CV_32F, cv::Scalar::all(0));
            cv::Mat matV1(3, 3, CV_32F, cv::Scalar::all(0));
            
            // 下面scan to map匹配，寻找匹配点的过程和aloam一致
            // 要求距离都小于1m
            if (pointSearchSqDis[4] < 1.0) {
                // 计算5个点的均值坐标，记为中心点
                float cx = 0, cy = 0, cz = 0;
                for (int j = 0; j < 5; j++) {
                    cx += laserCloudCornerFromMapDS->points[pointSearchInd[j]].x;
                    cy += laserCloudCornerFromMapDS->points[pointSearchInd[j]].y;
                    cz += laserCloudCornerFromMapDS->points[pointSearchInd[j]].z;
                }
                cx /= 5; cy /= 5;  cz /= 5; // 中心点

                // 计算协方差
                float a11 = 0, a12 = 0, a13 = 0, a22 = 0, a23 = 0, a33 = 0;
                for (int j = 0; j < 5; j++) {
                    // 计算点与中心点之间的距离
                    float ax = laserCloudCornerFromMapDS->points[pointSearchInd[j]].x - cx;
                    float ay = laserCloudCornerFromMapDS->points[pointSearchInd[j]].y - cy;
                    float az = laserCloudCornerFromMapDS->points[pointSearchInd[j]].z - cz;

                    a11 += ax * ax; a12 += ax * ay; a13 += ax * az;
                    a22 += ay * ay; a23 += ay * az;
                    a33 += az * az;
                }
                a11 /= 5; a12 /= 5; a13 /= 5; a22 /= 5; a23 /= 5; a33 /= 5;

                // 构建协方差矩阵
                matA1.at<float>(0, 0) = a11; matA1.at<float>(0, 1) = a12; matA1.at<float>(0, 2) = a13;
                matA1.at<float>(1, 0) = a12; matA1.at<float>(1, 1) = a22; matA1.at<float>(1, 2) = a23;
                matA1.at<float>(2, 0) = a13; matA1.at<float>(2, 1) = a23; matA1.at<float>(2, 2) = a33;

                // 特征值分解，最大特征值对应的特征向量为数据主方向
                cv::eigen(matA1, matD1, matV1);//用于分解得到输入矩阵matA1的特征值和特征向量，D1存储计算得到的特征值，V1存储特征向量

                // 如果最大的特征值相比次大特征值，大很多，认为构成了线，角点是合格的
                if (matD1.at<float>(0, 0) > 3 * matD1.at<float>(0, 1)) {
                    // 以下部分是在计算当前点pointSel到检索出的直线的距离和方向，如果距离够近，则认为匹配成功，否则认为匹配失败
                    // 当前帧角点坐标（map系下）
                    float x0 = pointSel.x;
                    float y0 = pointSel.y;
                    float z0 = pointSel.z;
                    // 局部map对应中心角点，沿着特征向量（直线方向）方向，前后各取一个点
                    // matV1（特征向量）的第一行就是5个点形成的直线的方向，cx,cy,cz是5个点的中心点
                    // 这里的中心点是在局部map中通过kd-tree找到的最近5个点的中心点
                    // 因此，x1,y1,z1和x2,y2,z2是经过中心点的直线上的另外两个点，两点之间的距离是0.2米
                    float x1 = cx + 0.1 * matV1.at<float>(0, 0);
                    float y1 = cy + 0.1 * matV1.at<float>(0, 1);
                    float z1 = cz + 0.1 * matV1.at<float>(0, 2);
                    float x2 = cx - 0.1 * matV1.at<float>(0, 0);
                    float y2 = cy - 0.1 * matV1.at<float>(0, 1);
                    float z2 = cz - 0.1 * matV1.at<float>(0, 2);

                    // area_012，也就是三个点组成的三角形面积*2，叉积的模|axb|=a*b*sin(theta)
                    // 这边是在求[(x0-x1),(y0-y1),(z0-z1)]与[(x0-x2),(y0-y2),(z0-z2)]叉乘得到的向量的模长
                    // 这个模长是由0.2*V1[0]和点[x0,y0,z0]构成的平行四边形的面积
                    // 垂直于0,1,2三点构成的平面的向量[XXX,YYY,ZZZ] = [(y0-y1)(z0-z2)-(y0-y2)(z0-z1),-(x0-x1)(z0-z2)+(x0-x2)(z0-z1),(x0-x1)(y0-y2)-(x0-x2)(y0-y1)]
                    // x0为当前帧角点坐标（map系下），x1和x2为局部map中搜索到的直线上的两个点
                    float a012 = sqrt(((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1)) * ((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1)) 
                                    + ((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1)) * ((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1)) 
                                    + ((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1)) * ((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1)));
                    
                    // line_12，底边边长
                    // 点x1,y1,z1到点x2,y2,z2的距离
                    float l12 = sqrt((x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2) + (z1 - z2)*(z1 - z2));
                    
                    // 两次叉积，得到点到直线的垂线段单位向量，x分量，下面同理
                    float la = ((y1 - y2)*((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1)) 
                              + (z1 - z2)*((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1))) / a012 / l12;

                    float lb = -((x1 - x2)*((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1)) 
                               - (z1 - z2)*((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1))) / a012 / l12;

                    float lc = -((x1 - x2)*((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1)) 
                               + (y1 - y2)*((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1))) / a012 / l12;

                    // 三角形的高，也就是点到直线距离
                    // ld2就是点pointSel(x0,y0,z0)到直线的距离
                    float ld2 = a012 / l12;

                    // 距离越大，s越小，是个距离惩罚因子（权重）
                    float s = 1 - 0.9 * fabs(ld2);

                    // 点到直线的垂线段单位向量
                    coeff.x = s * la;
                    coeff.y = s * lb;
                    coeff.z = s * lc;
                    // 点到直线距离
                    coeff.intensity = s * ld2;

                    if (s > 0.1) {
                        // 当前激光帧角点，加入匹配集合中
                        // 将当前帧与局部map匹配上的角点加入匹配集合中
                        // 这里添加的角点是坐标变换之前的角点（lidar坐标系下的）
                        laserCloudOriCornerVec[i] = pointOri;
                        // 角点的参数
                        coeffSelCornerVec[i] = coeff;
                        laserCloudOriCornerFlag[i] = true;
                    }
                }
            }
        }
    }

    /**
     * 当前激光帧平面点寻找局部map匹配点
     * 1、更新当前帧位姿，将当前帧平面点坐标变换到map系下，在局部map中查找5个最近点，距离小于1m，且5个点构成平面（最小二乘拟合平面），则认为匹配上了
     * 2、计算当前帧平面点到平面的距离、垂线的单位向量，存储为平面点参数
    */
    void surfOptimization()
    {
        // 更新当前帧位姿
        updatePointAssociateToMap();//计算将当前点从lidar系变换到map系下的矩阵

         // 遍历当前帧平面点集合
        #pragma omp parallel for num_threads(numberOfCores)
        for (int i = 0; i < laserCloudSurfLastDSNum; i++)
        {
            PointType pointOri, pointSel, coeff;
            std::vector<int> pointSearchInd;
            std::vector<float> pointSearchSqDis;
            
            // 平面点（坐标还是lidar系）
            pointOri = laserCloudSurfLastDS->points[i];
            // 根据当前帧位姿，变换到世界坐标系（map系）下
            pointAssociateToMap(&pointOri, &pointSel); 
            // 在局部平面点map中查找当前平面点相邻的5个平面点
            kdtreeSurfFromMap->nearestKSearch(pointSel, 5, pointSearchInd, pointSearchSqDis);

            // 下面的过程要求解Ax+By+Cz+1=0的平面方程
            // 由于有5个点，因此是求解超定方程
            // 假设5个点都在平面上，则matA0是系数矩阵，matB0是等号右边的值（都是-1）；matX0是求出来的A，B，C
            Eigen::Matrix<float, 5, 3> matA0;
            Eigen::Matrix<float, 5, 1> matB0;
            Eigen::Vector3f matX0;

            matA0.setZero();
            matB0.fill(-1);
            matX0.setZero();

            // 要求距离都小于1m
            if (pointSearchSqDis[4] < 1.0) {
                for (int j = 0; j < 5; j++) {
                    // 利用从局部map中搜索到的5个点来构建系数矩阵
                    matA0(j, 0) = laserCloudSurfFromMapDS->points[pointSearchInd[j]].x;
                    matA0(j, 1) = laserCloudSurfFromMapDS->points[pointSearchInd[j]].y;
                    matA0(j, 2) = laserCloudSurfFromMapDS->points[pointSearchInd[j]].z;
                }

                // 假设平面方程为ax+by+cz+1=0，这里就是求方程的系数abc，d=1
                // 这里是求解matA0XmatX0 = matB0方程
                matX0 = matA0.colPivHouseholderQr().solve(matB0);

                // 平面方程的系数，也是法向量的分量
                float pa = matX0(0, 0);
                float pb = matX0(1, 0);
                float pc = matX0(2, 0);
                float pd = 1;

                // 单位法向量
                // （pa,pb,pc)是平面的法向量，这里是对法向量规一化，变成单位法向量
                float ps = sqrt(pa * pa + pb * pb + pc * pc);
                pa /= ps; pb /= ps; pc /= ps; pd /= ps;

                // 检查平面是否合格，如果5个点中有点到平面的距离超过0.2m，那么认为这些点太分散了，不构成平面
                bool planeValid = true;
                for (int j = 0; j < 5; j++) {
                    if (fabs(pa * laserCloudSurfFromMapDS->points[pointSearchInd[j]].x +
                             pb * laserCloudSurfFromMapDS->points[pointSearchInd[j]].y +
                             pc * laserCloudSurfFromMapDS->points[pointSearchInd[j]].z + pd) > 0.2) {
                        planeValid = false;
                        break;
                    }
                }

                // 平面合格
                if (planeValid) {
                    // 当前激光帧点到平面距离
                    // （pa,pb,pc)是平面的法向量
                    float pd2 = pa * pointSel.x + pb * pointSel.y + pc * pointSel.z + pd;

                    float s = 1 - 0.9 * fabs(pd2) / sqrt(sqrt(pointSel.x * pointSel.x
                            + pointSel.y * pointSel.y + pointSel.z * pointSel.z));

                    // 点到平面垂线单位法向量（其实等价于平面法向量）
                    coeff.x = s * pa;
                    coeff.y = s * pb;
                    coeff.z = s * pc;
                    // 点到平面距离
                    coeff.intensity = s * pd2;

                    if (s > 0.1) {
                        // 当前激光帧平面点，加入匹配集合中
                        laserCloudOriSurfVec[i] = pointOri;
                        // 参数
                        coeffSelSurfVec[i] = coeff;
                        laserCloudOriSurfFlag[i] = true;
                    }
                }
            }
        }
    }

    /**
     * 提取当前帧中与局部map匹配上了的角点、平面点，加入同一集合
    */
    void combineOptimizationCoeffs()
    {
        // 遍历当前帧角点集合，提取出与局部map匹配上了的角点
        for (int i = 0; i < laserCloudCornerLastDSNum; ++i){
            if (laserCloudOriCornerFlag[i] == true){
                laserCloudOri->push_back(laserCloudOriCornerVec[i]);// 角点
                coeffSel->push_back(coeffSelCornerVec[i]);// 参数
            }
        }
        // 遍历当前帧平面点集合，提取出与局部map匹配上了的平面点
        for (int i = 0; i < laserCloudSurfLastDSNum; ++i){
            if (laserCloudOriSurfFlag[i] == true){
                laserCloudOri->push_back(laserCloudOriSurfVec[i]);
                coeffSel->push_back(coeffSelSurfVec[i]);
            }
        }
        // 清空标记
        std::fill(laserCloudOriCornerFlag.begin(), laserCloudOriCornerFlag.end(), false);
        std::fill(laserCloudOriSurfFlag.begin(), laserCloudOriSurfFlag.end(), false);
    }

    /**
     * scan-to-map优化
     * 对匹配特征点计算Jacobian矩阵，观测值为特征点到直线、平面的距离，构建高斯牛顿方程，迭代优化当前位姿，存transformTobeMapped
     * 公式推导：todo
     * int iterCount ：当前是第几次迭代
    */
    bool LMOptimization(int iterCount)
    {
        // 这部分应该是参考了别人的优化方法，该方法针对VSLAM，参考坐标系为相机坐标系。因此需要将当前点的数据从激光坐标系转到相机坐标系中。

        // This optimization is from the original loam_velodyne by Ji Zhang, need to cope with coordinate transformation
        // lidar <- camera      ---     camera <- lidar
        // x = z                ---     x = y
        // y = x                ---     y = z
        // z = y                ---     z = x
        // roll = yaw           ---     roll = pitch
        // pitch = roll         ---     pitch = yaw
        // yaw = pitch          ---     yaw = roll

        // lidar -> camera
        // 012存放旋转角  用 sin 和 cos 函数来计算绕 z、x 和 y 轴的旋转的 sin 和 cos 值
        // 计算三轴欧拉角的sin、cos，后面使用旋转矩阵对欧拉角求导中会使用到
        // 为了构建旋转矩阵，将点从激光雷达坐标系（lidar）转换到相机坐标系（camera）
        float srx = sin(transformTobeMapped[1]);
        float crx = cos(transformTobeMapped[1]);
        float sry = sin(transformTobeMapped[2]);
        float cry = cos(transformTobeMapped[2]);
        float srz = sin(transformTobeMapped[0]);
        float crz = cos(transformTobeMapped[0]);

        // 当前帧匹配特征点数太少
        // laserCloudOri：当前帧中与局部map匹配上的角点和面点集合
        int laserCloudSelNum = laserCloudOri->size();
        if (laserCloudSelNum < 50) {
            return false;
        }

        // matA是Jacobians矩阵J
        cv::Mat matA(laserCloudSelNum, 6, CV_32F, cv::Scalar::all(0));
        cv::Mat matAt(6, laserCloudSelNum, CV_32F, cv::Scalar::all(0));
        cv::Mat matAtA(6, 6, CV_32F, cv::Scalar::all(0));
        // matB是目标函数，也就是距离
        cv::Mat matB(laserCloudSelNum, 1, CV_32F, cv::Scalar::all(0));
        cv::Mat matAtB(6, 1, CV_32F, cv::Scalar::all(0));
        // matX是高斯-牛顿法计算出的更新向量
        cv::Mat matX(6, 1, CV_32F, cv::Scalar::all(0));

        PointType pointOri, coeff;

        // 遍历匹配特征点，构建Jacobian矩阵
        // 将点及对应的参数从雷达坐标系转到相机坐标系中
        for (int i = 0; i < laserCloudSelNum; i++) {
            // lidar -> camera todo
            // 点
            pointOri.x = laserCloudOri->points[i].y;
            pointOri.y = laserCloudOri->points[i].z;
            pointOri.z = laserCloudOri->points[i].x;
            // lidar -> camera
            // 参数
            coeff.x = coeffSel->points[i].y;
            coeff.y = coeffSel->points[i].z;
            coeff.z = coeffSel->points[i].x;
            coeff.intensity = coeffSel->points[i].intensity;
            // in camera
            // 求雅克比矩阵的值，也就是求目标函数（点到线、平面的距离）相对于tx,ty,tz,rx,ry,rz的导数
            // 具体的公式推导看仓库README中本项目博客，高斯牛顿法方程：J^{T}J\Delta{x} = -Jf(x)，\Delta{x}就是要求解的更新向量matX
            // arx是目标函数相对于roll的导数
            float arx = (crx*sry*srz*pointOri.x + crx*crz*sry*pointOri.y - srx*sry*pointOri.z) * coeff.x
                      + (-srx*srz*pointOri.x - crz*srx*pointOri.y - crx*pointOri.z) * coeff.y
                      + (crx*cry*srz*pointOri.x + crx*cry*crz*pointOri.y - cry*srx*pointOri.z) * coeff.z;
            // ary是目标函数相对于pitch的导数
            float ary = ((cry*srx*srz - crz*sry)*pointOri.x 
                      + (sry*srz + cry*crz*srx)*pointOri.y + crx*cry*pointOri.z) * coeff.x
                      + ((-cry*crz - srx*sry*srz)*pointOri.x 
                      + (cry*srz - crz*srx*sry)*pointOri.y - crx*sry*pointOri.z) * coeff.z;
            // arz是目标函数相对于yaw的导数
            float arz = ((crz*srx*sry - cry*srz)*pointOri.x + (-cry*crz-srx*sry*srz)*pointOri.y)*coeff.x
                      + (crx*crz*pointOri.x - crx*srz*pointOri.y) * coeff.y
                      + ((sry*srz + cry*crz*srx)*pointOri.x + (crz*sry-cry*srx*srz)*pointOri.y)*coeff.z;
            
            /*
            在求点到直线的距离时，coeff表示的是如下内容
            [la,lb,lc]表示的是点到直线的垂直连线方向，s是长度
            coeff.x = s * la;
            coeff.y = s * lb;
            coeff.z = s * lc;
            coeff.intensity = s * ld2;

            在求点到平面的距离时，coeff表示的是
            [pa,pb,pc]表示过外点的平面的法向量，s是线的长度
            coeff.x = s * pa;
            coeff.y = s * pb;
            coeff.z = s * pc;
            coeff.intensity = s * pd2;
            */

            // lidar -> camera
            // 构建雅可比矩阵，前三个分别是目标函数相对于rpy的导数
            matA.at<float>(i, 0) = arz;
            matA.at<float>(i, 1) = arx;
            matA.at<float>(i, 2) = ary;
            // 目标函数相对于tx的导数等于法向量的x
            matA.at<float>(i, 3) = coeff.z;
            // 目标函数相对于ty的导数等于法向量的y
            matA.at<float>(i, 4) = coeff.x;
            // 目标函数相对于tz的导数等于法向量的z
            matA.at<float>(i, 5) = coeff.y;

            // 点到直线距离、平面距离，作为观测值
            // 残差项
            // matB存储的是目标函数（距离）的负值，因为：J^{T}J\Delta{x} = -Jf(x)
            matB.at<float>(i, 0) = -coeff.intensity;
        }

        cv::transpose(matA, matAt);// 计算雅可比矩阵A的转置
        matAtA = matAt * matA;
        matAtB = matAt * matB;
        // J^T·J·delta_x = -J^T·f 高斯牛顿
        // QR分解计算matAtA* matX = matAtB
        // 求解高斯-牛顿法中的增量方程：J^{T}J\Delta{x} = -Jf(x)，这里解出来的matX就是更新向量
        // matA是雅克比矩阵J
        // matAtB是上面等式中等号的右边，负号在matB赋值的时候已经加入
        cv::solve(matAtA, matAtB, matX, cv::DECOMP_QR);

        // 首次迭代，检查近似Hessian矩阵（J^T·J）是否退化，或者称为奇异，行列式值=0 todo
        // 对于第一次迭代需要进行初始化，主要是针对退化问题。
        // 如果是第一次迭代，判断求解出来的近似Hessian矩阵，也就是J^{T}J:=matAtA是否退化
/**
    * 这部分的计算说实话没有找到很好的理论出处，这里只能大概说一下这段代码想要做的事情
    * 这里用matAtA也就是高斯-牛顿中的近似海瑟（Hessian）矩阵H。求解增量方程：J^{T}J\Delta{x} = -Jf(x)
    * 要求H:=J^{T}J可逆，但H不一定可逆。下面的代码通过H的特征值判断H是否退化，并将退化的方向清零matV2。而后又根据
    * matV.inv()*matV2作为更新向量的权重系数，matV是H的特征向量矩阵。
*/
        if (iterCount == 0) {

            cv::Mat matE(1, 6, CV_32F, cv::Scalar::all(0));
            cv::Mat matV(6, 6, CV_32F, cv::Scalar::all(0));
            cv::Mat matV2(6, 6, CV_32F, cv::Scalar::all(0));
            
            // 对近似Hessian矩阵做特征值分解，matE是特征值，matV是特征向量。opencv的matV中每一行是一个特征向量
            cv::eigen(matAtA, matE, matV);
            matV.copyTo(matV2);

            isDegenerate = false;
            float eignThre[6] = {100, 100, 100, 100, 100, 100};
            for (int i = 5; i >= 0; i--) {
                if (matE.at<float>(0, i) < eignThre[i]) {
                    for (int j = 0; j < 6; j++) {
                        matV2.at<float>(i, j) = 0;
                    }
                    isDegenerate = true;
                } else {
                    break;
                }
            }
            matP = matV.inv() * matV2;
        }

        // 当第一次迭代判断到海瑟矩阵退化，后面会使用计算出来的权重matP对增量matX做加权组合
        if (isDegenerate)
        {
            cv::Mat matX2(6, 1, CV_32F, cv::Scalar::all(0));
            matX.copyTo(matX2);
            matX = matP * matX2;
        }

        // 更新当前位姿 x = x + delta_x
        // 这里是计算 x = x + delta_x然后计算前后两次的delta_x的变化量。如果delta_x小于阈值，则认为结果收敛，则不再进行迭代.
        // 将增量matX叠加到变量（位姿）transformTobeMapped中
        transformTobeMapped[0] += matX.at<float>(0, 0);
        transformTobeMapped[1] += matX.at<float>(1, 0);
        transformTobeMapped[2] += matX.at<float>(2, 0);
        transformTobeMapped[3] += matX.at<float>(3, 0);
        transformTobeMapped[4] += matX.at<float>(4, 0);
        transformTobeMapped[5] += matX.at<float>(5, 0);

        //下面对收敛条件进行判断
        // 当增量步长达到一定阈值后，认为优化已经收敛，因此可以跳出后续迭代。这里用增量中的角度增量和平移增量的幅度做判断。
        // 计算roll、pitch、yaw的迭代步长
        float deltaR = sqrt(
                            pow(pcl::rad2deg(matX.at<float>(0, 0)), 2) +
                            pow(pcl::rad2deg(matX.at<float>(1, 0)), 2) +
                            pow(pcl::rad2deg(matX.at<float>(2, 0)), 2));
        // 计算tx，ty，tz的迭代步长
        float deltaT = sqrt(
                            pow(matX.at<float>(3, 0) * 100, 2) +
                            pow(matX.at<float>(4, 0) * 100, 2) +
                            pow(matX.at<float>(5, 0) * 100, 2));

        // delta_x很小，认为收敛
        // 如果迭代的步长达到设定阈值，则认为已经收敛
        if (deltaR < 0.05 && deltaT < 0.05) {
            return true; 
        }
        return false; 
    }

    /**
     * scan-to-map优化当前帧位姿
     * 1、要求当前帧特征点数量足够多，且匹配的点数够多，才执行优化
     * 2、迭代30次（上限）优化
     *   1) 当前激光帧角点寻找局部map匹配点
     *      a.更新当前帧位姿，将当前帧角点坐标变换到map系下，在局部map中查找5个最近点，距离小于1m，且5个点构成直线（用距离中心点的协方差矩阵，特征值进行判断），则认为匹配上了
     *      b.计算当前帧角点到直线的距离、垂线的单位向量，存储为角点参数
     *   2) 当前激光帧平面点寻找局部map匹配点
     *      a.更新当前帧位姿，将当前帧平面点坐标变换到map系下，在局部map中查找5个最近点，距离小于1m，且5个点构成平面（最小二乘拟合平面），则认为匹配上了
     *      b.计算当前帧平面点到平面的距离、垂线的单位向量，存储为平面点参数
     *   3) 提取当前帧中与局部map匹配上了的角点、平面点，加入同一集合
     *   4) 对匹配特征点计算Jacobian矩阵，观测值为特征点到直线、平面的距离，构建高斯牛顿方程，迭代优化当前位姿，存transformTobeMapped
     * 3、用imu原始RPY数据与scan-to-map优化后的位姿进行加权融合，更新当前帧位姿的roll、pitch，约束z坐标
    */
    void scan2MapOptimization()
    {
        // 要求有关键帧
        // 即第一帧激光数据无需匹配，跳过第一帧
        if (cloudKeyPoses3D->points.empty())
            return;

        // 降采样后的当前激光帧的角点、平面点数量足够多
        if (laserCloudCornerLastDSNum > edgeFeatureMinValidNum && laserCloudSurfLastDSNum > surfFeatureMinValidNum)
        {
            // kdtree输入为局部map点云
            kdtreeCornerFromMap->setInputCloud(laserCloudCornerFromMapDS);
            kdtreeSurfFromMap->setInputCloud(laserCloudSurfFromMapDS);

            // 迭代30次
            for (int iterCount = 0; iterCount < 30; iterCount++)
            {
                // 每次迭代清空特征点集合
                // 将当前帧scan和局部map匹配上的角点和面点加入相应的特征点集合
                laserCloudOri->clear();// 存储当前帧和局部地图匹配上的角点和面点
                coeffSel->clear();// 存储当前帧和局部地图匹配上的角点和面点的参数

                // 当前激光帧角点寻找局部map匹配点
                // 1、更新当前帧位姿，将当前帧角点坐标变换到map系下，在局部map中查找5个最近点，距离小于1m，且5个点构成直线（用距离中心点的协方差矩阵，特征值进行判断），则认为匹配上了
                // 2、计算当前帧角点到直线的距离、垂线的单位向量，存储为角点参数
                cornerOptimization();

                // 当前激光帧平面点寻找局部map匹配点
                // 1、更新当前帧位姿，将当前帧平面点坐标变换到map系下，在局部map中查找5个最近点，距离小于1m，且5个点构成平面（最小二乘拟合平面），则认为匹配上了
                // 2、计算当前帧平面点到平面的距离、垂线的单位向量，存储为平面点参数
                surfOptimization();

                // 提取当前帧中与局部map匹配上了的角点、平面点，加入同一集合
                combineOptimizationCoeffs();

                // scan-to-map优化
                // 对匹配特征点计算Jacobian矩阵，观测值为特征点到直线、平面的距离，构建高斯牛顿方程，迭代优化当前位姿，存transformTobeMapped
                // 满足收敛条件时LMOptimization返回true，跳出当前迭代循环
                if (LMOptimization(iterCount) == true)
                    break;              
            }
            // 用imu原始RPY数据与scan-to-map优化后的位姿进行加权融合，更新当前帧位姿的roll、pitch，约束z坐标
            transformUpdate(); //得到当前帧的位姿
        } else {
            ROS_WARN("Not enough features! Only %d edge and %d planar features available.", laserCloudCornerLastDSNum, laserCloudSurfLastDSNum);
        }
    }

    /**
     * 用imu原始RPY数据与scan-to-map优化后的位姿进行加权融合，更新当前帧位姿的roll、pitch，约束z坐标
    */
    void transformUpdate()
    {
        if (cloudInfo.imuAvailable == true)
        {
            // 俯仰角小于1.4
            // 从点云信息中提取imu原始数据
            if (std::abs(cloudInfo.imuPitchInit) < 1.4)
            {
                double imuWeight = imuRPYWeight;
                tf::Quaternion imuQuaternion;
                tf::Quaternion transformQuaternion;
                double rollMid, pitchMid, yawMid;

                // roll角求加权均值，用scan-to-map优化得到的位姿与imu原始RPY数据，进行加权平均
                transformQuaternion.setRPY(transformTobeMapped[0], 0, 0);
                imuQuaternion.setRPY(cloudInfo.imuRollInit, 0, 0);
                tf::Matrix3x3(transformQuaternion.slerp(imuQuaternion, imuWeight)).getRPY(rollMid, pitchMid, yawMid);
                transformTobeMapped[0] = rollMid;

                // pitch角求加权均值，用scan-to-map优化得到的位姿与imu原始RPY数据，进行加权平均
                transformQuaternion.setRPY(0, transformTobeMapped[1], 0);
                imuQuaternion.setRPY(0, cloudInfo.imuPitchInit, 0);
                tf::Matrix3x3(transformQuaternion.slerp(imuQuaternion, imuWeight)).getRPY(rollMid, pitchMid, yawMid);
                transformTobeMapped[1] = pitchMid;
            }
        }

        // 更新当前帧位姿的roll, pitch, z坐标；因为是小车，roll、pitch是相对稳定的，不会有很大变动，一定程度上可以信赖imu的数据，z是进行高度约束
        // 这三个limit值从参数yaml文件中读取
        transformTobeMapped[0] = constraintTransformation(transformTobeMapped[0], rotation_tollerance);
        transformTobeMapped[1] = constraintTransformation(transformTobeMapped[1], rotation_tollerance);
        transformTobeMapped[5] = constraintTransformation(transformTobeMapped[5], z_tollerance);

        // 当前帧位姿（转换为位姿矩阵）
        incrementalOdometryAffineBack = trans2Affine3f(transformTobeMapped);
    }

    /**
     * 值约束
    */
    float constraintTransformation(float value, float limit)
    {
        if (value < -limit)
            value = -limit;
        if (value > limit)
            value = limit;

        return value;
    }

    /**
     * 计算当前帧与前一帧位姿变换（关键帧之间的位姿变换），如果变化太小，不设为关键帧，反之设为关键帧
    */
    bool saveFrame()
    {
        // cloudKeyPoses3D：存放历史关键帧的3D位姿
        if (cloudKeyPoses3D->points.empty())
            return true; // 说明为第一帧，那么设置为关键帧

        // 后续帧的处理：

        // 前一帧位姿
        // cloudKeyPoses6D 存放历史关键帧6D位姿
        Eigen::Affine3f transStart = pclPointToAffine3f(cloudKeyPoses6D->back());
        // 当前帧位姿（上一步scan to map优化计算得到的）
        Eigen::Affine3f transFinal = pcl::getTransformation(transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5], 
                                                            transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);
        // 位姿变换增量
        Eigen::Affine3f transBetween = transStart.inverse() * transFinal;
        float x, y, z, roll, pitch, yaw;
        pcl::getTranslationAndEulerAngles(transBetween, x, y, z, roll, pitch, yaw);

        // 旋转和平移量都较小，当前帧不设为关键帧
        if (abs(roll)  < surroundingkeyframeAddingAngleThreshold &&
            abs(pitch) < surroundingkeyframeAddingAngleThreshold && 
            abs(yaw)   < surroundingkeyframeAddingAngleThreshold &&
            sqrt(x*x + y*y + z*z) < surroundingkeyframeAddingDistThreshold)
            return false;

        return true;
    }

    /**
     * 添加激光里程计因子
    */
    void addOdomFactor()
    {
        // cloudKeyPoses3D存放了历史关键帧位姿（只有位置）
        if (cloudKeyPoses3D->points.empty())
        {
            // 第一帧初始化先验因子
            // 第一帧需要创建先验因子并添加到因子图中，这里的位姿先验是一个较小的噪声模型
            // 先验因子约束了优化变量的初值
            noiseModel::Diagonal::shared_ptr priorNoise = noiseModel::Diagonal::Variances((Vector(6) << 1e-2, 1e-2, M_PI*M_PI, 1e8, 1e8, 1e8).finished()); // rad*rad, meter*meter
            // 向因子图中添加先验因子
            // PriorFactor<Pose3>：先验因子的类型；0：对索引为0的变量施加先验
            // trans2gtsamPose(transformTobeMapped)：将位姿转换为gtSAM库中的Pose3类型，用于指定初始位姿
            // priorNoise：噪声模型，用于表示先验信息的不确定性
            gtSAMgraph.add(PriorFactor<Pose3>(0, trans2gtsamPose(transformTobeMapped), priorNoise));
            // 变量节点设置初始值
            // initialEstimate 存放初始位姿值
            // trans2gtsamPose：将位姿数组转换为GTSAM格式的pose
            // 0表示插入的位置，这意味着将位姿对象插入到容器的第一个位置。
            initialEstimate.insert(0, trans2gtsamPose(transformTobeMapped));
        }else{
            // 添加激光里程计因子
            noiseModel::Diagonal::shared_ptr odometryNoise = noiseModel::Diagonal::Variances((Vector(6) << 1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4).finished());
            // 上一帧的位姿（位置+姿态）
            gtsam::Pose3 poseFrom = pclPointTogtsamPose3(cloudKeyPoses6D->points.back());
            // 当前激光帧的位姿
            gtsam::Pose3 poseTo   = trans2gtsamPose(transformTobeMapped);
            // 参数：前一帧id，当前帧id，前一帧与当前帧的位姿变换（作为观测值），噪声协方差
            // poseFrom.between(poseTo)  用于计算前一帧与当前帧之间的相对位姿变换，将其作为观测值传递给激光里程计因子
            // 因子图中的因子是观测值，激光里程计因子也就是当前Lidar帧与上一帧之间的相对位姿变换
            gtSAMgraph.add(BetweenFactor<Pose3>(cloudKeyPoses3D->size()-1, cloudKeyPoses3D->size(), poseFrom.between(poseTo), odometryNoise));
            // 变量节点设置初始值
            // cloudKeyPoses3D->size()表示当前帧的索引
            initialEstimate.insert(cloudKeyPoses3D->size(), poseTo);
        }
    }

    /**
     * 添加GPS因子
    */
    void addGPSFactor()
    {
        if (gpsQueue.empty())
            return;

        // 如果没有关键帧，或者首尾关键帧距离小于5m，不添加gps因子
        if (cloudKeyPoses3D->points.empty())
            return;
        else
        {
            if (pointDistance(cloudKeyPoses3D->front(), cloudKeyPoses3D->back()) < 5.0)
                return;
        }

        // 位姿协方差很小，没必要加入GPS数据进行校正
        if (poseCovariance(3,3) < poseCovThreshold && poseCovariance(4,4) < poseCovThreshold)
            return;

        static PointType lastGPSPoint;

        while (!gpsQueue.empty())
        {
            // 删除当前帧0.2s之前的里程计
            // timeLaserInfoCur：当前激光帧的时间戳
            if (gpsQueue.front().header.stamp.toSec() < timeLaserInfoCur - 0.2)
            {
                gpsQueue.pop_front();
            }
            // 超过当前帧0.2s之后，退出
            else if (gpsQueue.front().header.stamp.toSec() > timeLaserInfoCur + 0.2)
            {
                break;
            }
            else
            {
                nav_msgs::Odometry thisGPS = gpsQueue.front();
                gpsQueue.pop_front();//删除队头元素

                // GPS噪声协方差太大，不能用
                float noise_x = thisGPS.pose.covariance[0];
                float noise_y = thisGPS.pose.covariance[7];
                float noise_z = thisGPS.pose.covariance[14];
                if (noise_x > gpsCovThreshold || noise_y > gpsCovThreshold)
                    continue;

                // GPS里程计位置
                float gps_x = thisGPS.pose.pose.position.x;
                float gps_y = thisGPS.pose.pose.position.y;
                float gps_z = thisGPS.pose.pose.position.z;
                if (!useGpsElevation) // 从参数yaml文件中读取（设置的是false）
                {
                    gps_z = transformTobeMapped[5];
                    noise_z = 0.01;
                }

                // (0,0,0)无效数据
                if (abs(gps_x) < 1e-6 && abs(gps_y) < 1e-6)
                    continue;

                // 每隔5m添加一个GPS里程计
                PointType curGPSPoint;
                curGPSPoint.x = gps_x;
                curGPSPoint.y = gps_y;
                curGPSPoint.z = gps_z;
                if (pointDistance(curGPSPoint, lastGPSPoint) < 5.0)
                    continue;
                else
                    lastGPSPoint = curGPSPoint;

                // 添加GPS因子
                gtsam::Vector Vector3(3);
                Vector3 << max(noise_x, 1.0f), max(noise_y, 1.0f), max(noise_z, 1.0f);
                noiseModel::Diagonal::shared_ptr gps_noise = noiseModel::Diagonal::Variances(Vector3);
                gtsam::GPSFactor gps_factor(cloudKeyPoses3D->size(), gtsam::Point3(gps_x, gps_y, gps_z), gps_noise);
                gtSAMgraph.add(gps_factor);

                aLoopIsClosed = true;
                break;
            }
        }
    }

    /**
     * 添加闭环因子
    */
    void addLoopFactor()
    {
        if (loopIndexQueue.empty())
            return;

        // 闭环队列
        for (int i = 0; i < (int)loopIndexQueue.size(); ++i)
        {
            // 闭环边对应两帧的索引
            int indexFrom = loopIndexQueue[i].first;
            int indexTo = loopIndexQueue[i].second;
            // 闭环边的位姿变换
            gtsam::Pose3 poseBetween = loopPoseQueue[i];
            gtsam::noiseModel::Diagonal::shared_ptr noiseBetween = loopNoiseQueue[i];
            gtSAMgraph.add(BetweenFactor<Pose3>(indexFrom, indexTo, poseBetween, noiseBetween));
        }

        loopIndexQueue.clear();
        loopPoseQueue.clear();
        loopNoiseQueue.clear();
        aLoopIsClosed = true; // 添加回环检测因子完成标志位
    }

    /**
     * 设置当前帧为关键帧并执行因子图优化
     * 1、计算当前帧与前一帧位姿变换，如果变化太小，不设为关键帧，反之设为关键帧
     * 2、添加激光里程计因子、GPS因子、闭环因子
     * 3、执行因子图优化
     * 4、得到当前帧优化后位姿，位姿协方差
     * 5、添加cloudKeyPoses3D，cloudKeyPoses6D，更新transformTobeMapped，添加当前关键帧的角点、平面点集合
    */
    void saveKeyFramesAndFactor()
    {
        // 计算当前帧与前一帧位姿变换，如果变化太小，不设为关键帧，反之设为关键帧
        // 只对关键帧执行因子图优化
        if (saveFrame() == false)
            return;

        // 下面进入图优化过程
        // 激光里程计因子
        addOdomFactor();

        // GPS因子
        addGPSFactor();

        // 闭环因子
        addLoopFactor();

        // cout << "****************************************************" << endl;
        // gtSAMgraph.print("GTSAM Graph:\n");
        
        // 执行优化
        isam->update(gtSAMgraph, initialEstimate);
        isam->update();// 进行一次额外的优化

        if (aLoopIsClosed == true)// 添加回环检测因子完成标志位
        {
            isam->update();
            isam->update();
            isam->update();
            isam->update();
            isam->update();
        }

        // update之后要清空一下保存的因子图，注：历史数据不会清掉，ISAM保存起来了
        gtSAMgraph.resize(0);
        initialEstimate.clear();

        PointType thisPose3D;
        PointTypePose thisPose6D;
        Pose3 latestEstimate;

        // 优化结果
        // 获取优化后的估计值 
        isamCurrentEstimate = isam->calculateEstimate();
        // 当前帧位姿结果
        // 从经过iSAM2优化后的估计值中提取最新的位姿估计即当前帧位姿
        // isamCurrentEstimate.size()-1 要提取的变量节点的索引
        latestEstimate = isamCurrentEstimate.at<Pose3>(isamCurrentEstimate.size()-1);
        // cout << "****************************************************" << endl;
        // isamCurrentEstimate.print("Current estimate: ");

        // cloudKeyPoses3D加入当前帧位姿
        thisPose3D.x = latestEstimate.translation().x();
        thisPose3D.y = latestEstimate.translation().y();
        thisPose3D.z = latestEstimate.translation().z();
        // 索引
        thisPose3D.intensity = cloudKeyPoses3D->size(); 
        cloudKeyPoses3D->push_back(thisPose3D);

        // cloudKeyPoses6D加入当前帧位姿
        thisPose6D.x = thisPose3D.x;
        thisPose6D.y = thisPose3D.y;
        thisPose6D.z = thisPose3D.z;
        thisPose6D.intensity = thisPose3D.intensity ; 
        thisPose6D.roll  = latestEstimate.rotation().roll();
        thisPose6D.pitch = latestEstimate.rotation().pitch();
        thisPose6D.yaw   = latestEstimate.rotation().yaw();
        thisPose6D.time = timeLaserInfoCur;
        cloudKeyPoses6D->push_back(thisPose6D);

        // cout << "****************************************************" << endl;
        // cout << "Pose covariance:" << endl;
        // cout << isam->marginalCovariance(isamCurrentEstimate.size()-1) << endl << endl;
        // 获取指定变量节点的位姿协方差
        poseCovariance = isam->marginalCovariance(isamCurrentEstimate.size()-1);

        // transformTobeMapped更新当前帧位姿
        transformTobeMapped[0] = latestEstimate.rotation().roll();
        transformTobeMapped[1] = latestEstimate.rotation().pitch();
        transformTobeMapped[2] = latestEstimate.rotation().yaw();
        transformTobeMapped[3] = latestEstimate.translation().x();
        transformTobeMapped[4] = latestEstimate.translation().y();
        transformTobeMapped[5] = latestEstimate.translation().z();

        // 当前帧激光角点、平面点，降采样集合
        pcl::PointCloud<PointType>::Ptr thisCornerKeyFrame(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr thisSurfKeyFrame(new pcl::PointCloud<PointType>());
        pcl::copyPointCloud(*laserCloudCornerLastDS,  *thisCornerKeyFrame);
        pcl::copyPointCloud(*laserCloudSurfLastDS,    *thisSurfKeyFrame);

        // 保存特征点降采样集合
        cornerCloudKeyFrames.push_back(thisCornerKeyFrame);
        surfCloudKeyFrames.push_back(thisSurfKeyFrame);

        // 更新里程计轨迹
        updatePath(thisPose6D);
    }

    /**
     * 更新因子图中所有变量节点的位姿，也就是所有历史关键帧的位姿，更新里程计轨迹
    */
    void correctPoses()
    {
        if (cloudKeyPoses3D->points.empty())
            return;

        if (aLoopIsClosed == true)//当检测到新的回环时
        {
            // 清空局部map
            laserCloudMapContainer.clear();
            // 清空里程计轨迹
            globalPath.poses.clear();
            // 更新因子图中所有变量节点的位姿，也就是所有历史关键帧的位姿
            int numPoses = isamCurrentEstimate.size();
            for (int i = 0; i < numPoses; ++i)
            {
                cloudKeyPoses3D->points[i].x = isamCurrentEstimate.at<Pose3>(i).translation().x();
                cloudKeyPoses3D->points[i].y = isamCurrentEstimate.at<Pose3>(i).translation().y();
                cloudKeyPoses3D->points[i].z = isamCurrentEstimate.at<Pose3>(i).translation().z();

                cloudKeyPoses6D->points[i].x = cloudKeyPoses3D->points[i].x;
                cloudKeyPoses6D->points[i].y = cloudKeyPoses3D->points[i].y;
                cloudKeyPoses6D->points[i].z = cloudKeyPoses3D->points[i].z;
                cloudKeyPoses6D->points[i].roll  = isamCurrentEstimate.at<Pose3>(i).rotation().roll();
                cloudKeyPoses6D->points[i].pitch = isamCurrentEstimate.at<Pose3>(i).rotation().pitch();
                cloudKeyPoses6D->points[i].yaw   = isamCurrentEstimate.at<Pose3>(i).rotation().yaw();

                // 更新里程计轨迹
                updatePath(cloudKeyPoses6D->points[i]);
            }

            aLoopIsClosed = false;
        }
    }

    /**
     * 更新里程计轨迹
    */
    void updatePath(const PointTypePose& pose_in)
    {
        // 创建一个geometry_msgs::PoseStamped类型的消息来存储位姿信息
        geometry_msgs::PoseStamped pose_stamped;
        // 设置消息头部的时间戳
        pose_stamped.header.stamp = ros::Time().fromSec(pose_in.time);
        // 设置消息头部的坐标系
        pose_stamped.header.frame_id = odometryFrame;
        // 设置位姿信息为因子优化得到的当前帧的位姿
        pose_stamped.pose.position.x = pose_in.x;
        pose_stamped.pose.position.y = pose_in.y;
        pose_stamped.pose.position.z = pose_in.z;
        // 创建四元数来表示欧拉角
        tf::Quaternion q = tf::createQuaternionFromRPY(pose_in.roll, pose_in.pitch, pose_in.yaw);
        pose_stamped.pose.orientation.x = q.x();
        pose_stamped.pose.orientation.y = q.y();
        pose_stamped.pose.orientation.z = q.z();
        pose_stamped.pose.orientation.w = q.w();

        // 将构建完成的位姿消息添加到全局路径中
        globalPath.poses.push_back(pose_stamped);
    }

    /**
     * 发布激光里程计
     * pubLaserOdometryGlobal发布的是优化后的odom数据；
     * pubLaserOdometryIncremental则是没有经过优化的里程计数据。
     * 如果没有检测到闭环，则这两个是一样的。
    */
    void publishOdometry()
    {
        // 首先发布对应的里程计值及TF变换
        // 发布激光里程计，odom等价map
        nav_msgs::Odometry laserOdometryROS; // 创建ros中的里程计消息变量
        // 设置该里程计消息的时间戳和父子坐标系
        // 这个激光里程计消息所描述的运动是从odometryFrame到子帧"odom_mapping"的。
        laserOdometryROS.header.stamp = timeLaserInfoStamp;
        laserOdometryROS.header.frame_id = odometryFrame;
        laserOdometryROS.child_frame_id = "odom_mapping";
        // 设置里程计消息的位姿为优化得到的当前帧的位姿
        laserOdometryROS.pose.pose.position.x = transformTobeMapped[3];
        laserOdometryROS.pose.pose.position.y = transformTobeMapped[4];
        laserOdometryROS.pose.pose.position.z = transformTobeMapped[5];
        // 欧拉角转为四元数
        laserOdometryROS.pose.pose.orientation = tf::createQuaternionMsgFromRollPitchYaw(transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);
        // 下面发布的是优化后的里程计数据
        pubLaserOdometryGlobal.publish(laserOdometryROS); // 发布激光里程计消息

        // 发布TF变换消息，odom->lidar
        // 发布里程计到激光雷达的tf变换消息的目的是为了将激光雷达的坐标系（lidar_link）与里程计的坐标系（odom）之间的变换关系通知给其他ROS节点
        // 用于表示lidar相对于odom坐标系的位姿
        // odom坐标系的位置和姿态会随着机器人的移动和旋转而变化
        // "odom"坐标系的原点通常与机器人启动时的位置相对应。
        // 创建一个tf发布对象
        static tf::TransformBroadcaster br; // 用于发布坐标系间的变换关系，其他节点可以订阅这些变换信息以实时更新坐标系之间的关系。
        // 创建tf::Transform对象，表示odom到Lidar之间的坐标变换，该变换包括旋转和平移分量
        tf::Transform t_odom_to_lidar = tf::Transform(tf::createQuaternionFromRPY(transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]),
                                                      tf::Vector3(transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5]));
        // 将创建的tf变换包装成一个时间戳的tf变换，源坐标系：odom，目标坐标系：lidar_link
        // StampedTransform： 这是tf库中的一种变换类型，除了包含坐标变换信息外，还包含时间戳和坐标系信息。
        tf::StampedTransform trans_odom_to_lidar = tf::StampedTransform(t_odom_to_lidar, timeLaserInfoStamp, odometryFrame, "lidar_link");
        // 将包装好的tf变换信息发布出去
        br.sendTransform(trans_odom_to_lidar);

        // Publish odometry for ROS (incremental)
        // 然后发布里程计的增量，针对第一帧关键帧，里程计的增量等于当前帧的位姿
        // 里程计增量消息是否发布标志位
        static bool lastIncreOdomPubFlag = false;
        // 里程计增量消息
        static nav_msgs::Odometry laserOdomIncremental;
        static Eigen::Affine3f increOdomAffine; 

        // 对于第一帧，里程计增量等于当前帧的位姿
        if (lastIncreOdomPubFlag == false)
        {
            lastIncreOdomPubFlag = true;
            laserOdomIncremental = laserOdometryROS;
            // 将当前帧位姿转换为仿射变换矩阵
            increOdomAffine = trans2Affine3f(transformTobeMapped);
        } 
        // 对于其余关键帧，里程计的增量等于当前帧位姿与上一帧位姿的变化量
        else {
            // 计算当前帧与前一帧之间的位姿变换
            Eigen::Affine3f affineIncre = incrementalOdometryAffineFront.inverse() * incrementalOdometryAffineBack;
            // 还是当前帧位姿（打印一下数据，可以看到与激光里程计的位姿transformTobeMapped是一样的）
            // increOdomAffine是所有增量相乘的结果。
            increOdomAffine = increOdomAffine * affineIncre;
            float x, y, z, roll, pitch, yaw;
            // 从仿射变换矩阵中获取平移和欧拉角
            pcl::getTranslationAndEulerAngles (increOdomAffine, x, y, z, roll, pitch, yaw);

            // 点云中是否有Imu数据可用
            if (cloudInfo.imuAvailable == true)
            {
                if (std::abs(cloudInfo.imuPitchInit) < 1.4)
                {
                    double imuWeight = 0.1;
                    tf::Quaternion imuQuaternion;
                    tf::Quaternion transformQuaternion;
                    double rollMid, pitchMid, yawMid;

                    // roll姿态角加权平均
                    transformQuaternion.setRPY(roll, 0, 0);
                    imuQuaternion.setRPY(cloudInfo.imuRollInit, 0, 0);
                    tf::Matrix3x3(transformQuaternion.slerp(imuQuaternion, imuWeight)).getRPY(rollMid, pitchMid, yawMid);
                    roll = rollMid;

                    // pitch姿态角加权平均
                    transformQuaternion.setRPY(0, pitch, 0);
                    imuQuaternion.setRPY(0, cloudInfo.imuPitchInit, 0);
                    tf::Matrix3x3(transformQuaternion.slerp(imuQuaternion, imuWeight)).getRPY(rollMid, pitchMid, yawMid);
                    pitch = pitchMid;
                }
            }
            laserOdomIncremental.header.stamp = timeLaserInfoStamp;
            laserOdomIncremental.header.frame_id = odometryFrame;
            laserOdomIncremental.child_frame_id = "odom_mapping";
            laserOdomIncremental.pose.pose.position.x = x;
            laserOdomIncremental.pose.pose.position.y = y;
            laserOdomIncremental.pose.pose.position.z = z;
            laserOdomIncremental.pose.pose.orientation = tf::createQuaternionMsgFromRollPitchYaw(roll, pitch, yaw);
            if (isDegenerate) // 是否退化
                laserOdomIncremental.pose.covariance[0] = 1;
            else
                laserOdomIncremental.pose.covariance[0] = 0;
        }
        // 这里发布的增量消息是闭环检测优化之前的
        pubLaserOdometryIncremental.publish(laserOdomIncremental); 
    }

    /**
     * 发布里程计、点云、轨迹
     * 1、发布历史关键帧位姿集合
     * 2、发布局部map的降采样平面点集合
     * 3、发布历史帧（累加的）的角点、平面点降采样集合
     * 4、发布里程计轨迹
    */
    void publishFrames()
    {
        if (cloudKeyPoses3D->points.empty())
            return;
        // 发布历史关键帧位姿集合（优化后）
        // publishCloud：utility.h定义函数，将点云数据转换为ros格式的消息，用于话题发布
        publishCloud(pubKeyPoses, cloudKeyPoses3D, timeLaserInfoStamp, odometryFrame);
        // 发布局部map的降采样平面点集合
        publishCloud(pubRecentKeyFrames, laserCloudSurfFromMapDS, timeLaserInfoStamp, odometryFrame);
        
        // 发布历史帧（累加的）的角点、平面点降采样集合
        // 发布关键帧的点云（点云的位姿经过因子图优化） 
        if (pubRecentKeyFrame.getNumSubscribers() != 0)//检查该发布器是否有订阅者
        {
            // pubRecentKeyFrame为之前定义的发布器
            // 创建一个指向PCL点云的指针
            pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());
            // 将位姿数组转换为6D位姿格式
            PointTypePose thisPose6D = trans2PointTypePose(transformTobeMapped);
            // 将当前帧下采样后的角点云数据，应用坐标变换 thisPose6D 后，将结果点云追加到cloudOut中。
            *cloudOut += *transformPointCloud(laserCloudCornerLastDS,  &thisPose6D);
            // 将当前帧降采样后的面点数据叠加到点云中
            *cloudOut += *transformPointCloud(laserCloudSurfLastDS,    &thisPose6D);
            // 最终发布的关键帧点云包含角点和面点，且点云位姿经过优化
            publishCloud(pubRecentKeyFrame, cloudOut, timeLaserInfoStamp, odometryFrame);
        }
        // 发布当前帧原始点云配准之后的点云
        if (pubCloudRegisteredRaw.getNumSubscribers() != 0)
        {
            pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());
            pcl::fromROSMsg(cloudInfo.cloud_deskewed, *cloudOut);
            PointTypePose thisPose6D = trans2PointTypePose(transformTobeMapped);
            *cloudOut = *transformPointCloud(cloudOut,  &thisPose6D);
            publishCloud(pubCloudRegisteredRaw, cloudOut, timeLaserInfoStamp, odometryFrame);
        }
        // 发布里程计轨迹
        if (pubPath.getNumSubscribers() != 0)
        {
            globalPath.header.stamp = timeLaserInfoStamp;
            globalPath.header.frame_id = odometryFrame;
            pubPath.publish(globalPath);
        }
    }
};


int main(int argc, char** argv)
{
    ros::init(argc, argv, "lio_sam");// 初始化ROS节点并传递命令行参数 argc 和 argv，以及节点的名称 "lio_sam"。

    mapOptimization MO; // 实例化对象

    // 使用ROS的日志系统打印一条信息，表示地图优化已经开始
    ROS_INFO("\033[1;32m----> Map Optimization Started.\033[0m");

    // 创建两个独立的线程，并分别绑定到类中的成员函数上
    // 回环检测独立线程
    std::thread loopthread(&mapOptimization::loopClosureThread, &MO);
    // 全局地图可视化线程
    std::thread visualizeMapThread(&mapOptimization::visualizeGlobalMapThread, &MO);

    // 启动了ROS消息循环，使ROS节点可以接收和处理ROS消息。
    // 会一直运行，直到ROS节点被关闭。
    ros::spin();

    // 在ROS消息循环结束后，分别等待回环检测线程和地图可视化线程完成。
    loopthread.join();//join() 函数用于等待线程的执行完成。
    visualizeMapThread.join();

    return 0;//主函数返回0，表示正常退出
}
