#!/usr/bin/env python3
import rospy
import numpy as np
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from cv_bridge import CvBridge
import cv2

class DepthToPointCloud:
    def __init__(self):
        rospy.init_node('depth_to_pointcloud', anonymous=True)
        
        # 参数设置
        self.camera_frame = rospy.get_param("~camera_frame", "camera_depth_optical_frame")
        self.depth_topic = rospy.get_param("~depth_topic", "/camera/depth/image_raw")
        self.camera_info_topic = rospy.get_param("~camera_info_topic", "/camera/depth/camera_info")
        self.pointcloud_topic = rospy.get_param("~output_topic", "/pointcloud")
        
        # 初始化变量
        self.bridge = CvBridge()
        self.K = None  # 相机内参矩阵
        self.depth_scale = 1000.0  # 深度图缩放因子（假设深度图单位为毫米）
        
        # 订阅器和发布器
        rospy.Subscriber(self.depth_topic, Image, self.depth_callback)
        rospy.Subscriber(self.camera_info_topic, CameraInfo, self.camera_info_callback)
        self.pc_pub = rospy.Publisher(self.pointcloud_topic, PointCloud2, queue_size=10)
        
        rospy.loginfo("Depth to PointCloud转换器已启动，等待数据...")

    def camera_info_callback(self, msg):
        """获取相机内参"""
        if self.K is None:
            self.K = np.array(msg.K).reshape(3, 3)
            rospy.loginfo(f"相机内参矩阵:\n{self.K}")

    def depth_callback(self, msg):
        """深度图回调函数"""
        if self.K is None:
            rospy.logwarn_once("等待相机内参...")
            return
            
        try:
            # 转换为OpenCV格式的深度图
            depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            
            # 生成点云
            points = self.depth_to_points(depth_image)
            
            # 发布PointCloud2
            self.publish_pointcloud(points)
            
        except Exception as e:
            rospy.logerr(f"处理深度图时出错: {e}")

    def depth_to_points(self, depth_image):
        """将深度图转换为相机坐标系下的点云"""
        height, width = depth_image.shape
        u = np.arange(width)
        v = np.arange(height)
        u, v = np.meshgrid(u, v)
        
        # 转换为相机坐标系 (Z = depth / scale)
        Z = depth_image.astype(np.float32) / self.depth_scale
        valid = Z > 0  # 过滤无效深度
        
        # 计算X,Y坐标
        X = (u[valid] - self.K[0, 2]) * Z[valid] / self.K[0, 0]
        Y = (v[valid] - self.K[1, 2]) * Z[valid] / self.K[1, 1]
        Z_valid = Z[valid]
        
        # 合并为Nx3数组
        return np.column_stack((X, Y, Z_valid))

    def publish_pointcloud(self, points):
        """发布PointCloud2消息"""
        cloud = PointCloud2()
        cloud.header.stamp = rospy.Time.now()
        cloud.header.frame_id = self.camera_frame
        
        # 设置点云字段（x,y,z）
        cloud.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1)
        ]
        
        cloud.height = 1
        cloud.width = len(points)
        cloud.is_bigendian = False
        cloud.point_step = 12  # 3个float32 = 12字节
        cloud.row_step = cloud.point_step * cloud.width
        cloud.is_dense = True
        
        # 填充数据
        cloud.data = np.asarray(points, dtype=np.float32).tobytes()
        
        self.pc_pub.publish(cloud)

if __name__ == '__main__':
    try:
        converter = DepthToPointCloud()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass