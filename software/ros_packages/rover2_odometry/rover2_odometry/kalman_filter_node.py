import rclpy
from rclpy.node import Node

from std_msgs.msg import String
from nmea_msgs.msg import Sentence  # For GPS data
from sensor_msgs.msg import Imu     # For IMU data


import tf_transformations


import EKF_Class

#Set EKF variables
dt = 0.1    #Time step
std_vel = 0.1  #Velocity noise
std_gps = 2  #GPS noise (meters)
std_imu_bearing = 0.1 #IMU bearing noise
ekf = EKF_Class.RobotEKF(dt, std_vel, std_gps, std_imu_bearing)


class EKF_Node(Node):

    def __init__(self):
        super().__init__('ekf_node')
        self.imu_sub = self.create_subscription(Imu, 'imu/data', self.set_imu_val, 10)
        self.gps_sub = self.create_subscription(Sentence, 'gps/sentence', self.set_gps_val, 10)

        self.publisher_ = self.create_publisher(String, 'updated_ekf_position', 10)
       
        self.imu_data = None
        self.gps_data = None

    def set_imu_val(self,msg):
        self.imu_data = msg.data
        self.get_logger().info('IMU Data: "%s"' % msg.data)
    
    def set_gps_val(self,msg):
        self.gps_data = msg.data
        self.get_logger().info('GPS Data: "%s"' % msg.data)

    def update_ekf(self):
        if self.imu_data is not None and self.gps_data is not None:
            self.get_logger().info('Updating EKF')
        
            #___1. Get bearing and angular velocity from imu data___
            imu = self.imu_data #Retrieve imu data
            #Angular v:
            angular_velocity = imu.angular_velocity.z

            #Bearing:
            quaternion = (imu.orientation.x, imu.orientation.y, imu.orientation.z, imu.orientation.w)
            euler = tf_transformations.euler_from_quaternion(quaternion)
            bearing = euler[2]  #Bearing in radians

            

            #___2.  Convert gps data to x and y (East, North, Up from original position)___
            gps = self.gps_data  #Retrieve gps data
            #TODO: NEED TO PASS THE ORIGINAL GPS POSITION TO GET REF COORDS IN LLA FORMAT
            #Can store the first gps reading maybe??
            #ref_coords = %coords in NMEA format
            #ref_coords_lla = ekf.nmea_to_lla(ref_coords)

        
            #Need to convert to get it to lla so we can run enu func
        
            gps_lla = ekf.nmea_to_lla(gps) #Takes in full sentence, converts to lat/long in decimal form

            #Convert to ENU
            #gps_enu = ekf.lla_to_enu(gps_lla[0], gps_lla[1], ref_coords_lla[0], ref_coords_lla[1])
            x = gps_enu[0]
            y = gps_enu[1]

        
            #___3. Update EKF with gps and imu data___

            # TODO: 1. NEED TO SET ekf.x at the beginning TO INCLUDE THE INITIAL VELOCITY (look at run_localization example function in EKF_Class)
            # 2. AND ALSO FIGURE OUT HOW to SUBSCRIBE TO CONTROL VELOCITY 
            u = [linear_velocity, angular_velocity]
            z = [x, y, bearing]
            ekf.predict(u)
            ekf.update(z)
            
            #___4. Publish updated position___
            #get inromation array using ekf.x - 
            self.publisher_.publish(String(data=str(ekf.x)))



def main(args=None):
    rclpy.init(args=args)

    ekf_node = EKF_Node()

    rclpy.spin(ekf_node)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    ekf_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()