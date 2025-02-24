import numpy as np
from filterpy.kalman import ExtendedKalmanFilter as EKF
import matplotlib.pyplot as plt

class RobotEKF(EKF):
    def __init__(self, dt, std_vel, std_gps, std_imu_bearing):
        super().__init__(dim_x=4, dim_z=3)  # State: x, y, theta, velocity | Measurements: x, y, theta (IMU)
        self.dt = dt
        self.std_vel = std_vel
        self.std_gps = std_gps
        self.std_imu_bearing = std_imu_bearing  # Only use IMU bearing noise

        # Covariance matrices
        Q_coef = 1   # Trust in motion model
        P_coef = 0.12 # Initial uncertainty
        R_coef = 0.78  # Measurement noise scaling

        self.x = np.array([[0.0], [0.0], [0.0], [1.0]], dtype=np.float64)  # x, y, theta, velocity
        self.P = np.eye(4) * P_coef  
        self.R = np.diag([std_gps**R_coef, std_gps**R_coef, std_imu_bearing**R_coef])  # Updated to 3x3 for x, y, theta
        self.Q = np.diag([std_vel**Q_coef * dt,  # x process noise
                          std_vel**Q_coef * dt,  # y process noise
                          std_imu_bearing**Q_coef * dt,  # theta process noise
                          std_vel**Q_coef * dt])  # velocity process noise

    def predict(self, u):
        """Predict the next state based on linear and angular velocity"""
        v = u[0]  # Linear velocity
        w = u[1]  # Angular velocity (from IMU)
        theta = self.x[2, 0]

        if abs(w) > 1e-6:
            dx = (v / w) * (np.sin(theta + w * self.dt) - np.sin(theta))
            dy = (v / w) * (-np.cos(theta + w * self.dt) + np.cos(theta))
        else:
            dx = v * np.cos(theta) * self.dt
            dy = v * np.sin(theta) * self.dt
        
        dtheta = w * self.dt

        self.x[0, 0] += dx
        self.x[1, 0] += dy
        self.x[2, 0] += dtheta
        self.x[3, 0] = v  

        F = np.eye(4)
        F[0, 2] = dx
        F[1, 2] = dy

        self.P = F @ self.P @ F.T + self.Q  

    def update(self, z):
        """Update step with GPS (x, y) and IMU bearing (theta) measurements"""
        H = np.eye(3, 4)  # Measurement function mapping state to measurements (3 measurements)
        y = z - H @ self.x  # Difference between the measurement and predicted state
        S = H @ self.P @ H.T + self.R  # Innovation covariance
        K = self.P @ H.T @ np.linalg.inv(S)  # Kalman gain
        self.x = self.x + K @ y  # Update the state estimate
        self.P = self.P - K @ H @ self.P  # Update the covariance

    def nmea_to_lla(self, nmea_sentence):
        """Convert NMEA sentence to latitude, longitude"""
        nmea = nmea_sentence.split(',')
        lat = nmea[2]

        degrees_lat = int(lat // 100)  # Extracts the whole number of degrees
        minutes_lat = lat % 100        # Extracts the minutes (remainder after division by 100)
        decimal_degrees_lat = degrees_lat + (minutes_lat / 60)  # Convert minutes to degrees

        lon = nmea[4]

        degrees_lon = int(lon // 100)  # Extracts the whole number of degrees

        minutes_lon = lon % 100        # Extracts the minutes (remainder after division by 100)
        #LLA TO ENU WORKS WHEN LON IS NEGATIVE
        decimal_degrees_lon = -1*(degrees_lon + (minutes_lon / 60))  # Convert minutes to degrees

        return decimal_degrees_lat, decimal_degrees_lon

    def lla_to_enu(self,lat, lon, ref_lat, ref_lon):
        ###INPUT LONG SHOULD BE NEGATIVE TO WORK CORRECTLY (SEE NMEA TO LLA FUNCTION)
        """
        Convert Latitude, Longitude, Altitude (LLA) to East, North, Up (ENU) coordinates.
        
        Parameters:
            lat (float): Latitude in degrees
            lon (float): Longitude in degrees
            ref_lat (float): Reference latitude in degrees
            ref_lon (float): Reference longitude in degrees
        
        Returns:
            tuple: (east, north) / (x, y) in meters
        """
        EARTH_RADIUS = 6378137.0  # Earth's radius in meters
        DEG_TO_RAD = np.pi / 180.0  # Conversion factor from degrees to radians

        delta_lat = lat - ref_lat
        delta_lon = lon - ref_lon

        cos_ref_lat = np.cos(ref_lat * DEG_TO_RAD)

        east = delta_lon * DEG_TO_RAD * EARTH_RADIUS * cos_ref_lat
        north = delta_lat * DEG_TO_RAD * EARTH_RADIUS
        

        return east, north
    

    #Example for how to use this class
#     def run_localization(std_vel, std_gps, std_imu_bearing):
    #     dt = 0.5  
    #     ekf = RobotEKF(dt, std_vel, std_gps, std_imu_bearing)
        
    #     ekf.x = np.array([[2.0], [3.0], [np.pi/4], [1.0]])  # Initial state
    #     ekf.P = np.eye(4) * 0.1  # Initial uncertainty

    #     u = np.array([1.0, 0.2])  # Linear velocity = 1 m/s, Angular velocity = 0.2 rad/s
        
    #     actual_path = []
    #     num_steps = 50  # Number of time steps to simulate

    #     # Simulate robot movement
    #     for _ in range(num_steps):
    #         ekf.predict(u)
    #         actual_path.append(ekf.x.flatten())

    #     gps_measurements = []
    #     # Generate noisy GPS and IMU measurements
    #     for pos in actual_path:
    #         noisy_gps = pos[:2] + np.random.normal(0, std_gps, size=2)  # Add noise to GPS position
    #         noisy_bearing = pos[2] + np.random.normal(0, std_imu_bearing)  # Add noise to IMU bearing
    #         gps_measurements.append(np.array([*noisy_gps, noisy_bearing]))  # Only one bearing now

    #     estimated_path = []
    #     bearing_differences = []  # Store bearing errors

    #     # Run the Kalman filter with noisy measurements
    #     for i, gps_measurement in enumerate(gps_measurements):
    #         ekf.predict(u)
    #         ekf.update(gps_measurement.reshape(-1, 1))  # Update the state estimate with noisy measurements
    #         estimated_path.append(ekf.x.flatten())

    #         # Compute bearing error (absolute difference)
    #         bearing_error = abs(actual_path[i][2] - ekf.x[2, 0])
    #         bearing_differences.append(bearing_error)

    #     # Convert lists to numpy arrays for easier plotting
    #     actual_path = np.array(actual_path)
    #     gps_measurements = np.array(gps_measurements)
    #     estimated_path = np.array(estimated_path)

    #     # Plot the results
    #     plt.figure()
    #     plt.plot(actual_path[:, 0], actual_path[:, 1], 'g-', label='Actual Path')
    #     plt.plot(gps_measurements[:, 0], gps_measurements[:, 1], 'rx', label='GPS Measurements')
    #     plt.plot(estimated_path[:, 0], estimated_path[:, 1], 'b-', label='EKF Estimated Path')
    #     plt.legend()
    #     plt.xlabel('X position')
    #     plt.ylabel('Y position')
    #     plt.title('EKF Localization with IMU Bearing')

    #     # Calculate and print average localization error
    #     distance_differences = [
    #         np.sqrt((actual_path[i, 0] - estimated_path[i, 0])**2 + (actual_path[i, 1] - estimated_path[i, 1])**2)
    #         for i in range(num_steps)
    #     ]
    #     avg_pos_error = np.mean(distance_differences)
    #     avg_bearing_error = np.mean(bearing_differences)

    #     plt.show()
    #     print("Avg position error:", avg_pos_error, "meters")
    #     print("Avg bearing error:", avg_bearing_error, "radians")
    #     return avg_pos_error, avg_bearing_error

    # # Test with some standard deviations for the sensor noise
    # avg_pos_error, avg_bearing_error = run_localization(std_vel=0.1, std_gps=2, std_imu_bearing=0.05)
