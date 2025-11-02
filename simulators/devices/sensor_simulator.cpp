#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <cmath>
#include <string>
#include <thread>

class SensorSimulator {
private:
    std::string sensor_type;
    double sampling_rate_hz;
    int resolution_bits;
    double accuracy_percent;
    
    // 仿真数据
    std::vector<double> sensor_data;
    std::random_device rd;
    std::mt19937 gen;
    
public:
    SensorSimulator(const std::string& type, double rate, int bits, double accuracy)
        : sensor_type(type), sampling_rate_hz(rate), resolution_bits(bits), 
          accuracy_percent(accuracy), gen(rd()) {}
    
    void simulate_imu_sensor() {
        std::uniform_real_distribution<> noise(-0.01, 0.01);
        
        // 模拟IMU数据（加速度计、陀螺仪、磁力计）
        for (int i = 0; i < 1000; i++) {
            // 加速度计数据 (m/s²)
            double accel_x = sin(i * 0.1) + noise(gen);
            double accel_y = cos(i * 0.1) + noise(gen);
            double accel_z = 9.81 + noise(gen); // 重力加速度
            
            // 陀螺仪数据 (rad/s)
            double gyro_x = 0.1 * sin(i * 0.05) + noise(gen);
            double gyro_y = 0.1 * cos(i * 0.05) + noise(gen);
            double gyro_z = 0.05 * sin(i * 0.02) + noise(gen);
            
            // 磁力计数据 (μT)
            double mag_x = 25.0 + 5.0 * sin(i * 0.01) + noise(gen);
            double mag_y = 30.0 + 5.0 * cos(i * 0.01) + noise(gen);
            double mag_z = 40.0 + noise(gen);
            
            sensor_data.push_back(sqrt(accel_x*accel_x + accel_y*accel_y + accel_z*accel_z));
            
            // 模拟采样延迟
            std::this_thread::sleep_for(std::chrono::microseconds(static_cast<int>(1000000 / sampling_rate_hz)));
        }
    }
    
    void simulate_environmental_sensor() {
        std::uniform_real_distribution<> temp_noise(-0.5, 0.5);
        std::uniform_real_distribution<> pressure_noise(-10, 10);
        
        // 模拟环境传感器数据
        for (int i = 0; i < 500; i++) {
            // 温度传感器 (°C)
            double temperature = 25.0 + 10.0 * sin(i * 0.01) + temp_noise(gen);
            
            // 压力传感器 (Pa)
            double pressure = 101325.0 + 1000.0 * cos(i * 0.02) + pressure_noise(gen);
            
            // 湿度传感器 (%)
            double humidity = 50.0 + 20.0 * sin(i * 0.005) + temp_noise(gen);
            
            sensor_data.push_back(temperature);
            
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }
    
    void simulate_optical_sensor() {
        std::uniform_real_distribution<> noise(-0.1, 0.1);
        
        // 模拟光学传感器数据（星敏感器）
        for (int i = 0; i < 200; i++) {
            // 星点位置 (像素坐标)
            double star_x = 512 + 100 * sin(i * 0.1) + noise(gen);
            double star_y = 512 + 100 * cos(i * 0.1) + noise(gen);
            
            // 星等 (亮度)
            double magnitude = 5.0 + 2.0 * sin(i * 0.05) + noise(gen);
            
            sensor_data.push_back(magnitude);
            
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
    }
    
    void simulate_radiation_sensor() {
        std::uniform_real_distribution<> noise(-0.01, 0.01);
        std::exponential_distribution<> radiation_events(0.1);
        
        // 模拟辐射传感器数据
        for (int i = 0; i < 300; i++) {
            // 辐射剂量率 (mGy/h)
            double dose_rate = 0.1 + radiation_events(gen) + noise(gen);
            
            // 粒子计数
            int particle_count = static_cast<int>(dose_rate * 1000 + noise(gen) * 10);
            
            sensor_data.push_back(dose_rate);
            
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
    }
    
    double calculate_power_consumption() {
        // 根据传感器类型和采样率计算功耗
        double base_power = 10.0; // mW
        
        if (sensor_type == "IMU") {
            base_power = 15.0;
        } else if (sensor_type == "environmental") {
            base_power = 8.0;
        } else if (sensor_type == "optical") {
            base_power = 50.0; // 光学传感器功耗较高
        } else if (sensor_type == "radiation") {
            base_power = 20.0;
        }
        
        // 采样率影响功耗
        double rate_factor = sampling_rate_hz / 1000.0;
        return base_power * (1.0 + rate_factor);
    }
    
    void run_simulation() {
        std::cout << "🔍 启动传感器仿真: " << sensor_type << std::endl;
        std::cout << "采样率: " << sampling_rate_hz << " Hz" << std::endl;
        std::cout << "分辨率: " << resolution_bits << " bits" << std::endl;
        std::cout << "精度: " << accuracy_percent << "%" << std::endl;
        
        auto start = std::chrono::high_resolution_clock::now();
        
        if (sensor_type == "IMU") {
            simulate_imu_sensor();
        } else if (sensor_type == "environmental") {
            simulate_environmental_sensor();
        } else if (sensor_type == "optical") {
            simulate_optical_sensor();
        } else if (sensor_type == "radiation") {
            simulate_radiation_sensor();
        } else {
            std::cerr << "未知传感器类型: " << sensor_type << std::endl;
            return;
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        // 计算统计信息
        double mean = 0.0, variance = 0.0;
        for (double value : sensor_data) {
            mean += value;
        }
        mean /= sensor_data.size();
        
        for (double value : sensor_data) {
            variance += (value - mean) * (value - mean);
        }
        variance /= sensor_data.size();
        
        double power_consumption = calculate_power_consumption();
        
        std::cout << "✅ 传感器仿真完成" << std::endl;
        std::cout << "执行时间: " << duration.count() << " ms" << std::endl;
        std::cout << "数据点数: " << sensor_data.size() << std::endl;
        std::cout << "平均值: " << mean << std::endl;
        std::cout << "方差: " << variance << std::endl;
        std::cout << "功耗: " << power_consumption << " mW" << std::endl;
        
        // 输出JSON格式结果
        std::cout << "{" << std::endl;
        std::cout << "  \"sensor_type\": \"" << sensor_type << "\"," << std::endl;
        std::cout << "  \"sampling_rate_hz\": " << sampling_rate_hz << "," << std::endl;
        std::cout << "  \"resolution_bits\": " << resolution_bits << "," << std::endl;
        std::cout << "  \"accuracy_percent\": " << accuracy_percent << "," << std::endl;
        std::cout << "  \"execution_time_ms\": " << duration.count() << "," << std::endl;
        std::cout << "  \"data_points\": " << sensor_data.size() << "," << std::endl;
        std::cout << "  \"mean_value\": " << mean << "," << std::endl;
        std::cout << "  \"variance\": " << variance << "," << std::endl;
        std::cout << "  \"power_consumption_mw\": " << power_consumption << std::endl;
        std::cout << "}" << std::endl;
    }
};

int main(int argc, char* argv[]) {
    if (argc != 5) {
        std::cerr << "用法: " << argv[0] << " <sensor_type> <sampling_rate_hz> <resolution_bits> <accuracy_percent>" << std::endl;
        std::cerr << "传感器类型: IMU, environmental, optical, radiation" << std::endl;
        return 1;
    }
    
    std::string sensor_type = argv[1];
    double sampling_rate_hz = std::stod(argv[2]);
    int resolution_bits = std::stoi(argv[3]);
    double accuracy_percent = std::stod(argv[4]);
    
    SensorSimulator simulator(sensor_type, sampling_rate_hz, resolution_bits, accuracy_percent);
    simulator.run_simulation();
    
    return 0;
}