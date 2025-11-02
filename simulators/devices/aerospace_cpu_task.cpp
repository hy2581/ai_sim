#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <cmath>
#include <omp.h>
#include <string>

#ifdef AEROSPACE_FEATURES
#include <atomic>
#include <thread>
#endif

class AerospaceCPUTask {
private:
    int task_id;
    int iterations;
    int data_size;
    bool radiation_hardening;
    double seu_tolerance;
    
public:
    // 航空航天特性 - 改为public以便访问
    std::atomic<int> seu_events{0};
    std::atomic<int> corrected_errors{0};
    double aging_factor = 1.0;
    double temperature_factor = 1.0;
    
public:
    AerospaceCPUTask(int id, int iter, int size, bool rad_hard, double seu_tol) 
        : task_id(id), iterations(iter), data_size(size), 
          radiation_hardening(rad_hard), seu_tolerance(seu_tol) {}
    
    void simulate_radiation_effects() {
        #ifdef AEROSPACE_FEATURES
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0.0, 1.0);
        
        // 模拟单粒子翻转事件
        if (dis(gen) < seu_tolerance) {
            seu_events++;
            if (radiation_hardening) {
                // 辐射硬化设计可以纠正大部分错误
                if (dis(gen) < 0.95) {
                    corrected_errors++;
                }
            }
        }
        #endif
    }
    
    void simulate_aging_effects(double runtime_hours) {
        // NBTI/PBTI老化模型
        aging_factor = 1.0 - (runtime_hours / 100000.0) * 0.1; // 10万小时退化10%
        aging_factor = std::max(0.8, aging_factor); // 最大退化20%
    }
    
    void simulate_temperature_effects(double temp_c) {
        // 温度对性能的影响
        if (temp_c < -40) {
            temperature_factor = 0.8; // 低温降频
        } else if (temp_c > 85) {
            temperature_factor = 0.7; // 高温降频
        } else {
            temperature_factor = 1.0;
        }
    }
    
    double execute_compute_intensive_task() {
        auto start = std::chrono::high_resolution_clock::now();
        
        std::vector<double> data(data_size);
        std::vector<double> result(data_size);
        
        // 初始化数据
        #pragma omp parallel for
        for (int i = 0; i < data_size; i++) {
            data[i] = sin(i * 0.01) + cos(i * 0.02);
        }
        
        // 主计算循环
        for (int iter = 0; iter < iterations; iter++) {
            // 模拟航空航天环境效应
            if (iter % 1000 == 0) {
                simulate_radiation_effects();
                simulate_aging_effects(iter / 1000.0);
                simulate_temperature_effects(25.0 + (iter % 100) - 50); // 温度变化
            }
            
            // 计算密集型操作（考虑老化和温度影响）
            double effective_performance = aging_factor * temperature_factor;
            int effective_iterations = static_cast<int>(iterations * effective_performance);
            
            #pragma omp parallel for
            for (int i = 0; i < data_size; i++) {
                // 复杂数学运算
                result[i] = sqrt(data[i] * data[i] + 1.0);
                result[i] = log(result[i] + 1.0);
                result[i] = exp(result[i] * 0.1);
                
                // 模拟内存访问模式
                if (i > 0) {
                    result[i] += result[i-1] * 0.1;
                }
            }
            
            // 数据依赖计算
            double sum = 0.0;
            for (int i = 0; i < data_size; i++) {
                sum += result[i];
                data[i] = sum * 0.001; // 反馈到下一轮
            }
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        return duration.count() / 1000.0; // 返回毫秒
    }
    
    void print_aerospace_metrics() {
        std::cout << "=== 航空航天CPU性能指标 ===" << std::endl;
        std::cout << "任务ID: " << task_id << std::endl;
        std::cout << "辐射硬化: " << (radiation_hardening ? "启用" : "禁用") << std::endl;
        std::cout << "SEU事件数: " << seu_events.load() << std::endl;
        std::cout << "纠错次数: " << corrected_errors.load() << std::endl;
        std::cout << "老化因子: " << aging_factor << std::endl;
        std::cout << "温度因子: " << temperature_factor << std::endl;
        std::cout << "错误率: " << (seu_events > 0 ? (1.0 - (double)corrected_errors / seu_events) : 0.0) << std::endl;
    }
};

int main(int argc, char* argv[]) {
    if (argc != 6) {
        std::cerr << "用法: " << argv[0] << " <task_id> <iterations> <data_size> <radiation_hardening> <seu_tolerance>" << std::endl;
        return 1;
    }
    
    int task_id = std::stoi(argv[1]);
    int iterations = std::stoi(argv[2]);
    int data_size = std::stoi(argv[3]);
    bool radiation_hardening = (std::string(argv[4]) == "true");
    double seu_tolerance = std::stod(argv[5]);
    
    std::cout << "🚀 启动航空航天CPU任务 " << task_id << std::endl;
    std::cout << "配置: " << iterations << " 迭代, " << data_size << " 数据大小" << std::endl;
    std::cout << "辐射硬化: " << (radiation_hardening ? "启用" : "禁用") << std::endl;
    
    AerospaceCPUTask task(task_id, iterations, data_size, radiation_hardening, seu_tolerance);
    
    double execution_time = task.execute_compute_intensive_task();
    
    std::cout << "✅ 任务完成，执行时间: " << execution_time << " ms" << std::endl;
    
    task.print_aerospace_metrics();
    
    // 输出性能数据（JSON格式）
    std::cout << "{" << std::endl;
    std::cout << "  \"task_id\": " << task_id << "," << std::endl;
    std::cout << "  \"execution_time_ms\": " << execution_time << "," << std::endl;
    std::cout << "  \"iterations\": " << iterations << "," << std::endl;
    std::cout << "  \"data_size\": " << data_size << "," << std::endl;
    std::cout << "  \"radiation_hardening\": " << (radiation_hardening ? "true" : "false") << "," << std::endl;
    std::cout << "  \"seu_events\": " << task.seu_events.load() << "," << std::endl;
    std::cout << "  \"corrected_errors\": " << task.corrected_errors.load() << "," << std::endl;
    std::cout << "  \"performance_degradation\": " << (1.0 - task.aging_factor * task.temperature_factor) << std::endl;
    std::cout << "}" << std::endl;
    
    return 0;
}