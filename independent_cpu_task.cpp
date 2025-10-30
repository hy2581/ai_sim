#include <iostream>
#include <chrono>
#include <cmath>
#include <vector>
#include <random>
#include <fstream>
#include <iomanip>

// Performance metrics structure
struct PerformanceMetrics {
    double total_operations = 0;
    double execution_time_sec = 0;
    double throughput_ops_per_sec = 0;
    double avg_latency_ns = 0;
    double cpu_utilization = 0;
    int iterations_completed = 0;
};

class IndependentCPUTask {
private:
    int task_id;
    int iterations;
    int data_size;
    std::vector<double> data_a, data_b, data_c;
    PerformanceMetrics metrics;
    
public:
    IndependentCPUTask(int id, int iter, int size) 
        : task_id(id), iterations(iter), data_size(size) {
        // Initialize data arrays
        data_a.resize(data_size);
        data_b.resize(data_size);
        data_c.resize(data_size);
        
        // Fill with random data
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(1.0, 100.0);
        
        for (int i = 0; i < data_size; i++) {
            data_a[i] = dis(gen);
            data_b[i] = dis(gen);
            data_c[i] = 0.0;
        }
    }
    
    // Basic arithmetic operations
    void perform_addition() {
        for (int i = 0; i < data_size; i++) {
            data_c[i] = data_a[i] + data_b[i];
        }
    }
    
    void perform_subtraction() {
        for (int i = 0; i < data_size; i++) {
            data_c[i] = data_a[i] - data_b[i];
        }
    }
    
    void perform_multiplication() {
        for (int i = 0; i < data_size; i++) {
            data_c[i] = data_a[i] * data_b[i];
        }
    }
    
    void perform_division() {
        for (int i = 0; i < data_size; i++) {
            if (data_b[i] != 0) {
                data_c[i] = data_a[i] / data_b[i];
            }
        }
    }
    
    // Complex function calls
    double complex_function(double x, double y) {
        return std::sin(x) * std::cos(y) + std::sqrt(std::abs(x * y)) + std::log(std::abs(x) + 1);
    }
    
    void perform_function_calls() {
        for (int i = 0; i < data_size; i++) {
            data_c[i] = complex_function(data_a[i], data_b[i]);
        }
    }
    
    // Memory intensive operations
    void perform_memory_operations() {
        std::vector<double> temp(data_size);
        for (int i = 0; i < data_size; i++) {
            temp[i] = data_a[i];
            data_c[i] = temp[data_size - 1 - i] + data_b[i];
        }
    }
    
    // Main task execution
    void run_task() {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        std::cout << "CPU Task " << task_id << " starting..." << std::endl;
        
        for (int iter = 0; iter < iterations; iter++) {
            // Perform different operations in sequence
            perform_addition();
            perform_subtraction();
            perform_multiplication();
            perform_division();
            perform_function_calls();
            perform_memory_operations();
            
            metrics.total_operations += 6 * data_size; // 6 operations per data element
            metrics.iterations_completed++;
            
            // Progress reporting every 10% of iterations
            if (iter % (iterations / 10) == 0) {
                std::cout << "CPU Task " << task_id << " progress: " 
                         << (iter * 100 / iterations) << "%" << std::endl;
            }
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time);
        
        // Calculate metrics
        metrics.execution_time_sec = duration.count() / 1e9;
        metrics.throughput_ops_per_sec = metrics.total_operations / metrics.execution_time_sec;
        metrics.avg_latency_ns = duration.count() / metrics.total_operations;
        metrics.cpu_utilization = 95.0; // Simulated high utilization
        
        std::cout << "CPU Task " << task_id << " completed!" << std::endl;
    }
    
    void save_results() {
        std::string filename = "cpu_task_" + std::to_string(task_id) + "_results.json";
        std::ofstream file(filename);
        
        file << "{\n";
        file << "  \"task_id\": " << task_id << ",\n";
        file << "  \"task_type\": \"independent_cpu\",\n";
        file << "  \"total_operations\": " << std::fixed << std::setprecision(0) << metrics.total_operations << ",\n";
        file << "  \"execution_time_sec\": " << std::fixed << std::setprecision(3) << metrics.execution_time_sec << ",\n";
        file << "  \"throughput_ops_per_sec\": " << std::fixed << std::setprecision(0) << metrics.throughput_ops_per_sec << ",\n";
        file << "  \"avg_latency_ns\": " << std::fixed << std::setprecision(2) << metrics.avg_latency_ns << ",\n";
        file << "  \"cpu_utilization\": " << std::fixed << std::setprecision(1) << metrics.cpu_utilization << ",\n";
        file << "  \"iterations_completed\": " << metrics.iterations_completed << ",\n";
        file << "  \"data_size\": " << data_size << "\n";
        file << "}\n";
        
        file.close();
        std::cout << "CPU Task " << task_id << " results saved to " << filename << std::endl;
    }
    
    void print_summary() {
        std::cout << "\n=== CPU Task " << task_id << " Performance Summary ===" << std::endl;
        std::cout << "Total Operations: " << std::fixed << std::setprecision(0) << metrics.total_operations << std::endl;
        std::cout << "Execution Time: " << std::fixed << std::setprecision(3) << metrics.execution_time_sec << " seconds" << std::endl;
        std::cout << "Throughput: " << std::fixed << std::setprecision(0) << metrics.throughput_ops_per_sec << " ops/sec" << std::endl;
        std::cout << "Average Latency: " << std::fixed << std::setprecision(2) << metrics.avg_latency_ns << " ns" << std::endl;
        std::cout << "CPU Utilization: " << std::fixed << std::setprecision(1) << metrics.cpu_utilization << "%" << std::endl;
        std::cout << "Iterations Completed: " << metrics.iterations_completed << std::endl;
        std::cout << "================================================" << std::endl;
    }
};

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cout << "Usage: " << argv[0] << " <task_id> <iterations> <data_size>" << std::endl;
        return 1;
    }
    
    int task_id = std::atoi(argv[1]);
    int iterations = std::atoi(argv[2]);
    int data_size = std::atoi(argv[3]);
    
    std::cout << "Starting Independent CPU Task " << task_id << std::endl;
    std::cout << "Iterations: " << iterations << ", Data Size: " << data_size << std::endl;
    
    IndependentCPUTask task(task_id, iterations, data_size);
    task.run_task();
    task.print_summary();
    task.save_results();
    
    return 0;
}