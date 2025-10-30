#!/usr/bin/env python3
"""
Independent Chiplet Simulation Runner
Orchestrates independent chiplet performance testing and analysis
"""

import os
import sys
import json
import subprocess
import time
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
import matplotlib.pyplot as plt
from independent_chiplet_parser import IndependentChipletParser, SystemConfig
from independent_sim_generator import IndependentSimulationGenerator


class IndependentSimulationRunner:
    def __init__(self, simulator_root: str):
        self.simulator_root = Path(simulator_root)
        self.benchmark_root = self.simulator_root / "benchmark" / "ai_sim_gen"
        
    def setup_environment(self) -> bool:
        """Setup simulation environment"""
        try:
            # Set environment variables
            os.environ['SIMULATOR_ROOT'] = str(self.simulator_root)
            os.environ['BENCHMARK_ROOT'] = str(self.benchmark_root)
            
            return True
        except Exception as e:
            print(f"Error setting up environment: {e}")
            return False
    
    def build_benchmark(self) -> bool:
        """Build the independent benchmark binaries"""
        try:
            os.chdir(self.benchmark_root)
            
            # Run make
            result = subprocess.run(['make', '-j4'], 
                                  capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"Build failed: {result.stderr}")
                return False
                
            print("Independent chiplet tasks built successfully")
            return True
            
        except Exception as e:
            print(f"Error building benchmark: {e}")
            return False
    
    def run_simulation(self, yaml_file: str) -> bool:
        """Run the independent chiplet simulation"""
        try:
            os.chdir(self.benchmark_root)
            
            # Run interchiplet for independent tasks
            interchiplet_bin = self.simulator_root / "interchiplet" / "bin" / "interchiplet"
            if not interchiplet_bin.exists():
                print(f"Error: interchiplet binary not found at {interchiplet_bin}")
                return False
            
            cmd = [str(interchiplet_bin), yaml_file]
            print(f"Running independent chiplet simulation: {' '.join(cmd)}")
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"Simulation failed: {result.stderr}")
                return False
                
            print("Independent chiplet simulation completed successfully")
            return True
            
        except Exception as e:
            print(f"Error running simulation: {e}")
            return False
    
    def analyze_results(self) -> Dict[str, Any]:
        """Analyze independent chiplet simulation results"""
        results = {
            "status": "success",
            "metrics": {},
            "logs": [],
            "errors": [],
            "performance_summary": {}
        }
        
        try:
            os.chdir(self.benchmark_root)
            
            # Find result directories
            proc_dirs = list(Path('.').glob('proc_r*_p*_t*'))
            
            if not proc_dirs:
                results["status"] = "no_results"
                results["errors"].append("No simulation result directories found")
                return results
            
            # Analyze each process directory
            for proc_dir in proc_dirs:
                proc_name = proc_dir.name
                results["logs"].append(proc_name)
                
                # Look for result JSON files in the process directory
                json_files = list(proc_dir.glob('*_task_*_results.json'))
                for json_file in json_files:
                    try:
                        with open(json_file, 'r') as f:
                            task_result = json.load(f)
                            results["metrics"][f"{proc_name}_{json_file.stem}"] = task_result
                    except Exception as e:
                        results["errors"].append(f"Error reading {json_file}: {e}")
                
                # Look for log files
                log_files = list(proc_dir.glob('*.log'))
                for log_file in log_files:
                    log_type = log_file.stem
                    
                    # Parse different types of logs
                    if 'cpu_independent' in log_type:
                        cpu_metrics = self._parse_cpu_log(log_file)
                        results["metrics"][f"{proc_name}_{log_type}"] = cpu_metrics
                    elif 'gpu_task' in log_type:
                        gpu_metrics = self._parse_gpu_log(log_file)
                        results["metrics"][f"{proc_name}_{log_type}"] = gpu_metrics
            
            # Generate performance summary
            results["performance_summary"] = self._generate_performance_summary(results["metrics"])
            
        except Exception as e:
            results["status"] = "error"
            results["errors"].append(f"Analysis failed: {e}")
        
        return results
    
    def _parse_cpu_log(self, log_file: Path) -> Dict[str, Any]:
        """Parse CPU task log file"""
        metrics = {"type": "independent_cpu", "found": False}
        
        try:
            with open(log_file, 'r') as f:
                content = f.read()
                
            # Look for performance indicators
            if "completed" in content.lower():
                metrics["found"] = True
                lines = content.split('\n')
                for line in lines:
                    if "Total Operations" in line:
                        parts = line.split(':')
                        if len(parts) > 1:
                            metrics["total_operations"] = parts[1].strip()
                    elif "Throughput" in line:
                        parts = line.split(':')
                        if len(parts) > 1:
                            metrics["throughput"] = parts[1].strip()
                    elif "CPU Utilization" in line:
                        parts = line.split(':')
                        if len(parts) > 1:
                            metrics["utilization"] = parts[1].strip()
                            
        except Exception as e:
            metrics["error"] = str(e)
            
        return metrics
    
    def _parse_gpu_log(self, log_file: Path) -> Dict[str, Any]:
        """Parse GPU task log file"""
        metrics = {"type": "independent_gpu", "found": False}
        
        try:
            with open(log_file, 'r') as f:
                content = f.read()
                
            # Look for performance indicators
            if "completed" in content.lower():
                metrics["found"] = True
                lines = content.split('\n')
                for line in lines:
                    if "Total Operations" in line:
                        parts = line.split(':')
                        if len(parts) > 1:
                            metrics["total_operations"] = parts[1].strip()
                    elif "Throughput" in line:
                        parts = line.split(':')
                        if len(parts) > 1:
                            metrics["throughput"] = parts[1].strip()
                    elif "Memory Bandwidth" in line:
                        parts = line.split(':')
                        if len(parts) > 1:
                            metrics["memory_bandwidth"] = parts[1].strip()
                            
        except Exception as e:
            metrics["error"] = str(e)
            
        return metrics
    
    def _generate_performance_summary(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of independent chiplet performance"""
        summary = {
            "total_tasks": len(metrics),
            "cpu_tasks": 0,
            "gpu_tasks": 0,
            "status": "completed",
            "performance_highlights": []
        }
        
        cpu_throughputs = []
        gpu_throughputs = []
        
        for task_name, task_metrics in metrics.items():
            if task_metrics.get("type") == "independent_cpu":
                summary["cpu_tasks"] += 1
                if "throughput" in task_metrics:
                    try:
                        # Extract numeric value from throughput string
                        throughput_str = task_metrics["throughput"].replace(" ops/sec", "").replace(",", "")
                        cpu_throughputs.append(float(throughput_str))
                    except:
                        pass
            elif task_metrics.get("type") == "independent_gpu":
                summary["gpu_tasks"] += 1
                if "throughput" in task_metrics:
                    try:
                        # Extract numeric value from throughput string
                        throughput_str = task_metrics["throughput"].replace(" ops/sec", "").replace(",", "")
                        gpu_throughputs.append(float(throughput_str))
                    except:
                        pass
        
        # Calculate performance highlights
        if cpu_throughputs:
            avg_cpu_throughput = sum(cpu_throughputs) / len(cpu_throughputs)
            summary["performance_highlights"].append(f"Average CPU throughput: {avg_cpu_throughput:,.0f} ops/sec")
        
        if gpu_throughputs:
            avg_gpu_throughput = sum(gpu_throughputs) / len(gpu_throughputs)
            summary["performance_highlights"].append(f"Average GPU throughput: {avg_gpu_throughput:,.0f} ops/sec")
        
        if cpu_throughputs and gpu_throughputs:
            ratio = (sum(gpu_throughputs) / len(gpu_throughputs)) / (sum(cpu_throughputs) / len(cpu_throughputs))
            summary["performance_highlights"].append(f"GPU/CPU performance ratio: {ratio:.2f}x")
        
        return summary
    
    def generate_report(self, config: SystemConfig, results: Dict[str, Any]) -> str:
        """Generate human-readable report for independent chiplet testing"""
        report = f"""
=== 独立芯粒性能测试报告 ===

系统配置:
- CPU: {config.cpu.cores}-core {config.cpu.arch} @ {config.cpu.freq_ghz}GHz
- GPU: {config.gpu.sm_count}SM {config.gpu.arch} with {config.gpu.mem_gb}GB memory
- Memory: {config.memory.size_gb}GB {config.memory.type}
- 测试类型: 独立芯粒性能测试（无芯粒间通信）

CPU任务配置:
- 任务类型: {config.cpu.task.task_type}
- 操作类型: {', '.join(config.cpu.task.operations)}
- 迭代次数: {config.cpu.task.iterations:,}
- 数据大小: {config.cpu.task.data_size} KB

GPU任务配置:
- 任务类型: {config.gpu.task.task_type}
- 操作类型: {', '.join(config.gpu.task.operations)}
- 迭代次数: {config.gpu.task.iterations:,}
- 数据大小: {config.gpu.task.data_size} KB

测试结果:
- 状态: {results['status']}
- 总任务数: {results.get('performance_summary', {}).get('total_tasks', 0)}
- CPU任务: {results.get('performance_summary', {}).get('cpu_tasks', 0)}
- GPU任务: {results.get('performance_summary', {}).get('gpu_tasks', 0)}

性能亮点:
"""
        
        highlights = results.get('performance_summary', {}).get('performance_highlights', [])
        for highlight in highlights:
            report += f"- {highlight}\n"
        
        if results.get('errors'):
            report += f"\n错误信息:\n"
            for error in results['errors']:
                report += f"- {error}\n"
        
        report += f"\n测试持续时间: {config.test_duration_sec} 秒"
        report += f"\n性能指标: {', '.join(config.performance_metrics)}"
        
        return report


class IndependentChipletPipeline:
    def __init__(self, api_key: str, simulator_root: str):
        self.parser = IndependentChipletParser(api_key)
        self.generator = IndependentSimulationGenerator(simulator_root)
        self.runner = IndependentSimulationRunner(simulator_root)
        
    def run_pipeline(self, description: str) -> Dict[str, Any]:
        """Run the complete independent chiplet testing pipeline"""
        pipeline_result = {
            "status": "started",
            "steps": [],
            "config": None,
            "files": {},
            "results": {},
            "report": ""
        }
        
        try:
            # Step 1: Parse natural language
            print("Step 1: Parsing natural language description for independent chiplet testing...")
            config = self.parser.parse(description)
            if not config:
                pipeline_result["status"] = "failed"
                pipeline_result["steps"].append("Parse failed")
                return pipeline_result
            
            pipeline_result["config"] = self.parser.config_to_dict(config)
            pipeline_result["steps"].append("Parse completed")
            
            # Step 2: Generate simulation files
            print("Step 2: Generating independent chiplet simulation files...")
            output_dir = Path("/home/hao123/chiplet/benchmark/ai_sim_gen")
            files = self.generator.generate_all(config, output_dir)
            pipeline_result["files"] = files
            pipeline_result["steps"].append("Generation completed")
            
            # Step 3: Setup environment
            print("Step 3: Setting up environment...")
            if not self.runner.setup_environment():
                pipeline_result["status"] = "failed"
                pipeline_result["steps"].append("Environment setup failed")
                return pipeline_result
            pipeline_result["steps"].append("Environment setup completed")
            
            # Step 4: Build benchmark
            print("Step 4: Building independent chiplet tasks...")
            if not self.runner.build_benchmark():
                pipeline_result["status"] = "failed"
                pipeline_result["steps"].append("Build failed")
                return pipeline_result
            pipeline_result["steps"].append("Build completed")
            
            # Step 5: Run simulation
            print("Step 5: Running independent chiplet simulation...")
            if not self.runner.run_simulation("independent_simulation.yml"):
                pipeline_result["status"] = "failed"
                pipeline_result["steps"].append("Simulation failed")
                return pipeline_result
            pipeline_result["steps"].append("Simulation completed")
            
            # Step 6: Analyze results
            print("Step 6: Analyzing independent chiplet performance...")
            results = self.runner.analyze_results()
            pipeline_result["results"] = results
            pipeline_result["steps"].append("Analysis completed")
            
            # Step 7: Generate report
            print("Step 7: Generating performance report...")
            report = self.runner.generate_report(config, results)
            pipeline_result["report"] = report
            pipeline_result["steps"].append("Report generated")
            
            pipeline_result["status"] = "completed"
            
        except Exception as e:
            pipeline_result["status"] = "error"
            pipeline_result["steps"].append(f"Error: {e}")
        
        return pipeline_result


def main():
    """Main entry point for independent chiplet testing pipeline"""
    if len(sys.argv) < 2:
        print("Usage: python independent_sim_runner.py '<description>'")
        print("Example: python independent_sim_runner.py '测试一个8核x86 CPU和80SM Ampere GPU的独立性能，每个芯粒运行重复的加减乘除和函数调用运算，持续60秒'")
        return
    
    description = sys.argv[1]
    api_key = "sk-c5446da718c24bfab2536bf0b2d5abb0"
    simulator_root = "/home/hao123/chiplet"
    
    pipeline = IndependentChipletPipeline(api_key, simulator_root)
    result = pipeline.run_pipeline(description)
    
    print("\n" + "="*60)
    print("独立芯粒性能测试流水线执行结果")
    print("="*60)
    print(f"状态: {result['status']}")
    print(f"执行步骤: {' -> '.join(result['steps'])}")
    
    if result.get('report'):
        print(result['report'])
    
    # Save detailed results
    with open('/home/hao123/chiplet/benchmark/ai_sim_gen/independent_pipeline_result.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"\n详细结果已保存到: /home/hao123/chiplet/benchmark/ai_sim_gen/independent_pipeline_result.json")


if __name__ == "__main__":
    main()