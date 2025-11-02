#!/usr/bin/env python3
"""
Independent Chiplet Performance Analyzer
Collects and analyzes performance metrics from independent chiplet tasks
"""

import json
import glob
import sys
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
import argparse


class PerformanceAnalyzer:
    def __init__(self):
        self.results = []
        self.summary = {}
        
    def collect_results(self):
        """Collect results from all task result files"""
        # Find all result JSON files
        cpu_files = glob.glob("cpu_task_*_results.json")
        gpu_files = glob.glob("gpu_task_*_results.json")
        
        print(f"Found {len(cpu_files)} CPU task results and {len(gpu_files)} GPU task results")
        
        # Load CPU results
        for file in cpu_files:
            try:
                with open(file, 'r') as f:
                    data = json.load(f)
                    data['chiplet_type'] = 'CPU'
                    self.results.append(data)
            except Exception as e:
                print(f"Error loading {file}: {e}")
        
        # Load GPU results
        for file in gpu_files:
            try:
                with open(file, 'r') as f:
                    data = json.load(f)
                    data['chiplet_type'] = 'GPU'
                    self.results.append(data)
            except Exception as e:
                print(f"Error loading {file}: {e}")
    
    def analyze_performance(self):
        """Analyze collected performance data"""
        if not self.results:
            print("No results to analyze")
            return
            
        # Separate CPU and GPU results
        cpu_results = [r for r in self.results if r['chiplet_type'] == 'CPU']
        gpu_results = [r for r in self.results if r['chiplet_type'] == 'GPU']
        
        # Calculate summary statistics
        self.summary = {
            'total_chiplets': len(self.results),
            'cpu_chiplets': len(cpu_results),
            'gpu_chiplets': len(gpu_results),
            'cpu_performance': self._analyze_cpu_performance(cpu_results),
            'gpu_performance': self._analyze_gpu_performance(gpu_results),
            'comparative_analysis': self._comparative_analysis(cpu_results, gpu_results)
        }
    
    def _analyze_cpu_performance(self, cpu_results: List[Dict]) -> Dict:
        """Analyze CPU performance metrics"""
        if not cpu_results:
            return {}
            
        total_ops = sum(r['total_operations'] for r in cpu_results)
        total_time = sum(r['execution_time_sec'] for r in cpu_results)
        avg_throughput = sum(r['throughput_ops_per_sec'] for r in cpu_results) / len(cpu_results)
        avg_latency = sum(r['avg_latency_ns'] for r in cpu_results) / len(cpu_results)
        avg_utilization = sum(r['cpu_utilization'] for r in cpu_results) / len(cpu_results)
        
        return {
            'total_operations': total_ops,
            'total_execution_time_sec': total_time,
            'average_throughput_ops_per_sec': avg_throughput,
            'average_latency_ns': avg_latency,
            'average_utilization_percent': avg_utilization,
            'efficiency_score': avg_throughput / 1000000  # Ops per second in millions
        }
    
    def _analyze_gpu_performance(self, gpu_results: List[Dict]) -> Dict:
        """Analyze GPU performance metrics"""
        if not gpu_results:
            return {}
            
        total_ops = sum(r['total_operations'] for r in gpu_results)
        total_time = sum(r['execution_time_sec'] for r in gpu_results)
        avg_throughput = sum(r['throughput_ops_per_sec'] for r in gpu_results) / len(gpu_results)
        avg_latency = sum(r['avg_latency_ns'] for r in gpu_results) / len(gpu_results)
        avg_utilization = sum(r['gpu_utilization'] for r in gpu_results) / len(gpu_results)
        avg_bandwidth = sum(r.get('memory_bandwidth_gbps', 0) for r in gpu_results) / len(gpu_results)
        
        return {
            'total_operations': total_ops,
            'total_execution_time_sec': total_time,
            'average_throughput_ops_per_sec': avg_throughput,
            'average_latency_ns': avg_latency,
            'average_utilization_percent': avg_utilization,
            'average_memory_bandwidth_gbps': avg_bandwidth,
            'efficiency_score': avg_throughput / 1000000  # Ops per second in millions
        }
    
    def _comparative_analysis(self, cpu_results: List[Dict], gpu_results: List[Dict]) -> Dict:
        """Compare CPU and GPU performance"""
        analysis = {}
        
        if cpu_results and gpu_results:
            cpu_avg_throughput = sum(r['throughput_ops_per_sec'] for r in cpu_results) / len(cpu_results)
            gpu_avg_throughput = sum(r['throughput_ops_per_sec'] for r in gpu_results) / len(gpu_results)
            
            analysis['gpu_cpu_throughput_ratio'] = gpu_avg_throughput / cpu_avg_throughput if cpu_avg_throughput > 0 else 0
            analysis['performance_leader'] = 'GPU' if gpu_avg_throughput > cpu_avg_throughput else 'CPU'
            
            cpu_avg_latency = sum(r['avg_latency_ns'] for r in cpu_results) / len(cpu_results)
            gpu_avg_latency = sum(r['avg_latency_ns'] for r in gpu_results) / len(gpu_results)
            
            analysis['latency_leader'] = 'CPU' if cpu_avg_latency < gpu_avg_latency else 'GPU'
            analysis['cpu_avg_latency_ns'] = cpu_avg_latency
            analysis['gpu_avg_latency_ns'] = gpu_avg_latency
        
        return analysis
    
    def generate_visualizations(self):
        """Generate performance visualization charts"""
        if not self.results:
            return
            
        # Create performance comparison charts
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Throughput comparison
        chiplet_types = [r['chiplet_type'] for r in self.results]
        throughputs = [r['throughput_ops_per_sec'] / 1e6 for r in self.results]  # Convert to millions
        task_ids = [f"{r['chiplet_type']}-{r['task_id']}" for r in self.results]
        
        ax1.bar(task_ids, throughputs, color=['blue' if t == 'CPU' else 'red' for t in chiplet_types])
        ax1.set_title('Throughput Comparison (Million Ops/sec)')
        ax1.set_ylabel('Throughput (MOps/sec)')
        ax1.tick_params(axis='x', rotation=45)
        
        # Latency comparison
        latencies = [r['avg_latency_ns'] for r in self.results]
        ax2.bar(task_ids, latencies, color=['blue' if t == 'CPU' else 'red' for t in chiplet_types])
        ax2.set_title('Average Latency Comparison')
        ax2.set_ylabel('Latency (ns)')
        ax2.tick_params(axis='x', rotation=45)
        
        # Utilization comparison
        utilizations = [r.get('cpu_utilization', r.get('gpu_utilization', 0)) for r in self.results]
        ax3.bar(task_ids, utilizations, color=['blue' if t == 'CPU' else 'red' for t in chiplet_types])
        ax3.set_title('Utilization Comparison')
        ax3.set_ylabel('Utilization (%)')
        ax3.tick_params(axis='x', rotation=45)
        
        # Execution time comparison
        exec_times = [r['execution_time_sec'] for r in self.results]
        ax4.bar(task_ids, exec_times, color=['blue' if t == 'CPU' else 'red' for t in chiplet_types])
        ax4.set_title('Execution Time Comparison')
        ax4.set_ylabel('Time (seconds)')
        ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('independent_chiplet_performance.png', dpi=300, bbox_inches='tight')
        print("Performance visualization saved as 'independent_chiplet_performance.png'")
    
    def save_analysis_report(self):
        """Save detailed analysis report"""
        report_file = 'performance_analysis_report.json'
        
        report = {
            'analysis_summary': self.summary,
            'detailed_results': self.results,
            'recommendations': self._generate_recommendations()
        }
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"Detailed analysis report saved as '{report_file}'")
    
    def _generate_recommendations(self) -> List[str]:
        """Generate performance optimization recommendations"""
        recommendations = []
        
        if 'comparative_analysis' in self.summary:
            comp = self.summary['comparative_analysis']
            if comp.get('performance_leader') == 'GPU':
                recommendations.append("GPU chiplets show superior throughput performance - consider GPU-heavy workloads")
            else:
                recommendations.append("CPU chiplets show competitive performance - suitable for general-purpose tasks")
                
            if comp.get('latency_leader') == 'CPU':
                recommendations.append("CPU chiplets provide lower latency - ideal for latency-sensitive applications")
        
        # Add utilization recommendations
        if 'cpu_performance' in self.summary:
            cpu_util = self.summary['cpu_performance'].get('average_utilization_percent', 0)
            if cpu_util < 80:
                recommendations.append(f"CPU utilization is {cpu_util:.1f}% - consider increasing workload intensity")
        
        if 'gpu_performance' in self.summary:
            gpu_util = self.summary['gpu_performance'].get('average_utilization_percent', 0)
            if gpu_util < 85:
                recommendations.append(f"GPU utilization is {gpu_util:.1f}% - consider more parallel workloads")
        
        return recommendations
    
    def print_summary(self):
        """Print performance analysis summary"""
        print("\n" + "="*60)
        print("INDEPENDENT CHIPLET PERFORMANCE ANALYSIS SUMMARY")
        print("="*60)
        
        print(f"Total Chiplets Tested: {self.summary.get('total_chiplets', 0)}")
        print(f"CPU Chiplets: {self.summary.get('cpu_chiplets', 0)}")
        print(f"GPU Chiplets: {self.summary.get('gpu_chiplets', 0)}")
        
        if 'cpu_performance' in self.summary:
            cpu = self.summary['cpu_performance']
            print(f"\nCPU Performance:")
            print(f"  Average Throughput: {cpu.get('average_throughput_ops_per_sec', 0):,.0f} ops/sec")
            print(f"  Average Latency: {cpu.get('average_latency_ns', 0):.2f} ns")
            print(f"  Average Utilization: {cpu.get('average_utilization_percent', 0):.1f}%")
            print(f"  Efficiency Score: {cpu.get('efficiency_score', 0):.2f} MOps/sec")
        
        if 'gpu_performance' in self.summary:
            gpu = self.summary['gpu_performance']
            print(f"\nGPU Performance:")
            print(f"  Average Throughput: {gpu.get('average_throughput_ops_per_sec', 0):,.0f} ops/sec")
            print(f"  Average Latency: {gpu.get('average_latency_ns', 0):.2f} ns")
            print(f"  Average Utilization: {gpu.get('average_utilization_percent', 0):.1f}%")
            print(f"  Memory Bandwidth: {gpu.get('average_memory_bandwidth_gbps', 0):.2f} GB/s")
            print(f"  Efficiency Score: {gpu.get('efficiency_score', 0):.2f} MOps/sec")
        
        if 'comparative_analysis' in self.summary:
            comp = self.summary['comparative_analysis']
            print(f"\nComparative Analysis:")
            print(f"  Performance Leader: {comp.get('performance_leader', 'N/A')}")
            print(f"  Latency Leader: {comp.get('latency_leader', 'N/A')}")
            print(f"  GPU/CPU Throughput Ratio: {comp.get('gpu_cpu_throughput_ratio', 0):.2f}x")
        
        print("\n" + "="*60)


def main():
    parser = argparse.ArgumentParser(description='Analyze independent chiplet performance')
    parser.add_argument('--analyze-results', action='store_true', 
                       help='Analyze existing result files')
    
    args = parser.parse_args()
    
    if args.analyze_results:
        analyzer = PerformanceAnalyzer()
        analyzer.collect_results()
        analyzer.analyze_performance()
        analyzer.print_summary()
        analyzer.generate_visualizations()
        analyzer.save_analysis_report()
    else:
        print("Use --analyze-results to analyze performance data")


if __name__ == "__main__":
    main()
