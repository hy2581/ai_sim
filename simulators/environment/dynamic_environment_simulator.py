#!/usr/bin/env python3
"""
Dynamic Environment Simulator for Aerospace Microsystems
动态环境仿真器 - 模拟太空环境实时变化

模拟各种太空环境因素的动态变化，包括：
- 轨道动力学变化
- 辐射环境波动
- 热环境循环
- 磁场变化
- 太阳活动影响
- 大气阻力变化
"""

import json
import math
import time
import threading
import numpy as np
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from datetime import datetime, timedelta


class EnvironmentEventType(Enum):
    """环境事件类型"""
    SOLAR_FLARE = "solar_flare"                    # 太阳耀斑
    GEOMAGNETIC_STORM = "geomagnetic_storm"        # 地磁暴
    ECLIPSE_ENTRY = "eclipse_entry"                # 进入地影
    ECLIPSE_EXIT = "eclipse_exit"                  # 离开地影
    RADIATION_BELT_PASSAGE = "radiation_belt"      # 穿越辐射带
    THERMAL_SHOCK = "thermal_shock"                # 热冲击
    MICROMETEORITE_IMPACT = "micrometeorite"       # 微流星体撞击
    ATMOSPHERIC_DRAG_CHANGE = "drag_change"        # 大气阻力变化
    PLASMA_DENSITY_SPIKE = "plasma_spike"          # 等离子体密度峰值


@dataclass
class EnvironmentState:
    """环境状态"""
    timestamp: float                               # 仿真时间戳
    orbital_position: Tuple[float, float, float]   # 轨道位置 (x, y, z) km
    solar_flux: float                              # 太阳通量 W/m²
    temperature_external: float                    # 外部温度 °C
    radiation_dose_rate: float                     # 辐射剂量率 rad/s
    magnetic_field_strength: float                 # 磁场强度 nT
    plasma_density: float                          # 等离子体密度 cm^-3
    atmospheric_density: float                     # 大气密度 kg/m³
    eclipse_factor: float                          # 地影因子 (0-1)
    solar_activity_index: float                    # 太阳活动指数


@dataclass
class EnvironmentEvent:
    """环境事件"""
    event_type: EnvironmentEventType
    start_time: float                              # 开始时间
    duration: float                                # 持续时间
    severity: float                                # 严重程度 (0-1)
    parameters: Dict[str, Any]                     # 事件参数
    impact_factors: Dict[str, float]               # 影响因子


class OrbitDynamics:
    """轨道动力学计算器"""
    
    def __init__(self, semi_major_axis: float, eccentricity: float, 
                 inclination: float, initial_time: float = 0):
        self.a = semi_major_axis  # 半长轴 (km)
        self.e = eccentricity     # 偏心率
        self.i = inclination      # 倾角 (度)
        self.mu = 398600.4418     # 地球引力参数 km³/s²
        self.initial_time = initial_time
        
        # 计算轨道周期
        self.period = 2 * math.pi * math.sqrt(self.a**3 / self.mu)
        
    def get_position(self, time: float) -> Tuple[float, float, float]:
        """计算轨道位置"""
        # 简化的开普勒轨道计算
        mean_motion = 2 * math.pi / self.period
        mean_anomaly = mean_motion * (time - self.initial_time)
        
        # 偏近点角（简化计算）
        eccentric_anomaly = mean_anomaly + self.e * math.sin(mean_anomaly)
        
        # 真近点角
        true_anomaly = 2 * math.atan2(
            math.sqrt(1 + self.e) * math.sin(eccentric_anomaly / 2),
            math.sqrt(1 - self.e) * math.cos(eccentric_anomaly / 2)
        )
        
        # 轨道半径
        r = self.a * (1 - self.e * math.cos(eccentric_anomaly))
        
        # 轨道坐标系中的位置
        x_orbit = r * math.cos(true_anomaly)
        y_orbit = r * math.sin(true_anomaly)
        z_orbit = 0
        
        # 转换到地心坐标系（简化）
        i_rad = math.radians(self.i)
        x = x_orbit
        y = y_orbit * math.cos(i_rad)
        z = y_orbit * math.sin(i_rad)
        
        return (x, y, z)
    
    def get_altitude(self, time: float) -> float:
        """获取高度"""
        x, y, z = self.get_position(time)
        distance = math.sqrt(x**2 + y**2 + z**2)
        return distance - 6371  # 地球半径


class SolarActivityModel:
    """太阳活动模型"""
    
    def __init__(self, solar_cycle_phase: float = 0.5):
        self.solar_cycle_phase = solar_cycle_phase  # 太阳周期相位 (0-1)
        self.base_flux = 1361  # 太阳常数 W/m²
        
    def get_solar_flux(self, time: float, distance_au: float = 1.0) -> float:
        """计算太阳通量"""
        # 11年太阳周期
        cycle_variation = 0.1 * math.sin(2 * math.pi * self.solar_cycle_phase)
        
        # 27天自转周期
        rotation_variation = 0.05 * math.sin(2 * math.pi * time / (27 * 24 * 3600))
        
        # 随机波动
        random_variation = 0.02 * (np.random.random() - 0.5)
        
        flux = self.base_flux * (1 + cycle_variation + rotation_variation + random_variation)
        
        # 距离平方反比定律
        return flux / (distance_au ** 2)
    
    def generate_solar_flare(self, time: float) -> Optional[EnvironmentEvent]:
        """生成太阳耀斑事件"""
        # 基于太阳活动水平的概率
        flare_probability = 0.001 * (1 + self.solar_cycle_phase)
        
        if np.random.random() < flare_probability:
            severity = np.random.exponential(0.3)  # 指数分布
            severity = min(severity, 1.0)
            
            return EnvironmentEvent(
                event_type=EnvironmentEventType.SOLAR_FLARE,
                start_time=time,
                duration=np.random.uniform(600, 7200),  # 10分钟到2小时
                severity=severity,
                parameters={
                    "x_ray_class": self._get_flare_class(severity),
                    "proton_flux_increase": severity * 100,
                    "radio_blackout_level": severity
                },
                impact_factors={
                    "radiation_multiplier": 1 + severity * 10,
                    "communication_degradation": severity * 0.8,
                    "electronics_stress": severity * 0.6
                }
            )
        return None
    
    def _get_flare_class(self, severity: float) -> str:
        """根据严重程度确定耀斑等级"""
        if severity < 0.1:
            return "A"
        elif severity < 0.3:
            return "B"
        elif severity < 0.5:
            return "C"
        elif severity < 0.7:
            return "M"
        else:
            return "X"


class RadiationEnvironmentModel:
    """辐射环境模型"""
    
    def __init__(self):
        self.van_allen_inner = {"altitude_min": 700, "altitude_max": 10000, "intensity": 1000}
        self.van_allen_outer = {"altitude_min": 13000, "altitude_max": 60000, "intensity": 100}
        self.base_cosmic_ray_flux = 0.001  # rad/s
        
    def get_radiation_dose_rate(self, altitude: float, solar_activity: float, 
                               magnetic_latitude: float = 0) -> float:
        """计算辐射剂量率"""
        dose_rate = self.base_cosmic_ray_flux
        
        # 范艾伦辐射带
        if self.van_allen_inner["altitude_min"] <= altitude <= self.van_allen_inner["altitude_max"]:
            belt_factor = self._calculate_belt_intensity(altitude, self.van_allen_inner)
            dose_rate += belt_factor * self.van_allen_inner["intensity"] * 1e-6
            
        if self.van_allen_outer["altitude_min"] <= altitude <= self.van_allen_outer["altitude_max"]:
            belt_factor = self._calculate_belt_intensity(altitude, self.van_allen_outer)
            dose_rate += belt_factor * self.van_allen_outer["intensity"] * 1e-6
        
        # 太阳活动影响
        solar_modulation = 1 - 0.3 * solar_activity  # 太阳活动抑制宇宙射线
        dose_rate *= solar_modulation
        
        # 地磁屏蔽效应
        magnetic_shielding = math.cos(math.radians(magnetic_latitude)) ** 2
        dose_rate *= (1 - 0.5 * magnetic_shielding)
        
        return max(dose_rate, 1e-6)  # 最小值
    
    def _calculate_belt_intensity(self, altitude: float, belt_params: Dict) -> float:
        """计算辐射带强度因子"""
        center = (belt_params["altitude_min"] + belt_params["altitude_max"]) / 2
        width = belt_params["altitude_max"] - belt_params["altitude_min"]
        
        # 高斯分布
        return math.exp(-((altitude - center) / (width / 4)) ** 2)


class ThermalEnvironmentModel:
    """热环境模型"""
    
    def __init__(self, spacecraft_area: float = 10.0, emissivity: float = 0.85):
        self.spacecraft_area = spacecraft_area  # 航天器面积 m²
        self.emissivity = emissivity           # 发射率
        self.stefan_boltzmann = 5.67e-8        # 斯特藩-玻尔兹曼常数
        
    def calculate_temperature(self, solar_flux: float, eclipse_factor: float,
                            albedo_flux: float = 0, earth_ir_flux: float = 237) -> float:
        """计算航天器温度"""
        # 太阳辐射加热
        solar_heating = solar_flux * self.spacecraft_area * (1 - eclipse_factor) * 0.7  # 吸收率0.7
        
        # 地球反照辐射
        albedo_heating = albedo_flux * self.spacecraft_area * 0.3  # 地球反照率
        
        # 地球红外辐射
        earth_ir_heating = earth_ir_flux * self.spacecraft_area
        
        # 总加热功率
        total_heating = solar_heating + albedo_heating + earth_ir_heating
        
        # 辐射冷却
        # T^4 = Q / (ε * σ * A)
        equilibrium_temp_k = (total_heating / (self.emissivity * self.stefan_boltzmann * self.spacecraft_area)) ** 0.25
        
        # 转换为摄氏度
        return equilibrium_temp_k - 273.15
    
    def generate_thermal_shock_event(self, current_temp: float, time: float) -> Optional[EnvironmentEvent]:
        """生成热冲击事件"""
        # 进出地影时的快速温度变化
        shock_probability = 0.01  # 每次计算1%概率
        
        if np.random.random() < shock_probability:
            temp_change = np.random.uniform(-100, 100)  # 温度变化范围
            
            return EnvironmentEvent(
                event_type=EnvironmentEventType.THERMAL_SHOCK,
                start_time=time,
                duration=np.random.uniform(300, 1800),  # 5-30分钟
                severity=abs(temp_change) / 200,  # 归一化严重程度
                parameters={
                    "temperature_change": temp_change,
                    "rate_of_change": temp_change / 600  # °C/min
                },
                impact_factors={
                    "thermal_stress": abs(temp_change) / 100,
                    "component_reliability": 1 - abs(temp_change) / 500
                }
            )
        return None


class DynamicEnvironmentSimulator:
    """动态环境仿真器主类"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.current_time = 0.0
        self.time_step = config.get("time_step", 60.0)  # 默认60秒步长
        self.simulation_duration = config.get("duration", 86400)  # 默认24小时
        
        # 初始化子模型
        orbit_params = config.get("orbit", {})
        self.orbit_dynamics = OrbitDynamics(
            semi_major_axis=orbit_params.get("semi_major_axis", 7000),
            eccentricity=orbit_params.get("eccentricity", 0.001),
            inclination=orbit_params.get("inclination", 98.2)
        )
        
        self.solar_model = SolarActivityModel(
            solar_cycle_phase=config.get("solar_cycle_phase", 0.5)
        )
        
        self.radiation_model = RadiationEnvironmentModel()
        
        self.thermal_model = ThermalEnvironmentModel(
            spacecraft_area=config.get("spacecraft_area", 10.0),
            emissivity=config.get("emissivity", 0.85)
        )
        
        # 状态记录
        self.environment_history: List[EnvironmentState] = []
        self.event_history: List[EnvironmentEvent] = []
        self.active_events: List[EnvironmentEvent] = []
        
        # 回调函数
        self.state_callbacks: List[Callable[[EnvironmentState], None]] = []
        self.event_callbacks: List[Callable[[EnvironmentEvent], None]] = []
        
        # 仿真控制
        self.is_running = False
        self.simulation_thread = None
        
        # 日志设置
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def add_state_callback(self, callback: Callable[[EnvironmentState], None]):
        """添加状态变化回调"""
        self.state_callbacks.append(callback)
    
    def add_event_callback(self, callback: Callable[[EnvironmentEvent], None]):
        """添加事件回调"""
        self.event_callbacks.append(callback)
    
    def start_simulation(self, real_time: bool = False):
        """启动仿真"""
        if self.is_running:
            self.logger.warning("Simulation is already running")
            return
        
        self.is_running = True
        self.current_time = 0.0
        
        if real_time:
            self.simulation_thread = threading.Thread(target=self._run_real_time_simulation)
            self.simulation_thread.start()
        else:
            self._run_batch_simulation()
    
    def stop_simulation(self):
        """停止仿真"""
        self.is_running = False
        if self.simulation_thread and self.simulation_thread.is_alive():
            self.simulation_thread.join()
    
    def _run_batch_simulation(self):
        """批量仿真模式"""
        self.logger.info(f"Starting batch simulation for {self.simulation_duration} seconds")
        
        while self.current_time < self.simulation_duration and self.is_running:
            self._simulation_step()
            self.current_time += self.time_step
        
        self.logger.info("Batch simulation completed")
    
    def _run_real_time_simulation(self):
        """实时仿真模式"""
        self.logger.info("Starting real-time simulation")
        start_wall_time = time.time()
        
        while self.current_time < self.simulation_duration and self.is_running:
            step_start = time.time()
            
            self._simulation_step()
            self.current_time += self.time_step
            
            # 实时同步
            step_duration = time.time() - step_start
            sleep_time = max(0, self.time_step - step_duration)
            time.sleep(sleep_time)
        
        self.logger.info("Real-time simulation completed")
    
    def _simulation_step(self):
        """单步仿真"""
        # 计算当前环境状态
        current_state = self._calculate_environment_state()
        
        # 检查和生成新事件
        new_events = self._generate_environment_events(current_state)
        
        # 更新活跃事件列表
        self._update_active_events()
        
        # 应用事件影响
        modified_state = self._apply_event_effects(current_state)
        
        # 记录状态
        self.environment_history.append(modified_state)
        
        # 触发回调
        for callback in self.state_callbacks:
            try:
                callback(modified_state)
            except Exception as e:
                self.logger.error(f"State callback error: {e}")
        
        # 处理新事件
        for event in new_events:
            self.event_history.append(event)
            self.active_events.append(event)
            
            for callback in self.event_callbacks:
                try:
                    callback(event)
                except Exception as e:
                    self.logger.error(f"Event callback error: {e}")
    
    def _calculate_environment_state(self) -> EnvironmentState:
        """计算当前环境状态"""
        # 轨道位置
        position = self.orbit_dynamics.get_position(self.current_time)
        altitude = self.orbit_dynamics.get_altitude(self.current_time)
        
        # 太阳通量
        solar_flux = self.solar_model.get_solar_flux(self.current_time)
        
        # 地影计算（简化）
        eclipse_factor = self._calculate_eclipse_factor(position)
        
        # 辐射环境
        solar_activity = 0.5 + 0.3 * math.sin(2 * math.pi * self.current_time / (11 * 365 * 24 * 3600))
        radiation_dose = self.radiation_model.get_radiation_dose_rate(altitude, solar_activity)
        
        # 热环境
        temperature = self.thermal_model.calculate_temperature(solar_flux, eclipse_factor)
        
        # 其他环境参数
        magnetic_field = 50000 / (altitude / 6371) ** 3  # 简化磁场模型
        plasma_density = 1e6 / (altitude / 1000) ** 2    # 简化等离子体密度
        atmospheric_density = 1.225 * math.exp(-altitude / 8.5)  # 指数大气模型
        
        return EnvironmentState(
            timestamp=self.current_time,
            orbital_position=position,
            solar_flux=solar_flux,
            temperature_external=temperature,
            radiation_dose_rate=radiation_dose,
            magnetic_field_strength=magnetic_field,
            plasma_density=plasma_density,
            atmospheric_density=atmospheric_density,
            eclipse_factor=eclipse_factor,
            solar_activity_index=solar_activity
        )
    
    def _calculate_eclipse_factor(self, position: Tuple[float, float, float]) -> float:
        """计算地影因子"""
        x, y, z = position
        earth_radius = 6371
        
        # 简化的地影计算
        distance_from_earth = math.sqrt(x**2 + y**2 + z**2)
        
        # 如果在地球阴影中
        if x < 0 and math.sqrt(y**2 + z**2) < earth_radius:
            return 1.0  # 完全在地影中
        else:
            return 0.0  # 在阳光中
    
    def _generate_environment_events(self, state: EnvironmentState) -> List[EnvironmentEvent]:
        """生成环境事件"""
        events = []
        
        # 太阳耀斑
        solar_flare = self.solar_model.generate_solar_flare(self.current_time)
        if solar_flare:
            events.append(solar_flare)
        
        # 热冲击
        thermal_shock = self.thermal_model.generate_thermal_shock_event(
            state.temperature_external, self.current_time
        )
        if thermal_shock:
            events.append(thermal_shock)
        
        # 地影进出事件
        if hasattr(self, '_previous_eclipse_factor'):
            if self._previous_eclipse_factor == 0 and state.eclipse_factor > 0:
                events.append(EnvironmentEvent(
                    event_type=EnvironmentEventType.ECLIPSE_ENTRY,
                    start_time=self.current_time,
                    duration=0,
                    severity=1.0,
                    parameters={"eclipse_type": "umbra"},
                    impact_factors={"power_generation": 0.0, "thermal_shock": 0.8}
                ))
            elif self._previous_eclipse_factor > 0 and state.eclipse_factor == 0:
                events.append(EnvironmentEvent(
                    event_type=EnvironmentEventType.ECLIPSE_EXIT,
                    start_time=self.current_time,
                    duration=0,
                    severity=1.0,
                    parameters={"eclipse_type": "umbra"},
                    impact_factors={"power_generation": 1.0, "thermal_shock": 0.8}
                ))
        
        self._previous_eclipse_factor = state.eclipse_factor
        
        return events
    
    def _update_active_events(self):
        """更新活跃事件列表"""
        self.active_events = [
            event for event in self.active_events
            if self.current_time < event.start_time + event.duration
        ]
    
    def _apply_event_effects(self, state: EnvironmentState) -> EnvironmentState:
        """应用事件效果到环境状态"""
        modified_state = EnvironmentState(**asdict(state))
        
        for event in self.active_events:
            # 计算事件影响强度（随时间衰减）
            elapsed_time = self.current_time - event.start_time
            if elapsed_time >= 0 and elapsed_time < event.duration:
                # 简单的线性衰减模型
                intensity = event.severity * (1 - elapsed_time / event.duration)
                
                # 应用影响因子
                for factor, multiplier in event.impact_factors.items():
                    if factor == "radiation_multiplier":
                        modified_state.radiation_dose_rate *= (1 + intensity * (multiplier - 1))
                    elif factor == "thermal_shock":
                        temp_change = intensity * multiplier * 50  # 最大50°C变化
                        modified_state.temperature_external += temp_change
        
        return modified_state
    
    def get_current_state(self) -> Optional[EnvironmentState]:
        """获取当前环境状态"""
        return self.environment_history[-1] if self.environment_history else None
    
    def get_state_history(self, start_time: float = 0, end_time: float = None) -> List[EnvironmentState]:
        """获取状态历史"""
        if end_time is None:
            end_time = self.current_time
        
        return [
            state for state in self.environment_history
            if start_time <= state.timestamp <= end_time
        ]
    
    def get_event_history(self, event_type: Optional[EnvironmentEventType] = None) -> List[EnvironmentEvent]:
        """获取事件历史"""
        if event_type is None:
            return self.event_history.copy()
        else:
            return [event for event in self.event_history if event.event_type == event_type]
    
    def export_data(self, output_path: str, format: str = "json"):
        """导出仿真数据"""
        data = {
            "simulation_config": self.config,
            "environment_history": [asdict(state) for state in self.environment_history],
            "event_history": [asdict(event) for event in self.event_history],
            "statistics": self._calculate_statistics()
        }
        
        if format.lower() == "json":
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _calculate_statistics(self) -> Dict[str, Any]:
        """计算仿真统计信息"""
        if not self.environment_history:
            return {}
        
        temperatures = [state.temperature_external for state in self.environment_history]
        radiation_doses = [state.radiation_dose_rate for state in self.environment_history]
        
        return {
            "simulation_duration": self.current_time,
            "total_steps": len(self.environment_history),
            "total_events": len(self.event_history),
            "temperature_stats": {
                "min": min(temperatures),
                "max": max(temperatures),
                "mean": sum(temperatures) / len(temperatures),
                "std": np.std(temperatures)
            },
            "radiation_stats": {
                "min": min(radiation_doses),
                "max": max(radiation_doses),
                "mean": sum(radiation_doses) / len(radiation_doses),
                "total_dose": sum(radiation_doses) * self.time_step
            },
            "event_counts": {
                event_type.value: len([e for e in self.event_history if e.event_type == event_type])
                for event_type in EnvironmentEventType
            }
        }


def main():
    """主函数 - 演示动态环境仿真器"""
    # 配置仿真参数
    config = {
        "time_step": 60,  # 1分钟步长
        "duration": 86400,  # 24小时仿真
        "orbit": {
            "semi_major_axis": 7000,  # km
            "eccentricity": 0.001,
            "inclination": 98.2  # 度
        },
        "solar_cycle_phase": 0.7,  # 太阳活动高峰期
        "spacecraft_area": 15.0,   # m²
        "emissivity": 0.85
    }
    
    # 创建仿真器
    simulator = DynamicEnvironmentSimulator(config)
    
    # 添加回调函数
    def state_callback(state: EnvironmentState):
        if int(state.timestamp) % 3600 == 0:  # 每小时打印一次
            print(f"Time: {state.timestamp/3600:.1f}h, "
                  f"Temp: {state.temperature_external:.1f}°C, "
                  f"Radiation: {state.radiation_dose_rate:.2e} rad/s, "
                  f"Eclipse: {state.eclipse_factor:.1f}")
    
    def event_callback(event: EnvironmentEvent):
        print(f"Event: {event.event_type.value} at {event.start_time/3600:.1f}h, "
              f"severity: {event.severity:.2f}")
    
    simulator.add_state_callback(state_callback)
    simulator.add_event_callback(event_callback)
    
    # 运行仿真
    print("Starting dynamic environment simulation...")
    simulator.start_simulation(real_time=False)
    
    # 导出结果
    simulator.export_data("environment_simulation_results.json")
    print("Simulation completed. Results exported to environment_simulation_results.json")
    
    # 打印统计信息
    stats = simulator._calculate_statistics()
    print("\nSimulation Statistics:")
    print(f"Total events: {stats['total_events']}")
    print(f"Temperature range: {stats['temperature_stats']['min']:.1f} to {stats['temperature_stats']['max']:.1f}°C")
    print(f"Total radiation dose: {stats['radiation_stats']['total_dose']:.2e} rad")


if __name__ == "__main__":
    main()