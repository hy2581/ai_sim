# 航空航天微系统仿真平台

## 项目简介

这是一个AI增强的航空航天微系统需求定义与验证平台，专门用于航空航天领域的微系统仿真和验证。该平台集成了人工智能技术，能够自动化地进行任务需求分析、设备需求映射和仿真验证。

## 主要功能

- **任务需求分析**: 自动分析航空航天任务的具体需求
- **设备需求映射**: 将任务需求映射到具体的硬件设备需求
- **仿真验证**: 对系统设计进行仿真验证
- **AI增强**: 集成AI技术提升仿真精度和效率
- **多设备支持**: 支持CPU、GPU、传感器、通信等多种设备仿真

## 项目结构

```
├── main.py                    # 主程序入口
├── run_ai_simulation.py       # AI增强仿真入口
├── demo.py                    # 演示程序
├── config/                    # 配置文件目录
│   ├── requirements.txt       # Python依赖
│   ├── aerospace_simulation.yml # 仿真配置
│   └── Makefile*             # 编译配置
├── core/                      # 核心功能模块
│   ├── ai_integration/        # AI集成模块
│   ├── generators/            # 场景生成器
│   ├── main_control/          # 主控制模块
│   ├── parsers/              # 解析器
│   ├── requirements/         # 需求分析模块
│   └── verification/         # 验证模块
├── simulators/               # 仿真器
│   ├── devices/              # 设备仿真器
│   └── environment/          # 环境仿真器
└── analysis/                 # 分析工具
    ├── performance/          # 性能分析
    └── reports/             # 报告生成
```

## 安装和使用

### 环境要求

- Python 3.8+
- GCC/G++ (用于编译C++仿真器)
- CUDA (可选，用于GPU仿真)

### 安装依赖

```bash
# 安装Python依赖
pip install -r config/requirements.txt

# 编译C++仿真器 (可选)
cd config
make -f Makefile_aerospace
```

### 运行方式

```bash
python3 main.py --mode traditional --input "这款应用于深空探测的微型系统，高度集成于探测器机械臂末端，能够自主执行对天体表面的"接触-采样-原位分析"任务：它先以微型刷扫装置清除目标点浮尘，再驱动微型压电钻头破碎岩石，并利用气体流将颗粒样本送入内置分析舱，最终通过微型光谱仪实时解析其物质成分。该系统的卓越能力源于其精密的MEMS器件，包括用于物质识别的微型激光拉曼探头与高光谱成像仪、执行动作的压电钻头与MEMS气体泵阀、负责控制与数据处理的抗辐射片上系统（SoC），所有这些单 元均通过三维异构集成技术封装在一个轻巧坚固的陶瓷基框架内" --api-key "sk-045330d31ef1429cb907fca2232d8839" 
```


