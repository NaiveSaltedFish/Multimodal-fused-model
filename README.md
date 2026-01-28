=== 多模态情感分类项目===

=== 项目概述 ===

本项目实现了一个基于文本和图像的多模态情感分类系统。通过融合社交媒体帖子的文本内容和对应的图片特征，系统能够识别用户表达的三种情感：负面(negative)、中性(neutral)和正面(positive)。

=== 项目结构 ===

Multimodal_fused_model/

├── data/                                   # 数据目录

│   ├── 1.jpg                               # 图像文件（示例）

│   ├── 1.txt                               # 文本文件（示例）

│   ├── train.txt                           # 训练标签文件

│   ├── test_without_label.txt              # 测试集文件

│   ├──models/                              # 模型目录

│   │  ├──logistic_regression.pkl           # 模型文件（示例）

│   │  ├──logistic_regression_summary.txt   # 模型摘要（示例）

│   ├──splits/                              # 特征目录

│      ├──train_dataset.csv                 # 数据集信息（示例）

│      ├──train_features.npz                # 数据集特征（示例）

├── Prework.ipynb                           # 数据处理程序

├── ImprovedLogisticRegression.ipynb        # 改善的逻辑回归模型程序

├── LogisticRegression.ipynb                # 逻辑回归模型程序

├── Test.ipynb                              # 测试集运行程序

├── Experiment .ipynb                       # 消融实验结果程序

├── requirements.txt                        # 环境依赖

├── README.md                               # 项目说明

=== 环境要求 ===

基础科学计算

numpy>=1.24.0

pandas>=1.5.0

scipy>=1.10.0

机器学习

scikit-learn>=1.2.0

joblib>=1.2.0

深度学习框架

torch>=1.13.0

torchvision>=0.14.0

自然语言处理

nltk>=3.8.0

transformers>=4.25.0  # 可选，用于BERT等高级模型

图像处理

Pillow>=9.4.0

opencv-python>=4.7.0  # 可选

albumentations>=1.3.0  # 可选，图像增强

文本处理

gensim>=4.3.0  # 可选，用于词向量

chardet>=5.1.0  # 编码检测

数据可视化

matplotlib>=3.6.0

seaborn>=0.12.0

plotly>=5.13.0  # 可选，交互式可视化

数据处理

tqdm>=4.64.0  # 进度条

pyyaml>=6.0  # 配置文件

Jupyter支持

jupyter>=1.0.0

ipykernel>=6.0.0

开发工具

black>=23.0.0  # 代码格式化

flake8>=6.0.0  # 代码检查

pytest>=7.2.0  # 测试

其他

python-dotenv>=0.21.0  # 环境变量管理

你可以通过执行
pip install -r requirements.txt

安装依赖。
