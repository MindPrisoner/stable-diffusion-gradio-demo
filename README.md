# Stable Diffusion Gradio Demo

这是一个本地 Stable Diffusion 文生图 Demo，基于 Stable Diffusion v1.5 和 Gradio 构建，并额外保留了 LoRA 接口，便于做风格扩展和效果对比。

## 项目定位

- 提供本地文生图交互界面
- 支持步数、引导强度、分辨率和随机种子控制
- 支持本地 LoRA 权重加载
- 记录基础模型与 LoRA 版本的生成差异

## 技术栈

- PyTorch
- Diffusers
- Transformers
- Gradio

## 目录结构

```text
stable_diffusion_demo/
├── app.py                # Gradio 入口
├── models/
│   └── lora/             # LoRA 权重目录
├── outputs/              # 生成结果目录
├── requirements.txt      # 依赖列表
└── README.md
```

## 模型说明

项目基于本地 Stable Diffusion v1.5 模型加载，默认采用 FP16 推理配置。

当前实现支持：

- 基础文本生成
- LoRA 权重加载
- Base vs LoRA 对比

## 运行方式

先安装依赖：

```bash
pip install -r requirements.txt
```

然后启动 Demo：

```bash
python app.py
```

## 界面功能

界面中提供了以下控制项：

- Prompt
- Negative Prompt
- 推理步数
- Guidance Scale
- 分辨率选择
- Seed 控制
- LoRA 目录
- LoRA 权重文件名
- LoRA Scale

## 推荐配置

当前 Demo 中的推荐参数为：

- `Steps = 30`
- `Guidance = 10`
- `Resolution = 384 x 384`

## 输出结果

生成结果会保存在 `outputs/` 目录中，同时界面会返回：

- 生成图片
- 生成耗时
- 模型加载状态
- 保存路径

## 备注

- 模型从本地目录加载，不依赖在线下载
- `local_files_only=True` 已启用
- 需要本地准备好 Stable Diffusion 模型文件和 LoRA 权重文件
- `outputs/` 目录用于保存基础生成、LoRA 生成和对比结果

