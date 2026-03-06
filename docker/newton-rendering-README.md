# 在 Docker 中使用 Newton 渲染（无需 Isaac Sim）

在轻量级 Docker 容器中使用 Newton 物理后端和 Newton Warp 渲染器运行
Isaac Lab 训练——**不依赖 Isaac Sim / `isaacsim`**。

## 1. 构建 Docker 镜像

```bash
cd docker
docker build -f Dockerfile.newton -t isaaclab-newton ..
```

## 2. 启动容器

自己添加需要mount的路径；注意启动时需要映射 Viser 可视化端口（默认 `8080`），以便在浏览器中查看仿真画面：

```bash
docker run --rm -it --gpus all \
    -p 8080:8080 \ 
    isaaclab-newton
```

## 3. 在容器内启动训练

使用 `presets=` 选择 Newton 物理和渲染后端，推荐使用 `--visualizer viser`
通过浏览器查看仿真。


**使用 Viser 可视化（推荐）：**

```bash
python scripts/reinforcement_learning/rl_games/train.py \
    --task Isaac-Cartpole-Camera-Presets-Direct-v0 \
    presets=newton,newton_renderer \
    --visualizer viser
```

训练启动后，在浏览器中打开 `http://localhost:8080` 即可查看实时仿真画面。

不指定 `--visualizer` 时默认以 headless 模式运行，无需额外添加 `--headless`。
