# 基于 Qwen 大模型的铝/氧化钼铝热剂 CQD 掺杂燃速分析智能体

这是一个 AI4S 课程项目，包含 React + Vite 前端和 FastAPI 后端。后端通过阿里云百炼 Qwen 的 OpenAI 兼容接口生成安全报告，API Key 只从环境变量读取，不会进入前端代码。

当前数据结构面向 9 个样品：`m, mo1, mo2, mo3, mo4, mb1, mb2, mb3, mb4`。其中 `m` 为未掺杂基准，`mo1-mo4` 为橘子皮碳量子点样品，`mb1-mb4` 为香蕉皮碳量子点样品。

系统安全边界：不输出危险配方、不输出具体制备比例、不指导提升爆炸或燃烧威力，只做已测实验数据分析、图像证据质量评估、燃速趋势解释和复测候选排序。

## 项目结构

```text
ai4s-qwen-thermite-agent/
  backend/
    app/
      data/demo_experiments.csv
      main.py
      qwen_client.py
      safety.py
      tools.py
    .env.example
    Dockerfile
    requirements.txt
  frontend/
    src/App.jsx
    src/main.jsx
    src/styles.css
    index.html
    package.json
    vite.config.js
  .github/workflows/pages.yml
  README.md
```

## CSV 输入字段

必需字段：

```text
sample_id, thermite_system, cqd_source, cqd_concentration,
video_fps, frame_start, frame_end, burn_distance_mm,
burn_rate_mm_s, image_quality, note
```

可选图像特征字段：

```text
burn_time_s, flame_area_mm2, flame_length_mm, flame_brightness_mean
```

字段说明：

- `sample_id`：样品编号，例如 `m`, `mo1`, `mb2`。
- `thermite_system`：体系名称，例如 `Al-MoO3`。
- `cqd_source`：CQD 来源，建议用 `none`, `orange_peel`, `banana_peel`。
- `cqd_concentration`：碳量子点浓度或浓度等级。可填你的实际浓度数值，也可先用 0-4 等级。
- `video_fps`：高速视频拍摄帧率。
- `frame_start` / `frame_end`：石英玻璃管图像中燃烧前沿起止帧。
- `burn_distance_mm`：对应的燃烧距离。
- `burn_time_s`：可选；若为空，后端自动用 `(frame_end - frame_start) / video_fps` 计算。
- `burn_rate_mm_s`：目标性能参数；若为空或小于等于 0，后端自动用 `burn_distance_mm / burn_time_s` 计算。
- `flame_area_mm2`：可选，从高速燃烧截图分割得到的火焰面积。
- `flame_length_mm`：可选，从截图提取的火焰长度。
- `flame_brightness_mean`：可选，截图中火焰区域平均亮度或灰度指标。
- `image_quality`：图像证据质量，建议用 `high`, `medium`, `low`。
- `note`：备注，例如帧边界不确定、图像模糊、建议复测等。

## 后端工具函数

- `load_experiment_data`：读取用户上传 CSV 或 demo CSV，校验字段，并自动计算 `burn_time_s` 与 `burn_rate_mm_s`。
- `summarize_data`：生成燃速统计、浓度相关性、CQD 来源分组、缺失值和图像质量摘要。
- `train_surrogate_model`：使用 `RandomForestRegressor` 训练少样本燃速代理模型。
- `rank_candidates`：综合预测燃速和图像质量，对样品进行复测候选排序。
- `generate_report`：调用 Qwen OpenAI 兼容接口生成安全中文报告；未配置 Key 时返回本地 fallback 报告。

## 本地运行

### 后端

```powershell
cd "C:\Users\well done\ai4s-qwen-thermite-agent\backend"
python -m pip install -r requirements.txt -i https://pypi.org/simple
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --env-file .env
```

`.env` 示例：

```env
DASHSCOPE_API_KEY=你的百炼APIKey
QWEN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
QWEN_MODEL=qwen-plus
ALLOWED_ORIGINS=http://localhost:5173,http://127.0.0.1:5173
```

检查：

```powershell
Invoke-RestMethod http://127.0.0.1:8000/api/health
```

### 前端

```powershell
cd "C:\Users\well done\ai4s-qwen-thermite-agent\frontend"
npm install
npm run dev -- --host 127.0.0.1 --port 5173
```

打开：

```text
http://127.0.0.1:5173/
```

## 输出内容

前端会展示：

- 数据摘要：样品数、燃速均值、燃速范围、浓度范围、浓度-燃速相关性、模型 MAE。
- CQD 来源分组：未掺杂、橘子皮 CQD、香蕉皮 CQD 的平均燃速与浓度范围。
- 智能体回答：Qwen 基于数据摘要、模型指标和候选排序生成的安全分析。
- 复测候选表：样品编号、CQD 来源、浓度、fps、燃烧时间、距离、图像质量、实测燃速、预测燃速、排序理由。

## API

### `POST /api/analyze`

请求类型：`multipart/form-data`

字段：

- `question`：用户问题，必填。
- `file`：CSV 文件，可选；留空时使用内置 9 样品 demo CSV。

curl 测试：

```powershell
curl.exe -X POST "http://127.0.0.1:8000/api/analyze" `
  -F "question=请比较橘子皮 CQD 与香蕉皮 CQD 样品的燃速趋势"
```

## GitHub Pages 部署

前端可部署到 GitHub Pages，但后端不能放在 GitHub Pages 上，因为后端需要保存 API Key 并调用百炼接口。部署方式是：

1. 前端：GitHub Pages。
2. 后端：阿里云函数计算、阿里云 ECS/容器服务、Render、Railway、Fly.io 等支持环境变量的平台。
3. 在 GitHub 仓库 `Settings -> Secrets and variables -> Actions -> Variables` 添加：

```text
VITE_API_BASE_URL=https://你的后端域名
```

4. 在后端平台配置：

```text
DASHSCOPE_API_KEY=你的百炼APIKey
QWEN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
QWEN_MODEL=qwen-plus
ALLOWED_ORIGINS=https://你的GitHub用户名.github.io
```

5. 推送到 `main` 分支后，`.github/workflows/pages.yml` 会自动构建并部署前端。

## 安全说明

- 前端不保存、不读取、不传递 Qwen API Key。
- API Key 仅由后端从 `DASHSCOPE_API_KEY` 或 `BAILIAN_API_KEY` 环境变量读取。
- 后端固化系统安全提示词。
- 候选排序只用于复测优先级和数据质量改进，不生成新的配方比例或制备流程。
