# 基于 Qwen 大模型的铝/氧化钼铝热剂 CQD 掺杂燃速分析智能体

这是一个 AI4S 课程项目，前端使用 React + Vite，后端使用 FastAPI。系统通过阿里云百炼 Qwen 的 OpenAI 兼容接口生成安全的数据分析报告，Qwen API Key 只由后端从环境变量读取，不会写入前端代码、GitHub Pages 或浏览器。

当前项目面向 9 个少样本实验数据：`m, mo1, mo2, mo3, mo4, mb1, mb2, mb3, mb4`。其中 `m` 为未掺杂基准样品，`mo1-mo4` 为橘子皮碳量子点 CQD 掺杂样品，`mb1-mb4` 为香蕉皮碳量子点 CQD 掺杂样品。核心分析对象是石英玻璃管高速图像换算得到的燃速，以及 CQD 来源、浓度等级与燃速之间的趋势关系。

系统安全边界：不输出危险配方，不输出具体制备比例，不指导提升爆炸或燃烧威力；只做已测科研数据分析、趋势解释、模型不确定性说明和复测候选排序。

## 当前功能

- CSV 数据上传；未上传时使用内置 9 样品 demo CSV。
- 用户问题输入，可围绕 CQD 来源、浓度、燃速趋势、复测优先级提问。
- 数据摘要展示，包括样品数、燃速均值、燃速范围、浓度范围、相关性和模型 MAE。
- 数据来源诊断，显示当前读取的文件名、来源、样品编号和 12 位 SHA 数据指纹。
- Qwen 智能体回答，若云端 Key 未配置或调用失败，会回退到本地 deterministic report。
- 候选实验复测排序表，显示传播时间、实测燃速、预测燃速、排序分和复测理由。
- 前端连接云端后端，页面显示后端连接状态和 Qwen Key 是否已在云端配置。

## 项目结构

```text
ai4s-qwen-thermite-agent/
  backend/
    app/
      data/
        demo_experiments.csv       # 默认 demo 数据
        demo_thermite_cqd.csv      # 备用/课程演示 CSV
      __init__.py
      main.py                      # FastAPI 入口
      qwen_client.py               # Qwen OpenAI 兼容接口客户端
      safety.py                    # 安全提示词与危险意图过滤
      tools.py                     # 数据加载、摘要、模型、排序、报告工具函数
    .env.cloud.example             # 云端环境变量示例
    Dockerfile
    Procfile
    requirements.txt
  frontend/
    src/
      App.jsx
      main.jsx
      styles.css
    index.html
    package.json
    vite.config.js
  .github/workflows/pages.yml      # GitHub Pages 自动部署
  DEPLOYMENT.md
  README.md
```

## CSV 字段

### 必需字段

```text
sample_id, thermite_system, cqd_source, cqd_concentration,
video_fps, burn_distance_mm, note
```

### 可选字段

```text
replicate_id, frame_start, frame_end, frame_period_us,
wave_distance_m, wave_time_us, burn_time_s,
burn_rate_m_s, burn_rate_mm_s, burn_rate_mean_m_s,
performance_index,
flame_area_mm2, flame_length_mm, flame_brightness_mean,
morphology_note
```

### 字段说明

- `sample_id`：样品编号，例如 `m`, `mo1`, `mb4`。
- `thermite_system`：体系名称，例如 `Al-MoO3`。
- `cqd_source`：CQD 来源，建议使用 `none`, `orange_peel`, `banana_peel`。
- `cqd_concentration`：CQD 浓度或浓度等级；当前 demo 使用 `0-4` 等级。
- `video_fps`：高速视频拍摄帧率，例如 `250000`。
- `burn_distance_mm`：石英玻璃管或图像标定得到的传播距离，单位 mm。
- `note`：备注，例如基准样品、最高燃速、建议复测等。
- `wave_time_us`：传播时间，单位微秒；例如 `100` 表示 `100 μs`。
- `burn_time_s`：传播时间，单位秒；例如 `0.0001` 表示 `100 μs`。
- `burn_rate_m_s`：燃速，单位 m/s；这是当前模型的主目标列。
- `burn_rate_mm_s`：燃速，单位 mm/s；若只提供这一列，后端会自动换算成 m/s。
- `frame_start` / `frame_end`：可选，高速视频起止帧号。
- `wave_distance_m`：可选，传播距离，单位 m；若为空，后端用 `burn_distance_mm / 1000`。
- `performance_index`：可选，归一化性能指标或课程展示指标。
- `flame_area_mm2`、`flame_length_mm`、`flame_brightness_mean`：可选，来自火焰形态图片的图像特征。
- `morphology_note`：可选，火焰形态或图像观察备注。

### 时间与燃速计算优先级

后端会统一把目标燃速处理为 `burn_rate_m_s`，并统一把表格中的传播时间显示为 `μs`。

时间优先级如下：

1. 如果 CSV 有 `wave_time_us`，优先使用它。
2. 否则如果有 `burn_time_s`，使用它并换算为微秒显示。
3. 否则尝试用 `wave_distance_m / burn_rate_m_s` 反推时间。
4. 最后才尝试用 `(frame_end - frame_start) / video_fps` 计算。

因此，如果 demo 文件中 `burn_time_s = 0.0001`，前端候选表应显示 `100.0 μs`。

## 后端工具函数

后端在 `backend/app/tools.py` 中实现以下工具函数：

- `load_experiment_data`：读取上传 CSV 或内置 demo CSV，校验字段，补齐可选列，记录数据指纹，并规范化时间与燃速单位。
- `summarize_data`：生成数据摘要、缺失值统计、CQD 来源分组、浓度-燃速相关性和 top 样品。
- `train_surrogate_model`：使用 `RandomForestRegressor` 建立少样本燃速代理模型。
- `rank_candidates`：根据预测燃速生成复测候选排序，仅用于已测样品复核，不生成新配方。
- `generate_report`：调用 Qwen 生成中文安全分析报告；未配置 Key 或调用失败时返回本地 fallback 报告。

## 本地运行

### 1. 启动后端

在 `backend/.env` 中配置环境变量：

```env
DASHSCOPE_API_KEY=你的百炼APIKey
QWEN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
QWEN_MODEL=qwen-plus
ALLOWED_ORIGINS=http://localhost:5173,http://127.0.0.1:5173
```

启动：

```powershell
cd "C:\Users\well done\ai4s-qwen-thermite-agent\backend"
python -m pip install -r requirements.txt -i https://pypi.org/simple
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --env-file .env
```

健康检查：

```powershell
Invoke-RestMethod http://127.0.0.1:8000/api/health
```

正常返回示例：

```json
{"ok": true, "qwen_configured": true}
```

### 2. 启动前端

```powershell
cd "C:\Users\well done\ai4s-qwen-thermite-agent\frontend"
npm install
npm run dev -- --host 127.0.0.1 --port 5173
```

打开：

```text
http://127.0.0.1:5173/
```

## API

### `GET /api/health`

返回后端是否在线，以及 Qwen Key 是否已配置。

### `POST /api/analyze`

请求类型：`multipart/form-data`

字段：

- `question`：用户问题，必填。
- `file`：CSV 文件，可选；留空时使用 `backend/app/data/demo_experiments.csv`。

PowerShell 中建议使用 `curl.exe` 测试 multipart 表单：

```powershell
curl.exe -X POST "http://127.0.0.1:8000/api/analyze" `
  -F "question=请比较橘子皮 CQD 与香蕉皮 CQD 样品的燃速趋势"
```

上传 CSV 测试：

```powershell
curl.exe -X POST "http://127.0.0.1:8000/api/analyze" `
  -F "question=请分析当前 CSV 的燃速趋势" `
  -F "file=@C:\Users\well done\ai4s-qwen-thermite-agent\backend\app\data\demo_experiments.csv"
```

## 前端输出

前端会展示：

- 云端状态：后端连接状态、Qwen Key 是否云端已配置。
- 数据摘要：样品数、燃速均值、燃速范围、浓度范围、浓度-燃速相关、模型 MAE。
- 数据来源：当前 CSV 文件名、来源、数据指纹、样品编号。
- CQD 来源分组：未掺杂、橘子皮 CQD、香蕉皮 CQD 的平均燃速与浓度范围。
- 智能体回答：基于数据摘要、模型指标和候选排序生成的安全分析。
- 复测候选排序：样品编号、CQD 来源、浓度、传播时间 μs、实测燃速 m/s、预测燃速 m/s、排序分和理由。

## GitHub Pages 与云端后端部署

当前推荐架构：

```text
GitHub Pages React 前端
  -> VITE_API_BASE_URL
  -> 云端 FastAPI 后端
  -> DASHSCOPE_API_KEY 环境变量
  -> 阿里云百炼 Qwen OpenAI 兼容接口
```

前端部署到 GitHub Pages，后端部署到支持环境变量的平台，例如阿里云函数计算、ECS、容器服务、Render、Railway 或 Fly.io。后端不能部署到 GitHub Pages，因为它需要保存 API Key 并调用百炼接口。

### 前端 GitHub Pages

仓库包含 `.github/workflows/pages.yml`。在 GitHub 仓库设置中：

1. 进入 `Settings -> Pages`。
2. 将 `Build and deployment -> Source` 设为 `GitHub Actions`。
3. 进入 `Settings -> Secrets and variables -> Actions -> Variables`。
4. 添加变量：

```text
VITE_API_BASE_URL=https://你的后端域名
```

这里保存的是后端地址，不是 API Key。

推送到 `main` 分支后，GitHub Actions 会自动构建并部署前端。

### 阿里云函数计算后端

云端环境变量建议：

```env
DASHSCOPE_API_KEY=你的百炼APIKey
QWEN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
QWEN_MODEL=qwen-plus
ALLOWED_ORIGINS=https://你的GitHub用户名.github.io
```

如果你的 GitHub Pages 地址是：

```text
https://xiwowangyi626.github.io/ai4s-qwen-thermite-agent/
```

`ALLOWED_ORIGINS` 写 origin 即可：

```env
ALLOWED_ORIGINS=https://xiwowangyi626.github.io
```

后端部署包位置：

```text
backend/backend.zip
```

当后端代码有更新时，需要重新生成并上传 `backend/backend.zip` 到函数计算，否则云端仍然运行旧逻辑。

重新打包示例：

```powershell
cd "C:\Users\well done\ai4s-qwen-thermite-agent\backend"
Compress-Archive -Path "app","Dockerfile","Procfile","requirements.txt" -DestinationPath ".\backend.zip" -Force
```

部署完成后访问：

```text
https://你的后端域名/api/health
```

应返回：

```json
{"ok": true, "qwen_configured": true}
```

## 常见问题

### 为什么不同问题下候选排序一样？

候选排序表主要由 CSV 数据和代理模型决定，不由问题文本决定。同一个 CSV 换不同问题，候选排序通常会相同；智能体回答会根据问题重点变化。

### 为什么上传不同 CSV 结果仍然很像？

如果两个 CSV 的 `burn_rate_m_s` 基本一致，按燃速排序的候选表也会很像。页面会显示数据指纹，可以用来确认后端实际读取的是不是同一个文件内容。

### 为什么出现 `Failed to fetch` 或无法连接云端后端？

常见原因：

- 云函数没有部署或实例临时不可用。
- `VITE_API_BASE_URL` 指向了错误的后端地址。
- 后端 `ALLOWED_ORIGINS` 没有包含 GitHub Pages 的 origin。
- 后端代码更新后没有重新上传 `backend.zip`。

### 修改 CSV 后报缺列怎么办？

必需列名不能改，也不能删除：

```text
sample_id, thermite_system, cqd_source, cqd_concentration,
video_fps, burn_distance_mm, note
```

新增其它列可以，后端会忽略未知列；如果想让某列参与模型，需要在 `backend/app/tools.py` 中加入对应字段处理。

## 安全说明

- 前端不保存、不读取、不显示 Qwen API Key。
- API Key 只从后端环境变量读取，支持 `DASHSCOPE_API_KEY` 或 `BAILIAN_API_KEY`。
- 后端内置安全提示词与危险意图拒答逻辑。
- 系统只分析已测实验数据，不生成新配方比例、制备步骤、点火操作或威力提升建议。
- 少样本模型只用于课程展示、趋势解释和复测优先级排序，不能当作外推结论。

