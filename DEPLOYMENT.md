# 云端部署说明

目标效果：用户只打开 GitHub Pages 网页即可使用智能体；Qwen API Key 只保存在云端后端环境变量中，不进入浏览器。

## 架构

```text
GitHub Pages React 前端
  -> VITE_API_BASE_URL
  -> 云端 FastAPI 后端
  -> DASHSCOPE_API_KEY 环境变量
  -> 阿里云百炼 Qwen OpenAI 兼容接口
```

## 1. 部署后端

后端可以部署到任何支持 Python、HTTP 服务和环境变量的平台，例如阿里云函数计算、阿里云 ECS/容器服务、Render、Railway、Fly.io 等。

后端启动命令：

```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

如果平台没有 `$PORT`，使用：

```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

工作目录应指向：

```text
backend
```

依赖安装命令：

```bash
pip install -r requirements.txt
```

云端后端必须配置环境变量：

```env
DASHSCOPE_API_KEY=你的百炼APIKey
QWEN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
QWEN_MODEL=qwen-plus
ALLOWED_ORIGINS=https://你的GitHub用户名.github.io
```

如果 GitHub Pages 地址是项目页，一般形如：

```text
https://你的GitHub用户名.github.io/仓库名/
```

`ALLOWED_ORIGINS` 只需要写 origin：

```env
ALLOWED_ORIGINS=https://你的GitHub用户名.github.io
```

后端部署完成后，访问：

```text
https://你的后端域名/api/health
```

应返回：

```json
{"ok": true, "qwen_configured": true}
```

## 2. 部署前端到 GitHub Pages

仓库已经包含 `.github/workflows/pages.yml`。

在 GitHub 仓库中进入：

```text
Settings -> Secrets and variables -> Actions -> Variables
```

添加变量：

```text
VITE_API_BASE_URL=https://你的后端域名
```

注意：这里不是 API Key，只是后端地址，可以出现在前端。

然后进入：

```text
Settings -> Pages
```

选择：

```text
Build and deployment -> Source -> GitHub Actions
```

推送到 `main` 分支后，GitHub Actions 会自动构建和部署前端。

## 3. 前端状态显示

前端页面启动时会请求：

```text
/api/health
```

并显示：

- 后端连接：已连接 / 未连接
- Qwen Key：云端已配置 / 云端未配置
- 顶部状态：AI 在线

这个状态只是读取后端返回的布尔值，不会读取或暴露真实 Key。

## 4. 什么不能做

不要把以下内容写入前端代码、`frontend/.env*`、GitHub Pages 变量或浏览器 localStorage：

```env
DASHSCOPE_API_KEY=真实Key
BAILIAN_API_KEY=真实Key
```

前端只保存后端地址：

```env
VITE_API_BASE_URL=https://你的后端域名
```

真实 Key 只能放在云端后端的环境变量里。
