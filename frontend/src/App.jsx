import { useEffect, useMemo, useState } from "react";
import {
  BarChart3,
  Cloud,
  FileUp,
  FlaskConical,
  Loader2,
  Send,
  ShieldCheck,
  Wifi,
} from "lucide-react";

const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

const exampleQuestions = [
  "请比较橘子皮 CQD 与香蕉皮 CQD 样品的燃速趋势。",
  "碳量子点浓度与燃速之间是否存在先升后降的趋势？",
  "哪些样品最值得优先复测高速视频和石英玻璃管燃速？",
];

function App() {
  const [file, setFile] = useState(null);
  const [question, setQuestion] = useState(exampleQuestions[0]);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [backendStatus, setBackendStatus] = useState({
    checked: false,
    ok: false,
    qwenConfigured: false,
  });

  const apiLabel = useMemo(() => API_BASE.replace(/^https?:\/\//, ""), []);
  const aiOnline = backendStatus.ok && backendStatus.qwenConfigured;

  useEffect(() => {
    let active = true;
    async function checkBackend() {
      try {
        const response = await fetch(`${API_BASE}/api/health`);
        const data = await response.json();
        if (!active) return;
        setBackendStatus({
          checked: true,
          ok: Boolean(data.ok),
          qwenConfigured: Boolean(data.qwen_configured),
        });
      } catch {
        if (!active) return;
        setBackendStatus({ checked: true, ok: false, qwenConfigured: false });
      }
    }
    checkBackend();
    return () => {
      active = false;
    };
  }, []);

  async function handleSubmit(event) {
    event.preventDefault();
    setError("");
    setLoading(true);
    setResult(null);

    const form = new FormData();
    form.append("question", question);
    if (file) form.append("file", file);

    try {
      const response = await fetch(`${API_BASE}/api/analyze`, {
        method: "POST",
        body: form,
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || "分析请求失败");
      }
      setResult(data);
      setBackendStatus((current) => ({ ...current, checked: true, ok: true }));
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="app-shell">
      <section className="workspace">
        <header className="topbar">
          <div>
            <p className="eyebrow">AI4S Course Project</p>
            <h1>基于 Qwen 大模型的铝/氧化钼铝热剂 CQD 掺杂燃速分析智能体</h1>
          </div>
          <div className="cloud-status" aria-label="云端后端状态">
            <span className={aiOnline ? "status-dot online" : "status-dot"} />
            {aiOnline ? "AI 在线" : backendStatus.ok ? "后端在线" : "连接检测中"}
          </div>
        </header>

        <div className="layout">
          <form className="control-panel" onSubmit={handleSubmit}>
            <div className="panel-heading">
              <FlaskConical size={20} />
              <h2>燃速数据与问题</h2>
            </div>

            <CloudStatusPanel
              apiLabel={apiLabel}
              backendStatus={backendStatus}
            />

            <label className="upload-zone">
              <FileUp size={24} />
              <span>{file ? file.name : "上传 CSV，或留空使用 9 样品 demo 数据"}</span>
              <input
                type="file"
                accept=".csv,text/csv"
                onChange={(event) => setFile(event.target.files?.[0] || null)}
              />
            </label>

            <label className="field-label" htmlFor="question">
              用户问题
            </label>
            <textarea
              id="question"
              value={question}
              onChange={(event) => setQuestion(event.target.value)}
              rows={7}
              placeholder="输入关于 CQD 来源、浓度、视频燃速、图像质量或复测排序的问题"
            />

            <div className="quick-questions">
              {exampleQuestions.map((item) => (
                <button
                  key={item}
                  type="button"
                  onClick={() => setQuestion(item)}
                  className={question === item ? "chip active" : "chip"}
                >
                  {item}
                </button>
              ))}
            </div>

            <button className="primary-action" type="submit" disabled={loading || !backendStatus.ok}>
              {loading ? <Loader2 className="spin" size={18} /> : <Send size={18} />}
              {loading ? "分析中" : "运行智能体"}
            </button>

            <p className="safety-copy">
              Qwen API Key 只保存在云端后端环境变量中。浏览器前端只调用后端接口，不读取、不保存、不显示 Key。
            </p>
          </form>

          <section className="results-panel">
            {error && <div className="error-banner">{error}</div>}
            {!result && !error && (
              <div className="empty-state">
                <BarChart3 size={36} />
                <h2>等待分析</h2>
                <p>提交问题后，这里会显示燃速摘要、CQD 来源对比、智能体回答与复测候选排序。</p>
              </div>
            )}
            {result && (
              <>
                {result.warning && <div className="warning-banner">{result.warning}</div>}
                <SummaryGrid summary={result.summary} metrics={result.model_metrics} />
                <AnswerBlock answer={result.answer} provider={result.provider} />
                <CandidateTable rows={result.candidates} />
              </>
            )}
          </section>
        </div>
      </section>
    </main>
  );
}

function CloudStatusPanel({ apiLabel, backendStatus }) {
  return (
    <div className="connection-panel">
      <div className="connection-row">
        <Cloud size={17} />
        <span>云端后端</span>
        <strong>{apiLabel}</strong>
      </div>
      <div className="connection-grid">
        <StatusItem
          icon={<Wifi size={16} />}
          label="后端连接"
          value={backendStatus.ok ? "已连接" : backendStatus.checked ? "未连接" : "检测中"}
          good={backendStatus.ok}
        />
        <StatusItem
          icon={<ShieldCheck size={16} />}
          label="Qwen Key"
          value={backendStatus.qwenConfigured ? "云端已配置" : "云端未配置"}
          good={backendStatus.qwenConfigured}
        />
      </div>
    </div>
  );
}

function StatusItem({ icon, label, value, good }) {
  return (
    <div className={good ? "status-item good" : "status-item"}>
      {icon}
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function SummaryGrid({ summary, metrics }) {
  const burnRate = summary.numeric_stats?.burn_rate_mm_s || {};
  const concentration = summary.numeric_stats?.cqd_concentration || {};
  const corr = summary.correlations_to_burn_rate?.cqd_concentration;
  return (
    <section className="section-block">
      <div className="section-title">
        <BarChart3 size={18} />
        <h2>数据摘要</h2>
      </div>
      <div className="metric-grid">
        <Metric label="样品数" value={summary.rows} />
        <Metric label="燃速均值 mm/s" value={burnRate.mean ?? "N/A"} />
        <Metric label="燃速范围 mm/s" value={`${burnRate.min ?? "N/A"} - ${burnRate.max ?? "N/A"}`} />
        <Metric label="浓度范围" value={`${concentration.min ?? "N/A"} - ${concentration.max ?? "N/A"}`} />
        <Metric label="浓度-燃速相关" value={corr ?? "N/A"} />
        <Metric label="模型 MAE" value={metrics.mae ?? "N/A"} />
      </div>
      <SourceSummary groups={summary.grouped_by_source || []} />
    </section>
  );
}

function SourceSummary({ groups }) {
  if (!groups.length) return null;
  return (
    <div className="table-wrap compact-table">
      <table>
        <thead>
          <tr>
            <th>CQD 来源</th>
            <th>样品数</th>
            <th>平均燃速 mm/s</th>
            <th>浓度范围</th>
          </tr>
        </thead>
        <tbody>
          {groups.map((group) => (
            <tr key={group.cqd_source}>
              <td>{sourceLabel(group.cqd_source)}</td>
              <td>{group.sample_count}</td>
              <td>{group.mean_burn_rate_mm_s}</td>
              <td>{group.min_concentration} - {group.max_concentration}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function Metric({ label, value }) {
  return (
    <div className="metric">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function AnswerBlock({ answer, provider }) {
  return (
    <section className="section-block">
      <div className="section-title">
        <Send size={18} />
        <h2>智能体回答</h2>
        <span className="provider">{provider}</span>
      </div>
      <div className="answer">{answer}</div>
    </section>
  );
}

function CandidateTable({ rows }) {
  return (
    <section className="section-block">
      <div className="section-title">
        <FlaskConical size={18} />
        <h2>复测候选排序</h2>
      </div>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>样品</th>
              <th>CQD 来源</th>
              <th>浓度</th>
              <th>FPS</th>
              <th>燃烧时间 s</th>
              <th>距离 mm</th>
              <th>图像质量</th>
              <th>实测燃速</th>
              <th>预测燃速</th>
              <th>排序分</th>
              <th>理由</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={row.sample_id}>
                <td>{row.sample_id}</td>
                <td>{sourceLabel(row.cqd_source)}</td>
                <td>{row.cqd_concentration}</td>
                <td>{row.video_fps}</td>
                <td>{row.burn_time_s}</td>
                <td>{row.burn_distance_mm}</td>
                <td>{qualityLabel(row.image_quality)}</td>
                <td>{row.observed_burn_rate_mm_s}</td>
                <td>{row.predicted_burn_rate_mm_s}</td>
                <td>{row.ranking_score}</td>
                <td>{row.reason}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function sourceLabel(value) {
  const labels = {
    none: "未掺杂基准",
    orange_peel: "橘子皮 CQD",
    banana_peel: "香蕉皮 CQD",
  };
  return labels[value] || value;
}

function qualityLabel(value) {
  const labels = {
    high: "高",
    medium: "中",
    low: "低",
  };
  return labels[value] || value;
}

export default App;
