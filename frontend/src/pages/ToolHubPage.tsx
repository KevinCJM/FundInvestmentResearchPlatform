import { Link } from 'react-router-dom'

interface ToolLink {
  label: string
  description: string
  path: string
}

export default function ToolHubPage({ title, description, tools }: { title: string; description: string; tools: ToolLink[] }) {
  return (
    <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm sm:p-7">
      <span className="inline-flex rounded-full bg-amber-100 px-3 py-1 text-xs font-semibold text-amber-900">部分具备</span>
      <h2 className="mt-4 text-2xl font-bold text-slate-950">{title}</h2>
      <p className="mt-2 max-w-3xl text-sm leading-6 text-slate-600">{description}</p>
      <div className="mt-6 grid gap-4 md:grid-cols-2">
        {tools.map((tool) => (
          <Link key={tool.path} to={tool.path} className="group rounded-xl border border-slate-200 bg-slate-50 p-4 transition hover:border-accent-300 hover:bg-accent-50 focus:outline-none focus:ring-2 focus:ring-accent-500">
            <h3 className="font-semibold text-slate-900 group-hover:text-accent-700">{tool.label}</h3>
            <p className="mt-1 text-sm leading-6 text-slate-600">{tool.description}</p>
            <span className="mt-3 inline-block text-sm font-semibold text-accent-700">打开现有工具 →</span>
          </Link>
        ))}
      </div>
    </div>
  )
}
