import type { GraphNodeSchema } from '../components/computation-graph/types'

// Offline fixture for the backend graph_schemas contract; never production fallback.
export const etlGraphSchemas: GraphNodeSchema[] = [
  ['download', '下载原始数据', null, 'raw_batch', false],
  ['map', '字段映射', 'raw_batch', 'mapped_batch', false],
  ['resolve', '多源取值', 'mapped_batch', 'resolved_table', true],
  ['snapshot', '指标快照计算', 'resolved_table', 'snapshot', true],
  ['task', '数据集任务', 'workspace', 'workspace', false],
].map(([id, label, input, output, multiple]) => ({
  id: String(id), label: String(label), category_label: String(label),
  inputs: [...(input ? [{ id: 'data', label: '数据输入', value_type: String(input), multiple: Boolean(multiple), required: id !== 'task' }] : []), { id: 'after', label: '等待完成', value_type: 'control', multiple: true }],
  outputs: [{ id: 'data', label: '数据输出', value_type: String(output) }, { id: 'done', label: '完成', value_type: 'control' }],
}))
