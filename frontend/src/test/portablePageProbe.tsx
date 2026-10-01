import type { ComponentProps } from 'react'
import type PortableAgentMount from '../integrations/portable-agent/PortableAgentMount'

type Props = ComponentProps<typeof PortableAgentMount>
type Capture = { bindings: Array<Record<string, any>>; snapshots: Array<Record<string, any>> }

/** Page tests inspect host-owned bindings. SDK conversations are tested over real HTTP in E2E. */
export const portablePageProbe = {
  props: null as Props | null,
  target: null as Capture | null,
  capture() {
    const props = this.props
    if (!props || props.busy || props.active === false || !this.target) return
    const context = structuredClone(props.pageContext)
    if (!this.target.bindings.some(value => value.page_context.page_instance_id === context.page_instance_id)) {
      this.target.bindings.push({ page_context: context })
    }
    const snapshot = props.capturePageSnapshot?.()
    this.target.snapshots.push({ page_context: context, ...(snapshot ? { page_snapshot: structuredClone(snapshot) } : {}) })
  },
}

export function PortablePageProbe(props: Props) {
  portablePageProbe.props = props
  return <div data-testid="portable-page-binding">{props.renderStatus?.()}</div>
}
