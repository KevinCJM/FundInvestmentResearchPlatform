/** 投前产物版本与上游状态的线上类型，口径见 docs/pre-investment/versioning.md。 */
import type { UpstreamRef, Usability, UsabilityReason, VersionInfo } from './ltcmaContract.generated'

export type { UpstreamRef, Usability, UsabilityReason, VersionInfo }
export type Versioned = { version?: VersionInfo; upstream?: UpstreamRef[]; usable?: Usability }
