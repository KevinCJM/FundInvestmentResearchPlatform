"""Workspace access-control, responsibility separation, and audit contracts."""

from __future__ import annotations

from ..types import CategoryDefinition, TableDefinition, control_audit_fields, field, foreign_id, primary_id


CATEGORY = CategoryDefinition(
    category_id="security",
    label="权限、安全与审计",
    description="管理工作空间成员、角色、权限、职责分离、敏感数据访问和完整审计轨迹。",
    order=120,
)


def _internal_field(*args, **kwargs):
    kwargs["source_mappable"] = False
    return field(*args, **kwargs)


def _table(
    table_id: str,
    label: str,
    description: str,
    grain: str,
    primary_key: tuple[str, ...],
    fields: tuple,
    *,
    update_strategy: str = "upsert",
) -> TableDefinition:
    return TableDefinition(
        table_id=table_id,
        category_id=CATEGORY.category_id,
        label=label,
        description=description,
        layer="control",
        storage_engine="sqlite",
        storage_location="data/platform.db",
        delivery_phase="next",
        grain=grain,
        primary_key=primary_key,
        update_strategy=update_strategy,  # type: ignore[arg-type]
        fields=fields,
        source_mappable=False,
        pit_supported=False,
    )


TABLES = (
    _table(
        "security.principal",
        "访问主体",
        "统一用户、服务账户和自动化任务身份，不保存明文认证凭据。",
        "每个访问主体一条记录",
        ("principal_id",),
        (
            primary_id("principal_id", "访问主体 ID", "用户或服务身份稳定标识。"),
            _internal_field("principal_type", "主体类型", "string", "用户、服务账户或自动任务。", nullable=False, enum_values=("USER", "SERVICE", "SCHEDULER")),
            _internal_field("display_name", "显示名称", "string", "访问主体显示名称。", nullable=False),
            _internal_field("identity_provider", "身份提供方", "string", "本地、OIDC 或其他可信身份提供方。", nullable=False),
            _internal_field("external_subject", "外部身份标识", "string", "身份提供方中的 subject，不保存密码。"),
            _internal_field("email", "邮箱", "string", "用户联系邮箱。"),
            _internal_field("status", "状态", "string", "访问主体状态。", nullable=False, enum_values=("ACTIVE", "SUSPENDED", "DISABLED")),
            _internal_field("last_authenticated_at", "最近认证时间", "timestamp[us, UTC]", "最近成功认证时间。", role="audit"),
            *control_audit_fields(),
        ),
    ),
    _table(
        "security.workspace_member",
        "工作空间成员",
        "把访问主体加入工作空间并控制有效期和成员状态。",
        "每个工作空间和访问主体的一段成员关系",
        ("workspace_member_id",),
        (
            primary_id("workspace_member_id", "成员关系 ID", "工作空间成员关系稳定标识。"),
            foreign_id("workspace_id", "工作空间 ID", "portfolio.workspace.workspace_id", "所属工作空间。"),
            foreign_id("principal_id", "访问主体 ID", "security.principal.principal_id", "加入工作空间的访问主体。"),
            _internal_field("member_type", "成员类型", "string", "所有者、内部成员、外部复核人或服务账户。", nullable=False, enum_values=("OWNER", "INTERNAL", "EXTERNAL_REVIEWER", "SERVICE")),
            _internal_field("valid_from", "生效时间", "timestamp[us, UTC]", "成员关系开始生效时间。", nullable=False, role="effective_time"),
            _internal_field("valid_to", "失效时间", "timestamp[us, UTC]", "成员关系结束时间。", role="effective_time"),
            _internal_field("status", "成员状态", "string", "邀请中、有效、暂停或已移除。", nullable=False, enum_values=("INVITED", "ACTIVE", "SUSPENDED", "REMOVED")),
            *control_audit_fields(),
        ),
        update_strategy="scd2",
    ),
    _table(
        "security.role",
        "权限角色",
        "定义研究、数据、投资、运营、会计、复核和审批等职责角色。",
        "每个工作空间角色一条记录",
        ("role_id",),
        (
            primary_id("role_id", "角色 ID", "权限角色稳定标识。"),
            foreign_id("workspace_id", "工作空间 ID", "portfolio.workspace.workspace_id", "角色所属工作空间；系统角色可为空。", nullable=True),
            _internal_field("code", "角色代码", "string", "工作空间内唯一角色代码。", nullable=False),
            _internal_field("name", "角色名称", "string", "角色名称。", nullable=False),
            _internal_field("description", "说明", "string", "角色职责说明。"),
            _internal_field("system_role", "系统角色", "bool", "是否为平台预置角色。", nullable=False),
            _internal_field("status", "状态", "string", "角色状态。", nullable=False, enum_values=("ACTIVE", "INACTIVE")),
            *control_audit_fields(),
        ),
    ),
    _table(
        "security.permission",
        "权限点",
        "定义可执行的读取、创建、修改、运行、验证、发布、审批、导出和敏感字段访问动作。",
        "每个系统权限点一条记录",
        ("permission_id",),
        (
            primary_id("permission_id", "权限 ID", "系统权限点稳定标识。"),
            _internal_field("code", "权限代码", "string", "全平台唯一权限代码。", nullable=False),
            _internal_field("resource_type", "资源类型", "string", "数据源、研究、组合、交易、核算、报告等资源类型。", nullable=False),
            _internal_field("action", "动作", "string", "读、写、运行、验证、发布、审批、导出等动作。", nullable=False, enum_values=("READ", "CREATE", "UPDATE", "DELETE", "RUN", "VALIDATE", "PUBLISH", "APPROVE", "EXPORT", "READ_SENSITIVE")),
            _internal_field("description", "说明", "string", "权限点说明。"),
            _internal_field("sensitive", "敏感权限", "bool", "是否涉及敏感数据或高风险操作。", nullable=False),
            *control_audit_fields(),
        ),
    ),
    _table(
        "security.role_permission",
        "角色权限",
        "把角色绑定到具体权限点，并允许设置资源范围。",
        "每个角色和权限点的一条授权关系",
        ("role_id", "permission_id"),
        (
            foreign_id("role_id", "角色 ID", "security.role.role_id", "获得权限的角色。"),
            foreign_id("permission_id", "权限 ID", "security.permission.permission_id", "授予的权限点。"),
            _internal_field("resource_scope_json", "资源范围", "json", "工作空间、资产主体、组合或数据集范围。"),
            _internal_field("granted_at", "授权时间", "timestamp[us, UTC]", "角色权限生效时间。", nullable=False, role="audit"),
        ),
    ),
    _table(
        "security.member_role",
        "成员角色",
        "把工作空间成员分配到角色，并保存有效期。",
        "每个成员与角色的一段有效关系",
        ("member_role_id",),
        (
            primary_id("member_role_id", "成员角色 ID", "成员角色关系稳定标识。"),
            foreign_id("workspace_member_id", "成员关系 ID", "security.workspace_member.workspace_member_id", "获得角色的工作空间成员。"),
            foreign_id("role_id", "角色 ID", "security.role.role_id", "分配的角色。"),
            _internal_field("valid_from", "生效时间", "timestamp[us, UTC]", "角色开始生效时间。", nullable=False, role="effective_time"),
            _internal_field("valid_to", "失效时间", "timestamp[us, UTC]", "角色结束时间。", role="effective_time"),
            _internal_field("assignment_status", "分配状态", "string", "有效、暂停或撤销。", nullable=False, enum_values=("ACTIVE", "SUSPENDED", "REVOKED")),
            *control_audit_fields(),
        ),
        update_strategy="scd2",
    ),
    _table(
        "security.separation_rule",
        "职责分离规则",
        "限制同一访问主体同时承担研究、验证、发布、交易、核算和审批中的冲突职责。",
        "每条职责冲突规则一条记录",
        ("separation_rule_id",),
        (
            primary_id("separation_rule_id", "规则 ID", "职责分离规则稳定标识。"),
            foreign_id("workspace_id", "工作空间 ID", "portfolio.workspace.workspace_id", "规则所属工作空间；系统规则可为空。", nullable=True),
            _internal_field("name", "规则名称", "string", "职责分离规则名称。", nullable=False),
            _internal_field("left_role_code", "职责 A", "string", "第一项职责或角色代码。", nullable=False),
            _internal_field("right_role_code", "职责 B", "string", "与职责 A 冲突的职责或角色代码。", nullable=False),
            _internal_field("resource_type", "资源类型", "string", "规则适用资源类型。", nullable=False),
            _internal_field("enforcement", "执行方式", "string", "阻断或警告。", nullable=False, enum_values=("BLOCK", "WARN")),
            _internal_field("enabled", "是否启用", "bool", "规则是否生效。", nullable=False),
            _internal_field("description", "说明", "string", "规则业务说明。"),
            *control_audit_fields(),
        ),
    ),
    _table(
        "security.audit_event",
        "系统审计事件",
        "不可变记录数据、参数、模型、研究、交易、核算、权限、发布、导出和敏感访问行为。",
        "每次受审计操作一条记录",
        ("audit_event_id",),
        (
            primary_id("audit_event_id", "审计事件 ID", "不可变审计事件标识。"),
            foreign_id("workspace_id", "工作空间 ID", "portfolio.workspace.workspace_id", "事件所属工作空间。", nullable=True),
            foreign_id("principal_id", "访问主体 ID", "security.principal.principal_id", "执行操作的身份。", nullable=True),
            _internal_field("event_at", "事件时间", "timestamp[us, UTC]", "操作发生时间。", nullable=False, role="audit"),
            _internal_field("action", "操作", "string", "读取、创建、修改、运行、发布、审批、导出等。", nullable=False),
            _internal_field("resource_type", "资源类型", "string", "被操作资源类型。", nullable=False),
            _internal_field("resource_id", "资源 ID", "string", "被操作资源稳定标识。"),
            _internal_field("resource_version", "资源版本", "string", "被操作资源版本或修订。"),
            _internal_field("result", "操作结果", "string", "成功、拒绝或失败。", nullable=False, enum_values=("SUCCESS", "DENIED", "FAILED")),
            _internal_field("reason_code", "原因代码", "string", "拒绝或失败原因代码。"),
            _internal_field("request_id", "请求 ID", "string", "关联 API 或任务请求标识。"),
            _internal_field("source_ip_hash", "来源地址哈希", "string", "脱敏网络来源标识。"),
            _internal_field("before_hash", "变更前哈希", "string", "修改前资源内容哈希。"),
            _internal_field("after_hash", "变更后哈希", "string", "修改后资源内容哈希。"),
            _internal_field("metadata_json", "审计详情", "json", "不包含敏感明文的扩展审计信息。"),
        ),
        update_strategy="append",
    ),
    _table(
        "security.export_record",
        "数据导出记录",
        "记录 Excel、CSV、报告和研究制品导出的对象、范围、版本、保密级别和水印。",
        "每次数据或报告导出一条记录",
        ("export_record_id",),
        (
            primary_id("export_record_id", "导出记录 ID", "导出行为稳定标识。"),
            foreign_id("workspace_id", "工作空间 ID", "portfolio.workspace.workspace_id", "导出所属工作空间。"),
            foreign_id("principal_id", "访问主体 ID", "security.principal.principal_id", "执行导出的身份。"),
            _internal_field("exported_at", "导出时间", "timestamp[us, UTC]", "导出完成时间。", nullable=False, role="audit"),
            _internal_field("resource_type", "资源类型", "string", "报告、产品、指标、回测、组合或账簿数据。", nullable=False),
            _internal_field("resource_id", "资源 ID", "string", "被导出资源标识。"),
            _internal_field("resource_version", "资源版本", "string", "被导出资源版本。"),
            _internal_field("data_release_id", "数据版本 ID", "string", "导出结果依赖的数据版本。", reference="governance.data_release.data_release_id"),
            _internal_field("format", "导出格式", "string", "Excel、CSV、Parquet、HTML 或 PDF。", nullable=False, enum_values=("XLSX", "CSV", "PARQUET", "HTML", "PDF")),
            _internal_field("confidentiality", "保密级别", "string", "导出结果保密级别。", nullable=False, enum_values=("INTERNAL", "CONFIDENTIAL", "RESTRICTED")),
            _internal_field("watermark", "水印", "string", "导出文件水印内容。"),
            _internal_field("artifact_id", "制品 ID", "string", "生成的导出制品。", reference="research.artifact.artifact_id"),
            _internal_field("checksum", "校验和", "string", "导出文件内容校验和。", role="audit"),
            _internal_field("row_count", "导出行数", "int64", "表格导出行数。", role="measure", unit="row"),
            _internal_field("status", "导出状态", "string", "成功、失败或已撤销。", nullable=False, enum_values=("SUCCEEDED", "FAILED", "REVOKED")),
        ),
        update_strategy="append",
    ),
)
