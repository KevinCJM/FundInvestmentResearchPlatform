"""Explicit, user-requested dependency planning from registered task contracts."""
from .etl_models import EtlDefinition
from .models import CenterError
from .task_catalog import get_task


def plan_dependencies(definition: EtlDefinition) -> EtlDefinition:
    steps, producers = [], {}
    for original in definition.execution_steps():
        if original.kind != 'task':
            raise CenterError('ETL_DEPENDENCY_PLAN_UNSUPPORTED', '自动梳理只适用于登记的数据集任务；其他节点请手动设置连线。')
        spec = get_task(original.task_id)
        required = list(spec['requires'])
        # Coverage reads price files, not constituents/weights. Include only
        # explicitly selected histories; an absent history is not fabricated.
        if spec['action'] == 'index_coverage':
            required += [key for key in ('index_domestic', 'index_industry', 'index_concept',
                         'index_global', 'index_futures', 'index_valuation') if key in producers]
        missing = [key for key in required if key not in producers]
        if missing:
            raise CenterError('ETL_TASK_DEPENDENCY', f'{original.name} 缺少前置任务：{", ".join(missing)}。')
        inputs = list(dict.fromkeys(producers[key] for key in required))
        after = [steps[-1].id] if steps and steps[-1].id not in inputs else []
        steps.append(original.model_copy(update={'inputs': inputs, 'after': after}))
        producers.update({key: original.id for key in spec['provides']})
    return EtlDefinition.model_validate({**definition.model_dump(mode='json'), 'graph_version': 1,
                                        'steps': [s.model_dump(mode='json') for s in steps]})
