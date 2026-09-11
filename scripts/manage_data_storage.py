"""Local offline storage operations; never restart or cancel downloads implicitly."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.data_storage import StorageError, StorageManager
from backend.storage_migration import StorageMigration


def main():
    parser = argparse.ArgumentParser(description='数据存储：状态、启动前迁移、确认清理原副本')
    parser.add_argument('command', choices=('startup', 'status', 'cleanup-backup', 'cancel-plan', 'attach'))
    parser.add_argument('--path', help='attach：已有数据目录的绝对路径')
    parser.add_argument('--confirm', help='清理必须明确提供迁移 ID；删除不可恢复')
    args = parser.parse_args()
    manager = StorageManager(ROOT)
    migration = StorageMigration(manager, lambda message: print(message, flush=True))
    try:
        if args.command == 'attach':
            checked = manager.probe_existing(args.path)
            if args.confirm != checked['id']:
                raise StorageError('STORAGE_CONFIRMATION_REQUIRED', f"接入将共用配置、记录和凭据，不复制数据。请提供 --confirm {checked['id']} 确认该数据目录。")
            manager.save_attachment(args.path, manager.config()['revision'], args.confirm)
            migration.startup()
        elif args.command == 'startup':
            migration.startup()
        elif args.command == 'cleanup-backup':
            migration.cleanup_backup(args.confirm)
        elif args.command == 'cancel-plan':
            config = manager.config()
            if not config['pending'] or args.confirm != config['pending']['id']:
                raise StorageError('STORAGE_CONFIRMATION_REQUIRED', '请通过 status 查看并确认准确的待迁移 ID。')
            manager.cancel_plan(config['revision'])
            print('迁移计划已取消。原数据和旧暂存文件均保留，可重新启动服务。')
        else:
            print(json.dumps(manager.status(), ensure_ascii=False, indent=2))
    except (StorageError, OSError) as exc:
        print(f'[存储操作失败] {exc.message if isinstance(exc, StorageError) else type(exc).__name__}。未自动删除原数据。', file=sys.stderr)
        print('使用 ./start_services.sh storage-status 查看详情；复制阶段失败可 storage-cancel <迁移ID> 取消计划后恢复原目录服务。', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
