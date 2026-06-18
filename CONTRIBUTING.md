# 贡献说明

本仓库按比赛项目组织代码和文档。提交前请确认改动只包含源码、配置、文档和必要的小型报告。

## 分支与提交

- 功能或整理分支建议使用 `codex/<short-description>`、`feature/<short-description>` 或 `fix/<short-description>`。
- 提交信息保持简洁，例如 `organize world robot contest projects`。
- 不要把运行产物、缓存、虚拟环境、原始数据、模型权重或压缩包提交到仓库。

## 提交前检查

```powershell
git status -sb
git diff --stat
git check-ignore -v <path-to-large-or-generated-file>
```

如果 `git status` 中出现 `.venv/`、`__pycache__/`、`*.mat`、`*.npz`、`*.pth`、`*.pt`、`*.joblib`、`*.zip` 等文件，请先确认它们是否被 `.gitignore` 正确排除。

## 项目文档

每个项目建议至少保留：

- 项目目标和比赛背景
- 运行入口
- 依赖安装方式
- 数据获取方式或本地路径约定
- 训练、评估、提交或演示流程
- 参考论文、官方文档或比赛链接

大型数据和模型请放在外部存储，并在 README 中写明下载方式。
