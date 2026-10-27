# 处理脚本

完整安装、输入格式、建库、搜索和验证步骤见[仓库说明](../README.md)。本目录中的 Python 脚本在本目录运行。

主线为 `pdf_native.py` → `native_crop.py` → `embedding.py build` → `embedding.py merge` → `embedding.py search`。全库覆盖检查使用 `native_coverage.py`；修复已有裁片时使用 `native_sweep.py`，需在初始切题批次完成后执行。

`pdf-fast/resume-native.py --data-repair` 编排已有资料的增量重切、八卡消费与发布。`native_sweep.py --check` 使用本地真实校准资料；不含题库的全新克隆先运行仓库说明中的基础检查。

运行生成的目录、模型与索引已由 `.gitignore` 排除。请保留 `native-assets/` 中固定图片入口和索引的相对位置。
