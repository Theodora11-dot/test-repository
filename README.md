# 试卷逐题截图与图片向量检索

将试卷页面按题目切分，再用本地模型编码题图和老师的自然语言要求，返回最多 5 张相关题图。当前主线读取 PDF 自带的文字坐标，不做 OCR，不把答案解析送入题图 embedding。

本仓库集中保存采集、切题、编码、检索、修复、校验及测试代码。图片、PDF、模型权重、向量数据库、运行日志和登录凭证保存在本地；克隆代码不会自动获得这些数据。

## 目录

| 目录或文件 | 用途 |
| --- | --- |
| `北京题库_高中数学_20261021/` | 数学目录采集、预览逐页截图 |
| `北京题库_高中英语_20261021/` | 英语目录采集、预览逐页截图、公开同名目录匹配 |
| `题库处理/pdf_native.py` | PDF 原生坐标、题号、答案分界与页数检查 |
| `题库处理/native_crop.py` | 从原始 JPEG 按坐标裁剪单题 |
| `题库处理/embedding.py` | 本地编码、八分片增量消费、发布、自然语言检索 |
| `题库处理/native_sweep.py` | 复查并增量重切受影响图片 |
| `题库处理/native_coverage.py` | 对照有效裁片、向量、快照、图片指纹和尺寸 |
| `题库处理/pdf-fast/resume-native.py` | 切题、八卡编码及发布进程的异常重试 |
| `题库处理/pdf-fast/evaluate.py` | 固定自然语言要求的 Top5 测试与 HTML 图片报告 |
| `题库处理/pipeline.py` | 共用数据、硬链接工具及可选旧 OCR/标签流程 |
| `evaluation-summary.json` | 最终样本逐图核验结论，不含题库图片和原题正文 |

## 环境

已运行验证的环境为 Linux、Python 3.11、PyTorch 2.6.0+cu124、torchvision 0.21.0+cu124。系统需有 `curl`、Poppler 的 `pdftotext`、`pdfinfo`、`pdftoppm`。当前批处理按 8 个 GPU 分片设计；单卡可依次构建这 8 个分片后合并。

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

`requirements.txt` 记录实际使用的依赖版本。仅运行旧 OCR 流程时再安装 `requirements-ocr.txt`；坐标切题和图片编码不需要 PaddleOCR。

## 输入和模型

保留仓库内三个目录的名称及相对位置。脚本读取两个题库目录各自的 `预览图片清单.csv`，至少需要列 `status,id,pages,path,grade,title`，只处理 `status=已保存` 的资料。`path` 是相对于对应题库目录的试卷图片目录，页文件名为 `page-1.jpg`、`page-01.jpg` 等，数字宽度与该套试卷总页数的位数一致。

已有截图和清单时直接放入对应目录。需要采集时，先运行相应学科的 `collect_catalog.py`，再运行 `capture_previews.py --auth-file /外部私有路径/会话.json`。会话文件使用你自己的已授权登录，不要放进仓库。采集脚本复用已保存页面，认证或限流异常会停止网络请求。

下载本地 embedding 模型：

```bash
cd 题库处理
python download_models.py --owner Qwen Qwen3-VL-Embedding-2B
```

模型保存在 `题库处理/models/Qwen3-VL-Embedding-2B/`；下载器校验新文件的长度及可用的 SHA256，写入 `download-manifest.json`。编码时只读本地模型，输出 2048 维归一化向量。

## 从截图建立索引

以下命令均在 `题库处理/` 内运行，使用上面激活的 Python 环境。

```bash
# 首次构建生成八份资料清单；已有运行任务时不要重新分片。
python pipeline.py prepare

# 已有 PDF 缓存时离线检查，不调用 OCR、不访问网络。
# 缓存：pdf-fast/pdfs/math-资料ID.pdf 或 english-资料ID.pdf。
python pdf_native.py
```

如果只有 JPEG，可以用 `python pdf_native.py --auth-file /外部私有路径/会话.json` 从正常预览接口获取 PDF 并读取坐标。JPEG 本身不包含 PDF 文字坐标。预览 PDF 页数与旧截图不一致时记为 pending，不套用新坐标切旧图。

完成坐标检查后，在具有 8 个可见 GPU 的机器上启动：

```bash
python pdf-fast/resume-native.py
```

它启动 CPU 切图、8 个 GPU 编码消费者和一个发布进程；异常退出后重试。新安装无需先建立旧整页索引，来源标题向量会按资料生成并复用。

也可逐步执行。单卡机器先跑切题，再依次编码分片 0 到 7，最后发布：

```bash
python native_crop.py
for shard in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=0 python embedding.py build --all --shard "$shard" --native-crops
done
python embedding.py merge --native-crops
python native_coverage.py
```

进度在 `pdf-fast/crops-run/` 和 `embedding-native-questions/status-N.json`，覆盖报告为 `pdf-fast/vector-coverage.json`。图片未改变时复用已有向量；向量和日志断点在同一 SQLite 事务中提交。发布时先在临时盘构建快照，再复制到目标盘并原子替换，已有读者继续读取旧快照。

## 搜索和验证

```bash
CUDA_VISIBLE_DEVICES=0 python embedding.py search '给高一学生找三角函数图像与周期的题，难度适中' --limit 5 --native-crops
python embedding.py status --native-crops
python pdf-fast/evaluate.py evaluation-holdout --holdout
```

搜索返回 JSON，包含题图相对路径、来源、页码、原页坐标和得分。评估生成 JSON 和 HTML，可打开 HTML 检查截图。首次评估编码固定要求，有对应缓存时复用查询向量。

排序保留当前验证过的图片向量、来源标题向量和少量原生题干软加分。未命中的关键词不会报错，也不会作为硬条件排除全部候选。当前最多返回 5 条，低于本次最高分 85% 的候选可能不展示。难度等未确认维度不能保证命中；坐标主线不额外调用语言模型填满标签。

```bash
# 无需题库数据、模型下载或 GPU 的基础检查。
python test_pipeline.py
python test_pdf_native.py
python test_embedding.py
python native_coverage.py --check
python pdf-fast/resume-native.py --check
```

测试覆盖裁剪边界、输入校验、部分关键词匹配、硬链接移动、排序、幂等、没有旧整页数据库时的新资料编码、断点回滚恢复及原子发布。存在本地校准资料时，额外运行真实样本检查。

## 当前结果和边界

2026-10-27 的本地快照共 185,722 道单题：数学 116,987，英语 68,735。有效裁片、分片索引、发布快照和图片指纹尺寸的全量对照已通过。这是当前处理范围的覆盖检查，不能替代每张题图的人工校对。

10 条新要求的 Top5 逐张查看结果：33 张完整匹配、8 张部分匹配、7 张不符、2 张不完整；10/10 条要求均至少有一张完整匹配，符合本次约定标准。原 10 组回归测试为 46 张完整匹配、4 张部分匹配，同样 10/10 组有完整匹配。原 10 组曾用于调试规则，新 10 组在检索前冻结且未用于调规则。详见 `evaluation-summary.json`。

跨页题、无法确认公共材料的题和边界不明项暂缓。已知仍有重复、英语语法要求误匹配及个别写作条件漏切，样本结果不代表全库准确率。答案解析不参与题图 embedding；存在旧来源配对数据时保留同套答案链接，尚未普遍逐题对齐。

单题图片通过固定硬链接入口关联索引：同一文件系统中移动或改名整理后的图片不会破坏关联。需保留内部固定入口；跨文件系统复制及其后修改不具备自动同步保证。

可选旧 OCR 入口为 `pipeline.py`，模型为 `models/PaddleOCR-VL-1.6/` 和 `models/Qwen3-8B/`，标签模型路径可用环境变量 `TAG_MODEL` 指定。该流程独立于当前坐标主线，详情执行 `python pipeline.py --help`。
