# 如何启动推理

输入自然语言要求，从已有单题向量索引中检索相关题目截图。推理使用本地 `Qwen3-VL-Embedding-2B`，单张 GPU 即可；不会重新 OCR、切图或编码整个题库。

## 1. 准备环境

在仓库根目录执行。已有可用环境时直接激活即可；新环境使用 Python 3.11：

```bash
# 首次获取代码时跳过 LFS 自动下载，数据统一从 Release 获取。
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/Theodora11-dot/test-repository.git
cd test-repository
python3.11 -m venv .venv
source .venv/bin/activate
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

## 2. 放置模型、索引和题图

从已发布的 [data-v1 Release](https://github.com/Theodora11-dot/test-repository/releases/tag/data-v1) 下载索引、全部题图包和校验文件。全部数据约 21.5 GB；下载、解压和模型合计建议预留至少 50 GB 可用空间。

```bash
# 在仓库根目录执行；支持中断后继续下载。
mkdir -p data-v1
cd data-v1
base_url='https://github.com/Theodora11-dot/test-repository/releases/download/data-v1'
for name in index.sqlite3 release-manifest.json SHA256SUMS; do
  curl -fL --retry 5 -C - -O "$base_url/$name"
done
for part in $(seq -w 1 14); do
  curl -fL --retry 5 -C - -O "$base_url/question-images-$part.tar"
done
sha256sum -c SHA256SUMS
```

校验全部通过后，再放置索引并解压题图：

```bash
mkdir -p ../题库处理/embedding-native-questions
cp index.sqlite3 ../题库处理/embedding-native-questions/index.sqlite3
for archive in question-images-*.tar; do
  tar -xf "$archive" -C ../题库处理
done
cd ..
```

每个 `.tar` 包都能独立解压，无需拼接；JPEG 已压缩，打包保留原图片字节。解压后的资源位置如下：

```text
题库处理/
├── models/Qwen3-VL-Embedding-2B/       # 模型、处理器文件及 download-manifest.json
├── embedding-native-questions/
│   └── index.sqlite3                  # 已发布的单题向量索引
└── native-assets/                     # 索引对应的单题截图，保留内部目录结构
```

模型尚未下载时，在仓库根目录执行：

```bash
cd 题库处理
python download_models.py --owner Qwen Qwen3-VL-Embedding-2B
cd ..
```

Release 包含 185,722 张有效单题截图和对应索引。部分旧记录中的答案链接需要另备 `assets/` 中的答案页；不影响题目检索，答案图片不包含在本次资源包中。模型及编码设置须与建库时一致，程序会检查配置是否匹配。

## 3. 运行查询

在仓库根目录执行：

```bash
source .venv/bin/activate
cd 题库处理
CUDA_VISIBLE_DEVICES=0 python infer.py \
  '给高一学生找三角函数练习，考查正弦函数图像的周期或平移，难度适中' \
  --limit 5
```

替换引号内的要求即可搜索其他题目；`CUDA_VISIBLE_DEVICES=0` 指定使用哪张 GPU。每次启动会加载模型和索引，此命令执行一次查询后退出。`infer.py` 是独立推理入口，只读取现有索引；索引默认位置如上，也可用 `--index /路径/index.sqlite3` 指定。

标准输出返回 JSON 结果，模型加载信息写入标准错误。结果中的 `image` 是相对于 `题库处理/` 的截图路径，`score` 是排序得分，还会返回来源、页码等信息。打开对应图片即可查看原题。

`--limit 5` 最多返回 5 条；当前排序会过滤低于本次最高分 85% 的候选，因此可能少于 5 条。要求中仅部分关键词命中也能返回结果，未命中项不会报错；年级、难度等偏好不保证全部满足。答案解析不参与题图 embedding，检索结果仍需结合截图确认是否符合要求。

连续查询时，在同一个 Python 进程复用模型和索引快照，避免每条查询都重新加载。先进入 `题库处理/`，再使用：

```python
from contextlib import closing
from embedding import Encoder, MAX_PIXELS, check_encoder
from infer import open_index, prepare_search, search

with closing(open_index()) as db:
    check_encoder(db)
    encoder = Encoder(MAX_PIXELS)
    corpus = prepare_search(db, device=encoder.device)
    for query in ('高一三角函数的周期和平移', '高中英语邀请信写作'):
        results = search(db, encoder, query, limit=5, corpus=corpus)
        print(results)
```

这段代码只在开始时加载一次模型和全库向量；后续每条要求只编码查询文本并检索已有向量。
