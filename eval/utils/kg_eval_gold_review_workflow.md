# KG Eval Gold 预标注与人工审核操作说明

本文档用于记录 `AIGC-EDU` 图谱构建 gold 集的预标注、人工审核与 schema 变更后的重启流程，便于后续重复执行。

## 1. 当前文件约定

- 原始抽样样本：
  - `eval/kg_eval/dataset/AIGC-EDU-kgval.json`
- 旧版 gold 工作文件（冻结为历史参考，不再继续写入）：
  - `eval/kg_eval/dataset/AIGC-EDU-kgval.gold.json`
- 新版 gold 工作文件（当前应继续使用）：
  - `eval/kg_eval/dataset/AIGC-EDU-kgval.v2.gold.json`

说明：

- 当知识图谱生成逻辑、节点类型或属性口径发生变化时，不要继续在旧版 `.gold.json` 上累计人工修订。
- 当前新版 gold 统一在 `AIGC-EDU-kgval.v2.gold.json` 中维护。

## 2. 教师模型与运行口径

- 教师模型 profile：
  - `qwen_kg_candidate`
- 该 profile 当前绑定：
  - `qwen-max`

注意：

- 当前 `eval/kg_eval/config.yaml` 的默认 `sample_path` 仍可能指向旧文件。
- 因此在新版 gold 构建阶段，建议每次都显式传入：
  - `--sample-path eval/kg_eval/dataset/AIGC-EDU-kgval.v2.gold.json`

## 3. 状态字段含义

每条样本都带有如下结构：

```json
"kg_eval": {
  "gold": {
    "status": "pending|draft|approved",
    "generator_profile": "",
    "reviewer": "",
    "review_notes": "",
    "updated_at": "",
    "extraction": {
      "entity_types": {},
      "triples": [],
      "attributes": {}
    }
  }
}
```

状态含义如下：

- `pending`：尚未进行模型预标注
- `draft`：已经跑过模型预标注，但还未人工审核完成
- `approved`：已经人工审核完成，可作为后续评估使用的 gold

## 4. schema 变更后的重启流程

### 4.1 新建新版 gold 工作文件

如果需要从头重启一轮新版 gold，使用原始抽样样本重新复制：

```powershell
Copy-Item "eval/kg_eval/dataset/AIGC-EDU-kgval.json" "eval/kg_eval/dataset/AIGC-EDU-kgval.v2.gold.json"
```

原则：

- 从原始抽样样本复制，不要从旧版 `AIGC-EDU-kgval.gold.json` 复制
- 旧版 gold 仅作历史参考，不再作为新版预标注输入

### 4.2 第一批预标注

先跑前 20 条：

```bash
python -m eval.kg_eval.run --config eval/kg_eval/config.yaml generate_gold --sample-path eval/kg_eval/dataset/AIGC-EDU-kgval.v2.gold.json --profile qwen_kg_candidate --max-samples 20
```

执行后预期：

- 前 20 条样本变为 `draft`
- 其余样本保持 `pending`

### 4.3 人工审核

直接打开并修改：

- `eval/kg_eval/dataset/AIGC-EDU-kgval.v2.gold.json`

重点审核字段：

- `kg_eval.gold.extraction.entity_types`
- `kg_eval.gold.extraction.triples`
- `kg_eval.gold.extraction.attributes`

同时补充：

- `kg_eval.gold.reviewer`
- `kg_eval.gold.review_notes`
- `kg_eval.gold.updated_at`

审核完成后，将：

- `kg_eval.gold.status` 改为 `approved`

未审完的样本继续保留 `draft`。

## 5. 如何继续下一批

推荐按 20 篇一批推进：

1. `--max-samples 20`
2. 审核前 20 条并改为 `approved`
3. `--max-samples 40`
4. 审核第 21 到 40 条并改为 `approved`
5. `--max-samples 60`
6. 以此类推

对应命令示例：

```bash
python -m eval.kg_eval.run --config eval/kg_eval/config.yaml generate_gold --sample-path eval/kg_eval/dataset/AIGC-EDU-kgval.v2.gold.json --profile qwen_kg_candidate --max-samples 40
python -m eval.kg_eval.run --config eval/kg_eval/config.yaml generate_gold --sample-path eval/kg_eval/dataset/AIGC-EDU-kgval.v2.gold.json --profile qwen_kg_candidate --max-samples 60
```

原因：

- 当前代码只会跳过 `approved`
- 所有非 `approved` 样本在再次运行 `generate_gold` 时都会被覆盖

因此：

- 如果某条样本已经人工改过，但还停留在 `draft`，再次运行时仍会被新结果覆盖
- 一批审核完成后，必须先改成 `approved`，再继续跑下一批

## 6. 人工审核时建议重点检查的内容

### 6.1 实体类型

检查：

- 实体命名是否具体、稳定、可复用
- 类型是否符合当前新版图谱 schema
- 同一实体是否因为大小写、缩写或表述差异被拆成多个版本

### 6.2 三元组关系

检查：

- 三元组格式是否完整：`[头实体, 关系, 尾实体]`
- 关系方向是否正确
- 关系是否能从标题、摘要或论文元数据中得到明确支持
- 是否存在重复或语义等价但写法不同的三元组

### 6.3 属性

检查：

- 属性是否是简洁、事实性的短语
- 不要直接复制整段摘要
- 属性是否挂载到正确的实体上
- 属性表达是否符合你当前新版图谱的属性口径

## 7. 进度检查命令

### 7.1 查看状态分布

```bash
python -c "import json; from collections import Counter; data=json.load(open('eval/kg_eval/dataset/AIGC-EDU-kgval.v2.gold.json','r',encoding='utf-8')); print(Counter((x.get('kg_eval',{}).get('gold',{}).get('status','pending') for x in data)))"
```

典型结果示例：

- `draft: 20, pending: 80`
- `approved: 20, pending: 80`
- `approved: 40, pending: 60`

### 7.2 查看前 30 条状态

```bash
python -c "import json; data=json.load(open('eval/kg_eval/dataset/AIGC-EDU-kgval.v2.gold.json','r',encoding='utf-8')); [print(i+1, x['id'], x.get('kg_eval',{}).get('gold',{}).get('status','pending')) for i,x in enumerate(data[:30])]"
```

## 8. 常见问题

### Q1：为什么文件里还有很多 `pending`？

因为整个文件保存的是完整抽样集，例如 100 条样本。若你只跑了前 20 条预标注：

- 前 20 条会变成 `draft`
- 其余 80 条仍保持 `pending`

这是正常现象。

### Q2：为什么我人工改过的结果又被覆盖了？

因为当前逻辑只跳过 `approved`，不会跳过 `draft`。

也就是说：

- `approved`：安全，不会被重新生成覆盖
- `draft`：再次运行 `generate_gold` 时仍可能被覆盖

### Q3：什么时候再跑评估？

等新版 `v2.gold` 中积累出一批稳定的 `approved` 样本后，再将后续评估入口指向：

- `eval/kg_eval/dataset/AIGC-EDU-kgval.v2.gold.json`

当前阶段只做 gold 集构建，不做 `run` 评估。
