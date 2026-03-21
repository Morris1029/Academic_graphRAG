# KG Eval Gold 预标注与人工审核操作说明

本文档用于记录 `AIGC-EDU` 图谱构建 gold 集的预标注与人工审核流程，便于后续重复执行。

## 1. 当前使用的文件与默认配置

- 原始抽样样本文件：
  - `eval/kg_eval/dataset/AIGC-EDU-kgval.json`
- Gold 工作文件：
  - `eval/kg_eval/dataset/AIGC-EDU-kgval.gold.json`
- `kg_eval` 当前默认配置见：
  - `eval/kg_eval/config.yaml`

当前默认约定：

- 默认样本路径：`eval/kg_eval/dataset/AIGC-EDU-kgval.gold.json`
- 默认 gold 预标注模型：`qwen_kg_candidate`
- `qwen_kg_candidate` 实际绑定的是 `qwen-max`

因此，后续一般不需要再显式传 `--sample-path` 和 `--profile`。

## 2. 状态字段的含义

`eval/kg_eval/dataset/AIGC-EDU-kgval.gold.json` 中，每条样本都会带有：

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

其中：

- `pending`：尚未进行模型预标注
- `draft`：已经跑过模型预标注，但还未人工审核完成
- `approved`：已经人工审核完成，可作为后续评估使用的 gold 数据

## 3. 第一步：运行模型预标注

### 3.1 第一批 20 条

运行：

```bash
python -m eval.kg_eval.run --config eval/kg_eval/config.yaml generate_gold --max-samples 20
```

作用：

- 对前 20 条样本执行模型抽取
- 将结果直接写回 `eval/kg_eval/dataset/AIGC-EDU-kgval.gold.json`
- 这 20 条样本的 `status` 会变成 `draft`

### 3.2 后续批次

如果要继续下一批，则递增 `--max-samples`：

```bash
python -m eval.kg_eval.run --config eval/kg_eval/config.yaml generate_gold --max-samples 40
python -m eval.kg_eval.run --config eval/kg_eval/config.yaml generate_gold --max-samples 60
python -m eval.kg_eval.run --config eval/kg_eval/config.yaml generate_gold --max-samples 80
python -m eval.kg_eval.run --config eval/kg_eval/config.yaml generate_gold --max-samples 100
```

逻辑说明：

- 已经是 `approved` 的样本会被跳过
- 不是 `approved` 的样本会被新的模型结果重新覆盖

因此，进入下一批之前，必须先把上一批中已经审核完成的样本改成 `approved`。

## 4. 第二步：人工审核与修正

直接在下面这个文件里修改：

- `eval/kg_eval/dataset/AIGC-EDU-kgval.gold.json`

### 4.1 需要重点检查和修正的字段

每条样本主要审核以下内容：

- `kg_eval.gold.extraction.entity_types`
- `kg_eval.gold.extraction.triples`
- `kg_eval.gold.extraction.attributes`

同时补充：

- `kg_eval.gold.reviewer`
- `kg_eval.gold.review_notes`
- `kg_eval.gold.updated_at`

审核完成后，将：

- `kg_eval.gold.status` 改成 `approved`

尚未审核完的，保留 `draft`。

### 4.2 审核时建议关注的内容

#### 实体类型 `entity_types`

检查：

- 实体名是否具体、清晰，不要保留过于空泛的概念词
- 实体类型是否合理，例如论文、作者、机构、期刊、研究主题、研究方法等
- 同一个实体是否因为命名不一致被拆成多个版本

#### 关系三元组 `triples`

检查：

- 三元组是否完整，必须是 `[头实体, 关系, 尾实体]`
- 关系方向是否正确
- 关系是否来自论文元数据或摘要的明确信息，而不是模型臆断
- 同一含义是否被重复表达为多条几乎相同的三元组

#### 属性 `attributes`

检查：

- 属性是否是简洁的事实性短语
- 不要整段复制摘要
- 属性是否附着在正确的实体上

### 4.3 审核完成后的最小修改模板

可参考如下形式：

```json
"gold": {
  "status": "approved",
  "generator_profile": "qwen_kg_candidate",
  "reviewer": "你的名字",
  "review_notes": "已完成人工校对，修正了作者实体和部分关系方向。",
  "updated_at": "2026-03-21T15:30:00Z",
  "extraction": {
    "entity_types": {
      "实体A": "类型A"
    },
    "triples": [
      ["实体A", "关系", "实体B"]
    ],
    "attributes": {
      "实体A": ["属性1", "属性2"]
    }
  }
}
```

## 5. 第三步：如何继续下一批

推荐的固定节奏如下：

1. `--max-samples 20`
2. 审核前 20 条并改成 `approved`
3. `--max-samples 40`
4. 审核第 21 到 40 条并改成 `approved`
5. `--max-samples 60`
6. 依次推进，直到目标规模

不要这样做：

- 前 20 条还停留在 `draft`，就直接跑 `--max-samples 40`

因为这样会导致前 20 条被重新覆盖。

## 6. 如何检查当前进度

### 6.1 查看状态分布

```bash
python -c "import json; from collections import Counter; data=json.load(open('eval/kg_eval/dataset/AIGC-EDU-kgval.gold.json','r',encoding='utf-8')); print(Counter((x.get('kg_eval',{}).get('gold',{}).get('status','pending') for x in data)))"
```

示例结果：

- `draft: 20, pending: 80`
- 或 `approved: 20, pending: 80`

### 6.2 查看前 30 条的状态

```bash
python -c "import json; data=json.load(open('eval/kg_eval/dataset/AIGC-EDU-kgval.gold.json','r',encoding='utf-8')); [print(i+1, x['id'], x.get('kg_eval',{}).get('gold',{}).get('status','pending')) for i,x in enumerate(data[:30])]"
```

## 7. 常见问题

### Q1：为什么文件里有很多 `pending`？

因为这个文件中包含的是完整的样本集，例如 100 条。你只对前 20 条跑了预标注时：

- 前 20 条会变成 `draft`
- 其余样本仍然是默认的 `pending`

这属于正常现象。

### Q2：为什么我已经人工改过，又被覆盖了？

因为当前代码只会跳过 `approved`，不会跳过 `draft`。

也就是说：

- `approved`：安全，不会被重新生成覆盖
- `draft`：再次运行 `generate_gold` 时可能被覆盖

### Q3：什么时候再跑评估？

等你先积累出一批稳定的 `approved` 样本后，再运行后续评估流程。当前阶段只做 gold 集构建，不做 `run` 评估。

## 8. 推荐的本轮操作

如果你当前已经完成了前 20 条的模型预标注，建议下一步严格按下面顺序执行：

1. 打开 `eval/kg_eval/dataset/AIGC-EDU-kgval.gold.json`
2. 审核并修正前 20 条 `draft`
3. 将已完成的样本状态改成 `approved`
4. 检查状态分布，确认前 20 条都已 `approved`
5. 再运行：

```bash
python -m eval.kg_eval.run --config eval/kg_eval/config.yaml generate_gold --max-samples 40
```

这样系统就会跳过前 20 条，继续为第 21 到 40 条生成新的 `draft` 结果。
