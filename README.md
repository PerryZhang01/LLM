# LLM

大模型**结构解剖**文档集。每个模型一份自包含的 HTML：完整结构图、逐模块的张量形状、
从 checkpoint 分片头部实读的参数账，以及它在推理后端里究竟落在哪几个文件上。

面向的是要动这段代码的人——不是模型卡，也不是论文摘要。当前全部文档以
[ATOM](https://github.com/ROCm/ATOM) 为后端。

---

## 📖 在线阅读

> **[→ 打开模型结构手册](https://perryzhang01.github.io/LLM/)**

GitHub 的文件浏览器不渲染 HTML（只显示源码），所以文档通过 GitHub Pages 托管。
上面这个链接是总览页，点任意一张模型卡片即可进入对应的解剖页。

<details>
<summary>Pages 还没启用？两种办法</summary>

**启用 Pages（推荐，一次性设置）**
`Settings → Pages → Build and deployment → Source: Deploy from a branch`，
分支选 `main`、目录选 `/ (root)`，保存。约一分钟后 `https://perryzhang01.github.io/LLM/` 生效。
仓库根目录已放好 `index.html` 与 `.nojekyll`，无需额外的 workflow。

**临时预览（不改任何设置）**
把文件的 GitHub 地址丢给 htmlpreview：
`https://htmlpreview.github.io/?https://github.com/PerryZhang01/LLM/blob/main/index.html`

**本地看**
`git clone` 之后浏览器直接打开 `index.html`——所有页面都不依赖外部资源
（除 Google Fonts 外，断网时自动回退到系统字体）。

</details>

---

## 已收录的模型

| 模型 | 架构 | 总参数 / 激活 | 结构要点 | 后端实现 |
|---|---|---|---|---|
| **[GLM-5.3-Flash](https://perryzhang01.github.io/LLM/models/GLM/glm53_flash_anatomy.html)** | `glm5_next` | 320.8 B / 17.38 B | 34 层 KDA + 11 层稀疏 MLA 混合，mHC 四路残差，全模型 NoPE，k-pool 稀疏索引 | `atom/models/glm5_next.py` |
| **[GLM-5.3](https://perryzhang01.github.io/LLM/models/GLM/glm53_anatomy.html)** | `glm_moe_dsa` | 753.3 B / 41.25 B | 78 层 MLA + DSA，256 专家，IndexShare；结构与 5.2 逐字段相同，只多一个 `moe_router_dtype` | `atom/models/deepseek_v2.py` |
| **[GLM-5.2](https://perryzhang01.github.io/LLM/models/glm52_anatomy.html)** | `glm_moe_dsa` | 753.3 B / 41.25 B | 引入 IndexShare：78 层里只有 21 层真的算 top-k | `atom/models/deepseek_v2.py` |

GLM-5.2 另有一份 Markdown 版，GitHub 可直接阅读：[`models/GLM/model.md`](models/GLM/model.md)。

---

## 每份文档包含什么

- **完整模型栈图 + 单层展开图**，每个模块标注张量形状
- **核心机制的专门图示**——稀疏索引怎么选、残差怎么走、注意力变体差在哪
- **权重实测形状**：直接读 safetensors 分片头部，不抄 config
- **参数账与显存账**：单 token 激活量、KV cache、每请求状态缓存
- **该模型在后端里的落点**，以及一条读码路线

所有数字来自 checkpoint 与源码实读，不引用模型卡上的标称值。

---

## 目录结构

```
.
├── index.html                          # 模型总览页（GitHub Pages 入口）
├── models/
│   ├── glm52_anatomy.html              # GLM-5.2
│   └── GLM/
│       ├── glm53_anatomy.html          # GLM-5.3（旗舰）
│       ├── glm53_flash_anatomy.html    # GLM-5.3-Flash
│       └── model.md                    # GLM-5.2 的 Markdown 版
└── skills/                             # 驱动这些文档生成的 skill
    ├── models.md                       # 模型结构解剖
    ├── accuracy.md                     # 精度调查
    ├── atom-ops-debug.md               # 算子调试
    └── vllm_plugin_upgrade.md          # vLLM 插件升级
```

---

## 新增一个模型

文档由 [`skills/models.md`](skills/models.md) 驱动生成。改三行，然后交给 Claude Code 执行该 skill：

```markdown
# 模型
-glm5.3

# 框架后端
-atom

# 输出路径
-LLM/models/GLM
```

产物是一份不依赖构建步骤的 HTML（深浅色自适应、响应式、图全部是内联 SVG）。
生成后把它加进 `index.html` 的卡片网格，README 表格里补一行，就完成了收录。
