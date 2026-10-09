---
name: model structure
目标: 请你按照最新代码分析一下指定模型的结构，以html形式呈现，结果放置LLM/models，然后向远端推送PR。
---

# 要求

- 有完整的模型结构，模型里要包含shape信息，对于复杂的attention模块可用大块表示
- 若主模型中有大块表示，下面必须要有一个单独的模型结构对该块做详细拆解
- 一些特殊的模块也需要单独拆解

# 模型

-DeepSeekV4.1

# 框架后端

-atom

# 输出路径

-LLM/models/DeepSeek