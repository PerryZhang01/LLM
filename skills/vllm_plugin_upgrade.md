---
Name: 升级vllm版本并完成vllm plugin atom适配
Description:  对vllm-atom框架进行vllm版本升级，并解决因升级导致的vllm plugin模型功能精度等兼容问题
---

# 版本升级

先按照指定版本完成vllm的版本升级安装，然后对部分plugin模型做兼容性适配，最后对整个模型列表进行完整测试，要求升级后的所有模型还能保持原有精度。


请你按照以下步骤进行

```text
实现进度：
- [x] 1) 升级vllm版本
- [x] 2) CI模型验证
- [x] 3) nightly模型精度测试
- [x] 4) benchmark模型性能测试
```

### 1) 升级vllm版本

请按照以下步骤对vllm版本进行升级：

1. 参考docker/vllm_release.dockerfile构建vllm image vllm-0.27.1, vllm commit使用v0.28.0，你需要修改部分dockerfile文件按照dockerfile的运行流程来构建vllm升级镜像；
2. 请使用最新image并参考以下命令创建并进入docker(如果当前已在docker环境中，则无需创建docker，跳过前两步);
```
podman run -it --cap-add=SYS_PTRACE --network=host --security-opt seccomp=unconfined --name xxx --device=/dev/kfd --device=/dev/dri -v /shared/data:/data --group-add keep-groups --ipc=host docker.io/rocm/atom-dev:vllm-upgrade /bin/bash
```
3. 请你在新的文档里记录升级的关键修改，以及升级前后哪些依赖发生版本变更。

注意事项：
- 在vllm中的安装依赖，如果有改变torch、transformers、triton版本的请直接在dockerfile中删除这些依赖并用--no-deps安装。

### 2) CI模型验证

请你按照ATOM的CI列表对vllm backend CI模型进行精度验证(模型权重在挂载目录/data/amd_int/models)：

1. 在ATOM/.github下找到CI模型的vllm backend精度测试脚本；
2. 分别运行以上模型列表，每次运行前请先清除显存再启动server，模型运行失败时请直接修复；
3. 模型精度不足时，请先尝试解决精度问题，如未能解决，可以基于rocm/atom-dev:latest重建docker，在老版本中运行精度测试，或者只跑ATOM backend不带vllm测试精度，对比再定位解决升级精度问题；
3. 统计每个模型的运行结果。

### 3) nightly模型精度测试

CI模型测试完成后，请你接下来完成nightly全部MI355模型测试，你要先完成全部vllm backend的模型测试，然后对有错误的模型尝试进行修复，按照以下步骤执行

1. 在ATOM/.github下找到所有模型的nighly accuracy精度测试脚本；
2. 分别运行以上模型列表，每次运行前请先清除显存再启动server，你可以利用所有的空闲卡同时跑多个任务；
3. 模型精度不足时，请先尝试解决精度问题，如未能解决，可以重建docker，不升级在vllm在0.22.1版本中运行精度测试做对比再定位解决；
3. 统计每个模型的运行结果。

### 3) benchmark模型性能测试

完成精度验证后，请你接下来按步骤完成MI355的benchmark测试，并将测试结果保存下来。

1. 在ATOM/.github下找到vllm nightly模型的benchmark测试脚本；
2. 分别运行以上模型列表，每次运行前请先清除显存再启动server；
3. 统计每个模型的运行结果。


