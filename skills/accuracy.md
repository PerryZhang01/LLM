---
name: accuracy-commit-debug
目标: 模型精度运行依赖框架ATOM和算子aiter，现在给你一个模型和一组精度正常的 ATOM 与 aiter commit，以及一组精度异常的ATOM 和 aiter commit；现在请你在正确commit和错误commit之间找出首次导致精度错误的commit，它可能来自 ATOM，也可能来自 aiter。
---

# 输入

正确commit:
- ATOM: 368cd515d
- aiter: 95c8be607

错误commit:
- ATOM: 368cd515d
- aiter: 5228838e5

框架:
- ATOM

模型：
- Qwen3.5-397B-A17B-FP8 MTP

Host路径：
- /shared/data/amd_int/models/Qwen/Qwen3.5-397B-A17B-FP8

## 可选配置

模型精度threshold(可以不提供，如果不提供，则使用ATOM仓库里的threshold)：0.80

image(可以不提供，如果不提供，使用默认镜像)：
atom默认镜像：rocm/atom-dev:latest
vllm-atom默认镜像：rocm/atom-dev:vllm-latest
sglang-atom默认镜像：rocm/atom-dev:sglang-latest

docker启动命令(请使用podman启动docker)
```bash
podman run -it --cap-add=SYS_PTRACE --network=host --security-opt seccomp=unconfined --name xxx --device=/dev/kfd --device=/dev/dri -v host_path:/model_path --group-add keep-groups --ipc=host image /bin/bash
```

## 执行流程

```text
- [ ] 1) 参考 `.github` 下的 workflows，在本地拉取镜像，并使用 `podman` 或 `docker` 启动容器，进入容器中查看ATOM和aiter的安装位置；
- [ ] 2) 按照ATOM仓库.github目录下的accuracy workflow，根据model名称和框架类型，为该模型构建 server 和 accuracy client 脚本；
- [ ] 3) 每次切换commit请先清理cache，每次运行结束，请关闭相关进程；
- [ ] 4) 首先将ATOM和aiter切换到正确commit和错误commit启动server和client进行精度验证，精度结果大于threshold表示该commit组合正确(如果是MTP模型，则接受率大于50%才表示正确)，否则则有问题；
- [ ] 5) 然后对 ATOM 或 aiter 的commit进行二分，或者你认为更合适的方法，更换commit后继续进行精度测试，直到找出**首次**出错的那个commit；
- [ ] 6) 切换commit时遇到ATOM与aiter的commit不兼容问题而非精度问题请你自行调整commit,兼容性问题不算精度问题；
- [ ] 7) 找出首次出错的commit后根据该commit的内容进行代码修复并验证其正确性；
- [ ] 8) 最后对你整个测试过程生成一份测试报告，并删除额外调试容器和log。
```

## 停止条件

按照执行流程进行测试，直到你找出第一个commit导致精度flexible-extract的value明显低于模型设定的threshold或者接受率小于50%，如果模型未设定threshold，请你去ATOM .github目录中查看该配置下模型的精度threshold。