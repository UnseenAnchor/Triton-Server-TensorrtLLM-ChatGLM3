# Triton-Server-TensorrtLLM-ChatGLM3
<a name="o5JjG"></a>
## 介绍
<a name="fOrL4"></a>
### Triton inference server
Triton 推理服务器是英伟达 NVIDIA AIE 的组成部分，同时也是一个**开源**的推理服务软件，用于**简化 AI 模型的部署和推理过程，并提供高性能的推理服务**。<br />Triton 推理服务器提供了标准化的 AI 推理流程，**支持部署各种深度学习和机器学习框架的AI模型**， 包括 TensorRT、TensorFlow、PyTorch、ONNX、OpenVINO、Python、RAPIDS FIL等 。Triton 推理服务器可以在 NVIDIA GPU、x86 和 ARM CPU 以及 AWS Inferentia 等设备上进行**云端**、**数据中心**、**边缘**和**嵌入式设备**的推理。

![](https://cdn.nlark.com/yuque/0/2024/png/1604247/1710901115274-7c2317af-ce82-4271-9899-70e1aabeca0e.png#averageHue=%23f7f7f7&clientId=u8e1ff727-14c4-4&from=paste&id=ue4b41d51&originHeight=788&originWidth=627&originalType=url&ratio=1&rotation=0&showTitle=false&status=done&style=none&taskId=ucc6f6597-998b-401e-a49f-11c176e0ad1&title=)<br />Triton的**主要特性**包括：

- 支持多种机器学习/深度学习框架
- 并发模型执行
- 动态批处理
- 序列批处理和隐式状态管理用于有状态模型
- 提供后端API，允许添加自定义后端和前/后处理操作
- 使用集成（ Ensembles）和业务逻辑脚本（ BLS）构建模型Pipeline
- 基于社区开发的KServe协议的HTTP/REST和GRPC推理协议
- 支持C API和Java API直接链接到应用程序
- 指示GPU利用率 、服务器吞吐量 、服务器延迟等指标

Triton 推理服务器对多种查询类型提供高效的推理，支持**实时查询**、**批处理查询**、**集成模型查询**和**音视频流查询**等。

<a name="atp6I"></a>
### TensorRT-LLM
TensorRT-LLM是针对大型语言模型构建最优化的 TensorRT 引擎，以在 NVIDIA GPU 上高效执行推理 。<br />TensorRT-LLM 包含用于创建执行这些 TensorRT 引擎的 Python 和 C++ 运行时的组件，还包括与 NVIDIA Triton 推理服务器集成的后端，用于提供大模型服务的生产级系统。**TensorRT-LLM 支持单个 GPU 到多节点多 GPU 的各种配置环境的使用，同时支持近30余种国内外流行大模型的优化**。<br />TensorRT-LLM 的具体性能可以查看官方性能页面，其优势在一些测试和报道中也已经得到体现：NVIDIA TensorRT-LLM 在 NVIDIA H*GPU （80GB）上大幅提升大型语言模型的推理速度。<br />TensorRT-LLM 优化特性覆盖了以下几个方面：<br />**1. 注意力优化（Attention Optimizations）**

- Multi-head Attention (MHA)：将注意力计算分解为多个头，提高并行性，并允许模型关注输入的不同维度语义空间的信息，然后再进行拼接。
- Multi-query Attention (MQA)：与MHA不同的，MQA 让所有的头之间共享同一份 Key 和 Value 矩阵，每个头只单独保留了一份 Query 参数，从而大大减少 Key 和 Value 矩阵的参数量，提高吞吐量并降低延迟。
- Group-query Attention (GQA)：介于MHA和MQA，将查询分组以减少内存访问和计算，提高效率。
- In-flight Batching：重叠计算和数据传输以隐藏延迟并提高性能。
- Paged KV Cache for the Attention ：在注意力层中缓存键值对，减少内存访问并加快计算速度。

**2. 并行性（ Parallelism）**

- Tensor Parallelism ：将模型层分布在多个 GPU 上，使其能够扩展到大型模型。
- Pipeline Parallelism ：重叠不同层的计算，降低整体延迟。

**3. 量化（ Quantization）**

- INT4/INT8 weight-only (W4A16 和 W8A16)：将权重存储为 4 位或 8 位整型减少模型大小和内存占用，同时保持激活在 16 位浮点精度。
- SmoothQuant：为注意力层提供平滑量化，保留准确性。
- GPTQ：一次性权重量化方法，针对 GPT 类似模型架构量身定制的量化技术，同时保持精度。
- AWQ：自适应权重量化，动态调整不同部分模型的量化精度，确保高精度和效率。
- FP8：在支持的 GPU（ 如 NVIDIA Hopper）上利用 8 位浮点精度进行计算，进一步减少内存占用并加速处理。

**4. 解码优化（ Decoding Optimizations）**

- Greedy-search：贪婪搜索，一次生成一个文本令牌，通过选择最可能的下一个令牌，快速但可能不太准确。
- Beam-search：束搜索，跟踪多个可能的令牌序列，提高准确性但增加计算成本。

**5. 其他**

- RoPE (相对位置编码)：高效地嵌入令牌的相对位置信息，增强模型对上下文的理解。

能否使用特定优化取决于模型架构、硬件配置和所需的性能权衡，目前最新版本中，并非所有模型都支持上述优化。TensorRT-LLM 提供了一个灵活的框架，可用于尝试不同的优化策略，以实现特定用例的最佳结果。通过一系列的优化技术，能显著提高大语言模型在 NVIDIA GPU 上的性能和效率。

TensorRT-LLM 使用的流程：<br />![b39f6a448aab35464ed4f878ac83e5e8.png](https://cdn.nlark.com/yuque/0/2024/png/1604247/1710484083589-71f79757-3e57-4e12-bbfe-96e2cd697f98.png#averageHue=%23f5f5f5&clientId=u222aebfd-4f12-4&from=paste&height=403&id=ue9eeff89&originHeight=5625&originWidth=10000&originalType=binary&ratio=1&rotation=0&showTitle=false&size=3072830&status=done&style=none&taskId=u244c08de-daaf-4ec8-bd19-ae978f9e3cf&title=&width=717)
<a name="Fgnuc"></a>
## 部署实践
<a name="BuJXo"></a>
### 系统环境

- GPU: A30 24G *4
- Memory: 256GB
- Host OS：Ubuntu 22.04
- GPU Driver：545.29.06
- Cuda Toolkit：cuda_12.3
<a name="QBx0Q"></a>
### 版本说明

- Docker Image：nvcr.io/nvidia/tritonserver:24.01-trtllm-python-py3 
- Docker Image：baseten/tensorrt-llm-benchmarks:v0.7.1

我们使用的triton版本是24.01，与此对应的tensorrtllm版本需要是v0.7.1，[这里](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html)可以看到不同版本对驱动、cuda以及pytorch、tensorrt-llm的要求。
<a name="JGsWv"></a>
### 拉取tritonserver镜像
```shell
docker pull nvcr.io/nvidia/tritonserver:24.01-trtllm-python-py3
```
<a name="iSy2L"></a>
### clone tensorrtllm项目
```shell
git clone -b v0.7.1 --depth=1  https://github.com/triton-inference-server/tensorrtllm_backend.git
```
<a name="YeJEE"></a>
### 复制文件
```shell
cd tensorrtllm_backend
mkdir triton_model_repo
cp -r all_models/inflight_batcher_llm/* triton_model_repo/
```
<a name="nY6HO"></a>
### 在tensorrt-llm容器里编译chatglm3-6b的engine
因为我在本地无法执行`make -C docker release_build`，来构建tensorrt-llm的docker镜像，因此选择直接拉取镜像文件，然后将本地模型文件挂载在容器内，来进行模型的编译。
```shell
sudo docker run --gpus all \
  --name trt_llm \
  -d \
  --ipc=host \
  --ulimit memlock=-1 \
  -v /home/lead/models:/mnt\
  --restart=always \
  --ulimit stack=67108864 \
  baseten/tensorrt-llm-benchmarks:v0.7.1 sleep 8640000

sudo docker exec -it trt_llm /bin/bash


```
然后是构建推理引擎
```shell
python3 chatglm/build.py \
        -m chatglm3_6b \
        --model_dir /mnt/chatglm3-6b/ \
        --world_size 4 --tp_size 4 \
        --max_batch_size 256 \
        --max_output_len 2048 \
        --max_input_len 2048 \
        --enable_context_fmha \
        --use_gpt_attention_plugin \
        --paged_kv_cache \
        --output_dir chatglm/trt_engines/chatglm3_6b/fp16/triton_4-gpu_v2
```
build.py 参数选择
1. --model_name {chatglm_6b,chatglm2_6b,chatglm2_6b_32k,chatglm3_6b,chatglm3_6b_base,chatglm3_6b_32k,glm_10b}：指定要构建的模型名称。使用下划线而不是连字符来连接名称部分。
2. --world_size WORLD_SIZE：指定世界大小，目前只支持张量并行。
3. --tp_size TP_SIZE：指定张量并行的大小。
4. --pp_size PP_SIZE：指定流水线并行的大小。
5. --model_dir MODEL_DIR：指定模型目录。
6. --quant_ckpt_path QUANT_CKPT_PATH：指定量化检查点路径。
7. --dtype {float32,float16,bfloat16}：指定数据类型，可以是float32、float16或bfloat16。
8. --logits_dtype {float16,float32}：指定logits的数据类型。
9. --timing_cache TIMING_CACHE：指定从哪里读取时间缓存的路径，如果文件不存在则会被忽略。
10. --log_level {verbose,info,warning,error,internal_error}：选择日志级别。
11. --max_batch_size MAX_BATCH_SIZE：指定最大批量大小。
12. --max_input_len MAX_INPUT_LEN：指定最大输入长度。
13. --max_output_len MAX_OUTPUT_LEN：指定最大输出长度。
14. --max_beam_width MAX_BEAM_WIDTH：指定最大beam宽度。
15. --use_gpt_attention_plugin [{float32,float16,bfloat16,False}]：激活注意力插件。可以指定插件数据类型或留空以使用模型数据类型。
16. --use_gemm_plugin [{float32,float16,bfloat16,False}]：激活GEMM插件。可以指定插件数据类型或留空以使用模型数据类型。
17. --use_layernorm_plugin [{float32,float16,bfloat16,False}]：激活层归一化插件。可以指定插件数据类型或留空以使用模型数据类型。
18. --use_rmsnorm_plugin [{float32,float16,bfloat16,False}]：激活rmsnorm插件。可以指定插件数据类型或留空以使用模型数据类型。
19. --gather_all_token_logits：是否收集所有token的logits。
20. --parallel_build：是否并行构建。
21. --enable_context_fmha：启用上下文全连接注意力。
22. --enable_context_fmha_fp32_acc：启用上下文全连接注意力的FP32累积。
23. --multi_block_mode：将长的KV序列分割成多个块（应用于生成MHA内核）。当batchxnum_heads不能完全利用GPU时，这很有帮助。
24. --visualize：是否可视化。
25. --enable_debug_output：是否启用调试输出。
26. --gpus_per_node GPUS_PER_NODE：每个节点的GPU数量。
27. --builder_opt BUILDER_OPT：构建器选项。
28. --output_dir OUTPUT_DIR：保存序列化引擎文件、时间缓存文件和模型配置的路径。
29. --strongly_typed：这个选项是在trt 9.1.0.1+中引入的，可以显著减少fp8的构建时间。
30. --remove_input_padding：移除输入填充。
31. --paged_kv_cache：启用分页KV缓存。
32. --use_inflight_batching：激活gptAttentionPlugin的inflight批处理模式。
33. --use_smooth_quant：使用SmoothQuant方法量化激活和权重。查看--per_channel和--per_token以获取更细粒度的量化选项。
34. --use_weight_only：仅量化各种GEMM的权重为INT4/INT8。查看--weight_only_precision以设置精度。
35. --weight_only_precision {int8,int4,int4_awq}：定义权重的精度，当使用权重量化时。你必须同时使用--use_weight_only才能使该参数生效。
36. --per_channel：默认情况下，我们使用单个静态缩放因子来缩放GEMM的结果。per_channel则为每个通道使用不同的静态缩放因子。后者通常更准确，但稍慢。
37. --per_token：默认情况下，我们使用单个静态缩放因子来缩放int8范围内的激活。per_token在运行时选择，为每个token选择一个自定义缩放因子。后者通常更准确，但稍慢。
38. --per_group：默认情况下，我们使用单个静态缩放因子来缩放int4范围内的权重。per_group在运行时选择，为每个组选择一个自定义缩放因子。这个标志是为GPTQ/AWQ量化构建的。
39. --group_size GROUP_SIZE：在GPTQ/AWQ量化中使用的组大小。
40. --int8_kv_cache：默认情况下，我们使用数据类型为KV缓存。int8_kv_cache选择int8量化为KV。
41. --random_seed RANDOM_SEED：为torch初始化随机数生成器时使用的种子。
42. --tokens_per_block TOKENS_PER_BLOCK：分页KV缓存中每个块的token数量。
43. --enable_fp8：在Attention QKV/Dense和MLP中使用FP8线性层。
44. --fp8_kv_cache：默认情况下，我们使用数据类型为KV缓存。fp8_kv_cache选择fp8量化为KV。
45. --max_num_tokens MAX_NUM_TOKENS：定义引擎支持的最大token数量。
46. --use_custom_all_reduce：激活用于all-reduce的延迟优化算法，而不是NCCL。

这些参数允许用户根据特定的硬件配置和性能需求来定制模型的构建过程。例如，通过调整并行大小、数据类型和量化选项，用户可以优化模型在特定GPU或GPU集群上的性能。

构建好的推理引擎：<br />![image.png](https://cdn.nlark.com/yuque/0/2024/png/1604247/1710983690872-51d4c5fa-2341-4c70-a500-4941d29d49ce.png#averageHue=%23e1f6e6&clientId=uefaefa63-262e-4&from=paste&height=114&id=u488b140a&originHeight=114&originWidth=305&originalType=binary&ratio=1&rotation=0&showTitle=false&size=4030&status=done&style=none&taskId=u2b17288f-ab5c-4aaa-81fe-bf9d873b08d&title=&width=305)
<a name="b4vu7"></a>
### 启动triton server容器
```shell
sudo docker run -it \
                --net host \
                --name triton_v2 \
                -d \
                --shm-size=2g \
                --ulimit memlock=-1 \
                --ulimit stack=67108864 \
                --gpus all \
                -v /mnt/liyanbo/tensorrtllm_backend:/opt/tritonserver/tensorrtllm_backend:rw \
                -v /mnt/llm_models/chatglm3-6b:/chatglm3-6b \
                nvcr.io/nvidia/tritonserver:24.01-trtllm-python-py3 bash

# cp -r /chatglm3-6b /opt/tritonserver/tensorrtllm_backend/triton_model_repo/tensorrt_llm/
```
这里从host主机挂载两个文件夹在容器内：

1. host内的tensorrtllm_backend；
2. host上的模型文件，用来指定tokenizer的路径使用。【后来因为读取tokenizer路径总有问题，改为从huggingface镜像上读取，因此这一步对我来说不是必须的】
<a name="doj1M"></a>
### 安装相关库
在容器内安装相关库
```shell
pip install torch torchvision torchaudio sentence_transformers transformers tiktoken accelerate packaging ninja transformers_stream_generator einops optimum bitsandbytes -i https://pypi.tuna.tsinghua.edu.cn/simple/

# 安装requirements.txt
pip install transformers==4.31.0 -i https://pypi.tuna.tsinghua.edu.cn/simple/
pip install tritonclient[all,cuda] -i https://pypi.tuna.tsinghua.edu.cn/simple/
pip install regex fire pandas tabulate -i https://pypi.tuna.tsinghua.edu.cn/simple/
```
<a name="cRmTV"></a>
### 修改triton_model_repo下配置文件
<a name="Xapxs"></a>
#### 需要修改的config.pbtxt

- /tensorrtllm_backend/triton_model_repo/ensemble/config.pbtxt
```json
max_batch_size: 512
```

- /tensorrtllm_backend/triton_model_repo/postprocessing/config.pbtxt
```json
max_batch_size: 512
parameters {
  key: "tokenizer_dir"
  value: {
    string_value: "THUDM/chatglm3-6b"
  }
}
parameters {
  key: "tokenizer_type"
  value: {
    string_value: "auto"
  }
}
```

- /tensorrtllm_backend/triton_model_repo/preprocessing/config.pbtxt
```json
max_batch_size: 512
parameters {
  key: "tokenizer_dir"
  value: {
    string_value: "THUDM/chatglm3-6b"
  }
}

parameters {
  key: "tokenizer_type"
  value: {
    string_value: "auto"
  }
}
```

- /tensorrtllm_backend/triton_model_repo/tensorrt_llm/config.pbtxt
```json
max_batch_size: 512

model_transaction_policy {
  decoupled: True
}

dynamic_batching {
    preferred_batch_size: [ 512 ]
    max_queue_delay_microseconds: 10000
}

instance_group [
  {
    count: 1
    kind: KIND_CPU
  }
]
parameters: {
  key: "max_beam_width"
  value: {
    string_value: "${max_beam_width}"
  }
}
parameters: {
  key: "FORCE_CPU_ONLY_INPUT_TENSORS"
  value: {
    string_value: "no"
  }
}
parameters: {
  key: "gpt_model_type"
  value: {
    string_value: "V1"
  }
}
parameters: {
  key: "gpt_model_path"
  value: {
    string_value: "/opt/tritonserver/tensorrtllm_backend/triton_model_repo/tensorrt_llm/1"
  }
}
parameters: {
  key: "max_tokens_in_paged_kv_cache"
  value: {
    string_value: "${max_tokens_in_paged_kv_cache}"
  }
}
parameters: {
  key: "max_attention_window_size"
  value: {
    string_value: "${max_attention_window_size}"
  }
}
parameters: {
  key: "batch_scheduler_policy"
  value: {
    string_value: "max_utilization"
  }
}
parameters: {
  key: "kv_cache_free_gpu_mem_fraction"
  value: {
    string_value: "0.9"
  }
}
parameters: {
  key: "max_num_sequences"
  value: {
    string_value: "${max_num_sequences}"
  }
}
parameters: {
  key: "enable_trt_overlap"
  value: {
    string_value: "false"
  }
}
parameters: {
  key: "exclude_input_in_output"
  value: {
    string_value: "true"
  }
}
parameters: {
  key: "enable_kv_cache_reuse"
  value: {
    string_value: "false"
  }
}
```

- /tensorrtllm_backend/triton_model_repo/tensorrt_llm_bls/config.pbtxt
```json
max_batch_size: 512

model_transaction_policy {
  decoupled: True
}
```
<a name="G2BGv"></a>
#### 需要修改的model.py

- /tensorrtllm_backend/triton_model_repo/preprocessing/1/model.py
```python
# 一定要在导入transformers之前使用
import os
os.environ['HF_ENDPOINT'] ='https://hf-mirror.com'

# 修改前
# self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, padding_side='left', trust_remote_code=False)
# 修改后
self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, legacy=False, padding_side='left', truncation_side='left', trust_remote_code=True)

# 修改前
# self.tokenizer.pad_token = self.tokenizer.eos_token
# self.pad_id = self.tokenizer.encode(self.tokenizer.pad_token, add_special_tokens=False)[0]
# 修改后
if self.tokenizer.pad_token_id is None:
    self.tokenizer_pad_id = self.tokenizer.eos_token_id
self.tokenizer_pad_id = self.tokenizer.pad_token_id
self.tokenizer_end_id = self.tokenizer.eos_token_id

# 修改前
# self.pad_id → self.tokenizer_pad_id
# 修改后
start_ids = np.stack([np.pad(seq, (0, max_len - seq.shape[0]), 'constant', constant_values=(0, self.tokenizer_pad_id)) for seq in start_ids])
```

- /tensorrtllm_backend/triton_model_repo/postprocessing/1/model.py
```python
# 一定要在导入transformers之前使用
import os
os.environ['HF_ENDPOINT'] ='https://hf-mirror.com'
# 修改前
# self.tokenizer = AutoTokenizer.from_pretrained(
#     tokenizer_dir, padding_side='left', trust_remote_code=False)
# 修改后
self.tokenizer = AutoTokenizer.from_pretrained(
    tokenizer_dir, legacy=False,padding_side='left',truncation_side='left', trust_remote_code=True)

```

<a name="arsMR"></a>
### 启动 Triton 服务器
```shell
python3 /opt/tritonserver/tensorrtllm_backend/scripts/launch_triton_server.py \
        --world_size 4 \
        --model_repo /opt/tritonserver/tensorrtllm_backend/triton_model_repo
```
出现如下图的情况则表示部署成功<br />![](https://cdn.nlark.com/yuque/0/2024/png/1604247/1710825296399-f0077d05-fb50-4a32-8862-f8c5bad662cd.png?x-oss-process=image%2Fformat%2Cwebp#averageHue=%23050302&from=url&id=qcuV4&originHeight=784&originWidth=1380&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=)
<a name="cNjzl"></a>
### 测试
一共有两种测试方式：curl推理和client script
<a name="c3DVg"></a>
#### curl推理
```shell
# curl 推理测试
# ensemble
curl -X POST localhost:8000/v2/models/ensemble/generate -d '{"text_input": "你好", "max_tokens": 200, "bad_words": "", "stop_words": ""}' 

# bls
curl -X POST localhost:8000/v2/models/tensorrt_llm_bls/generate -d '{"text_input": "无锡有什么好吃的", "max_tokens": 200, "bad_words": "", "stop_words": ""}'
```
![image.png](https://cdn.nlark.com/yuque/0/2024/png/1604247/1710830764574-f9d004a6-7a67-49d5-8f33-210b51f4653f.png#averageHue=%2315120e&clientId=ucb5a8d20-e87a-4&from=paste&height=147&id=u66660a48&originHeight=147&originWidth=859&originalType=binary&ratio=1&rotation=0&showTitle=false&size=28261&status=done&style=none&taskId=udf09e22f-fb92-45c7-9241-268cdf8aa62&title=&width=859)
<a name="oF2AW"></a>
#### tilize the provided client script to send a request
```shell
cd /opt/tritonserver/tensorrtll_backend

# 修改preprocessing下的model.py中的self.pad_id
python3 inflight_batcher_llm/client/end_to_end_grpc_client.py --streaming --output-len 200 --prompt "你对中国和美国的过力对比怎么看，字数2000字"
```

<a name="eHCdh"></a>
### 总结
由于使用triton inference server部署tensorrtllmv0.7.1版本的 chatglm3能参考的帖子不多，因此基本也算是摸着石头过河，到目前为止也只是部署成功，后续还有大量的优化可以做，比如，开启 In-flight Batching，模型复读机问题等等。

<a name="RJVwO"></a>
### 踩坑记录
至于踩坑过程中遇到的一些bug，慢慢记录

<a name="Ejud8"></a>
### 参考资料

- [GitHub - DataXujing/TensorRT-LLM-ChatGLM3: :fire: 大模型部署实战：TensorRT-LLM, Triton Inference Server, vLLM](https://github.com/DataXujing/TensorRT-LLM-ChatGLM3)
- [深度学习部署神器——triton inference server入门教程指北_triton调用深度学习模型服务端代码如何编写-CSDN博客](https://blog.csdn.net/IAMoldpan/article/details/127350748)
- [大模型推理实践-1：基于TensorRT-LLM和Triton部署ChatGLM2-6B模型推理服务](https://zhuanlan.zhihu.com/p/663338695)
- [Triton部署TensorRT-LLM](https://zhuanlan.zhihu.com/p/663378231)
- [Triton23.12部署TensorRT-LLM,实现http查询](https://zhuanlan.zhihu.com/p/678864791)
- [NVIDIA AI Enterprise 科普 | Triton 推理服务器 & TensorRT-LLM 两大组件介绍及实践](https://mp.weixin.qq.com/s/zz12uvkPKuwoxsX1iiWyCA)
- [英伟达开源TensorRT-LLM，可优化类ChatGPT开源模型！](https://mp.weixin.qq.com/s/zNmlthYs2cCfrOjLthyrQQ)
- [TensorRT-LLM大模型部署速通](https://blog.nghuyong.top/2023/12/01/NLP/tensorrt_llm/)

