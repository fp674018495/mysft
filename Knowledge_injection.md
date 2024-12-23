
<!-- TOC -->

- [1. 大模型微调注入新知识方案总结](#1-大模型微调注入新知识方案总结)
- [2. 通过大模型微调注入新知识 方法和技术下细节](#2-通过大模型微调注入新知识-方法和技术下细节)
    - [2.1. 模型选择](#21-模型选择)
    - [2.2. 数据构建](#22-数据构建)
    - [2.3. 微调方法](#23-微调方法)
    - [2.4. 微调框架](#24-微调框架)
    - [2.5. 强化训练](#25-强化训练)
    - [2.6. 可能会存在问题](#26-可能会存在问题)
    - [2.7. 多模态开源模型](#27-多模态开源模型)
- [3. 参考文献](#3-参考文献)

<!-- /TOC -->

# 1. 大模型微调注入新知识方案总结
- 模型选择：先采用qwen2.5 1.8b进行尝试 后可尝试微调qwen2.5 7b
- 微调方式: 采用 Lora 
- 数据构建 :    
    * 处理文档   
    * Prompt设计，构建问答对  
    * 通过生成的问题，利用带有本地知识的rag大模型构建问答对
- 对齐：
    * 构建人类偏好数据
    * 利用DPO进行对齐 
    
# 2. 通过大模型微调注入新知识 方法和技术下细节



## 2.1. 模型选择
https://rank.opencompass.org.cn/home 评测榜单

chinese-llm-benchmark  2024/10/20 发布v2.3版本评测榜单
![alt text](src/image.png)
![alt text](src/image-1.png)
![alt text](src/image-2.png)

常见底座模型细节概览：
| 底座     | 包含模型                    | 模型参数大小      | 训练token数  | 训练最大长度 | 是否可商用 |
|----------|---------------------------|-----------------|-------------|------------|-------   |
| ChatGLM  | ChatGLM/2/3 Base&Chat     | 6B              | 1T/1.4      | 2K/32K     | 可商用   |
| LLaMA    | LLaMA/2/3 Base&Chat       | 7B/8B/13B/33B/70B | 1T/2T       | 2k/4k      | 部分可商用  |
| Baichuan | Baichuan/2 Base&Chat      | 7B/13B          | 1.2T/1.4T | 4k     | 可商用   |
| Qwen     | Qwen/2.5 Base&Chat        | 7B/14B/72B/110B | 2.2T/3T      | 8k/32k     | 可商用   |
| BLOOM    | BLOOM                     | 1B/7B/176B-MT   | 1.5T      | 2k     | 可商用   |
| Aquila   | Aquila/2 Base/Chat        | 7B/34B          | -         | 2k     | 可商用   |
| InternLM | InternLM/2 Base/Chat/Code | 7B/20B          | -         | 200k | 可商用 |
| Mixtral  | Base&Chat                 | 8x7B            | -         | 32k | 可商用 |
| Yi       | Base&Chat                 | 6B/9B/34B       | 3T        | 200k | 可商用 |
| DeepSeek | Base&Chat                 | 1.3B/7B/33B/67B | -         | 4k | 可商用 |
| XVERSE   | Base&Chat                 | 7B/13B/65B/A4.2B| 2.6T/3.2T | 8k/16k/256k | 可商用 |

国内开源模型有
* ChatGLM：
  * 地址：https://github.com/THUDM/ChatGLM-6B 
  * 简介：中文领域效果最好的开源底座模型之一，针对中文问答和对话进行了优化。经过约 1T 标识符的中英双语训练，辅以监督微调、反馈自助、人类反馈强化学习等技术的加持
* ChatGLM2-6B
  * 地址：https://github.com/THUDM/ChatGLM2-6B 
  * 简介：基于开源中英双语对话模型 ChatGLM-6B 的第二代版本，在保留了初代模型对话流畅、部署门槛较低等众多优秀特性的基础之上，引入了GLM 的混合目标函数，经过了 1.4T 中英标识符的预训练与人类偏好对齐训练；基座模型的上下文长度扩展到了 32K，并在对话阶段使用 8K 的上下文长度训练；基于 Multi-Query Attention 技术实现更高效的推理速度和更低的显存占用；允许商业使用。
* ChatGLM3-6B
  * 地址：https://github.com/THUDM/ChatGLM3 
  * 简介：ChatGLM3-6B 是 ChatGLM3 系列中的开源模型，在保留了前两代模型对话流畅、部署门槛低等众多优秀特性的基础上，ChatGLM3-6B 引入了如下特性：更强大的基础模型： ChatGLM3-6B 的基础模型 ChatGLM3-6B-Base 采用了更多样的训练数据、更充分的训练步数和更合理的训练策略；更完整的功能支持： ChatGLM3-6B 采用了全新设计的 Prompt 格式，除正常的多轮对话外。同时原生支持工具调用（Function Call）、代码执行（Code Interpreter）和 Agent 任务等复杂场景；更全面的开源序列： 除了对话模型 ChatGLM3-6B 外，还开源了基础模型 ChatGLM3-6B-Base、长文本对话模型 ChatGLM3-6B-32K。以上所有权重对学术研究完全开放，在填写问卷进行登记后亦允许免费商业使用
* GLM-4
  * 地址：https://github.com/THUDM/GLM-4
    ![](https://img.shields.io/github/stars/THUDM/GLM-4.svg)
  * 简介：GLM-4-9B 是智谱 AI 推出的最新一代预训练模型 GLM-4 系列中的开源版本。 在语义、数学、推理、代码和知识等多方面的数据集测评中， **GLM-4-9B** 及其人类偏好对齐的版本 **GLM-4-9B-Chat** 均表现出超越 Llama-3-8B 的卓越性能。除了能进行多轮对话，GLM-4-9B-Chat 还具备网页浏览、代码执行、自定义工具调用（Function Call）和长文本推理（支持最大 128K 上下文）等高级功能。本代模型增加了多语言支持，支持包括日语，韩语，德语在内的 26 种语言。我们还推出了支持 1M 上下文长度（约 200 万中文字符）的 **GLM-4-9B-Chat-1M** 模型和基于 GLM-4-9B 的多模态模型 GLM-4V-9B。**GLM-4V-9B** 具备 1120 * 1120 高分辨率下的中英双语多轮对话能力，在中英文综合能力、感知推理、文字识别、图表理解等多方面多模态评测中，GLM-4V-9B 表现出超越 GPT-4-turbo-2024-04-09、Gemini 1.0 Pro、Qwen-VL-Max 和 Claude 3 Opus 的卓越性能。
* Chinese-LLaMA-Alpaca：
  * 地址：https://github.com/ymcui/Chinese-LLaMA-Alpaca
    ![](https://img.shields.io/github/stars/ymcui/Chinese-LLaMA-Alpaca.svg)
  * 简介：中文LLaMA&Alpaca大语言模型+本地CPU/GPU部署，在原版LLaMA的基础上扩充了中文词表并使用了中文数据进行二次预训练
* Chinese-LLaMA-Alpaca-2：
  * 地址：https://github.com/ymcui/Chinese-LLaMA-Alpaca-2
    ![](https://img.shields.io/github/stars/ymcui/Chinese-LLaMA-Alpaca-2.svg)
  * 简介：该项目将发布中文LLaMA-2 & Alpaca-2大语言模型，基于可商用的LLaMA-2进行二次开发。
* Chinese-LlaMA2：
  * 地址：https://github.com/michael-wzhu/Chinese-LlaMA2
    ![](https://img.shields.io/github/stars/michael-wzhu/Chinese-LlaMA2.svg)
  * 简介：该项目基于可商用的LLaMA-2进行二次开发决定在次开展Llama 2的中文汉化工作，包括Chinese-LlaMA2: 对Llama 2进行中文预训练；第一步：先在42G中文预料上进行训练；后续将会加大训练规模；Chinese-LlaMA2-chat: 对Chinese-LlaMA2进行指令微调和多轮对话微调，以适应各种应用场景和多轮对话交互。同时我们也考虑更为快速的中文适配方案：Chinese-LlaMA2-sft-v0: 采用现有的开源中文指令微调或者是对话数据，对LlaMA-2进行直接微调 (将于近期开源)。
* Llama2-Chinese：
  * 地址：https://github.com/FlagAlpha/Llama2-Chinese
    ![](https://img.shields.io/github/stars/FlagAlpha/Llama2-Chinese.svg)
  * 简介：该项目专注于Llama2模型在中文方面的优化和上层建设，基于大规模中文数据，从预训练开始对Llama2模型进行中文能力的持续迭代升级。
* Qwen/Qwen1.5/Qwen2.5
  * 地址：https://github.com/QwenLM/Qwen
    ![](https://img.shields.io/github/stars/QwenLM/Qwen.svg)
  * 简介：通义千问 是阿里云研发的通义千问大模型系列模型，包括参数规模为18亿（1.8B）、70亿（7B）、140亿（14B）、720亿（72B）和1100亿（110B）。各个规模的模型包括基础模型Qwen，以及对话模型。数据集包括文本和代码等多种数据类型，覆盖通用领域和专业领域，能支持8K的上下文长度，针对插件调用相关的对齐数据做了特定优化，当前模型能有效调用插件以及升级为Agent。
 Baichuan-7B
  * 地址：https://github.com/baichuan-inc/baichuan-7B
    ![](https://img.shields.io/github/stars/baichuan-inc/baichuan-7B.svg)
  * 简介：Baichuan-13B 是由百川智能继 Baichuan-7B 之后开发的包含 130 亿参数的开源可商用的大规模语言模型，在权威的中文和英文 benchmark 上均取得同尺寸最好的效果。该项目发布包含有预训练 (Baichuan-13B-Base) 和对齐 (Baichuan-13B-Chat) 两个版本。
* Baichuan-13B
  * 地址：https://github.com/baichuan-inc/Baichuan-13B
    ![](https://img.shields.io/github/stars/baichuan-inc/baichuan-13B.svg)
  * 简介：由百川智能开发的一个开源可商用的大规模预训练语言模型。基于Transformer结构，在大约1.2万亿tokens上训练的70亿参数模型，支持中英双语，上下文窗口长度为4096。在标准的中文和英文权威benchmark（C-EVAL/MMLU）上均取得同尺寸最好的效果。
* Baichuan2
  * 地址：https://github.com/baichuan-inc/Baichuan2
    ![](https://img.shields.io/github/stars/baichuan-inc/Baichuan2.svg)
  * 简介：由百川智能推出的新一代开源大语言模型，采用 2.6 万亿 Tokens 的高质量语料训练，在多个权威的中文、英文和多语言的通用、领域 benchmark上取得同尺寸最佳的效果，发布包含有7B、13B的Base和经过PPO训练的Chat版本，并提供了Chat版本的4bits量化。


## 2.2. 数据构建
  * 预训练数据集构建  略

  * SFT数据集构建
    * 处理文档   
        先完成文档处理的操作的内容是文本数据，这里需要对PPT、word、pdf、txt等数据进行处理：
    * Prompt设计，构建问答对

        以往我们只能通过人工的方式完成，现可以借助大模型的能力。大致思路就是让大模型根据文本，总结出对话、问答内容。这点可以通过Prompt工程实现。Prompt中，强调根据上下文内容，让模型提取对话、问答等内容。比如：
        ```
        QA_PAIRS_SYSTEM_PROMPT = """  
        <Context></Context> 标记中是一段文本，学习和分析它，并整理学习成果：  
        - 提出问题并给出每个问题的答案。  
        - 答案需详细完整，尽可能保留原文描述。  
        - 答案可以包含普通文字、链接、代码、表格、公示、媒体链接等 Markdown 元素。  
        - 最多提出 30 个问题。  
        """
        ```
    * 通过生成的问题，利用带有本地知识的rag大模型构建问答对

 * 偏好数据集构建（偏好对其）   
   利用多种大模型（微调后和未微调的模型）生成数据，通过模型生成多个不同的回复，然后根据人类的价值观或偏好，手工或自动地为这些成对的响应进行反馈标注；
。



## 2.3. 微调方法
 * 全参数，  
    * 特点：   
        调整所有参数：在微调过程中，模型的所有参数都会被更新。
        高性能：能够充分利用微调数据，提高模型在特定任务上的表现。
        高资源需求：需要大量的计算资源和显存，特别是对于大型模型（如7B、65B参数）的微调。  
    * 适用场景： 资源充足，且需要模型在特定任务上达到最佳性能的情况。
 * LoRA，  
    * 特点：  
        只调整部分参数：通过在模型的某些层插入低秩矩阵，仅调整这些额外的参数，而保持原有参数不变。
        资源高效：大幅减少需要调整的参数数量，降低计算资源和显存的需求。
        灵活性：可以在不影响原模型参数的情况下，进行多任务或多领域的微调。  
    * 适用场景：资源有限，但仍希望进行高效微调的情况。需要频繁切换不同任务或领域的微调。

 * QLoRA  
    * 特点：  
        - ‌内存优化‌：QLoRA通过冻结的4位量化预训练语言模型将梯度反向传播到低秩适配器（LoRA），显著减少了内存使用。例如，可以在单个48GB GPU上微调65B参数模型，而传统的16位精度微调需要超过780GB的GPU内存‌12。   
        - ‌性能保持‌：QLoRA在减少内存需求的同时，保持了与16位全精度微调相当的性能，没有损失模型精度‌12。  
        - ‌创新技术‌：QLoRA引入了多种创新技术，包括4位NormalFloat（NF4）数据类型、双重量化和分页优化器，进一步优化了内存使用和计算效率‌12。

 * 不同SFT下GPU占用情况  
    在单GPU使用LoRA（LoRA (emb)指的是embedding和输出层参与训练，而LoRA则不优化这部分参数）和QLoRA时处理不同长度输入的显存占用和训练速度的情况。使用CUDA 11.8和Pytorch 2.0，并使用了flash attention 2。我们统一使用batch size为1，gradient accumulation为8的训练配置，记录输入长度分别为256、512、1024、2048、4096和8192的显存占用（GB）和训练速度（s/iter）。
    ![alt text](src/image-3.png)

## 2.4. 微调框架


 - LLaMA-Factory https://github.com/hiyouga/LLaMA-Factory 适合那些需要在多种硬件环境下进行微调的用户，特别是对于需要量化模型以适应资源受限设备的场景。它的模块化设计、多硬件支持和量化技术，为用户提供了强大的工具，助力他们在人工智能领域中取得更好的成果
 - Firefly https://github.com/yangjianxin1/Firefly   
 支持预训练、指令微调、DPO，支持全量参数训练、LoRA、QLoRA高效训练。通过配置文件的方式训练不同的模型，小白亦可快速上手训练模型。支持使用Unsloth加速训练，并且节省显存。支持绝大部分主流的开源大模型，如Llama3、Gemma、MiniCPM、Llama、InternLM、Baichuan、ChatGLM、Yi、Deepseek、Qwen、Orion、Ziya、Xverse、Mistral、Mixtral-8x7B、Zephyr、Vicuna、Bloom，训练时与各个官方的chat模型的template对齐。
 - unsloth https://github.com/unslothai/unsloth   Unsloth‌是一个开源的大模型训练加速项目，旨在提升大模型的训练速度并减少显存使用。Unsloth通过重写模型的计算过程，使用OpenAI的Triton进行加速，同时保证模型训练的精度不会损失‌1。优势：微调速度快   
 - SWIFT https://github.com/modelscope/ms-swift SWIFT框架微调支持300+ LLM和80+ MLLM（多模态大模型,MiniCPM-V ）的训练(预训练、微调、对齐)、推理、评测和部署。开发者可以直接将框架应用到自己的Research和生产环境中，实现模型训练评测到应用的完整链路。除支持了PEFT提供的轻量训练方案外，也提供了一个完整的Adapters库以支持最新的训练技术，如NEFTune、LoRA+、LLaMA-PRO等，这个适配器库可以脱离训练脚本直接使用在自己的自定流程中。
 ![alt text](src/image-5.png)

| 方法                   |   LLama-factory       |   Firefly    |       unsloth  |       SWIFT      |
| ---------------------- | ------------------ | ------------------ | ------------------ | ------------------ |
| 预训练                 | √ |   √|√  | √ |
| 指令监督微调            | √ |  √ |  √| √ |
| 奖励模型训练            | √ |  √ |  √|  √|
| PPO 训练               | √ |   √|  √|  √|
| DPO 训练               | √ |  √ | √ |  √|
| KTO 训练               | √ | - |  -|  √|
| ORPO 训练              |  √|  -| - | √ |
| SimPO 训练             | √ | - |  -|  √|




## 2.5. 强化训练
 * RLHF

 * DPO  
    RLHF是一个复杂、不稳定、难训练的过程（用reward model进行ppo强化学习等），而DPO可以避开训练奖励模型这个步骤，直接对排序数据集进行直接偏好学习。将对奖励函数的损失转为对策略的损失，优化和RLHF相同的目标函数（KL散度限制下，最大化reward）。将强化学习的训练阶段转化为一个二分类问题，减少了内存消耗并提高了训练稳定性

## 2.6. 可能会存在问题
    1. 使用事实数据，微调大模型更多epoch，可以记住知识，但是会产生过拟合，造成模型性能严重下降；
    2. 如何评估、是否记住了知识，而不是只记住了训练数据模式：评估集，同样是基于事实数据，使用不同的prompt，生成5-10个多样且全面的问答对，然后评估模型对于所有这些问答对的准确率；

![alt text](src/image-4.png)

## 2.7. 多模态开源模型





# 3. 参考文献
https://zhuanlan.zhihu.com/p/642357133

https://zhuanlan.zhihu.com/p/693661316

https://arxiv.org/abs/2404.00213?context=cs

https://zhuanlan.zhihu.com/p/676193188

https://blog.csdn.net/HUANGXIN9898/article/details/144023855 Qwen2大模型微调 




实际测试真实显存占用和模型加载的显存占用不是正相关的，如下表所示（尚未开始推理）。这一点与vllm的显存预分配机制有关：

1）除了加载模型，还需要初始化kv_cache，这部分就是预分配的显存，这就导致使用vllm时显存占用高于使用transformers；

2）为什么总的显存是这个数字比如下面的18GB而不是其他的数值呢？vllm中使用gpu_memory_utilization来限制最大显存占用，默认是0.9，计算可知大概是21.6GB，它的组成为：模型加载使用的显存A、模型推理消耗的最大显存B以及kv_cache的显存C；而当下的18GB是A + C；A是已知的，B可以通过一次前向模拟获得，那么C就能被计算出来；因而只要将这部分显存先占用起来，就能得到当前18GB的占用。