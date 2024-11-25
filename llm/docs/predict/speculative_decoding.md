# 投机解码教程

投机解码是一个通过投机性地一次性猜测多个 token 然后进行验证和接收的算法，通过投机解码可以极大地减小推理时延。PaddleNLP 提供了简单、高效的投机解码推理流程。下面提供 PaddleNLP 中各种投机解码算法的使用说明。

## Inference with reference

该算法通过 n-gram 窗口从 prompt 中匹配 draft tokens，适合输入和输出有很大 overlap 的场景如代码编辑、文档查询等，更多信息查看查看[论文地址](https://arxiv.org/pdf/2304.04487)。

### 使用命令

```shell
# 动态图模型推理命令参考
python ./predict/predictor.py --model_name_or_path meta-llama/Llama-2-7b-chat --inference_model --dtype float16 --speculate_method inference_with_reference --speculate_max_draft_token_num 5 --speculate_max_ngram_size 2
```

**Note:**

1. 该算法目前只支持 llama 系列模型
2. 投机解码同时支持量化推理，具体命令参考[推理示例](./inference.md)，将 speculate_method 等投机解码参数加上即可。
