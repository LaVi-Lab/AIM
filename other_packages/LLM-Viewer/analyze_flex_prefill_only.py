from flexible_analyzer import FlexibleAnalyzer
import torch.nn as nn
import numpy as np
import os
import importlib
import argparse
import ipdb

# Fit the curve
num_input_tokens = 49*32
num_text_tokens = 100
num_list = [max(0, 1.0 - (layer_i - 14 + 1) * 0.125) * num_input_tokens + num_text_tokens if layer_i >= 14 else 1.0 * num_input_tokens + num_text_tokens for layer_i in range(28)]
# num_list = [max(0, 1.0 - (layer_i - 13 + 1) * 0.125) * num_input_tokens + num_text_tokens if layer_i >= 13 else 1.0 * num_input_tokens + num_text_tokens for layer_i in range(32)]
num_list = [int(item) for item in num_list]
print(num_list)

parser = argparse.ArgumentParser()
parser.add_argument("model_id", type=str, help="model id")
parser.add_argument(
    "hardware",
    type=str,
    help="name of hardware, for example nvidia_V100 or nvidia_A6000",
)
parser.add_argument("--config_file", type=str, default=None, help="config file")
parser.add_argument("--batchsize", type=int, default=1, help="batch size")
parser.add_argument("--promptlen", type=str, default=128, help="prompt sequence length, token number for each layer")
parser.add_argument("--w_bit", type=int, default=16, help="weight bitwidth")
parser.add_argument("--a_bit", type=int, default=16, help="temporary activation bitwidth")
parser.add_argument("--kv_bit", type=int, default=16, help="kv cache bitwidth")
parser.add_argument("--use_flashattention", action="store_true", help="use flash attention")
parser.add_argument("--source", dest="source",  type=str, default="huggingface", help="use flash attention")
parser.add_argument("--skip-mlp-bias", dest="skip_mlp_bias", action="store_true", help="if we using te ")

parser.add_argument(
    "--tp-size",
    type=int,
    default=1,
    help="the number of devices for tensor parallelism to use"
)
args = parser.parse_args()

analyzer = FlexibleAnalyzer(args.model_id, args.hardware, args.config_file,source=args.source)

number_of_layer_of_model = analyzer.config.get_num_hidden_layers(analyzer.model_params)
num_attention_heads = analyzer.config.get_num_attention_heads(analyzer.model_params)
promptlen = num_list
# promptlen = int(args.promptlen) if ',' not in args.promptlen else [int(ele) for ele in args.promptlen.split(',')]
if isinstance(promptlen, int):
    promptlen = [promptlen]*number_of_layer_of_model
elif isinstance(promptlen, list):
    assert len(promptlen) == number_of_layer_of_model
else:
    raise NotImplementedError

results = analyzer.analyze_all_layers(
    prompt_len=promptlen,
    num_heads=[num_attention_heads]*number_of_layer_of_model,
    batchsize=args.batchsize,
    w_bit=args.w_bit,
    a_bit=args.a_bit,
    kv_bit=args.kv_bit,
    use_flashattention=args.use_flashattention,
    tp_size=args.tp_size,
    skip_mlp_bias=args.skip_mlp_bias,
)

# standardize the output and the print

print('Flops (TB): ', round(results['prefill']['OPs'] / 10**12, 4) , 
      'inference_time (ms): ', round(results['prefill']['inference_time'] * 1000, 4),
      'memory_consumption (GB): ', round(results['prefill']['memory_consumption'] / 10**9, 4))