# encoding=utf-8
# 将HUBERT和VITS模型导出为ONNX格式
# 示例
# python onnx_export.py \
# --config_file "./configs/nyarumul.json" \
# --input_hubert_pt "./pth/hubert.pt" \
# --input_vits_pt "./pth/121_epochs.pth" \
# --output_hubert_onnx "./pth/hubert.onnx" \
# --output_vits_onnx "./pth/121_epochs.onnx"
import argparse
import time

import numpy as np
import onnx
#from onnxsim import simplify
import onnxruntime as ort
#import onnxoptimizer
import torch

from sovits.infer_tool import Svc


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", required=True)
    parser.add_argument("--input_hubert_pt", required=True)
    parser.add_argument("--input_vits_pt", required=True)
    parser.add_argument("--output_hubert_onnx", required=True)
    parser.add_argument("--output_vits_onnx", required=True)
    return parser.parse_args()


def main(args):
    device = torch.device("cuda")
    # 实例化模型
    svc = Svc(net_g_path=args.input_vits_pt,
              config_path=args.config_file,
              hubert_path=args.input_hubert_pt,
              onnx=True)
    hubert = svc.hubert_soft

    # 转为ONNX
    test_input = torch.rand(1, 1, 16000)
    input_names = ["source", ]
    output_names = ["embed", ]
    torch.onnx.export(hubert,
                      test_input.to(device),
                      args.output_hubert_onnx,
                      dynamic_axes={
                          "source": {
                              2: "sample_length"
                          }
                      },
                      verbose=False,
                      opset_version=13,
                      input_names=input_names,
                      output_names=output_names)
    # 使用不同shape的数据对ONNX输出结果进行验证
    origin_output = hubert(test_input.to(device)).detach().cpu().numpy()
    ort_session = ort.InferenceSession(args.output_hubert_onnx, providers=['CPUExecutionProvider', ])

    outputs = ort_session.run(
        output_names,
        {"source": test_input.numpy()}
    )
    onnx_output = outputs[0]
    right = np.allclose(origin_output, onnx_output, rtol=0.1, atol=0.1)
    print("Hubert ONNX right: {}".format(right))

    # 实例化模型
    net_g = svc.net_g_ms
    for i in net_g.parameters():
        i.requires_grad = False
    # 转为ONNX
    test_hidden_unit = torch.rand(1, 50, 256)
    test_lengths = torch.LongTensor([50])
    test_pitch = (torch.rand(1, 50) * 128).long()
    test_sid = torch.LongTensor([0])
    input_names = ["hidden_unit", "lengths", "pitch", "sid"]
    output_names = ["audio", ]
    net_g.eval()
    # traced = torch.jit.trace(net_g,
    #                          (
    #                              test_hidden_unit.to(device),
    #                              test_lengths.to(device),
    #                              test_pitch.to(device),
    #                              test_sid.to(device)
    #                          )
    #                          )
    torch.onnx.export(net_g,
                      (
                          test_hidden_unit.to(device),
                          test_lengths.to(device),
                          test_pitch.to(device),
                          test_sid.to(device)
                      ),
                      args.output_vits_onnx,
                      dynamic_axes={
                          "hidden_unit": [0, 1],
                          "pitch": [1]
                      },
                      do_constant_folding=False,
                      opset_version=13,
                      verbose=False,
                      input_names=input_names,
                      output_names=output_names)

    # 使用不同shape的数据对ONNX输出结果进行验证
    origin_output = net_g(*(
        test_hidden_unit.to(device),
        test_lengths.to(device),
        test_pitch.to(device),
        test_sid.to(device)
    )).detach().cpu().numpy()

    # model = onnx.load(args.output_vits_onnx)
    # model_opti = onnxoptimizer.optimize(model)
    # model_simp, check = simplify(model_opti)
    # onnx.save(model_opti, args.output_vits_onnx)
    # assert check, "Simplified ONNX model could not be validated"
    sess_options = ort.SessionOptions()
    # sess_options.add_session_config_entry('session.dynamic_block_base', '4')
    # sess_options.enable_profiling = True
    # sess_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
    # sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    ort_session = ort.InferenceSession(args.output_vits_onnx,
                                       sess_options=sess_options,
                                       providers=["CUDAExecutionProvider", ])
    # ort_session = ort.InferenceSession(r"D:\codes\sovits_aishell3\onnxmodel334.onnx",
    #                                    sess_options=sess_options,
    #                                    providers=['CUDAExecutionProvider', ])

    # test_hidden_unit = torch.rand(1, 50, 258)
    print("vits onnx benchmark")
    use_time_list = []
    for i in range(30):
        start = time.time()
        outputs = ort_session.run(
            output_names,
            {
                "hidden_unit": test_hidden_unit.numpy(),
                "lengths": test_lengths.numpy(),
                "pitch": test_pitch.numpy(),
                "sid": test_sid.numpy(),

                # "x": test_hidden_unit.numpy(),
                # 'x_lengths': test_lengths.numpy(),
                # 'sid': test_sid.numpy(),
                # "noise_scale": [0.667], "length_scale": [1.0], "noise_scale_w": [0.8]
            }
        )
        use_time = time.time() - start
        use_time_list.append(use_time)
        print("use time:{}".format(use_time))
    use_time_list = use_time_list[5:]
    mean_use_time = sum(use_time_list) / len(use_time_list)
    print("mean_use_time:{}".format(mean_use_time))
    onnx_output = outputs[0]
    # VITS的推理过程有随机过程，这里不验证输出结果的数值是否一致
    #assert onnx_output.shape == origin_output.shape
    print("Done")


if __name__ == '__main__':
    args = get_args()
    print(args)
    main(args)
