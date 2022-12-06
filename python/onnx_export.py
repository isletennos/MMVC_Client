import argparse
import time
import json

import numpy as np
import onnx
import onnxruntime as ort
import torch

from models import SynthesizerTrn
from symbols import symbols


def get_hparams_from_file(config_path):
  with open(config_path, "r", encoding="utf-8") as f:
    data = f.read()
  config = json.loads(data)

  hparams =HParams(**config)
  return hparams


class HParams():
  def __init__(self, **kwargs):
    for k, v in kwargs.items():
      if type(v) == dict:
        v = HParams(**v)
      self[k] = v
    
  def keys(self):
    return self.__dict__.keys()

  def items(self):
    return self.__dict__.items()

  def values(self):
    return self.__dict__.values()

  def __len__(self):
    return len(self.__dict__)

  def __getitem__(self, key):
    return getattr(self, key)

  def __setitem__(self, key, value):
    return setattr(self, key, value)

  def __contains__(self, key):
    return key in self.__dict__

  def __repr__(self):
    return self.__dict__.__repr__()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", required=True)
    parser.add_argument("--input_vits_pt", required=True)
    parser.add_argument("--output_vits_onnx", required=True)
    return parser.parse_args()


def main(args):
    hps = get_hparams_from_file(args.config_file)
    device = torch.device("cuda")
    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model)
    for i in net_g.parameters():
        i.requires_grad = False
    _ = net_g.eval()

    # ONNXへの切り替え
    test_hidden_unit = torch.rand(1, 50, 256)
    test_lengths = torch.LongTensor([50])
    test_pitch = (torch.rand(1, 50) * 128).long()
    test_sid = torch.LongTensor([0])
    input_names = ["hidden_unit", "lengths", "pitch", "sid"]
    output_names = ["audio", ]

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

    sess_options = ort.SessionOptions()
    ort_session = ort.InferenceSession(args.output_vits_onnx,
                                       sess_options=sess_options,
                                       providers=["CUDAExecutionProvider", ])

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
    origin_output = net_g(*(
        test_hidden_unit.to(device),
        test_lengths.to(device),
        test_pitch.to(device),
        test_sid.to(device)
    )).detach().cpu().numpy()
    assert onnx_output.shape == origin_output.shape

    print("Done")


if __name__ == '__main__':
    args = get_args()
    print(args)
    main(args)
