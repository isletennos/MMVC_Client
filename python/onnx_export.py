import os
import argparse
import time
import json

import onnx
from onnxsim import simplify
import onnxruntime as ort
import torch

from models import SynthesizerTrn
from symbols import symbols


def get_hparams_from_file(config_path):
  with open(config_path, "r", encoding="utf-8") as f:
    data = f.read()
  config = json.loads(data)
  hparams = HParams(**config)
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


def load_checkpoint(checkpoint_path, model, optimizer=None):
  assert os.path.isfile(checkpoint_path), f"No such file or directory: {checkpoint_path}"
  checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
  iteration = checkpoint_dict['iteration']
  learning_rate = checkpoint_dict['learning_rate']
  if optimizer is not None:
    optimizer.load_state_dict(checkpoint_dict['optimizer'])
  saved_state_dict = checkpoint_dict['model']
  if hasattr(model, 'module'):
    state_dict = model.module.state_dict()
  else:
    state_dict = model.state_dict()
  new_state_dict= {}
  for k, v in state_dict.items():
    try:
      new_state_dict[k] = saved_state_dict[k]
    except:
      new_state_dict[k] = v
  if hasattr(model, 'module'):
    model.module.load_state_dict(new_state_dict)
  else:
    model.load_state_dict(new_state_dict)
  return model, optimizer, learning_rate, iteration


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", required=True)
    parser.add_argument("--input_vits_pt", required=True)
    parser.add_argument("--output_vits_onnx", required=True)
    return parser.parse_args()


def main(args):
    hps = get_hparams_from_file(args.config_file)
    #device = torch.device("cuda")
    device = torch.device("cpu")

    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model)
    for i in net_g.parameters():
        i.requires_grad = False
    _ = net_g.eval()
    _ = load_checkpoint(args.input_vits_pt, net_g, None)
    print("Model data loading succeeded.\nConverting start.")

    # ONNXへの切り替え
    dummy_specs = torch.rand(1, 257, 60)
    dummy_lengths = torch.LongTensor([60])
    dummy_sid_src = torch.LongTensor([0])
    dummy_sid_tgt = torch.LongTensor([1])
    inputs = (dummy_specs, dummy_lengths, dummy_sid_src, dummy_sid_tgt)

    torch.onnx.export(
        net_g,
        inputs,
        args.output_vits_onnx,
        do_constant_folding=False,
        opset_version=13,
        verbose=False,
        input_names=["specs", "lengths", "sid_src", "sid_tgt"],
        output_names=["audio"],
        dynamic_axes={
            "specs": {2: "length"}
        })
    model_onnx2 = onnx.load(args.output_vits_onnx)
    model_simp, check = simplify(model_onnx2)
    onnx.save(model_simp, args.output_vits_onnx)

    sess_options = ort.SessionOptions()
    ort_session = ort.InferenceSession(
        args.output_vits_onnx,
        sess_options=sess_options,
        providers=["CPUExecutionProvider"])
#        providers=["CUDAExecutionProvider"])

    print("vits onnx benchmark")
    use_time_list = []
    for i in range(30):
        start = time.time()
        output = ort_session.run(
            ["audio"],
            {
                "specs": dummy_specs.numpy(),
                "lengths": dummy_lengths.numpy(),
                "sid_src": dummy_sid_src.numpy(),
                "sid_tgt": dummy_sid_tgt.numpy()
            }
        )
        use_time = time.time() - start
        use_time_list.append(use_time)
        #print("use time:{}".format(use_time))
    use_time_list = use_time_list[5:]
    mean_use_time = sum(use_time_list) / len(use_time_list)
    print(f"onnx mean_use_time: {mean_use_time}")
    onnx_output = output[0]

    use_time_list = []
    for i in range(30):
        start = time.time()
        origin_output = net_g(*(
            dummy_specs.to(device),
            dummy_lengths.to(device),
            dummy_sid_src.to(device),
            dummy_sid_tgt.to(device)
        )).data.cpu().float().numpy()
        use_time = time.time() - start
        use_time_list.append(use_time)
        #print("use time:{}".format(use_time))
    use_time_list = use_time_list[5:]
    mean_use_time = sum(use_time_list) / len(use_time_list)
    print(f"{device} origin mean_use_time: {mean_use_time}")

    assert onnx_output.shape == origin_output.shape

    print("Done")


if __name__ == '__main__':
    args = get_args()
    print(args)
    main(args)
