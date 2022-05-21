# -*- coding: utf-8 -*
import pyaudio
import sounddevice as sd
import wave
import numpy as np
import time
import json

class VCPrifile():
  def __init__(self, **kwargs):
    for k, v in kwargs.items():
      if type(v) == dict:
        v = VCPrifile(**v)
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

def config_get(conf):
    config_path = conf
    with open(config_path, "r") as f:
        data = f.read()
    config = json.loads(data)
    hparams = VCPrifile(**config)
    return hparams

def MakeWavFile():
    chunk = 1024
    while True:  # 無限ループ
        print('学習済みモデルのサンプリングレートを指定してください。')
        try:
            sr = int(input('>> '))
        except ValueError:
            # ValueError例外を処理するコード
            print('数字以外が入力されました。数字のみを入力してください')
            continue
        break

    while True:  # 無限ループ
        print('「myprofile.json」のパスを入力してください。')
        profile_path = input('>> ')
        try:
            if profile_path:
                break
            else:
                print('ファイルが存在しません')
                continue
    
        except ValueError:
            # ValueError例外を処理するコード
            print('パスを入力してください・')
            continue
    
    params = config_get(profile_path)
    print(params.device.input_device1)
    if type(params.device.input_device1) == str:
        device_index = sd.query_devices().index(sd.query_devices(params.device.input_device1, 'input'))
    else:
        device_index = params.device.input_device1

    p = pyaudio.PyAudio()
    stream = p.open(format = pyaudio.paInt16,
                    channels = 1,
                    rate = sr,
                    input = True,
                    input_device_index=params.device.input_device1,
                    frames_per_buffer = chunk)
    #レコード開始
    print("あなたの環境ノイズを録音します。マイクの電源を入れて、何もせずに待機していてください。")
    print("5秒後に録音を開始します。5秒間ノイズを録音します。完了するまで待機していてください。")
    Record_Seconds = 5
    MAX_Value = 32768.0
    all = []
    time.sleep(5)
    print("録音を開始しました。")
    for i in range(0, int(sr / chunk * Record_Seconds)):
        data = stream.read(chunk) #音声を読み取って、
        data = np.frombuffer(data, dtype='int16')
        audio1 = data * MAX_Value
        audio1 = audio1.astype(np.int16).tobytes()
        all.append(data) #データを追加
    #レコード終了
    print("録音が完了しました。")
    print("ファイルに書き込みを行っています。")
    stream.close()
    p.terminate()
    wavFile = wave.open("noise.wav", 'wb')
    wavFile.setnchannels(1)
    wavFile.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    wavFile.setframerate(sr)
    #wavFile.writeframes(b''.join(all)) #Python2 用
    wavFile.writeframes(b"".join(all)) #Python3用
    wavFile.close()
    print("ファイルの書き込み完了しました。")
    print("このウィンドウは閉じて問題ありません。")
    input()

if __name__ == '__main__':
    MakeWavFile()