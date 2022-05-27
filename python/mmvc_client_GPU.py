# -*- coding:utf-8 -*-
#use thread limit
import os
os.environ["OMP_NUM_THREADS"] = "1"
import sys
import pyaudio
import numpy as np

#voice conversion
import torch

#user lib
import utils
from models import SynthesizerTrn
from text.symbols import symbols
# from Shifter.shifter import Shifter
import sounddevice as sd
import soundfile as sf
#noice reduce
import noisereduce as nr
import json
#use logging
from logging import getLogger

#ファイルダイアログ関連
import tkinter as tk #add
from tkinter import filedialog #add

import readchar


class Hyperparameters():
    CHANNELS = 1 #モノラル
    FORMAT = pyaudio.paInt16
    INPUT_DEVICE_1 = None
    INPUT_DEVICE_2 = None
    OUTPUT_DEVICE_1 = None
    CONFIG_JSON_PATH = None
    MODEL_PATH = None
    NOISE_FILE = None
    FLAME_LENGTH = None
    SOURCE_ID = None
    TARGET_ID = None
    USE_NR = None
    VOICE_LIST = None
    #jsonから取得
    SAMPLE_RATE = None
    MAX_WAV_VALUE = None
    FILTER_LENGTH = None
    HOP_LENGTH = None
    SEGMENT_SIZE = None
    N_SPEAKERS = None
    CONFIG_JSON_Body = None
    DELAY_FLAMES = None
    #thread share var
    REC_NOISE_END_FLAG = False
    VC_END_FLAG = False
    OVERLAP = None
    

    def set_input_device_1(self, value):
        Hyperparameters.INPUT_DEVICE_1 = value

    def set_input_device_2(self, value):
        Hyperparameters.INPUT_DEVICE_2 = value

    def set_output_device_1(self, value):
        Hyperparameters.OUTPUT_DEVICE_1 = value

    def set_config_path(self, value):
        Hyperparameters.CONFIG_JSON_PATH = value
        config = utils.get_hparams_from_file(Hyperparameters.CONFIG_JSON_PATH)
        Hyperparameters.CONFIG_JSON_Body = config
        Hyperparameters.SAMPLE_RATE = config.data.sampling_rate
        Hyperparameters.MAX_WAV_VALUE = config.data.max_wav_value
        Hyperparameters.FILTER_LENGTH = config.data.filter_length
        Hyperparameters.HOP_LENGTH = config.data.hop_length
        Hyperparameters.SEGMENT_SIZE = config.train.segment_size
        Hyperparameters.N_SPEAKERS = config.data.n_speakers

    def set_model_path(self, value):
        Hyperparameters.MODEL_PATH = value

    def set_NOISE_FILE(self, value):
        Hyperparameters.NOISE_FILE = value

    def set_FLAME_LENGTH(self, value):
        Hyperparameters.FLAME_LENGTH = value

    def set_SOURCE_ID(self, value):
        Hyperparameters.SOURCE_ID = value

    def set_TARGET_ID(self, value):
        Hyperparameters.TARGET_ID = value

    def set_OVERLAP(self, value):
        Hyperparameters.OVERLAP = value

    def set_USE_NR(self, value):
        Hyperparameters.USE_NR = value

    def set_VOICE_LIST(self, value):
        Hyperparameters.VOICE_LIST = value

    def set_DELAY_FLAMES(self, value):
        Hyperparameters.DELAY_FLAMES = value

    def set_profile(self, profile):
        sound_devices = sd.query_devices()
        if type(profile.device.input_device1) == str:
            self.set_input_device_1(sound_devices.index(sd.query_devices(profile.device.input_device1, 'input')))
        else:
            self.set_input_device_1(profile.device.input_device1)
        
        if type(profile.device.input_device2) == str:
            self.set_input_device_2(sound_devices.index(sd.query_devices(profile.device.input_device2, 'input')))
        else:
            self.set_input_device_2(profile.device.input_device2)
        
        if type(profile.device.output_device) == str:
            self.set_output_device_1(sound_devices.index(sd.query_devices(profile.device.output_device, 'output')))
        else:
            self.set_output_device_1(profile.device.output_device)
        
        self.set_config_path(profile.path.json)
        self.set_model_path(profile.path.model)
        self.set_NOISE_FILE(profile.path.noise)
        self.set_FLAME_LENGTH(profile.vc_conf.frame_length)
        self.set_SOURCE_ID(profile.vc_conf.source_id)
        self.set_TARGET_ID(profile.vc_conf.target_id)
        self.set_OVERLAP(profile.vc_conf.overlap)
        self.set_USE_NR(profile.others.use_nr)
        self.set_VOICE_LIST(profile.others.voice_list)
        self.set_DELAY_FLAMES(profile.vc_conf.delay_flames)

    def launch_model(self):
        hps = utils.get_hparams_from_file(Hyperparameters.CONFIG_JSON_PATH)
        net_g = SynthesizerTrn(
            len(symbols),
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            n_speakers=hps.data.n_speakers,
            **hps.model)
        _ = net_g.eval()
        #暫定872000
        _ = utils.load_checkpoint(Hyperparameters.MODEL_PATH, net_g, None)
        print("モデルの読み込みが完了しました。音声の入出力の準備を行います。少々お待ちください。")
        return net_g
        
    def audio_trans_GPU(self, tdbm, input, net_g, noise_data, target_id):
        # byte => torch
        signal = np.frombuffer(input, dtype='int16')
        #signal = torch.frombuffer(input, dtype=torch.float32)
        signal = signal / Hyperparameters.MAX_WAV_VALUE
        if Hyperparameters.USE_NR:
            signal = nr.reduce_noise(y=signal, sr=Hyperparameters.SAMPLE_RATE, y_noise = noise_data, n_std_thresh_stationary=2.5,stationary=True)
        # any to many への取り組み(失敗)
        # f0を変えるだけでは枯れた声は直らなかった
        #f0trans = Shifter(Hyperparameters.SAMPLE_RATE, 1.75, frame_ms=20, shift_ms=10)
        #transformed = f0trans.transform(signal)
        signal = torch.from_numpy(signal.astype(np.float32)).clone()

        #voice conversion
        with torch.no_grad():
            #SID
            data = tdbm.get_audio_text_speaker_pair(signal.view(1,Hyperparameters.FLAME_LENGTH), ["m", Hyperparameters.SOURCE_ID, "m"])
            data = TextAudioSpeakerCollate()([data])
            x, x_lengths, spec, spec_lengths, y, y_lengths, sid_src = [x.cuda() for x in data]

            sid_tgt1 = torch.LongTensor([target_id]).cuda() # 話者IDはJVSの番号を100で割った余りです
            audio1 = net_g.cuda().voice_conversion(spec, spec_lengths, sid_src=sid_src, sid_tgt=sid_tgt1)[0][0,0].data.cpu().float().numpy()

        audio1 = audio1 * Hyperparameters.MAX_WAV_VALUE
        audio1 = audio1.astype(np.int16).tobytes()

        return audio1

    def over_lap_marge(self,trance_data_A,trance_data_B,overlap):
        a = np.arange(overlap)/overlap
        #overlap部分の処理
        trance_data_A_lap = trance_data_A
        trance_data_B_lap = trance_data_B
        signal_A = np.frombuffer(trance_data_A_lap, dtype='int16')
        signal_B = np.frombuffer(trance_data_B_lap, dtype='int16')
        signal_Z = (1-a) * signal_A + a * signal_B
        signal = np.round(signal_Z,decimals= 0)
        signal = signal.astype(np.int16).tobytes()
        return signal

    def vc_run(self):
        audio = pyaudio.PyAudio()
        print("モデルを読み込んでいます。少々お待ちください。")
        net_g = self.launch_model()
        tdbm = Transform_Data_By_Model()

        if Hyperparameters.USE_NR:
            noise_data, noise_rate = sf.read(Hyperparameters.NOISE_FILE)
        else:
            noise_data = 0

        # audio stream voice
        #マイク
        audio_input_stream = audio.open(format=Hyperparameters.FORMAT,
                            channels=1,
                            rate=Hyperparameters.SAMPLE_RATE,
                            frames_per_buffer=Hyperparameters.DELAY_FLAMES,
                            input_device_index=Hyperparameters.INPUT_DEVICE_1,
                            input=True)

        #Realtek Digital Output
        audio_output_stream = audio.open(format=Hyperparameters.FORMAT,
                            channels=1,
                            rate=Hyperparameters.SAMPLE_RATE,
                            frames_per_buffer=Hyperparameters.DELAY_FLAMES,
                            output_device_index = Hyperparameters.OUTPUT_DEVICE_1,
                            output=True)

        #CABLE Output
        if Hyperparameters.INPUT_DEVICE_2 != False:
            back_audio_input_stream = audio.open(format=Hyperparameters.FORMAT,
                                channels=1,
                                rate=Hyperparameters.SAMPLE_RATE,
                                frames_per_buffer=Hyperparameters.DELAY_FLAMES,
                                input_device_index=Hyperparameters.INPUT_DEVICE_2,
                                input=True)
        else:
            back_audio_input_stream = audio.open(format=Hyperparameters.FORMAT,
                                channels=1,
                                rate=Hyperparameters.SAMPLE_RATE,
                                frames_per_buffer=Hyperparameters.DELAY_FLAMES,
                                input_device_index=Hyperparameters.INPUT_DEVICE_1,
                                input=True)

        #Realtek Digital Output
        back_audio_output_stream = audio.open(format=Hyperparameters.FORMAT,
                            channels=1,
                            rate=Hyperparameters.SAMPLE_RATE,
                            frames_per_buffer=Hyperparameters.DELAY_FLAMES,
                            output_device_index = Hyperparameters.OUTPUT_DEVICE_1,
                            output=True)

        overlap = Hyperparameters.OVERLAP
        target_id = Hyperparameters.TARGET_ID
        target_index = 0

        #第一節を取得する
        try:
            print("準備が完了しました。VC開始します。")
            #マイクから音声読み込み
            #最初のデータAを取得する
            #rawdataのsizeは(frame_length * 2 - overlap)の2倍になっている type=byte 30720
            ##8192
            in_raw_data_A = audio_input_stream.read(Hyperparameters.FLAME_LENGTH, exception_on_overflow = False)
            #背景BGMを取得
            back_in_raw_data_A = back_audio_input_stream.read(Hyperparameters.FLAME_LENGTH, exception_on_overflow = False)
            #ボイチェン(取得した音声の前半)
            #trancedataのsizeは(frame_length*2)となっている type=byte 16384
            trance_data_A = self.audio_trans_GPU(tdbm, in_raw_data_A, net_g, noise_data, target_id)
            #Hyperparameters.DELAY_FLAMES + overlap を後半部分から取る
            #ゴミ+Hyperparameters.DELAY_FLAMES+overlap >> Hyperparameters.DELAY_FLAMES+overlap
            tmp = trance_data_A
            tmp2 = back_in_raw_data_A
            trance_data_A = trance_data_A[-(Hyperparameters.DELAY_FLAMES + overlap)*2:-overlap*2]
            back_trance_data_A = back_in_raw_data_A[-(Hyperparameters.DELAY_FLAMES + overlap)*2:-overlap*2]
            overlap_trance_data = tmp[-overlap*2:]
            overlap_back_trance_data = tmp2[-overlap*2:]

            while True:
                #声id変更
                kb = readchar.readchar()
                if kb == b'n':
                    target_index += 1
                    if target_index >= len(Hyperparameters.VOICE_LIST) :
                        target_index = 0
                    target_id = Hyperparameters.VOICE_LIST[target_index]
                    print(f"next - voice id: {target_id}")
                if kb == b'p':
                    target_index -= 1
                    if target_index < 0 :
                        target_index = len(Hyperparameters.VOICE_LIST) - 1
                    target_id = Hyperparameters.VOICE_LIST[target_index]
                    print(f"prev - voice id: {target_id}")

                if Hyperparameters.VC_END_FLAG: #エスケープ
                    print("vc_finish")
                    break
                #音声後半のoverlapを取得する
                overlap_trance_data_A = trance_data_A[-overlap*2:]
                trance_data_A = trance_data_A[:-overlap*2]
                overlap_back_trance_data_A = back_trance_data_A[-overlap*2:]
                back_trance_data_A = back_trance_data_A[:-overlap*2]

                #(overlap(処理済み) + Hyperparameters.DELAY_FLAMES - 次のoverlap)の音声を出力する
                out_trance_data_A = overlap_trance_data + trance_data_A
                out_back_trance_data_A = overlap_back_trance_data + back_trance_data_A
                #overlap + Hyperparameters.DELAY_FLAMESの音声を出力
                audio_output_stream.write(out_trance_data_A)
                if Hyperparameters.INPUT_DEVICE_2 != False:
                    back_audio_output_stream.write(out_back_trance_data_A)
                
                #音声が出力されている間に次の音声の準備をする
                #Hyperparameters.DELAY_FLAMESだけ、音声を取得する
                in_raw_data_B = audio_input_stream.read(Hyperparameters.DELAY_FLAMES)
                back_in_raw_data_B = back_audio_input_stream.read(Hyperparameters.DELAY_FLAMES)
                #取得したサイズと前のデータの後ろを組み合わせて、segment_size(8192)にする。
                in_raw_data_A = in_raw_data_A[-Hyperparameters.DELAY_FLAMES*2:] + in_raw_data_B
                back_in_raw_data_A = back_in_raw_data_A[-Hyperparameters.DELAY_FLAMES*2:] + back_in_raw_data_B

                #ボイチェン
                #trancedataのsizeは(frame_length*2)となっている type=byte 16384
                trance_data_B = self.audio_trans_GPU(tdbm, in_raw_data_A, net_g, noise_data, target_id)
                #overlap + 後半部分のみ使う
                trance_data_B = trance_data_B[-(overlap + Hyperparameters.DELAY_FLAMES)*2:]
                #back 処理用
                back_trance_data_B = back_in_raw_data_B[-(overlap + Hyperparameters.DELAY_FLAMES)*2:]
                #overlap対応(今度は前半文)
                overlap_trance_data_B = trance_data_B[:overlap*2]
                overlap_back_trance_data_B = back_trance_data_B[:overlap*2]
                trance_data_B = trance_data_B[overlap*2:]
                back_trance_data_B = back_trance_data_B[:overlap*2:]
                #overlap マージ
                overlap_trance_data = self.over_lap_marge(overlap_trance_data_A,overlap_trance_data_B,overlap)
                overlap_back_trance_data = self.over_lap_marge(overlap_back_trance_data_A,overlap_back_trance_data_B,overlap)

                trance_data_A = trance_data_B
                back_trance_data_A = back_trance_data_B
                

        except KeyboardInterrupt:
            audio_input_stream.stop_stream()
            audio_input_stream.close()
            audio_output_stream.stop_stream()
            audio_output_stream.close()
            back_audio_input_stream.stop_stream()
            back_audio_input_stream.close()
            back_audio_output_stream.stop_stream()
            back_audio_output_stream.close()
            audio.terminate()
            print("Stop Streaming")    

class Transform_Data_By_Model():
    hann_window = {}
    FILTER_LENGTH = 0
    HOP_LENGTH = 0
    SAMPLE_RATE = 0
    HPS = None
    CONFIG = None
    def __init__(self):
        self.G_HP = Hyperparameters()
        self.HPS = utils.get_hparams_from_file(self.G_HP.CONFIG_JSON_PATH)
        #define samplerate
        self.SAMPLE_RATE =self.HPS.data.sampling_rate
        #define filter size
        self.FILTER_LENGTH = self.HPS.data.filter_length
        self.HOP_LENGTH = self.HPS.data.hop_length

    def spectrogram_torch(self, y, n_fft, sampling_rate, hop_size, win_size, center=False):
        if torch.min(y) < -1.:
            print('min value is ', torch.min(y))
        if torch.max(y) > 1.:
            print('max value is ', torch.max(y))

        dtype_device = str(y.dtype) + '_' + str(y.device)
        wnsize_dtype_device = str(win_size) + '_' + dtype_device
        if wnsize_dtype_device not in self.hann_window:
            self.hann_window[wnsize_dtype_device] = torch.hann_window(win_size).to(dtype=y.dtype, device=y.device)

        y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
        y = y.squeeze(1)

        spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=self.hann_window[wnsize_dtype_device],
                        center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex=True)
        spec = torch.view_as_real(spec)
        
        spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)
        return spec

    def get_audio_text_speaker_pair(self, wav, audiopath_sid_text):
        _, sid, text = audiopath_sid_text[0], audiopath_sid_text[1], audiopath_sid_text[2]
        text = self.get_text(text)
        spec = self.get_spec(wav)
        sid = self.get_sid(sid)
        return (text, spec, wav, sid)

    def get_spec(self, audio_norm):
        filter_length = self.FILTER_LENGTH
        sampling_rate = self.SAMPLE_RATE
        hop_length = self.HOP_LENGTH
        win_length = self.FILTER_LENGTH
        spec = self.spectrogram_torch(audio_norm, filter_length,
            sampling_rate, hop_length, win_length,
            center=False)
        spec = torch.squeeze(spec, 0)
        return spec

    def get_text(self, text):
        return text

    def get_sid(self, sid):
        sid = torch.LongTensor([int(sid)])
        return sid

class TextAudioSpeakerCollate():
    """ Zero-pads model inputs and targets
    """
    def __init__(self, return_ids=False):
        self.return_ids = return_ids

    def __call__(self, batch):
        """Collate's training batch from normalized text, audio and speaker identities
        PARAMS
        ------
        batch: [text_normalized, spec_normalized, wav_normalized, sid]
        """
        # Right zero-pad all one-hot text sequences to max input length
        _, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([x[1].size(1) for x in batch]),
            dim=0, descending=True)

        max_text_len = max([len(x[0]) for x in batch])
        max_spec_len = max([x[1].size(1) for x in batch])
        max_wav_len = max([x[2].size(1) for x in batch])

        text_lengths = torch.LongTensor(len(batch))
        spec_lengths = torch.LongTensor(len(batch))
        wav_lengths = torch.LongTensor(len(batch))
        sid = torch.LongTensor(len(batch))

        text_padded = torch.LongTensor(len(batch), max_text_len)
        spec_padded = torch.FloatTensor(len(batch), batch[0][1].size(0), max_spec_len)
        wav_padded = torch.FloatTensor(len(batch), 1, max_wav_len)
        text_padded.zero_()
        spec_padded.zero_()
        wav_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]

            spec = row[1]
            spec_padded[i, :, :spec.size(1)] = spec
            spec_lengths[i] = spec.size(1)

            wav = row[2]
            wav_padded[i, :, :wav.size(1)] = wav
            wav_lengths[i] = wav.size(1)

            sid[i] = row[3]

        if self.return_ids:
            return text_padded, text_lengths, spec_padded, spec_lengths, wav_padded, wav_lengths, sid, ids_sorted_decreasing
        return text_padded, text_lengths, spec_padded, spec_lengths, wav_padded, wav_lengths, sid

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
    with open(config_path, "r", encoding="utf-8") as f:
        data = f.read()
    config = json.loads(data)
    hparams = VCPrifile(**config)
    return hparams

if __name__ == '__main__':
    try: #add
        args = sys.argv
        if len(args) < 2:
            end_counter = 0
            while True:  # 無限ループ
                tkroot = tk.Tk()
                tkroot.withdraw()
                print('myprofile.json を選択して下さい')
                typ = [('jsonファイル','*.json')]
                dir = './'
                profile_path = filedialog.askopenfilename(filetypes = typ, initialdir = dir)
                tkroot.destroy()
                try:
                    if profile_path:
                        break
                    else:
                        print('ファイルが存在しません')
                        end_counter = end_counter + 1
                        print(end_counter)
                        if end_counter > 3:
                            break
                        continue
            
                except ValueError:
                    # ValueError例外を処理するコード
                    print('パスを入力してください・')
                    continue
        else:
            profile_path = args[1]
            print("起動時にmyprofile.jsonのパスが指定されました。")
            print(profile_path)

        params = config_get(profile_path)
        vc_main = Hyperparameters()

        print(params.path.json)
        vc_main.set_profile(params)
        vc_main.vc_run()
    
    except Exception as e:
        print('エラーが発生しました。')
        print(e)
        os.system('PAUSE')


    
