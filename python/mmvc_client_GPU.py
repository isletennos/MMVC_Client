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

import keyboard
import wave


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
    DISPOSE_STFT_SPECS = 0
    DISPOSE_CONV1D_SPECS = 0
    INPUT_FILENAME = None
    OUTPUT_FILENAME = None
    

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

    def set_DISPOSE_STFT_SPECS(self, value):
        Hyperparameters.DISPOSE_STFT_SPECS = value

    def set_DISPOSE_CONV1D_SPECS(self, value):
        Hyperparameters.DISPOSE_CONV1D_SPECS = value

    def set_INPUT_FILENAME(self, value):
        Hyperparameters.INPUT_FILENAME = value

    def set_OUTPUT_FILENAME(self, value):
        Hyperparameters.OUTPUT_FILENAME = value

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
        self.set_DISPOSE_STFT_SPECS(profile.vc_conf.dispose_stft_specs)
        self.set_DISPOSE_CONV1D_SPECS(profile.vc_conf.dispose_conv1d_specs)
        if hasattr(profile.others, "input_filename"):
            self.set_INPUT_FILENAME(profile.others.input_filename)
        if hasattr(profile.others, "output_filename"):
            self.set_OUTPUT_FILENAME(profile.others.output_filename)

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
        
    def audio_trans_GPU(self, tdbm, input, net_g, noise_data, target_id, dispose_stft_specs, dispose_conv1d_specs):
        hop_length = Hyperparameters.HOP_LENGTH
        dispose_stft_length = dispose_stft_specs * hop_length
        dispose_conv1d_length = dispose_conv1d_specs * hop_length
    
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
            trans_length = signal.size()[0]
            text, spec, wav, sid = tdbm.get_audio_text_speaker_pair(signal.view(1, trans_length), ["m", Hyperparameters.SOURCE_ID, "m"])
            if dispose_stft_specs != 0:
                # specの頭と終がstft paddingの影響受けるので2コマを削る
                # wavもspecで削るぶんと同じだけ頭256と終256を削る
                spec = spec[:, dispose_stft_specs:-dispose_stft_specs]
                wav = wav[:, dispose_stft_length:-dispose_stft_length]
            data = TextAudioSpeakerCollate()([(text, spec, wav, sid)])
            x, x_lengths, spec, spec_lengths, y, y_lengths, sid_src = [x.cuda() for x in data]

            sid_target = torch.LongTensor([target_id]).cuda() # 話者IDはJVSの番号を100で割った余りです
            audio = net_g.cuda().voice_conversion(spec, spec_lengths, sid_src=sid_src, sid_tgt=sid_target)[0][0,0].data.cpu().float().numpy()

        if dispose_conv1d_specs != 0:
            # 出力されたwavでconv1d paddingの影響受けるところを削る
            audio = audio[dispose_conv1d_length:-dispose_conv1d_length]
        audio = audio * Hyperparameters.MAX_WAV_VALUE
        audio = audio.astype(np.int16).tobytes()

        return audio

    def overlap_merge(self, now_wav, prev_wav, overlap_length):
        """
        生成したwavデータを前回生成したwavデータとoverlap_lengthだけ重ねてグラデーション的にマージします
        終端のoverlap_lengthぶんは次回マージしてから再生するので削除します

        Parameters
        ----------
        now_wav: 今回生成した音声wavデータ
        prev_wav: 前回生成した音声wavデータ
        overlap_length: 重ねる長さ
        """
        gradation = np.arange(overlap_length) / overlap_length
        now = np.frombuffer(now_wav, dtype='int16')
        prev = np.frombuffer(prev_wav, dtype='int16')
        now_head = now[:overlap_length]
        prev_tail = prev[-overlap_length:]
        merged = prev_tail * (1 - gradation) + now_head * gradation
        overlapped = np.append(merged, now[overlap_length:-overlap_length])
        signal = np.round(overlapped, decimals=0)
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

        # テストファイル入出力のモックアップ
        mock_stream = MockStream(Hyperparameters.SAMPLE_RATE)
        if Hyperparameters.INPUT_FILENAME != None:
            mock_stream.open_inputfile(Hyperparameters.INPUT_FILENAME)
            audio_input_stream = mock_stream
        if Hyperparameters.OUTPUT_FILENAME != None:
            mock_stream.open_outputfile(Hyperparameters.OUTPUT_FILENAME)
            audio_output_stream = mock_stream

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

        with_bgm = (Hyperparameters.INPUT_DEVICE_2 != False)
        delay_frames = Hyperparameters.DELAY_FLAMES
        overlap_length = Hyperparameters.OVERLAP
        target_id = Hyperparameters.TARGET_ID
        wav_bytes = 2 # 1音声データあたりのデータサイズ(2bytes) (math.log2(max_wav_value)+1)/8 で算出してもよいけど
        hop_length = Hyperparameters.HOP_LENGTH
        dispose_stft_specs = Hyperparameters.DISPOSE_STFT_SPECS
        dispose_conv1d_specs = Hyperparameters.DISPOSE_CONV1D_SPECS
        dispose_specs =  dispose_stft_specs * 2 + dispose_conv1d_specs * 2
        dispose_length = dispose_specs * hop_length
        assert(delay_frames >= dispose_length + overlap_length)

        #第一節を取得する
        try:
            print("準備が完了しました。VC開始します。")

            prev_wav_tail = bytes(0)
            in_wav = prev_wav_tail + audio_input_stream.read(delay_frames, exception_on_overflow=False)
            trans_wav = self.audio_trans_GPU(tdbm, in_wav, net_g, noise_data, target_id, 0, 0) # 遅延減らすため初回だけpadding対策使わない
            overlapped_wav = trans_wav
            prev_trans_wav = trans_wav
            if dispose_length + overlap_length != 0:
                prev_wav_tail = in_wav[-((dispose_length + overlap_length) * wav_bytes):] # 次回の頭のデータとして終端データを保持する
            if with_bgm:
                back_in_raw = back_audio_input_stream.read(delay_frames, exception_on_overflow = False) # 背景BGMを取得
            while True:
                audio_output_stream.write(overlapped_wav)
                in_wav = prev_wav_tail + audio_input_stream.read(delay_frames, exception_on_overflow=False)
                trans_wav = self.audio_trans_GPU(tdbm, in_wav, net_g, noise_data, target_id, dispose_stft_specs, dispose_conv1d_specs)
                overlapped_wav = self.overlap_merge(trans_wav,  prev_trans_wav, overlap_length)
                prev_trans_wav = trans_wav
                if dispose_length + overlap_length != 0:
                    prev_wav_tail = in_wav[-((dispose_length + overlap_length) * wav_bytes):] # 今回の終端の捨てデータぶんだけ次回の頭のデータとして保持する
                if with_bgm:
                    back_audio_output_stream.write(back_in_raw)
                    back_in_raw = back_audio_input_stream.read(delay_frames, exception_on_overflow=False) # 背景BGMを取得

                #声id変更 数字キーの0～9で切り替え
                for k in range(10) :
                    if keyboard.is_pressed(str(k)) :
                        if k < len(Hyperparameters.VOICE_LIST) :
                            target_id = Hyperparameters.VOICE_LIST[k]
                            print(f"voice id: {target_id}")

                if Hyperparameters.VC_END_FLAG: #エスケープ
                    print("vc_finish")
                    break

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

class MockStream:
    """
    オーディオストリーミング入出力をファイル入出力にそのまま置き換えるためのモック
    """
    def __init__(self, sampling_rate):
        self.sampling_rate = sampling_rate
        self.start_count = 2
        self.end_count = 2
        self.fr = None
        self.fw = None

    def open_inputfile(self, input_filename):
        self.fr = wave.open(input_filename, 'rb')

    def open_outputfile(self, output_filename):
        self.fw = wave.open(output_filename, 'wb')
        self.fw.setnchannels(1)
        self.fw.setsampwidth(2)
        self.fw.setframerate(self.sampling_rate)

    def read(self, length, exception_on_overflow=False):
        if self.start_count > 0:
            wav = bytes(length * 2)
            self.start_count -= 1 # 最初の2回はダミーの空データ送る
        else:
            wav = self.fr.readframes(length)
        if len(wav) <= 0: # データなくなってから最後の2回はダミーの空データを送る
            wav = bytes(length * 2)
            self.end_count -= 1
            if self.end_count < 0:
                raise ValueError("End of data.")
        return wav

    def write(self, wav):
        self.fw.writeframes(wav)

    def stop_stream(self):
        pass
    
    def close(self):
        if self.fr != None:
            self.fr.close()
            self.fr = None
        if self.fw != None:
            self.fw.close()
            self.fw = None

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


    
