# -*- coding: utf-8 -*
import pyaudio
import wave
import numpy as np
import time

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

    p = pyaudio.PyAudio()
    stream = p.open(format = pyaudio.paInt16,
                    channels = 1,
                    rate = sr,
                    input = True,
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