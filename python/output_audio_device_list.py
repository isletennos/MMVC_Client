# -*- coding: utf-8 -*
import pyaudio

def main():
    audio = pyaudio.PyAudio()
    audio_devices = list()
    isInOut = ""

    # 音声デバイス毎のインデックス番号を一覧表示
    for x in range(0, audio.get_device_count()): 
        devices = audio.get_device_info_by_index(x)
        if devices['maxInputChannels'] > 0:
            isInOut = isInOut + "入"
        if devices['maxOutputChannels'] > 0:
            isInOut = isInOut + "出"
        isInOut = isInOut + "力："
        
        audio_devices.append(isInOut + "Index : " + str(devices['index']) + "  デバイス名 : " + devices['name'] + "\n")
        
        isInOut = ""

    with open('audio_device_list.txt', 'w', encoding='utf-8') as f:
        f.writelines(audio_devices)

    print(" 使用可能なデバイス一覧の取得が完了しました。\n audio_device_list.txt を参照してください。\n このウィンドウは閉じて問題ありません。")

if __name__ == '__main__':
    main()