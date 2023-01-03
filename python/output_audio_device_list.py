# -*- coding: utf-8 -*
import pyaudio
from os import linesep

def main():
    audio = pyaudio.PyAudio()
    audio_devices = list()
    host_apis = list()
    
    for api_index in range(audio.get_host_api_count()):
        host_apis.append(audio.get_host_api_info_by_index(api_index)['name'])
    
    # 音声デバイス毎のインデックス番号を一覧表示
    for x in range(0, audio.get_device_count()):
        devices = audio.get_device_info_by_index(x)
        try:
            device_name = devices['name'].encode('shift-jis').decode('utf-8')
        except UnicodeDecodeError:
            device_name = devices['name']
        
        device_name = device_name.replace(linesep, '') + ", " + host_apis[devices['hostApi']]
        
        isInOut = ""
        if devices['maxInputChannels'] > 0:
            isInOut += "入"
        if devices['maxOutputChannels'] > 0:
            isInOut += "出"
        
        audio_devices.append(f"{isInOut}力： Index：{devices['index']} デバイス名：\"{device_name}\"\n")

    with open('audio_device_list.txt', 'w', encoding='utf-8') as f:
        f.writelines(audio_devices)

    print(" 使用可能なデバイス一覧の取得が完了しました。\n audio_device_list.txt を参照してください。\n このウィンドウは閉じて問題ありません。")

if __name__ == '__main__':
    main()