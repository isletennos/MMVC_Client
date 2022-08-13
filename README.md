MMVC_Client
====

AIを使ったリアルタイムボイスチェンジャー

## Description
AIを使ったリアルタイムボイスチェンジャー「MMVC(RealTime-Many to Many Voice Conversion)」  
の本体です。  
MMVC_Trainerで学習したモデルを使ってリアルタイムでVCを行います。
## MMVC_Trainer
https://github.com/isletennos/MMVC_Trainer
## concept
「簡単」「だれでも」「好きな声に」「リアルタイムで」
## Requirement
・MMVC_Trainerで学習したモデルとそのコンフィグ  
## Install
### windows かつ 実行ファイルを利用する方
下記URLからGPU版をダウンロードして、展開してください。(ファイルサイズが非常に大きいので注意)  
[MMVC_client v0.3.0.0(GPU ver)](https://drive.google.com/file/d/1QXJQAnTOr8vE5nwxInUROtj-fiHeJsXH/view?usp=sharing)  
#### ファイルサイズが大きすぎてDLできない人向けの分割版
[MMVC_client v0.3.0.0(GPU ver)](https://drive.google.com/drive/folders/1eoDBw37WT7wJsAXh-RIXvXLvbSwnDtt9?usp=sharing)  

### 旧ver
[MMVC_client v0.2.0.1(GPU ver)](https://drive.google.com/file/d/1JEvYw4vjiBwhsZq79Pb0Doh7Fy16dK76/view?usp=sharing)  
[MMVC_client 無印(CPU_ver) (現在非推奨)](https://drive.google.com/file/d/1KLqo_q-qbahPRzNo2kUhCqHqnb8lTjMJ/view?usp=sharing)  
[MMVC_client 無印(GPU ver)](https://drive.google.com/file/d/1XNdfT3BFGKlxDm43hEbYvnoJSecjLedt/view?usp=sharing)  

#### TrainnerとClientの対応表
| ver 対応表                | MMVC Trainner v1.2.0.x | MMVC Trainner v1.2.1.x | MMVC Trainner v1.3.0.x | 
| ------------------------- | ---------------------- | ---------------------- | ---------------------- | 
| MMVC Client 無印(CPU/GPU) | 〇                     | 〇                     | ×
| MMVC Client v0.2.0.x(GPU) | 〇                     | 〇                     | ×
| MMVC Client v0.3.0.x(GPU) | ×                     | ×                     | 〇

## Install(python)
このリポジトリをダウンロードして、展開してください。  
また、下記.exeの実行を.pyの実行に置き換えて実行してください。  

## Usage
### 1. 使用可能なオーディオデバイス一覧の取得
「output_audio_device_list.exe」を実行します。  
「audio_device_list.txt」が実行ファイルと同じディレクトリに出力されます。  
こちらに入出力のオーディオデバイス名およびIDが出力されており、下記セクション以降で利用します。  
### 2. myprofile.confの書き換え
myprofile.confの下記項目を環境に合わせて変更します。  
```
  "device": {
    "input_device1":1,
    "input_device2":false,
    "output_device":12,
    "gpu_id":0
  },
```

```
  "vc_conf": {
    "frame_length":8192,
    "delay_flames":4096,
    "overlap":1024,
    "dispose_stft_specs":2,
    "dispose_conv1d_specs":10,
    "source_id":107,
    "target_id":100
  },
```

```
  "path": {
    "json":"D:\\GitRepository\\MMVC_Client\\brunch\\models\\config.json",
    "model":"D:\\GitRepository\\MMVC_Client\\brunch\\models\\G_30000.pth",
    "noise":"D:\\GitRepository\\MMVC_Client\\brunch\\MMVC_Client_old\\noise.wav"
  },
```

```
  "others": {
    "use_nr":false,
    "voice_selector":false,
    "voice_list": [100, 108, 107, 6, 30, 108],
    "voice_label": ["ずんだもん", "目標話者", "自分の声", "女性の声", "男性の低い声", "女性の高い声"]
  }
```
### 2.1 myprofile.confの書き換え(device)
このセクションでは、下記項目の変更方法について記載します。  
```
  "device": {
    "input_device1":1,
    "input_device2":false,
    "output_device":12,
    "gpu_id":0
  },
```
各要素はそれぞれ  
**input_device1 : マイク入力のデバイスID or デバイス名を指定します。**  


**input_device2 : 背景音声の入力のデバイスID or デバイス名を指定します。**  
主にカラオケ等背景のBGMと自分の変換後の音声のラグを0にしたいときに使います。  


**output_device : 変換した音声の出力先のデバイスID or デバイス名を指定します。**  


**gpu_id : 複数GPUをPCに搭載している場合、数字で指定できます。**  
使い分けが不要な場合は0のまま変更は不要です。 

### 2.2 myprofile.confの書き換え(vc_conf)
このセクションでは、下記項目の変更方法について記載します。  
```
  "vc_conf": {
    "frame_length":8192,
    "delay_flames":4096,
    "overlap":1024,
    "dispose_stft_specs":2,
    "dispose_conv1d_specs":10,
    "source_id":107,
    "target_id":100
  },
```
この項目では、下記2項目のみ変更します。それ以外の項目については割愛します。  
**source_id : 変換元の音声の話者IDになります。**  
Trainerで特に弄っていなければ、107のままで問題ありません。  


**target_id : 変換先の音声の話者IDになります。**  
学習時に生成した「./filelists/train_config_Correspondence.txt」を参考に話者IDを指定してください。  
チュートリアルもんであれば100のままで問題ありません。  

### 2.3 myprofile.confの書き換え(path)
このセクションでは、下記項目の変更方法について記載します。  
```
  "path": {
    "json":"D:\\GitRepository\\MMVC_Client\\brunch\\models\\config.json",
    "model":"D:\\GitRepository\\MMVC_Client\\brunch\\models\\G_30000.pth",
    "noise":"D:\\GitRepository\\MMVC_Client\\brunch\\MMVC_Client_old\\noise.wav"
  },
```
各要素はそれぞれ  
**json : 学習時に生成したconfigファイルのパスを指定します。**  
./configs/train_config_zundamon.json  
等のファイルがあるはずです。  


**model : 学習したモデルのパスを指定します。**  
./logs/xxxx/G_xxxx.pth といった感じにファイルが配置されているはずです。  
また、上記で指定するjsonファイルは  
./logs/xxxx/config.json  
でも問題ありません。  


**noise : 現在非推奨で使わないのでそのままでいいです。**  
使いたい方は下記おまけセクションを参考ください。  
**※ここで指定するパスは必ず「\」ではなく「\\\\」で区切ってください。**  

### 2.4 myprofile.confの書き換え(others)
このセクションでは、下記項目の変更方法について記載します。  
```
  "others": {
    "use_nr":false,
    "voice_selector":false,
    "voice_list": [100, 108, 107, 6, 30, 108],
    "voice_label": ["ずんだもん", "目標話者", "自分の声", "女性の声", "男性の低い声", "女性の高い声"]
  }
```
各要素はそれぞれ  
**use_nr : ノイズリダクションを有効化するかしないか指定します。**  
現状は品質が下がるため、自前で用意することを推奨します。  
この機能を使う場合、 true に書き換えてください。  


**voice_selector : MMVC起動中にターゲット話者をリアルタイムで変更する機能を有効化するかしないか指定します。**  
この機能を十全に使うには、複数話者の同時学習を行う必要があります。  
複数話者の同時学習を行っていない場合はfalseのままにしておいてください。  


**voice_list : voice_selectorを有効化したときに利用する項目です。学習した話者IDを記載します。**  


**voice_label : voice_selectorを有効化したときに利用する項目です。話者IDのラベルになります。**  

### 3. ソフトウェアの起動
パターン1
「mmvc_client_GPU.bat」を実行  
正しく「myprofile.conf」が設定されていればそのまま起動します。

パターン2
「mmvc_client_GPU.exe」を実行してください。  
起動に少しだけ時間がかかります。  
起動すると「myprofile.conf」のパスを聞かれるので、パスを指定して下さい。  

### おまけ:ノイズリダクションの有効化
####  1. ノイズ音取得の実行
「rec_environmental_noise.exe」を実行します。  
実行したら、モデルを学習したときに設定したサンプリングレートを入力してください。  
(MMVC_Trainerの設定を変えていなければ24000です)  
次にmyprofile.confのパスを聞かれるため、編集したmyprofile.confのパスを入力してください。  
以下の入力パスの例のように、.confファイルまで含めて入力して下さい。  
```
D:\mmvc_client_GPU\myprofile.conf
```
※注意として、入力パスの両端に”（ダブルクォーテーション）は付けないでください。  
パスの入力とmyprofile.confに問題が無ければ、ノイズの録音が開始されます。  
ノイズの録音が完了するまで、マイクに話しかけたり等しないで、待ちます。  
「noise.wav」が実行ファイルと同じディレクトリに出力されます。  

####  2. myprofile.confの書き換え
```
  "path": {
    "json":"D:\\GitRepository\\MMVC_Client\\config.json",
    "model":"D:\\GitRepository\\MMVC_Client\\G_348000.pth",
    "noise":"D:\\GitRepository\\MMVC_Client\\noise.wav"
  }
```
上記項目の"noise"に 1. ノイズ音取得の実行 で作成した「noise.wav」のパスを入力します。  
```
  "others": {
    "use_nr":false,
    "voice_selector":false,
    "voice_list": [100, 108, 107, 6, 30, 108],
    "voice_label": ["ずんだもん", "目標話者", "自分の声", "女性の声", "男性の低い声", "女性の高い声"]
  }
```
上記項目の"use_nr"をtrueに変えます。  

## Reference
https://arxiv.org/abs/2106.06103  
https://github.com/jaywalnut310/vits  
https://github.com/timsainb/noisereduce
## Author
Isle Tennos  
Twitter : https://twitter.com/IsleTennos

