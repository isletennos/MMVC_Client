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
## Demo
作成中
## Requirement
・MMVC_Trainerで学習したモデルとそのコンフィグ  
## Install
### windows かつ 実行ファイルを利用する方
下記URLからGPU版 or CPU版をダウンロードして、展開してください。(ファイルサイズが非常に大きいので注意)  
[MMVC_client(GPU ver) v0.2.0.1](https://drive.google.com/file/d/1MyIvxZ19BovpSsTZSNedofwSARmeaNH6/view?usp=sharing)  
[MMVC_client(CPU_ver) (現在非推奨)](https://drive.google.com/file/d/1KLqo_q-qbahPRzNo2kUhCqHqnb8lTjMJ/view?usp=sharing)

### 旧ver
[MMVC_client(GPU ver)](https://drive.google.com/file/d/1XNdfT3BFGKlxDm43hEbYvnoJSecjLedt/view?usp=sharing)  

#### TrainnerとClientの対応表
| ver 対応表                | MMVC Trainner v1.2.0.x | MMVC Trainner v1.2.1.x | 
| ------------------------- | ---------------------- | ---------------------- | 
| MMVC Client 無印(CPU/GPU) | 〇                     | 〇                     | 
| MMVC Client v0.2.0.x(GPU) | 〇                     | 〇                     | 

### pythonを利用する方
このリポジトリをダウンロードして、展開してください。  
また、下記.exeの実行を.pyの実行に置き換えて実行してください。
## Usage
### 1. 使用可能なオーディオデバイス一覧の取得
「output_audio_device_list.exe」を実行します。  
「audio_device_list.txt」が実行ファイルと同じディレクトリに出力されます。  

### 2. 環境によるノイズ音を取得
####  2.1. myprofileでの入力デバイスの設定
「myprofile.json」の"input_device1"を編集します。
"input_device1"に、実際に声を変換するマイクのオーディオデバイスのIDを指定します。
オーディオデバイスのIDは 
「### 1. 使用可能なオーディオデバイス一覧の取得」で作成した「audio_device_list.txt」
に書かれています。
####  2.2. ノイズ音取得の実行
「rec_environmental_noise.exe」を実行します。  
実行したら、モデルを学習したときに設定したサンプリングレートを入力してください。  
(MMVC_Trainerの設定を変えていなければ24000です)  
次にmyprofile.jsonのパスを聞かれるため、
「####  2.1. myprofileでの入力デバイスの設定」
で編集したmyprofile.jsonのパスを入力してください。
以下の入力パスの例のように、.jsonファイルまで含めて入力して下さい。
```
D:\mmvc_client_GPU\myprofile.json
```
※注意として、入力パスの両端に”（ダブルクォーテーション）は付けないでください。
パスの入力とmyprofile.jsonに問題が無ければ、ノイズの録音が開始されます。
ノイズの録音が完了するまで、マイクに話しかけたり等しないで、待ちます。  
「noise.wav」が実行ファイルと同じディレクトリに出力されます。  

### 3. 「myprofile.json」の設定
####  3.1. オーディオデバイスの設定(ここ詰まりポイント！)
「myprofile.json」の下記3項目を設定します。
```
{
  "device": {
    "input_device1":1,
    "input_device2":false,
    "output_device":9
  },
```
"input_device1"には、実際に声を変換するマイクなどのオーディオデバイスのIDを指定します。  
「audio_device_list.txt」にオーディオデバイス名とそのデバイスに対応するIDがリスト化されています。  
※同名のオーディオデバイスがリストに複数ある場合、基本的に若番を指定すればいいはずです。たぶん… 


"input_device2"は、声質変換しない入力を一緒に流したい場合に利用します。  
音声の遅延に合わせてBGMを出力することができます。


"output_device"には出力するオーディオデバイスを指定します。


####  3.2. VCのコンフィグ
「myprofile.json」の下記4項目を設定します。
```
  "vc_conf": {
    "frame_length":8192,
    "overlap":1024,
    "source_id":104,
    "target_id":105
  },
```
"frame_length" および "overlap" のパラメータはソフトウェア上での音声の遅延時間に影響します。  
このソフトウェア上で発生する遅延時間は  
```
[("frame_length" - "overlap") / サンプリングレート]秒
```
になります。  
"frame_length"は4096×nの値を入力してください。  
"overlap"は1024以上の値を指定することを推奨します。  

"source_id"には自分の声の話者IDを入力してください  
"target_id"には変換先の声の話者IDを入力してください  
話者IDについては、MMVC_Trainerのfilelistsディレクトリに「xxxx_Correspondence.txt」に音声データと話者IDの対応表があります。

####  3.3. 必要なファイルのパスの指定
「myprofile.json」の下記3項目を設定します。
```
  "path": {
    "json":"D:\\GitRepository\\MMVC_Client\\config.json",
    "model":"D:\\GitRepository\\MMVC_Client\\G_348000.pth",
    "noise":"D:\\GitRepository\\MMVC_Client\\noise.wav"
  }
```
"json"には学習に使ったコンフィグファイルのパスを  
"model"にはMMVC_Trainerで学習したモデルのパスを  
"noise"には 2. 環境によるノイズ音を取得 で作成した「noise.wav」のパスを入力します。
※注意として、.jsonファイル内でのパスは\を必ず2個続けて記述(\\)してください。
\が1個の場合、読み込めずエラーになります。

####  3.4. ノイズ削減機能のオン/オフ
```
  "others": {
    "use_nr":true
  }
```
"use_nr"をfalseにすると無効になります。  
不要な方はfalseにしてみてください。

### 4. ソフトウェアの起動
パターン1
「mmvc_client_GPU.bat」を実行  
正しく「myprofile.json」が設定されていればそのまま起動します。

パターン2
「mmvc_client_GPU.exe」を実行してください。  
起動に少しだけ時間がかかります。  
起動すると「myprofile.json」のパスを聞かれるので、パスを指定して下さい。  
```
UserWarning: stft will soon require the return_complex parameter be given for real inputs, and will further require that return_complex=True in a future PyTorch release. (Triggered internally at  ..\aten\src\ATen\native\SpectralOps.cpp:664.)
```
と表示されていれば、問題なく起動しています。  

## Q&A
順次更新
## Note
なにか不明点があればお気軽にご連絡ください。
## Reference
https://arxiv.org/abs/2106.06103  
https://github.com/jaywalnut310/vits  
https://github.com/timsainb/noisereduce
## Author
Isle Tennos  
Twitter : https://twitter.com/IsleTennos

