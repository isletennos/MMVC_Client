import logging
import platform
from multiprocessing import freeze_support

# 以下必要な外部ライブラリ
# pip install --upgrade py-cpuinfo
# pip install --upgrade psutil
# pip install --upgrade nvgpu

# Pipfileとpipenvを使用する場合
# cd pyhon
# pipenv install --dev
# pipenv run python setup_check.py


# cpuinfo.get_cpu_info()とpyinstallerを組み合わせる場合に必要
freeze_support()



# 定数
MMVC_INFO:        str = "MMVC_Client"
OUTPUT_FILE_NAME: str = "device_check.txt"

LOWER_LIMIT_MEMORY:     int = 4 * 1024**3 # 4 GiB
LOWER_LIMIT_GPU_MEMORY: int = 1 * 1024**3 # 1 GiB

ONNX_TEXT: str = "このPCでは、onnxモデルを出力することで動作する可能性があります"



# ログファイル出力の設定
logging.basicConfig(
    filename = OUTPUT_FILE_NAME,
    filemode = "w",
    encoding = "utf-8",
    level = logging.INFO,
    format = "%(levelname)s%(message)s")
logging.addLevelName(logging.INFO, "")
logging.addLevelName(logging.WARNING, "\n[警告]\n")
logging.addLevelName(logging.ERROR, "\n[エラー]\n")



# 基本的な情報
logging.info(f"バージョン: {MMVC_INFO}")
logging.info(f"Python: {platform.python_version()}")
logging.info(f"アーキテクチャ: {platform.machine()}")
logging.info(f"OS: {platform.system()}")



# CPU関連
try:
    from cpuinfo import get_cpu_info
    
    cpu_info = get_cpu_info()
    logging.info(f"CPU: {cpu_info['brand_raw']}")

except ModuleNotFoundError:
    logging.info(f"CPU: {platform.processor()}")
    logging.warning("py-cpuinfoライブラリがインストールされていません\n" +
                    "以下のコマンドを実行して、py-cpuinfoをインストールするとより詳細な情報を得られます\n" +
                    "pip install --upgrade py-cpuinfo\n")


# メモリ
try:
    from psutil import virtual_memory
    memory = virtual_memory().total
    logging.info(f"メモリ: {round(memory / 1024**3, 0)} GiB")

    if memory < LOWER_LIMIT_MEMORY:
        logging.error("メモリが不足しています\n" +
                      "メモリを増設することで動作不良が改善される場合があります\n")
except ModuleNotFoundError:
    logging.error("psutilライブラリがインストールされていません\n" +
                  "以下のコマンドを実行して、psutilをインストールする必要があります\n" +
                  "pip install --upgrade psutil\n")


# GPU
try:
    import nvgpu
    
    gpu_infos = nvgpu.gpu_info()
    gpu_memory = 0
    
    for gpu_info in gpu_infos:
        temp_gpt_memory = gpu_info["mem_total"] * 1024**2
        logging.info(f"GPU {gpu_info['index']} 名称: {gpu_info['type']}")
        logging.info(f"GPU {gpu_info['index']} メモリ: {round(temp_gpt_memory / 1024**3, 1)} GiB")
        gpu_memory = max(gpu_info["mem_total"] * 1024**2, gpu_memory)
    
    if len(gpu_infos) == 0:
        logging.warning(f"NvidiaのGPUが存在しません\n{ONNX_TEXT}\n")
    
    elif gpu_memory < LOWER_LIMIT_GPU_MEMORY:
        logging.warning(f"GPUのメモリ量が不足しています\n{ONNX_TEXT}\n")

except ModuleNotFoundError:
    logging.error("nvgpuライブラリがインストールされていません\n" +
                  "以下のコマンドを実行して、nvgpuをインストールする必要があります\n" +
                  "pip install --upgrade nvgpu\n")

except FileNotFoundError:
    # nvidia-smiパッケージが見つからない場合
    logging.warning(f"NvidiaのGPUもしくはGPUドライバーが存在しません\n{ONNX_TEXT}\n")



logging.info("デバイス情報取得完了")
print(f"デバイス情報の取得が完了しました。\n{OUTPUT_FILE_NAME} を確認してください。\nこのウィンドウは閉じて問題ありません。")