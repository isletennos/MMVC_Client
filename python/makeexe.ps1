pipenv run pyinstaller mmvc_client.py --add-binary "./.venv/Lib/site-packages/onnxruntime/capi/onnxruntime_providers_shared.dll;./onnxruntime/capi/" --add-binary "./.venv/Lib/site-packages/onnxruntime/capi/DirectML.dll;./onnxruntime/capi/" --collect-data librosa --onedir --clean -y
pipenv run pyinstaller output_audio_device_list.py --onefile
pipenv run pyinstaller rec_environmental_noise.py --onefile
