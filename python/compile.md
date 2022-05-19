nuitka --standalone --mingw64 --follow-imports --windows-icon-from-ico=D:\GitRepository\RT-MMVC_Client\use_exe.ico --enable-plugin=torch --enable-plugin=anti-bloat --enable-plugin=numpy --enable-plugin=multiprocessing --enable-plugin=tk-inter --assume-yes-for-downloads --user-plugin=D:\GitRepository\MMVC_Client\brunch\MMVC_Client\python\FixBuildPlugin_pytorch.py --include-plugin-directory=D:\GitRepository\MMVC_Client\brunch\MMVC_Client\python --nofollow-import-to=torchvision --no-prefer-source-code D:\GitRepository\MMVC_Client\brunch\MMVC_Client\python\mmvc_client_GPU.py  

1)_soundfile_data\... がないといわれるので、pythonの環境から_soundfile_dataディレクトリを直接持ってくる
2)llvmlite.dll がないといわれるので、pythonの環境からllvmliteディレクトリを直接持ってくる
3)librosa\... がないといわれるので、
4)cannot load filter definition for kaiser best と言われるので、python環境から、resampyを持ってくる

