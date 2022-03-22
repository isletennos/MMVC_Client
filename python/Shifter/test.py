import numpy as np
import soundfile as sf

from shifter import Shifter
from scipy.io import wavfile

def main():
    # read input .wav file
    fs, x = wavfile.read("D:\GitRepository\RT-MMVC_Client\python\Shifter\in.wav")
    print(fs)

    # F0 transoformation based on WSOLA and resampling
    f0trans = Shifter(fs, 1.0, frame_ms=20, shift_ms=10)
    transformed = f0trans.transform(x)
    print(type(transformed))

    # write output .wav file
    wavfile.write("out.wav", fs, np.array(transformed, dtype=np.int16))

if __name__ == '__main__':
    main()