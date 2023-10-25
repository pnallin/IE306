import numpy as np
import sounddevice as sd
from scipy import signal
from scipy.io import wavfile
import plotly.graph_objects as go


sampleRate = 44100 # samples per second - intrinsec to sound system
Ts = 0.1 # seconds - symbol time
nSequencias = 3 # to repeat payload data


audioFile = 'signal.wav'
plot = False

# ----- SEQUENCER
nSequencer = 4
seq = signal.max_len_seq(nSequencer)[0]
seqStr = ''.join(chr(i + 48) for i in seq)


# ----- FSK
nSymbols = 4
f0 = 440
f1 = 770
f2 = 550
f3 = 660
f = [f0, f1, f2, f3]
t = np.arange(0, Ts, 1/sampleRate)

x = []
for freq in f:
    x.append(np.sin(2*np.pi*freq*t))

symbolMapping = {
    "0":  x[0],
    "1":  x[1],
    "00": x[0],
    "01": x[1],
    "11": x[2],
    "10": x[3]
}


# ----- NOISE
mean_noise = 0
stdev = 0.75







# MESSAGE + SEQUENCER
mensagem = 'Transmissao de dados via FSK'

mensagemBitsStr = ''.join(format(ord(i), '08b') for i in mensagem)

seqEnviada = ''
for _ in range(nSequencias):
    seqEnviada += seqStr + mensagemBitsStr
seqEnviada += seqStr

while(len(seqEnviada) % int(np.log2(nSymbols))):
    seqEnviada = "0" + seqEnviada



# MAPPING SYMBOLS -> FSK
sinalEnviado = np.array([])
i=0
while(i < len(seqEnviada)):
    ini = i
    i += int(np.log2(nSymbols))
    end = i

    c = seqEnviada[ini:end]

    if c in symbolMapping.keys():
        sinalEnviado = np.concatenate((sinalEnviado, symbolMapping[c]))
    else:
        print("Symbol {} not mapped !".format(c))
   


# TO BE SENT
gt = sinalEnviado
gt += np.random.normal(mean_noise, stdev, len(gt))


# PLAY AUDIO
sd.play(gt, blocking=True)


# SAVE AUDIO INTO *.WAV FILE
scaled = np.int16(gt / np.max(np.abs(gt)) * 32767)
wavfile.write(audioFile, sampleRate, scaled)


