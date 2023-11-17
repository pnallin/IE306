import numpy as np
import sounddevice as sd
from scipy import signal
from scipy.io import wavfile
import plotly.graph_objects as go


sampleRate = 44100 # samples per second - intrinsec to sound system
Ts = 0.01 # seconds - symbol time
nSequencias = 3 # to repeat payload data


audioFile = 'signal.wav'
plot = True

# ----- SEQUENCER
nSequencer = 6
nbitsSequencer = (2**nSequencer) - 1
seq = signal.max_len_seq(nSequencer)[0]
seqStr = '0' + ''.join(chr(i + 48) for i in seq)
print(seqStr)


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







# MESSAGE
mensagem = 'Transmissao de dados via FSK'
mensagemBitsStr = ''.join(format(ord(i), '08b') for i in mensagem)




# MAPPING MESSAGE SYMBOLS -> FSK
msgMapped = np.array([])
i=0
while(i < len(mensagemBitsStr)):
    ini = i
    i += int(np.log2(nSymbols))
    end = i

    c = mensagemBitsStr[ini:end]

    if c in symbolMapping.keys():
        msgMapped = np.concatenate((msgMapped, symbolMapping[c]))
    else:
        print("Symbol {} not mapped !".format(c))


   

# MAPPING SEQ SYMBOLS -> FSK
seqMapped = np.array([])
i=0
while(i < len(seqStr)):
    ini = i
    i += int(np.log2(nSymbols))
    end = i

    c = seqStr[ini:end]

    if c in symbolMapping.keys():
        seqMapped = np.concatenate((seqMapped, 2*symbolMapping[c]))
    else:
        print("Symbol {} not mapped !".format(c))



# MESSAGE + SEQUENCER
gt = np.array([])
for _ in range(nSequencias):
    gt = np.concatenate((gt, seqMapped, msgMapped))
gt = np.concatenate((gt, seqMapped))





# TO BE SENT
gt += 2*np.random.normal(mean_noise, stdev, len(gt))
print("{} bits mapped !".format((nSequencias+1)*len(seqStr)+nSequencias*len(mensagemBitsStr)))

# PLAY AUDIO
sd.play(gt, blocking=True)


# SAVE AUDIO INTO *.WAV FILE
scaled = np.int16(gt / np.max(np.abs(gt)) * 32767)
wavfile.write(audioFile, sampleRate, scaled)


if plot:
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=gt))

    fig.update_layout(
        title={
            'text': "Sent data",
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'},
        xaxis_title="t [1/{} sec]".format(sampleRate),
    )
    fig.show()


