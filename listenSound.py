import numpy as np
import sounddevice as sd
from scipy import signal
from scipy.io import wavfile
import plotly.graph_objects as go
from scipy.ndimage import gaussian_filter1d





sampleRate = 44100 # samples per second - intrinsec to sound system
Ts = 0.1 # seconds - symbol time


audioFile = 'test.wav'
plot = True

# Sequencer
nSeq = 4
seq = signal.max_len_seq(nSeq)[0]
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


symbolMapping = [b'00', b'01', b'11', b'10']



# Moving average filter
w = 1000
mAvg = (np.ones((1,w))/w)[0]


# Sequencer
nSeq = 4
seq = signal.max_len_seq(nSeq)[0]
seqStr = ''.join(chr(i + 48) for i in seq)
seqIntValue = int(seqStr,2)


# Audio File
sampleRate, receivedData = wavfile.read(audioFile)

if (receivedData.dtype == 'int16'):
    receivedData = receivedData / float(2^15)


if plot:
    fftSignal = np.abs(np.fft.fft(receivedData))
    ft = np.fft.fftfreq(len(receivedData),1/(sampleRate))

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=fftSignal,
                             x=ft,
                        name='fft'))
    fig.update_xaxes(range=[0, 1000])
    fig.update_yaxes(type="log")
    fig.update_layout(
        title={
            'text': "Input signal - FFT",
            'xanchor': 'center',
            'yanchor': 'top'},
        yaxis_title="Intensity (int16)",
        xaxis_title="[Hz]",
    )

    fig.show()




# Filter h(t)
ht = []
for freq in f:
    ht.append(np.sin(2*np.pi*freq*t))


# Convolucao com h(t) e valor absoluto
rt = []
for h in ht:
    rt.append(abs(np.convolve(receivedData, h, mode='same')))


# APLICACAO DA MEDIA MOVEL E NORMALIZACAO
mt = []
for r in rt:
    movingAvg = np.convolve(r,mAvg,mode='same')
    mt.append(movingAvg/max(movingAvg))


if plot:
    fig = go.Figure()
    for i in range(len(f)):
        fig.add_trace(go.Scatter(y=mt[i],
                            name='{} Hz'.format(f[i])))

    fig.update_layout(
        title={
            'text': "Signal after convolution w/ sin(x), normalization and moving avg filtering",
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'},
        yaxis_title="to be decoded",
        xaxis_title="sample #",
    )
    fig.show()
# ------------------------------



# INDICE INICIAL PARA AMOSTRAGEM
initialValue = mt[0][0]

for idx in range(len(mt[0])):
    if(initialValue > 0.5):
        if(mt[0][idx] < 0.5):
            firstIndex = int(idx + Ts*sampleRate/2)
            break
    else:
        if(mt[0][idx] > 0.5):
            firstIndex = int(idx + Ts*sampleRate/2)
            break




# CRIA ARRAY DE BITS
incomingBits = b''
evaluateIndex = firstIndex

while (evaluateIndex < len(mt[0])):
    if(nSymbols == 2):
        if(mt[0][evaluateIndex] > mt[1][evaluateIndex]):
            incomingBits += b'0'
        else:
            incomingBits += b'1'


    elif(nSymbols == 4):
        maxValue = 0
        for m in mt:
            maxValue = max(m[evaluateIndex],maxValue)
        
        for i in range(len(mt)):
            if maxValue == mt[i][evaluateIndex]:
                incomingBits += symbolMapping[i]
    evaluateIndex += int(sampleRate*Ts)



# FIND SEQUENCER DELIMITATION INDEX - MINIMAL DISTANCE METHOD
distances = []
minIndex = []
for i in range(len(incomingBits) - (2**nSeq - 1) + 1):
    recebidoInt = int(incomingBits[i:i+15],2)
    distances.append(abs(seqIntValue - recebidoInt))

for idx in range(len(distances)):
    if distances[idx] == min(distances):
        minIndex.append(idx)


# QUANTIDADE DE SEQUENCIAS RECEBIDAS
decodedMessage = {}
receivedSequences = len(minIndex) - 1


# INFORMACOES DE CADA SEQUENCIA
for i in range(receivedSequences):
    decodedMessage[i] = {'payloadIndex': minIndex[i]+15,
                         'payloadBytes': int(len(incomingBits[minIndex[i]+15:minIndex[i+1]])/8),
                         'receivedMessage': ""}

# DECODIFICACAO DA MENSAGEM ASCII
for sequence in decodedMessage.keys():
    idx = decodedMessage[sequence]['payloadIndex']
    for n in range(decodedMessage[sequence]['payloadBytes']):
        byte = incomingBits[idx+8*n:idx+8*n+8]
        caracter = (chr(int(byte,2)))
        decodedMessage[sequence]['receivedMessage'] += caracter

print(decodedMessage)