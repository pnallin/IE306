import sounddevice as sd
import numpy as np
from scipy import signal
from scipy.io import wavfile
import plotly
import plotly.graph_objects as go

sampleRate = 44100 # samples per second - intrinsec to sound system
Ts = 0.05 # seconds - symbol time



audioFile = 'signal.wav'
plot = True

# ----- SEQUENCER - Sequencia de treinamento
nSequencer = 6
nbitsSequencer = (2**nSequencer) - 1
seq = signal.max_len_seq(nSequencer)[0]
seqStr = '0' + ''.join(chr(i + 48) for i in seq)
seqIntValue = int(seqStr,2)


# ----- FSK - Frequencias de audio para representação dos simbolos
nSymbols = 4
f0 = 440
f1 = 770
f2 = 550
f3 = 660
f = [f0, f1, f2, f3]
t = np.arange(0, Ts, 1/sampleRate)



symbolMapping = {
    # 2 simbolos:
    2: {
        "0":  np.sin(2*np.pi*f0*t),
        "1":  np.sin(2*np.pi*f1*t)
    },    
    # 4 simbolos:
    4: {
        "00": np.sin(2*np.pi*f0*t),
        "01": np.sin(2*np.pi*f1*t),
        "11": np.sin(2*np.pi*f2*t),
        "10": np.sin(2*np.pi*f3*t)
    }
}

receivedSymbolMapping = {
    2: [b'0', b'1'],
    4: [b'00', b'01', b'11', b'10']
}

# ----- Sequencer
minDistanceAccepted = 0.25

# ----- Moving average filter
w = 100
mAvg = (np.ones((1,w))/w)[0]


# ARQUIVO DE AUDIO
nSequencias = 3
audioFile = "amostraSom_seq{}_rept{}_nSymbols{}_Ts{}ms.wav".format(nSequencer, nSequencias, nSymbols, int(Ts*1000))




# Open Audio File
sampleRate, receivedData = wavfile.read("audios/" + audioFile)

if (receivedData.dtype == 'int16'):
    receivedData = receivedData / float(2**15)

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
        title={'text': "Received signal - FFT",
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'},
        yaxis_title="Intensity (int16)",
        xaxis_title="[Hz]",
    )

    figname = "recv__fft_mvAvg{}_seq{}_nSymbols{}_Ts{}ms.html".format(w, nSequencer, nSymbols, int(Ts*1000))
    plotly.offline.plot(fig, filename = "graficos/" + figname, auto_open=False)
    fig.show(renderer="png")
    print("GRAFICO INTERATIVO CORRESPONDENTE: {}".format(figname))



# Filter h(t) == senoides representativas de cada simbolo
# Convolucao com h(t) e valor absoluto
rt = []
for sinf in symbolMapping[nSymbols].keys():
    rt.append(abs(np.convolve(receivedData, symbolMapping[nSymbols][sinf], mode='same')))


# APLICACAO DA MEDIA MOVEL E NORMALIZACAO
mt = []
for r in rt:
    movingAvg = np.convolve(r,mAvg,mode='same')
    mt.append(movingAvg/max(movingAvg))


if plot:
    fig = go.Figure()
    tx=np.linspace(0, len(mt[0])/sampleRate, len(mt[0]))
    for i in range(len(f)):
        fig.add_trace(go.Scatter(y=mt[i],
                              #  x=tx,
                            name='{} Hz'.format(f[i])))

    fig.update_layout(
        title={
            'text': "Signal after convolution w/ sin(x), normalization and moving avg filtering",
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'},
        yaxis_title="to be decoded [au]",
        xaxis_title="sample #",
    )
    fig.show(renderer="png")
    figname = "recv_mvAvg{}_seq{}_nSymbols{}_Ts{}ms.html".format(w, nSequencer, nSymbols, int(Ts*1000))
    plotly.offline.plot(fig, filename = "graficos/" + figname, auto_open=False)
    
    
    fig = go.Figure()
    tx=np.linspace(0, len(mt[0])/sampleRate, len(mt[0]))
    for i in range(len(f)):
        fig.add_trace(go.Scatter(y=mt[i][:50000],
                              #  x=tx,
                            name='{} Hz'.format(f[i])))

    fig.update_layout(
        title={
            'text': "Zoom - Signal after convolution, normalization and moving avg filtering",
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'},
        yaxis_title="to be decoded [au]",
        xaxis_title="sample #",
    )
    fig.show(renderer="png")

    
    
    print("GRAFICO INTERATIVO CORRESPONDENTE: {}".format(figname))

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
sampleIndex = []
sampleValue = []

while (evaluateIndex < len(mt[0])):
    sampleIndex.append(evaluateIndex)
    if(nSymbols == 2):
        if(mt[0][evaluateIndex] > mt[1][evaluateIndex]):
            incomingBits += b'0'
            sampleValue.append(mt[0][evaluateIndex])
        else:
            incomingBits += b'1'
            sampleValue.append(mt[1][evaluateIndex])


    elif(nSymbols == 4):
        maxValue = 0
        for m in mt:
            maxValue = max(m[evaluateIndex],maxValue)
        sampleValue.append(maxValue)
        
        for i in range(len(mt)):
            if maxValue == mt[i][evaluateIndex]:
                incomingBits += receivedSymbolMapping[nSymbols][i]
    evaluateIndex += int(sampleRate*Ts)



if plot:
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=mt[0][:50000],
                            name='{} Hz'.format(f[0])))
        
    fig.add_vline(x=firstIndex-Ts*sampleRate/2, 
                  line_dash="dot", 
                  annotation_text="First middle point index: {} Hz".format(int(firstIndex-Ts*sampleRate/2)), 
                  line_color="green")
    fig.update_layout(
        title={
            'text': "Selecting sampling starting point using signal at {} Hz".format(f[0]),
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'},
        yaxis_title=" to be decoded - Convolution with {} Hz [au]".format(f[0]),
        xaxis_title="sample #",
    )
    fig.update_yaxes(range=[0, 1.5])
    fig.show(renderer="png")
    
    
    fig = go.Figure()
    for i in range(len(f)):
        fig.add_trace(go.Scatter(y=mt[i][:50000],
                            name='{} Hz'.format(f[i])))
    
    fig.add_trace(go.Scatter(x=sampleIndex[:16], y=sampleValue[:16],
                    mode='markers', name='sampled', marker_symbol='x'))

    fig.update_layout(
        title={
            'text': " Sampling time",
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'},
        yaxis_title="to be decoded [au]",
        xaxis_title="sample #",
    )
    fig.show(renderer="png")
    
    fig = go.Figure()
    for i in range(len(f)):
        fig.add_trace(go.Scatter(y=mt[i],
                            name='{} Hz'.format(f[i])))
    
    fig.add_trace(go.Scatter(x=sampleIndex, y=sampleValue,
                    mode='markers', name='sampled', marker_symbol='x'))

    fig.update_layout(
        title={
            'text': " Sampling time - Processed signal",
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'},
        yaxis_title="to be decoded [au]",
        xaxis_title="sample #",
    )
    figname = "recv_mvAvg{}_sampling_seq{}_nSymbols{}_Ts{}ms.html".format(w, nSequencer, nSymbols, int(Ts*1000))
    plotly.offline.plot(fig, filename = "graficos/" + figname, auto_open=False)
    print("GRAFICO INTERATIVO CORRESPONDENTE: {}".format(figname))
    
    
print("")
print("")
print("TOTAL DE BITS DECODIFICADOS: {}".format(len(incomingBits)))
print("SEQUENCIA DECODIFICADA: {}".format(incomingBits))

# FIND SEQUENCER DELIMITATION INDEX - MINIMAL DISTANCE METHOD
minIndex = []
distances = []

for i in range(len(incomingBits) - nbitsSequencer + 1):
    recebidoInt = int(incomingBits[i:i+nbitsSequencer],2)
    normalizedDistances = (float(bin(recebidoInt ^ seqIntValue).count("1"))) / (2**nSequencer)
    distances.append(normalizedDistances*100)
    if normalizedDistances < minDistanceAccepted:
        minIndex.append(i)
        

if plot:
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=distances))
    
    fig.add_hline(y=minDistanceAccepted*100, 
                  line_dash="dot", 
                  annotation_text="minDistance", 
                  line_color="green")


    fig.update_layout(
        title={
            'text': "Distances from sequencer",
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'},
        yaxis_title="Deviation from {}-bit sequencer [%]".format(2**nSequencer-1),
        xaxis_title="bit index",
    )
    figname = "recv_distances_mvAvg{}_sampling_seq{}_nSymbols{}_Ts{}ms.html".format(w, nSequencer, nSymbols, int(Ts*1000))
    plotly.offline.plot(fig, filename = "graficos/" + figname, auto_open=False)
    fig.show(renderer="png")
    print("GRAFICO INTERATIVO CORRESPONDENTE: {}".format(figname))    


    # QUANTIDADE DE SEQUENCIAS RECEBIDAS
decodedMessage = {}
receivedSequences = len(minIndex) - 1


# INFORMACOES DE CADA SEQUENCIA
for i in range(receivedSequences): 
    decodedMessage[i] = {'payloadIndex': minIndex[i]+nbitsSequencer,
                         'payloadBytes': int(len(incomingBits[minIndex[i]+nbitsSequencer:minIndex[i+1]])/8),
                         'receivedMessage': ""}

# DECODIFICACAO DA MENSAGEM ASCII
for sequence in decodedMessage.keys():
    idx = decodedMessage[sequence]['payloadIndex']
    for n in range(decodedMessage[sequence]['payloadBytes']):
        byte = incomingBits[idx+8*n:idx+8*n+8]
        caracter = (chr(int(byte,2)))
        decodedMessage[sequence]['receivedMessage'] += caracter



print("SEQUENCIAS DE TREINAMENTO ENCONTRADAS: {}".format(len(minIndex)))
print("")
print("")

for msg in (decodedMessage.keys()):
    print("----- Mensagem #{}".format(msg+1))
    print("Index inicial: {}".format(decodedMessage[msg]["payloadIndex"]))
    print("Quantidade de caracteres: {}".format(decodedMessage[msg]["payloadBytes"]))
    print("Mensagem: {}".format(decodedMessage[msg]["receivedMessage"]))
    print("")