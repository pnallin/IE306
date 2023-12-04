import sounddevice as sd
import numpy as np
from scipy import signal
from scipy.io import wavfile
import plotly
import plotly.graph_objects as go

sampleRate = 44100 # samples per second - intrinsec to sound system
Ts = 0.05 # seconds - symbol time



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

    if c in symbolMapping[nSymbols].keys():
        msgMapped = np.concatenate((msgMapped, symbolMapping[nSymbols][c]))
    else:
        print("Bit(s) {} not mapped !".format(c))


   

# MAPPING SEQ SYMBOLS -> FSK
seqMapped = np.array([])
i=0
while(i < len(seqStr)):
    ini = i
    i += int(np.log2(nSymbols))
    end = i

    c = seqStr[ini:end]

    if c in symbolMapping[nSymbols].keys():
        seqMapped = np.concatenate((seqMapped, 2*symbolMapping[nSymbols][c]))
    else:
        print("Symbol {} not mapped !".format(c))

        
print("TAMANHO DA MENSAGEM: {} caracteres".format(len(mensagem)))
print("QUANTIDADE DE BITS NA MENSAGEM: {}".format(len(mensagemBitsStr)))
print("QUANTIDADE DE BITS NA SEQ.TREINAMENTO: {}".format(len(seqStr)))



# MESSAGE + SEQUENCER
nSequencias = 3 # to repeat payload data
gt = np.array([])

for _ in range(nSequencias):
    gt = np.concatenate((gt, seqMapped, msgMapped))
gt = np.concatenate((gt, seqMapped))



# ADD NOISE, IF WANTED
mean_noise = 0
stdev = 0.75
add_noise = False
if add_noise:
    gt += np.random.normal(mean_noise, stdev, len(gt))

# PLAY AUDIO
#sd.play(gt, blocking=True)


# SAVE AUDIO INTO *.WAV FILE
audioFile = "amostraSom_seq{}_rept{}_nSymbols{}_Ts{}ms.wav".format(nSequencer, nSequencias, nSymbols, int(Ts*1000))
scaled = np.int16(gt / np.max(np.abs(gt)) * 32767)
wavfile.write("audios/" + audioFile, sampleRate, scaled)



print("Total de {} bits mapeados!".format((nSequencias+1)*len(seqStr)+nSequencias*len(mensagemBitsStr)))
print("Arquivo de audio gerado: {}".format(audioFile))




# GRAFICOS ABAIXO
if plot:
    # SINAL GERADO    
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=gt, x=np.linspace(0, len(gt)/sampleRate, len(gt))))

    fig.update_layout(
        title={
            'text': "Generated signal",
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'},
        xaxis_title="t [sec]",
        yaxis_title="Intensity (au)",
    )
    figname = "sinal_seq{}_rept{}_nSymbols{}_Ts{}ms.html".format(nSequencer, nSequencias, nSymbols, int(Ts*1000))
    plotly.offline.plot(fig, filename = "graficos/" + figname, auto_open=False)
    fig.show(renderer="png")
    print("GRAFICO INTERATIVO CORRESPONDENTE: {}".format(figname))
    
    
    # "ZOOM" NO INICIO DO SINAL GERADO
    symbols2show = 3
    tzoom = int(symbols2show*sampleRate*Ts)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=gt[0:tzoom], x=np.linspace(0, len(gt)/sampleRate, len(gt))[0:tzoom]))

    fig.update_layout(
        title={
            'text': "Generated signal (Zoom)",
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'},
        xaxis_title="t [sec]",
        yaxis_title="Intensity (au)",
    )
    fig.show(renderer="png")
    
    
    # FFT DO SINAL
    fftSignal = np.abs(np.fft.fft(gt))
    ft = np.fft.fftfreq(len(gt),1/(sampleRate))
    

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=fftSignal,
                             x=ft,
                        name='fft'))
    fig.update_xaxes(range=[0, 1000])
    fig.update_yaxes(type="log")
    fig.update_layout(
        title={
            'text': "Generated signal - FFT",
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'},
        yaxis_title="Intensity (int16)",
        xaxis_title="[Hz]",
    )

    figname = "sinal_FFT_seq{}_rept{}_nSymbols{}_Ts{}ms.html".format(nSequencer, nSequencias, nSymbols, int(Ts*1000))
    plotly.offline.plot(fig, filename = "graficos/" + figname, auto_open=False)
    fig.show(renderer="png")
    print("GRAFICO INTERATIVO CORRESPONDENTE: {}".format(figname))

