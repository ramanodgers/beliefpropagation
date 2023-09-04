import numpy as np 
import gradio as gr
from BPtools import *

description = "This is a toy  syndrome-based belief propagation decoder with OSD post-processing. Pick an input number, error distance and encoding. \
Your number will convert to binary, then encode, then be corrupted by your error distance, then produce a syndrome used to decode the error. See the decoded result given your choice of parity-check matrix. \
The decoded result may not be the same as the encoded input string. It should just be equal up to a stabilizer.\n\n (It refreshes with every change to the input and might miss a change if the previous one is loading)"

file1  = 'Dv2Dc3_G18_N114.npy'
file2  = 'Dv2Dc6_G8_N36.npy'

#classical
CM1 = np.load('files/'+file1)
CM2 = np.load('files/'+file2)

#quantum
surface25 = np.array([[1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0],
                      [0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1]],dtype=int)

rep3 = np.array([[1,1,0],[0,1,1]])
HGPrep3  = HGP(rep3,rep3)
project_stabilizers = ['ZZZZIIIIIIIIIIIII','ZIZIZZIIIIIIIIIII',
                       'IIIIZZIIZZIIIIIII','IIIIIIZZIIZZIIIII',
                      'IIIIIIIIZZIIZZIII','IIIIIIIIIIZZIIZZI',
                       'IIIIIIIZIIIZIIIZZ','IIZZIZZIIZZIIZZII']
steane_stabs = ['ZZZZIII','IZIZIZZ','IIZZZZI']
pcm17 = convert(project_stabilizers, 17)
steane7 = convert(steane_stabs, 7)
HCHQ = np.kron(CM2,steane7)


def greet(number, dist, choice):
    
    matrices = {"Big CLDPC": CM1, "Small CLDPC" : CM2, "Surface 25" : surface25, "Steane 17": pcm17}
    #outputs encoded binary, corrupted binary, syndrome, the estimated error, the recovered message binary, and an int output(which need not be the same) for a success, 
    M = matrices[choice]
    H1 = M
    G1 = generator(M)
    
    try:
        number = int(number)
    except: 
        raise ValueError("not a number")
        
    #string
    binary = arr2string(format_input(decimalToBinary(number), H1.shape[1]-H1.shape[0]))
    
    #np array
    encbin = word_gen(G1, x=format_input(binary, G1.shape[1]))
    
    #np array 
    corrupted = dist_error(encbin, dist)
    
    syndrome = np.remainder(np.dot(H1, corrupted),2)
    
    BP = syndrome_BP(H1,syndrome,30,0.02,higher = True )
    
    #np array 
    est_error, _ = BP.decoder()
    
    #np array 
    recovered = np.remainder(corrupted-est_error,2)
    
    x = solver(G1,recovered)
    # int from bin
    out = int(arr2string(x),2 )

    return (
        arr2string(encbin),
        arr2string(corrupted),
        arr2string(syndrome),
        arr2string(est_error),
        arr2string(recovered),
        out)


demo = gr.Interface(
    fn=greet,
    inputs=["text", 
            gr.Slider(0, 10, step = 1, label = "Error Distance"), 
            gr.Radio(["Big CLDPC", "Small CLDPC","Surface 25","Steane 17"], label="Parity-Check matrix", info="What matrix should we use?", value = 'Big CLDPC')],
    outputs=[gr.Textbox(label="Encoded Binary"), gr.Textbox(label="Corrupted Binary"),
             gr.Textbox(label="Syndrome Binary"),gr.Textbox(label="Estimated Error After Syndrome-based Belief Propagation Decoding"),
             gr.Textbox(label="Recovered Message Binary"), gr.Number(label = "Int Output")],
    theme=gr.themes.Monochrome(),
    title="Quantum Syndrome-Based Belief Propagation decoding",
    description = description, 
    allow_flagging = 'never',
    live = True

)
demo.launch( share = False)