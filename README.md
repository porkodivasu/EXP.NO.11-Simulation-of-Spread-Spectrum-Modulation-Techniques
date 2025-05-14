EXP.NO.11-Simulation-of-Spread-Spectrum-Modulation-Techniques
11.Simulation of Spread Spectrum Modulation Techniques

AIM
To simulate the spread spectrum modulation techniques using Python programming.

SOFTWARE REQUIRED
Python Colab

ALGORITHMS
Initialize System Parameters

Calculate chip_rate = bit_rate × chips_per_bit

Calculate samples_per_chip = sample_rate / chip_rate

Generate Random Binary Data

Create a binary data array of length data_length with values 0 or 1

Generate PN Sequence

Create a pseudorandom noise (PN) sequence of length chips_per_bit with values ±1

Spread Data using DSSS

For each data bit:

Map bit 0 to -1 and bit 1 to +1 (BPSK mapping)

Multiply the mapped bit by the PN sequence (chip-wise)

Append the result to the spread_signal

Modulate Spread Signal onto Carrier

Repeat each chip in the spread signal samples_per_chip times to match sampling rate

Generate a cosine carrier wave with frequency carrier_freq

Multiply the chip samples with the carrier wave (BPSK modulation)

Plot Signals

Plot the DSSS spread signal (baseband chip values)

Plot the BPSK-modulated carrier waveform

Display the Plots

PROGRAM
import numpy as np

import matplotlib.pyplot as plt

data_length = 4

chips_per_bit = 8

bit_rate = 1e3

chip_rate = bit_rate * chips_per_bit

carrier_freq = 20e3

sample_rate = 160e3

samples_per_chip = int(sample_rate / chip_rate)

def generate_data(length):

return np.random.randint(0, 2, length)
def generate_pn_sequence(length):

return np.random.choice([-1, 1], length)
def bpsk_modulate(bit):

return 2 * bit - 1
def dsss_spread(data, pn_sequence):

spread = []

for bit in data:

    bpsk_bit = bpsk_modulate(bit)

    spread.extend(bpsk_bit * pn_sequence)

return np.array(spread)
def carrier_modulate(spread_signal, carrier_freq, sample_rate, samples_per_chip):

total_samples = len(spread_signal) * samples_per_chip

t = np.arange(total_samples) / sample_rate

carrier_wave = np.cos(2 * np.pi * carrier_freq * t)

chip_samples = np.repeat(spread_signal, samples_per_chip)

return chip_samples * carrier_wave, t
if name == "main":

data = generate_data(data_length)

pn_seq = generate_pn_sequence(chips_per_bit)

print("Original Data Bits:     ", data)

print("PN Sequence:            ", pn_seq)

spread_signal = dsss_spread(data, pn_seq)

bpsk_waveform, t = carrier_modulate(spread_signal, carrier_freq, sample_rate, samples_per_chip)

plt.figure(figsize=(12, 3))

plt.plot(spread_signal, drawstyle='steps-mid')

plt.title("DSSS Spread Signal (Baseband)")

plt.xlabel("Chip Index")

plt.ylabel("Amplitude")

plt.grid(True)

plt.tight_layout()

plt.figure(figsize=(12, 3))

plt.plot(t, bpsk_waveform)

plt.title("BPSK Modulated Waveform (Carrier)")

plt.xlabel("Time (s)")

plt.ylabel("Amplitude")

plt.grid(True)

plt.tight_layout()

plt.show()
OUTPUT
![Screenshot (122)](https://github.com/user-attachments/assets/18c62cc1-1162-48e9-89a8-d3d8a950a671)


RESULT / CONCLUSIONS
Thus, simulation of the spread spectrum modulation techniques using Python programming was implemented successfully.
