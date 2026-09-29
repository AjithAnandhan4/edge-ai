# Tap-Rhythm Classifier: edge AI on FRDM-MCXN236

Tap **SW2** six times. A tiny int8 neural network running on the Cortex-M33
(TensorFlow Lite Micro + CMSIS-NN kernels) classifies the rhythm, and the RGB LED
shows the result.

| Class       | How to tap                         | LED     |
|-------------|------------------------------------|---------|
| `steady`    | all gaps equal: `x . x . x . x`    | green   |
| `gallop`    | short-long-short-long: `xx . xx .` | blue    |
| `speed_up`  | each gap shorter (accelerando)     | red     |
| `slow_down` | each gap longer (ritardando)       | magenta |
| `irregular` | anything else (random gaps)        | white   |

`irregular` is a "none of the above" class. A classifier without it confidently
forces random tapping into one of the four rhythms (we saw `steady 1.00` for
gaps of 637/171/167/198/689 ms on the real board). Out-of-distribution inputs
are a classic edge-AI pitfall.

**SW3** cancels a half-finished sequence. A pause longer than 3 s also resets it.

## The edge-AI pipeline you are learning

```
 train/train.py (PC)                                src/main.cpp (MCU)
 ───────────────────                                ───────────────────
 1. synthetic data  ─┐                          ┌─► 6. SW2 ISR records tap times
 2. features         │  gap_i / mean(gaps)  ◄───┼── 7. same features on device
 3. Keras MLP 5-16-16-5                         │   8. quantize  q = x/scale + zp
 4. int8 quantization (representative data)     │   9. interpreter->Invoke()
 5. model_data.cc  (C array, 16-byte aligned) ──┘  10. dequantize, argmax -> LED
```

Things worth noticing:

- **Feature engineering beats model size.** Dividing the gaps by their mean makes
  the features tempo-invariant. The network only has to learn the *shape* of the
  rhythm, so ~450 parameters are enough.
- **Quantization.** Weights and activations are int8. Look at the
  `Input quantization: scale=… zero_point=…` line printed at boot, and at
  `classify()`, to see how float values are mapped to int8 and back.
- **Only link the ops you use.** `MicroMutableOpResolver<2>` registers just
  `FULLY_CONNECTED` and `SOFTMAX` (the list printed by the TFLite analyzer).
- **Tensor arena.** All tensors live in one static 4 KB buffer. The boot log
  prints how much of it is actually used.
- **No NPU on the MCXN236.** Inference runs on the CPU. CMSIS-NN kernels
  use the M33's DSP/SIMD instructions to speed it up (compare with Stage 2 in
  `../README.rst`).

## Build and flash

```console
source ~/zephyrproject/.venv/bin/activate
cd ~/zephyrproject
west build -b frdm_mcxn236 -p always -d edge-ai/tap_rhythm/build edge-ai/tap_rhythm
west flash -d edge-ai/tap_rhythm/build
screen /dev/ttyACM0 115200      # then press RESET on the board
```

On boot, a self-test classifies one clean example per class and should end with
`Self-test: 5/5 PASS`.

Measured on the board: ~65 us per inference, 1316 B of tensor arena, 92.7 KB
flash for the whole application. The int8 model reaches 97.3% on the synthetic
test set.

**Troubleshooting: no console output.** If the board runs (LEDs work) but
`/dev/ttyACM0` stays silent, the MCU-Link virtual COM port is stuck. Unplug and
replug the USB cable. `LinkServer probe '#1' wiretimedreset 100` resets the target
from the command line without pressing RESET.

## Retrain

```console
python edge-ai/tap_rhythm/train/train.py          # regenerates src/model_data.*
```

### Train on your own taps (exercise)

1. Hold **SW3** while pressing **RESET** to enter log mode.
2. Press **SW3** to choose the label you are about to tap (the LED shows the
   class colour), then tap SW2 six times. Repeat many times per class.
3. Copy the `CSV,…` lines from the console into `my_taps.csv`, without the `CSV,`
   prefix (`grep ^CSV log.txt | cut -d, -f2- > my_taps.csv`).
4. `python edge-ai/tap_rhythm/train/train.py --real my_taps.csv`, then rebuild and flash.

Ideas for going further: add another rhythm class, make the tap count
variable, or feed ADC samples from a sensor on the Arduino header instead of
button taps.
