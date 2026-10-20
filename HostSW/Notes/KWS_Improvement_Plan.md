I've analysed your capture, researched what other KWS projects do, and simulated a smoothing fix. Here's what I found and the plan I'd suggest.

## What `capture.csv` shows

- **Speech gets classified as keywords:** 245 of 621 windows (40%) have a keyword as their top prediction, and the label changes 166 times in 62 s. The model is unsure almost everywhere: the median top probability is only 0.45.
- **Long silences are mostly handled correctly:** the 10–20 s and 52–62 s pauses come out as `_background_noise_`. What fails is the short gaps between words, where most of each 1 s window is still speech.

I think there are four causes, roughly in order of impact:

1. **The model has never seen continuous speech or partial words.** Every training example is one isolated word, roughly centred, surrounded by near-silence. In your speech, a window holding the end of one word and the start of another looks more like a keyword than anything it saw labelled `other`. Also, `other` is cut down to 20% of the data, so the model rarely sees non-keyword speech.
2. **There's no augmentation.** `augment_clip()` in the notebook does nothing, so there's no time shift, no mixed-in noise and no gain variation.
3. **The background class is mostly synthetic noise.** 1,140 of the 1,478 clips are blue, brown or violet noise at about −14 dBFS. There's no quiet room tone, no background chatter or TV, and no music.
4. **The board mic sounds different from the dataset:**
   - Its noise floor during pauses is about −45 dBFS, against about −65 dBFS in Speech Commands clips.
   - Your speech peaks about 8 dB quieter than the dataset (about −24 vs −15 dBFS).
   - There's a large DC offset that drifts from −0.33 to −0.07 of full scale. That's because the BSP leaves the ADF high-pass filter off (`HighPassFilter.Activation = DISABLE`, `b_u585i_iot02a_audio.c:427`).

   DC sits below the 125 Hz lowest mel filter, so it only slightly affects the features. But it uses up headroom: 2 samples hit full scale and 22 went past 0.9, all on the negative side.

**A correction to what I said earlier:** I told you there was no normalisation. There is: it happens inside the C library, which subtracts `LOG_MEL_MEAN` and divides by `LOG_MEL_STDDEV`. All the scripts call the library, so nothing changes.

**Smoothing alone helps, but isn't enough.** I applied TF Micro-style post-processing to your capture: average the outputs, require a threshold, then ignore new detections for 1.5 s. With 0.5 s averaging and a 0.7 threshold, 245 keyword windows reduce to 12 detections:

> 10.0 forward, 17.9 stop, 23.7 forward, 27.9 backward, 31.5 stop, 34.2 forward, 39.3 stop, 41.9 forward, 45.2 forward, 48.1 forward, 51.3 backward, 54.1 forward

**Could you tell me which of these were keywords you actually said?** That gives us a first measured baseline. Even unsmoothed, the model reaches 0.9+ on some windows that are probably word fragments, so smoothing can't fix those.

## What others do

- **Google's Speech Commands training script** ([train.py](https://raw.githubusercontent.com/tensorflow/tensorflow/v2.4.0/tensorflow/examples/speech_commands/train.py)) mixes background noise into 80% of samples at up to 0.1 volume. It shifts samples randomly by up to ±100 ms and adds 10% silence and 10% unknown words. Your pipeline currently has none of these.
- **Wake-word projects built for streaming audio** ([microWakeWord](https://docsearch.algolia.com/mcp/docs/repo/ohf-voice/micro-wake-word), [openWakeWord](https://github.com/alfiedennen/openwakeword-colab-2026)):
  - They train on very large amounts of negative audio, about 2,000 hours of general audio in openWakeWord's case.
  - They use roughly 10–20 negative windows per positive.
  - Augmentation includes room reverb (MIT room impulse responses), noise and music from MUSAN, FSD50K and WHAM!, and pitch and speed changes.
  - They add confusable words as extra negatives, and often use positive samples generated with Piper text-to-speech.
  - They measure **false accepts per hour on long audio**, not per-clip accuracy.
  - The reason that metric matters: with 10 windows per second, even a 5% per-window false-positive rate is [about 1,800 false alarms an hour](https://www.ioa.org.uk/system/files/publications/MJ%20CAREY%20ES%20PARRIS%20IMPROVEMENTS%20TO%20A%20KEYWORD%20RECOGNITION%20ALGORITHM.pdf).
- **TF Micro's `RecognizeCommands`** ([source](https://raw.githubusercontent.com/tensorflow/tensorflow/v2.4.0/tensorflow/lite/micro/examples/micro_speech/recognize_commands.h)) averages outputs over 1000 ms, needs a score of 200/255 (about 0.78), suppresses repeats for 1500 ms and needs at least 3 results in the average.
- **Datasets:**
  - For continuous speech: [LibriSpeech](https://arxiv.org/pdf/2505.24820) (960 h).
  - For many more isolated words: the [Multilingual Spoken Words Corpus](https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/hash/fe131d7f5a6b38b23cc967316c13dae2-Abstract-round2.html) (23 M one-second words).
  - For noise: MUSAN, FSD50K or ESC-50.
- **Edge Impulse** [recommends](https://docs.edgeimpulse.com/docs/responding-to-your-voice) recording on the target device itself, with at least 10 minutes per class including noise and "unknown".

## Is it worth recording your own data on the board?

Yes. Minute for minute, board recordings are your most valuable data, because they're the only data with your mic's noise floor, level and DC behaviour. In priority order:

1. **Negatives: the cheapest and most useful.** Use `receive_and_play.py -o` and record long stretches of yourself talking or reading aloud, TV and podcasts, music, typing, kitchen noise and plain room tone. No labelling is needed, since every 1 s window is `other` or `_background_noise_`. Aim for hours, not minutes.
2. **Keywords in context.** Say sentences like "go forward a bit" and "please stop", as well as the keywords alone. Vary distance (0.3–2 m), loudness and speed. These need timestamps; an Audacity label track exported to the `_truth.csv` format works, and then `view_results.py` shows it as an overlay.
3. **Other speakers if possible.** Otherwise the model risks learning your voice; Speech Commands still provides speaker variety.
4. **A held-out test set.** Keep some recording sessions out of training entirely and use them for evaluation.

## Suggested plan, cheapest first

0. **Measure first.** Record a labelled board evaluation set of about 5 minutes: speech, occasional keywords and pauses. Then add a `score_results.py` that reports keyword recall and false accepts per minute after smoothing. Every later step is judged by these two numbers, not per-clip accuracy.
1. **Add smoothing, a threshold and suppression to the host scripts**, and tune them on the evaluation set. The firmware will need the same logic later anyway.
2. **Add augmentation and better negatives in the notebook. I expect this to be the biggest improvement.**
   - Time shift of ±100–200 ms.
   - **Partial-keyword negatives:** windows where only a fragment of a keyword is inside, labelled `other`. This targets the label flipping you're seeing directly.
   - 1 s crops of LibriSpeech, skipping utterances whose transcript contains a keyword.
   - Use far more of Speech Commands' `other` words, especially confusable ones like `follow` and `four`. Weight the classes so keyword recall doesn't collapse.
   - Mix real noise into about 80% of samples at 0–20 dB SNR.
   - Random gain of ±12 dB.
   - Rebuild `_background_noise_` around real noise and quiet room tone, with less synthetic noise.
   - Optionally, room reverb and SpecAugment.

   All of this happens on the waveform before the C preprocessing, so the ctypes pipeline stays as it is. To speed it up, run it across several processes; each process loads its own copy of the library, so its internal state isn't shared.
3. **Add the board recordings** from the previous section to training.
4. **Remove the DC offset.** Subtracting each frame's mean inside the C library keeps firmware and training identical. You'll be retraining anyway; it does mean regenerating the test vectors and re-running Ceedling.
5. **Then the model.** Find where the retrained model gives false keywords on hours of negative audio, add those windows to training, and repeat. If the harder data exceeds what the small 16/24/24 CNN can learn, try a slightly larger model or a DS-CNN, the architecture from ARM's [Hello Edge](https://arxiv.org/abs/1711.07128v1) paper.

LibriSpeech, MUSAN and similar datasets are separate downloads, from hundreds of MB to several GB, so I'd check with you before adding any of them. A good starting point would be steps 0 and 1 (the scoring script and smoothing), then the time-shift, noise and partial-keyword augmentation in the notebook.