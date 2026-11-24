# Keyword Spotting Embedded ML Project for STM32 B-U585I-IOT02A Discovery Kit

## Project Overview
Personal learning/portfolio project: on-device keyword spotting (KWS) on an STM32U5.
Pipeline goal: digital mic → 16 kHz PCM → log-mel spectrogram (30×45) → int8 CNN (ST Edge AI) → class.
Classes (model output order): `forward`, `backward`, `stop`, `_background_noise_`, `other`.

**Current phase: audio capture.** Firmware records the on-board mic and streams raw PCM over
UART to the host (`HostSW/receive_and_play.py`) to verify audio quality. Preprocessing is built
and host-tested but not yet called on target; inference is initialised but not reached.

## Hardware Platform
- **Board**: B-U585I-IOT02A Discovery kit (on-board STLINK-V3E, USB Virtual COM port)
- **MCU**: STM32U585AII6Q: Cortex-M33 @ 160 MHz, FPU (fpv5-sp-d16), TrustZone disabled
- **Memory**: 2 MB internal flash, 768 KB SRAM (+16 KB SRAM4); 64 MB octo-SPI flash (MX25LM51245G, unused)
- **Clock**: MSI 4 MHz → PLL1 (M=1, N=80, R=2) = 160 MHz SYSCLK; VOS1, flash latency 4
- **Microphones**: 2× MP23DB01HP digital MEMS. MIC1 → ADF1_Filter0 / GPDMA1_Channel6 (in use);
  MIC2 → MDF1_Filter0 / GPDMA1_Channel0
- **UART**: USART1 (PA9 TX / PA10 RX → ST-Link VCP), 921600 8N1, polling TX
- **GPIO**: LED_RED PH6, LED_GREEN PH7, USER button PC13 (configured, unused)

## Architecture
- **Bare metal**, no RTOS: main-loop super-loop; DMA/ISR callbacks only set flags
- **Audio input is owned by the BSP** (`b_u585i_iot02a_audio.c` configures clocks, ADF/MDF, pins, DMA).
  CubeMX's MDF/ADF init is intentionally disabled (see Known Issues).
- **Audio capture** (`Core/Src/audio.c`): circular ping-pong DMA, streamed over UART
- **UART stream protocol**: custom framed binary protocol shared with `HostSW/receive_and_play.py`.
  `printf` shares the same UART; the host resyncs on magic numbers.
- **Preprocessing** (`Lib/AudioPreprocessing`): log-mel spectrogram built on ST's audio preprocessing
  library + CMSIS-DSP. Parameters must match the training pipeline.
- **Inference** (`AI/App/`): ST Edge AI generated int8 CNN (`HostSW/tflite_models/kws_v7_int8_softmax.tflite`).
  App hooks in `app_x-cube-ai.c` are still stubs. The float spectrogram must be quantised with the model's
  input scale/zp and its layout checked against training.

## Performance Targets
- **Latency goal**: preprocessing + inference ≤ ~100 ms per 1 s window, so it can run up to 10×/s
  (sliding window). This is exploratory; measure it with DWT cycle counts before optimising.
- Audio capture must never drop DMA half-buffers. ISR work must finish well within one half-buffer period.
- No power budget: the board is USB-powered.

## Build System & Toolchain
- **Toolchain**: STM32Cube for VS Code bundles: GNU Tools for STM32 14.3.1 (`arm-none-eabi-gcc`),
  CMake 4.3 + Ninja, st-arm-clangd, ST-LINK GDB server, STM32CubeProgrammer 2.23
- **Flags**: `-mcpu=cortex-m33 -mfpu=fpv5-sp-d16 -mfloat-abi=hard`, C11, `nano.specs`, `--gc-sections`
- **Configs**: Debug `-O0 -g3`, Release `-Os -g0` (alternative clang toolchain in `cmake/starm-clang.cmake`, unused)
- **Build**: always use `cube-cmake` (the STM32Cube extension's wrapper; it puts the ST bundle's
  gcc 14.3.1 and Ninja on PATH). Plain `cmake` from a terminal fails: Ninja isn't on PATH and
  `/Applications/ArmGNUToolchain/Latest` (Arm GNU 12.2) shadows the ST toolchain.
  ```
  CUBE_CMAKE=$(ls ~/.vscode/extensions/stmicroelectronics.stm32cube-ide-build-cmake-*/resources/cube-cmake/darwin/aarch64/cube-cmake | tail -1)
  $CUBE_CMAKE --preset Debug -DCMAKE_COMMAND=cube-cmake && $CUBE_CMAKE --build build/Debug   # → build/Debug/B-U585I-IOT02A.elf
  ```
- The toolchain file sets `CMAKE_AR`/`CMAKE_RANLIB` explicitly. Without them CMake can pick macOS
  `/usr/bin/ar`, whose archive index GNU ld can't read → "undefined reference" link errors for BSP/library symbols.
- **Adding source files**: user sources go in the root `CMakeLists.txt`
  (`target_sources`). `cmake/stm32cubemx/CMakeLists.txt` is CubeMX-generated.
- **Flash/debug**: via the STM32Cube VS Code extension (ST-Link)
- **Code generation**: `B-U585I-IOT02A.ioc` (STM32CubeMX) and `project.stai` / `.ai/` (STM32Cube AI Studio)

## Coding Standards
- **Language**: C11 (firmware); Python 3 (host)
- **Match the style of the file you're editing.** There are two styles in the repo:
  - ST/CubeMX style (`Core/`, `AI/`, `audio.c`): 2-space indent, Allman braces, `/* */` comments,
    Doxygen `@brief/@param/@retval`, `#ifndef` guards. User functions and globals are PascalCase
    (`SendChunk`, `RecordBuff`), types `Name_t`, macros `ALL_CAPS` with `U` suffix.
  - Library style (`Lib/AudioPreprocessing`): 4-space indent, K&R braces, `module_prefix_snake_case`,
    `MODULE_STATUS_*` enums, `#pragma once`
- Mark ISR-shared flags `volatile`
- Doxygen comments on public functions

## Critical Don'ts
- **NEVER** edit CubeMX-generated files outside `/* USER CODE BEGIN */ … /* USER CODE END */` blocks.
  Never hand-edit generated `AI/App/network*.c` or `Middlewares/`. Regenerate them instead.
- **NEVER** use `malloc()`/`free()` or other dynamic allocation. Use static buffers only.
- **NEVER** block or do heavy work in ISRs or DMA callbacks (no UART TX, no `HAL_Delay`,
  no preprocessing/inference). Set flags and do the work in the main loop.
- **NEVER** commit, or regenerate CubeMX / ST Edge AI code, without asking first.
- Don't change the UART protocol on one side only. Firmware and `receive_and_play.py` must match.

## Allowed Libraries
- STM32U5 HAL + CMSIS (`Drivers/`), B-U585I-IOT02A BSP v1.4.2 and stm32-bsp-common v7.3.0 (submodules)
- CMSIS-DSP v1.18 (submodule, `Middlewares/ARM/CMSIS-DSP`)
- ST AI runtime (`Middlewares/ST/AI`, `NetworkRuntime1201_CM33_GCC.a`) and the STM32 AI Audio Preprocessing Library
- Ask before adding anything else

## Memory Budget
- Flash: 2 MB. Target being able to deploy on smaller devices, would like to use much less than the available 2 MB
- RAM: 768 KB. Target being able to deploy on smaller devices, would like to use much less than the available 768
- Heap 2 KB (effectively unused), stack 8 KB

## Testing Strategy
- **Host unit tests**: Ceedling 1.0.1 + Unity (clang) in `Test/`. Run `cd Test && ceedling test:all`
  (coverage: `ceedling gcov:all`). Currently, they test CMSIS-DSP RFFT and the preprocessing pipeline against
  reference vectors (tolerance 1e-4). Add more tests as project progresses
- **Test vectors**: generated by `HostSW/Notebooks/Preprocessing.ipynb` into `HostSW/test_vectors/`, then copied
  into `Test/support/`
- **On-target audio check**: flash, run `python HostSW/receive_and_play.py <serial-port>`
  (e.g. `/dev/tty.usbmodem2021303`), listen
- **Model validation**: STM32Cube AI Studio validate using `HostSW/TestDatasetST/*.npz`
- **Model on long audio** (host, sliding window, scripts in `HostSW/EvalScripts/`): `make_test_wav.py -o t.wav` (or a board capture via
  `receive_and_play.py -o`) → `run_model_on_wav.py -m tflite_models/<model>.tflite -w t.wav` →
  `view_results.py -w t.wav`
- **Eval splits**: both eval sets have `val/` (compare models, make decisions; scored every training run) and
  `test/` (held back: only scored when asked for by name, `--split test` / `RUN_TEST_EVAL = True` in the notebook;
  look rarely, never tune on it). Never suggest tuning on, or routinely reporting, test results
- **Board eval set** (`HostSW/eval_recordings/{val,test}/`, held out, never train on it; see its README; test is
  empty until new recordings are made): record with
  `receive_and_play.py -o`, label keywords (+ `ignore`) in Audacity → `audacity_labels_to_truth.py` →
  `run_model_on_wav.py` → `score_results.py --sweep` (keyword recall, false accepts/min) → `view_results.py`.
  `eval_wav.py -w <wav>` runs those last three steps in one go
- **Public eval set** (`HostSW/eval_recordings_public/`, see its README): LibriSpeech, AMI and Spoken Wikipedia pieces with keyword
  spans from word alignments, same WAV + `_truth.csv` format, hundreds of speakers, plus about as many hours of speech-free
  background noise (MUSAN instrumental music + noise, DEMAND, Sounds of Home; empty truth files, kept at their recorded
  level), and 1 s keyword / `other` / `_background_noise_` clips. Built by `make_public_eval.py` from `dataset_public/` and
  `dataset_noise/`; WAVs are gitignored (rebuild, deterministic). Speakers and noise groups (artist, recorder, environment,
  home) are split 50/25/25 train/val/test in `splits.csv`: only `split = train` groups may ever be trained on. Noise groups
  keep their split once assigned
- **Scoring** (`snapshot_eval.py -v <ver> -m <model> --dataset board|public --split val|test --jobs 6`): sliding-window
  recall + false accepts/hour over a split → one summary row in `<split dir>/eval_history.csv` (board: one per run;
  public: one per model version, with per-source columns and false accepts/hour on speech and on noise apart:
  `fa_per_hour_speech` / `_noise`, also at each budget) with `mlflow_run_id` + `run_type`. From the notebook
  (eval section, `mlflow_run_id=run.info.run_id`) metrics like `board_val_recall` go into the training run
  (`run_type = training`); as a script it creates a `run_type = eval` MLflow run nested under the version's training run.
  Rows also have recall at false accept budgets (`recall_at_300fa`/`100`/`30`; `FA_BUDGETS` = FA/hour for the device, all
  keywords together): the main model comparison. One threshold shared by all keywords, chosen per budget on val only;
  test reuses it (score val first). Add budgets rather than replacing them: each has history columns.
  Every snapshot also logs the split's whole history as MLflow tables (`reports/<dataset>_<split>_history/`: default
  threshold, by source, one per budget, run settings)

## Host Software (`HostSW/`)
- Notebooks are in `Notebooks/`, eval scripts in `EvalScripts/`. Both use paths relative to `HostSW/`:
  the notebooks' first cell `chdir`s there; run the scripts from `HostSW/` (`python EvalScripts/<script>.py`).
- `TrainModel.ipynb` (current, v7): Google Speech Commands v0.02 → features → CNN → int8 PTQ →
  `tflite_models/`. It computes features by calling the **same C preprocessing library** via ctypes
  (`HostSW/build/.../libAudioPreprocessing.dylib`, built from `HostSW/CMakeLists.txt`), so training
  features match the firmware. `MODEL_VERSION` (first cell) names the checkpoint/log dirs, tflite files
  and MLflow run. "Run All" does a full run (train → export → board eval → MLflow); the Optuna
  Optimize cells only run with `RUN_OPTIMIZE = True`.
- MLflow tracking: the notebook logs one run per training (params, dataset manifest, metrics, models, plots)
  to `HostSW/mlflow/` (SQLite + artifacts, Git LFS, see its README). Logging cells call `require_mlflow_run()`
  so they fail instead of silently starting a new run. UI: `cd HostSW && mlflow ui --backend-store-uri sqlite:///mlflow/mlflow.db`
- Datasets: `dataset/` = Speech Commands (downloaded, gitignored; `_background_noise_split_/` is derived
  and rebuilt by the notebook). `dataset_board/` = our board-mic training recordings (Git LFS, see its
  README); its `_background_noise_/` long recordings are split into the background class. `dataset_public/` =
  LibriSpeech, its MFA alignments, AMI and Spoken Wikipedia (gitignored), only used for the public eval set so far.
  `dataset_noise/` = background noise from MUSAN (noise + music, vocals flagged), DEMAND and Sounds of Home (gitignored,
  `download_noise.py`; `manifest.csv` with a split `group` per file: artist / recorder / file / environment / home), and
  where its speech is (`speech.npz`, `noise_speech.py`). Used by the public eval set; not in training yet (use only
  `split = train` groups, speech-free audio). Synthetic coloured noise is capped at `SYNTHETIC_NOISE_SECONDS` per colour
- `receive_and_play.py`: UART audio receiver and playback (pyserial, sounddevice, numpy)
- `EvalScripts/`:
  - `run_model_on_wav.py`: sliding 1 s window over a WAV → C preprocessing dylib → quantise → tflite →
    per-window CSV. Its `AudioPreprocessor` is a copy of the notebook's; keep them in sync
  - `make_test_wav.py`: stitches random Speech Commands clips into a WAV + `<stem>_truth.csv`
  - `view_results.py`: plays the WAV with the model output animated in sync (matplotlib, sounddevice)
  - `audacity_labels_to_truth.py`: Audacity label export ↔ `<stem>_truth.csv`
  - `score_results.py`: smoothing + threshold + suppression (`detect()`, or `keyword_curve()` for a threshold sweep) over a results CSV, matched
    against truth → recall and false accepts/min, `<stem>_detections.csv` for the viewer
  - `eval_wav.py`: run model → score → view on one WAV
  - `snapshot_eval.py`: scores a model on one split of the board or public eval set → that split's `eval_history.csv` + MLflow
  - `eval_tracking.py`: MLflow store / experiment setup shared with the notebook, standalone eval runs
  - `make_public_eval.py`: splits speakers / noise groups and builds `eval_recordings_public/{val,test}/` from
    `dataset_public/` and `dataset_noise/` (`--stats` for keyword counts and usable noise hours, `--check` to verify a build)
  - `download_ami.py`: resumable AMI audio download from the AMI site's wget script (Mix-Headset only by default;
    `--channels` picks other recordings). Its downloader is shared, and can fetch a byte range (one member of a remote zip)
  - `download_noise.py`: resumable download of the non-speech noise in `dataset_noise/` (`--dry-run` for sizes)
  - `noise_speech.py`: finds speech in `dataset_noise/` → `speech.npz`: the datasets' own labels (Sounds of Home's PANNs
    labels alone, MUSAN's vocals flag) and YAMNet (MUSAN; DEMAND confident speech only). Re-run after downloads
  - `check_keyword_clips.py`: flags partial / wrong Speech Commands keyword clips (Whisper large-v3-turbo on the
    Apple GPU via mlx-whisper, plus clip-edge / length / level energy checks) → `dataset/keyword_check.csv` and a
    listening file `dataset/keyword_check_spotcheck.wav` (+ Audacity labels). Resumable; `--report-only`
  - `review_keyword_clips.py`: terminal tool to hear each flagged clip and keep (y) / remove (n) it, likeliest cuts
    first (shuffled within a tier). Pause / resume, report on pause; hand labels → `keyword_clip_review.csv`
    (committed). `--report`, `--unsure`, `--sample N` (each tier's first N: a fair sample). `rule_decision()` (fitted to
    the labels, leans to removing) decides unreviewed clips (keep / remove / check: kept only once heard); `--rule-keeps` reviews the checks, then the keeps.
    `--unflagged [DB]` reviews unflagged forward / backward clips just past `cut_end`'s limit (saved as tier 0);
    `--unflagged-groups [N]` samples two more unflagged groups (files under 1 s; stop clips past the limit), also tier 0.
    `--public [N]` reviews public val keyword clips (up to N random per source + edge cases: is the whole word in the
    clip?) → `public_clip_review.csv`
- Main deps: tensorflow, tensorflow-io, ai-edge-litert, optuna, mlflow, numpy, pandas,
  scikit-learn, matplotlib, seaborn, soundfile, pyserial, sounddevice, mlx-whisper (pulls in torch)
- Uses conda env "kws", pinned in `HostSW/environment.yml` (Python 3.10 from conda, all else pip).
  `HostSW/setup_env.sh` creates it, builds the preprocessing dylib and smoke-tests imports.
  If you add packages, regenerate the pip list from `pip freeze` (drop `pip @ file://...` lines)

## Known Issues & Workarounds
- `MX_ADF1_Init()` in `main.c` and `HAL_MDF_MspInit/DeInit` in `stm32u5xx_hal_msp.c` are commented
  out so they don't override the BSP audio config. **These edits are outside USER CODE blocks, so
  CubeMX regeneration will restore them.** Re-apply after regenerating.
- `MX_UARTx_Init` call in `app_x-cube-ai.c` / `bsp_ai.h` removed (duplicate USART init). Re-apply after AI regeneration.

## File Structure
```
Core/Src/main.c, audio.c         # App entry + audio capture/UART stream (user code)
Core/Src/stm32u5xx_it.c          # IRQs (GPDMA1 Ch0/Ch6 → BSP audio handlers)
Core/Inc/b_u585i_iot02a_conf.h   # BSP config
AI/App/                          # ST Edge AI generated network + app_x-cube-ai.c glue
Lib/AudioPreprocessing/          # Log-mel spectrogram wrapper (our code)
Drivers/                         # HAL, CMSIS, BSP submodules
Middlewares/                     # CMSIS-DSP (submodule), ST AI runtime, ST audio preproc lib
Test/                            # Ceedling host tests + test vectors
HostSW/                          # tflite models, host lib build, UART receiver, datasets
HostSW/Notebooks/                # Training + preprocessing notebooks
HostSW/EvalScripts/              # Model-on-WAV eval, scoring, viewer
cmake/, CMakeLists.txt, CMakePresets.json
B-U585I-IOT02A.ioc, project.stai, .ai/   # CubeMX + AI Studio projects
```

## Important Notes for AI Assistance
- This is a learning project. Explain the STM32U5 and embedded-ML reasoning when proposing changes.
- Check the BSP (`b_u585i_iot02a_audio.c`) before touching audio clocks, ADF/MDF or DMA. It owns that config.
- Any change to preprocessing must stay identical between firmware and the host training path.
  Re-run the Ceedling tests and regenerate test vectors if the parameters change.
- HAL functions return `HAL_StatusTypeDef` (`HAL_OK` = 0); BSP returns `BSP_ERROR_NONE` = 0.
