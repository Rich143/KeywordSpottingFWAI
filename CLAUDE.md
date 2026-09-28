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
- **Build**:
  ```
  cmake --preset Debug && cmake --build --preset Debug   # → build/Debug/B-U585I-IOT02A.elf
  ```
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
- **Test vectors**: generated by `HostSW/Preprocessing.ipynb` into `HostSW/test_vectors/`, then copied
  into `Test/support/`
- **On-target audio check**: flash, run `python HostSW/receive_and_play.py` (edit `SERIAL_PORT`), listen
- **Model validation**: STM32Cube AI Studio validate using `HostSW/TestDatasetST/*.npz`

## Host Software (`HostSW/`)
- `TrainModel.ipynb` (current, v7): Google Speech Commands v0.02 → features → CNN → int8 PTQ →
  `tflite_models/`. It computes features by calling the **same C preprocessing library** via ctypes
  (`HostSW/build/.../libAudioPreprocessing.dylib`, built from `HostSW/CMakeLists.txt`), so training
  features match the firmware.
- `receive_and_play.py`: UART audio receiver and playback (pyserial, sounddevice, numpy)
- No requirements.txt. Main deps: tensorflow, tensorflow-io, ai-edge-litert, optuna, numpy, pandas,
  scikit-learn, matplotlib, seaborn, soundfile, pyserial, sounddevice
- Uses conda env "kws"

## Known Issues & Workarounds
- `MX_ADF1_Init()` in `main.c` and `HAL_MDF_MspInit/DeInit` in `stm32u5xx_hal_msp.c` are commented
  out so they don't override the BSP audio config. **These edits are outside USER CODE blocks, so
  CubeMX regeneration will restore them.** Re-apply after regenerating.
- `MX_UARTx_Init` call in `app_x-cube-ai.c` / `bsp_ai.h` removed (duplicate USART init). Re-apply after AI regeneration.
- `Core/Src/audio_cb_rate.c` is an untracked experiment that duplicates symbols in `audio.c`. Don't add it to the build.

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
HostSW/                          # Training notebooks, tflite models, host lib build, UART receiver
cmake/, CMakeLists.txt, CMakePresets.json
B-U585I-IOT02A.ioc, project.stai, .ai/   # CubeMX + AI Studio projects
```

## Important Notes for AI Assistance
- This is a learning project. Explain the STM32U5 and embedded-ML reasoning when proposing changes.
- Check the BSP (`b_u585i_iot02a_audio.c`) before touching audio clocks, ADF/MDF or DMA. It owns that config.
- Any change to preprocessing must stay identical between firmware and the host training path.
  Re-run the Ceedling tests and regenerate test vectors if the parameters change.
- HAL functions return `HAL_StatusTypeDef` (`HAL_OK` = 0); BSP returns `BSP_ERROR_NONE` = 0.
