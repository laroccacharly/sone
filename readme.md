# sone

Zero-shot choice decisions on images: a small vision-language model scores each
question's options in one forward pass on llama.cpp, fast enough for a Raspberry Pi 5 CPU.

## Demo

[![sone demo](https://img.youtube.com/vi/Psu5gCObilg/maxresdefault.jpg)](https://youtu.be/Psu5gCObilg)

## Requirements

- Python 3.12 or newer and [uv](https://docs.astral.sh/uv/)
- A C++17 compiler (on Debian/Raspberry Pi OS: `sudo apt install build-essential`)

CMake and Ninja are fetched by the build when missing. Installing compiles llama.cpp
for the machine's CPU, which takes a few minutes on a Raspberry Pi.

## Installation

```sh
uv pip install "sone @ git+https://github.com/laroccacharly/sone"
```

## Notes

https://huggingface.co/convaiinnovations/laya
https://huggingface.co/Cloudflare/clef
https://huggingface.co/nvkudva/vega-onnx-q8
https://huggingface.co/LiquidAI/d1-3B
