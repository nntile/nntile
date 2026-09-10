#!/usr/bin/env python3
"""Run modern U-Net HF(cuda) / HF(nntile) overhead ladder."""

from run_cnn_overhead_benchmark import main

if __name__ == "__main__":
    raise SystemExit(main(default_family="unet_modern"))
