#!/usr/bin/bash
uv run poe p01
uv run poe p02
uv run poe p03
uv run poe p04
uv run poe p04_layout_tensor
uv run poe p05
uv run poe p05_layout_tensor
uv run poe p06
uv run poe p07
uv run poe p07_layout_tensor
uv run poe p08
uv run poe p08_layout_tensor
uv run poe p09
uv run poe p09_layout_tensor
uv run poe p10
uv run poe p10_layout_tensor
uv run poe p11 --block-boundary
uv run poe p11 --simple
uv run poe p12 --complete
uv run poe p12 --simple
uv run poe p13
uv run poe p14 --naive
uv run poe p14 --single-block
uv run poe p14 --tiled
uv run poe p15
uv run poe p16
uv run poe p16-test-kernels
