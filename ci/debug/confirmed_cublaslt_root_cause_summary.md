# Confirmed cuBLASLt Root Cause Summary

## Problem

The active failure family is the CUDA `complex64` QR derivative path in
[tests/linalg_sharding_test.py](https://github.com/jax-ml/jax/blob/e2e6d0c0239df3ecc3b9acebaaf27e78e9f1af54/tests/linalg_sharding_test.py#L55-L210),
specifically the multi-device batch-axis sharding JVP/VJP coverage in
`LinalgShardingTest`.

## Confirming CI run

This was confirmed on the real CI path with a reduced current-head A/B run:

- workflow run:
  - https://github.com/jax-ml/jax/actions/runs/23915714662
- head baseline:
  - failing job:
    https://github.com/jax-ml/jax/actions/runs/23915714662/job/69748992617
- head with `XLA_FLAGS=--xla_gpu_enable_cublaslt=false`:
  - passing job:
    https://github.com/jax-ml/jax/actions/runs/23915714662/job/69748992717
- head with a source-level default flip back to `xla_gpu_enable_cublaslt=false`:
  - passing job:
    https://github.com/jax-ml/jax/actions/runs/23915714662/job/69748992506

This is the strongest confirmation point because it holds on the actual failing
CI workflow shape, not only in a local probe.

## Local probe that motivated the CI A/B

A focused local root-cause probe reproduced the relevant QR derivative shape on
a single GPU by comparing:

- the normal batch-`8` derivative computation
- a manual local split into `4 + 4`

That probe showed:

- head/nightly diverges under the split
- release `0.9.2` does not
- head/nightly becomes stable when `--xla_gpu_enable_cublaslt=false`
- release becomes unstable when `--xla_gpu_enable_cublaslt=true`

It also showed that the important backend difference is:

- unstable path: `__cublas$lt$matmul`
- stable path: `__cublas$gemm`

while the QR solver custom calls remain present. That made the right CI check
an A/B on cuBLASLt itself, not another broad historical bisect.

## Source-level explanation

Release `0.9.2` did not use stock XLA defaults for this path.

- JAX commit
  [e0bb88afc](https://github.com/jax-ml/jax/commit/e0bb88afcc93a4f950b5309828b19d475db11919)
  added XLA patch entries in
  [third_party/xla/workspace.bzl](https://github.com/jax-ml/jax/blob/e0bb88afcc93a4f950b5309828b19d475db11919/third_party/xla/workspace.bzl#L20-L29).
- That release branch carried
  [third_party/xla/xla_gpu_cublaslt_default.patch](https://github.com/jax-ml/jax/blob/e0bb88afcc93a4f950b5309828b19d475db11919/third_party/xla/xla_gpu_cublaslt_default.patch#L1-L15),
  which flips:
  - `opts.set_xla_gpu_enable_cublaslt(true);`
  - to `opts.set_xla_gpu_enable_cublaslt(false);`

Current JAX main does not carry that downstream override.

- Current
  [third_party/xla/workspace.bzl](https://github.com/jax-ml/jax/blob/e2e6d0c0239df3ecc3b9acebaaf27e78e9f1af54/third_party/xla/workspace.bzl#L19-L29)
  has an empty XLA `patch_file` list.

Upstream XLA currently defaults cuBLASLt on.

- Current
  [xla/debug_options_flags.cc](https://github.com/openxla/xla/blob/6e9c133d38ff253a4627ad62fbed0af677a067f3/xla/debug_options_flags.cc#L241-L248)
  sets `opts.set_xla_gpu_enable_cublaslt(true);`.
- XLA commit
  [ba4f5f2336](https://github.com/openxla/xla/commit/ba4f5f23363480275bcdeb4106f2e75c64ba3f91)
  explicitly re-enabled that default on March 23, 2026.

## Conclusion

The operative cause is the cuBLASLt default-on path for this QR derivative
failure family.

In practical terms:

- release `0.9.2` stays stable because JAX patched the XLA cuBLASLt default off
- head/nightly takes the cuBLASLt path because it no longer carries that patch
  and upstream XLA defaults it on
- disabling cuBLASLt on head, either by runtime flag or by source-level default
  flip, removes the flake on the real CI path

So this is no longer just a theory or a local-only explanation. The reduced CI
confirmation run demonstrates it directly.

## What remains open

What is still unresolved is narrower:

- not whether the cuBLASLt default-on path is the cause of the observed flake
- but what deeper numerical issue inside that cuBLASLt path makes these QR
  derivative matmuls unstable
