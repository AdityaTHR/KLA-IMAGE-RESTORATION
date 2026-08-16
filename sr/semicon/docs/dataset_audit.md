# KLA PS01 Dataset Audit

**Audit date:** 2026-08-12
**Method:** Non-destructive archive inspection with selective NumPy loading

## Archive inventory

| Property | `train.zip` | `Test_NoisyLR.zip` |
|---|---:|---:|
| Compressed size | 918,994,209 bytes | 23,419,125 bytes |
| Uncompressed size | 1,050,449,253 bytes | 26,330,963 bytes |
| Data arrays | 6,400 | 400 |
| GT arrays | 3,200 | 0 |
| NoisyLR arrays | 3,200 | 400 |

The archives also contain macOS resource-fork metadata. `scripts/prepare_data.py`
filters `__MACOSX` entries and `.DS_Store` during extraction.

## Dataset contract

| Property | Training NoisyLR | Training GT | Blind-test NoisyLR |
|---|---|---|---|
| Count | 3,200 | 3,200 | 400 |
| Shape | `(128, 128)` | `(256, 256)` | `(128, 128)` |
| dtype | `float32` | `float32` | `float32` |
| Observed minimum | -0.080085 | 0.000000 | -0.029556 |
| Observed maximum | 1.722575 | 1.000000 | 1.540614 |
| GT available | Yes | Yes | No |

All arrays are single-channel images stored without an explicit channel axis. The
loader adds that axis and pairs training inputs and targets by identical filename.
Training filenames are contiguous from `000000.npy` through `003199.npy`; blind-test
filenames run from `000000.npy` through `000399.npy`.

Training and blind-test filenames overlap numerically but belong to separate
namespaces and contain different images. Full paths must therefore remain distinct.

## Input range

NoisyLR values outside `[0,1]` are valid observations, not data errors. The audit
found values below zero and above one in both training and blind-test inputs. These
values carry information about degradation magnitude and must not be clipped or
normalized into `[0,1]` before the model.

GT arrays are finite and bounded in `[0,1]`. Model outputs are constrained to this
range at the final output mapping and are validated again before saving.

## Degradation characteristics

Residual analysis against bicubic-downsampled GT showed signal-dependent noise:

| Sample | Residual std | Corr(signal, abs residual) | Bright/dark noise ratio |
|---|---:|---:|---:|
| `000000.npy` | 0.0512 | 0.671 | 7.52 |
| `000160.npy` | 0.0543 | 0.498 | 3.72 |
| `000320.npy` | 0.0948 | 0.272 | 3.43 |
| `000640.npy` | 0.1043 | 0.433 | 3.21 |
| `001760.npy` | 0.1054 | 0.583 | 5.53 |
| `002560.npy` | 0.1068 | 0.499 | 4.92 |

The positive correlation and larger bright-region variance are consistent with a
speckle-like component. Residual standard deviation also varies substantially across
images, supporting the mixed-severity degradation experiments in later phases.

## Canonical split

The seed-42 manifests under `data/splits/` are immutable:

| Manifest | Count | Role |
|---|---:|---|
| `train.txt` | 2,560 | Training only |
| `val.txt` | 320 | Checkpoint selection and development |
| `dev_test.txt` | 320 | Final internal comparison |

The manifests are unique, mutually disjoint, and exactly reproducible from the
sorted 3,200 paired filenames with Python's seed-42 shuffle. Blind-test data is not
included in any manifest and must not influence training or model selection.

## Storage

Extracted arrays require approximately 1.0 GB:

| Component | Approximate size |
|---|---:|
| Training GT | 800 MB |
| Training NoisyLR | 200 MB |
| Blind-test NoisyLR | 25 MB |

The project loads samples on demand, so the complete dataset does not need to remain
resident in memory.

## Verification

Run the current exhaustive data and split checks from the project root:

```bash
python scripts/prepare_data.py --seed 42
python scripts/audit_data.py
python scripts/validate_experiments.py
```

These checks validate counts, pairing, shapes, dtypes, finite values, GT bounds,
split reproducibility, split isolation, and preservation of raw NoisyLR values.
