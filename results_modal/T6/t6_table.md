Measured on NVIDIA A100-SXM4-40GB (capped at 31.7 GB) + SSIS on NVIDIA A10; sample 0.66 s/img, train 0.69 s/timestep, SSIS 0.055 s/img, c_train 10.5.

| config | images/epoch | s/image (sample + reward + train) | GPU-h (measured GPU) | GPU-h on V100 (×2–×3, rough) |
|---|---|---|---|---|
| flowgrpo_default_100ep | 1152 | 0.66 + 0.06 + 6.22 = 6.9 | 221.9 | 444–666 |
| flowgrpo_default_1000ep | 1152 | 0.66 + 0.06 + 6.22 = 6.9 | 2219.3 | 4439–6658 |
| cutdown_8x8_100ep | 64 | 0.66 + 0.06 + 6.22 = 6.9 | 12.3 | 25–37 |
| cutdown_8x8_300ep_tf0.5 | 64 | 0.66 + 0.06 + 3.46 = 4.2 | 22.2 | 44–67 |
| cutdown_16x8_200ep | 128 | 0.66 + 0.06 + 6.22 = 6.9 | 49.3 | 99–148 |
