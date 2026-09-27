| arch | stage | tok/s | vs dense 4K | step (s) | peak mem GB | TFLOPS | MFU % | power W | SM MHz | util % | backends |
|---|---|---|---|---|---|---|---|---|---|---|---|
| dense | trunk | 95,042 | 1.00x | 1.379 | 15.9 | 126.1 | 12.7 | 595.9 | 1980.0 | 99.7 | sdpa_mask |
| dsa | trunk | 30,391 | 0.32x | 4.313 | 31.2 | 34.0 | 3.4 | 504.0 | 1980.0 | 99.9 | sdpa_mask |
| kda_full | trunk | 71,351 | 0.75x | 1.837 | 23.0 | 83.4 | 8.4 | 527.2 | 1980.0 | 96.6 | sdpa_mask, kda=fla |
| kda_dsa | trunk | 50,855 | 0.54x | 2.577 | 26.8 | 56.8 | 5.7 | 506.1 | 1980.0 | 97.8 | sdpa_mask, kda=fla |
| dense | s2_16k | 48,049 | 0.51x | 2.728 | 7.2 | 96.4 | 9.7 | 665.8 | 1977.6 | 99.9 | sdpa_mask |
| dsa | s2_16k | 5,834 | 0.06x | 22.468 | 15.3 | 6.5 | 0.7 | 498.0 | 1980.0 | 100.0 | sdpa_mask |
| kda_full | s2_16k | 51,806 | 0.55x | 2.53 | 7.2 | 69.3 | 7.0 | 534.5 | 1980.0 | 96.4 | sdpa_mask, kda=fla |
