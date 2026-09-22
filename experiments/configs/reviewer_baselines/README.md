# Reviewer-requested traffic baselines

These configurations run STAEformer and PDFormer under the same protocol as
the TKDE revision experiments: input length 48, horizons 24/48/72/96, the
project's chronological splits and per-node standardization, three seeds, MSE,
50 epochs, and patience 7.

The models use micro-batches of 4 with 8-step gradient accumulation, giving an
effective batch size of 32. This is needed because spatial attention at 48
input steps is substantially more memory-intensive than the original 12-step
protocol.

PDFormer requires a dataset-specific asset file containing its DTW semantic
matrix, KShape pattern keys, shortest-hop matrix, and Laplacian positional
encoding. DTW and pattern keys are generated from the training period only.
Generate the assets before building/running PDFormer:

```bash
python -m experiments.preprocess_pdformer \
  --data_path datasets/PEMS08/data.dat \
  --adj_path datasets/PEMS08/adj_mx.pkl \
  --output_path datasets/PEMS08/pdformer_assets.npz \
  --steps_per_day 288 --train_size 0.6

python -m experiments.preprocess_pdformer \
  --data_path datasets/air_yrd/data.nc \
  --variable O3 \
  --adj_path datasets/air_yrd/adj_mx.pkl \
  --output_path datasets/air_yrd/pdformer_assets_O3.npz \
  --steps_per_day 24 --train_size 0.5
```

The PEMS08 comparison addresses the traffic-forecasting reviewer request.
YRD-O3 is an out-of-domain transfer test and must not be described as an
official PDFormer/STAEformer benchmark result.
