# Kaggle CPU Execution

This repository includes a Kaggle-ready notebook at:

```text
kaggle/s3forecaster_s3fastsketch_m4_cpu.ipynb
```

The matching metadata file is:

```text
kaggle/kernel-metadata.json
```

`kernel-metadata.json` is generated with the configured Kaggle username, defaulting to `alejopatio` in this workspace. Override it by setting `KAGGLE_USERNAME` before running `tools/make_m4_kaggle_notebook.py`.

## CPU Settings

The notebook and metadata are configured for CPU execution:

- `enable_gpu`: `false`;
- notebook metadata `accelerator`: `none`;
- notebook setup cell sets `CUDA_VISIBLE_DEVICES=""`.

Internet is enabled so the notebook can read the official M4 Monthly CSVs from `Mcompetitions/M4-methods`.

## Source Availability

The notebook imports `s3paper`, so Kaggle must be able to see the repository source. The generated notebook includes an embedded zipped copy of `s3paper/` as a fallback. The metadata also references the private source dataset:

```text
alejopatio/s3forecaster-s3fastsketch-source
```

For future maintenance, use one of these approaches:

1. Attach this repository as a Kaggle input dataset containing the `s3paper/` folder.
2. Run the notebook from an environment where the repository has been checked out.
3. Publish the repository remotely and install it in a setup cell before the package imports.

## CLI Commands

After configuring Kaggle credentials and metadata:

```bash
kaggle kernels push -p kaggle
kaggle kernels status alejopatio/s3forecaster-s3fastsketch-m4-cpu
```

Kaggle credentials are expected in the standard Kaggle CLI location or environment variables. Without those credentials, the notebook files can be prepared locally but cannot be submitted from this machine.

Current successful CPU run:

```text
https://www.kaggle.com/code/alejopatio/s3forecaster-s3fastsketch-m4-cpu
```
