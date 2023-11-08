
---

# Sep-TFAnet <sup>VAD</sup> :Single-Microphone Speaker Separation and Voice Activity Detection in Noisy and Reverberant Environments



## 🚀 Getting Started

### 1. Pull the Docker Image

To pull the Docker image `mordehay/separation_vad:1.0`, use the following command:

```bash
docker pull mordehay/separation_vad:1.0
```

### 2. Creating a Container

Once the image has been pulled, you can create and start a container:

```bash
docker run -it mordehay/separation_vad:1.0 /bin/bash
```

You will be placed into an interactive terminal inside the container.

## 🎯 Running the Inference Script

With the container up and running, execute the `only_inference.py` script. Here are the available arguments for the script:

```bash
python only_inference.py \
-c/--config [CONFIG_PATH] \
-r/--resume [CHECKPOINT_PATH] \
-d/--device [DEVICE] \
-sp/--save_test_path [SAVE_PATH] \
-o/--online [TRUE/FALSE] \
-ps/--precision_save [16/32] \
-pm/--path_mix [MIX_SIGNAL_PATH] \
-ikw/--inference_kw [INFERENCE_OPTIONS]
```

### 📜 Argument Descriptions:

- **`-c/--config`**: Path to the config file. 
  - Default: `config_with_vad.json`
  
- **`-r/--resume`**: Path to the latest checkpoint.
  - Default: `model_without_vad.pth`
  
- **`-d/--device`**: Indices of GPUs to enable. 
  - Default: All GPUs.
  
- **`-sp/--save_test_path`**: Directory where separated signals will be saved. 
  - Default: Current directory.
  
- **`-o/--online`**: Online mode toggle.
  - Default: `True`
  
- **`-ps/--precision_save`**: Precision of the separated signals. 
  - Choices: `16` or `32`
  - Default: `32`
  
- **`-pm/--path_mix`**: Path to the mixed signal for running the separation. Required argument.

- **`-ikw/--inference_kw`**: Options for the inference phase, such as:
  - `filter_signals_by_smo_vad`: Filter by smoothed VAD (Default: `False`).
  - `filter_signals_by_unsmo_vad`: Filter by unsmoothed VAD (Default: `False`).
  - `length_smoothing_filter`: Length of smoothing filter (Default: `3`).
  - `threshold_activated_vad`: VAD threshold (Default: `0.5`).
  - `return_smoothed_vad`: Toggle to return smoothed VAD (Default: `False`).

  This argument must be dictionary as JSON string for an exmaple: 
  ```
  '{"filter_signals_by_smo_vad": false, 
    "length_smoothing_filter": 3, 
    "return_smoothed_vad": true}'
  ```



<!-- ## For Further Details and Support:
For more information, detailed documentation, and further resources, please visit the [official Sep-TFAnet VAD website](https://sep-tfanet.github.io/). -->

---
