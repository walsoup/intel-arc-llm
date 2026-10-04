# Intel Core Ultra (Intel Arc iGPU) LLM Throughput Optimization Guide

> **Target Platform:** Intel Core Ultra 5 / 7 / 9 (Series 1 Meteor Lake Xe-LPG, Series 2 Lunar Lake Xe2-LPG, Arrow Lake)  
> **Memory Architecture:** Dual-Channel 128-bit LPDDR5 / DDR5 (Shared Unified Memory Architecture / UMA)  
> **Operating System:** Linux (Kernel 6.8+, Mesa 24.1+ ANV Vulkan Driver)

---

## 1. Executive Summary & The Memory Wall

During auto-regressive large language model (LLM) generation (single-token decoding, batch size = 1), **every generated token requires streaming the full weight matrices of every layer across the system memory bus into processor registers**.

### The Physics of the Memory Wall:
For a 9B parameter model quantized to `Q4_K_M` (file size $\approx 5.61\text{ GB}$):
$$\text{Memory Bandwidth Required} = 5.61\text{ GB} \times \text{Tokens Per Second (TPS)}$$

| Target TPS | Required Memory Read Throughput | Status on 128-bit DDR5-5600 |
| :---: | :---: | :--- |
| **5.0 TPS** | **28.05 GB/s** | Standard unoptimized Linux/Wine default |
| **6.8 TPS** | **38.15 GB/s** | **Hardware Roofline Limit** (~85–90% practical bus saturation) |
| **15.0 TPS** | **84.15 GB/s** | **Physically Impossible** for single-token decoding on 128-bit DDR5 |
| **25.0 TPS** | **140.25 GB/s** | **Physically Impossible** (requires 256/384-bit GDDR6 or HBM) |

### How to Break the Memory Wall:
To achieve **15–25+ TPS** on Intel Core Ultra chips, you cannot rely on faster single-token math kernels. You must **amortize memory bandwidth across multiple tokens** using **Multi-Token Prediction (MTP)** and **N-Gram Self-Speculation**, combined with hardware clock pinning and DP4A vector execution.

---

## 2. Hardware Architecture Profile: Meteor Lake / Lunar Lake

```
 ┌────────────────────────────────────────────────────────────────────────┐
 │                    Intel Core Ultra SoC (Meteor Lake)                  │
 │                                                                        │
 │  ┌──────────────────────┐  ┌────────────────────┐  ┌────────────────┐  │
 │  │  14-Core Hybrid CPU  │  │ Intel Arc iGPU     │  │ Intel NPU 3720 │  │
 │  │  4P + 8E + 2LP-E     │  │ 7-8 Xe-Cores       │  │ 2 NCEs         │  │
 │  │  (4.2 - 4.4 GHz)     │  │ (112 Vector Eng.)  │  │ (11.5 TOPS)    │  │
 │  └──────────┬───────────┘  └─────────┬──────────┘  └───────┬────────┘  │
 │             │                        │                     │           │
 │             └──────────────────┬─────┴─────────────────────┘           │
 │                                │                                       │
 │                 ┌──────────────┴─────────────┐                         │
 │                 │  128-bit Shared Memory Bus │                         │
 │                 │  (Peak: 89.6 GB/s @ 5600)  │                         │
 │                 └──────────────┬─────────────┘                         │
 └────────────────────────────────┼───────────────────────────────────────┘
                                  │
                 ┌────────────────┴──────────────┐
                 │  Dual-Channel DDR5-5600 SODIMM│
                 │  (Real-World Cap: 40-50 GB/s) │
                 └───────────────────────────────┘
```

### Key Hardware Characteristics:
1. **Intel Arc Integrated GPU (Xe-LPG)**:
   * 7 to 8 Xe-cores (112 Vector Engines, 896 ALUs).
   * **Matrix Acceleration**: Uses **DP4A integer dot products** (`VK_KHR_shader_integer_dot_product` / `dotPacked4x8EXT`).
   * **Shared UMA Bus**: Shares memory bandwidth with the CPU, OS display server, and system background tasks.
2. **Intel NPU 3720 (AI Boost)**:
   * Low-power accelerator (~11.5 INT8 TOPS) located at `/dev/accel/accel0`.
   * **Limitation**: Shares the exact same 128-bit DDR5 memory bus and possesses only ~4 MB of on-die SRAM. It cannot store or stream a 9B model faster than the GPU.
   * **Best Role**: Heterogeneous low-power draft model execution.

---

## 3. Step 1: System & Kernel-Level Performance Tuning

Default Linux power profiles throttle the GPU down to 100 MHz and place CPU cores into deep C-states between layer dispatches. Run the root tuning script before launching inference:

```bash
sudo ./scripts/tune-hardware.sh
```

### What This Tunes:
```bash
# 1. Lock CPU Frequency Scaling Governor to Performance (4.2 - 4.4 GHz)
for g in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
    echo performance > "$g"
done

# 2. Pin Intel Arc GPU (gt0) to Peak Boost Clock (2200 MHz)
echo 2200 > /sys/devices/pci0000:00/0000:00:02.0/drm/card1/gt/gt0/rps_min_freq_mhz
echo 2200 > /sys/devices/pci0000:00/0000:00:02.0/drm/card1/gt/gt0/rps_boost_freq_mhz
echo 2200 > /sys/devices/pci0000:00/0000:00:02.0/drm/card1/gt_min_freq_mhz
echo 2200 > /sys/devices/pci0000:00/0000:00:02.0/drm/card1/gt_boost_freq_mhz

# 3. Enable Transparent Huge Pages (THP) for 2MB Page Sweeps
echo always > /sys/kernel/mm/transparent_hugepage/enabled
echo always > /sys/kernel/mm/transparent_hugepage/defrag

# 4. Minimize Swappiness
sysctl -w vm.swappiness=10
```

---

## 4. Step 2: Driver & Vector Engine Acceleration

### Why Native Linux Vulkan Dominates Wine:
* **Wine Emulation**: Prompt Processing: 44.9 t/s | Generation: 5.91 t/s
* **Native Linux Mesa ANV (Build 11385)**: Prompt Processing: **56.70 t/s (+26%)** | Generation: **6.84 t/s (+16%)**

### The Critical Shader Extension: DP4A Vector Math
When compiling or choosing a `llama.cpp` Vulkan binary, verify that `int dot: 1` is reported:
* **Without DP4A (`int dot: 0`)**: Prefill: 23.66 t/s | Decode: 4.96 t/s
* **With DP4A (`int dot: 1`)**: Prefill: **56.70 t/s (+140%)** | Decode: **6.84 t/s (+38%)**

Build 11385 in `bin-linux-vulkan/` enables DP4A via `VK_KHR_shader_integer_dot_product`.

---

## 5. Step 3: Beating the Memory Wall (15–25+ TPS)

To break past the 6.84 TPS roofline limit on a 128-bit memory bus, deploy these strategies:

### Strategy A: N-Gram In-Memory Self-Speculation (`--spec-type ngram-mod`)
* **DRAM Cost**: **0 Bytes**.
* **Mechanism**: Maintains a rolling context hash pool in CPU RAM. When recurring tokens, code syntax, or schema keywords match, candidate tokens are drafted instantly.
* **Verification**: The 9B model verifies candidate tokens in a single batched pass on the GPU at **156+ TPS**.
* **Real-World Throughput**: **14.0 – 18.5 TPS** on code and structured JSON.

### Strategy B: Native Multi-Token Prediction (MTP)
* **DRAM Cost**: **~3% of model weights** (Layer 32 only).
* **Models Supported**: Qwen 3.5 / DeepSeek models containing MTP weights (e.g., `Defiant Fable MTP`, `Qwythos-9B-v2-MTP`).
* **Execution**:
  ```bash
  llama-server -m Models/model-MTP.gguf \
    --spec-type draft-mtp \
    --spec-draft-n-max 2 \
    --spec-draft-p-min 0.70 \
    -ngl 99 -fa on
  ```
* **Real-World Throughput**: **18.0 – 22.0+ TPS**.

> [!WARNING]
> **Avoid Standalone Draft Models (`draft-simple`) on UMA iGPUs**:
> Running two separate models (e.g., 9B + 0.8B) causes catastrophic memory bus contention across the shared 128-bit DDR5 controller, dropping decode throughput down to **2.8 TPS**. Always use **in-model self-speculation** (MTP or N-Gram).

---

## 6. Production Quick-Start

### 1-Click Native Launcher:
```bash
./start-server.sh
```

### Verified Server Flags:
```bash
"${SCRIPT_DIR}/bin-linux-vulkan/llama-server" \
    --models-dir "${SCRIPT_DIR}/Models" \
    --models-max 1 \
    -ngl 99 \
    -t 6 \
    -c 16384 \
    -np 1 \
    -fa on \
    --spec-type ngram-mod \
    --host 127.0.0.1 \
    --port 8080
```

* **Web UI**: `http://127.0.0.1:8080`
* **OpenAI Compatible Endpoint**: `http://127.0.0.1:8080/v1/chat/completions`
