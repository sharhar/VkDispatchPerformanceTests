# GPU Clock Locking for Performance Testing

## Overview
This guide documents the procedure for locking NVIDIA GPUs to maximum performance states during benchmarking and performance testing. This is critical for eliminating dynamic voltage and frequency scaling (DVFS) noise in scientific measurements.

**Hardware:** NVIDIA RTX 6000 Ada Generation (4x)  
**OS:** Linux  
**APIs:** CUDA and Vulkan  
**Use Case:** Performance testing for IEEE Access paper on Vulkan FFTs

## Why Lock GPU Clocks?

By default, NVIDIA GPUs use dynamic boost algorithms that adjust clock frequencies based on power, thermal, and utilization conditions. This introduces variance between benchmark runs, which is unacceptable for rigorous performance analysis. Locking clocks ensures:

- **Deterministic Performance:** Identical runs produce identical timing results
- **Fair Comparisons:** Eliminates DVFS as a confounding variable
- **Reproducible Results:** Other researchers can replicate your exact configuration

## Prerequisites

All commands require `root` privileges. Ensure you have:
- NVIDIA drivers installed (`nvidia-smi` available)
- `sudo` access
- Adequate cooling (server-grade airflow recommended for sustained loads)

---

## Step 1: Enable Persistence Mode

Persistence mode keeps the NVIDIA driver loaded even when no GPU processes are running. **Without this, clock locks reset between test runs.**

```bash
sudo nvidia-smi -pm 1
```

**Verification:**
```bash
nvidia-smi
```
Look for `Persistence-M: On` in the output.

---

## Step 2: Query Maximum Clocks

Identify the theoretical maximum clocks supported by your GPU architecture.

```bash
nvidia-smi --query-gpu=name,clocks.max.gr,clocks.max.mem --format=csv
```

**Example Output (RTX 6000 Ada):**
```
name, clocks.max.graphics [MHz], clocks.max.memory [MHz]

NVIDIA RTX 6000 Ada Generation, 3105 MHz, 10001 MHz
```
**Important:** The "max graphics" value (3105 MHz) is the theoretical boost ceiling. Under sustained compute workloads, attempting to lock at this frequency often causes thermal throttling, which defeats the purpose.

---

## Step 3: Lock Clocks to Recommended Values

### Recommended Configuration (RTX 6000 Ada)

- **Graphics Clock:** `2505 MHz` (official Boost Clock specification)
- **Memory Clock:** `10001 MHz` (maximum stable bandwidth)

**Rationale:**
- **2505 MHz Graphics:** High enough for excellent performance, low enough to sustain without throttling under heavy FFT workloads
- **10001 MHz Memory:** FFTs are memory-bandwidth bound; lock memory at maximum

### Apply the Lock (Sequential Commands)

```bash
# Lock Graphics Clock
sudo nvidia-smi -lgc 2505

# Lock Memory Clock
sudo nvidia-smi -lmc 10001
```

## Cleanup: Restoring Default Behavior

After completing your benchmarking session, it is critical to restore the GPUs to their default dynamic boost behavior. Leaving clocks locked at maximum frequencies will:

- Waste power during idle periods or light workloads
- Generate unnecessary heat
- Reduce hardware lifespan through sustained high-frequency operation
- Prevent the GPU from entering low-power states

### Complete Reset Procedure

Run these commands in sequence to fully restore default operation:

```bash
# 1. Reset Graphics Clock to default boost behavior
sudo nvidia-smi -rgc

# 2. Reset Memory Clock to default behavior
sudo nvidia-smi -rmc

# 3. Reset Power Limit (if modified)
sudo nvidia-smi -rpl

# 4. Disable Persistence Mode
sudo nvidia-smi -pm 0
```

### Verification

Confirm the GPUs have returned to dynamic boost mode:

```bash
nvidia-smi --query-gpu=clocks.gr,clocks.mem,persistence_mode --format=csv
```

**Expected Output:**
- Graphics and memory clocks should now show **variable** frequencies (not locked at 2505/10001)
- `Persistence Mode` should show `Disabled`
- At idle, graphics clock should drop to ~200-500 MHz

### Quick Check: Idle Power

At system idle, power draw should drop significantly:

```bash
nvidia-smi --query-gpu=power.draw --format=csv
```

**Expected:** ~20-40W per GPU at idle (vs ~300W under locked load)
