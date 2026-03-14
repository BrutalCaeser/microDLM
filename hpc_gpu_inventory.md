# Northeastern Explorer HPC Cluster - GPU Inventory

**Cluster:** Explorer
**Hostname:** `explorer.northeastern.edu`
**Username:** `gupta.yashv`
**Documented:** 2026-03-12

---

## GPU Summary

| GPU Type | Count per Node | Node(s) | Status |
|----------|---------------|---------|--------|
| V100-SXM2 | 4 | d1002, d1007, d1009, d1010, d1011, d1012, d1019, d1027 | mixed/allocated |
| V100-SXM2 | 3-4 | d1013, d1015, d1017 | drained/mixed |
| V100-PCIE | 2 | c2204, c2205, c2206, c2207 | idle/mixed |
| V100 | 4 | d3091, d3092, d3093, d3094, d3095, d3096, d3098 | idle/mixed |
| A100 | 8 | d3146, d3203 | **idle** |
| A100 | 7-8 | d3149, d3204 | idle/mixed |
| A100 | 3 | d1026, d1033 | mixed |
| H200 | 8 | d4052, d4053, d4054, d4055 | mixed |
| H100 | 4 | d4041 | mixed |
| L40S | 4-8 | d4042, d4043, d4044, d4047, d4050, d4051 | mixed |
| L40 | 10 | d3230, d3231 | mixed |
| T4 | 4 | d1025 | mixed |
| A5000 | 8 | d3165, d3166, d3170, d3171, d3194 | mixed |
| A6000 | 2-8 | d3168, d3232, d4056 | mixed |
| A30 | 6 | d4100 | mixed |
| MI50 | 8 | d3163 | drained |
| P100 | 3-4 | c2184, c2185, c2186, c2187, c2188, c2193, c2194, c2195 | idle |
| Quadro | 3 | d3089, d3090 | mixed |

---

## GPU Specifications

### High-End GPUs (AI/ML Recommended)

| GPU | Memory | FP16 (TFLOPS) | FP32 (TFLOPS) | Best For |
|-----|--------|---------------|---------------|----------|
| H200 | ~141 GB HBM3 | N/A | N/A | Large model training |
| H100 | 80 GB HBM3 | N/A | N/A | Large model training |
| A100 | 40/80 GB HBM2e | 312 | 19.5 | AI/ML, Scientific |
| V100 | 16/32 GB HBM2 | 125 | 15.7 | Deep Learning |

### Mid-Range GPUs

| GPU | Memory | Best For |
|-----|--------|----------|
| L40S | 48 GB GDDR6 | AI inference, Graphics |
| L40 | 48 GB GDDR6 | AI inference, Graphics |
| A5000 | 24 GB GDDR6 | CAD, DCC, AI |
| A6000 | 48 GB GDDR6 | Professional viz |
| T4 | 16 GB GDDR6 | Inference, Training |

### Legacy GPUs

| GPU | Memory | Notes |
|-----|--------|-------|
| P100 | 16 GB HBM2 | Older gen, still capable |
| Quadro | Varies | Professional graphics |
| A30 | 24 GB HBM2 | Data center GPU |
| MI50 | 16/32 GB HBM2 | AMD GPU |

---

## Node Status Legend

| Status | Meaning |
|--------|---------|
| `idle` | Available for jobs |
| `mixed` | Some resources in use |
| `allocated` | Fully allocated to jobs |
| `drained` | Admin-disabled, not available |
| `down` | Node offline |
| `draining` | Being drained for maintenance |
| `reserved` | Reserved for specific use |

---

## Available Resources Summary

### Currently Idle (Ready for Jobs)
- **A100 (8x)**: Nodes `d3146`, `d3203` - Best for AI/ML
- **V100 (4x)**: Nodes `d3092-d3096`, `d3098` - Great for DL
- **V100-PCIE (2x)**: Nodes `c2206`, `c2207` - Good general purpose
- **P100 (3-4x)**: Nodes `c2184-c2188`, `c2193-c2195` - General compute

### Recommended Partitions for Different Workloads

| Workload | Recommended GPU | Target Nodes |
|----------|-----------------|--------------|
| Deep Learning Training | A100, V100 | d3146, d3203, d3092-d3098 |
| Large Model Training | H200, H100 | d4052-d4055, d4041 |
| Inference | L40S, T4 | d4042-d4051, d1025 |
| General GPU Compute | V100, P100 | d3092-d3098, c2184-c2195 |
| Visualization | Quadro, A5000 | d3089-d3090, d3165-d3194 |

---

## Partitions Overview

| Partition | Default | Max Time | Max Nodes | GPUs | Best For |
|-----------|---------|----------|-----------|------|----------|
| `gpu` | No | 8 hours | Unlimited | 102 | Long GPU jobs |
| `short` | **Yes** | 48 hours | 2 | 4 | General CPU jobs |
| `courses-gpu` | No | 24 hours | Unlimited | 34 | Course GPU work |
| `courses` | No | 24 hours | Unlimited | - | Course CPU work |
| `sharing` | No | 1 hour | 2 | 277 | Low priority jobs |
| `gpu-short` | No | 2 hours | Unlimited | 102 | Quick GPU tests |
| `gpu-interactive` | No | 2 hours | Unlimited | 102 | Interactive sessions |

### Partition Details

#### `gpu` (GPU Jobs)
- **Max Time:** 8 hours
- **Max Nodes:** Unlimited
- **GPUs:** 102 total (V100, A100, H200, etc.)
- **Nodes:** c2204-c2207, d1002-d1029, d4052-d4055
- **Use for:** Standard GPU workloads up to 8 hours

#### `short` (Default CPU Partition)
- **Max Time:** 48 hours (2 days)
- **Max Nodes:** 2
- **GPUs:** 4 total (limited GPU availability)
- **Nodes:** 321 nodes (c0160-c0690, d0002-d0150)
- **Use for:** General CPU jobs, default partition

#### `courses-gpu` (Course GPU Work)
- **Max Time:** 24 hours
- **Max Nodes:** Unlimited
- **GPUs:** 34 total
- **Nodes:** c2184-c2195, d1004
- **Access:** Only for `rc` and `courses` groups
- **Use for:** Course-related GPU assignments

#### `gpu-short` (Quick GPU Jobs)
- **Max Time:** 2 hours
- **Max Nodes:** Unlimited
- **GPUs:** 102 total
- **Use for:** Quick GPU tests, debugging, small runs

#### `gpu-interactive` (Interactive Sessions)
- **Max Time:** 2 hours
- **Max Nodes:** Unlimited
- **GPUs:** 102 total
- **Use for:** Interactive development with `srun --pty`

#### `sharing` (Low Priority)
- **Max Time:** 1 hour
- **Max Nodes:** 2
- **GPUs:** 277 total (largest GPU pool)
- **QoS:** lowpriority (can be preempted)
- **Use for:** Non-urgent, fill-in jobs

---

## Useful Commands

```bash
# Check GPU availability
sinfo -o "%N %T %G"

# Check partition details
sinfo -p

# Show all partition info
scontrol show partitions

# Check quotas
squota

# Submit a GPU job (gpu partition, 4 hours, 1 V100)
sbatch --partition=gpu --time=04:00:00 --gres=gpu:v100:1 job_script.sh

# Quick GPU test (gpu-short partition, 1 hour)
sbatch --partition=gpu-short --time=01:00:00 --gres=gpu:1 job_script.sh

# Interactive GPU session (2 hours max)
srun --partition=gpu-interactive --time=02:00:00 --gres=gpu:v100:1 --pty /bin/bash

# Interactive with specific GPU type
srun --partition=gpu-interactive --gres=gpu:a100:1 --pty /bin/bash
```

---

## Notes

- **Drained nodes** (`d1015`, `d1017`, `d3163`, `d1004`, `d3150`) are temporarily unavailable
- **Mixed status** means some GPUs on the node are in use but others may be available
- **H200/H100** nodes are the newest and most powerful but currently all allocated/mixed
- For best queue times, target **V100** or **P100** nodes which have more availability

---

## Contact/References

- Northeastern HPC Support: [Insert link if available]
- Documentation: https://northeastern.edu/hpc (verify)
- Job submission: `sbatch <script.sh>`
- Interactive jobs: `srun --pty /bin/bash`
