# Greyhound: Hunting Fail-Slows in Hybrid-Parallel Training at Scale  
*Artifact Evaluation Guidelines for ATC'25 Paper #164*  


## Testbed Environment  
We provide an **AWS VM instance with 4 NVIDIA A10 GPUs** to reproduce key results from the paper, including:  
- **Real-Time Performance Tracking** (Section 4.1)  
  - ACF-based iteration time estimation  
  - Slow iteration detection via Bayesian Online Change Point Detection (BOCD)  
- **Profiling & Validation** (Section 4.2)  
  - Pre-training computation/communication performance checks  
  - Reactive profiling upon fail-slow detection  
- **Adaptive Straggler Mitigation** (Section 5.1)  
- **Micro-Batch & Parallelism Adjustments** (Section 5.2):  
  - S1: Baseline (no action)  
  - S2: Micro-batch tuning for computation stragglers  
  - S3: Communication-aware pipeline reconfiguration  
  - S4: Checkpoint-restart strategy  

*Note:* Full reproduction of large-scale experiments and topology switches requires a cluster with hundreds of GPUs. While we cannot provide such infrastructure, reviewers with access to large-scale resources may replicate these experiments.

---

## Installation Guide  

### Option 1: Docker Setup (Recommended)  
1. Pull the pre-built image:  
   ```bash
   docker pull tianyuanwu/greyhound:ae
   ```
2. Launch the container:  
   ```bash
   bash ./start_container.sh
   ```
   - The Greyhound codebase is already at `/workspace`.
   - To use an external codebase: Keep this line in `start_container.sh`:  
     ```bash
     -v `pwd`/../Greyhound:/workspace/Greyhound  # Mount external repo
     ```
   - To use the default internal codebase: Delete/comment the above line.

---

### Option 2: Manual Local Setup  
1. Clone the repository:  
   ```bash
   git clone https://github.com/wutianyuan1/Greyhound
   ```
2. Install dependencies:  
   ```bash
   sudo apt-get install redis-server libboost-all-dev
   pip install redis numpy matplotlib pandas statsmodels Rbeast cvxpy ecos
   ```
3. Build components:  
   ```bash
   # Build detector
   cd ${REPO_PATH}/detector && mkdir build && cd build
   cmake .. && make -j

   # Install controllers
   cd ${REPO_PATH}/detector
   python setup.py bdist_wheel && python setup.py install
   ```

---

## Reproducing Experiments  

### Step 1: Launch Training  

**Option A – Docker Environment**  
- Basic detection (Parallelism `[2TP, 1DP, 2PP]`):  
  ```bash
  python run_training.py
  ```
- Detection + mitigation (Parallelism `[1TP, 4DP, 1PP]`):  
  ```bash
  python run_training_dp.py
  ```

**Option B – Custom Cluster Setup**  
1. Edit `run_training_dp.py`:  
   ```python
   # Line 102: Set detector library path
   my_env['LD_PRELOAD'] = "<PATH_TO_DETECTOR_SO>"  # ${REPO_PATH}/detector/build/*.so

   # Line 103: Set controller wheel path
   my_env['CONTROL_PLANE_WHL_PATH'] = "<PATH_TO_CONTROLLER_WHL>"  # ${REPO_PATH}/detector/dist/*.whl

   # Lines 153-155: Configure cluster topology
   master = "<MASTER_IP>"
   nnodes = <NODE_COUNT>
   rank = <NODE_RANK>  # 0-indexed

   # Lines 214-215: Set dataset paths
   vocab_file = "<PATH_TO_GPT2_VOCAB>"
   merge_file = "<PATH_TO_GPT2_MERGES>"
   data_path = "<PATH_TO_GPT2_DATASET>"

   # Line 224: Configure checkpoints
   save = "<CHECKPOINT_SAVE_PATH>"
   load = "<CHECKPOINT_LOAD_PATH>"

   # Line 208: Set parallelism
   tensor_model_parallel_size=<TP_SIZE>, pipeline_model_parallel_size=<PP_SIZE>,
   ```
2. Execute:  
   ```bash
   python run_training_dp.py --logdir <LOG_PATH>
   ```

---

### Step 2: Inject Fail-Slow Scenarios  

**Computation Degradation**  
Lock GPU clock frequency:  
```bash
nvidia-smi -i <GPU_ID> -lgc <FREQ_MHz>  # e.g., 100 MHz for 10-30% slowdown on A10
```

**Communication Congestion**
Simulate network contention (no effects on single node, and the performance varies a lot):  
```bash
python ${REPO_PATH}/detector/injection/single_comm.py \
  --tensor-size <MEGA_BYTES> \          # e.g., 200 for 200MB
  --duration <SECONDS> \           # Congestion duration
  --logdir <OUTPUT_PATH>
```
Or you can use a [NCCL network plugin](https://github.com/NVIDIA/nccl/tree/master/ext-net) to add sleeps
for certain NCCL calls, which can get a more stable performance.

---

### Step 3: Analyze Results  

#### Log Structure  
Experiment logs are stored in `trainlog/` with format:  
```
trainlog/
  └── log_<TIMESTAMP>_noderank{N}/  # One folder per node
      ├── global_controller_[MASTER_IP].log    # Global coordination
      ├── local_controller_[NODE_RANK].log     # Node-level analysis
      ├── ncclprobe.log                       # NCCL interception logs
      └── megatron_output_[NODE_RANK].log     # Training process output
```

#### Key Log Indicators  

**Global Controller Log** (`global_controller_*.log`):  
- **Pre-Check Phase**:  
  ```
  ===== Performing pre-check before training start =====
  Build [TP/DP/PP] clique XXX        # Identified communication groups
  Computation tasks dispatched!      # Benchmarking computation
  Computation result of rank XX: min=YYY ms, max=ZZZ ms, avg=AAA ms
  Communication test: [S1→R1, ..., Sk→Rk]  # Concurrent p2p tests
  ```

- **Fail-Slow Mitigation**:
- The following example shows a typical mitigation plan in handling computation stragglers.
  ```
  Mitigating fail-slow...
  Root cause: comp/comm              # Identified bottleneck
  [Mitigation Plan] DPcost=2096.1, PPcost=60000, time_since_slow=3005.2  # Ski-rental based mitigation timing decison
  [DP solver] New DP plan: [9, 6, 9, 8]  # Micro-batch redistribution
  ```

**Local Controller Log** (`local_controller_*.log`):  
- Pattern Recognition:  
  ```
  Repeat pattern starts at XXX, period=YYY, pattern=[ZZZ]  # NCCL call IDs
  Estimated iteration time: {'rank0': 1234567µs, ...}      # Per-rank timing
  ```
- Additionally, `No peaks in ACF, continues...` indicates cannot estimate iteration time due to inadequate data points collected. In the repeat pattern of NCCL calls recognized by ACF algorithm `pattern=[ZZZ]`, items are the NCCL call IDs defined in `detector/config.hpp`:
    ```c++
    enum NcclNumber {
        SEND,
        RECV,
        BCAST,
        BROADCAST,
        ALL_GATHER,
        REDUCE_SCATTER,
        ALL_REDUCE,
        INVALID
    };
    ```
---

## Validation Metrics  

**Accuracy Testing**  
Compare iteration time estimates in `local_controller_*.log` against ground truth values from Megatron-LM logs.  

**Overhead Measurement**  
To measure detector overhead:  
1. Comment out Line 102 in `run_training.py`:  
   ```python
   # my_env['LD_PRELOAD'] = <DETECTOR_LIB_PATH>  # Disable detector
   ```
2. Compare iteration times with/without this line enabled.
