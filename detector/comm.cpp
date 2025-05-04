#include "comm.hpp"
#include "utils.hpp"


struct HackedComm {
#if NCCL_VERSION > 22000
    uint64_t startMagic;
#endif
    struct ncclMemoryStack memPermanent, memScoped;
    // List of destructors to run when comm is destructed
    struct ncclDestructor* destructorHead;

    struct ncclSharedResources* sharedRes;
    /* map to top parent ranks. */
    int* topParentRanks;
    int* topParentLocalRanks;
    struct ncclChannel channels[MAXCHANNELS];
    struct ncclPeerInfo* peerInfo;
    struct ncclTopoSystem* topo;

    void* ncclNet;  // Hack: it should be ncclNet_t*
    void* ncclCollNet;  // Hack: it should be ncclCollNet_t*
    void* bootstrap;
    // Bitmasks for ncclTransportP2pSetup
    uint64_t* connectSend;
    uint64_t* connectRecv;

    uint64_t magic; // Magic number for all network communication. Not a security key -- only goal is to detect mismatches.

    uint64_t commHash;
    int rank;    // my rank in the communicator
    int nRanks;  // number of GPUs in communicator
    int cudaDev; // my cuda device index
    int nvmlDev; // my nvml device index
    int compCap; // compute capability of the GPU
    int minCompCap, maxCompCap; // min/max compute capability in the communicator
    int64_t busId;   // my PCI bus ID in int format
    cpu_set_t cpuAffinity; // CPU affinity of the GPU
    int cudaArch; // matches __CUDA_ARCH__ of device

    int node;
    int nNodes;
    int localRank;
    int localRanks;
    int maxLocalRanks;
    int* rankToNode;
    int* rankToLocalRank;
    int* localRankToRank;
    // localRanks and localRanktoRank for all nodes
    struct ncclNodeRanks* nodeRanks;

    bool checkPointers;
    bool dmaBufSupport;

    // Counter for tracking CUDA launches (P2P and collectives included)
    uint64_t opCount;

    // Channels for collectives
    int nChannels;
    int nvlsChannels;
    int collNetChannels;
    // Channels (per peer) for p2p
    int p2pnChannels;
    int p2pnChannelsPerPeer;
    int p2pChannels[MAXCHANNELS];

    // Should this comm allocate LL buffers for network P2P connections?
    bool allocP2pNetLLBuffers;

    // Buffer sizes
    int buffSizes[NCCL_NUM_PROTOCOLS];
    int p2pChunkSize;

    // Algorithm/Protocols thresholds
    ssize_t threadThresholds[NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS];
    float latencies[NCCL_NUM_FUNCTIONS][NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS];
    float bandwidths[NCCL_NUM_FUNCTIONS][NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS];
    float ringbdw[NCCL_NUM_FUNCTIONS][NCCL_NUM_PROTOCOLS];
    int maxThreads[NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS];

    /* This attribute can indicate the states of communicators and return code of
    * asynchronous NCCL operations. */
    ncclResult_t asyncResult;

    // Flag to ask NCCL kernels to abort
    volatile uint32_t *abortFlag;
    volatile uint32_t *childAbortFlag;
    volatile uint32_t *abortFlagRefCount;

    // Device side of the communicator (for cudaFree's)
    struct ncclDevComm* devComm; // actually = &ncclDevCommAndChannels::comm

    // Operation pool.
    int workFifoDepth; // size of workFifoHeap[], power of 2
    struct ncclWork* workFifoHeap;
    struct ncclWork* devWorkFifoHeap;
    void* workFifoHeapGdrHandle;

    // Work completion notificaion
    uint32_t* workFifoDone/*[MAXCHANNELS]*/; // in cudaHost memory
    uint32_t workFifoSent; // Monotonic (mod 1<<32) index of next unused fifo slot.
    uint32_t workFifoAckdMin; // Monotonic index of least unprocessed fifo slot over all channels.

    // Intra-process sync
    struct ncclComm* intraComm0; // leader of intra-process comms (self possible)
    struct ncclComm* intraNext; // next of intra-process comms, intraComm0 is head
    int intraRank;
    int intraRanks;
    uint32_t intraBarrierPhase;
    char intraPad1[64 - sizeof(uint64_t)];
    uint64_t intraBarrierCounter; // only used if this is intraComm0
    char intraPad2[64 - sizeof(uint64_t)];
    uint64_t intraBarrierGate; // only used if this is intraComm0

    struct ncclProxyState* proxyState;
    int proxyRefCountOld; /* store proxy post-atomic-sub refcount */
    // Whether this communicator uses collNet
    int collNetSupport;
    uint8_t collNetSupportMatrix[4/*sum,prod,min,max*/][ncclNumTypes];
    int intraHighestTransportType;
    int* collNetHeads;
    int collNetHeadsNum;
    /* sharable collNet proxy progress resource. */
    struct ncclCollNetSharedRes* collNetSharedRes;

    // NVLink SHARP (NVLS) support
    int nvlsSupport;
    int nvlsRegSupport;
    /* sharable NVLS resource. */
    struct ncclNvlsSharedRes* nvlsResources;
    struct ncclShmemCollBuff nvlsShmem;
    void *nvlsShmemHandle;

    ssize_t channelSize; // User requested work size (bytes) for channel partitions

    /** !!! The following fields are not important for us, just remove them!!! **/
    // // pools backed by comm->memPermanent
    // struct ncclMemoryPool memPool_ncclProxyOp;
    // struct ncclMemoryPool memPool_ncclKernelPlan;
    // struct ncclMemoryPool memPool_ncclPointerList;
    // struct ncclMemoryPool memPool_ncclNvlsHandleList;
    // // Next comm in this thread's active ncclGroup[Start|End](). Holds "0x1" when
    // // this comm is not yet in a group.
    // struct ncclComm* groupNext;
    // // Subset of those in groupNext list. Holds 0x1 if not needing preconnect.
    // struct ncclComm* preconnectNext;
    // int persistentRefs; // number of persistent plan-lists capturing this comm
    // struct ncclTasks tasks;

    // // user-created reduction ops
    // int userRedOpCapacity, userRedOpFreeHead;
    // void *userRedOps; // Hack: it should be ncclUserRedOp

    // // Queue of things for the main thread to do
    // struct ncclIntruQueueMpsc<struct ncclCommCallback, &ncclCommCallback::next> callbackQueue;

    // // List of kernel plans built form tasks.
    // struct ncclIntruQueue<struct ncclKernelPlan, &ncclKernelPlan::next> planQueue;
    // // First of the unlaunched kernels in `planQueue`
    // struct ncclKernelPlan* unlaunchedPlansHead;

    // ncclConfig_t config;
    // // initState is to more conveniently reclaim resources when errors happen.
    // ncclResult_t initState;
    // // flag to indicate if ncclCommFinalize() is called
    // bool finalizeCalled;
    // // shared structures for finalization
    // int finalizeRankCnt;
    // // group job to support multi-thread FT
    // struct ncclGroupJob *groupJob;

    //   /* store to buffer register request */
    //   struct ncclIntruQueue<struct ncclRegRequest, &ncclRegRequest::next> regRequestQueue;
    //   /* store registered buffer */
    //   struct ncclIntruQueue<struct ncclRegRecord, &ncclRegRecord::next> regRecordQueue;

    //   // Tuning plugin
    //   ncclTuner_t* tuner;
};


void parse_communicator(ncclComm_t hidden_comm, Communicator* parsed_comm)
{
    // pure reverse engineering, I'm sb
    HackedComm* hcomm = reinterpret_cast<HackedComm*>(hidden_comm);
    parsed_comm->comm_addr = reinterpret_cast<uint64_t>(hcomm);
    parsed_comm->num_devices = hcomm->nRanks;

    // Note: the `rank` of a communicator is the rank in its communication group
    parsed_comm->group_rank = hcomm->rank;

    // For torchrun/ompi, RANK and LOCAL_RANK are properly set in environment variables
    parsed_comm->local_rank = get_local_rank(DistEngine::auto_find);
    parsed_comm->global_rank = get_rank(DistEngine::auto_find);

    parsed_comm->num_channels = hcomm->nChannels;

    for (int i = 0; i < hcomm->nChannels; i++){
      parsed_comm->add_ring(hcomm->channels[i].ring);
      parsed_comm->add_tree(hcomm->channels[i].tree);
    }
}
