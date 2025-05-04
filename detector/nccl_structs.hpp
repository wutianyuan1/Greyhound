#pragma once
#include <cstdint>
#include <cstdlib>
#include <sched.h>
#include <unistd.h>
#include "config.hpp"

#define MAXCHANNELS 32
#define NCCL_NUM_PROTOCOLS 3 // Simple/LL/LL128
#define NCCL_NUM_ALGORITHMS 6 // Tree/Ring/CollNet*
#define NCCL_NUM_FUNCTIONS 5 // Send/Recv not included for now
// Hack: we do not need them
struct ncclChannelPeer {};
struct ncclDevChannelPeer {};


struct ncclShmemCollBuff {
  volatile size_t *cnt[2];
  volatile void *ptr[2];
  int round;
  size_t maxTypeSize;
};


struct ncclMemoryStack {
    struct Hunk {
        struct Hunk* above; // reverse stack pointer
        size_t size; // size of this allocation (including this header struct)
    };
    struct Unhunk { // proxy header for objects allocated out-of-hunk
        struct Unhunk* next;
        void* obj;
    };
    struct Frame {
        struct Hunk* hunk; // top of non-empty hunks
        uintptr_t bumper, end; // points into top hunk
        struct Unhunk* unhunks;
        struct Frame* below;
    };

    static void* allocateSpilled(struct ncclMemoryStack* me, size_t size, size_t align);
    static void* allocate(struct ncclMemoryStack* me, size_t size, size_t align);

    struct Hunk stub;
    struct Frame topFrame;
};


struct ncclRing
{
  // Shortcuts for userRanks[1] and userRanks[n-1]
  int prev;
  int next;

  // Maps an internal nccl index to user-specified rank order. This is necessary
  // since we need to know how the user expects data to be ordered across
  // devices. Ordered from current device.
  int* userRanks;

  int index; // This rank's index in the ring
};


// The root of each tree only has one node down (+1 intra-node).
#define NCCL_MAX_TREE_ARITY_TOP 2
// Nodes inside the binary tree can have to two nodes down (+1 intra-node).
#define NCCL_MAX_TREE_ARITY 3
struct ncclTree {
  int depth;
  int up;
  int down[NCCL_MAX_TREE_ARITY];
};

#define NCCL_MAX_DIRECT_ARITY 7
struct ncclDirect {
  int depth;
  int out;
  int nHeads;   // Number of parallel N<->1<->net operations we'll do in parallel; size of up/down
  int headRank; // Index in 0..nHeads-1 I am the head rank of. -1 if I'm not a head rank (no local NIC)
  int shift;    // Shuffling of send/recv for scatter/gather operations, basically localRank%nHeads
  int up[NCCL_MAX_DIRECT_ARITY];
  int down[NCCL_MAX_DIRECT_ARITY];
};

#if NCCL_VERSION_CODE > 22000
#define NCCL_MAX_NVLS_ARITY 32
#else
#define NCCL_MAX_NVLS_ARITY 8
#endif

#define NCCL_MAX_NVLS_TREE_ARITY 3
struct ncclNvls {
  int out;
  int nHeads;   // Number of parallel N<->1<->net operations we'll do in parallel; size of up/down
  int headRank; // Index in 0..nHeads-1 I am the head rank of. -1 if I'm not a head rank (no local NIC)
  int up[NCCL_MAX_NVLS_ARITY];
  int down;
  int treeUp;
  int treeDown[NCCL_MAX_NVLS_TREE_ARITY];
  int node;
  int nNodes;
};


struct ncclChannel {
  struct ncclChannelPeer** peers;
  struct ncclDevChannelPeer** devPeers;
#if NCCL_VERSION_CODE > 21800
  /* devPeer pointer array used for host side access */
  struct ncclDevChannelPeer** devPeersHostPtr;
#endif
  struct ncclRing ring;
  int* devRingUserRanks;
  struct ncclTree tree;

  struct ncclTree collnetChain;
  struct ncclDirect collnetDirect;

  struct ncclNvls nvls;

  int id; // index of this channel
  uint32_t workFifoSent; // last used work index+1

  /* comm split sharable resources */
  struct ncclChannelPeer* collnetPeers;
  struct ncclDevChannelPeer* collnetDevPeers;
  struct ncclChannelPeer* nvlsPeers;
  struct ncclDevChannelPeer* nvlsDevPeers;
};
