#pragma once
#include <map>
#include <string>
#include <sstream>
#include <cstddef>
#include <cpp_redis/cpp_redis>
#include <boost/log/trivial.hpp>
#include <boost/interprocess/shared_memory_object.hpp>
#include <boost/interprocess/mapped_region.hpp>
#include <boost/interprocess/sync/interprocess_mutex.hpp>
#include <boost/interprocess/sync/scoped_lock.hpp>
#include "nccl_structs.hpp"

#define MAXFUNCTIONS 7


uint64_t hash_nccl_id(const char* str, size_t len);

struct Communicator
{
    uint64_t id_hash;
    uint64_t num_channels;
    uint64_t last_ring_id, last_tree_id;
    uint64_t comm_addr;
    int num_devices;
    int global_rank;
    int local_rank;
    int group_rank;
    ncclRing rings[MAXCHANNELS];
    ncclTree trees[MAXCHANNELS];
    int comm_ops[MAXFUNCTIONS];
public:
    Communicator();
    Communicator(const Communicator& other);
    Communicator(uint64_t addr, int my_rank, uint64_t num_channels);
    ~Communicator();
    void serialize(std::ostream& out);
    void from_bytes(std::istream& in);
    void add_ring(ncclRing& ring);
    void add_tree(ncclTree& tree);
    void debug_print();
};


class NcclTopoConnection
{
    cpp_redis::client client;
public:
    NcclTopoConnection(int n_ranks);
    ~NcclTopoConnection();
    std::shared_ptr<Communicator> add_comm(Communicator& comm);
    std::shared_ptr<Communicator> find(uint64_t comm_addr);
    std::vector<std::shared_ptr<Communicator> > find_linked_ranks(const std::shared_ptr<Communicator> comm);
    uint64_t num_comms();
};
