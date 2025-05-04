#pragma once
#include <chrono>
#include <memory>
#include <iostream>
#include <vector>
#include <exception>
#include <system_error>
#include <cuda.h>
#include <nccl.h>
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <cpp_redis/cpp_redis>
#include <boost/log/trivial.hpp>
#include <boost/log/core.hpp>
#include <boost/log/expressions.hpp>
#include <boost/process.hpp>
#include <boost/log/utility/setup/common_attributes.hpp>
#include <boost/log/utility/setup/file.hpp>
#include "config.hpp"
#include "shm_storage.hpp"
#include "shm_topo.hpp"
#include "event_handler.hpp"

using namespace std::chrono;


struct GlobalStatus {
    // NCCL dynamic linked library
    std::string nccl_lib_path;
    void* nccl_lib_handle;

    // storage buffer on shared memory
    std::shared_ptr<NcclRecordStorage> storage_buffer;

    // Topo links between GPUs
    std::shared_ptr<NcclTopoConnection> topo_buffer;
    std::vector<Communicator> local_comms;
    ncclComm_t comm_in_group;
    uint64_t comm_nccl_id_hash;

    // timing utils
    ControlState state;
    cudaEvent_t group_op_start, group_op_stop;
    cudaStream_t curr_stream;
    NcclNumber event_op;
    bool has_events_in_group;
    bool in_group;

    // TP related compression operations
    NcclNumber last_call_id;
    ncclComm_t last_comm;
    uint64_t repeated_call_num;
    uint64_t accumulated_count;
    float accumulated_duration;

    // Running time since training starts
    system_clock::time_point start_time;

    // Temporary record buffer
    std::vector<Record> tmp_record_buffer;

    // To implement worker-side response to controllers
    std::shared_ptr<EventHandler> event_handler;
    bool should_check;
    bool transparent;

    // global controller & local controllers
    std::shared_ptr<boost::process::child> global_controller_proc;
    std::shared_ptr<boost::process::child> local_controller_proc;
private:
    int install_python_packages(std::string whl_path);
    void wait_installation_done();
    int start_global_controller();
    int start_local_controller();
public:
    GlobalStatus() = default;
    GlobalStatus(const char* nccl_path_);
    ~GlobalStatus();

    // Initializes all status
    void initialize(const char* nccl_path_);

    // Returns a function pointer of given `func_name` from NCCL lib
    void* get_function_ptr(const char* func_name);

    // Returns the time since probe initialization in microseconds (us)
    double time_since_initialize();

    // Entry & Exit of NCCL groups
    void group_start();
    void group_end();

    // Updates the TP accumulated calls (AllGather, ReduceScatter)
    void update_accumulation(NcclNumber last_call_number, uint64_t count, ncclComm_t tp_comm);

    // Resets the TP accumulated calls (AllGather, ReduceScatter)
    void reset_accumulation(NcclNumber last_call_number);

    // Add a cudaEvent before an NCCL operation for timing 
    void add_timing_event(NcclNumber op, uint64_t count, cudaStream_t stream);

    // Logs this event cost in microseconds (us)
    double get_communication_time();
};