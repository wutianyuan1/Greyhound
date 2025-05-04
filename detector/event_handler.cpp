#include "event_handler.hpp"
#include "comm.hpp"
#include "utils.hpp"
#include "matmul.cuh"

#define NUM_WARMUP 3
#define NUM_REPEATS 10
#define MATRIX_N 4096

static void parse_task(std::string& task, int* role, int* target_peer, uint64_t* comm_addr)
{
    std::stringstream ss(task);
    std::string segment;

    std::getline(ss, segment, '_');
    *role = std::stoi(segment);
    std::getline(ss, segment, '_');
    *target_peer = std::stoi(segment);
    std::getline(ss, segment, '_');
    *comm_addr = std::stoul(segment);
    return;
}


ProfileResult p2p_profile_task(
    int role, int peer, ncclSendFuncPtr send_func, ncclRecvFuncPtr recv_func,
    ncclComm_t comm = nullptr, cudaStream_t stream = nullptr)
{
    int count = 20 * 1024 * 1024;
    int *buf = nullptr;
    cudaEvent_t start, stop;
    float duration = 0.0f;
    cudaMalloc(&buf, count * sizeof(int));
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 3 warmup runs, this will not be recorded
    for (int i = 0; i < NUM_WARMUP; i++)
    {
        if (role == (int)ProcessRole::ROLE_SENDER) {
            cudaEventRecord(start, stream);
            ncclResult_t res = send_func(buf, count, ncclInt, peer, comm, stream);
            cudaEventRecord(stop, stream);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&duration, start, stop);
        } else if (role == (int)ProcessRole::ROLE_RECVER) {
            cudaEventRecord(start, stream);
            ncclResult_t res = recv_func(buf, count, ncclInt, peer, comm, stream);
            cudaEventRecord(stop, stream);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&duration, start, stop);
        }
    }

    std::vector<double> durations;
    for (int i = 0; i < NUM_REPEATS; i++)
    {
        if (role == (int)ProcessRole::ROLE_SENDER) {
            cudaEventRecord(start, stream);
            ncclResult_t res = send_func(buf, count, ncclInt, peer, comm, stream);
            cudaEventRecord(stop, stream);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&duration, start, stop);
            durations.push_back(duration);
            // printf("[Rank %d] I am sender, my target is %d, addr is %p, result is %d, duration=%f\n",
            //     get_rank(DistEngine::auto_find), peer, comm, 0, duration);
        } else if (role == (int)ProcessRole::ROLE_RECVER) {
            cudaEventRecord(start, stream);
            ncclResult_t res = recv_func(buf, count, ncclInt, peer, comm, stream);
            cudaEventRecord(stop, stream);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&duration, start, stop);
            durations.push_back(duration);
            // printf("[Rank %d] I am receiver, my sender is %d, addr is %p, result is %d, duration=%f\n",
            //     get_rank(DistEngine::auto_find), peer, comm, 0, duration);
        }
    }

    // Find min, max, and average
    double min = *std::min_element(durations.begin(), durations.end());
    double max = *std::max_element(durations.begin(), durations.end());
    double avg = std::accumulate(durations.begin(), durations.end(), 0.0) / durations.size();

    // Find standard deviation
    double sum = 0.0;
    for (double duration : durations) {
        sum += (duration - avg) * (duration - avg);
    }
    double std_dev = sqrt(sum / durations.size());

    return ProfileResult(min, max, avg, std_dev);
}


ProfileResult::ProfileResult(double minl, double maxl, double avgl, double stdl)
    : min_lat(minl), max_lat(maxl), avg_lat(avgl), std_lat(stdl)
{};

std::string ProfileResult::serialize()
{
    std::stringstream ss;
    ss.write(reinterpret_cast<const char*>(&min_lat), sizeof(min_lat));
    ss.write(reinterpret_cast<const char*>(&max_lat), sizeof(max_lat));
    ss.write(reinterpret_cast<const char*>(&avg_lat), sizeof(avg_lat));
    ss.write(reinterpret_cast<const char*>(&std_lat), sizeof(std_lat));
    return ss.str();
}


EventHandler::EventHandler(std::string master_addr, int port, ncclSendFuncPtr sptr, ncclRecvFuncPtr rptr)
{
    world_comm = nullptr;
    send_ptr = sptr;
    recv_ptr = rptr;
    client = std::shared_ptr<cpp_redis::client>(new cpp_redis::client());
    while (true) {
        try {
            client->connect(master_addr, port);
            break;
        }
        catch (std::exception& e) {
            std::cout << "Eventhandler:" << e.what() << std::endl;
            sleep(1);
        }
    }

    if (get_rank(DistEngine::auto_find) == 0)
    {
        std::string world_size = std::to_string(get_world_size(DistEngine::auto_find));
        client->set("world_size", world_size);
        client->sync_commit();
    }
    cudaMalloc(&this->control_state, sizeof(int) * 4);
}

EventHandler::~EventHandler()
{
    cudaFree(this->control_state);
    client->disconnect();
}

bool EventHandler::has_world_comm() const
{
    return this->world_comm != nullptr;
}

void EventHandler::set_world_comm(ncclComm_t comm)
{
    this->world_comm = comm;
    parse_communicator(world_comm, &(this->parsed_comm));
}

void EventHandler::fetech_and_exec_task()
{
    // check my task to do in this term.
    std::string task_name = std::string("validtask_rank_") + std::to_string(get_rank(DistEngine::auto_find));
    std::string task_content;
    client->get(task_name,
        [&](const cpp_redis::reply& reply) {
            if (reply.is_string())
                task_content = reply.as_string();
        }
    );
    client->sync_commit();
    // skip emptys
    if (task_content.length() <= 5)
        return;

    // If a task is already acked by the worker, just skip it.
    if (task_content.size() >= 10 && task_content == std::string(TASK_ACKED))
        return;
    // Otherwise, receive this task and prepare to execute it.
    client->set(task_name, TASK_ACKED);
    client->sync_commit();

    if (task_content == "ComputationTest")
    {
        auto result = perf_gemm(MATRIX_N);
        // Add the result to redis
        client->set(task_name + std::string("_result"), result.serialize());
        client->sync_commit();
    }
    else
    {
        int role, peer;
        uint64_t comm_addr;
        parse_task(task_content, &role, &peer, &comm_addr);
        // Perform p2p send/recv job and get exec time metrics
        auto result = p2p_profile_task(role, peer,
            this->send_ptr, this->recv_ptr,
            reinterpret_cast<ncclComm_t>(comm_addr), cudaStreamLegacy);
        // Add the result to redis
        client->set(task_name + std::string("_result"), result.serialize());
        client->sync_commit();
    }
}

void EventHandler::handle_control_signal(cudaStream_t curr_stream, ControlState* state)
{
    if (parsed_comm.group_rank == 0)
    {
        ControlState master_state = ControlState::STATE_MONITOR;
        client->get("control_state",
            [&](const cpp_redis::reply& reply) {
                if (!reply.is_string()) {
                    BOOST_LOG_TRIVIAL(error) << "Reply is not a string!";
                    return;
                }
                auto reply_str = reply.as_string();
                if (reply_str[0] != 0)
                    master_state = ControlState(reply_str[0] - '0');
            }
        );
        client->sync_commit();
        if (master_state != *state)  // broadcast if state changes
            cudaMemcpy(control_state, &master_state, sizeof(int), cudaMemcpyHostToDevice);
        ncclBroadcast(control_state, control_state, 1, ncclInt, 0, world_comm, curr_stream);
    } 
    else
        ncclBroadcast(nullptr, control_state, 1, ncclInt, 0, world_comm, curr_stream);

    ControlState sync_state = ControlState::STATE_MONITOR;
    cudaMemcpy(&sync_state, control_state, sizeof(int), cudaMemcpyDeviceToHost);
    *state = sync_state;

    // If we should pause, then loop to wait the "continue" (pause=0) signal 
    if (sync_state == ControlState::STATE_VALIDATE)
    {
        BOOST_LOG_TRIVIAL(info) << "Rank " << get_rank(DistEngine::auto_find) << " receives pause signal, paused!";
        while (true)
        {
            // fetch a task from redis and execute it
            fetech_and_exec_task();
        
            // check if we should continue
            if (parsed_comm.group_rank == 0)
            {
                int my_pause = 1;
                client->get("control_state",
                    [&](const cpp_redis::reply& reply) {
                        if (reply.is_string() && reply.as_string()[0] != '2') my_pause = 0;
                    }
                );
                client->sync_commit();
                if (my_pause == 0)
                    cudaMemcpy(control_state, &my_pause, sizeof(int), cudaMemcpyHostToDevice);
                ncclBroadcast(control_state, control_state, 1, ncclInt, 0, world_comm, curr_stream);
            }
            else
                ncclBroadcast(nullptr, control_state, 1, ncclInt, 0, world_comm, curr_stream);

            int do_validation = 0;
            cudaMemcpy(&do_validation, control_state, sizeof(int), cudaMemcpyDeviceToHost);
            if (!do_validation)
            {
                BOOST_LOG_TRIVIAL(info) << "Rank " << get_rank(DistEngine::auto_find) << " receives continue signal, will continue!";
                break;   
            }
            usleep(500 * 1000);  // wait for 0.5 seconds
        }
    }
}
