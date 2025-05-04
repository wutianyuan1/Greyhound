#include "shm_topo.hpp"
#include "utils.hpp"
using namespace boost::interprocess;


uint64_t hash_nccl_id(char const* str, unsigned long len)
{
    uint64_t hash = 5381;
    for (size_t i = 0; i < len; i++)
        hash = ((hash << 5) + hash) + (int)str[i]; /* hash * 33 + c */
    return hash;
}

Communicator::Communicator(uint64_t addr, int my_rank, uint64_t num_channels_) 
    : num_channels(num_channels_), last_ring_id(0), last_tree_id(0), comm_addr(addr), global_rank(my_rank)
{}

Communicator::Communicator() 
    : num_channels(0), last_ring_id(0), last_tree_id(0), comm_addr(0), global_rank(0)
{}

Communicator::Communicator(const Communicator& other) {
    id_hash = other.id_hash;
    num_channels = other.num_channels;
    last_ring_id = other.last_ring_id;
    last_tree_id = other.last_tree_id;
    comm_addr = other.comm_addr;
    num_devices = other.num_devices;
    global_rank = other.global_rank;
    local_rank = other.local_rank;
    group_rank = other.group_rank;

    // Deep copy of rings and trees arrays
    memcpy(this->rings, other.rings, sizeof(ncclRing) * num_channels);
    memcpy(this->trees, other.trees, sizeof(ncclTree) * num_channels);
    memcpy(this->comm_ops, other.comm_ops, sizeof(int) * MAXFUNCTIONS);
}

Communicator::~Communicator()
{}

void Communicator::add_ring(ncclRing& ring)
{
    memcpy(rings + last_ring_id, &ring, sizeof(struct ncclRing));
    last_ring_id++;
}

void Communicator::add_tree(ncclTree& tree)
{
    memcpy(trees + last_tree_id, &tree, sizeof(struct ncclTree));
    last_tree_id++;
}

void Communicator::serialize(std::ostream& out) {
    out.write(reinterpret_cast<const char*>(&id_hash), sizeof(id_hash));
    out.write(reinterpret_cast<const char*>(&num_channels), sizeof(num_channels));
    out.write(reinterpret_cast<const char*>(&last_ring_id), sizeof(last_ring_id));
    out.write(reinterpret_cast<const char*>(&last_tree_id), sizeof(last_tree_id));
    out.write(reinterpret_cast<const char*>(&comm_addr), sizeof(comm_addr));
    out.write(reinterpret_cast<const char*>(&num_devices), sizeof(num_devices));
    out.write(reinterpret_cast<const char*>(&global_rank), sizeof(global_rank));
    out.write(reinterpret_cast<const char*>(&local_rank), sizeof(local_rank));
    out.write(reinterpret_cast<const char*>(&group_rank), sizeof(group_rank));
    // Serialize rings and trees
    out.write(reinterpret_cast<const char*>(rings), sizeof(ncclRing) * MAXCHANNELS);
    out.write(reinterpret_cast<const char*>(trees), sizeof(ncclTree) * MAXCHANNELS);
}

void Communicator::from_bytes(std::istream& in) {
    in.read(reinterpret_cast<char*>(&id_hash), sizeof(id_hash));
    in.read(reinterpret_cast<char*>(&num_channels), sizeof(num_channels));
    in.read(reinterpret_cast<char*>(&last_ring_id), sizeof(last_ring_id));
    in.read(reinterpret_cast<char*>(&last_tree_id), sizeof(last_tree_id));
    in.read(reinterpret_cast<char*>(&comm_addr), sizeof(comm_addr));
    in.read(reinterpret_cast<char*>(&num_devices), sizeof(num_devices));
    in.read(reinterpret_cast<char*>(&global_rank), sizeof(global_rank));
    in.read(reinterpret_cast<char*>(&local_rank), sizeof(local_rank));
    in.read(reinterpret_cast<char*>(&group_rank), sizeof(group_rank));
    // Deserialize rings and trees
    in.read(reinterpret_cast<char*>(rings), sizeof(ncclRing) * MAXCHANNELS);
    in.read(reinterpret_cast<char*>(trees), sizeof(ncclTree) * MAXCHANNELS);
}

void Communicator::debug_print()
{
    std::stringstream ss;
    ss << "<GPU Connection Info>\n"
        << "  Global Rank:" << global_rank << ", Group Rank: " << group_rank << ", Local Rank: "\
        << local_rank << ", #devs" << num_devices << ", #channels: " << num_channels << "\n";
    for (int i = 0; i < num_channels; i++)
        ss << "  channel[" << i << "]:\n"
            << "    (Ring id=" << rings[i].index << ", prev=" << rings[i].prev << ", next=" << rings[i].next << ")\n"
            << "    (Tree depth=" << trees[i].depth << ", up=" << trees[i].up << ", down=("\
            << trees[i].down[0] << ", " << trees[i].down[1] <<"))\n";
    BOOST_LOG_TRIVIAL(debug) << ss.str();
}




NcclTopoConnection::NcclTopoConnection(int n_ranks)
{
    while (true) {
        try {
            client.connect(get_master_addr(), get_redis_port());
            break;
        }
        catch (std::exception& e) {
            std::cout << "NcclTopoConnection: " << e.what() << std::endl;
            sleep(1);
        }
    }
}

NcclTopoConnection::~NcclTopoConnection()
{
    client.disconnect();
}

std::shared_ptr<Communicator>
NcclTopoConnection::add_comm(Communicator& comm)
{
    auto ret = this->find(comm.comm_addr);
    if (ret) {
        BOOST_LOG_TRIVIAL(debug) << "Communicator " << comm.comm_addr << "is found in cache, will not be repeatly added";
        return ret;
    }
    std::stringstream ss;
    comm.serialize(ss);
    std::string data = ss.str();
    client.set(std::string("Communicator_") + std::to_string(comm.comm_addr), data);
    client.sync_commit();
    return nullptr;
}

std::shared_ptr<Communicator>
NcclTopoConnection::find(uint64_t comm_addr)
{
    auto reply_future = client.get(std::to_string(comm_addr));
    client.sync_commit();
    auto reply = reply_future.get();
    if (reply.is_null()) {
        BOOST_LOG_TRIVIAL(debug) << "RANK" << get_rank(DistEngine::auto_find) << ", Communicator not found";
        return nullptr;
    }
    std::string binary_data = reply.as_string();
    std::stringstream ss(binary_data);
    std::shared_ptr<Communicator> comm(new Communicator());
    comm->from_bytes(ss);
    BOOST_LOG_TRIVIAL(debug) << "RANK" << get_rank(DistEngine::auto_find) << ", Communicator found at " << comm_addr;
    return comm;
}

std::vector<std::shared_ptr<Communicator> >
NcclTopoConnection::find_linked_ranks(const std::shared_ptr<Communicator> comm)
{
    uint64_t ncomms = num_comms();
    std::vector<std::shared_ptr<Communicator> > ret;
    std::vector<std::string> all_keys;
    client.keys("*", [&all_keys](const cpp_redis::reply& reply) {
        if (reply.is_array())
            for (const auto& key : reply.as_array())
                all_keys.push_back(key.as_string());
    });
    client.sync_commit();

    auto is_linked_peer = [&](std::shared_ptr<Communicator> comm_i) {
        // skip the communicators that are not in the same commId
        if (comm_i->id_hash != comm->id_hash)
            return false;
        // skip itself
        if (comm_i->comm_addr == comm->comm_addr)
            return false;
        bool is_linked_peer = false;
        int min_channels = comm->num_channels < comm_i->num_channels ? comm->num_channels : comm_i->num_channels;
        for (int c = 0; c < min_channels; c++)
        {
            // ring links
            if (comm->rings[c].prev == comm_i->group_rank)
                is_linked_peer = true;
            if (comm->rings[c].next == comm_i->group_rank)
                is_linked_peer = true;
            // tree links
            if (comm->trees[c].down[0] == comm_i->group_rank)
                is_linked_peer = true;
            if (comm->trees[c].down[1] == comm_i->group_rank)
                is_linked_peer = true;
            if (comm->trees[c].up == comm_i->group_rank)
                is_linked_peer = true;
        }
        if (is_linked_peer)
            return true;
        return false;    
    };

    // Check for each key if the corresponding Comm is linked with this
    for (const auto& key : all_keys)
    {
        client.get(key, 
            [=, &ret](const cpp_redis::reply& reply) {
                if (reply.is_string())
                {
                    std::stringstream ss(reply.as_string());
                    std::shared_ptr<Communicator> comm_ptr(new Communicator());
                    comm_ptr->from_bytes(ss);
                    if (is_linked_peer(comm_ptr))
                        ret.push_back(comm_ptr);

                }
            }
        );
    }
    client.sync_commit();
    return ret;
}

uint64_t NcclTopoConnection::num_comms()
{
    auto future_reply = client.dbsize();
    client.sync_commit();  // Synchronize the commit to ensure the command is processed
    auto reply = future_reply.get();  // Get the reply from Redis

    if (!reply.is_integer())
        throw std::runtime_error("Failed to get database size");

    return static_cast<uint64_t>(reply.as_integer());
}
