#include "shm_storage.hpp"
#include <vector>
#include <iostream>
#include <csignal>

auto s = new NcclRecordStorage(5, 10);

void signal_handler(int signal)
{
    delete s;
    exit(0);
}

int main() {
    std::signal(SIGINT, signal_handler);
    
    for (uint64_t i = 0; i < 10; i++)
        s->addRecord(std::vector<uint64_t>({i,i,i,i,i}));
    while (true) {
        uint64_t n;
        std::cin >> n;
        s->addRecord(std::vector<uint64_t>(5, n));
    }
    return 0;
}
