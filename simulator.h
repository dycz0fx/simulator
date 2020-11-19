#include <iostream>
#include <vector>
#include <queue>
#include <unordered_map>
#include <string>
#include <time.h>
#include <boost/functional/hash.hpp>

#define SEG_SIZE 262144
#define MAX_NUM_SEGS 1
#define LEGION_OVERHEAD 0
class Device
{
public:
    enum DeviceType {
        DEVICE_COMP,
        DEVICE_MEM,
        DEVICE_COMM,
    };
    Device(std::string name, DeviceType type, int node_id, int socket_id, int device_id);
    std::string name;
    DeviceType type;
    int node_id;
    int socket_id;
    int device_id;
};

class Comp_device : public Device
{
public:
    enum CompDevType {
        LOC_PROC,   //CPU
        TOC_PROC,   //GPU
    };
    CompDevType comp_type;
    Comp_device(std::string name, CompDevType comp_type, int node_id, int socket_id, int device_id);
};

class Mem_device : public Device
{
public:
    enum MemDevType {
        //DISK_MEM,     // Disk memory on a single node
        //HDF_MEM,      // HDF framebuffer memory for a single GPU
        //FILE_MEM,     // File memory on a single node
        //GLOBAL_MEM,   // RDMA addressable memory when running with GASNet
        SYSTEM_MEM,     // DRAM on a single node
        //REGDMA_MEM,   // Pinned memory on a single node
        //SOCKET_MEM,   // A memory associated with a single socket
        Z_COPY_MEM,     // Zero-copy memory betweeen CPU DRAM and all GPUs on a single node
        GPU_FB_MEM,     // GPU framebuffer memory for a single GPU
    };
    MemDevType mem_type;
    Mem_device(std::string name, MemDevType mem_type, int node_id, int socket_id, int device_id);
};

class Comm_device : public Device
{
public:
    enum CommDevType {
        MEMBUS_COMM,
        UPI_IN_COMM,
        UPI_OUT_COMM,
        NIC_IN_COMM,
        NIC_OUT_COMM,
        PCI_IN_COMM,
        PCI_OUT_COMM,
        NVLINK_COMM,
    };
    CommDevType comm_type;
    float latency;
    float bandwidth;
    Comm_device(std::string name, CommDevType comm_type, int node_id, int socket_id, int device_id, float latency, float bandwidth);
};


class Machine
{
private:
    int num_nodes;
    int num_sockets_per_node;
    int num_cpus_per_socket;
    int num_gpus_per_socket;
    int num_sockets;
    int num_cpus;
    int num_gpus;
    int num_nvlinks_per_node;
    std::vector<std::vector<Comp_device *>> cpus;   // socket_id, local_id
    std::vector<std::vector<Comp_device *>> gpus;   // socket_id, local_id
    std::vector<Mem_device *> sys_mems;             // socket_id
    std::vector<Mem_device *> z_copy_mems;          // socket_id
    std::vector<std::vector<Mem_device *>> gpu_fb_mems;     // socket_id, local_id
    std::vector<Comm_device *> membuses;            // socket_id
    std::vector<Comm_device *> upi_ins;             // socket_id
    std::vector<Comm_device *> upi_outs;            // socket_id
    std::vector<Comm_device *> nic_ins;             // socket_id
    std::vector<Comm_device *> nic_outs;            // socket_id
    std::vector<Comm_device *> pci_ins;             // from gpu to main memory, socket_id
    std::vector<Comm_device *> pci_outs;            // from main memory to gpu, socket_id
    std::vector<std::vector<Comm_device *>> nvlinks;    // node_id, local_id
    std::unordered_map<std::pair<int, int>, Comm_device *, boost::hash<std::pair<int, int>>> mem_to_nvlink;
    void add_cpus();
    void add_gpus();

public:
    Machine(int num_nodes, int num_sockets_per_node, int num_cpus_per_socket, int num_gpus_per_socket);
    Comp_device *get_cpu(int device_id);
    Comp_device *get_cpu(int socket_id, int local_id);
    Comp_device *get_gpu(int device_id);
    Comp_device *get_gpu(int socket_id, int local_id);
    Mem_device *get_sys_mem(int socket_id);
    Mem_device *get_z_copy_mem(int socket_id);
    Mem_device *get_gpu_fb_mem(int device_id);
    Mem_device *get_gpu_fb_mem(int socket_id, int local_id);
    void add_membuses(float latency, float bandwidth);
    void add_upis(float latency, float bandwidth);
    void add_nics(float latency, float bandwidth);
    void add_pcis(float latency, float bandwidth);
    void add_nvlinks(int num_nvlinks_per_node, float latency, float bandwidth);
    void attach_nvlink(Mem_device *src_mem, Mem_device *tar_mem, Comm_device *comm);    // nvlinks between GPUs
    std::vector<Comm_device *> get_comm_path(Mem_device *src_mem, Mem_device *tar_mem);
    std::string to_string();
};

class Task
{
public:
    Task(std::string name, Device *device);
    std::string name;
    Device *device;
    float ready_time;
    std::vector<Task *> next_tasks;
    int counter;
    bool is_main;       // whether is a part of main loop
    void add_next_task(Task *task);
    virtual std::string to_string();
};

class Comp_task : public Task
{
public:
    Comp_task(std::string name, Comp_device *comp_deivce, float run_time, Mem_device *mem_device);
    Mem_device *mem;
    float run_time;
    float cost();
    std::string to_string();
};

class Comm_task : public Task
{
public:
    Comm_task(std::string name, Comm_device *comm_device, int message_size);
    int message_size;
    float cost();
    std::string to_string();
};

class TaskCompare {
public:
    bool operator() (Task *lhs, Task *rhs) {
        if (lhs->ready_time == rhs->ready_time) {
            return rand() % 2;
        }
        return lhs->ready_time > rhs->ready_time;
    }
};

class Simulator
{
private:
    std::priority_queue<Task *, std::vector<Task *>, TaskCompare> ready_queue;
public:
    Machine *machine;
    Simulator(Machine *machine);
    Task *new_comp_task(std::string name, Comp_device *comp_device, float run_time, Mem_device *mem_device);
    void new_comm_task(Task *src_task, Task *tar_task, int message_size);
    void enter_ready_queue(Task *task);
    void add_dependency(std::vector<Task *> prev_tasks, Task *cur_task);
    void add_dependency(Task *prev_task, Task *cur_task);
    void simulate();
};

inline bool starts_with(std::string s, std::string sub){
    return s.find(sub) == 0 ? 1 : 0;
}
