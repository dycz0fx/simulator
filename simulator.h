#include <iostream>
#include <vector>
#include <queue>
#include <unordered_map>
#include <string>
#include <time.h>
#include <boost/functional/hash.hpp>

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

class CompDevice : public Device
{
public:
    enum CompDevType {
        LOC_PROC,   //CPU
        TOC_PROC,   //GPU
    };
    CompDevType comp_type;
    CompDevice(std::string name, CompDevType comp_type, int node_id, int socket_id, int device_id);
};

class MemDevice : public Device
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
    MemDevice(std::string name, MemDevType mem_type, int node_id, int socket_id, int device_id);
};

class CommDevice : public Device
{
public:
    enum CommDevType {
        MEMBUS_COMM,
        UPI_IN_COMM,
        UPI_OUT_COMM,
        NIC_IN_COMM,
        NIC_OUT_COMM,
        PCI_TO_HOST_COMM,
        PCI_TO_DEV_COMM,
        NVLINK_COMM,
    };
    CommDevType comm_type;
    float latency;
    float bandwidth;
    CommDevice(std::string name, CommDevType comm_type, int node_id, int socket_id, int device_id, float latency, float bandwidth);
};

class MachineModel {
public:
  virtual ~MachineModel() = default;
  virtual int get_version() const = 0;
  virtual CompDevice *get_cpu(int device_id) const = 0;
  virtual MemDevice *get_sys_mem(int devicd_id) const = 0;
  virtual CompDevice *get_gpu(int device_id) const = 0;
  virtual MemDevice *get_gpu_fb_mem(int devicd_id) const = 0;
  virtual int get_num_gpus() const = 0;
  virtual float get_intra_node_gpu_bandwidth() const = 0;
  virtual float get_inter_node_gpu_bandwidth() const = 0;
  virtual std::vector<CommDevice *> get_comm_path(MemDevice *src_mem, MemDevice *tar_mem) const = 0;
  virtual std::string to_string() const = 0;
  int version;
  size_t default_seg_size;
  int max_num_segs;
  float realm_comm_overhead;
};

class SimpleMachineModel : public MachineModel {
public:
  SimpleMachineModel(int num_nodes, int num_cpus_per_node, int num_gpus_per_node);
  ~SimpleMachineModel();
  int get_version() const;
  CompDevice *get_cpu(int device_id) const;
  MemDevice *get_sys_mem(int socket_id) const;
  CompDevice *get_gpu(int device_id) const;
  MemDevice *get_gpu_fb_mem(int devicd_id) const;
  int get_num_gpus() const;
  float get_intra_node_gpu_bandwidth() const;
  float get_inter_node_gpu_bandwidth() const;
  std::vector<CommDevice *> get_comm_path(MemDevice *src_mem, MemDevice *tar_mem) const;
  std::string to_string() const;
private:
  int num_nodes;
  int num_cpus_per_node;
  int num_gpus_per_node;
  int num_gpus;
  float inter_gpu_bandwidth;
  float inter_node_bandwidth;
  float gpu_dram_bandwidth;
  std::map<int, CompDevice*> id_to_cpu;
  std::map<int, MemDevice*> id_to_sys_mem;
  std::map<int, CompDevice*> id_to_gpu;
  std::map<int, MemDevice*> id_to_gpu_fb_mem;
  std::map<int, CommDevice*> id_to_gputodram_comm_device;
  std::map<int, CommDevice*> id_to_dramtogpu_comm_device;
  std::map<size_t, CommDevice*> ids_to_inter_gpu_comm_device;
  std::map<size_t, CommDevice*> ids_to_inter_node_comm_device;
};

/**
 * An enhanced machine model supports the following features:
 * 1. Customize the machine model with a configuration file.
 * 2. Support socket-level simulation.
 * 3. Simulate congestions on a communication device. In this machine model, some communication 
 *    devices, such as NIC_IN and NIC_OUT, represent the communication ports instead of the links 
 *    in the simple machine model. In this way, for example, concurrent inter-node communications 
 *    from node A to node B and from node A to node C share the same NIC_OUT device on node A, 
 *    which simulates the slowdown of concurrent communications when transferring big messages.
 * 4. When passing big messages, the messages usually are divided into segments and transferred 
 *    one-by-one to overlap the communications on different devices. This machine model can 
 *    simulate this kind of pipelining.
 */ 
class EnhancedMachineModel : public MachineModel {
public:
    enum NicDistribution {
      PER_NODE,
      PER_SOCKET,
    };
    EnhancedMachineModel(std::string file);
    ~EnhancedMachineModel();
    int get_version() const;
    CompDevice *get_cpu(int device_id) const;
    CompDevice *get_cpu(int socket_id, int local_id) const;
    CompDevice *get_gpu(int device_id) const;
    CompDevice *get_gpu(int socket_id, int local_id) const;
    MemDevice *get_sys_mem(int socket_id) const;
    MemDevice *get_z_copy_mem(int socket_id) const;
    MemDevice *get_gpu_fb_mem(int device_id) const;
    MemDevice *get_gpu_fb_mem(int socket_id, int local_id) const;
    CommDevice *get_nvlink(MemDevice *src_mem, MemDevice *tar_mem) const;
    int get_num_gpus() const;
    float get_intra_node_gpu_bandwidth() const;
    float get_inter_node_gpu_bandwidth() const;
    std::vector<CommDevice *> get_comm_path(MemDevice *src_mem, MemDevice *tar_mem) const;
    std::string to_string() const;
private:
    int num_nodes;
    int num_sockets_per_node;
    int num_cpus_per_socket;
    int num_gpus_per_socket;
    int num_sockets;
    int num_cpus;
    int num_gpus;
    int num_nvlinks_per_node;
    float membus_latency;
    float membus_bandwidth;
    float upi_latency;
    float upi_bandwidth;
    float nic_latency;
    float nic_bandwidth;
    NicDistribution nic_distribution;
    float pci_latency;
    float pci_bandwidth;
    float nvlink_latency;
    float nvlink_bandwidth;
    std::vector<CommDevice::CommDevType> intra_socket_sys_mem_to_sys_mem;
    std::vector<CommDevice::CommDevType> inter_socket_sys_mem_to_sys_mem;
    std::vector<CommDevice::CommDevType> inter_node_sys_mem_to_sys_mem;
    std::vector<CommDevice::CommDevType> intra_socket_sys_mem_to_gpu_fb_mem;
    std::vector<CommDevice::CommDevType> inter_socket_sys_mem_to_gpu_fb_mem;
    std::vector<CommDevice::CommDevType> inter_node_sys_mem_to_gpu_fb_mem;
    std::vector<CommDevice::CommDevType> intra_socket_gpu_fb_mem_to_sys_mem;
    std::vector<CommDevice::CommDevType> inter_socket_gpu_fb_mem_to_sys_mem;
    std::vector<CommDevice::CommDevType> inter_node_gpu_fb_mem_to_sys_mem;
    std::vector<CommDevice::CommDevType> intra_socket_gpu_fb_mem_to_gpu_fb_mem;
    std::vector<CommDevice::CommDevType> inter_socket_gpu_fb_mem_to_gpu_fb_mem;
    std::vector<CommDevice::CommDevType> inter_node_gpu_fb_mem_to_gpu_fb_mem;
    std::vector<std::vector<CompDevice *> > cpus;   // socket_id, local_id
    std::vector<std::vector<CompDevice *> > gpus;   // socket_id, local_id
    std::vector<MemDevice *> sys_mems;             // socket_id
    std::vector<MemDevice *> z_copy_mems;          // socket_id
    std::vector<std::vector<MemDevice *> > gpu_fb_mems;     // socket_id, local_id
    std::vector<CommDevice *> membuses;            // socket_id
    std::vector<CommDevice *> upi_ins;             // socket_id
    std::vector<CommDevice *> upi_outs;            // socket_id
    std::vector<CommDevice *> nic_ins;             // socket_id
    std::vector<CommDevice *> nic_outs;            // socket_id
    std::vector<CommDevice *> pcis_to_host;             // from gpu to main memory, socket_id
    std::vector<CommDevice *> pcis_to_device;            // from main memory to gpu, socket_id
    std::vector<std::vector<CommDevice *> > nvlinks;    // node_id, local_id
    std::unordered_map<size_t, CommDevice *> mem_to_nvlink;
    // set up communication paths from a config file
    void set_comm_path(std::vector<CommDevice::CommDevType> &comm_path, std::string device_str);
    void add_cpus();
    void add_gpus();
    void add_membuses(float latency, float bandwidth);
    void add_upis(float latency, float bandwidth);
    void add_nics(float latency, float bandwidth, NicDistribution nic_distribution);
    void add_pcis(float latency, float bandwidth);
    void add_nvlinks(float latency, float bandwidth);
    // attach a nvlink communication device to a pair of GPU framebuffer memories
    void attach_nvlink(MemDevice *src_mem, MemDevice *tar_mem, CommDevice *comm);
    // return a list of specific communication devices based on the descriptions of a communication path
    void add_comm_path(std::vector<CommDevice::CommDevType> const &comm_device_list, MemDevice *src_mem, MemDevice *tar_mem, std::vector<CommDevice *> &ret) const;
};


class Task
{
public:
    Task(std::string name, Device *device);
    size_t id;
    static size_t cur_id;
    std::string name;
    Device *device;
    float ready_time;
    std::vector<Task *> next_tasks;
    int counter;
    bool is_main;   // whether is a part of main loop
    void add_next_task(Task *task);
    virtual float cost() const = 0 ;
    virtual std::string to_string() const = 0;
};

class CompTask : public Task
{
public:
    CompTask(std::string name, CompDevice *comp_deivce, float run_time, MemDevice *mem_device);
    MemDevice *mem;
    float run_time;
    float cost() const;
    std::string to_string() const;
};

class CommTask : public Task
{
public:
    CommTask(std::string name, CommDevice *comm_device, size_t message_size);
    size_t message_size;
    float cost() const;
    std::string to_string() const;
};

class TaskCompare {
public:
    bool operator() (Task *lhs, Task *rhs) {
        if (lhs->ready_time == rhs->ready_time) {
            return lhs->id > rhs->id;
        }
        return lhs->ready_time > rhs->ready_time;
    }
};

class Simulator
{
private:
    std::priority_queue<Task *, std::vector<Task *>, TaskCompare> ready_queue;
public:
    MachineModel *machine;
    Simulator(MachineModel *machine);
    Task *new_comp_task(std::string name, CompDevice *comp_device, float run_time, MemDevice *mem_device);
    void new_comm_task(Task *src_task, Task *tar_task, size_t message_size);
    void enter_ready_queue(Task *task);
    void add_dependency(std::vector<Task *> prev_tasks, Task *cur_task);
    void add_dependency(Task *prev_task, Task *cur_task);
    void simulate();
};

inline bool starts_with(std::string s, std::string sub){
    return s.find(sub) == 0 ? 1 : 0;
}
