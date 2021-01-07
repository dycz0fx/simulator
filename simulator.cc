#include "simulator.h"
#include <chrono>
#include <algorithm> 

using std::string;
using std::vector;
using std::cout;
using std::endl;
using std::to_string;
using std::unordered_map;
using std::max;
using std::pair;

// class Device
Device::Device(string name, DeviceType type, int node_id, int socket_id, int device_id)
: name(name), type(type), node_id(node_id), socket_id(socket_id), device_id(device_id)
{
}

// class Comp_device
Comp_device::Comp_device(std::string name, CompDevType comp_type, int node_id, int socket_id, int device_id)
: Device(name, Device::DEVICE_COMP, node_id, socket_id, device_id), comp_type(comp_type)
{
}

// class Mem_device
Mem_device::Mem_device(std::string name, MemDevType mem_type, int node_id, int socket_id, int device_id)
: Device(name, Device::DEVICE_MEM, node_id, socket_id, device_id), mem_type(mem_type)
{
}

// class Comm_device
Comm_device::Comm_device(std::string name, CommDevType comm_type, int node_id, int socket_id, int device_id, float latency, float bandwidth)
: Device(name, Device::DEVICE_COMM, node_id, socket_id, device_id), comm_type(comm_type), latency(latency), bandwidth(bandwidth)
{
}

// class Machine_Sherlock
Machine_Sherlock::Machine_Sherlock(int num_nodes, int num_sockets_per_node, int num_cpus_per_socket, int num_gpus_per_socket) 
: num_nodes(num_nodes), num_sockets_per_node(num_sockets_per_node), num_cpus_per_socket(num_cpus_per_socket), num_gpus_per_socket(num_gpus_per_socket)
{
    num_sockets = num_nodes * num_sockets_per_node;
    num_cpus = num_sockets * num_cpus_per_socket;
    num_gpus = num_sockets * num_gpus_per_socket;    
    mem_to_nvlink.clear();
    add_cpus();
    add_gpus();
}

void Machine_Sherlock::add_cpus()
{
    for (int i = 0; i < num_nodes; i++) {
        int node_id = i;
        for (int j = 0; j < num_sockets_per_node; j++) {
            int socket_id = i * num_sockets_per_node + j;
            int device_id = socket_id;
            // add system memory
            string sys_mem_name = "SYSTEM_MEM " + std::to_string(device_id);
            Mem_device *sys_mem = new Mem_device(sys_mem_name, Mem_device::SYSTEM_MEM, node_id, socket_id, device_id);
            sys_mems.push_back(sys_mem);
            // add cpus
            cpus.push_back({});
            for (int k = 0; k < num_cpus_per_socket; k++) {
                device_id = socket_id * num_cpus_per_socket + k;
                string cpu_name = "CPU " + std::to_string(device_id);
                cpus[socket_id].emplace_back(new Comp_device(cpu_name, Comp_device::LOC_PROC, node_id, socket_id, device_id));
            }
        }
    }
}

void Machine_Sherlock::add_gpus()
{
    for (int i = 0; i < num_nodes; i++) {
        int node_id = i;
        for (int j = 0; j < num_sockets_per_node; j++) {
            int socket_id = i * num_sockets_per_node + j;
            int device_id = socket_id;
            // add zero copy memory
            string z_copy_mem_name = "Z_COPY_MEM " + std::to_string(device_id);
            Mem_device *z_copy_mem = new Mem_device(z_copy_mem_name, Mem_device::Z_COPY_MEM, node_id, socket_id, device_id);
            z_copy_mems.push_back(z_copy_mem);
            // add gpus and gpu framebuffer memories
            gpus.push_back({});
            gpu_fb_mems.push_back({});
            for (int k = 0; k < num_gpus_per_socket; k++) {
                device_id = socket_id * num_gpus_per_socket + k;
                string gpu_name = "GPU " + std::to_string(device_id);
                gpus[socket_id].emplace_back(new Comp_device(gpu_name, Comp_device::TOC_PROC, node_id, socket_id, device_id));
                string gpu_mem_name = "GPU_FB_MEM " + std::to_string(device_id);
                Mem_device *gpu_mem = new Mem_device(gpu_mem_name, Mem_device::GPU_FB_MEM, node_id, socket_id, device_id);
                gpu_fb_mems[socket_id].push_back({gpu_mem});
            }
        }
    }
}

void Machine_Sherlock::add_membuses(float latency, float bandwidth)
{
    for (int i = 0; i < num_nodes; i++) {
        int node_id = i;
        for (int j = 0; j < num_sockets_per_node; j++) {
            int socket_id = i * num_sockets_per_node + j;
            int device_id = socket_id;
            string membus_name = "MEMBUS " + std::to_string(device_id);
            Comm_device *membus = new Comm_device(membus_name, Comm_device::MEMBUS_COMM, node_id, socket_id, device_id, latency, bandwidth);
            membuses.push_back(membus);
        }
    }    
}

void Machine_Sherlock::add_upis(float latency, float bandwidth)
{
    for (int i = 0; i < num_nodes; i++) {
        int node_id = i;
        for (int j = 0; j < num_sockets_per_node; j++) {
            int socket_id = i * num_sockets_per_node + j;
            int device_id = socket_id;
            string upi_in_name = "UPI_IN " + std::to_string(device_id);
            Comm_device *upi_in = new Comm_device(upi_in_name, Comm_device::UPI_IN_COMM, node_id, socket_id, device_id, latency, bandwidth);
            upi_ins.push_back(upi_in);
            string upi_out_name = "UPI_OUT " + std::to_string(device_id);
            Comm_device *upi_out = new Comm_device(upi_out_name, Comm_device::UPI_OUT_COMM, node_id, socket_id, device_id, latency, bandwidth);
            upi_outs.push_back(upi_out);
        }
    }    
}

void Machine_Sherlock::add_nics(float latency, float bandwidth)
{
    for (int i = 0; i < num_nodes; i++) {
        int node_id = i;
        for (int j = 0; j < num_sockets_per_node; j++) {
            int socket_id = i * num_sockets_per_node + j;
            int device_id = socket_id;
            string nic_in_name = "NIC_IN " + std::to_string(device_id);
            Comm_device *nic_in = new Comm_device(nic_in_name, Comm_device::UPI_IN_COMM, node_id, socket_id, device_id, latency, bandwidth);
            nic_ins.push_back(nic_in);
            string nic_out_name = "NIC_OUT " + std::to_string(device_id);
            Comm_device *nic_out = new Comm_device(nic_out_name, Comm_device::UPI_OUT_COMM, node_id, socket_id, device_id, latency, bandwidth);
            nic_outs.push_back(nic_out);
        }
    }    
}

void Machine_Sherlock::add_pcis(float latency, float bandwidth)
{
    for (int i = 0; i < num_nodes; i++) {
        int node_id = i;
        for (int j = 0; j < num_sockets_per_node; j++) {
            int socket_id = i * num_sockets_per_node + j;
            int device_id = socket_id;
            string pci_in_name = "PCI_IN " + std::to_string(device_id);    // pcie to memory
            Comm_device *pci_in = new Comm_device(pci_in_name, Comm_device::PCI_IN_COMM, node_id, socket_id, socket_id, latency, bandwidth);
            pci_ins.push_back(pci_in);
            string pci_out_name = "PCI_OUT " + std::to_string(device_id);  // memory to pcie
            Comm_device *pci_out = new Comm_device(pci_out_name, Comm_device::PCI_OUT_COMM, node_id, socket_id, socket_id, latency, bandwidth);
            pci_outs.push_back(pci_out);
        }
    }    
}

void Machine_Sherlock::add_nvlinks(int num_nvlinks_per_node, float latency, float bandwidth)
{
    assert(num_gpus_per_socket == 2);
    assert(num_sockets_per_node == 2);
    this->num_nvlinks_per_node = num_nvlinks_per_node;
    for (int i = 0; i < num_nodes; i++) {
        int node_id = i;
        int socket_id = i * num_sockets_per_node;
        nvlinks.push_back({});
        for (int j = 0; j < num_nvlinks_per_node * 2; j++) {
            int nvlink_id = node_id * num_nvlinks_per_node * 2 + j;
            string nvlink_name = "NVLINK " + std::to_string(nvlink_id);
            if (j < 8) {
                nvlinks[i].push_back(new Comm_device(nvlink_name, Comm_device::NVLINK_COMM, node_id, socket_id, nvlink_id, latency, bandwidth));
            }
            else {
                nvlinks[i].push_back(new Comm_device(nvlink_name, Comm_device::NVLINK_COMM, node_id, socket_id, nvlink_id, latency, bandwidth * 2));
            }
        }

        for (int j = 0; j < num_sockets_per_node; j++) {
            for (int k = 0; k < num_gpus_per_socket; k++) {
                socket_id = i * num_sockets_per_node + j;
                int local_gpu_fb_mem_id = j * num_gpus_per_socket + k;
                Mem_device *gpu_fb_mem = gpu_fb_mems[socket_id][k];
                switch (local_gpu_fb_mem_id)
                {
                case 0:
                    attach_nvlink(gpu_fb_mem, gpu_fb_mems[socket_id][1], nvlinks[i][0]);
                    attach_nvlink(gpu_fb_mem, gpu_fb_mems[socket_id+1][0], nvlinks[i][7]);
                    attach_nvlink(gpu_fb_mem, gpu_fb_mems[socket_id+1][1], nvlinks[i][8]);
                    break;
                case 1:
                    attach_nvlink(gpu_fb_mem, gpu_fb_mems[socket_id][0], nvlinks[i][1]);
                    attach_nvlink(gpu_fb_mem, gpu_fb_mems[socket_id+1][1], nvlinks[i][2]);
                    attach_nvlink(gpu_fb_mem, gpu_fb_mems[socket_id+1][0], nvlinks[i][11]);
                    break;
                case 2:
                    attach_nvlink(gpu_fb_mem, gpu_fb_mems[socket_id][1], nvlinks[i][5]);
                    attach_nvlink(gpu_fb_mem, gpu_fb_mems[socket_id-1][0], nvlinks[i][6]);
                    attach_nvlink(gpu_fb_mem, gpu_fb_mems[socket_id-1][1], nvlinks[i][10]);
                    break;
                case 3:
                    attach_nvlink(gpu_fb_mem, gpu_fb_mems[socket_id-1][1], nvlinks[i][3]);
                    attach_nvlink(gpu_fb_mem, gpu_fb_mems[socket_id][0], nvlinks[i][4]);
                    attach_nvlink(gpu_fb_mem, gpu_fb_mems[socket_id-1][0], nvlinks[i][9]);
                    break;
                default:
                    break;
                }
            }
        }
    }
}

Comp_device *Machine_Sherlock::get_cpu(int device_id)
{
    return cpus[device_id / num_cpus_per_socket][device_id % num_cpus_per_socket];
}

Comp_device *Machine_Sherlock::get_cpu(int socket_id, int local_id)
{
    return cpus[socket_id][local_id];
} 

Comp_device *Machine_Sherlock::get_gpu(int device_id)
{
    return gpus[device_id / num_gpus_per_socket][device_id % num_gpus_per_socket];
}

Comp_device *Machine_Sherlock::get_gpu(int socket_id, int local_id)
{
    return gpus[socket_id][local_id];
}

Mem_device *Machine_Sherlock::get_sys_mem(int socket_id)
{
    return sys_mems[socket_id];
}

Mem_device *Machine_Sherlock::get_z_copy_mem(int socket_id)
{
    return z_copy_mems[socket_id];
}

Mem_device *Machine_Sherlock::get_gpu_fb_mem(int device_id)
{
    return gpu_fb_mems[device_id / num_gpus_per_socket][device_id % num_gpus_per_socket];
}

Mem_device *Machine_Sherlock::get_gpu_fb_mem(int socket_id, int local_id)
{
    return gpu_fb_mems[socket_id][local_id];
}

void Machine_Sherlock::attach_nvlink(Mem_device *src_mem, Mem_device *tar_mem, Comm_device *comm) 
{
    assert(comm->comm_type == Comm_device::NVLINK_COMM);
    pair<int, int> key(src_mem->device_id, tar_mem->device_id);
    if (mem_to_nvlink.find(key) == mem_to_nvlink.end()) {
        mem_to_nvlink[key] = comm;
    }
}

vector<Comm_device *> Machine_Sherlock::get_comm_path(Mem_device *src_mem, Mem_device *tar_mem)
{
    vector<Comm_device *> ret;
    // on the same memory
    if (src_mem->mem_type == tar_mem->mem_type and src_mem->device_id == tar_mem->device_id) {
        return ret;
    }
    if ((src_mem->mem_type == Mem_device::SYSTEM_MEM and tar_mem->mem_type == Mem_device::SYSTEM_MEM) or
        (src_mem->mem_type == Mem_device::Z_COPY_MEM and tar_mem->mem_type == Mem_device::Z_COPY_MEM) or
        (src_mem->mem_type == Mem_device::Z_COPY_MEM and tar_mem->mem_type == Mem_device::SYSTEM_MEM) or
        (src_mem->mem_type == Mem_device::SYSTEM_MEM and tar_mem->mem_type == Mem_device::Z_COPY_MEM)) {
        // on the same socket
        if (src_mem->socket_id == tar_mem->socket_id) {
            return ret;
        }
        // on the same node
        else if (src_mem->node_id == tar_mem->node_id) {
            ret.emplace_back(upi_outs[src_mem->socket_id]);
            ret.emplace_back(upi_ins[tar_mem->socket_id]);
        }
        // on different nodes
        else {
            ret.emplace_back(nic_outs[src_mem->socket_id]);
            ret.emplace_back(nic_ins[tar_mem->socket_id]);
            ret.emplace_back(membuses[tar_mem->socket_id]);
        }
    }
    else if (src_mem->mem_type == Mem_device::GPU_FB_MEM and tar_mem->mem_type == Mem_device::GPU_FB_MEM) {
        // on the same node
        if (src_mem->node_id == tar_mem->node_id) {
            pair<int, int> key(src_mem->device_id, tar_mem->device_id);
            ret.emplace_back(mem_to_nvlink[key]);
        }
        // on different nodes
        else {
            ret.emplace_back(pci_ins[src_mem->socket_id]);
            ret.emplace_back(nic_outs[src_mem->socket_id]);
            ret.emplace_back(nic_ins[tar_mem->socket_id]);
            ret.emplace_back(pci_outs[tar_mem->socket_id]);
        }
    }
    else if (src_mem->mem_type == Mem_device::SYSTEM_MEM and tar_mem->mem_type == Mem_device::GPU_FB_MEM) {
        // on the same socket
        if (src_mem->socket_id == tar_mem->socket_id) {
            ret.emplace_back(membuses[tar_mem->socket_id]);
            ret.emplace_back(pci_outs[tar_mem->socket_id]);
        }
        // on the same node
        else if (src_mem->node_id == tar_mem->node_id) {
            ret.emplace_back(membuses[src_mem->socket_id]);
            ret.emplace_back(upi_outs[src_mem->socket_id]);
            ret.emplace_back(upi_ins[tar_mem->socket_id]);
            ret.emplace_back(pci_outs[tar_mem->socket_id]);
        }
        // on different nodes
        else {
            //ret.emplace_back(membuses[src_mem->socket_id]);
            ret.emplace_back(nic_outs[src_mem->socket_id]);
            ret.emplace_back(nic_ins[tar_mem->socket_id]);
            ret.emplace_back(pci_outs[tar_mem->socket_id]);            
        }
    }
    else if (src_mem->mem_type == Mem_device::GPU_FB_MEM and tar_mem->mem_type == Mem_device::SYSTEM_MEM) {
        // on the same socket
        if (src_mem->socket_id == tar_mem->socket_id) {
            ret.emplace_back(pci_ins[tar_mem->socket_id]);
            ret.emplace_back(membuses[tar_mem->socket_id]);
        }
        // on the same node
        if (src_mem->node_id == tar_mem->node_id) {
            ret.emplace_back(pci_ins[src_mem->socket_id]);
            ret.emplace_back(upi_outs[src_mem->socket_id]);
            ret.emplace_back(upi_ins[tar_mem->socket_id]);
            ret.emplace_back(membuses[tar_mem->socket_id]);
        }
        // on different nodes
        else {
            ret.emplace_back(pci_ins[src_mem->socket_id]);  
            ret.emplace_back(nic_outs[src_mem->socket_id]);
            ret.emplace_back(nic_ins[tar_mem->socket_id]);
            ret.emplace_back(membuses[tar_mem->socket_id]);
        }
    }
    else if (src_mem->mem_type == Mem_device::Z_COPY_MEM and tar_mem->mem_type == Mem_device::GPU_FB_MEM) {
        // on the same socket
        if (src_mem->socket_id == tar_mem->socket_id) {
            ret.emplace_back(pci_outs[tar_mem->socket_id]);
        }
        // on the same node
        else if (src_mem->node_id == tar_mem->node_id) {
            ret.emplace_back(upi_outs[src_mem->socket_id]);
            ret.emplace_back(upi_ins[tar_mem->socket_id]);
            ret.emplace_back(pci_outs[tar_mem->socket_id]);
        }
        // on different nodes
        else {
            ret.emplace_back(nic_outs[src_mem->socket_id]);
            ret.emplace_back(nic_ins[tar_mem->socket_id]);
            ret.emplace_back(pci_outs[tar_mem->socket_id]);            
        }
    }
    else if (src_mem->mem_type == Mem_device::GPU_FB_MEM and tar_mem->mem_type == Mem_device::Z_COPY_MEM) {
        // on the same socket
        if (src_mem->socket_id == tar_mem->socket_id) {
            ret.emplace_back(pci_ins[tar_mem->socket_id]);
        }
        // on the same node
        if (src_mem->node_id == tar_mem->node_id) {
            ret.emplace_back(pci_ins[src_mem->socket_id]);
            ret.emplace_back(upi_outs[src_mem->socket_id]);
            ret.emplace_back(upi_ins[tar_mem->socket_id]);
        }
        // on different nodes
        else {
            ret.emplace_back(pci_ins[src_mem->socket_id]);            
            ret.emplace_back(nic_outs[src_mem->socket_id]);
            ret.emplace_back(nic_ins[tar_mem->socket_id]);
            ret.emplace_back(membuses[tar_mem->socket_id]);
        }
    }
    else {
        cout << "No path found between " << src_mem->name << " and " << tar_mem->name << endl;
    }

    return ret;
}

string Machine_Sherlock::to_string()
{
    string s;
    for (int i = 0; i < num_nodes; i++) {
        int node_id = i;
        s += "==========================================\n";
        s += "Node " + std::to_string(node_id) + '\n';
        for (int j = 0; j < num_sockets_per_node; j++) {
            s += "------------------------------------------\n";
            int socket_id = i * num_sockets_per_node + j;
            s += "Socket " + std::to_string(socket_id) + '\n';
            s += "COMP: \n";
            for (int k = 0; k < num_cpus_per_socket; k++) {
                s += cpus[socket_id][k]->name + '\n';
            }
            for (int k = 0; k < num_gpus_per_socket; k++) {
                s += gpus[socket_id][k]->name + '\n';
            }
            s += '\n';
            s += "MEM: \n";
            s += sys_mems[socket_id]->name + '\n';
            s += z_copy_mems[socket_id]->name + '\n';
            for (int k = 0; k < num_gpus_per_socket; k++) {
                s += gpu_fb_mems[socket_id][k]->name + '\n';
            }
            s += '\n';
            s += "COMM: \n";
            s += membuses[socket_id]->name + '\n';
            s += upi_ins[socket_id]->name + '\n';
            s += upi_outs[socket_id]->name + '\n';
            s += nic_ins[socket_id]->name + '\n';
            s += nic_outs[socket_id]->name + '\n';
            s += pci_ins[socket_id]->name + '\n';
            s += pci_outs[socket_id]->name + '\n';
        }
        s += "------------------------------------------\n";
        for (int j = 0; j < num_nvlinks_per_node * 2; j++) {
            s += nvlinks[node_id][j]->name + '\n';
        }
    }
    return s;
}

// class Machine_Old
Machine_Old::Machine_Old(int num_nodes, int num_cpus_per_node, int num_gpus_per_node)
: num_nodes(num_nodes), num_cpus_per_node(num_cpus_per_node), num_gpus_per_node(num_gpus_per_node)
{
    float inter_gpu_bandwidth = 20 * 1024 * 1024.0f; /* B/ms*/
    float inter_node_bandwidth = 12 * 1024 * 1024.0f / num_nodes; /* B/ms*/
    float gpu_dram_bandwidth = 16 * 1024 * 1024.0f; /* B/ms*/

    total_num_cpus = num_cpus_per_node * num_nodes;
    total_num_gpus = num_gpus_per_node * num_nodes;
    
    // Create CPU compute device
    for (int i = 0; i < num_nodes; i++) {
        // add system memory
        string sys_mem_name = "SYSTEM_MEM " + std::to_string(i);
        id_to_sys_mem[i] = new Mem_device(sys_mem_name, Mem_device::SYSTEM_MEM, i, i, i);
        for (int j = 0; j < num_cpus_per_node; j++) {
            int device_id = i * num_cpus_per_node + j;
            string cpu_name = "CPU " + std::to_string(device_id);
            id_to_cpu[device_id] = new Comp_device(cpu_name, Comp_device::LOC_PROC, i, i, device_id);
        }
    }

    // Create GPU compute device
    for (int i = 0; i < num_nodes; i++) {
        for (int j = 0; j < num_gpus_per_node; j++) {
            int device_id = i * num_gpus_per_node + j;
            string gpu_name = "GPU " + std::to_string(device_id);
            id_to_gpu[device_id] = new Comp_device(gpu_name, Comp_device::TOC_PROC, i, i, device_id);
            string gpu_mem_name = "GPU_FB_MEM " + std::to_string(device_id);
            id_to_gpu_fb_mem[device_id] = new Mem_device(gpu_mem_name, Mem_device::GPU_FB_MEM, i, i, device_id);
        }
    }

    // Create inter GPU comm devices (NVLinks)
    for (int i = 0; i < total_num_gpus; i++) {
        for (int j = 0; j < total_num_gpus; j++) {
            Device* src = id_to_gpu[i];
            Device* dst = id_to_gpu[j];
            if (src->node_id == dst->node_id && src != dst) {
                int device_id = i * total_num_gpus + j;
                string nvlink_name = "NVLINK " + std::to_string(device_id);
                ids_to_inter_gpu_comm_device[device_id] = new Comm_device(nvlink_name, Comm_device::NVLINK_COMM, src->node_id, src->node_id, device_id, 0, inter_gpu_bandwidth);
            }
        }
    }

    // Create gpu<->dram comm devices
    for (int i = 0; i < total_num_gpus; i++) {
        int node_id = total_num_gpus / num_gpus_per_node;
        string pci_in_name = "PCI_IN " + std::to_string(i);
        id_to_gputodram_comm_device[i] = new Comm_device(pci_in_name, Comm_device::PCI_IN_COMM, node_id, node_id, i, 0, gpu_dram_bandwidth);
        string pci_out_name = "PCI_OUT " + std::to_string(i);
        id_to_dramtogpu_comm_device[i] = new Comm_device(pci_out_name, Comm_device::PCI_OUT_COMM, node_id, node_id, i, 0, gpu_dram_bandwidth);
    }

    // Create inter node comm devices
    for (int i = 0; i < num_nodes; i++) {
        for (int j = 0; j < num_nodes; j++) {
            if (i != j) {
                int device_id = i * num_nodes + j;
                string nic_name = "NIC " + std::to_string(device_id);
                ids_to_inter_node_comm_device[device_id] = new Comm_device(nic_name, Comm_device::NIC_OUT_COMM, -1, -1, device_id, 0, inter_node_bandwidth);
            }
        }
    }
}

Comp_device *Machine_Old::get_cpu(int device_id) 
{
    assert(id_to_cpu.find(device_id) != id_to_cpu.end());
    return id_to_cpu[device_id];
}

Comp_device *Machine_Old::get_gpu(int device_id) 
{
    assert(id_to_gpu.find(device_id) != id_to_gpu.end());
    return id_to_gpu[device_id];
}

Mem_device *Machine_Old::get_sys_mem(int node_id) 
{
    assert(id_to_sys_mem.find(node_id) != id_to_sys_mem.end());
    return id_to_sys_mem[node_id];
}

Mem_device *Machine_Old::get_gpu_fb_mem(int device_id) 
{
    assert(id_to_gpu_fb_mem.find(device_id) != id_to_gpu_fb_mem.end());
    return id_to_gpu_fb_mem[device_id];
}

vector<Comm_device *> Machine_Old::get_comm_path(Mem_device *src_mem, Mem_device *tar_mem)
{
    vector<Comm_device *> ret;
    // on the same memory
    if (src_mem->mem_type == tar_mem->mem_type and src_mem->device_id == tar_mem->device_id) {
        return ret;
    }
    if (src_mem->mem_type == Mem_device::SYSTEM_MEM and tar_mem->mem_type == Mem_device::SYSTEM_MEM) {
        if (src_mem->node_id == tar_mem->node_id) {
            return ret;
        }
        else {
            int device_id = src_mem->node_id * num_nodes + tar_mem->node_id;
            ret.emplace_back(ids_to_inter_node_comm_device[device_id]);
        }
    }
    else if (src_mem->mem_type == Mem_device::GPU_FB_MEM and tar_mem->mem_type == Mem_device::GPU_FB_MEM) {
        if (src_mem->node_id == tar_mem->node_id) {
            int device_id = src_mem->device_id * total_num_gpus + tar_mem->device_id;
            ret.emplace_back(ids_to_inter_gpu_comm_device[device_id]);
        }
        else {
            ret.emplace_back(id_to_gputodram_comm_device[src_mem->device_id]);
            int device_id = src_mem->node_id * num_nodes + tar_mem->node_id;
            ret.emplace_back(ids_to_inter_node_comm_device[device_id]);
            ret.emplace_back(id_to_dramtogpu_comm_device[tar_mem->device_id]);
        }
    }
    else if (src_mem->mem_type == Mem_device::SYSTEM_MEM and tar_mem->mem_type == Mem_device::GPU_FB_MEM) {
        if (src_mem->node_id == tar_mem->node_id) {
            ret.emplace_back(id_to_dramtogpu_comm_device[tar_mem->device_id]);
        }
        else {
            int device_id = src_mem->node_id * num_nodes + tar_mem->node_id;
            ret.emplace_back(ids_to_inter_node_comm_device[device_id]);
            ret.emplace_back(id_to_dramtogpu_comm_device[tar_mem->device_id]);
        }
    }
    else if (src_mem->mem_type == Mem_device::GPU_FB_MEM and tar_mem->mem_type == Mem_device::SYSTEM_MEM) {
        if (src_mem->node_id == tar_mem->node_id) {
            ret.emplace_back(id_to_gputodram_comm_device[src_mem->device_id]);
        }
        else {
            ret.emplace_back(id_to_gputodram_comm_device[src_mem->device_id]);
            int device_id = src_mem->node_id * num_nodes + tar_mem->node_id;
            ret.emplace_back(ids_to_inter_node_comm_device[device_id]);
        }
    }
    else {
        cout << "No path found between " << src_mem->name << " and " << tar_mem->name << endl;
    }

    return ret;
}




// class Task
Task::Task(string name, Device *device) 
: name(name), device(device), ready_time(0.0f), counter(0), is_main(false)
{
    next_tasks.clear();
} 

void Task::add_next_task(Task *task)
{
    next_tasks.push_back(task);
    task->counter++;
}

string Task::to_string()
{
    return name + " on " + device->name;
}

// class Comp_task
Comp_task::Comp_task(std::string name, Comp_device *comp_deivce, float run_time, Mem_device *mem_device)
: Task(name, comp_deivce), run_time(run_time), mem(mem_device)
{
}

string Comp_task::to_string()
{
    return name + "(" + device->name + ',' + std::to_string(counter) + ',' + std::to_string(run_time) + "ms," + mem->name + ")";
}

float Comp_task::cost()
{
    return run_time;
}

// class Comm_task
Comm_task::Comm_task(string name, Comm_device *comm_device, int message_size)
: Task(name, comm_device), message_size(message_size)
{
}

string Comm_task::to_string()
{
    return name + "(" + device->name + ',' + std::to_string(counter) + ',' + std::to_string(message_size) + "B)";
}

float Comm_task::cost()
{
    Comm_device *comm = (Comm_device *) device;
    return comm->latency + message_size / comm->bandwidth;
}

// class Simulator
Simulator::Simulator(Machine *machine) : machine(machine)
{
}

Task *Simulator::new_comp_task(string name, Comp_device *comp_device, float run_time, Mem_device *mem_device)
{
    Task *cur_task = (Task *) new Comp_task(name, comp_device, run_time, mem_device);
    return cur_task;
}

void Simulator::new_comm_task(Task *src_task, Task *tar_task, int message_size)
{
    vector<Comm_device *> path = machine->get_comm_path(((Comp_task *)src_task)->mem, ((Comp_task *)tar_task)->mem);
    if (path.empty()) {
        add_dependency(src_task, tar_task);
        return;
    }
    assert(message_size > 0);
    vector<vector<Task *>> all_tasks;
    // Limit the max number of segments per message
    int seg_size = SEG_SIZE;
    int num_segment = message_size / seg_size;
    if (message_size % seg_size != 0) {
        num_segment += 1;
    }
    if (num_segment > MAX_NUM_SEGS) {
        num_segment = MAX_NUM_SEGS;
        seg_size = message_size / num_segment;
    }
    // Create all the comm tasks
    // Divide messages into segments
    for (int i = 0; i < path.size(); i++) {
        all_tasks.push_back({});
        for (int j = 0; j < num_segment; j++) {
            int cur_seg_size = seg_size;
            if (j == num_segment - 1) {
                cur_seg_size = message_size - (num_segment - 1) * seg_size;
            }
            string name = "seg " + to_string(j) + " from " + src_task->name + " to " + tar_task->name;
            Task *cur_task = (Task *) new Comm_task(name, path[i], cur_seg_size);
            all_tasks[i].push_back(cur_task);
        }
    }

    // Add dependencies among the comm tasks
    for (int i = 0; i < path.size(); i++) {
        for (int j = 0; j < num_segment; j++) {
            if (i == 0) {
                add_dependency(src_task, all_tasks[i][j]);
            }
            if (i == path.size() - 1) {
                add_dependency(all_tasks[i][j], tar_task);
            }
            if (i > 0) {
                add_dependency(all_tasks[i-1][j], all_tasks[i][j]);
            }
        }
    }

}

void Simulator::enter_ready_queue(Task *task)
{
    ready_queue.push(task);
}

void Simulator::add_dependency(vector<Task *> prev_tasks, Task *cur_task)
{
    for (int i = 0; i < prev_tasks.size(); i++) {
        prev_tasks[i]->add_next_task(cur_task);
    }
    
}

void Simulator::add_dependency(Task *prev_task, Task *cur_task)
{
    prev_task->add_next_task(cur_task);
}

void Simulator::simulate()
{
    srand(time(NULL));
    bool measure_main_loop = true;
    float main_loop_start = std::numeric_limits<float>::max();
    float main_loop_stop = 0.0f;
    float sim_time = 0.0f;
    float comp_time = 0.0f;
    float comm_time = 0.0f;
    unordered_map<Device*, float> device_times;
    while (!ready_queue.empty()) {
        // Find the task with the earliest start time
        Task* cur_task = ready_queue.top();
        ready_queue.pop();
        float ready_time = 0;
        if (device_times.find((Device *)cur_task->device) != device_times.end()) {
            ready_time = device_times[(Device *)cur_task->device];
        }
        float start_time = max(ready_time, cur_task->ready_time);
        if (cur_task->device->type == Device::DEVICE_COMP) {
            start_time += LEGION_OVERHEAD;
        }
        float run_time = 0;
        if (cur_task->device->type == Device::DEVICE_COMP) {
            run_time = ((Comp_task *)cur_task)->cost();
            comp_time += run_time;
        }
        else {
            run_time = ((Comm_task *)cur_task)->cost();
            comm_time += run_time;
        }
        float end_time = start_time + run_time;
        device_times[cur_task->device] = end_time;
        if (measure_main_loop and cur_task->is_main) {
            main_loop_start = fminf(main_loop_start, start_time);
            main_loop_stop = fmaxf(main_loop_stop, end_time);
        }
        //cout << cur_task->name << " --- " << cur_task->device->name << " --- " << "device_ready(" << ready_time << ") start("  << start_time << ") run(" << run_time << ") end(" <<  end_time << ")" << endl;
        if (end_time > sim_time)
            sim_time = end_time;
        for (size_t i = 0; i < cur_task->next_tasks.size(); i++) {
            Task* next = cur_task->next_tasks[i];
            next->ready_time = max(next->ready_time, end_time);
            next->counter--;
            if (next->counter == 0) {
                ready_queue.push(next);
            }
        }
    }
    if (measure_main_loop) {
        cout << "main_loop " << main_loop_stop - main_loop_start << "ms" << endl;
    }
    cout << "sim_time " << sim_time << "ms" << endl;
    cout << "total_comp_time " << comp_time << "ms" << endl;
    cout << "total_comm_time " << comm_time << "ms" << endl;
    return;
}

