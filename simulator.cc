#include "simulator.h"
#include <unordered_set>
#include <fstream>

using std::string;
using std::vector;
using std::cout;
using std::endl;
using std::to_string;
using std::unordered_map;
using std::max;
using std::unordered_set;

// class Device
Device::Device(string name, DeviceType type, int node_id, int socket_id, int device_id)
: name(name), type(type), node_id(node_id), socket_id(socket_id), device_id(device_id)
{
}

// class Comp_device
Comp_device::Comp_device(string name, int node_id, int socket_id, int device_id)
: Device(name, Device::DEVICE_COMP, node_id, socket_id, device_id)
{
}

// class Comm_device
Comm_device::Comm_device(string name, int node_id, int socket_id, int device_id, float latency, float bandwidth)
: Device(name, Device::DEVICE_COMM, node_id, socket_id, device_id), latency(latency), bandwidth(bandwidth)
{
}

// class Machine
Machine::Machine() {
    comp_to_dram.clear();
    comp_to_qpi_in.clear();
    comp_to_qpi_out.clear();
    comp_to_nic_in.clear();
    comp_to_nic_out.clear();
}

void Machine::attach_dram(Comp_device *comp, Comm_device *comm)
{
    assert(starts_with(comm->name, "DRAM"));
    if (comp_to_dram.find(comp->socket_id) == comp_to_dram.end()) {
        comp_to_dram[comp->socket_id] = comm;
    }
}

void Machine::attach_qpi_in(Comp_device *comp, Comm_device *comm)
{
    assert(starts_with(comm->name, "QPI"));
    if (comp_to_qpi_in.find(comp->socket_id) == comp_to_qpi_in.end()) {
        comp_to_qpi_in[comp->socket_id] = comm;
    }
}

void Machine::attach_qpi_out(Comp_device *comp, Comm_device *comm)
{
    assert(starts_with(comm->name, "QPI"));
    if (comp_to_qpi_out.find(comp->socket_id) == comp_to_qpi_out.end()) {
        comp_to_qpi_out[comp->socket_id] = comm;
    }
}

void Machine::attach_nic_in(Comp_device *comp, Comm_device *comm)
{
    assert(starts_with(comm->name, "NIC"));
    if (comp_to_nic_in.find(comp->socket_id) == comp_to_nic_in.end()) {
        comp_to_nic_in[comp->socket_id] = comm;
    }
}

void Machine::attach_nic_out(Comp_device *comp, Comm_device *comm)
{
    assert(starts_with(comm->name, "NIC"));
    if (comp_to_nic_out.find(comp->socket_id) == comp_to_nic_out.end()) {
        comp_to_nic_out[comp->socket_id] = comm;
    }
}

vector<Comm_device *> Machine::get_comm_path(Comp_device *source, Comp_device *target)
{
    vector<Comm_device *> ret;
    // on the same device
    if (source->device_id == target->device_id) {
        return ret;
    }
    // on the same socket
    else if (source->socket_id == target->socket_id) {
        ret.emplace_back(comp_to_dram[source->socket_id]);

    }
    // on the same node
    else if (source->node_id == target->node_id) {
        //ret.emplace_back(comp_to_dram[source->socket_id]);
        ret.emplace_back(comp_to_qpi_out[source->socket_id]);
        ret.emplace_back(comp_to_qpi_in[target->socket_id]);
        //ret.emplace_back(comp_to_dram[target->socket_id]);
    }
    // on different nodes
    else {
        //ret.emplace_back(comp_to_dram[source->socket_id]);
        ret.emplace_back(comp_to_nic_out[source->socket_id]);
        ret.emplace_back(comp_to_nic_in[target->socket_id]);
        //ret.emplace_back(comp_to_dram[target->socket_id]);
    }
    return ret;
}


// class Task
Task::Task(string name, Device *device) 
: name(name), device(device), ready_time(0.0f), counter(0)
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
Comp_task::Comp_task(string name, Comp_device *device, float run_time)
: Task(name, device), run_time(run_time)
{
}

string Comp_task::to_string()
{
    return name + "(" + device->name + "," + std::to_string(run_time) + "ms)";
}

float Comp_task::cost()
{
    return run_time;
}

// class Comm_task
Comm_task::Comm_task(string name, Comm_device *device, int message_size)
: Task(name, device), message_size(message_size)
{
}

string Comm_task::to_string()
{
    return name + "(" + device->name + "," + std::to_string(message_size) + "B)";
}

float Comm_task::cost()
{
    Comm_device *comm = (Comm_device *) device;
    return comm->latency + message_size / comm->bandwidth;
}

// class Simulator
Task *Simulator::new_comp_task(string name, Comp_device *device, float run_time)
{
    Task *cur_task = (Task *) new Comp_task(name, device, run_time);
    return cur_task;
}

void Simulator::new_comm_task(Task *source_task, Task *target_task, int message_size)
{
    assert(message_size > 0);
    vector<Comm_device *> path = machine.get_comm_path((Comp_device *)(source_task->device), (Comp_device *)(target_task->device));
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
            string name = "seg " + to_string(j) + " from " + source_task->name + " to " + target_task->name;
            Task *cur_task = (Task *) new Comm_task(name, path[i], cur_seg_size);
            all_tasks[i].push_back(cur_task);
        }
    }

    // Add dependencies among the comm tasks
    for (int i = 0; i < path.size(); i++) {
        for (int j = 0; j < num_segment; j++) {
            if (i == 0) {
                add_dependency(source_task, all_tasks[i][j]);
            }
            if (i == path.size() - 1) {
                add_dependency(all_tasks[i][j], target_task);
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
    float sim_time = 0.0f;
    unordered_map<Device*, float> device_times;
    while (!ready_queue.empty()) {
        // Find the task with the earliest start time
        Task* cur_task = ready_queue.top();
        ready_queue.pop();
        float ready_time = 0;
        if (device_times.find((Device *)cur_task->device) != device_times.end()) {
            ready_time = device_times[(Device *)cur_task->device];
        }
        float start_time = std::max(ready_time, cur_task->ready_time);
        if (cur_task->device->type == Device::DEVICE_COMP) {
            start_time += LEGION_OVERHEAD;
        }
        float run_time = 0;
        if (cur_task->device->type == Device::DEVICE_COMP) {
            run_time = ((Comp_task *)cur_task)->cost();
        }
        else {
            run_time = ((Comm_task *)cur_task)->cost();
        }
        float end_time = start_time + run_time;
        device_times[cur_task->device] = end_time;
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
    cout << "sim_time " << sim_time << "ms" << endl;
    return;
}

void stencil_1d()
{
    Simulator simulator;

    int num_tasks = 18;
    int num_nodes = 1;
    int num_sockets_per_node = 2;
    int num_sockets = num_nodes * num_sockets_per_node;
    int num_tasks_per_socket = num_tasks / num_sockets;

    vector<vector<Comp_device *>> cpus;
    for (int i = 0; i < num_nodes; i++) {
        for (int j = 0; j < num_sockets_per_node; j++) {
            cpus.push_back({});
            for (int k = 0; k < num_tasks_per_socket; k++) {
                int node_id = i;
                int socket_id = i * num_sockets_per_node + j;
                int device_id = socket_id * num_tasks_per_socket + k;
                string cpu_name = "CPU " +to_string(device_id) + " on SOCKET " + to_string(socket_id) + " on NODE " + to_string(node_id);
                cpus[socket_id].emplace_back(new Comp_device(cpu_name, node_id, socket_id, device_id));
            }
            
        }
    }

    for (int i = 0; i < num_nodes; i++) {
        int node_id, socket_id;
        node_id = i;
        socket_id = i * num_sockets_per_node + 0;
        string nic_in_name = "NIC_IN on SOCKET " + to_string(socket_id) + " NODE " + to_string(node_id);
        Comm_device *nic_in = new Comm_device(nic_in_name, node_id, socket_id, socket_id, 0.000507, 20.9545 * 1024 * 1024);
        string nic_out_name = "NIC_OUT on SOCKET " + to_string(socket_id) + " NODE " + to_string(node_id);
        Comm_device *nic_out = new Comm_device(nic_out_name, node_id, socket_id, socket_id, 0.000507, 20.9545 * 1024 * 1024);
        for (int j = 0; j < num_sockets_per_node; j++) {
            node_id = i;
            socket_id = i * num_sockets_per_node + j;
            string dram_name = "DRAM on SOCKET " + to_string(socket_id) + " NODE " + to_string(node_id);
            Comm_device *dram = new Comm_device(dram_name, node_id, socket_id, socket_id, 0.00003, 8.75 * 1024 * 1024); // ms, B/ms or kB/s
            string qpi_in_name = "QPI_IN on SOCKET " + to_string(socket_id) + " NODE " + to_string(node_id);
            Comm_device *qpi_in = new Comm_device(qpi_in_name, node_id, socket_id, socket_id, 0.0003965, 6.65753 * 1024 * 1024); // ms, B/ms or kB/s
            string qpi_out_name = "QPI_OUT on SOCKET " + to_string(socket_id) + " NODE " + to_string(node_id);
            Comm_device *qpi_out = new Comm_device(qpi_out_name, node_id, socket_id, socket_id, 0.0003965, 6.65753 * 1024 * 1024); // ms, B/ms or kB/s
            simulator.machine.attach_dram(cpus[socket_id][0], dram);
            simulator.machine.attach_qpi_in(cpus[socket_id][0], qpi_in);
            simulator.machine.attach_qpi_out(cpus[socket_id][0], qpi_out);
            simulator.machine.attach_nic_in(cpus[socket_id][0], nic_in);
            simulator.machine.attach_nic_out(cpus[socket_id][0], nic_out);
        }
    }

    // 1D stencil
    // 0: 0 1
    // 1: 0 1 2
    // 2: 1 2 3
    // 3: 2 3

    // init comp tasks first
    int iters = 2;
    vector<vector <Task *>> comp_tasks;
    for (int t = 0; t < iters; t++) {
        comp_tasks.push_back({});
        for (int i = 0; i < num_nodes; i++) {
            for (int j = 0; j < num_sockets_per_node; j++) {
                for (int k = 0; k < num_tasks_per_socket; k++) {
                    int socket_id = i * num_sockets_per_node + j;
                    int task_id = socket_id * num_tasks_per_socket + k;
                    string task_name = "comp_task " + to_string(task_id) + " iter " + to_string(t);
                    float run_time = 0 * 0.333605;
                    comp_tasks[t].emplace_back(simulator.new_comp_task(task_name, cpus[socket_id][k], run_time));
                }
            }
        }
    }

    for (int i = 0; i < comp_tasks[0].size(); i++) {
        simulator.enter_ready_queue((Task *)comp_tasks[0][i]);
    }

    // add comm tasks between iters
    for (int i = 1; i < iters; i++) {
        for (int j = 0; j < num_tasks; j++){
            int message_size = 262144;
            // left
            int left = j - 1;
            if (left >= 0) {
                simulator.new_comm_task(comp_tasks[i-1][j], comp_tasks[i][left], message_size);
            }
            // right
            int right = j + 1;
            if (right < num_tasks) {
                simulator.new_comm_task(comp_tasks[i-1][j], comp_tasks[i][right], message_size);
            }
            // mid
            simulator.new_comm_task(comp_tasks[i-1][j], comp_tasks[i][j], message_size);
        }
    }


    simulator.simulate();
}

vector<string> split(string srcStr, const string& delim)
{
	int nPos = 0;
	vector<string> vec;
	nPos = srcStr.find(delim.c_str());
	while(-1 != nPos)
	{
		string temp = srcStr.substr(0, nPos);
		vec.push_back(temp);
		srcStr = srcStr.substr(nPos+1);
		nPos = srcStr.find(delim.c_str());
	}
	vec.push_back(srcStr);
	return vec;
}

void circuit(int argc, char **argv)
{
    Simulator simulator;

    // set up comp devices
    int num_cpus = 18;
    int num_nodes = 2;
    int num_sockets_per_node = 2;
    int num_sockets = num_nodes * num_sockets_per_node;
    int num_cpus_per_socket = num_cpus / num_sockets;
    vector<vector<Comp_device *>> cpus;
    for (int i = 0; i < num_nodes; i++) {
        for (int j = 0; j < num_sockets_per_node; j++) {
            cpus.push_back({});
            for (int k = 0; k < num_cpus_per_socket; k++) {
                int node_id = i;
                int socket_id = i * num_sockets_per_node + j;
                int device_id = socket_id * num_cpus_per_socket + k;
                string cpu_name = "CPU " +to_string(device_id) + " on SOCKET " + to_string(socket_id) + " on NODE " + to_string(node_id);
                cpus[socket_id].emplace_back(new Comp_device(cpu_name, node_id, socket_id, device_id));
            }
        }
    }
    unordered_map<string, Comp_device *> cpu_id_map;
    cpu_id_map["0x1d00000000000001"] = cpus[0][0];
    cpu_id_map["0x1d00000000000002"] = cpus[0][1];
    cpu_id_map["0x1d00010000000001"] = cpus[2][0];
    cpu_id_map["0x1d00010000000002"] = cpus[2][1];


    // set up comm devices
    for (int i = 0; i < num_nodes; i++) {
        int node_id, socket_id;
        node_id = i;
        socket_id = i * num_sockets_per_node + 0;
        string nic_in_name = "NIC_IN on SOCKET " + to_string(socket_id) + " NODE " + to_string(node_id);
        Comm_device *nic_in = new Comm_device(nic_in_name, node_id, socket_id, socket_id, 0.000507, 20.9545 * 1024 * 1024);
        string nic_out_name = "NIC_OUT on SOCKET " + to_string(socket_id) + " NODE " + to_string(node_id);
        Comm_device *nic_out = new Comm_device(nic_out_name, node_id, socket_id, socket_id, 0.000507, 20.9545 * 1024 * 1024);
        for (int j = 0; j < num_sockets_per_node; j++) {
            node_id = i;
            socket_id = i * num_sockets_per_node + j;
            string dram_name = "DRAM on SOCKET " + to_string(socket_id) + " NODE " + to_string(node_id);
            Comm_device *dram = new Comm_device(dram_name, node_id, socket_id, socket_id, 0.00003, 8.75 * 1024 * 1024); // ms, B/ms or kB/s
            string qpi_in_name = "QPI_IN on SOCKET " + to_string(socket_id) + " NODE " + to_string(node_id);
            Comm_device *qpi_in = new Comm_device(qpi_in_name, node_id, socket_id, socket_id, 0.0003965, 6.65753 * 1024 * 1024); // ms, B/ms or kB/s
            string qpi_out_name = "QPI_OUT on SOCKET " + to_string(socket_id) + " NODE " + to_string(node_id);
            Comm_device *qpi_out = new Comm_device(qpi_out_name, node_id, socket_id, socket_id, 0.0003965, 6.65753 * 1024 * 1024); // ms, B/ms or kB/s
            simulator.machine.attach_dram(cpus[socket_id][0], dram);
            simulator.machine.attach_qpi_in(cpus[socket_id][0], qpi_in);
            simulator.machine.attach_qpi_out(cpus[socket_id][0], qpi_out);
            simulator.machine.attach_nic_in(cpus[socket_id][0], nic_in);
            simulator.machine.attach_nic_out(cpus[socket_id][0], nic_out);
        }
    }

    string folder;
    if (argc == 2) {
        folder = argv[1];
    }
    unordered_map<int, float> cost_map;
    // get costs of tasks
    std::ifstream cost_file(folder + "/cost");
    int uid;
    float cost;
    while (cost_file >> uid >> cost) {
        cout << uid << " " << cost << endl;
        if (cost_map.find(uid) == cost_map.end()) {
            cost_map[uid] = cost * 0.001;  //us -> ms
        }
        else {
            cout << "Has duplicate uid in cost file" << endl;
        }
    }

    unordered_map<string, Task *> comp_tasks_map;
    unordered_map<string, int> messages_map;
    unordered_set<string> left;
    unordered_set<string> right;
    // get dependencies
    std::ifstream dag_file(folder + "/dag");
    if (dag_file.is_open()) {
        std::string line;
        while (std::getline(dag_file, line)) {
            vector<string> line_array = split(line, " ");
            /*
            for (int i = 0; i < line_array.size(); i++) {
                cout << line_array[i] << " ";
            }
            cout << endl;
            */
            if (line_array[0] == "comp:") {
                int task_id = -1;
                for (int i = 0; i < line_array.size(); i++) {
                    if (line_array[i] == "(UID:") {
                        task_id = stoi(line_array[i+1].substr(0, line_array[i+1].size() - 1));
                        break;
                    }
                }
                string task_name = "op_node_" + to_string(task_id);
                Task *cur_task = simulator.new_comp_task(task_name, cpu_id_map[line_array.back()], cost_map[task_id]);
                cout << cur_task->to_string() << endl;
                comp_tasks_map[task_name] = cur_task;
            }
            else if (line_array[0] == "comm:") {
                cout << line << endl;
                if (line_array[4] == "'Realm" and (line_array[5] == "Copy" or line_array[5] == "Fill")) {
                    string task_name;
                    string realm_id = line_array[6].substr(1, line_array[6].size() - 2);
                    if (line_array[5] == "Copy")
                        task_name = "realm_copy_" + realm_id;
                    else
                        task_name = "realm_fill_" + realm_id;
                    string cpu_id = line_array.back().substr(0, line_array.back().size() - 3);
                Task *cur_task = simulator.new_comp_task(task_name, cpu_id_map[cpu_id], 0);
                cout << cur_task->to_string() << endl;
                comp_tasks_map[task_name] = cur_task;
                int index_size = 0, field_size = 0;
                for (int i = 0; i < line_array.size(); i++) {
                    if (line_array[i] == "Index_Space_Size:") {
                        index_size = stoi(line_array[i+1]);
                    }
                    if (line_array[i] == "Field_Size:") {
                        field_size = stoi(line_array[i+1]);
                    }
                }
                assert(index_size > 0 and field_size > 0);
                messages_map[task_name] = index_size * field_size;
                }
                else {
                    cout << "comm: has other types" << endl;
                }
            }
            else if (line_array[0] == "deps:") {
                cout << line << endl;
                //cout << line_array[1] << " " << line_array[3] << endl;
                if (comp_tasks_map.find(line_array[1]) != comp_tasks_map.end() and
                    comp_tasks_map.find(line_array[3]) != comp_tasks_map.end()) {
                    int message_size = 1;
                    if (starts_with(line_array[3], "realm")) {
                        message_size = messages_map[line_array[3]];
                    }
                    else if (starts_with(line_array[1], "realm")) {
                        message_size = messages_map[line_array[1]];
                    }
                    cout << message_size << endl;
                    simulator.new_comm_task(comp_tasks_map[line_array[1]], comp_tasks_map[line_array[3]], message_size);
                    if (left.find(line_array[1]) == left.end()) {
                        left.insert(line_array[1]);
                    }
                    if (right.find(line_array[3]) == right.end()) {
                        right.insert(line_array[3]);
                    }
                }
                else {
                    if (comp_tasks_map.find(line_array[1]) == comp_tasks_map.end())
                        cout << "deps: cannot find " << line_array[1] << endl;
                    if (comp_tasks_map.find(line_array[3]) == comp_tasks_map.end())
                        cout << "deps: cannot find " << line_array[3] << endl;
                }

            }
            else {
                cout << "error" << endl;
            }
        }
        dag_file.close();
    }

    for (auto i : left) {
        if (right.find(i) == right.end()) {
            cout << "starts with:" << i << endl;
            simulator.enter_ready_queue(comp_tasks_map[i]);
        }
    }



    simulator.simulate();
}

int main(int argc, char **argv)
{
    circuit(argc, argv);
}


