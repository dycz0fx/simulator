#include "simulator.h"
#include <unordered_set>
#include <fstream>
#include <utility>

using std::string;
using std::vector;
using std::cout;
using std::endl;
using std::to_string;
using std::unordered_map;
using std::unordered_set;
using std::pair;

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

void stencil_1d_cpu()
{
    int num_nodes = 2;
    int num_sockets_per_node = 2;
    int num_cpus_per_socket = 2;
    int num_gpus_per_socket = 0;
    int num_sockets = num_nodes * num_sockets_per_node;
    int num_cpus = num_sockets * num_cpus_per_socket;

    Machine_Sherlock machine(num_nodes, num_sockets_per_node, num_cpus_per_socket, num_gpus_per_socket);
    machine.add_membuses(0.00003, 8.75 * 1024 * 1024);
    machine.add_upis(0.0003965, 6.65753 * 1024 * 1024);
    machine.add_nics(0.000507, 20.9545 * 1024 * 1024);
    
    Simulator simulator(&machine);

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
                for (int k = 0; k < num_cpus_per_socket; k++) {
                    int socket_id = i * num_sockets_per_node + j;
                    int task_id = socket_id * num_cpus_per_socket + k;
                    string task_name = "comp_task " + to_string(task_id) + " iter " + to_string(t);
                    float run_time = 1;
                    comp_tasks[t].emplace_back(simulator.new_comp_task(task_name, machine.get_cpu(socket_id, k), run_time, machine.get_sys_mem(socket_id)));
                }
            }
        }
    }

    for (int i = 0; i < comp_tasks[0].size(); i++) {
        simulator.enter_ready_queue((Task *)comp_tasks[0][i]);
    }

    // add comm tasks between iters
    for (int i = 1; i < iters; i++) {
        for (int j = 0; j < num_cpus; j++){
            int message_size = 262144;
            // left
            int left = j - 1;
            if (left >= 0) {
                simulator.new_comm_task(comp_tasks[i-1][j], comp_tasks[i][left], message_size);
            }
            // right
            int right = j + 1;
            if (right < num_cpus) {
                simulator.new_comm_task(comp_tasks[i-1][j], comp_tasks[i][right], message_size);
            }
            // mid
            simulator.new_comm_task(comp_tasks[i-1][j], comp_tasks[i][j], message_size);
        }
    }

    simulator.simulate();
}

void stencil_1d_gpu()
{
    int num_nodes = 2;
    int num_sockets_per_node = 1;
    int num_cpus_per_socket = 10;
    int num_gpus_per_socket = 4;
    int num_sockets = num_nodes * num_sockets_per_node;
    int num_cpus = num_sockets * num_cpus_per_socket;

    // Machine_Sherlock machine(num_nodes, num_sockets_per_node, num_cpus_per_socket, num_gpus_per_socket);
    // machine.add_membuses(0.00003, 8.75 * 1024 * 1024);
    // machine.add_upis(0.0003965, 6.65753 * 1024 * 1024);
    // machine.add_nics(0.000507, 20.9545 * 1024 * 1024);
    // machine.add_pcis(0.001, 13.2 * 2 * 1024 * 1024);
    // machine.add_nvlinks(6, 0.001, 18.52 * 1024 * 1024);
    
    Machine_Old machine(num_nodes, num_sockets_per_node * num_cpus_per_socket, num_sockets_per_node * num_gpus_per_socket);
    Simulator simulator(&machine);

    // init comp tasks first
    int iters = 2;
    vector<vector <Task *>> comp_tasks;
    for (int t = 0; t < iters; t++) {
        comp_tasks.push_back({});
        for (int i = 0; i < num_nodes; i++) {
            for (int j = 0; j < num_sockets_per_node; j++) {
                for (int k = 0; k < num_gpus_per_socket; k++) {
                    int socket_id = i * num_sockets_per_node + j;
                    int task_id = socket_id * num_gpus_per_socket + k;
                    string task_name = "comp_task " + to_string(task_id) + " iter " + to_string(t);
                    float run_time = 1;
                    comp_tasks[t].emplace_back(simulator.new_comp_task(task_name, machine.get_gpu(task_id), run_time, machine.get_gpu_fb_mem(task_id)));
                }
            }
        }
    }

    for (int i = 0; i < comp_tasks[0].size(); i++) {
        simulator.enter_ready_queue((Task *)comp_tasks[0][i]);
    }

    // add comm tasks between iters
    int num_tasks = num_nodes * num_sockets_per_node * num_gpus_per_socket;
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

std::unordered_map<std::string, pair<int, int>> cpu_id_map;
std::unordered_map<std::string, pair<int, int>> gpu_id_map;
// create machine model (sherlock)
Machine_Sherlock *create_machine_model_sherlock()
{
    Machine_Sherlock *machine = new Machine_Sherlock(4, 2, 10, 2);
    machine->add_membuses(0.00003, 8.75 * 1024 * 1024);     // ms, KB/s
    machine->add_upis(0.0003965, 6.65753 * 1024 * 1024);
    machine->add_nics(0.000507, 20.9545 * 1024 * 1024);
    machine->add_pcis(0.001, 13.2 * 2 * 1024 * 1024);
    machine->add_nvlinks(6, 0.001, 18.52 * 1024 * 1024);
    
    // set up cpu_id_map: socket_id, local_id
    {
    cpu_id_map["0x1d00000000000000"] = {0, 0};
    cpu_id_map["0x1d00000000000001"] = {0, 1};
    cpu_id_map["0x1d00000000000002"] = {0, 2};
    cpu_id_map["0x1d00000000000003"] = {0, 3};
    cpu_id_map["0x1d00000000000004"] = {0, 4};
    cpu_id_map["0x1d00000000000005"] = {0, 5};
    cpu_id_map["0x1d00000000000006"] = {0, 6};
    cpu_id_map["0x1d00000000000007"] = {0, 7};
    cpu_id_map["0x1d00000000000008"] = {0, 8};
    cpu_id_map["0x1d00000000000009"] = {0, 9};
    cpu_id_map["0x1d0000000000000a"] = {1, 0};
    cpu_id_map["0x1d0000000000000b"] = {1, 1};
    cpu_id_map["0x1d0000000000000c"] = {1, 2};
    cpu_id_map["0x1d0000000000000d"] = {1, 3};
    cpu_id_map["0x1d0000000000000e"] = {1, 4};
    cpu_id_map["0x1d0000000000000f"] = {1, 5};
    cpu_id_map["0x1d00000000000010"] = {1, 6};
    cpu_id_map["0x1d00000000000011"] = {1, 7};
    cpu_id_map["0x1d00000000000012"] = {1, 8};
    cpu_id_map["0x1d00000000000013"] = {1, 9};
    cpu_id_map["0x1d00010000000000"] = {2, 0};
    cpu_id_map["0x1d00010000000001"] = {2, 1};
    cpu_id_map["0x1d00010000000002"] = {2, 2};
    cpu_id_map["0x1d00010000000003"] = {2, 3};
    cpu_id_map["0x1d00010000000004"] = {2, 4};
    cpu_id_map["0x1d00010000000005"] = {2, 5};
    cpu_id_map["0x1d00010000000006"] = {2, 6};
    cpu_id_map["0x1d00010000000007"] = {2, 7};
    cpu_id_map["0x1d00010000000008"] = {2, 8};
    cpu_id_map["0x1d00010000000009"] = {2, 9};
    cpu_id_map["0x1d0001000000000a"] = {3, 0};
    cpu_id_map["0x1d0001000000000b"] = {3, 1};
    cpu_id_map["0x1d0001000000000c"] = {3, 2};
    cpu_id_map["0x1d0001000000000d"] = {3, 3};
    cpu_id_map["0x1d0001000000000e"] = {3, 4};
    cpu_id_map["0x1d0001000000000f"] = {3, 5};
    cpu_id_map["0x1d00010000000010"] = {3, 6};
    cpu_id_map["0x1d00010000000011"] = {3, 7};
    cpu_id_map["0x1d00010000000012"] = {3, 8};
    cpu_id_map["0x1d00010000000013"] = {3, 9};
    cpu_id_map["0x1d00020000000000"] = {4, 0};
    cpu_id_map["0x1d00020000000001"] = {4, 1};
    cpu_id_map["0x1d00020000000002"] = {4, 2};
    cpu_id_map["0x1d00020000000003"] = {4, 3};
    cpu_id_map["0x1d00020000000004"] = {4, 4};
    cpu_id_map["0x1d00020000000005"] = {4, 5};
    cpu_id_map["0x1d00020000000006"] = {4, 6};
    cpu_id_map["0x1d00020000000007"] = {4, 7};
    cpu_id_map["0x1d00020000000008"] = {4, 8};
    cpu_id_map["0x1d00020000000009"] = {4, 9};
    cpu_id_map["0x1d0002000000000a"] = {5, 0};
    cpu_id_map["0x1d0002000000000b"] = {5, 1};
    cpu_id_map["0x1d0002000000000c"] = {5, 2};
    cpu_id_map["0x1d0002000000000d"] = {5, 3};
    cpu_id_map["0x1d0002000000000e"] = {5, 4};
    cpu_id_map["0x1d0002000000000f"] = {5, 5};
    cpu_id_map["0x1d00020000000010"] = {5, 6};
    cpu_id_map["0x1d00020000000011"] = {5, 7};
    cpu_id_map["0x1d00020000000012"] = {5, 8};
    cpu_id_map["0x1d00020000000013"] = {5, 9};
    cpu_id_map["0x1d00030000000000"] = {6, 0};
    cpu_id_map["0x1d00030000000001"] = {6, 1};
    cpu_id_map["0x1d00030000000002"] = {6, 2};
    cpu_id_map["0x1d00030000000003"] = {6, 3};
    cpu_id_map["0x1d00030000000004"] = {6, 4};
    cpu_id_map["0x1d00030000000005"] = {6, 5};
    cpu_id_map["0x1d00030000000006"] = {6, 6};
    cpu_id_map["0x1d00030000000007"] = {6, 7};
    cpu_id_map["0x1d00030000000008"] = {6, 8};
    cpu_id_map["0x1d00030000000009"] = {6, 9};
    cpu_id_map["0x1d0003000000000a"] = {7, 0};
    cpu_id_map["0x1d0003000000000b"] = {7, 1};
    cpu_id_map["0x1d0003000000000c"] = {7, 2};
    cpu_id_map["0x1d0003000000000d"] = {7, 3};
    cpu_id_map["0x1d0003000000000e"] = {7, 4};
    cpu_id_map["0x1d0003000000000f"] = {7, 5};
    cpu_id_map["0x1d00030000000010"] = {7, 6};
    cpu_id_map["0x1d00030000000011"] = {7, 7};
    cpu_id_map["0x1d00030000000012"] = {7, 8};
    cpu_id_map["0x1d00030000000013"] = {7, 9};
    }

    // set up gpu_id_map
    {
    gpu_id_map["0x1d00000000000006"] = {0, 0};
    gpu_id_map["0x1d00000000000007"] = {0, 1};
    gpu_id_map["0x1d00000000000008"] = {1, 0};
    gpu_id_map["0x1d00000000000009"] = {1, 1};
    gpu_id_map["0x1d00010000000006"] = {2, 0};
    gpu_id_map["0x1d00010000000007"] = {2, 1};
    gpu_id_map["0x1d00010000000008"] = {3, 0};
    gpu_id_map["0x1d00010000000009"] = {3, 1};
    gpu_id_map["0x1d00020000000006"] = {4, 0};
    gpu_id_map["0x1d00020000000007"] = {4, 1};
    gpu_id_map["0x1d00020000000008"] = {5, 0};
    gpu_id_map["0x1d00020000000009"] = {5, 1};
    gpu_id_map["0x1d00030000000006"] = {6, 0};
    gpu_id_map["0x1d00030000000007"] = {6, 1};
    gpu_id_map["0x1d00030000000008"] = {7, 0};
    gpu_id_map["0x1d00030000000009"] = {7, 1};
    }

    return machine;
}

Machine_Old *create_machine_model_old()
{
    Machine_Old *machine = new Machine_Old(4, 20, 4);
    
    // set up cpu_id_map: node_id, device_id
    {
    cpu_id_map["0x1d00000000000000"] = {0, 0};
    cpu_id_map["0x1d00000000000001"] = {0, 1};
    cpu_id_map["0x1d00000000000002"] = {0, 2};
    cpu_id_map["0x1d00000000000003"] = {0, 3};
    cpu_id_map["0x1d00000000000004"] = {0, 4};
    cpu_id_map["0x1d00000000000005"] = {0, 5};
    cpu_id_map["0x1d00000000000006"] = {0, 6};
    cpu_id_map["0x1d00000000000007"] = {0, 7};
    cpu_id_map["0x1d00000000000008"] = {0, 8};
    cpu_id_map["0x1d00000000000009"] = {0, 9};
    cpu_id_map["0x1d0000000000000a"] = {0, 10};
    cpu_id_map["0x1d0000000000000b"] = {0, 11};
    cpu_id_map["0x1d0000000000000c"] = {0, 12};
    cpu_id_map["0x1d0000000000000d"] = {0, 13};
    cpu_id_map["0x1d0000000000000e"] = {0, 14};
    cpu_id_map["0x1d0000000000000f"] = {0, 15};
    cpu_id_map["0x1d00000000000010"] = {0, 16};
    cpu_id_map["0x1d00000000000011"] = {0, 17};
    cpu_id_map["0x1d00000000000012"] = {0, 18};
    cpu_id_map["0x1d00000000000013"] = {0, 19};
    cpu_id_map["0x1d00010000000000"] = {1, 10};
    cpu_id_map["0x1d00010000000001"] = {1, 21};
    cpu_id_map["0x1d00010000000002"] = {1, 22};
    cpu_id_map["0x1d00010000000003"] = {1, 23};
    cpu_id_map["0x1d00010000000004"] = {1, 24};
    cpu_id_map["0x1d00010000000005"] = {1, 25};
    cpu_id_map["0x1d00010000000006"] = {1, 26};
    cpu_id_map["0x1d00010000000007"] = {1, 27};
    cpu_id_map["0x1d00010000000008"] = {1, 28};
    cpu_id_map["0x1d00010000000009"] = {1, 29};
    cpu_id_map["0x1d0001000000000a"] = {1, 20};
    cpu_id_map["0x1d0001000000000b"] = {1, 31};
    cpu_id_map["0x1d0001000000000c"] = {1, 32};
    cpu_id_map["0x1d0001000000000d"] = {1, 33};
    cpu_id_map["0x1d0001000000000e"] = {1, 34};
    cpu_id_map["0x1d0001000000000f"] = {1, 35};
    cpu_id_map["0x1d00010000000010"] = {1, 36};
    cpu_id_map["0x1d00010000000011"] = {1, 37};
    cpu_id_map["0x1d00010000000012"] = {1, 38};
    cpu_id_map["0x1d00010000000013"] = {1, 39};
    cpu_id_map["0x1d00020000000000"] = {2, 40};
    cpu_id_map["0x1d00020000000001"] = {2, 41};
    cpu_id_map["0x1d00020000000002"] = {2, 42};
    cpu_id_map["0x1d00020000000003"] = {2, 43};
    cpu_id_map["0x1d00020000000004"] = {2, 44};
    cpu_id_map["0x1d00020000000005"] = {2, 45};
    cpu_id_map["0x1d00020000000006"] = {2, 46};
    cpu_id_map["0x1d00020000000007"] = {2, 47};
    cpu_id_map["0x1d00020000000008"] = {2, 48};
    cpu_id_map["0x1d00020000000009"] = {2, 49};
    cpu_id_map["0x1d0002000000000a"] = {2, 50};
    cpu_id_map["0x1d0002000000000b"] = {2, 51};
    cpu_id_map["0x1d0002000000000c"] = {2, 52};
    cpu_id_map["0x1d0002000000000d"] = {2, 53};
    cpu_id_map["0x1d0002000000000e"] = {2, 54};
    cpu_id_map["0x1d0002000000000f"] = {2, 55};
    cpu_id_map["0x1d00020000000010"] = {2, 56};
    cpu_id_map["0x1d00020000000011"] = {2, 57};
    cpu_id_map["0x1d00020000000012"] = {2, 58};
    cpu_id_map["0x1d00020000000013"] = {2, 59};
    cpu_id_map["0x1d00030000000000"] = {3, 60};
    cpu_id_map["0x1d00030000000001"] = {3, 61};
    cpu_id_map["0x1d00030000000002"] = {3, 62};
    cpu_id_map["0x1d00030000000003"] = {3, 63};
    cpu_id_map["0x1d00030000000004"] = {3, 64};
    cpu_id_map["0x1d00030000000005"] = {3, 65};
    cpu_id_map["0x1d00030000000006"] = {3, 66};
    cpu_id_map["0x1d00030000000007"] = {3, 67};
    cpu_id_map["0x1d00030000000008"] = {3, 68};
    cpu_id_map["0x1d00030000000009"] = {3, 69};
    cpu_id_map["0x1d0003000000000a"] = {3, 70};
    cpu_id_map["0x1d0003000000000b"] = {3, 71};
    cpu_id_map["0x1d0003000000000c"] = {3, 72};
    cpu_id_map["0x1d0003000000000d"] = {3, 73};
    cpu_id_map["0x1d0003000000000e"] = {3, 74};
    cpu_id_map["0x1d0003000000000f"] = {3, 75};
    cpu_id_map["0x1d00030000000010"] = {3, 76};
    cpu_id_map["0x1d00030000000011"] = {3, 77};
    cpu_id_map["0x1d00030000000012"] = {3, 78};
    cpu_id_map["0x1d00030000000013"] = {3, 79};
    }

    // set up gpu_id_map
    {
    gpu_id_map["0x1d00000000000006"] = {0, 0};
    gpu_id_map["0x1d00000000000007"] = {0, 1};
    gpu_id_map["0x1d00000000000008"] = {0, 2};
    gpu_id_map["0x1d00000000000009"] = {0, 3};
    gpu_id_map["0x1d00010000000006"] = {1, 4};
    gpu_id_map["0x1d00010000000007"] = {1, 5};
    gpu_id_map["0x1d00010000000008"] = {1, 6};
    gpu_id_map["0x1d00010000000009"] = {1, 7};
    gpu_id_map["0x1d00020000000006"] = {2, 8};
    gpu_id_map["0x1d00020000000007"] = {2, 9};
    gpu_id_map["0x1d00020000000008"] = {2, 10};
    gpu_id_map["0x1d00020000000009"] = {2, 11};
    gpu_id_map["0x1d00030000000006"] = {3, 12};
    gpu_id_map["0x1d00030000000007"] = {3, 13};
    gpu_id_map["0x1d00030000000008"] = {3, 14};
    gpu_id_map["0x1d00030000000009"] = {3, 15};
    }

    return machine;
}



void run_dag_file(int argc, char **argv)
{
    Machine_Sherlock *machine = create_machine_model_sherlock();
    //Machine_Old *machine = create_machine_model_old();
    Simulator simulator((Machine *) machine);

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
        // cout << uid << " " << cost << endl;
        if (cost_map.find(uid) == cost_map.end()) {
            cost_map[uid] = cost * 0.001;  //us -> ms
        }
        else {
            cout << "Has duplicate uid in cost file" << endl;
        }
    }

    int num_comp_tasks = 0;
    int num_comm_tasks = 0;
    unordered_map<string, Task *> comp_tasks_map;
    unordered_map<string, int> messages_map;
    unordered_set<string> left;
    unordered_set<string> right;
    // get comp tasks
    std::ifstream comp_file(folder + "/comp");
    if (comp_file.is_open()) {
        std::string line;
        while (std::getline(comp_file, line)) {
            vector<string> line_array = split(line, " ");
            /*
            for (int i = 0; i < line_array.size(); i++) {
                cout << line_array[i] << " ";
            }
            cout << endl;
            */
            if (line_array[0] == "comp:") {
                num_comp_tasks++;
                int task_id = -1;
                bool is_main = false;
                Comp_device *comp_device = nullptr;
                Mem_device *mem_device = nullptr;
                for (int i = 0; i < line_array.size(); i++) {
                    if ((line_array[1] == "Load" and line_array[2] == "Input" and line_array[3] == "Task") or 
                        (line_array[1] == "SGD" and line_array[2] == "Update" and line_array[3] == "Task")) {
                        is_main = true;
                    }
                    if (line_array[i] == "(UID:") {
                        task_id = stoi(line_array[i+1].substr(0, line_array[i+1].size() - 1));
                    }
                    if (line_array[i] == "Processor:") {
                        if (line_array[i+1] == "CPU") {
                            pair<int, int> ids = cpu_id_map[line_array.back()];
                            comp_device = machine->get_cpu(ids.first, ids.second);
                            mem_device = machine->get_sys_mem(ids.first);
                            //comp_device = machine->get_cpu(ids.second);
                            //mem_device = machine->get_sys_mem(ids.first);
                        }
                        else if (line_array[i+1] == "GPU") {
                            pair<int, int> ids = gpu_id_map[line_array.back()];
                            comp_device = machine->get_gpu(ids.first, ids.second);
                            mem_device = machine->get_gpu_fb_mem(ids.first);
                            //comp_device = machine->get_gpu(ids.second);
                            //mem_device = machine->get_gpu_fb_mem(ids.second);
                        }
                        else {
                            cout << "Unknow type of processor" << endl;
                        }
                        break;
                    }
                }
                string task_name = "op_node_" + to_string(task_id);
                Task *cur_task = simulator.new_comp_task(task_name, comp_device, cost_map[task_id], mem_device);
                cur_task->is_main = is_main;
                // cout << cur_task->to_string() << endl;
                comp_tasks_map[task_name] = cur_task;
            }
            else {
                cout << "error" << endl;
            }
        }
        comp_file.close();
    }
    // get comm tasks
    std::ifstream comm_file(folder + "/comm");
    if (comm_file.is_open()) {
        std::string line;
        while (std::getline(comm_file, line)) {
            vector<string> line_array = split(line, " ");
            /*
            for (int i = 0; i < line_array.size(); i++) {
                cout << line_array[i] << " ";
            }
            cout << endl;
            */
            if (line_array[0] == "comm:") {
                num_comm_tasks++;
                int loc = 2;
                if (line_array[loc] == "'Realm" and (line_array[loc+1] == "Copy" or line_array[loc+1] == "Fill")) {
                    string task_name;
                    string realm_id = line_array[loc+2].substr(1, line_array[6].size() - 2);
                    if (line_array[loc+1] == "Copy")
                        task_name = "realm_copy_" + realm_id;
                    else
                        task_name = "realm_fill_" + realm_id;
                    string device_id = line_array.back().substr(0, line_array.back().size() - 3);
                    string device_type = line_array[line_array.size()-3];
                    device_type = device_type.substr(device_type.size()-3, 3);
                    Task *cur_task;
                    if (device_type == "CPU") {
                        pair<int, int> ids = cpu_id_map[device_id];
                        Comp_device *comp_device = machine->get_cpu(ids.first, ids.second);
                        Mem_device *mem_device = machine->get_sys_mem(ids.first);
                        // Comp_device *comp_device = machine->get_cpu(ids.second);
                        // Mem_device *mem_device = machine->get_sys_mem(ids.first);
                        cur_task = simulator.new_comp_task(task_name, comp_device, 0, mem_device);
                    }
                    else if (device_type == "GPU") {
                        pair<int, int> ids = gpu_id_map[device_id];
                        Comp_device *comp_device = machine->get_gpu(ids.first, ids.second);
                        Mem_device *mem_device = machine->get_gpu_fb_mem(ids.first, ids.second);
                        // Comp_device *comp_device = machine->get_gpu(ids.second);
                        // Mem_device *mem_device = machine->get_gpu_fb_mem(ids.second);
                        cur_task = simulator.new_comp_task(task_name, comp_device, 0, mem_device);
                    }
                    else {
                        cout << "wrong device_type" << endl;
                        assert(0);
                    }
                    // cout << cur_task->to_string() << endl;
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
                    cout << line << endl;
                    cout << "comm: has other types" << endl;
                    assert(0);
                }
            }
            else {
                cout << "error" << endl;
            }
        }
        comp_file.close();
    }    
    // get deps
    std::ifstream deps_file(folder + "/deps");
    if (deps_file.is_open()) {
        std::string line;
        while (std::getline(deps_file, line)) {
            vector<string> line_array = split(line, " ");
            /*
            for (int i = 0; i < line_array.size(); i++) {
                cout << line_array[i] << " ";
            }
            cout << endl;
            */
            if (line_array[0] == "deps:") {
                // cout << line << endl;
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
                        // cout << "deps: cannot find " << line_array[1] << endl;
                        ;
                    if (comp_tasks_map.find(line_array[3]) == comp_tasks_map.end())
                        // cout << "deps: cannot find " << line_array[3] << endl;
                        ;
                }

            }
            else {
                cout << "error" << endl;
            }
        }
        deps_file.close();
    }

    for (auto i : left) {
        if (right.find(i) == right.end()) {
            cout << "starts with:" << i << endl;
            simulator.enter_ready_queue(comp_tasks_map[i]);
        }
    }
    
    simulator.simulate();
    cout << "num_comp_tasks " << num_comp_tasks << endl;
    cout << "num_comm_tasks " << num_comm_tasks << endl;

}

void test_comm()
{
    Machine_Sherlock *machine = create_machine_model_sherlock();
    //Machine_Old *machine = create_machine_model_old();
    Simulator simulator((Machine *) machine);
    string src_task_name = "src_task";
    int src_socket_id = 0;
    int src_local_id = 0;
    simulator.new_comp_task(src_task_name, machine->get_cpu(src_socket_id, src_local_id), 0, machine->get_sys_mem(src_socket_id));
    
}

int main(int argc, char **argv)
{
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    run_dag_file(argc, argv);
    // stencil_1d_gpu();
    std::chrono::steady_clock::time_point stop = std::chrono::steady_clock::now();
    std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double> >(stop-start);
    cout << "simulator runs: " << time_span.count() << " seconds" <<  endl;
}
