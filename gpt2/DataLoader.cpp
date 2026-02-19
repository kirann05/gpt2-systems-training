#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <Eigen/Dense>
#include <vector>
#include <string>
#include <cassert>
#include <dirent.h>
#include <algorithm>
#include <iostream>
#include "cnpy.h"

namespace py = pybind11;
using namespace Eigen;
using namespace std;

vector<int> load_tokens(const string& filename) {
    vector<int> tokens;
    try {
        cnpy::NpyArray arr = cnpy::npy_load(filename);

        if (arr.word_size == 2) {  // Assuming 2 bytes per element means the data type is 'short'
            short* data = arr.data<short>();
            tokens.assign(data, data + arr.shape[0]);
        } else if (arr.word_size == 4) {  // Assuming 4 bytes per element means the data type is 'int'
            int* data = arr.data<int>();
            tokens.assign(data, data + arr.shape[0]);
        } else {
            cerr << "Unsupported data type." << endl;
            exit(EXIT_FAILURE);
        }
    } catch (const std::exception& e) {
        cerr << "Error loading file: " << e.what() << endl;
        exit(EXIT_FAILURE);
    }
    return tokens;
}

class DataLoader {
public:
    DataLoader(int B, int T, const string& split) : B(B), T(T) {
        assert(split == "train" || split == "val");

        string data_root = "edu_fineweb10B";
        vector<string> shards;

        // Read directory
        DIR *dir;
        struct dirent *ent;
        if ((dir = opendir(data_root.c_str())) != nullptr) {
            while ((ent = readdir(dir)) != nullptr) {
                string filename = ent->d_name;
                if (filename.find(split) != string::npos) {
                    shards.push_back(data_root + "/" + filename);
                }
            }
            closedir(dir);
        } else {
            perror("Could not open directory");
            exit(EXIT_FAILURE);
        }

        if (shards.empty()) {
            cerr << "No shards found for split: " << split << endl;
            exit(EXIT_FAILURE);
        }

        sort(shards.begin(), shards.end());
        this->shards = shards;

        current_shard = 0;
        tokens = load_tokens(shards[current_shard]);
        current_pos = B * T;

        reset();
    }

    void reset() {
        current_shard = 0;
        tokens = load_tokens(shards[current_shard]);
        current_pos = B * T;
    }

    pair<MatrixXi, MatrixXi> next_batch() {
        int buf_size = B * T + 1;
        if (current_pos + buf_size > tokens.size()) {
            current_shard = (current_shard + 1) % shards.size();
            tokens = load_tokens(shards[current_shard]);
            current_pos = B * T;
        }
        if (current_pos + buf_size > tokens.size()) {
            cerr << "Buffer size exceeds token size. Exiting." << endl;
            exit(EXIT_FAILURE);
        }
        vector<int> buf(tokens.begin() + current_pos, tokens.begin() + current_pos + buf_size);
        Map<MatrixXi> buf_map(buf.data(), B, T + 1);

        MatrixXi x = buf_map.leftCols(T);
        MatrixXi y = buf_map.rightCols(T);

        current_pos += B * T;

        return make_pair(x, y);
    }

private:
    int B, T;
    vector<string> shards;
    int current_shard;
    vector<int> tokens;
    std::vector<int>::size_type current_pos;
};

PYBIND11_MODULE(dataloader, m) {
    py::class_<DataLoader>(m, "DataLoader")
        .def(py::init<int, int, const string&>())
        .def("reset", &DataLoader::reset)
        .def("next_batch", &DataLoader::next_batch);
}
