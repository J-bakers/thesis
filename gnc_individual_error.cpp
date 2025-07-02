#include <gtsam/slam/dataset.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/GncOptimizer.h>
#include <gtsam/nonlinear/LevenbergMarquardtParams.h>

#include <iostream>
#include <vector>
#include <map>
#include <set>
#include <cmath>
#include <fstream>
#include <algorithm>

using namespace std;
using namespace gtsam;

int main(int argc, char** argv) {
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " input.g2o" << endl;
        return 1;
    }

    string filename = argv[1];

    // Read G2O file
    auto graphAndValues = readG2o(filename, true);
    NonlinearFactorGraph graph = *(graphAndValues.first);
    Values initialEstimate = *(graphAndValues.second);

    // Optimize globally with GNC
    LevenbergMarquardtParams lmParams;
    GncParams<LevenbergMarquardtParams> gncParams(lmParams);
    gncParams.setLossType(GncLossType::GM);
    gncParams.setMaxIterations(100);
    

    GncOptimizer optimizer(graph, initialEstimate, gncParams);
    Values global_result = optimizer.optimize();

    // Determine robot IDs based on vertex ID jumps of >= 1000
    map<size_t, size_t> robot_id_from_key;
    vector<size_t> robot_start_ids;
    size_t prev_id = 0;
    size_t robot_id = 0;

    vector<size_t> keys = initialEstimate.keys();
    sort(keys.begin(), keys.end());

    for (size_t i = 0; i < keys.size(); ++i) {
        size_t current_id = keys[i];
        if (i > 0 && current_id - prev_id >= 1000) {
            robot_id++;
        }
        robot_id_from_key[current_id] = robot_id;
        prev_id = current_id;
    }
    size_t num_robots = robot_id + 1;

    cout << "Detected " << num_robots << " robots" << endl;

    // Build per-robot subgraphs and compute error
    vector<NonlinearFactorGraph> subgraphs(num_robots);
    for (const auto& factor : graph) {
        auto keys = factor->keys();
        if (keys.size() == 2) {
            size_t rid1 = robot_id_from_key[keys[0]];
            size_t rid2 = robot_id_from_key[keys[1]];
            if (rid1 == rid2) {
                subgraphs[rid1].add(factor);
            }
        }
    }

    for (size_t rid = 0; rid < num_robots; ++rid) {
        double error = subgraphs[rid].error(global_result);
        cout << "Robot " << rid << " individual error: " << error << endl;
    }

    return 0;
}

