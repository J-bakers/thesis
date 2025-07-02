#include <gtsam/slam/dataset.h>
#include <gtsam/nonlinear/LevenbergMarquardtParams.h>
#include <gtsam/nonlinear/GncParams.h>
#include <gtsam/nonlinear/GncOptimizer.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <iostream>

using namespace gtsam;
using namespace std;

int main(int argc, char** argv) {
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " input.g2o [output.g2o]" << endl;
        return 1;
    }

    string input_file = argv[1];
    string output_file = (argc >= 3) ? argv[2] : "optimized_output.g2o";

    // ✅ Leggi il file G2O (versione moderna)
    auto graph_and_values = readG2o(input_file, true);  // true -> 3D
    NonlinearFactorGraph graph = *graph_and_values.first;
    Values initialEstimate = *graph_and_values.second;

    cout << "Loaded graph with " << graph.size() << " factors and "
         << initialEstimate.size() << " initial values." << endl;

    // ✅ Parametri LM
    LevenbergMarquardtParams lmParams;
    lmParams.setVerbosity("ERROR");
    lmParams.maxIterations = 100;

    // ✅ Parametri GNC
    using MyGncParams = GncParams<LevenbergMarquardtParams>;
    MyGncParams gncParams(lmParams);
    gncParams.setLossType(GncLossType::GM);
    gncParams.setMuStep(1.4);
    gncParams.setWeightsTol(1e-5);
    gncParams.setMaxIterations(50);
    

    // ✅ Ottimizzatore GNC
    GncOptimizer<MyGncParams> optimizer(graph, initialEstimate, gncParams);
    Values resultValues = optimizer.optimize();

    // ✅ Salva su file
    writeG2o(graph, resultValues, output_file);
    cout << "Optimized graph written to: " << output_file << endl;

    return 0;
}

