#pragma once

#include <Eigen/Dense>

#include <iostream>
#include <sstream>
#include <fstream>
#include <math.h>
#include <vector>
#include <unordered_map>

class GNB {
public:
    /**
      * Constructor
      */
    explicit GNB(size_t n_params = 4);

    /**
     * Destructor
     */
    virtual ~GNB();

    using Samples = std::vector<std::vector<double>>;

    void train(Samples data, std::vector<std::string> labels);

    std::string predict(std::vector<double>);

private:
    struct LabelStats {
        const Eigen::ArrayXd means;
        const Eigen::ArrayXd stddevs;
        const double prior;
    };

    const std::vector<std::string> possible_labels = {"left", "keep", "right"};

    size_t n_params;
    std::unordered_map<std::string, LabelStats> statsPerLabel;

    using SamplesPerLabel = std::unordered_map<std::string, std::vector<std::vector<double>>>;

    SamplesPerLabel segregatePerLabel(Samples, std::vector<std::string> labels);
};
