#include "classifier.h"

#include <iostream>
#include <sstream>
#include <fstream>
#include <math.h>
#include <vector>
#include <cassert>
#include <numeric>


/**
 * Initializes GNB
 */

GNB::GNB(size_t n_params_) : n_params(n_params_) {};

GNB::~GNB() = default;

GNB::SamplesPerLabel GNB::segregatePerLabel(Samples data, std::vector<std::string> labels) {
    // Some sanity checks
    assert (!data.empty());
    assert (data.size() == labels.size());
    assert (data[0].size() == n_params);

    // Create a map to separate the incoming samples
    SamplesPerLabel samplesPerLabel(possible_labels.size());
    for (auto &label : possible_labels) {
        samplesPerLabel.emplace(label, std::vector<std::vector<double>>());
    }

    // Separate the incoming samples into the map
    for (size_t i = 0; i < data.size(); i++) {
        std::string &label = labels[i];
        std::vector<double> &sample = data[i];

        assert (std::find(possible_labels.begin(), possible_labels.end(), label) != possible_labels.end());

        auto &totalsForLabel = samplesPerLabel[label];
        totalsForLabel.push_back(sample);
    }

    return samplesPerLabel;
}


void GNB::train(Samples data, std::vector<std::string> labels) {

    /*
        Trains the classifier with N data points and labels.

        INPUTS
        data - array of N observations
          - Each observation is a tuple with 4 values: s, d,
            s_dot and d_dot.
          - Example : [
                  [3.5, 0.1, 5.9, -0.02],
                  [8.0, -0.3, 3.0, 2.2],
                  ...
              ]

        labels - array of N labels
          - Each label is one of "left", "keep", or "right".
    */

    // Some sanity checks
    assert (!data.empty());
    assert (data.size() == labels.size());
    assert (data[0].size() == n_params);

    // Create a map to separate the incoming samples
    SamplesPerLabel samplesPerLabel = segregatePerLabel(data, labels);

    // Calculate mean/stdev/prior per label
    for (auto &label : possible_labels) {
        const std::vector<std::vector<double>> &samplesForLabel = samplesPerLabel[label];
        double sampleSize = samplesForLabel.size();

        // In case there is no data
        if (sampleSize == 0) {
            std::cout << "WARN: no data for label: " << label << std::endl;
            LabelStats stats{Eigen::ArrayXd::Zero(n_params),
                             Eigen::ArrayXd::Zero(n_params),
                             0};
            statsPerLabel.emplace(label, std::move(stats));
            continue;
        }

        // Means
        Eigen::ArrayXd means = Eigen::ArrayXd::Zero(n_params);
        for (const std::vector<double> &sample : samplesForLabel) {
            assert(sample.size() == n_params);
            means += Eigen::ArrayXd::Map(sample.data(), sample.size());
        }
        means /= sampleSize;

        //stddev
        Eigen::ArrayXd stddevs = Eigen::ArrayXd::Zero(n_params);
        for (const std::vector<double> &sample : samplesForLabel) {
            assert(sample.size() == n_params);
            stddevs += (Eigen::ArrayXd::Map(sample.data(), sample.size()) - means).square();
        }
        stddevs = (stddevs / sampleSize).sqrt();

        // Prior
        double prior = sampleSize / labels.size();

        // Store for prediction
        LabelStats stats{std::move(means), std::move(stddevs), prior};
        statsPerLabel.emplace(label, std::move(stats));
    }

}

std::string GNB::predict(std::vector<double> sample) {
    /*
        Once trained, this method is called and expected to return
        a predicted behavior for the given observation.

        INPUTS

        observation - a 4 tuple with s, d, s_dot, d_dot.
          - Example: [3.5, 0.1, 8.5, -0.2]

        OUTPUT

        A label representing the best guess of the classifier. Can
        be one of "left", "keep" or "right".
        """
    */

    // Sanity checks
    assert (sample.size() == n_params);

    Eigen::ArrayXd probs = Eigen::ArrayXd::Ones(possible_labels.size());

    for (size_t i = 0; i < possible_labels.size(); i++) {
        const LabelStats &stats = statsPerLabel.at(possible_labels[i]);
        for (size_t j = 0; j < sample.size(); j++) {
            probs[i] *= (1.0 / sqrt(2.0 * M_PI * pow(stats.stddevs[j], 2))) *
                        exp(-0.5 * pow(sample[j] - stats.means[j], 2) / pow(stats.stddevs[j], 2));
        }

        probs[i] *= stats.prior;
    }

    double max = -1;
    size_t maxIndex = 0;
    for (size_t i = 0; i < possible_labels.size(); i++) {
        if (probs[i] > max) {
            max = probs[i];
            maxIndex = i;
        }
    }

    return possible_labels[maxIndex];
}