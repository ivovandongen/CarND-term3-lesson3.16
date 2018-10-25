#pragma once

#include <iostream>
#include <sstream>
#include <fstream>
#include <math.h>
#include <vector>

class GNB {
public:
    /**
      * Constructor
      */
    GNB();

    /**
     * Destructor
     */
    virtual ~GNB();

    void train(std::vector<std::vector<double>> data, std::vector<std::string> labels);

    std::string predict(std::vector<double>);

private:
    std::vector<std::string> possible_labels = {"left", "keep", "right"};
};
