#include <test.hpp>


#include <vector>

#define private public

#include <classifier.h>

#undef private

TEST(Classifier, TestSegregation) {

    GNB gnb(4);

    GNB::Samples samples{
            {10, 14, 8, 2},
            {5,  14, 3, 2},
            {10, 2,  8, 5}
    };

    std::vector<std::string> labels{
            gnb.possible_labels[0], gnb.possible_labels[1], gnb.possible_labels[0]
    };

    auto segregated = gnb.segregatePerLabel(samples, labels);


    ASSERT_EQ(segregated.size(), gnb.possible_labels.size());
    ASSERT_EQ(segregated[gnb.possible_labels[0]].size(), 2);
    ASSERT_EQ(segregated[gnb.possible_labels[0]][0].size(), 4);
    ASSERT_EQ(segregated[gnb.possible_labels[1]].size(), 1);
    ASSERT_EQ(segregated[gnb.possible_labels[1]][0].size(), 4);
    ASSERT_EQ(segregated[gnb.possible_labels[2]].size(), 0);
}

TEST(Classifier, TestTrain) {

    const int num_params = 2;
    GNB gnb(num_params);

    GNB::Samples samples{
            {10, 0},
            {5,  0}
    };

    std::vector<std::string> labels(samples.size());
    std::fill_n(labels.begin(), samples.size(), gnb.possible_labels[0]);

    gnb.train(samples, labels);

    auto &statsPerLabel = gnb.statsPerLabel;
    ASSERT_EQ(statsPerLabel.size(), gnb.possible_labels.size());

    auto& statsForLabel = statsPerLabel[gnb.possible_labels[0]];
    ASSERT_EQ(statsForLabel.means.size(), num_params);

    ASSERT_EQ(statsForLabel.means[0], 7.5);
    ASSERT_EQ(statsForLabel.means[1], 0);

    ASSERT_EQ(statsForLabel.stddevs.size(), num_params);
    ASSERT_EQ(statsForLabel.stddevs[0], 2.5);
    ASSERT_EQ(statsForLabel.stddevs[1], 0);

    ASSERT_EQ(statsForLabel.priors.size(), num_params);
}