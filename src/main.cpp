#include "ClusterIndex.h"

int main()
{
    int dim = 16;              // Dimension of the elements
    int max_elements = 10000;  // Maximum number of elements, should be known beforehand
    int M = 16;                // Tightly connected with internal dimensionality of the data
                               // strongly affects the memory consumption
    int ef_construction = 200; // Controls index search speed/build speed tradeoff

    // Initing index
    ClusterIndex<float> index(Distance::kL2, dim, max_elements, M, ef_construction);

    // Generate random data
    std::mt19937 rng;
    rng.seed(47);
    std::uniform_real_distribution<float> distrib_real;
    ClusterIndex<float>::VectorList data(max_elements, std::vector<float>(dim));
    for (int i = 0; i < max_elements; i++)
    {
        for (int j = 0; j < dim; j++)
        {
            data[i][j] = distrib_real(rng);
        }
    }

    // Add data to index
    index.InsertBatch(data);

    // Query the elements for themselves and measure recall
    float correct = 0;
    for (int i = 0; i < max_elements; i++)
    {
        ClusterIndex<float>::QueryResult result = index.SearchKNN(data[i], 1);
        auto label = result.top().second;
        if (label == i)
            correct++;
    }
    float recall = correct / max_elements;
    std::cout << "Recall: " << recall << "\n";

    return 0;
}
