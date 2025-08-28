#pragma once

#include <vector>
#include <string>
#include <iostream>
#include <queue>

enum class Distance
{
    kL2,
    kInnerProduct,
    kCosine
};

template <typename dist_t>
class ClusterIndex
{
    typedef std::vector<std::vector<dist_t>> VectorList;
    typedef std::priority_queue<std::pair<dist_t, size_t>> QueryResult;
    typedef std::vector<QueryResult> QueryResultList;

public:
    ClusterIndex(Distance dist, size_t dim, size_t max_elements, size_t M, size_t ef_construction);
    void Insert(const VectorList &vector_list);
    QueryResultList SearchKNN(const VectorList &vector_list, size_t k);

private:
    Distance dist_;
    size_t dim_;
    size_t max_elements_;
    size_t M_;
    size_t ef_construction_;
};