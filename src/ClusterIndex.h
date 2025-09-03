/**
 * @file ClusterIndex.h
 * @author Yousif Alkhayyat
 * @brief Contains declaration and definition of the ClusterIndex class, a wrapper of hnswlib.
 * @version 0.1
 * @date 2025-09-02
 *
 * @copyright Copyright (c) 2025
 *
 */

#pragma once

#include <vector>
#include <queue>
#include <memory>
#include "lib/hnswlib/hnswlib.h"

enum class Distance
{
    kL2,
    kInnerProduct,
    kCosine
};

template <typename dist_t>
class ClusterIndex
{
public:
    using VectorList = std::vector<std::vector<dist_t>>;
    using QueryResult = std::priority_queue<std::pair<dist_t, hnswlib::labeltype>>;
    using QueryResultList = std::vector<QueryResult>;

private:
    Distance dist_;
    size_t dim_;
    size_t max_elements_;
    size_t M_;
    size_t ef_construction_;
    size_t next_id_;

    std::unique_ptr<hnswlib::SpaceInterface<dist_t>> space_;
    std::unique_ptr<hnswlib::HierarchicalNSW<dist_t>> hnsw_;

public:
    /**
     * @brief Construct a new Cluster Index object.
     *
     * @param dist L2, Cosine, or InnerProduct.
     * @param dim Vector space dimension.
     * @param max_elements Maximum number of elements in index.
     * @param M Tightly connected with internal dimensionality of the data, strongly affects memory consumption.
     * @param ef_construction Controls index search/build speed tradeoff.
     */
    ClusterIndex(Distance dist, size_t dim, size_t max_elements, size_t M, size_t ef_construction)
        : dist_(dist), dim_(dim), max_elements_(max_elements), M_(M), ef_construction_(ef_construction), next_id_(0)
    {
        if (dist == Distance::kL2)
            space_ = std::make_unique<hnswlib::L2Space>(dim);
        else
            space_ = std::make_unique<hnswlib::InnerProductSpace>(dim);

        hnsw_ = std::make_unique<hnswlib::HierarchicalNSW<dist_t>>(space_.get(), max_elements_, M_, ef_construction_);
    }

    /**
     * @brief Insert single vector into the index. Unique ID is always generated.
     *
     * @param vector
     */
    void Insert(const std::vector<dist_t> &vector)
    {
        hnsw_->addPoint(vector.data(), next_id_++);
    }

    /**
     * @brief Insert a batch of vectors into the index. Each vector will have a unique ID.
     *
     * @param vector_list
     */
    void InsertBatch(const VectorList &vector_list)
    {
        for (const auto &vec : vector_list)
        {
            Insert(vec);
        }
    }

    /**
     * @brief Run a single KNN query.
     *
     * @param vector
     * @param k
     * @return QueryResult
     */
    QueryResult SearchKNN(const std::vector<dist_t> &vector, size_t k)
    {
        return hnsw_->searchKnn(vector.data(), k);
    }

    /**
     * @brief Run a batch of KNN queries on the index.
     *
     * @param vector_list
     * @param k
     * @return QueryResultList
     */
    QueryResultList SearchKNNBatch(const VectorList &vector_list, size_t k)
    {
        QueryResultList results;
        for (const auto &vec : vector_list)
        {
            results.push_back(SearchKNN(vec, k));
        }
        return results;
    }
};
