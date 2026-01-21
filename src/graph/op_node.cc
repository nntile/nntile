/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/op_node.cc
 * Implementation of OpNode class.
 *
 * @version 1.1.0
 * */

// Include corresponding header
#include "nntile/graph/op_node.hh"

// Include standard headers
#include <stdexcept>

// Include third-party headers

// Include other NNTile headers

namespace nntile::graph
{

//! Convert OpType to string
std::string op_type_to_string(OpType type)
{
    switch(type)
    {
        case OpType::GEMM:
            return "GEMM";
        case OpType::GELU:
            return "GELU";
        case OpType::GELU_BACKWARD:
            return "GELU_BACKWARD";
        case OpType::ADD_FIBER:
            return "ADD_FIBER";
        case OpType::SUM_FIBER:
            return "SUM_FIBER";
        default:
            throw std::invalid_argument("Unknown OpType");
    }
}

//! An operation node in the logical graph
OpNode::OpNode(NodeId id, OpType type, OpAttrs attrs, LogicalGraph* graph)
    : id_(id)
    , type_(type)
    , attrs_(std::move(attrs))
    , graph_(graph)
{
}

//! String representation
std::string OpNode::to_string() const
{
    std::string result = op_type_to_string(type_) + "(id=" +
        std::to_string(id_) + ", inputs=[";
    for(size_t i = 0; i < inputs_.size(); ++i)
    {
        if(i > 0)
        {
            result += ", ";
        }
        result += inputs_[i]->name();
    }
    result += "], outputs=[";
    for(size_t i = 0; i < outputs_.size(); ++i)
    {
        if(i > 0)
        {
            result += ", ";
        }
        result += outputs_[i]->name();
    }
    result += "])";
    return result;
}

//! Only LogicalGraph can modify
void OpNode::add_input(LogicalGraphTensorNode* t)
{
    inputs_.push_back(t);
    t->add_consumer(this);
}

//! Only LogicalGraph can modify
void OpNode::add_output(LogicalGraphTensorNode* t)
{
    outputs_.push_back(t);
    t->set_producer(this);
}

} // namespace nntile::graph
