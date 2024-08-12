/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/model/deep_linear.hh
 * Deep linear network model
 *
 * @version 1.1.0
 * */

#pragma once

#include <nntile/model/base.hh>
#include <nntile/tensor.hh>
#include <nntile/starpu.hh>
#include <nntile/layer.hh>

namespace nntile::model
{

template<typename T>
class DeepLinear: public Base<T>
{
    static std::vector<std::shared_ptr<layer::Base<T>>> _gen_layers(
            const std::vector<Index> &shape,
            const std::vector<Index> &basetile,
            const std::vector<int> mpi_grid,
            starpu_mpi_tag_t &last_tag)
    {
        Index nlayers = shape.size() - 2;
        int mpi_size = starpu_mpi_world_size();
        std::vector<std::shared_ptr<layer::Base<T>>> layers;
        for(Index i = 0; i < nlayers; ++i)
        {
            std::vector<Index> layer_shape(2), layer_basetile(2),
                input_shape(2), input_basetile(2), output_shape(2),
                output_basetile(2);
            layer_shape[0] = shape[i+2];
            layer_shape[1] = shape[i+1];
            layer_basetile[0] = basetile[i+2];
            layer_basetile[1] = basetile[i+1];
            input_shape[0] = shape[i+1];
            input_shape[1] = shape[0];
            input_basetile[0] = basetile[i+1];
            input_basetile[1] = basetile[0];
            output_shape[0] = shape[i+2];
            output_shape[1] = shape[0];
            output_basetile[0] = basetile[i+2];
            output_basetile[1] = basetile[0];
            tensor::TensorTraits traits(layer_shape, layer_basetile),
                input_traits(input_shape, input_basetile),
                output_traits(output_shape, output_basetile);
            std::vector<int> distr = tensor::distributions::block_cyclic(
                    traits.grid.shape, mpi_grid, 0, mpi_size);
            tensor::Tensor<T> weight(traits, distr, last_tag),
                grad_weight(traits, distr, last_tag);
            auto layer = new layer::Linear<T>(input_traits, output_traits,
                    {weight}, {grad_weight});
            layers.push_back(std::shared_ptr<layer::Base<T>>(layer));
        }
        return layers;
    }
public:
    DeepLinear(const std::vector<Index> &shape,
            const std::vector<Index> &basetile,
            const std::vector<int> mpi_grid,
            starpu_mpi_tag_t &last_tag):
        Base<T>(_gen_layers(shape, basetile, mpi_grid, last_tag))
    {
    }
};

// Explicit instantiation
extern template
class DeepLinear<fp32_t>;

extern template
class DeepLinear<fp64_t>;

} // namespace nntile::model
