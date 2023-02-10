/*! @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
 *                           (Skoltech). All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file wrappers/python/nntile/nntile_core.cc
 * Extension module with NNTile wrappers
 *
 * @version 1.0.0
 * @author Aleksandr Mikhalev
 * @date 2023-02-02
 * */

#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <nntile.hh>
#include <sstream>
#include <cstring>

using namespace nntile;
namespace py = pybind11;

// Extend (sub)module with nntile::starpu functionality
void def_mod_starpu(py::module_ &m)
{
    using namespace nntile::starpu;
    py::class_<Config>(m, "Config").
        def(py::init<int, int, int>()).
        def("init", &Config::init).
        def("shutdown", &Config::shutdown);
    m.def("init", init);
    m.def("pause", starpu_pause);
    m.def("resume", starpu_resume);
    m.def("wait_for_all", [](){starpu_task_wait_for_all();
            starpu_mpi_wait_for_all(MPI_COMM_WORLD);});
}

// numpy.ndarray -> Tile
template<typename T>
void tile_from_array(const tile::Tile<T> &tile,
        const py::array_t<T, py::array::f_style | py::array::forcecast> &array)
{
    // Treat special 0-dimensional case, where NNTile assumes 1 element in a
    // tensor, while 0-dimensional numpy array assumes there no array elements
    if(tile.ndim == 0)
    {
        if(array.ndim() != 1)
        {
            throw std::runtime_error("array.ndim() != 1");
        }
        if(array.shape()[0] != 1)
        {
            throw std::runtime_error("array.shape()[0] != 1");
        }
        // Acquire tile and copy a single element
        auto tile_local = tile.acquire(STARPU_W);
        std::memcpy(tile_local.get_ptr(), array.data(), sizeof(T));
        tile_local.release();
        return;
    }
    // Treat other cases
    if(tile.ndim != array.ndim())
    {
        throw std::runtime_error("tile.ndim != array.ndim()");
    }
    for(Index i = 0; i < tile.ndim; ++i)
    {
        if(array.shape()[i] != tile.shape[i])
        {
            throw std::runtime_error("array.shape()[i] != tile.shape[i]");
        }
    }
    // Acquire tile and copy data
    auto tile_local = tile.acquire(STARPU_W);
    std::memcpy(tile_local.get_ptr(), array.data(),
            tile.nelems*sizeof(T));
    tile_local.release();
}

// Tile -> numpy.ndarray
template<typename T>
void tile_to_array(const tile::Tile<T> &tile,
        py::array_t<T, py::array::f_style> &array)
{
    // Treat special 0-dimensional case, where NNTile assumes 1 element in a
    // tensor, while 0-dimensional numpy array assumes there no array elements
    if(tile.ndim == 0)
    {
        if(array.ndim() != 1)
        {
            throw std::runtime_error("array.ndim() != 1");
        }
        if(array.shape()[0] != 1)
        {
            throw std::runtime_error("array.shape()[0] != 1");
        }
        // Acquire tile and copy a single element
        auto tile_local = tile.acquire(STARPU_R);
        std::memcpy(array.mutable_data(), tile_local.get_ptr(), sizeof(T));
        tile_local.release();
        return;
    }
    // Treat other cases
    if(tile.ndim != array.ndim())
    {
        throw std::runtime_error("tile.ndim != array.ndim()");
    }
    for(Index i = 0; i < tile.ndim; ++i)
    {
        if(array.shape()[i] != tile.shape[i])
        {
            throw std::runtime_error("array.shape()[i] != tile.shape[i]");
        }
    }
    // Acquire tile and copy data
    auto tile_local = tile.acquire(STARPU_R);
    std::memcpy(array.mutable_data(), tile_local.get_ptr(),
            tile.nelems*sizeof(T));
    tile_local.release();
}

// Extend (sub)module with nntile::tile::Tile<T>
template<typename T>
void def_class_tile(py::module_ &m, const char *name)
{
    using namespace nntile::tile;
    py::class_<Tile<T>, TileTraits>(m, name, py::multiple_inheritance()).
        def(py::init<const TileTraits &>()).
        def("unregister", &Tile<T>::unregister).
        def("from_array", tile_from_array<T>).
        def("to_array", tile_to_array<T>);
    m.def("tile_from_array", tile_from_array<T>);
    m.def("tile_to_array", tile_to_array<T>);
}

// Extend (sub)module with nntile::tile functionality
void def_mod_tile(py::module_ &m)
{
    using namespace nntile::tile;
    // Define wrapper for the Class
    py::class_<TileTraits>(m, "TileTraits").
        // Constructor
        def(py::init<const std::vector<Index> &>()).
        // __repr__ function for print(object)
        def("__repr__", [](const TileTraits &data){
                std::stringstream stream;
                stream << data;
                return stream.str();}).
        // Number of dimensions
        def_readonly("ndim", &TileTraits::ndim).
        // Shape of a tile
        def_readonly("shape", &TileTraits::shape).
        // Number of elements of a tile
        def_readonly("nelems", &TileTraits::nelems).
        // Linear to index
        def("linear_to_index", &TileTraits::linear_to_index).
        // Index to linear
        def("index_to_linear", &TileTraits::index_to_linear);
    // Define wrappers for Tile<T>
    def_class_tile<fp32_t>(m, "Tile_fp32");
    def_class_tile<fp64_t>(m, "Tile_fp64");
}

// numpy.ndarray -> Tensor
template<typename T>
void tensor_from_array(const tensor::Tensor<T> &tensor,
        const py::array_t<T, py::array::f_style | py::array::forcecast> &array)
{
    // Treat special 0-dimensional case, where NNTile assumes 1 element in a
    // tensor, while 0-dimensional numpy array assumes there no array elements
    if(tensor.ndim == 0)
    {
        if(array.ndim() != 1)
        {
            throw std::runtime_error("array.ndim() != 1");
        }
        if(array.shape()[0] != 1)
        {
            throw std::runtime_error("array.shape()[0] != 1");
        }
        // Acquire tile and copy a single element
        int mpi_rank = starpu_mpi_world_rank();
        auto tile = tensor.get_tile(0);
        if(mpi_rank == tile.mpi_get_rank())
        {
            auto tile_local = tile.acquire(STARPU_W);
            std::memcpy(tile_local.get_ptr(), array.data(), sizeof(T));
            tile_local.release();
        }
        tile.mpi_flush();
        return;
    }
    // Treat other cases
    if(tensor.ndim != array.ndim())
    {
        throw std::runtime_error("tensor.ndim != array.ndim()");
    }
    for(Index i = 0; i < tensor.ndim; ++i)
    {
        if(array.shape()[i] != tensor.shape[i])
        {
            throw std::runtime_error("array.shape()[i] != tensor.shape[i]");
        }
    }
    if(tensor.grid.nelems != 1)
    {
        throw std::runtime_error("tensor.grid.nelems != 1");
    }
    // Acquire tile and copy data
    int mpi_rank = starpu_mpi_world_rank();
    auto tile = tensor.get_tile(0);
    if(mpi_rank == tile.mpi_get_rank())
    {
        auto tile_local = tile.acquire(STARPU_W);
        std::memcpy(tile_local.get_ptr(), array.data(),
                tile.nelems*sizeof(T));
        tile_local.release();
    }
    tile.mpi_flush();
}

// Tensor -> numpy.ndarray
template<typename T>
void tensor_to_array(const tensor::Tensor<T> &tensor,
        py::array_t<T, py::array::f_style> &array)
{
    // Treat special 0-dimensional case, where NNTile assumes 1 element in a
    // tensor, while 0-dimensional numpy array assumes there no array elements
    if(tensor.ndim == 0)
    {
        if(array.ndim() != 1)
        {
            throw std::runtime_error("array.ndim() != 1");
        }
        if(array.shape()[0] != 1)
        {
            throw std::runtime_error("array.shape()[0] != 1");
        }
        // Acquire tile and copy a single element
        int mpi_rank = starpu_mpi_world_rank();
        auto tile = tensor.get_tile(0);
        if(mpi_rank == tile.mpi_get_rank())
        {
            auto tile_local = tile.acquire(STARPU_R);
            std::memcpy(array.mutable_data(), tile_local.get_ptr(), sizeof(T));
            tile_local.release();
        }
        tile.mpi_flush();
        return;
    }
    // Treat other cases
    if(tensor.ndim != array.ndim())
    {
        throw std::runtime_error("tensor.ndim != array.ndim()");
    }
    for(Index i = 0; i < tensor.ndim; ++i)
    {
        if(array.shape()[i] != tensor.shape[i])
        {
            throw std::runtime_error("array.shape()[i] != tensor.shape[i]");
        }
    }
    if(tensor.grid.nelems != 1)
    {
        throw std::runtime_error("tensor.grid.nelems != 1");
    }
    // Acquire tile and copy data
    int mpi_rank = starpu_mpi_world_rank();
    auto tile = tensor.get_tile(0);
    if(mpi_rank == tile.mpi_get_rank())
    {
        auto tile_local = tile.acquire(STARPU_R);
        std::memcpy(array.mutable_data(), tile_local.get_ptr(),
                tile.nelems*sizeof(T));
        tile_local.release();
    }
}

// Extend (sub)module with nntile::tensor::Tensor<T>
template<typename T>
void def_class_tensor(py::module_ &m, const char *name)
{
    using namespace nntile::tensor;
    py::class_<Tensor<T>, TensorTraits>(m, name, py::multiple_inheritance()).
        def(py::init<const TensorTraits &, const std::vector<int> &,
                starpu_mpi_tag_t &>()).
        def_readonly("next_tag", &Tensor<T>::next_tag).
        def("unregister", &Tensor<T>::unregister).
        def("from_array", tensor_from_array<T>).
        def("to_array", tensor_to_array<T>).
        // Get tile
        def("get_tile", static_cast<tile::Tile<T>(Tensor<T>::*)(Index) const>(
                    &Tensor<T>::get_tile));
    m.def("tensor_to_array", tensor_to_array<T>);
    m.def("tensor_from_array", tensor_from_array<T>);
}

// Extend (sub)module with nntile::tensor::distributions functionality
void def_tensor_distributions(py::module_ &m)
{
    using namespace nntile::tensor::distributions;
    m.def("block_cyclic", &block_cyclic);
}

// Extend (sub)module with nntile::tensor functionality
void def_mod_tensor(py::module_ &m)
{
    using namespace nntile::tensor;
    // Define wrapper for TensorTraits
    py::class_<TensorTraits, tile::TileTraits>(m, "TensorTraits",
            py::multiple_inheritance()).
        // Constructor
        def(py::init<const std::vector<Index> &,
                const std::vector<Index> &>()).
        // __repr__ function for print(object)
        def("__repr__", [](const TensorTraits &data){
                std::stringstream stream;
                stream << data;
                return stream.str();}).
        // Shape of corresponding tile
        def("get_tile_shape", &TensorTraits::get_tile_shape).
        // Shape of a grid
        def("get_grid_shape", [](const TensorTraits &data){
                return data.grid.shape;}).
        // Get grid (TileTraits)
        def_readonly("grid", &TensorTraits::grid);
    // Define wrappers for Tensor<T>
    def_class_tensor<fp32_t>(m, "Tensor_fp32");
    def_class_tensor<fp64_t>(m, "Tensor_fp64");
    // Add tensor.distributions submodule
    auto distributions = m.def_submodule("distributions");
    def_tensor_distributions(distributions);
    // Add functions for Tensor<T>
    m.def("gemm_async_fp32", &gemm_async<fp32_t>);
    m.def("gemm_fp32", &gemm<fp32_t>);
    m.def("gemm_async_fp64", &gemm_async<fp64_t>);
    m.def("gemm_fp64", &gemm<fp64_t>);
    // Add activation functions for Tensor<T>
    m.def("relu_async_fp64", &relu_async<fp64_t>);
    m.def("relu_async_fp32", &relu_async<fp32_t>);
    m.def("relu_fp64", &relu<fp64_t>);
    m.def("relu_fp32", &relu<fp32_t>);
    m.def("drelu_async_fp64", &drelu_async<fp64_t>);
    m.def("drelu_async_fp32", &drelu_async<fp32_t>);
    m.def("drelu_fp64", &drelu<fp64_t>);
    m.def("drelu_fp32", &drelu<fp32_t>);
    // Add other functions for Tensor<T>
    m.def("sumnorm_async_fp64", &sumnorm_async<fp64_t>);
    m.def("sumnorm_async_fp32", &sumnorm_async<fp32_t>);
    m.def("sumnorm_fp64", &sumnorm<fp64_t>);
    m.def("sumnorm_fp32", &sumnorm<fp32_t>);
    m.def("softmax_async_fp64", &softmax_async<fp64_t>);
    m.def("softmax_async_fp32", &softmax_async<fp32_t>);
    m.def("softmax_fp64", &softmax<fp64_t>);
    m.def("softmax_fp32", &softmax<fp32_t>);
    m.def("scatter_async_fp64", &scatter_async<fp64_t>);
    m.def("scatter_async_fp32", &scatter_async<fp32_t>);
    m.def("scatter_fp64", &scatter<fp64_t>);
    m.def("scatter_fp32", &scatter<fp32_t>);
    m.def("randn_async_fp64", &randn_async<fp64_t>);
    m.def("randn_async_fp32", &randn_async<fp32_t>);
    m.def("randn_fp64", &randn<fp64_t>);
    m.def("randn_fp32", &randn<fp32_t>);
    m.def("prod_async_fp64", &prod_async<fp64_t>);
    m.def("prod_async_fp32", &prod_async<fp32_t>);
    m.def("prod_fp64", &prod<fp64_t>);
    m.def("prod_fp32", &prod<fp32_t>);
    m.def("nrm2_async_fp64", &nrm2_async<fp64_t>);
    m.def("nrm2_async_fp32", &nrm2_async<fp32_t>);
    m.def("nrm2_fp64", &nrm2<fp64_t>);
    m.def("nrm2_fp32", &nrm2<fp32_t>);
    m.def("normalize_async_fp64", &normalize_async<fp64_t>);
    m.def("normalize_async_fp32", &normalize_async<fp32_t>);
    m.def("normalize_fp64", &normalize<fp64_t>);
    m.def("normalize_fp32", &normalize<fp32_t>);
    m.def("maxsumexp_async_fp64", &maxsumexp_async<fp64_t>);
    m.def("maxsumexp_async_fp32", &maxsumexp_async<fp32_t>);
    m.def("maxsumexp_fp64", &maxsumexp<fp64_t>);
    m.def("maxsumexp_fp32", &maxsumexp<fp32_t>);
    m.def("bias_async_fp64", &bias_async<fp64_t>);
    m.def("bias_async_fp32", &bias_async<fp32_t>);
    m.def("bias_fp64", &bias<fp64_t>);
    m.def("bias_fp32", &bias<fp32_t>);
    m.def("gather_async_fp64", &gather_async<fp64_t>);
    m.def("gather_async_fp32", &gather_async<fp32_t>);
    m.def("gather_fp64", &gather<fp64_t>);
    m.def("gather_fp32", &gather<fp32_t>);
    m.def("copy_intersection_async_fp64", &copy_intersection_async<fp64_t>);
    m.def("copy_intersection_async_fp32", &copy_intersection_async<fp32_t>);
    m.def("copy_intersection_fp64", &copy_intersection<fp64_t>);
    m.def("copy_intersection_fp32", &copy_intersection<fp32_t>);
    m.def("copy_async_fp64", &copy_async<fp64_t>);
    m.def("copy_async_fp32", &copy_async<fp32_t>);
    m.def("copy_fp64", &copy<fp64_t>);
    m.def("copy_fp32", &copy<fp32_t>);
    m.def("clear_async_fp64", &clear_async<fp64_t>);
    m.def("clear_async_fp32", &clear_async<fp32_t>);
    m.def("clear_fp64", &clear<fp64_t>);
    m.def("clear_fp32", &clear<fp32_t>);
        
    m.def("axpy_async_fp64", &axpy_async<fp64_t>);
    m.def("axpy_async_fp32", &axpy_async<fp32_t>);
    m.def("axpy_fp64", &axpy<fp64_t>);
    m.def("axpy_fp32", &axpy<fp32_t>);

    m.def("axpy2_async_fp64", &axpy2_async<fp64_t>);
    m.def("axpy2_async_fp32", &axpy2_async<fp32_t>);
    m.def("axpy2_fp64", &axpy2<fp64_t>);
    m.def("axpy2_fp32", &axpy2<fp32_t>);

    m.def("sqrt_async_fp64", &sqrt_async<fp64_t>);
    m.def("sqrt_async_fp32", &sqrt_async<fp32_t>);
    m.def("sqrt_fp64", &sqrt<fp64_t>);
    m.def("sqrt_fp32", &sqrt<fp32_t>);
}

// Main extension module with all wrappers
PYBIND11_MODULE(nntile_core, m)
{
    // Add starpu submodule
    auto starpu = m.def_submodule("starpu");
    def_mod_starpu(starpu);
    // Add tile submodule
    auto tile = m.def_submodule("tile");
    def_mod_tile(tile);
    // Add tensor submodule
    auto tensor = m.def_submodule("tensor");
    def_mod_tensor(tensor);
    // Define TransOp class and corresponding constants
    py::class_<TransOp>(m, "TransOp").
        // Constructor
        def(py::init<const enum TransOp::Value &>());
    m.attr("notrans") = py::cast(new TransOp(TransOp::NoTrans));
    m.attr("trans") = py::cast(new TransOp(TransOp::Trans));
}

