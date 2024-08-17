/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file wrappers/python/nntile/nntile_core.cc
 * Extension module with NNTile wrappers
 *
 * @version 1.1.0
 * */

#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <nntile.hh>
#include <sstream>
#include <cstring>
#include <thread>

using pybind11::literals::operator""_a;
using namespace nntile;
namespace py = pybind11;

constexpr auto _wait_for_all_sleep_time = std::chrono::milliseconds(1);

// Extend (sub)module with nntile::starpu functionality
void def_mod_starpu(py::module_ &m)
{
    using namespace nntile::starpu;
    using namespace std::chrono_literals;
    py::class_<Config>(m, "Config").
        def(py::init<int, int, int, int, const char *, int>(),
                py::arg("ncpus_")=-1, py::arg("ncuda_")=-1,
                py::arg("cublas_")=-1, py::arg("logger")=0,
                py::arg("logger_server_addr")="",
                py::arg("logger_server_port")=5001).
        def("shutdown", &Config::shutdown);
    m.def("init", init);
    m.def("pause", starpu_pause);
    m.def("resume", starpu_resume);
    m.def("wait_for_all", [](){
            while(true)
            {
                int nsubmitted = starpu_task_nsubmitted();
                //int nready = starpu_task_nready();
                //std::cout << "S=" << nsubmitted << " R=" << nready << "\n";
                //if (nready > nsubmitted)
                //{
                //    std::cout << "======================\n";
                //}
                std::this_thread::sleep_for(_wait_for_all_sleep_time);
                if(nsubmitted == 0)
                {
                    break;
                }
                if(PyErr_CheckSignals() != 0)
                {
                    throw py::error_already_set();
                }
            }
            starpu_mpi_wait_for_all(MPI_COMM_WORLD);});
    m.def("restrict_cuda", [](){restrict_where(STARPU_CUDA);});
    m.def("restrict_cpu", [](){restrict_where(STARPU_CPU);});
    m.def("restrict_restore", [](){restore_where();});
    m.def("profiling_init", [](){
            //starpu_profiling_init();
            });
    m.def("profiling_enable", [](){
            //starpu_profiling_status_set(STARPU_PROFILING_ENABLE);
            starpu_fxt_start_profiling();});
    m.def("profiling_disable", [](){
            //starpu_profiling_status_set(STARPU_PROFILING_DISABLE);
            starpu_fxt_stop_profiling();});
}

//! Copy from raw pointer to a raw pointer with a possible conversion
template<typename T, typename Y, bool trivial_copy>
void copy_raw(Index nelems, const T *src, Y *dst)
{
    if constexpr (trivial_copy)
    {
        std::memcpy(dst, src, nelems*sizeof(T));
    }
    else
    {
        for(Index i = 0; i < nelems; ++i)
        {
            dst[i] = static_cast<Y>(src[i]);
        }
    }
}

// numpy.ndarray -> Tile
template<typename T>
void tile_from_array(const tile::Tile<T> &tile,
        const py::array_t<typename T::repr_t,
            py::array::f_style | py::array::forcecast> &array)
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
#ifndef STARPU_SIMGRID
        using Y = typename T::repr_t;
        constexpr bool triv = T::trivial_copy_from_compat;
        copy_raw<Y, T, triv>(1, array.data(), tile_local.get_ptr());
#endif // STARPU_SIMGRID
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
#ifndef STARPU_SIMGRID
    using Y = typename T::repr_t;
    constexpr bool triv = T::trivial_copy_from_compat;
    copy_raw<Y, T, triv>(tile.nelems, array.data(), tile_local.get_ptr());
#endif // STARPU_SIMGRID
    tile_local.release();
}

// Tile -> numpy.ndarray
template<typename T>
void tile_to_array(const tile::Tile<T> &tile,
        py::array_t<typename T::repr_t, py::array::f_style> &array)
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
#ifndef STARPU_SIMGRID
        using Y = typename T::repr_t;
        constexpr bool triv = T::trivial_copy_from_compat;
        copy_raw<T, Y, triv>(1, tile_local.get_ptr(), array.mutable_data());
#endif // STARPU_SIMGRID
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
#ifndef STARPU_SIMGRID
    using Y = typename T::repr_t;
    constexpr bool triv = T::trivial_copy_from_compat;
    copy_raw<T, Y, triv>(tile.nelems, tile_local.get_ptr(),
        array.mutable_data());
#endif // STARPU_SIMGRID
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
        // Strides of a tile
        def_readonly("stride", &TileTraits::stride).
        // Number of elements of a tile
        def_readonly("nelems", &TileTraits::nelems).
        // Linear to index
        def("linear_to_index", &TileTraits::linear_to_index).
        // Index to linear
        def("index_to_linear", &TileTraits::index_to_linear);
    // Define wrappers for Tile<T>
    def_class_tile<fp32_t>(m, "Tile_fp32");
    def_class_tile<fp32_fast_tf32_t>(m, "Tile_fp32_fast_tf32");
    def_class_tile<fp64_t>(m, "Tile_fp64");
    def_class_tile<bf16_t>(m, "Tile_bf16");
}

// numpy.ndarray -> Tensor
template<typename T>
void tensor_from_array(const tensor::Tensor<T> &tensor,
        const py::array_t<typename T::repr_t,
            py::array::f_style | py::array::forcecast> &array)
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
#ifndef STARPU_SIMGRID
            using Y = typename T::repr_t;
            constexpr bool triv = T::trivial_copy_from_compat;
            copy_raw<Y, T, triv>(1, array.data(), tile_local.get_ptr());
#endif // STARPU_SIMGRID
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
    // Create temporary single-tile tensor
    tensor::TensorTraits tmp_traits(tensor.shape, tensor.shape);
    std::int64_t tmp_tag = 0;
    int flag;
    //starpu_mpi_comm_get_attr(MPI_COMM_WORLD, STARPU_MPI_TAG_UB, &tmp_tag, \
    //        &flag);
    std::vector<int> tmp_distr{0};
    tensor::Tensor<T> tmp(tmp_traits, tmp_distr, tmp_tag);
    // Acquire tile and copy data
    int mpi_rank = starpu_mpi_world_rank();
    auto tile = tmp.get_tile(0);
    if(mpi_rank == tile.mpi_get_rank())
    {
        auto tile_local = tile.acquire(STARPU_W);
#ifndef STARPU_SIMGRID
        using Y = typename T::repr_t;
        constexpr bool triv = T::trivial_copy_from_compat;
        copy_raw<Y, T, triv>(tile.nelems, array.data(), tile_local.get_ptr());
#endif // STARPU_SIMGRID
        tile_local.release();
    }
    tensor::scatter<T>(tmp, tensor);
    tmp.unregister();
    tensor.mpi_flush();
}

// Tensor -> numpy.ndarray
template<typename T>
void tensor_to_array(const tensor::Tensor<T> &tensor,
        py::array_t<typename T::repr_t, py::array::f_style> &array)
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
#ifndef STARPU_SIMGRID
            using Y = typename T::repr_t;
            constexpr bool triv = T::trivial_copy_from_compat;
            copy_raw<T, Y, triv>(1, tile_local.get_ptr(),
                array.mutable_data());
#endif // STARPU_SIMGRID
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
    // Create temporary single-tile tensor
    tensor::TensorTraits tmp_traits(tensor.shape, tensor.shape);
    std::int64_t tmp_tag = 0;
    int flag;
    //starpu_mpi_comm_get_attr(MPI_COMM_WORLD, STARPU_MPI_TAG_UB, &tmp_tag, \
            &flag);
    std::vector<int> tmp_distr{0};
    tensor::Tensor<T> tmp(tmp_traits, tmp_distr, tmp_tag);
    tensor::gather<T>(tensor, tmp);
    // Acquire tile and copy data
    int mpi_rank = starpu_mpi_world_rank();
    auto tile = tmp.get_tile(0);
    if(mpi_rank == tile.mpi_get_rank())
    {
        auto tile_local = tile.acquire(STARPU_R);
#ifndef STARPU_SIMGRID
        using Y = typename T::repr_t;
        constexpr bool triv = T::trivial_copy_from_compat;
        copy_raw<T, Y, triv>(tile.nelems, tile_local.get_ptr(),
            array.mutable_data());
#endif // STARPU_SIMGRID
        tile_local.release();
    }
    tmp.unregister();
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
        // Temporary disable invalidate_submit and use wont_use instead
        def("invalidate_submit", &Tensor<T>::invalidate_submit).
        //def("invalidate_submit", &Tensor<T>::wont_use).
        def("wont_use", &Tensor<T>::wont_use).
        def("from_array", &tensor_from_array<T>).
        def("to_array", &tensor_to_array<T>).

        def("set_reduction_add", &Tensor<T>::set_reduction_add).
        def("set_reduction_hypot", &Tensor<T>::set_reduction_hypot).
        def("set_reduction_maxsumexp", &Tensor<T>::set_reduction_maxsumexp).
        def("print_scalar_async", &Tensor<T>::print_scalar_async).
        // Get tile
        def("get_tile", static_cast<tile::Tile<T>(Tensor<T>::*)(Index) const>(
                    &Tensor<T>::get_tile)).
        def("get_nbytes", &Tensor<T>::get_nbytes).
        def_readonly("distribution", &Tensor<T>::tile_distr);
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
        // Get basetile shape
        def_readonly("basetile_shape", &TensorTraits::basetile_shape).
        // Shape of corresponding tile
        def("get_tile_shape", &TensorTraits::get_tile_shape).
        // Shape of a grid
        def("get_grid_shape", [](const TensorTraits &data){
                return data.grid.shape;}).
        // Get grid (TileTraits)
        def_readonly("grid", &TensorTraits::grid);
    // Define wrappers for Tensor<T>
    def_class_tensor<nntile::int64_t>(m, "Tensor_int64");
    def_class_tensor<bool_t>(m, "Tensor_bool");
    def_class_tensor<fp64_t>(m, "Tensor_fp64");
    def_class_tensor<fp32_fast_tf32_t>(m, "Tensor_fp32_fast_tf32");
    def_class_tensor<fp32_t>(m, "Tensor_fp32");
    def_class_tensor<bf16_t>(m, "Tensor_bf16");
    // def_class_tensor<fp16_t>(m, "Tensor_fp16");
    // Add tensor.distributions submodule
    auto distributions = m.def_submodule("distributions");
    def_tensor_distributions(distributions);

    // Add functions for Tensor<T>
    m.def("gemm_async_fp64", &gemm_async<fp64_t>);
    m.def("gemm_async_fp32", &gemm_async<fp32_t>);
    m.def("gemm_async_fp32_fast_tf32", &gemm_async<fp32_fast_tf32_t>);
    m.def("gemm_async_bf16", &gemm_async<bf16_t>);
    //m.def("gemm_async_fp16", &gemm_async<fp16_t>);

    m.def("gemm_fp64", &gemm<fp64_t>);
    m.def("gemm_fp32", &gemm<fp32_t>);
    m.def("gemm_fp32_fast_tf32", &gemm<fp32_fast_tf32_t>);
    m.def("gemm_bf16", &gemm<bf16_t>);
    //m.def("gemm_fp16", &gemm<fp16_t>);

    // Add activation functions for Tensor<T>
    m.def("relu_async_fp64", &relu_async<fp64_t>);
    m.def("relu_async_fp32_fast_tf32", &relu_async<fp32_fast_tf32_t>);
    m.def("relu_async_fp32", &relu_async<fp32_t>);
    m.def("relu_fp64", &relu<fp64_t>);
    m.def("relu_fp32", &relu<fp32_t>);
    m.def("relu_fp32_fast_tf32", &relu<fp32_fast_tf32_t>);

    m.def("relu_forward_async_fp64", &relu_forward_async<fp64_t>);
    m.def("relu_forward_async_fp32", &relu_forward_async<fp32_t>);
    m.def("relu_forward_async_bf16", &relu_forward_async<bf16_t>);
    m.def("relu_forward_async_fp32_fast_tf32", &relu_forward_async<fp32_fast_tf32_t>);
    m.def("relu_forward_fp64", &relu_forward<fp64_t>);
    m.def("relu_forward_fp32", &relu_forward<fp32_t>);
    m.def("relu_forward_bf16", &relu_forward<bf16_t>);
    m.def("relu_forward_fp32_fast_tf32", &relu_forward<fp32_fast_tf32_t>);

    m.def("silu_forward_async_fp64", &silu_forward_async<fp64_t>);
    m.def("silu_forward_async_fp32", &silu_forward_async<fp32_t>);
    m.def("silu_forward_async_bf16", &silu_forward_async<bf16_t>);
    m.def("silu_forward_async_fp32_fast_tf32", &silu_forward_async<fp32_fast_tf32_t>);
    m.def("silu_forward_fp64", &silu_forward<fp64_t>);
    m.def("silu_forward_fp32", &silu_forward<fp32_t>);
    m.def("silu_forward_bf16", &silu_forward<bf16_t>);
    m.def("silu_forward_fp32_fast_tf32", &silu_forward<fp32_fast_tf32_t>);

    m.def("relu_backward_async_fp64", &relu_backward_async<fp64_t>);
    m.def("relu_backward_async_bf16", &relu_backward_async<bf16_t>);
    m.def("relu_backward_async_fp32", &relu_backward_async<fp32_t>);
    m.def("relu_backward_async_fp32_fast_tf32", &relu_backward_async<fp32_fast_tf32_t>);
    m.def("relu_backward_fp64", &relu_backward<fp64_t>);
    m.def("relu_backward_bf16", &relu_backward<bf16_t>);
    m.def("relu_backward_fp32", &relu_backward<fp32_t>);
    m.def("relu_backward_fp32_fast_tf32", &relu_backward<fp32_fast_tf32_t>);

    m.def("silu_backward_async_fp64", &silu_backward_async<fp64_t>);
    m.def("silu_backward_async_bf16", &silu_backward_async<bf16_t>);
    m.def("silu_backward_async_fp32", &silu_backward_async<fp32_t>);
    m.def("silu_backward_async_fp32_fast_tf32", &silu_backward_async<fp32_fast_tf32_t>);
    m.def("silu_backward_fp64", &silu_backward<fp64_t>);
    m.def("silu_backward_bf16", &silu_backward<bf16_t>);
    m.def("silu_backward_fp32", &silu_backward<fp32_t>);
    m.def("silu_backward_fp32_fast_tf32", &silu_backward<fp32_fast_tf32_t>);

    m.def("drelu_async_fp64", &drelu_async<fp64_t>);
    m.def("drelu_async_fp32", &drelu_async<fp32_t>);
    m.def("drelu_fp64", &drelu<fp64_t>);
    m.def("drelu_fp32", &drelu<fp32_t>);
    // Add other functions for Tensor<T>
    m.def("fill_async_fp64", &fill_async<fp64_t>);
    m.def("fill_async_bf16", &fill_async<bf16_t>);
    m.def("fill_async_fp32", &fill_async<fp32_t>);
    m.def("fill_async_fp32_fast_tf32", &fill_async<fp32_fast_tf32_t>);
    m.def("fill_fp64", &fill<fp64_t>);
    m.def("fill_bf16", &fill<bf16_t>);
    m.def("fill_fp32", &fill<fp32_t>);
    m.def("fill_fp32_fast_tf32", &fill<fp32_fast_tf32_t>);

    m.def("sum_slice_async_fp64", &sum_slice_async<fp64_t>);
    m.def("sum_slice_async_bf16", &sum_slice_async<bf16_t>);
    m.def("sum_slice_async_fp32", &sum_slice_async<fp32_t>);
    m.def("sum_slice_async_fp32_fast_tf32", &sum_slice_async<fp32_fast_tf32_t>);
    m.def("sum_slice_fp64", &sum_slice<fp64_t>);
    m.def("sum_slice_fp32", &sum_slice<fp32_t>);
    m.def("sum_slice_fp32_fast_tf32", &sum_slice<fp32_fast_tf32_t>);
    m.def("sum_slice_bf16", &sum_slice<bf16_t>);

    m.def("sum_fiber_async_fp64", &sum_fiber_async<fp64_t>);
    m.def("sum_fiber_async_bf16", &sum_fiber_async<bf16_t>);
    m.def("sum_fiber_async_fp32", &sum_fiber_async<fp32_t>);
    m.def("sum_fiber_async_fp32_fast_tf32", &sum_fiber_async<fp32_fast_tf32_t>);
    m.def("sum_fiber_fp64", &sum_fiber<fp64_t>);
    m.def("sum_fiber_fp32", &sum_fiber<fp32_t>);
    m.def("sum_fiber_bf16", &sum_fiber<bf16_t>);
    m.def("sum_fiber_fp32_fast_tf32", &sum_fiber<fp32_fast_tf32_t>);

    m.def("norm_fiber_async_fp64", &norm_fiber_async<fp64_t>);
    m.def("norm_fiber_async_bf16", &norm_fiber_async<bf16_t>);
    m.def("norm_fiber_async_fp32", &norm_fiber_async<fp32_t>);
    m.def("norm_fiber_async_fp32_fast_tf32", &norm_fiber_async<fp32_fast_tf32_t>);
    m.def("norm_fiber_fp64", &norm_fiber<fp64_t>);
    m.def("norm_fiber_fp32", &norm_fiber<fp32_t>);
    m.def("norm_fiber_bf16", &norm_fiber<bf16_t>);
    m.def("norm_fiber_fp32_fast_tf32", &norm_fiber<fp32_fast_tf32_t>);

    m.def("norm_slice_async_fp64", &norm_slice_async<fp64_t>);
    m.def("norm_slice_async_bf16", &norm_slice_async<bf16_t>);
    m.def("norm_slice_async_fp32", &norm_slice_async<fp32_t>);
    m.def("norm_slice_async_fp32_fast_tf32", &norm_slice_async<fp32_fast_tf32_t>);
    m.def("norm_slice_fp64", &norm_slice<fp64_t>);
    m.def("norm_slice_fp32", &norm_slice<fp32_t>);
    m.def("norm_slice_bf16", &norm_slice<bf16_t>);
    m.def("norm_slice_fp32_fast_tf32", &norm_slice<fp32_fast_tf32_t>);

    m.def("pow_async_fp64", &pow_async<fp64_t>);
    m.def("pow_async_fp32", &pow_async<fp32_t>);
    m.def("pow_fp64", &pow<fp64_t>);
    m.def("pow_fp32", &pow<fp32_t>);

    m.def("sumnorm_async_fp64", &sumnorm_async<fp64_t>);
    m.def("sumnorm_async_fp32", &sumnorm_async<fp32_t>);
    m.def("sumnorm_fp64", &sumnorm<fp64_t>);
    m.def("sumnorm_fp32", &sumnorm<fp32_t>);

    m.def("flash_softmax_gemm_async_fp64", &flash_softmax_gemm_async<fp64_t>);
    m.def("flash_softmax_gemm_async_bf16", &flash_softmax_gemm_async<bf16_t>);
    m.def("flash_softmax_gemm_async_fp32", &flash_softmax_gemm_async<fp32_t>);
    m.def("flash_softmax_gemm_async_fp32_fast_tf32", &flash_softmax_gemm_async<fp32_fast_tf32_t>);
    m.def("flash_softmax_gemm_fp64", &flash_softmax_gemm<fp64_t>);
    m.def("flash_softmax_gemm_bf16", &flash_softmax_gemm<bf16_t>);
    m.def("flash_softmax_gemm_fp32", &flash_softmax_gemm<fp32_t>);
    m.def("flash_softmax_gemm_fp32_fast_tf32", &flash_softmax_gemm<fp32_fast_tf32_t>);

    m.def("flash_softmax_gemm_backward_async_fp64", &flash_softmax_gemm_backward_async<fp64_t>);
    m.def("flash_softmax_gemm_backward_async_bf16", &flash_softmax_gemm_backward_async<bf16_t>);
    m.def("flash_softmax_gemm_backward_async_fp32", &flash_softmax_gemm_backward_async<fp32_t>);
    m.def("flash_softmax_gemm_backward_async_fp32_fast_tf32", &flash_softmax_gemm_backward_async<fp32_fast_tf32_t>);
    m.def("flash_softmax_gemm_backward_fp64", &flash_softmax_gemm_backward<fp64_t>);
    m.def("flash_softmax_gemm_backward_fp32", &flash_softmax_gemm_backward<fp32_t>);
    m.def("flash_softmax_gemm_backward_bf16", &flash_softmax_gemm_backward<bf16_t>);
    m.def("flash_softmax_gemm_backward_fp32_fast_tf32", &flash_softmax_gemm_backward<fp32_fast_tf32_t>);

    m.def("softmax_async_fp64", &softmax_async<fp64_t>);
    m.def("softmax_async_bf16", &softmax_async<bf16_t>);
    m.def("softmax_async_fp32", &softmax_async<fp32_t>);
    m.def("softmax_async_fp32_fast_tf32", &softmax_async<fp32_fast_tf32_t>);
    m.def("softmax_fp64", &softmax<fp64_t>);
    m.def("softmax_fp32", &softmax<fp32_t>);
    m.def("softmax_bf16", &softmax<bf16_t>);
    m.def("softmax_fp32_fast_tf32", &softmax<fp32_fast_tf32_t>);

    m.def("softmax_inplace_async_fp64", &softmax_inplace_async<fp64_t>);
    m.def("softmax_inplace_async_bf16", &softmax_inplace_async<bf16_t>);
    m.def("softmax_inplace_async_fp32", &softmax_inplace_async<fp32_t>);
    m.def("softmax_inplace_async_fp32_fast_tf32", &softmax_inplace_async<fp32_fast_tf32_t>);
    m.def("softmax_inplace_fp64", &softmax_inplace<fp64_t>);
    m.def("softmax_inplace_bf16", &softmax_inplace<bf16_t>);
    m.def("softmax_inplace_fp32", &softmax_inplace<fp32_t>);
    m.def("softmax_inplace_fp32_fast_tf32", &softmax_inplace<fp32_fast_tf32_t>);

    m.def("scatter_async_fp64", &scatter_async<fp64_t>);
    m.def("scatter_async_fp32", &scatter_async<fp32_t>);
    m.def("scatter_async_int64", &scatter_async<nntile::int64_t>);
    m.def("scatter_async_bool", &scatter_async<bool_t>);
    m.def("scatter_async_bf16", &scatter_async<bf16_t>);
    m.def("scatter_fp64", &scatter<fp64_t>);
    m.def("scatter_fp32", &scatter<fp32_t>);
    m.def("scatter_int64", &scatter<nntile::int64_t>);
    m.def("scatter_bool", &scatter<bool_t>);
    m.def("scatter_bf16", &scatter<bf16_t>);

    m.def("randn_async_fp64", &randn_async<fp64_t>);
    m.def("randn_async_fp32", &randn_async<fp32_t>);
    m.def("randn_async_fp32_fast_tf32", &randn_async<fp32_fast_tf32_t>);
    m.def("randn_async_bf16", &randn_async<bf16_t>);
    m.def("randn_fp64", &randn<fp64_t>);
    m.def("randn_fp32", &randn<fp32_t>);
    m.def("randn_fp32_fast_tf32", &randn<fp32_fast_tf32_t>);
    m.def("randn_bf16", &randn<bf16_t>);

    m.def("prod_async_fp64", &prod_async<fp64_t>);
    m.def("prod_async_bf16", &prod_async<bf16_t>);
    m.def("prod_async_fp32", &prod_async<fp32_t>);
    m.def("prod_async_fp32_fast_tf32", &prod_async<fp32_fast_tf32_t>);
    m.def("prod_fp64", &prod<fp64_t>);
    m.def("prod_fp32", &prod<fp32_t>);
    m.def("prod_bf16", &prod<bf16_t>);
    m.def("prod_fp32_fast_tf32", &prod<fp32_fast_tf32_t>);

    m.def("prod_inplace_async_fp64", &prod_inplace_async<fp64_t>);
    m.def("prod_inplace_async_bf16", &prod_inplace_async<bf16_t>);
    m.def("prod_inplace_async_fp32", &prod_inplace_async<fp32_t>);
    m.def("prod_inplace_async_fp32_fast_tf32",
            &prod_inplace_async<fp32_fast_tf32_t>);
    m.def("prod_inplace_fp64", &prod_inplace<fp64_t>);
    m.def("prod_inplace_fp32", &prod_inplace<fp32_t>);
    m.def("prod_inplace_bf16", &prod_inplace<bf16_t>);
    m.def("prod_inplace_fp32_fast_tf32", &prod_inplace<fp32_fast_tf32_t>);

    m.def("nrm2_async_fp64", &nrm2_async<fp64_t>);
    m.def("nrm2_async_fp32", &nrm2_async<fp32_t>);
    m.def("nrm2_fp64", &nrm2<fp64_t>);
    m.def("nrm2_fp32", &nrm2<fp32_t>);

    m.def("normalize_async_fp64", &normalize_async<fp64_t>, "gamma_beta"_a, "src"_a, "dst"_a, "size"_a, "eps"_a, "axis"_a);
    m.def("normalize_async_fp32", &normalize_async<fp32_t>, "gamma_beta"_a, "src"_a, "dst"_a, "size"_a, "eps"_a, "axis"_a);
    m.def("normalize_fp64", &normalize<fp64_t>, "gamma_beta"_a, "src"_a, "dst"_a, "size"_a, "eps"_a, "axis"_a);
    m.def("normalize_fp32", &normalize<fp32_t>, "gamma_beta"_a, "src"_a, "dst"_a, "size"_a, "eps"_a, "axis"_a);

    m.def("flash_maxsumexp_async_fp64", &flash_maxsumexp_async<fp64_t>);
    m.def("flash_maxsumexp_async_bf16", &flash_maxsumexp_async<bf16_t>);
    m.def("flash_maxsumexp_async_fp32", &flash_maxsumexp_async<fp32_t>);
    m.def("flash_maxsumexp_async_fp32_fast_tf32", &flash_maxsumexp_async<fp32_fast_tf32_t>);
    m.def("flash_maxsumexp_fp64", &flash_maxsumexp<fp64_t>);
    m.def("flash_maxsumexp_fp32", &flash_maxsumexp<fp32_t>);
    m.def("flash_maxsumexp_bf16", &flash_maxsumexp<bf16_t>);
    m.def("flash_maxsumexp_fp32_fast_tf32", &flash_maxsumexp<fp32_fast_tf32_t>);

    m.def("maxsumexp_async_fp64", &maxsumexp_async<fp64_t>);
    m.def("maxsumexp_async_bf16", &maxsumexp_async<bf16_t>);
    m.def("maxsumexp_async_fp32", &maxsumexp_async<fp32_t>);
    m.def("maxsumexp_async_fp32_fast_tf32", &maxsumexp_async<fp32_fast_tf32_t>);
    m.def("maxsumexp_fp64", &maxsumexp<fp64_t>);
    m.def("maxsumexp_bf16", &maxsumexp<bf16_t>);
    m.def("maxsumexp_fp32", &maxsumexp<fp32_t>);
    m.def("maxsumexp_fp32_fast_tf32", &maxsumexp<fp32_fast_tf32_t>);

    m.def("add_slice_async_fp64", &add_slice_async<fp64_t>);
    m.def("add_slice_async_bf16", &add_slice_async<bf16_t>);
    m.def("add_slice_async_fp32", &add_slice_async<fp32_t>);
    m.def("add_slice_async_fp32_fast_tf32", &add_slice_async<fp32_fast_tf32_t>);
    m.def("add_slice_fp64", &add_slice<fp64_t>);
    m.def("add_slice_bf16", &add_slice<bf16_t>);
    m.def("add_slice_fp32", &add_slice<fp32_t>);
    m.def("add_slice_fp32_fast_tf32", &add_slice<fp32_fast_tf32_t>);

    m.def("add_slice3_async_fp64", &add_slice3_async<fp64_t>);
    m.def("add_slice3_async_bf16", &add_slice3_async<bf16_t>);
    m.def("add_slice3_async_fp32", &add_slice3_async<fp32_t>);
    m.def("add_slice3_async_fp32_fast_tf32", &add_slice3_async<fp32_fast_tf32_t>);
    m.def("add_slice3_fp64", &add_slice3<fp64_t>);
    m.def("add_slice3_fp32", &add_slice3<fp32_t>);
    m.def("add_slice3_bf16", &add_slice3<bf16_t>);
    m.def("add_slice3_fp32_fast_tf32", &add_slice3<fp32_fast_tf32_t>);

    m.def("add_async_fp64", &add_async<fp64_t>);
    m.def("add_async_fp32", &add_async<fp32_t>);
    m.def("add_async_fp32_fast_tf32", &add_async<fp32_fast_tf32_t>);
    m.def("add_async_bf16", &add_async<bf16_t>);
    m.def("add_fp64", &add<fp64_t>);
    m.def("add_bf16", &add<bf16_t>);
    m.def("add_fp32", &add<fp32_t>);
    m.def("add_fp32_fast_tf32", &add<fp32_fast_tf32_t>);

    m.def("add_scalar_async_fp64", &add_scalar_async<fp64_t>);
    m.def("add_scalar_async_fp32", &add_scalar_async<fp32_t>);
    m.def("add_scalar_fp64", &add_scalar<fp64_t>);
    m.def("add_scalar_fp32", &add_scalar<fp32_t>);

    m.def("add_fiber_async_fp64", &add_fiber_async<fp64_t>);
    m.def("add_fiber_async_bf16", &add_fiber_async<bf16_t>);
    m.def("add_fiber_async_fp32", &add_fiber_async<fp32_t>);
    m.def("add_fiber_async_fp32_fast_tf32", &add_fiber_async<fp32_fast_tf32_t>);
    m.def("add_fiber_fp64", &add_fiber<fp64_t>);
    m.def("add_fiber_fp32", &add_fiber<fp32_t>);
    m.def("add_fiber_bf16", &add_fiber<bf16_t>);
    m.def("add_fiber_fp32_fast_tf32", &add_fiber<fp32_fast_tf32_t>);

    m.def("prod_slice_async_fp64", &prod_slice_async<fp64_t>);
    m.def("prod_slice_async_bf16", &prod_slice_async<bf16_t>);
    m.def("prod_slice_async_fp32", &prod_slice_async<fp32_t>);
    m.def("prod_slice_async_fp32_fast_tf32", &prod_slice_async<fp32_fast_tf32_t>);
    m.def("prod_slice_fp64", &prod_slice<fp64_t>);
    m.def("prod_slice_fp32", &prod_slice<fp32_t>);
    m.def("prod_slice_bf16", &prod_slice<bf16_t>);
    m.def("prod_slice_fp32_fast_tf32", &prod_slice<fp32_fast_tf32_t>);

    m.def("prod_fiber_async_fp64", &prod_fiber_async<fp64_t>);
    m.def("prod_fiber_async_fp32", &prod_fiber_async<fp32_t>);
    m.def("prod_fiber_fp64", &prod_fiber<fp64_t>);
    m.def("prod_fiber_fp32", &prod_fiber<fp32_t>);

    m.def("prod_fiber3_async_fp64", &prod_fiber3_async<fp64_t>);
    m.def("prod_fiber3_async_bf16", &prod_fiber3_async<bf16_t>);
    m.def("prod_fiber3_async_fp32", &prod_fiber3_async<fp32_t>);
    m.def("prod_fiber3_async_fp32_fast_tf32", &prod_fiber3_async<fp32_fast_tf32_t>);
    m.def("prod_fiber3_fp64", &prod_fiber3<fp64_t>);
    m.def("prod_fiber3_fp32", &prod_fiber3<fp32_t>);
    m.def("prod_fiber3_bf16", &prod_fiber3<bf16_t>);
    m.def("prod_fiber3_fp32_fast_tf32", &prod_fiber3<fp32_fast_tf32_t>);

    m.def("gather_async_fp64", &gather_async<fp64_t>);
    m.def("gather_async_fp32", &gather_async<fp32_t>);
    m.def("gather_async_int64", &gather_async<nntile::int64_t>);
    m.def("gather_async_bool", &gather_async<bool_t>);
    m.def("gather_async_bf16", &gather_async<bf16_t>);
    m.def("gather_fp64", &gather<fp64_t>);
    m.def("gather_fp32", &gather<fp32_t>);
    m.def("gather_int64", &gather<nntile::int64_t>);
    m.def("gather_bool", &gather<bool_t>);
    m.def("gather_bf16", &gather<bf16_t>);

    m.def("copy_intersection_async_bool", &copy_intersection_async<bool_t>);
    m.def("copy_intersection_async_fp64", &copy_intersection_async<fp64_t>);
    m.def("copy_intersection_async_fp32", &copy_intersection_async<fp32_t>);
    m.def("copy_intersection_async_int64", &copy_intersection_async<nntile::int64_t>);

    m.def("copy_intersection_bool", &copy_intersection<bool_t>);
    m.def("copy_intersection_fp64", &copy_intersection<fp64_t>);
    m.def("copy_intersection_fp32", &copy_intersection<fp32_t>);
    m.def("copy_intersection_int64", &copy_intersection<nntile::int64_t>);

    m.def("copy_async_fp64", &copy_async<fp64_t>);
    m.def("copy_async_bf16", &copy_async<bf16_t>);
    m.def("copy_async_fp32", &copy_async<fp32_t>);
    m.def("copy_async_fp32_fast_tf32", &copy_async<fp32_fast_tf32_t>);
    m.def("copy_async_int64", &copy_async<nntile::int64_t>);

    m.def("copy_fp64", &copy<fp64_t>);
    m.def("copy_bf16", &copy<bf16_t>);
    m.def("copy_fp32", &copy<fp32_t>);
    m.def("copy_fp32_fast_tf32", &copy<fp32_fast_tf32_t>);
    m.def("copy_int64", &copy<nntile::int64_t>);

    m.def("clear_async_fp64", &clear_async<fp64_t>);
    m.def("clear_async_fp32", &clear_async<fp32_t>);
    m.def("clear_async_fp32_fast_tf32", &clear_async<fp32_fast_tf32_t>);
    m.def("clear_async_bf16", &clear_async<bf16_t>);
    //m.def("clear_async_fp16", &clear_async<fp16_t>);
    m.def("clear_fp64", &clear<fp64_t>);
    m.def("clear_fp32", &clear<fp32_t>);
    m.def("clear_bf16", &clear<bf16_t>);
    m.def("clear_fp32_fast_tf32", &clear<fp32_fast_tf32_t>);
    //m.def("clear_fp16", &clear<fp16_t>);

    m.def("axpy_async_fp64", py::overload_cast<Scalar, const Tensor<fp64_t>&,
            const Tensor<fp64_t>&>(&axpy_async<fp64_t>));
    m.def("axpy_async_fp32", py::overload_cast<Scalar, const Tensor<fp32_t>&,
            const Tensor<fp32_t>&>(&axpy_async<fp32_t>));
    m.def("axpy_async_fp32_fast_tf32", py::overload_cast<Scalar, const Tensor<fp32_fast_tf32_t>&,
            const Tensor<fp32_fast_tf32_t>&>(&axpy_async<fp32_fast_tf32_t>));
    m.def("axpy_fp64", py::overload_cast<Scalar, const Tensor<fp64_t>&,
            const Tensor<fp64_t>&>(&axpy<fp64_t>));
    m.def("axpy_fp32", py::overload_cast<Scalar, const Tensor<fp32_t>&,
            const Tensor<fp32_t>&>(&axpy<fp32_t>));
    m.def("axpy_fp32_fast_tf32", py::overload_cast<Scalar, const Tensor<fp32_fast_tf32_t>&,
            const Tensor<fp32_fast_tf32_t>&>(&axpy<fp32_fast_tf32_t>));

    m.def("axpy_async_fp64", py::overload_cast<const Tensor<fp64_t>&,
            const Tensor<fp64_t>&,
            const Tensor<fp64_t>&>(&axpy_async<fp64_t>));
    m.def("axpy_async_fp32", py::overload_cast<const Tensor<fp32_t>&,
            const Tensor<fp32_t>&,
            const Tensor<fp32_t>&>(&axpy_async<fp32_t>));
    m.def("axpy_fp64", py::overload_cast<const Tensor<fp64_t>&,
            const Tensor<fp64_t>&, const Tensor<fp64_t>&>(&axpy<fp64_t>));
    m.def("axpy_fp32", py::overload_cast<const Tensor<fp32_t>&,
            const Tensor<fp32_t>&, const Tensor<fp32_t>&>(&axpy<fp32_t>));

    m.def("sqrt_async_fp64", &sqrt_async<fp64_t>);
    m.def("sqrt_async_fp32", &sqrt_async<fp32_t>);
    m.def("sqrt_fp64", &sqrt<fp64_t>);
    m.def("sqrt_fp32", &sqrt<fp32_t>);
    m.def("sqrt_inplace_async_fp64", &sqrt_inplace_async<fp64_t>);
    m.def("sqrt_inplace_async_fp32", &sqrt_inplace_async<fp32_t>);
    m.def("sqrt_inplace_fp64", &sqrt_inplace<fp64_t>);
    m.def("sqrt_inplace_fp32", &sqrt_inplace<fp32_t>);
    m.def("maximum_async_fp64", &maximum_async<fp64_t>);
    m.def("maximum_async_fp32", &maximum_async<fp32_t>);
    m.def("maximum_fp64", &maximum<fp64_t>);
    m.def("maximum_fp32", &maximum<fp32_t>);

    m.def("addcdiv_async_fp64", &addcdiv_async<fp64_t>);
    m.def("addcdiv_async_bf16", &addcdiv_async<bf16_t>);
    m.def("addcdiv_async_fp32", &addcdiv_async<fp32_t>);
    m.def("addcdiv_async_fp32_fast_tf32", &addcdiv_async<fp32_fast_tf32_t>);
    m.def("addcdiv_fp64", &addcdiv<fp64_t>);
    m.def("addcdiv_fp32", &addcdiv<fp32_t>);
    m.def("addcdiv_bf16", &addcdiv<bf16_t>);
    m.def("addcdiv_fp32_fast_tf32", &addcdiv<fp32_fast_tf32_t>);

    m.def("logsumexp_async_fp64", &logsumexp_async<fp64_t>);
    m.def("logsumexp_async_bf16", &logsumexp_async<bf16_t>);
    m.def("logsumexp_async_fp32", &logsumexp_async<fp32_t>);
    m.def("logsumexp_async_fp32_fast_tf32", &logsumexp_async<fp32_fast_tf32_t>);
    m.def("logsumexp_fp64", &logsumexp<fp64_t>);
    m.def("logsumexp_bf16", &logsumexp<bf16_t>);
    m.def("logsumexp_fp32", &logsumexp<fp32_t>);
    m.def("logsumexp_fp32_fast_tf32", &logsumexp<fp32_fast_tf32_t>);

    m.def("total_sum_accum_async_fp64", &total_sum_accum_async<fp64_t>);
    m.def("total_sum_accum_async_bf16", &total_sum_accum_async<bf16_t>);
    m.def("total_sum_accum_async_fp32", &total_sum_accum_async<fp32_t>);
    m.def("total_sum_accum_async_fp32_fast_tf32", &total_sum_accum_async<fp32_fast_tf32_t>);
    m.def("total_sum_accum_fp64", &total_sum_accum<fp64_t>);
    m.def("total_sum_accum_fp32", &total_sum_accum<fp32_t>);
    m.def("total_sum_accum_bf16", &total_sum_accum<bf16_t>);
    m.def("total_sum_accum_fp32_fast_tf32", &total_sum_accum<fp32_fast_tf32_t>);

    m.def("subtract_indexed_outputs_async_fp64",
            &subtract_indexed_outputs_async<fp64_t>);
    m.def("subtract_indexed_outputs_async_bf16",
            &subtract_indexed_outputs_async<bf16_t>);
    m.def("subtract_indexed_outputs_async_fp32",
            &subtract_indexed_outputs_async<fp32_t>);
    m.def("subtract_indexed_outputs_async_fp32_fast_tf32",
            &subtract_indexed_outputs_async<fp32_fast_tf32_t>);
    m.def("subtract_indexed_outputs_fp64", &subtract_indexed_outputs<fp64_t>);
    m.def("subtract_indexed_outputs_bf16", &subtract_indexed_outputs<bf16_t>);
    m.def("subtract_indexed_outputs_fp32", &subtract_indexed_outputs<fp32_t>);
    m.def("subtract_indexed_outputs_fp32_fast_tf32", &subtract_indexed_outputs<fp32_fast_tf32_t>);

    m.def("scal_async_fp64", &scal_async<fp64_t>);
    m.def("scal_async_bf16", &scal_async<bf16_t>);
    m.def("scal_async_fp32", &scal_async<fp32_t>);
    m.def("scal_async_fp32_fast_tf32", &scal_async<fp32_fast_tf32_t>);
    m.def("scal_fp64", &scal<fp64_t>);
    m.def("scal_fp32", &scal<fp32_t>);
    m.def("scal_bf16", &scal<bf16_t>);
    m.def("scal_fp32_fast_tf32", &scal<fp32_fast_tf32_t>);

    m.def("adam_step_async_fp64", &adam_step_async<fp64_t>);
    m.def("adam_step_async_bf16", &adam_step_async<bf16_t>);
    m.def("adam_step_async_fp32", &adam_step_async<fp32_t>);
    m.def("adam_step_async_fp32_fast_tf32", &adam_step_async<fp32_fast_tf32_t>);
    m.def("adam_step_fp64", &adam_step<fp64_t>);
    m.def("adam_step_bf16", &adam_step<bf16_t>);
    m.def("adam_step_fp32", &adam_step<fp32_t>);
    m.def("adam_step_fp32_fast_tf32", &adam_step<fp32_fast_tf32_t>);

    m.def("adamw_step_async_fp64", &adamw_step_async<fp64_t>);
    m.def("adamw_step_async_bf16", &adamw_step_async<bf16_t>);
    m.def("adamw_step_async_fp32", &adamw_step_async<fp32_t>);
    m.def("adamw_step_async_fp32_fast_tf32", &adamw_step_async<fp32_fast_tf32_t>);
    m.def("adamw_step_fp64", &adamw_step<fp64_t>);
    m.def("adamw_step_bf16", &adamw_step<bf16_t>);
    m.def("adamw_step_fp32", &adamw_step<fp32_t>);
    m.def("adamw_step_fp32_fast_tf32", &adamw_step<fp32_fast_tf32_t>);

    m.def("scal_inplace_async_fp64", &scal_inplace_async<fp64_t>);
    m.def("scal_inplace_async_fp32", &scal_inplace_async<fp32_t>);
    m.def("scal_inplace_async_fp32_fast_tf32", &scal_inplace_async<fp32_fast_tf32_t>);
    m.def("scal_inplace_fp64", &scal_inplace<fp64_t>);
    m.def("scal_inplace_fp32", &scal_inplace<fp32_t>);
    m.def("scal_inplace_fp32_fast_tf32", &scal_inplace<fp32_fast_tf32_t>);

    m.def("sumprod_slice_async_fp64", &sumprod_slice_async<fp64_t>);
    m.def("sumprod_slice_async_bf16", &sumprod_slice_async<bf16_t>);
    m.def("sumprod_slice_async_fp32", &sumprod_slice_async<fp32_t>);
    m.def("sumprod_slice_async_fp32_fast_tf32", &sumprod_slice_async<fp32_fast_tf32_t>);
    m.def("sumprod_slice_fp64", &sumprod_slice<fp64_t>);
    m.def("sumprod_slice_fp32", &sumprod_slice<fp32_t>);
    m.def("sumprod_slice_bf16", &sumprod_slice<bf16_t>);
    m.def("sumprod_slice_fp32_fast_tf32", &sumprod_slice<fp32_fast_tf32_t>);

    m.def("sumprod_fiber_async_fp64", &sumprod_fiber_async<fp64_t>);
    m.def("sumprod_fiber_async_bf16", &sumprod_fiber_async<bf16_t>);
    m.def("sumprod_fiber_async_fp32", &sumprod_fiber_async<fp32_t>);
    m.def("sumprod_fiber_async_fp32_fast_tf32", &sumprod_fiber_async<fp32_fast_tf32_t>);
    m.def("sumprod_fiber_fp64", &sumprod_fiber<fp64_t>);
    m.def("sumprod_fiber_fp32", &sumprod_fiber<fp32_t>);
    m.def("sumprod_fiber_bf16", &sumprod_fiber<bf16_t>);
    m.def("sumprod_fiber_fp32_fast_tf32", &sumprod_fiber<fp32_fast_tf32_t>);

    // gelu and dgelu
    m.def("gelu_async_fp64", &gelu_async<fp64_t>);
    m.def("gelu_async_fp32", &gelu_async<fp32_t>);
    m.def("gelu_fp64", &gelu<fp64_t>);
    m.def("gelu_fp32", &gelu<fp32_t>);
    m.def("gelu_backward_async_fp64", &gelu_backward_async<fp64_t>);
    m.def("gelu_backward_async_fp32", &gelu_backward_async<fp32_t>);
    m.def("gelu_backward_fp64", &gelu_backward<fp64_t>);
    m.def("gelu_backward_fp32", &gelu_backward<fp32_t>);

    m.def("gelutanh_async_fp64", &gelutanh_async<fp64_t>);
    m.def("gelutanh_async_bf16", &gelutanh_async<bf16_t>);
    m.def("gelutanh_async_fp32", &gelutanh_async<fp32_t>);
    m.def("gelutanh_async_fp32_fast_tf32", &gelutanh_async<fp32_fast_tf32_t>);
    m.def("gelutanh_fp64", &gelutanh<fp64_t>);
    m.def("gelutanh_fp32", &gelutanh<fp32_t>);
    m.def("gelutanh_bf16", &gelutanh<bf16_t>);
    m.def("gelutanh_fp32_fast_tf32", &gelutanh<fp32_fast_tf32_t>);

    m.def("gelutanh_inplace_async_fp64", &gelutanh_inplace_async<fp64_t>);
    m.def("gelutanh_inplace_async_fp32", &gelutanh_inplace_async<fp32_t>);
    m.def("gelutanh_inplace_fp64", &gelutanh_inplace<fp64_t>);
    m.def("gelutanh_inplace_fp32", &gelutanh_inplace<fp32_t>);

    m.def("gelutanh_backward_async_fp64", &gelutanh_backward_async<fp64_t>);
    m.def("gelutanh_backward_async_bf16", &gelutanh_backward_async<bf16_t>);
    m.def("gelutanh_backward_async_fp32", &gelutanh_backward_async<fp32_t>);
    m.def("gelutanh_backward_async_fp32_fast_tf32", &gelutanh_backward_async<fp32_fast_tf32_t>);
    m.def("gelutanh_backward_fp64", &gelutanh_backward<fp64_t>);
    m.def("gelutanh_backward_fp32", &gelutanh_backward<fp32_t>);
    m.def("gelutanh_backward_bf16", &gelutanh_backward<bf16_t>);
    m.def("gelutanh_backward_fp32_fast_tf32", &gelutanh_backward<fp32_fast_tf32_t>);

    m.def("dgelu_async_fp64", &dgelu_async<fp64_t>);
    m.def("dgelu_async_fp32", &dgelu_async<fp32_t>);
    m.def("dgelu_fp64", &dgelu<fp64_t>);
    m.def("dgelu_fp32", &dgelu<fp32_t>);
    m.def("dgelutanh_async_fp64", &dgelutanh_async<fp64_t>);
    m.def("dgelutanh_async_fp32", &dgelutanh_async<fp32_t>);
    m.def("dgelutanh_fp64", &dgelutanh<fp64_t>);
    m.def("dgelutanh_fp32", &dgelutanh<fp32_t>);

    // Embedding forward pass
    m.def("embedding_async_fp64", &embedding_async<fp64_t>);
    m.def("embedding_async_fp32", &embedding_async<fp32_t>);
    m.def("embedding_async_bf16", &embedding_async<bf16_t>);
    m.def("embedding_async_fp32_fast_tf32", &embedding_async<fp32_fast_tf32_t>);
    m.def("embedding_fp64", &embedding<fp64_t>);
    m.def("embedding_fp32", &embedding<fp32_t>);
    m.def("embedding_bf16", &embedding<bf16_t>);
    m.def("embedding_fp32_fast_tf32", &embedding<fp32_fast_tf32_t>);

    // Embedding backward pass
    m.def("embedding_backward_async_fp64", &embedding_backward_async<fp64_t>);
    m.def("embedding_backward_async_bf16", &embedding_backward_async<bf16_t>);
    m.def("embedding_backward_async_fp32", &embedding_backward_async<fp32_t>);
    m.def("embedding_backward_async_fp32_fast_tf32", &embedding_backward_async<fp32_fast_tf32_t>);
    m.def("embedding_backward_fp64", &embedding_backward<fp64_t>);
    m.def("embedding_backward_fp32", &embedding_backward<fp32_t>);
    m.def("embedding_backward_bf16", &embedding_backward<bf16_t>);
    m.def("embedding_backward_fp32_fast_tf32", &embedding_backward<fp32_fast_tf32_t>);

    // FP32 <-> FP16
    //m.def("fp32_to_fp16_async", &fp32_to_fp16_async);
    //m.def("fp16_to_fp32_async", &fp16_to_fp32_async);

    m.def("mask_scalar_async_fp64", &mask_scalar_async<fp64_t>);
    m.def("mask_scalar_async_bf16", &mask_scalar_async<bf16_t>);
    m.def("mask_scalar_async_fp32", &mask_scalar_async<fp32_t>);
    m.def("mask_scalar_async_fp32_fast_tf32", &mask_scalar_async<fp32_fast_tf32_t>);
    m.def("mask_scalar_fp64", &mask_scalar<fp64_t>);
    m.def("mask_scalar_fp32", &mask_scalar<fp32_t>);
    m.def("mask_scalar_bf16", &mask_scalar<bf16_t>);
    m.def("mask_scalar_fp32_fast_tf32", &mask_scalar<fp32_fast_tf32_t>);

    m.def("hypot_async_fp64", &hypot_async<fp64_t>);
    m.def("hypot_async_bf16", &hypot_async<bf16_t>);
    m.def("hypot_async_fp32", &hypot_async<fp32_t>);
    m.def("hypot_async_fp32_fast_tf32", &hypot_async<fp32_fast_tf32_t>);
    m.def("hypot_fp64", &hypot<fp64_t>);
    m.def("hypot_bf16", &hypot<bf16_t>);
    m.def("hypot_fp32", &hypot<fp32_t>);
    m.def("hypot_fp32", &hypot<fp32_fast_tf32_t>);

    m.def("hypot_scalar_inverse_async_fp64", &hypot_scalar_inverse_async<fp64_t>);
    m.def("hypot_scalar_inverse_async_bf16", &hypot_scalar_inverse_async<bf16_t>);
    m.def("hypot_scalar_inverse_async_fp32", &hypot_scalar_inverse_async<fp32_t>);
    m.def("hypot_scalar_inverse_async_fp32_fast_tf32", &hypot_scalar_inverse_async<fp32_fast_tf32_t>);
    m.def("hypot_scalar_inverse_fp64", &hypot_scalar_inverse<fp64_t>);
    m.def("hypot_scalar_inverse_bf16", &hypot_scalar_inverse<bf16_t>);
    m.def("hypot_scalar_inverse_fp32", &hypot_scalar_inverse<fp32_t>);
    m.def("hypot_scalar_inverse_fp32_fast_tf32", &hypot_scalar_inverse<fp32_fast_tf32_t>);

    m.def("transpose_async_fp64", &transpose_async<fp64_t>);
    m.def("transpose_async_bf16", &transpose_async<bf16_t>);
    m.def("transpose_async_fp32", &transpose_async<fp32_t>);
    m.def("transpose_async_fp32_fast_tf32", &transpose_async<fp32_fast_tf32_t>);
    m.def("transpose_fp64", &transpose<fp64_t>);
    m.def("transpose_fp32", &transpose<fp32_t>);
    m.def("transpose_bf16", &transpose<bf16_t>);
    m.def("transpose_fp32_fast_tf32", &transpose<fp32_fast_tf32_t>);

    m.def("conv2d_inplace_async_fp64", &conv2d_inplace_async<fp64_t>);
    m.def("conv2d_inplace_async_fp32", &conv2d_inplace_async<fp32_t>);
    m.def("conv2d_inplace_async_fp32_fast_tf32",
            &conv2d_inplace_async<fp32_fast_tf32_t>);
    m.def("conv2d_inplace_async_bf16", &conv2d_inplace_async<bf16_t>);
    m.def("conv2d_inplace_fp64", &conv2d_inplace<fp64_t>);
    m.def("conv2d_inplace_fp32", &conv2d_inplace<fp32_t>);
    m.def("conv2d_inplace_fp32_fast_tf32",
            &conv2d_inplace<fp32_fast_tf32_t>);

    m.def("conv2d_bwd_input_inplace_bf16",
            &conv2d_bwd_input_inplace<bf16_t>);
    m.def("conv2d_bwd_input_inplace_async_fp64",
            &conv2d_bwd_input_inplace_async<fp64_t>);
    m.def("conv2d_bwd_input_inplace_async_fp32",
            &conv2d_bwd_input_inplace_async<fp32_t>);
    m.def("conv2d_bwd_input_inplace_async_fp32_fast_tf32",
            &conv2d_bwd_input_inplace_async<fp32_fast_tf32_t>);
    m.def("conv2d_bwd_input_inplace_async_bf16",
            &conv2d_bwd_input_inplace_async<bf16_t>);
    m.def("conv2d_bwd_input_inplace_fp64",
            &conv2d_bwd_input_inplace<fp64_t>);
    m.def("conv2d_bwd_input_inplace_fp32",
            &conv2d_bwd_input_inplace<fp32_t>);
    m.def("conv2d_bwd_input_inplace_fp32_fast_tf32",
            &conv2d_bwd_input_inplace<fp32_fast_tf32_t>);
    m.def("conv2d_bwd_input_inplace_bf16",
            &conv2d_bwd_input_inplace<bf16_t>);

    m.def("conv2d_bwd_weight_inplace_bf16",
            &conv2d_bwd_weight_inplace<bf16_t>);
    m.def("conv2d_bwd_weight_inplace_async_fp64",
            &conv2d_bwd_weight_inplace_async<fp64_t>);
    m.def("conv2d_bwd_weight_inplace_async_fp32",
            &conv2d_bwd_weight_inplace_async<fp32_t>);
    m.def("conv2d_bwd_weight_inplace_async_fp32_fast_tf32",
            &conv2d_bwd_weight_inplace_async<fp32_fast_tf32_t>);
    m.def("conv2d_bwd_weight_inplace_async_bf16",
            &conv2d_bwd_weight_inplace_async<bf16_t>);
    m.def("conv2d_bwd_weight_inplace_fp64",
            &conv2d_bwd_weight_inplace<fp64_t>);
    m.def("conv2d_bwd_weight_inplace_fp32",
            &conv2d_bwd_weight_inplace<fp32_t>);
    m.def("conv2d_bwd_weight_inplace_fp32_fast_tf32",
            &conv2d_bwd_weight_inplace<fp32_fast_tf32_t>);
    m.def("conv2d_bwd_weight_inplace_bf16",
            &conv2d_bwd_weight_inplace<bf16_t>);

    m.def("rope_async_fp64", &rope_async<fp64_t>);
    m.def("rope_async_fp32", &rope_async<fp32_t>);
    m.def("rope_async_fp32_fast_tf32", &rope_async<fp32_fast_tf32_t>);
    m.def("rope_async_bf16", &rope_async<bf16_t>);
    m.def("rope_fp64", &rope<fp64_t>);
    m.def("rope_fp32", &rope<fp32_t>);
    m.def("rope_fp32_fast_tf32", &rope<fp32_fast_tf32_t>);
    m.def("rope_bf16", &rope<bf16_t>);

    m.def("rope_backward_async_fp64", &rope_backward_async<fp64_t>);
    m.def("rope_backward_async_fp32", &rope_backward_async<fp32_t>);
    m.def("rope_backward_async_fp32_fast_tf32",
            &rope_backward_async<fp32_fast_tf32_t>);
    m.def("rope_backward_async_bf16", &rope_backward_async<bf16_t>);
    m.def("rope_backward_fp64", &rope_backward<fp64_t>);
    m.def("rope_backward_fp32", &rope_backward<fp32_t>);
    m.def("rope_backward_fp32_fast_tf32", &rope_backward<fp32_fast_tf32_t>);
    m.def("rope_backward_bf16", &rope_backward<bf16_t>);
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
