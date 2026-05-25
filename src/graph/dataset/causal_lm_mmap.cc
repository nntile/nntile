/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/dataset/causal_lm_mmap.cc
 * Mmap token stream and causal LM batch iterator.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/dataset/causal_lm_mmap.hh"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <random>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace nntile::graph::dataset
{

namespace
{

void close_fd(int& fd)
{
    if(fd >= 0)
    {
        ::close(fd);
        fd = -1;
    }
}

} // namespace

TokenMemoryMap::TokenMemoryMap(std::string path)
    : path_(std::move(path))
{
    fd_ = ::open(path_.c_str(), O_RDONLY);
    if(fd_ < 0)
    {
        throw std::runtime_error(
            "TokenMemoryMap: cannot open " + path_ + ": " +
            std::strerror(errno));
    }
    struct stat st
    {
    };
    if(::fstat(fd_, &st) != 0)
    {
        close_fd(fd_);
        throw std::runtime_error(
            "TokenMemoryMap: fstat failed for " + path_ + ": " +
            std::strerror(errno));
    }
    if(st.st_size < 0)
    {
        close_fd(fd_);
        throw std::runtime_error("TokenMemoryMap: negative file size");
    }
    size_ = static_cast<std::size_t>(st.st_size);
    if((size_ % sizeof(std::uint16_t)) != 0)
    {
        close_fd(fd_);
        throw std::runtime_error(
            "TokenMemoryMap: file size not multiple of 2: " + path_);
    }
    if(size_ == 0)
    {
        close_fd(fd_);
        throw std::runtime_error("TokenMemoryMap: empty file: " + path_);
    }
    void* p = ::mmap(nullptr, size_, PROT_READ, MAP_SHARED, fd_, 0);
    if(p == MAP_FAILED)
    {
        close_fd(fd_);
        throw std::runtime_error(
            "TokenMemoryMap: mmap failed for " + path_ + ": " +
            std::strerror(errno));
    }
    addr_ = p;
}

TokenMemoryMap::~TokenMemoryMap()
{
    if(addr_ != nullptr && addr_ != MAP_FAILED)
    {
        ::munmap(addr_, size_);
        addr_ = nullptr;
        size_ = 0;
    }
    close_fd(fd_);
}

TokenMemoryMap::TokenMemoryMap(TokenMemoryMap&& other) noexcept
    : path_(std::move(other.path_))
    , addr_(other.addr_)
    , size_(other.size_)
    , fd_(other.fd_)
{
    other.addr_ = nullptr;
    other.size_ = 0;
    other.fd_ = -1;
}

TokenMemoryMap& TokenMemoryMap::operator=(TokenMemoryMap&& other) noexcept
{
    if(this != &other)
    {
        if(addr_ != nullptr && addr_ != MAP_FAILED)
        {
            ::munmap(addr_, size_);
        }
        close_fd(fd_);
        path_ = std::move(other.path_);
        addr_ = other.addr_;
        size_ = other.size_;
        fd_ = other.fd_;
        other.addr_ = nullptr;
        other.size_ = 0;
        other.fd_ = -1;
    }
    return *this;
}

std::size_t TokenMemoryMap::num_tokens() const noexcept
{
    return size_ / sizeof(std::uint16_t);
}

std::uint16_t const* TokenMemoryMap::data() const noexcept
{
    return static_cast<std::uint16_t const*>(addr_);
}

std::uint16_t TokenMemoryMap::token_u16(std::size_t i) const
{
    if(i >= num_tokens())
    {
        throw std::out_of_range("TokenMemoryMap::token_u16");
    }
    return data()[i];
}

std::string const& TokenMemoryMap::path() const noexcept
{
    return path_;
}

CausalLmBatchIterator::CausalLmBatchIterator(
    TokenMemoryMap const& tokens,
    CausalLmBatchConfig const& cfg,
    Index vocab_size)
    : tokens_(&tokens)
    , cfg_(cfg)
    , vocab_size_(vocab_size)
{
    if(cfg_.n_seq <= 0 || cfg_.n_batch <= 0)
    {
        throw std::invalid_argument(
            "CausalLmBatchIterator: n_seq and n_batch must be positive");
    }
    const std::size_t window =
        static_cast<std::size_t>(cfg_.n_seq) + 1U;
    const std::size_t ntok = tokens_->num_tokens();
    num_seq_ = ntok / window;
    num_batches_ = num_seq_ / static_cast<std::size_t>(cfg_.n_batch);
    seq_order_.resize(num_seq_);
    for(std::size_t i = 0; i < num_seq_; ++i)
    {
        seq_order_[i] = i;
    }
    if(cfg_.shuffle && num_seq_ > 1)
    {
        std::mt19937 gen(cfg_.seed);
        std::shuffle(seq_order_.begin(), seq_order_.end(), gen);
    }
}

bool CausalLmBatchIterator::next(CausalLmBatch& batch)
{
    if(batch_idx_ >= num_batches_)
    {
        return false;
    }
    const Index n_seq = cfg_.n_seq;
    const Index n_batch = cfg_.n_batch;
    const std::size_t nelem =
        static_cast<std::size_t>(n_seq) * static_cast<std::size_t>(n_batch);
    batch.input_ids.resize(nelem);
    batch.target_ids.resize(nelem);
    const std::size_t window =
        static_cast<std::size_t>(n_seq) + 1U;
    std::uint16_t const* const tok = tokens_->data();
    for(Index b = 0; b < n_batch; ++b)
    {
        const std::size_t seq_slot =
            seq_order_[batch_idx_ * static_cast<std::size_t>(n_batch) + b];
        const std::size_t off = seq_slot * window;
        for(Index s = 0; s < n_seq; ++s)
        {
            const std::uint16_t ti = tok[off + static_cast<std::size_t>(s)];
            const std::uint16_t tt =
                tok[off + static_cast<std::size_t>(s) + 1U];
            if(vocab_size_ > 0)
            {
                if(static_cast<Index>(ti) >= vocab_size_ ||
                    static_cast<Index>(tt) >= vocab_size_)
                {
                    throw std::runtime_error(
                        "CausalLmBatchIterator: token id >= vocab_size");
                }
            }
            const std::size_t idx =
                static_cast<std::size_t>(s) +
                static_cast<std::size_t>(n_seq) * static_cast<std::size_t>(b);
            batch.input_ids[idx] = static_cast<std::int64_t>(ti);
            batch.target_ids[idx] = static_cast<std::int64_t>(tt);
        }
    }
    ++batch_idx_;
    return true;
}

std::size_t CausalLmBatchIterator::num_batches() const noexcept
{
    return num_batches_;
}

std::size_t CausalLmBatchIterator::batch_index() const noexcept
{
    return batch_idx_;
}

} // namespace nntile::graph::dataset
