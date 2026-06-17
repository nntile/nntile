/*! @copyright (c) 2026-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *
 * @file torch_nntile/csrc/nntile_generator.cpp
 */

#include "nntile_generator.h"

#include <ATen/Functions.h>
#include <ATen/core/MT19937RNGEngine.h>
#include <c10/core/GeneratorImpl.h>
#include <c10/core/TensorImpl.h>

#include <cstring>
#include <mutex>
#include <unordered_map>

namespace torch_nntile
{

namespace
{

struct NntileGeneratorImpl final : public c10::GeneratorImpl
{
    explicit NntileGeneratorImpl(c10::DeviceIndex device_index) :
        c10::GeneratorImpl(
            {c10::DeviceType::PrivateUse1, device_index},
            c10::DispatchKeySet(c10::DispatchKey::PrivateUse1)),
        engine_(c10::default_rng_seed_val)
    {
        seed_ = c10::default_rng_seed_val;
    }

    void set_current_seed(uint64_t seed) override
    {
        seed_ = seed;
        engine_ = at::mt19937(seed);
    }

    void set_offset(uint64_t offset) override
    {
        offset_ = offset;
    }

    uint64_t get_offset() const override
    {
        return offset_;
    }

    uint64_t current_seed() const override
    {
        return seed_;
    }

    uint64_t seed() override
    {
        const uint64_t random_seed = static_cast<uint64_t>(engine_());
        set_current_seed(random_seed);
        return random_seed;
    }

    void set_state(const c10::TensorImpl &new_state) override
    {
        TORCH_CHECK(
            new_state.dtype() == at::ScalarType::Byte,
            "nntile generator state must be byte tensor");
        TORCH_CHECK(
            new_state.numel() ==
                static_cast<int64_t>(sizeof(at::mt19937_data_pod)),
            "nntile generator state has wrong size");
        at::mt19937_data_pod pod{};
        std::memcpy(
            &pod,
            new_state.storage().data(),
            sizeof(at::mt19937_data_pod));
        engine_.set_data(pod);
        seed_ = pod.seed_;
    }

    c10::intrusive_ptr<c10::TensorImpl> get_state() const override
    {
        at::Tensor state = at::empty(
            {static_cast<int64_t>(sizeof(at::mt19937_data_pod))},
            at::TensorOptions().dtype(at::kByte).device(at::kCPU));
        const at::mt19937_data_pod pod = engine_.data();
        std::memcpy(state.data_ptr(), &pod, sizeof(at::mt19937_data_pod));
        return state.getIntrusivePtr();
    }

    NntileGeneratorImpl *clone_impl() const override
    {
        auto *clone = new NntileGeneratorImpl(device().index());
        clone->seed_ = seed_;
        clone->offset_ = offset_;
        clone->engine_ = engine_;
        return clone;
    }

    uint64_t seed_ = c10::default_rng_seed_val;
    uint64_t offset_ = 0;
    at::mt19937 engine_;
};

} // namespace

at::Generator make_nntile_generator(c10::DeviceIndex device_index)
{
    return at::make_generator<NntileGeneratorImpl>(device_index);
}

const at::Generator &get_default_nntile_generator(c10::DeviceIndex device_index)
{
    static std::mutex mutex;
    static std::unordered_map<c10::DeviceIndex, at::Generator> generators;
    std::lock_guard<std::mutex> lock(mutex);
    auto it = generators.find(device_index);
    if (it == generators.end())
    {
        it = generators.emplace(device_index, make_nntile_generator(device_index))
                 .first;
    }
    return it->second;
}

} // namespace torch_nntile
