#include <nntile/common.hh>
/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * @file nntile/src/core/execution_schedule.cc
 *
 * @version 1.1.0
 * */

#include "nntile/core/execution_schedule.hh"
#include "nntile/core/execution_worker.hh"

#include "nntile/starpu_c.hh"
#include "nntile/tile/graph_data_node.hh"
#include "nntile/tile/graph_decl.hh"
#include "nntile/tile/graph_op_node.hh"

#include <algorithm>
#include <fstream>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace nntile
{

namespace
{

std::string tile_label(TileGraph::TileNode const *t)
{
    if (t == nullptr)
    {
        return "";
    }
    if (!t->name().empty())
    {
        return t->name();
    }
    return "tile@" + std::to_string(static_cast<unsigned long long>(t->id()));
}

void collect_writable_tiles(
    TileGraph::OpNode const &op,
    std::vector<TileGraph::TileNode const *> &out)
{
    std::unordered_set<TileGraph::TileNode const *> seen;
    auto add = [&](TileGraph::TileNode const *t) {
        if (t != nullptr && seen.insert(t).second)
        {
            out.push_back(t);
        }
    };
    for (TileGraph::TileNode *o : op.outputs())
    {
        add(o);
    }
}

std::map<int, size_t> writable_bytes_by_worker(
    std::vector<TileGraph::TileNode const *> const &writable,
    std::map<TileGraph::TileNode const *, int> const &tile_worker)
{
    std::map<int, size_t> bytes;
    for (TileGraph::TileNode const *t : writable)
    {
        auto it = tile_worker.find(t);
        if (it == tile_worker.end())
        {
            continue;
        }
        bytes[it->second] += static_cast<size_t>(t->size_bytes());
    }
    return bytes;
}

int pick_worker_max_writable_dependency(
    std::vector<TileGraph::TileNode const *> const &writable,
    std::map<TileGraph::TileNode const *, int> const &tile_worker)
{
    std::map<int, size_t> const by_worker =
        writable_bytes_by_worker(writable, tile_worker);
    if (by_worker.empty())
    {
        return 0;
    }
    int best_worker = by_worker.begin()->first;
    size_t best_bytes = by_worker.begin()->second;
    for (auto const &[w, nbytes] : by_worker)
    {
        if (nbytes > best_bytes || (nbytes == best_bytes && w < best_worker))
        {
            best_bytes = nbytes;
            best_worker = w;
        }
    }
    return best_worker;
}

void assign_tensor_tiles_round_robin(
    TileGraph::TensorDescriptor const &td,
    int num_workers,
    std::map<TileGraph::TileNode const *, int> &tile_worker,
    std::map<std::string, int> &tile_virtual_worker)
{
    Index const vol = static_cast<Index>(td.tiles.size());
    for (Index lin = 0; lin < vol; ++lin)
    {
        TileGraph::TileNode *t = td.tiles[static_cast<size_t>(lin)];
        if (t == nullptr)
        {
            continue;
        }
        int const w =
            num_workers > 0 ? static_cast<int>(lin % num_workers) : 0;
        tile_worker[t] = w;
        tile_virtual_worker[tile_label(t)] = w;
    }
}

void assign_tensor_tiles_affinity_batch(
    TileGraph::TensorDescriptor const &td,
    int num_workers,
    std::map<TileGraph::TileNode const *, int> &tile_worker,
    std::map<std::string, int> &tile_virtual_worker)
{
    Index const vol = static_cast<Index>(td.tiles.size());
    Index batch_stride = 1;
    Index batch_extent = 1;
    if (!td.grid_shape.empty())
    {
        batch_extent = td.grid_shape.back();
        for (size_t d = 0; d + 1 < td.grid_shape.size(); ++d)
        {
            batch_stride *= td.grid_shape[d];
        }
    }
    for (Index lin = 0; lin < vol; ++lin)
    {
        TileGraph::TileNode *t = td.tiles[static_cast<size_t>(lin)];
        if (t == nullptr)
        {
            continue;
        }
        Index const batch_idx =
            batch_stride > 0 ? (lin / batch_stride) % batch_extent : lin;
        int const w = num_workers > 0
            ? static_cast<int>(batch_idx % num_workers)
            : 0;
        tile_worker[t] = w;
        tile_virtual_worker[tile_label(t)] = w;
    }
}

ExecutionSchedule build_execution_schedule(
    TileGraph const &graph,
    std::vector<std::shared_ptr<TileGraph::OpNode>> const &execution_order,
    std::string policy,
    void (*assign_tiles)(TileGraph::TensorDescriptor const &,
        int,
        std::map<TileGraph::TileNode const *, int> &,
        std::map<std::string, int> &))
{
    ExecutionSchedule schedule;
    schedule.policy = std::move(policy);
    schedule.num_workers = sched::count_execution_workers();
    schedule.use_cuda_workers =
        starpu_is_initialized() &&
        starpu_worker_get_count_by_type(STARPU_CUDA_WORKER) > 0;
    if (schedule.num_workers <= 0)
    {
        schedule.num_workers = 1;
    }

    std::map<TileGraph::TileNode const *, int> tile_worker;
    Index orphan_counter = 0;

    for (auto const &td_uptr : graph.tensor_descriptors())
    {
        assign_tiles(
            *td_uptr, schedule.num_workers, tile_worker, schedule.tile_virtual_worker);
    }

    std::vector<TileGraph::TileNode const *> orphans;
    for (auto const &node_uptr : graph.tile_nodes())
    {
        TileGraph::TileNode const *t = node_uptr.get();
        if (tile_worker.count(t) != 0)
        {
            continue;
        }
        orphans.push_back(t);
    }
    std::sort(orphans.begin(),
        orphans.end(),
        [](TileGraph::TileNode const *a, TileGraph::TileNode const *b) {
            return a->id() < b->id();
        });
    for (TileGraph::TileNode const *t : orphans)
    {
        int const w = schedule.num_workers > 0
            ? static_cast<int>(orphan_counter++ % schedule.num_workers)
            : 0;
        tile_worker[t] = w;
        schedule.tile_virtual_worker[tile_label(t)] = w;
    }

    schedule.ops.reserve(execution_order.size());
    for (size_t i = 0; i < execution_order.size(); ++i)
    {
        TileGraph::OpNode const &op = *execution_order[i];
        ScheduledOpEntry entry;
        entry.execution_index = i;
        entry.op_name = op.op_name();
        entry.task_label = op.name().empty()
            ? entry.op_name + "@" +
                  std::to_string(static_cast<unsigned long long>(op.id()))
            : op.name();

        std::vector<TileGraph::TileNode const *> writable;
        collect_writable_tiles(op, writable);

        if (writable.size() == 1)
        {
            auto it = tile_worker.find(writable[0]);
            entry.worker =
                it != tile_worker.end() ? it->second : 0;
        }
        else if (!writable.empty())
        {
            entry.worker = pick_worker_max_writable_dependency(
                writable, tile_worker);
        }
        else
        {
            entry.worker = 0;
        }

        for (TileGraph::TileNode const *t : writable)
        {
            entry.writable_tiles.push_back(tile_label(t));
        }
        for (TileGraph::TileNode *t : op.inputs())
        {
            if (t == nullptr)
            {
                continue;
            }
            bool is_w = false;
            for (TileGraph::TileNode const *w : writable)
            {
                if (w == t)
                {
                    is_w = true;
                    break;
                }
            }
            if (!is_w)
            {
                entry.read_tiles.push_back(tile_label(t));
            }
        }

        schedule.ops.push_back(std::move(entry));
    }

    schedule.fingerprint = make_execution_schedule_fingerprint(execution_order);
    return schedule;
}

} // namespace

ExecutionScheduleFingerprint make_execution_schedule_fingerprint(
    std::vector<std::shared_ptr<TileGraph::OpNode>> const &execution_order)
{
    ExecutionScheduleFingerprint fp;
    fp.op_count = execution_order.size();
    fp.op_names.reserve(execution_order.size());
    for (auto const &op : execution_order)
    {
        fp.op_names.push_back(op->op_name());
    }
    return fp;
}

void validate_execution_schedule_fingerprint(
    ExecutionScheduleFingerprint const &fp,
    std::vector<std::shared_ptr<TileGraph::OpNode>> const &execution_order)
{
    if (fp.op_count != execution_order.size())
    {
        throw std::runtime_error(
            "execution schedule fingerprint: op_count (" +
            std::to_string(fp.op_count) + ") != compiled graph (" +
            std::to_string(execution_order.size()) +
            "); regenerate execution.json");
    }
    if (fp.op_names.size() != execution_order.size())
    {
        throw std::runtime_error(
            "execution schedule fingerprint: op_names length mismatch; "
            "regenerate execution.json");
    }
    for (size_t i = 0; i < execution_order.size(); ++i)
    {
        if (fp.op_names[i] != execution_order[i]->op_name())
        {
            throw std::runtime_error(
                "execution schedule fingerprint: op_names[" +
                std::to_string(i) + "] mismatch (json '" + fp.op_names[i] +
                "' vs graph '" + execution_order[i]->op_name() +
                "'); regenerate execution.json");
        }
    }
}

int ExecutionSchedule::worker_for_op(size_t execution_index) const
{
    if (execution_index >= ops.size())
    {
        throw std::out_of_range(
            "ExecutionSchedule::worker_for_op: bad execution_index");
    }
    return ops[execution_index].worker;
}

namespace sched
{

int count_execution_workers()
{
    if (!starpu_is_initialized())
    {
        return 1;
    }
    int const ncuda = starpu_worker_get_count_by_type(STARPU_CUDA_WORKER);
    if (ncuda > 0)
    {
        return ncuda;
    }
    int const ncpu = starpu_worker_get_count_by_type(STARPU_CPU_WORKER);
    return ncpu > 0 ? ncpu : 1;
}

int logical_worker_to_starpu_id(int logical_worker, bool use_cuda_workers)
{
    if (!starpu_is_initialized() || logical_worker < 0)
    {
        return -1;
    }
    if (use_cuda_workers)
    {
        return starpu_worker_get_by_type(STARPU_CUDA_WORKER, logical_worker);
    }
    return starpu_worker_get_by_type(STARPU_CPU_WORKER, logical_worker);
}

} // namespace sched

ExecutionSchedule generate_round_robin_execution_schedule(
    TileGraph const &graph,
    std::vector<std::shared_ptr<TileGraph::OpNode>> const &execution_order)
{
    return build_execution_schedule(graph,
        execution_order,
        "round_robin_virtual_tensor_split",
        assign_tensor_tiles_round_robin);
}

ExecutionSchedule generate_affinity_batch_execution_schedule(
    TileGraph const &graph,
    std::vector<std::shared_ptr<TileGraph::OpNode>> const &execution_order)
{
    return build_execution_schedule(graph,
        execution_order,
        "affinity_batch_virtual_tensor_split",
        assign_tensor_tiles_affinity_batch);
}

std::string execution_schedule_to_json(ExecutionSchedule const &schedule)
{
    nlohmann::json j;
    j["policy"] = schedule.policy.empty()
        ? "round_robin_virtual_tensor_split"
        : schedule.policy;
    j["hardware"] = {
        {"num_workers", schedule.num_workers},
        {"worker_kind", schedule.use_cuda_workers ? "cuda" : "cpu"},
    };
    nlohmann::json fp = nlohmann::json::object();
    fp["op_count"] = schedule.fingerprint.op_count;
    fp["op_names"] = schedule.fingerprint.op_names;
    j["schedule_fingerprint"] = fp;
    nlohmann::json tiles = nlohmann::json::array();
    for (auto const &[name, worker] : schedule.tile_virtual_worker)
    {
        tiles.push_back({{"tile", name}, {"virtual_worker", worker}});
    }
    j["virtual_tile_workers"] = tiles;

    nlohmann::json ops = nlohmann::json::array();
    for (ScheduledOpEntry const &e : schedule.ops)
    {
        nlohmann::json o;
        o["index"] = e.execution_index;
        o["op"] = e.op_name;
        o["name"] = e.task_label;
        o["worker"] = e.worker;
        o["writable_tiles"] = e.writable_tiles;
        o["read_tiles"] = e.read_tiles;
        ops.push_back(o);
    }
    j["ops"] = ops;
    return j.dump(2) + "\n";
}

void write_execution_schedule_json(
    ExecutionSchedule const &schedule,
    std::string const &path)
{
    std::ofstream f(path);
    if (!f.good())
    {
        throw std::runtime_error(
            "Cannot write execution schedule file: " + path);
    }
    f << execution_schedule_to_json(schedule);
}

ExecutionSchedule load_execution_schedule_json(std::string const &path)
{
    std::ifstream f(path);
    if (!f.good())
    {
        throw std::runtime_error(
            "Cannot open execution schedule file: " + path);
    }
    nlohmann::json j = nlohmann::json::parse(f);

    ExecutionSchedule schedule;
    if (j.contains("policy") && j["policy"].is_string())
    {
        schedule.policy = j["policy"].get<std::string>();
    }
    if (j.contains("hardware") && j["hardware"].is_object())
    {
        auto const &hw = j["hardware"];
        if (hw.contains("num_workers") && hw["num_workers"].is_number_integer())
        {
            schedule.num_workers = hw["num_workers"].get<int>();
        }
        if (hw.contains("worker_kind") && hw["worker_kind"].is_string())
        {
            schedule.use_cuda_workers =
                hw["worker_kind"].get<std::string>() == "cuda";
        }
    }
    if (j.contains("schedule_fingerprint") &&
        j["schedule_fingerprint"].is_object())
    {
        auto const &fp = j["schedule_fingerprint"];
        schedule.fingerprint.op_count = fp.at("op_count").get<size_t>();
        if (fp.contains("op_names") && fp["op_names"].is_array())
        {
            for (auto const &n : fp["op_names"])
            {
                schedule.fingerprint.op_names.push_back(n.get<std::string>());
            }
        }
    }
    if (j.contains("virtual_tile_workers") &&
        j["virtual_tile_workers"].is_array())
    {
        for (auto const &el : j["virtual_tile_workers"])
        {
            if (!el.is_object() || !el.contains("tile") ||
                !el.contains("virtual_worker"))
            {
                throw std::runtime_error(
                    "execution.json: bad virtual_tile_workers entry");
            }
            schedule.tile_virtual_worker[el["tile"].get<std::string>()] =
                el["virtual_worker"].get<int>();
        }
    }
    if (!j.contains("ops") || !j["ops"].is_array())
    {
        throw std::runtime_error("execution.json: missing ops array");
    }
    for (auto const &o : j["ops"])
    {
        ScheduledOpEntry e;
        e.execution_index = o.at("index").get<size_t>();
        e.op_name = o.at("op").get<std::string>();
        if (o.contains("name"))
        {
            e.task_label = o["name"].get<std::string>();
        }
        e.worker = o.at("worker").get<int>();
        if (o.contains("writable_tiles") && o["writable_tiles"].is_array())
        {
            for (auto const &t : o["writable_tiles"])
            {
                e.writable_tiles.push_back(t.get<std::string>());
            }
        }
        if (o.contains("read_tiles") && o["read_tiles"].is_array())
        {
            for (auto const &t : o["read_tiles"])
            {
                e.read_tiles.push_back(t.get<std::string>());
            }
        }
        schedule.ops.push_back(std::move(e));
    }
    int const runtime_workers = sched::count_execution_workers();
    if (schedule.num_workers <= 0)
    {
        schedule.num_workers = runtime_workers;
    }
    else if (schedule.num_workers != runtime_workers)
    {
        throw std::runtime_error(
            "execution.json: hardware.num_workers (" +
            std::to_string(schedule.num_workers) +
            ") != runtime worker count (" +
            std::to_string(runtime_workers) + ")");
    }
    for (ScheduledOpEntry const &e : schedule.ops)
    {
        if (e.worker < 0 || e.worker >= runtime_workers)
        {
            throw std::runtime_error(
                "execution.json: ops[" +
                std::to_string(e.execution_index) + "] worker " +
                std::to_string(e.worker) + " out of range [0, " +
                std::to_string(runtime_workers) + ")");
        }
    }
    for (auto const &[tile, worker] : schedule.tile_virtual_worker)
    {
        if (worker < 0 || worker >= runtime_workers)
        {
            throw std::runtime_error(
                "execution.json: tile '" + tile + "' virtual_worker " +
                std::to_string(worker) + " out of range [0, " +
                std::to_string(runtime_workers) + ")");
        }
    }
    return schedule;
}

void generate_round_robin_execution_json(
    TileGraph const &graph,
    std::vector<std::shared_ptr<TileGraph::OpNode>> const &execution_order,
    std::string const &path)
{
    write_execution_schedule_json(
        generate_round_robin_execution_schedule(graph, execution_order), path);
}

} // namespace nntile
