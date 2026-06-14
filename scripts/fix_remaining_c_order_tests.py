#!/usr/bin/env python3
"""Fix remaining C-order test shape/buffer mismatches."""

from __future__ import annotations

from pathlib import Path

REPLACEMENTS: list[tuple[str, str, str]] = [
    # Rope: rope pairs on first dim when sin is 1D {2}
    ("nntile/tests/core/rope.cc",
     "Tile<T> sin({2}), cos({2}), src({5, 4}), dst({5, 4}), dst_ref({5, 4});",
     "Tile<T> sin({2}), cos({2}), src({4, 5}), dst({4, 5}), dst_ref({4, 5});"),
    ("nntile/tests/core/rope_backward.cc",
     "Tile<T> sin({2}), cos({2}), dy({5, 4}), dx({5, 4}), dx_ref({5, 4});",
     "Tile<T> sin({2}), cos({2}), dy({4, 5}), dx({4, 5}), dx_ref({4, 5});"),
    ("nntile/tests/tile/ops/rope.cc",
     "const std::vector<Index> sh = {2}, tsh = {5, 4};",
     "const std::vector<Index> sh = {2}, tsh = {4, 5};"),
    ("nntile/tests/tile/ops/rope_backward.cc",
     "const std::vector<Index> sh = {2}, tsh = {5, 4};",
     "const std::vector<Index> sh = {2}, tsh = {4, 5};"),
    # sum_fiber output buffers
    ("nntile/tests/tile/ops/sum_fiber.cc",
     "std::vector<float> dd(3, 0.f);",
     "std::vector<float> dd(5, 0.f);"),
    ("nntile/tests/tile/ops/sum_fiber.cc",
     "for(Index j = 0; j < 3; ++j) { b[j] = Y(0); }",
     "for(Index j = 0; j < 5; ++j) { b[j] = Y(0); }"),
    ("nntile/tests/tile/ops/sum_fiber.cc",
     "std::vector<float> tref(3);",
     "std::vector<float> tref(5);"),
    ("nntile/tests/tile/ops/sum_fiber.cc",
     "for(Index j = 0; j < 3; ++j) { tref[static_cast<size_t>(j)] = static_cast<float>(l2[j]); }",
     "for(Index j = 0; j < 5; ++j) { tref[static_cast<size_t>(j)] = static_cast<float>(l2[j]); }"),
    ("nntile/tests/tile/ops/sum_fiber.cc",
     "for(size_t j = 0; j < 3; ++j) { REQUIRE(std::abs(gout[j] - tref[j]) < tol); }",
     "for(size_t j = 0; j < 5; ++j) { REQUIRE(std::abs(gout[j] - tref[j]) < tol); }"),
    ("nntile/tests/tile/ops/sumprod_fiber.cc",
     "std::vector<float> dd(3, 0.f);",
     "std::vector<float> dd(5, 0.f);"),
    # slice at axis 1 on {5,4,3} -> {5,3}
    ("nntile/tests/tile/ops/add_slice_inplace.cc",
     "const std::vector<Index> t1s = {4, 3}, t2s = {5, 4, 3};",
     "const std::vector<Index> t1s = {5, 3}, t2s = {5, 4, 3};"),
    ("nntile/tests/tile/ops/scale_slice.cc",
     "const std::vector<Index> t1s = {4, 3}, t2s = {5, 4, 3};",
     "const std::vector<Index> t1s = {5, 3}, t2s = {5, 4, 3};"),
    # logsumexp tile: drop last dim of {3,2,2} -> {3,2}
    ("nntile/tests/tile/ops/logsumexp.cc",
     "const std::vector<Index> sh_dst = {2, 2};",
     "const std::vector<Index> sh_dst = {3, 2};"),
    # embedding output shapes
    ("nntile/tests/tensor/ops/embedding.cc",
     "{4, 5, 100}",
     "{5, 4, 100}"),
    ("nntile/tests/tensor/ops/embedding_backward.cc",
     "{4, 5, 100}",
     "{5, 4, 100}"),
    ("nntile/tests/module/embedding.cc",
     "std::vector<Index>({100, 10})",
     "std::vector<Index>({10, 100})"),
    ("nntile/tests/module/embedding.cc",
     "std::vector<Index>({4, 5, 100})",
     "std::vector<Index>({5, 4, 100})"),
    ("nntile/tests/module/embedding.cc",
     "std::vector<Index>({100, 10})",
     "std::vector<Index>({10, 100})"),
    ("nntile/tests/module/embedding.cc",
     "auto *vocab = g.tensor({50, 8}, DataType::FP32)->set_name(\"shared_vocab\");",
     "auto *vocab = g.tensor({8, 50}, DataType::FP32)->set_name(\"shared_vocab\");"),
    ("nntile/tests/module/embedding.cc",
     "REQUIRE(emb.num_embeddings() == 8);\n    REQUIRE(emb.embed_dim() == 50);",
     "REQUIRE(emb.num_embeddings() == 8);\n    REQUIRE(emb.embed_dim() == 50);"),
    # cross_entropy: class axis is last in C-order
    ("nntile/tests/nn/ops/cross_entropy.cc",
     "GENERATE(std::tuple{std::vector<Index>{5, 7}, std::vector<Index>{7}},\n"
     "            std::tuple{std::vector<Index>{4, 3, 2}, std::vector<Index>{3, 2}});",
     "GENERATE(std::tuple{std::vector<Index>{7, 5}, std::vector<Index>{7}},\n"
     "            std::tuple{std::vector<Index>{4, 3, 2}, std::vector<Index>{4, 3}});"),
    ("nntile/tests/nn/ops/cross_entropy.cc",
     "std::vector<Index> x_shape{5, 7};\n    std::vector<Index> labels_shape{7};",
     "std::vector<Index> x_shape{7, 5};\n    std::vector<Index> labels_shape{7};",
     ),
]


def main() -> None:
    changed = 0
    for path_str, old, new in REPLACEMENTS:
        path = Path(path_str)
        if not path.exists():
            print("skip missing", path)
            continue
        text = path.read_text()
        if old not in text:
            print("no match in", path)
            continue
        path.write_text(text.replace(old, new))
        print("fixed", path)
        changed += 1
    print(f"total: {changed}")


if __name__ == "__main__":
    main()
