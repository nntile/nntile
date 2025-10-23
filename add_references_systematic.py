#!/usr/bin/env python3

import os
import re

# Mapping from file names to their reference function names
file_to_ref_func = {
    "tests/kernel/pow.cc": "pow",
    "tests/kernel/hypot.cc": "hypot",
    "tests/kernel/norm_slice.cc": "norm_slice",
    "tests/kernel/gelu.cc": "gelu",
    "tests/kernel/maxsumexp.cc": "maxsumexp",
    "tests/kernel/embedding_backward.cc": "embedding_backward",
    "tests/kernel/relu.cc": "relu",
    "tests/kernel/norm_slice_inplace.cc": "norm_slice_inplace",
    "tests/kernel/adamw_step.cc": "adamw_step",
    "tests/kernel/add_fiber.cc": "add_fiber",
    "tests/kernel/adam_step.cc": "adam_step",
    "tests/kernel/scale_fiber.cc": "scale_fiber",
    "tests/kernel/gelutanh_inplace.cc": "gelutanh_inplace",
    "tests/kernel/embedding.cc": "embedding",
    "tests/kernel/silu_inplace.cc": "silu_inplace",
    "tests/kernel/norm_fiber.cc": "norm_fiber",
    "tests/kernel/fill.cc": "fill",
    "tests/kernel/silu_backward.cc": "silu_backward",
    "tests/kernel/relu_inplace.cc": "relu_inplace",
    "tests/kernel/sum_fiber.cc": "sum_fiber",
    "tests/kernel/softmax.cc": "softmax",
    "tests/kernel/hypot_inplace.cc": "hypot_inplace",
    "tests/kernel/multiply_fiber.cc": "multiply_fiber",
    "tests/kernel/add_fiber_inplace.cc": "add_fiber_inplace",
    "tests/kernel/gelu_backward.cc": "gelu_backward",
    "tests/kernel/accumulate_maxsumexp.cc": "accumulate_maxsumexp",
    "tests/kernel/hypot_scalar_inverse.cc": "hypot_scalar_inverse",
    "tests/kernel/multiply_inplace.cc": "multiply_inplace",
    "tests/kernel/relu_backward.cc": "relu_backward",
    "tests/kernel/gelu_inplace.cc": "gelu_inplace",
    "tests/kernel/logsumexp.cc": "logsumexp",
    "tests/kernel/add_slice.cc": "add_slice",
    "tests/kernel/add_slice_inplace.cc": "add_slice_inplace",
    "tests/kernel/multiply_fiber_inplace.cc": "multiply_fiber_inplace",
    "tests/kernel/add_inplace.cc": "add_inplace",
    "tests/kernel/gelutanh_backward.cc": "gelutanh_backward",
    "tests/kernel/gelutanh.cc": "gelutanh",
}

def add_reference_computation(filepath, ref_func):
    """Add reference computation to verification test in the file"""
    with open(filepath, 'r') as f:
        content = f.read()

    # Pattern to find verification tests and add reference computation
    pattern = r'const DataGen strategy = GENERATE\(DataGen::PRESET, DataGen::RANDOM\);\s*\n\s*auto data = get_test_input_data<T>\(([^)]+)\);\s*\n\s*SECTION'

    def replacement(match):
        params = match.group(1)
        return f'''const DataGen strategy = GENERATE(DataGen::PRESET, DataGen::RANDOM);

    auto data = get_test_input_data<T>({params});

    // Compute reference outputs for verification
    reference_{ref_func}(data);

    SECTION('''

    new_content = re.sub(pattern, replacement, content, flags=re.MULTILINE | re.DOTALL)

    if new_content != content:
        with open(filepath, 'w') as f:
            f.write(new_content)
        print(f"  Added reference_{ref_func}(data) to verification test in {filepath}")
        return True
    else:
        print(f"  No verification test pattern found in {filepath}")
        return False

def main():
    """Add reference computation to all verification tests"""
    success_count = 0
    total_count = 0

    for filepath, ref_func in file_to_ref_func.items():
        if os.path.exists(filepath):
            total_count += 1
            if add_reference_computation(filepath, ref_func):
                success_count += 1

    print(f"Added reference computation to {success_count}/{total_count} files")

if __name__ == "__main__":
    main()