#!/usr/bin/env python3

import os
import re
import glob

def get_reference_function(filepath):
    """Extract the reference function name from the file"""
    with open(filepath, 'r') as f:
        content = f.read()

    # Look for the reference function call pattern
    match = re.search(r'reference_(\w+)\(data\)', content)
    if match:
        return match.group(1)
    return None

def add_reference_to_verification_test(filepath, ref_func):
    """Add reference computation to verification test"""
    with open(filepath, 'r') as f:
        content = f.read()

    # Pattern to find verification tests and add reference computation
    # This looks for tests with DataGen::PRESET, DataGen::RANDOM
    pattern = r'const DataGen strategy = GENERATE\(DataGen::PRESET, DataGen::RANDOM\);\s*\n\s*auto data = get_test_input_data<T>\(([^)]+)\);\s*\n\s*SECTION'

    def replacement(match):
        params = match.group(1)
        # Add reference computation before SECTION
        return f'''const DataGen strategy = GENERATE(DataGen::PRESET, DataGen::RANDOM);

    auto data = get_test_input_data<T>({params});

    // Compute reference outputs for verification
    reference_{ref_func}(data);

    SECTION('''

    new_content = re.sub(pattern, replacement, content, flags=re.MULTILINE | re.DOTALL)

    if new_content != content:
        with open(filepath, 'w') as f:
            f.write(new_content)
        print(f"  Added reference computation to verification test in {filepath}")
    else:
        print(f"  No verification test found in {filepath}")

def main():
    """Fix verification tests in all files"""
    files = [
        "tests/kernel/pow.cc",
        "tests/kernel/hypot.cc",
        "tests/kernel/norm_slice.cc",
        "tests/kernel/gelu.cc",
        "tests/kernel/maxsumexp.cc",
        "tests/kernel/embedding_backward.cc",
        "tests/kernel/relu.cc",
        "tests/kernel/norm_slice_inplace.cc",
        "tests/kernel/adamw_step.cc",
        "tests/kernel/add_fiber.cc",
        "tests/kernel/adam_step.cc",
        "tests/kernel/scale_fiber.cc",
        "tests/kernel/gelutanh_inplace.cc",
        "tests/kernel/embedding.cc",
        "tests/kernel/silu_inplace.cc",
        "tests/kernel/norm_fiber.cc",
        "tests/kernel/fill.cc",
        "tests/kernel/silu_backward.cc",
        "tests/kernel/relu_inplace.cc",
        "tests/kernel/sum_fiber.cc",
        "tests/kernel/softmax.cc",
        "tests/kernel/hypot_inplace.cc",
        "tests/kernel/multiply_fiber.cc",
        "tests/kernel/add_fiber_inplace.cc",
        "tests/kernel/gelu_backward.cc",
        "tests/kernel/accumulate_maxsumexp.cc",
        "tests/kernel/hypot_scalar_inverse.cc",
        "tests/kernel/multiply_inplace.cc",
        "tests/kernel/relu_backward.cc",
        "tests/kernel/gelu_inplace.cc",
        "tests/kernel/logsumexp.cc",
        "tests/kernel/add_slice.cc",
        "tests/kernel/add_slice_inplace.cc",
        "tests/kernel/multiply_fiber_inplace.cc",
        "tests/kernel/add_inplace.cc",
        "tests/kernel/gelutanh_backward.cc",
        "tests/kernel/gelutanh.cc"
    ]

    for filepath in files:
        if os.path.exists(filepath):
            ref_func = get_reference_function(filepath)
            if ref_func:
                print(f"Processing {filepath} with reference function: {ref_func}")
                add_reference_to_verification_test(filepath, ref_func)
            else:
                print(f"No reference function found in {filepath}")

    print("Verification test fixing completed!")

if __name__ == "__main__":
    main()