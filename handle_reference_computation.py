#!/usr/bin/env python3

import os
import re
import glob

def find_reference_calls(content):
    """Find reference computation calls in the content"""
    # Look for patterns like "reference_<function>(data);"
    pattern = r'reference_(\w+)\(data\);'
    matches = re.findall(pattern, content)
    return matches

def remove_reference_from_function(content, reference_functions):
    """Remove reference computation from function definition"""
    for ref_func in reference_functions:
        pattern = r'(\s+)// Compute reference outputs\s*\n\s*reference_' + re.escape(ref_func) + r'\(data\);\s*\n\s*return data;'
        replacement = r'\1return data;'
        content = re.sub(pattern, replacement, content, flags=re.MULTILINE)

        # Also handle case without comment
        pattern = r'(\s+)reference_' + re.escape(ref_func) + r'\(data\);\s*\n\s*return data;'
        replacement = r'\1return data;'
        content = re.sub(pattern, replacement, content, flags=re.MULTILINE)

    return content

def add_reference_to_verification_tests(content, reference_functions):
    """Add reference computation to verification tests"""
    # Find verification tests (those with DataGen::RANDOM)
    # and add reference computation after get_test_input_data call

    for ref_func in reference_functions:
        # Pattern for verification test calls
        pattern = r'const DataGen strategy = GENERATE\(DataGen::PRESET, DataGen::RANDOM\);\s*\n\s*auto data = get_test_input_data<T>\(([^)]+)\);\s*\n\s*SECTION'

        def replacement(match):
            params = match.group(1)
            return f'const DataGen strategy = GENERATE(DataGen::PRESET, DataGen::RANDOM);\n\n    auto data = get_test_input_data<T>({params});\n\n    // Compute reference outputs for verification\n    reference_{ref_func}(data);\n\n    SECTION'

        content = re.sub(pattern, replacement, content, flags=re.MULTILINE)

    return content

def process_file(filepath):
    """Process a single file"""
    print(f"Processing {filepath}...")

    with open(filepath, 'r') as f:
        content = f.read()

    # Find reference functions used in this file
    reference_functions = find_reference_calls(content)

    if not reference_functions:
        print(f"  No reference calls found in {filepath}")
        return

    print(f"  Found reference functions: {reference_functions}")

    # Remove reference computation from function definition
    content = remove_reference_from_function(content, reference_functions)

    # Add reference computation to verification tests
    content = add_reference_to_verification_tests(content, reference_functions)

    # Save the file
    with open(filepath, 'w') as f:
        f.write(content)

    print(f"  Updated {filepath}")

def main():
    """Process files that need reference computation handling"""
    files_to_process = [
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

    for filepath in files_to_process:
        if os.path.exists(filepath):
            process_file(filepath)

    print("Reference computation handling completed!")

if __name__ == "__main__":
    main()