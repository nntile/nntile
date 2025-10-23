#!/usr/bin/env python3

import os
import re
import glob

def refactor_file(filepath):
    """Refactor a single test file"""
    print(f"Processing {filepath}...")

    with open(filepath, 'r') as f:
        content = f.read()

    original_content = content

    # Step 1: Update function definition comment and name
    content = re.sub(
        r'// Get test data and reference results',
        '// Get test input data (reference computation is done separately)',
        content
    )

    # Step 2: Rename function
    content = re.sub(r'get_test_data\s*\(', 'get_test_input_data(', content)

    # Step 3: Remove reference computation call
    # This is tricky - we need to find the pattern "reference_<something>(data);" and remove it
    # Let's find all reference calls first
    reference_calls = re.findall(r'reference_\w+\(data\);', content)

    if reference_calls:
        print(f"  Found reference calls: {reference_calls}")
        # Remove the first reference call (usually the only one in the function)
        content = re.sub(r'(\s+)// Compute reference outputs\s*\n\s*reference_\w+\(data\);', '', content)

    # Step 4: Update function calls in tests
    # For verification tests (those with DataGen::RANDOM), add reference computation
    # For benchmark tests (those with only DataGen::PRESET), don't add reference computation

    # Find verification test calls and add reference computation
    # Pattern: test that has both PRESET and RANDOM
    verification_pattern = r'const DataGen strategy = GENERATE\(DataGen::PRESET, DataGen::RANDOM\);\s*\n\s*auto data = get_test_input_data<T>\(([^)]+)\);'
    verification_matches = re.finditer(verification_pattern, content, re.MULTILINE)

    for match in verification_matches:
        params = match.group(1)
        # Add reference computation after the function call
        replacement = f'const DataGen strategy = GENERATE(DataGen::PRESET, DataGen::RANDOM);\n\n    auto data = get_test_input_data<T>({params});\n\n    // Compute reference outputs for verification\n    reference_\\1(data);\n\n    SECTION('
        # We need to identify the specific reference function for each test
        # This is file-specific, so we'll need to handle this case by case

    # For now, let's just update the function calls without adding reference computation
    # The reference computation addition needs to be done manually for each file

    # Save the file if there were changes
    if content != original_content:
        with open(filepath, 'w') as f:
            f.write(content)
        print(f"  Updated {filepath}")
    else:
        print(f"  No changes needed for {filepath}")

def main():
    """Process all test files"""
    test_files = [
        "tests/kernel/multiply_slice.cc",
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

    # Skip files we've already processed
    processed_files = [
        "tests/kernel/sqrt.cc",
        "tests/kernel/add.cc",
        "tests/kernel/scale_slice.cc",
        "tests/kernel/multiply.cc"
    ]

    for filepath in test_files:
        if filepath not in processed_files and os.path.exists(filepath):
            refactor_file(filepath)

    print("Processing completed!")

if __name__ == "__main__":
    main()