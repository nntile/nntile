#!/bin/bash

# Script to handle reference computation for all remaining files
# This removes reference calls from function definitions and adds them to verification tests

files=(
    "tests/kernel/pow.cc"
    "tests/kernel/hypot.cc"
    "tests/kernel/norm_slice.cc"
    "tests/kernel/gelu.cc"
    "tests/kernel/maxsumexp.cc"
    "tests/kernel/embedding_backward.cc"
    "tests/kernel/relu.cc"
    "tests/kernel/norm_slice_inplace.cc"
    "tests/kernel/adamw_step.cc"
    "tests/kernel/add_fiber.cc"
    "tests/kernel/adam_step.cc"
    "tests/kernel/scale_fiber.cc"
    "tests/kernel/gelutanh_inplace.cc"
    "tests/kernel/embedding.cc"
    "tests/kernel/silu_inplace.cc"
    "tests/kernel/norm_fiber.cc"
    "tests/kernel/fill.cc"
    "tests/kernel/silu_backward.cc"
    "tests/kernel/relu_inplace.cc"
    "tests/kernel/sum_fiber.cc"
    "tests/kernel/softmax.cc"
    "tests/kernel/hypot_inplace.cc"
    "tests/kernel/multiply_fiber.cc"
    "tests/kernel/add_fiber_inplace.cc"
    "tests/kernel/gelu_backward.cc"
    "tests/kernel/accumulate_maxsumexp.cc"
    "tests/kernel/hypot_scalar_inverse.cc"
    "tests/kernel/multiply_inplace.cc"
    "tests/kernel/relu_backward.cc"
    "tests/kernel/gelu_inplace.cc"
    "tests/kernel/logsumexp.cc"
    "tests/kernel/add_slice.cc"
    "tests/kernel/add_slice_inplace.cc"
    "tests/kernel/multiply_fiber_inplace.cc"
    "tests/kernel/add_inplace.cc"
    "tests/kernel/gelutanh_backward.cc"
    "tests/kernel/gelutanh.cc"
)

echo "Handling reference computation for all files..."

for file in "${files[@]}"; do
    echo "Processing $file..."

    if [ ! -f "$file" ]; then
        echo "  Warning: $file does not exist, skipping"
        continue
    fi

    # Create backup
    cp "$file" "${file}.backup"

    # Step 1: Find and remove reference computation from function definition
    # This is tricky with sed, so let's use a more targeted approach

    # Find the reference function name used in this file
    ref_func=$(grep -n "reference_.*(data)" "$file" | head -1 | sed 's/.*reference_\([^()]*\)(data).*/\1/')

    if [ -n "$ref_func" ]; then
        echo "  Found reference function: $ref_func"

        # Remove the reference computation line and the comment before it
        sed -i "/\/\/ Compute reference outputs/,+1d" "$file"
        # Also try to remove just the reference line if the above didn't work
        sed -i "/reference_${ref_func}(data);/d" "$file"
    fi

    # Step 2: Add reference computation to verification tests
    # Find verification tests (those with DataGen::RANDOM) and add reference computation

    # Check if this file has verification tests
    if grep -q "DataGen::PRESET, DataGen::RANDOM" "$file"; then
        echo "  Adding reference computation to verification tests"

        # This is complex to do with sed, so let's use a Python script for this part
        python3 -c "
import re

# Read the file
with open('$file', 'r') as f:
    content = f.read()

# Find the reference function name
ref_match = re.search(r'reference_(\w+)\(data\)', content)
if ref_match:
    ref_func = ref_match.group(1)

    # Add reference computation to verification tests
    def add_reference(match):
        params = match.group(1)
        return f'''const DataGen strategy = GENERATE(DataGen::PRESET, DataGen::RANDOM);

    auto data = get_test_input_data<T>({params});

    // Compute reference outputs for verification
    reference_{ref_func}(data);

    SECTION('''

    # Pattern for verification test
    pattern = r'const DataGen strategy = GENERATE\(DataGen::PRESET, DataGen::RANDOM\);\s*\n\s*auto data = get_test_input_data<T>\(([^)]+)\);\s*\n\s*SECTION'
    content = re.sub(pattern, add_reference, content, flags=re.MULTILINE | re.DOTALL)

    # Save the file
    with open('$file', 'w') as f:
        f.write(content)
"
    fi

    echo "  Completed $file"
done

echo "Reference computation handling completed!"
echo "Please verify the changes and fix any issues manually if needed."