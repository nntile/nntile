#!/bin/bash

# Script to refactor test files in tests/kernel directory
# Rename get_test_data to get_test_input_data and move reference computation

files=(
    "tests/kernel/multiply_slice.cc"
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
    "tests/kernel/scale.cc"
    "tests/kernel/scale_fiber.cc"
    "tests/kernel/gelutanh_inplace.cc"
    "tests/kernel/embedding.cc"
    "tests/kernel/silu_inplace.cc"
    "tests/kernel/norm_fiber.cc"
    "tests/kernel/scale_slice.cc"
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
    "tests/kernel/multiply.cc"
    "tests/kernel/hypot_scalar_inverse.cc"
    "tests/kernel/multiply_inplace.cc"
    "tests/kernel/silu.cc"
    "tests/kernel/relu_backward.cc"
    "tests/kernel/gelu_inplace.cc"
    "tests/kernel/logsumexp.cc"
    "tests/kernel/add_slice.cc"
    "tests/kernel/add_slice_inplace.cc"
    "tests/kernel/multiply_fiber_inplace.cc"
    "tests/kernel/add_inplace.cc"
    "tests/kernel/gelutanh_backward.cc"
    "tests/kernel/gelutanh.cc"
    "tests/kernel/sqrt.cc"
    "tests/kernel/add.cc"
)

echo "Starting refactoring of test files..."

for file in "${files[@]}"; do
    echo "Processing $file..."

    # Skip if file doesn't exist
    if [ ! -f "$file" ]; then
        echo "  Warning: $file does not exist, skipping"
        continue
    fi

    # Create backup
    cp "$file" "${file}.backup"

    # Step 1: Rename function definition and update comment
    sed -i 's|// Get test data and reference results|// Get test input data (reference computation is done separately)|g' "$file"

    # Step 2: Rename function name
    sed -i 's|get_test_data|get_test_input_data|g' "$file"

    # Step 3: Remove reference computation call from function (this is the tricky part)
    # We need to find the pattern where reference computation is called and remove it
    # This is typically a line like "reference_<operation>(data);"
    # Let's use a more sophisticated approach with awk to handle this

    # First, let's find all reference computation calls in the files
    reference_calls=$(grep -n "reference_.*(data)" "$file" | head -1 | cut -d: -f2 | tr -d ' ')
    if [ -n "$reference_calls" ]; then
        echo "  Found reference call: $reference_calls"
        # Remove the reference computation line (this is complex with sed, so let's use a different approach)
        # For now, let's manually handle this with a more targeted approach
    fi

    # Step 4: Update function calls - this is handled by the global rename above

    echo "  Completed $file"
done

echo "Refactoring completed. Please review the changes and handle reference computation removal manually if needed."

# Now we need to handle the reference computation removal and addition to verification tests
# This is more complex and needs to be done carefully for each file

echo "Now need to manually handle reference computation for each file..."
echo "For each verification test, add reference computation after get_test_input_data call"
echo "For each benchmark test, just use get_test_input_data without reference computation"