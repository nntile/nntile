#!/bin/bash

# Script to refactor remaining test files
# This handles the function renaming and call updates

remaining_files=(
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

echo "Starting refactoring of remaining files..."

for file in "${remaining_files[@]}"; do
    echo "Processing $file..."

    if [ ! -f "$file" ]; then
        echo "  Warning: $file does not exist, skipping"
        continue
    fi

    # Create backup
    cp "$file" "${file}.backup"

    # Step 1: Update function comment
    sed -i 's|// Get test data and reference results|// Get test input data (reference computation is done separately)|g' "$file"

    # Step 2: Rename function definition
    sed -i 's/get_test_data</get_test_input_data</g' "$file"

    # Step 3: Rename function calls
    sed -i 's/get_test_data</get_test_input_data</g' "$file"

    # Step 4: Update function names in comments if any
    sed -i 's/get_test_data(/get_test_input_data(/g' "$file"

    echo "  Completed basic refactoring for $file"
done

echo "Basic refactoring completed."
echo "Next steps:"
echo "1. Manually remove reference computation calls from function definitions"
echo "2. Manually add reference computation to verification tests"
echo "3. Verify all changes work correctly"

echo "Files that need manual reference computation handling:"
for file in "${remaining_files[@]}"; do
    if [ -f "$file" ]; then
        echo "  - $file"
    fi
done