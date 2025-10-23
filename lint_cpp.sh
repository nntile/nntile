#!/bin/bash

# Simple C++ linting script for NNTile project
# This script runs basic checks on C++ files

echo "Running C++ linting checks..."

# Check for basic formatting issues
echo "1. Checking for trailing whitespace..."
find . -name "*.cc" -o -name "*.hh" -o -name "*.hpp" -o -name "*.cpp" | grep -v external | xargs grep -l " $" | while read file; do
    echo "  Trailing whitespace found in: $file"
done

# Check for tabs
echo "2. Checking for tabs..."
find . -name "*.cc" -o -name "*.hh" -o -name "*.hpp" -o -name "*.cpp" | grep -v external | xargs grep -l "	" | while read file; do
    echo "  Tabs found in: $file"
done

# Check for missing final newlines
echo "3. Checking for missing final newlines..."
find . -name "*.cc" -o -name "*.hh" -o -name "*.hpp" -o -name "*.cpp" | grep -v external | while read file; do
    if [ -n "$(tail -c1 "$file")" ]; then
        echo "  Missing final newline in: $file"
    fi
done

# Check for long lines (over 100 characters)
echo "4. Checking for long lines (>100 characters)..."
find . -name "*.cc" -o -name "*.hh" -o -name "*.hpp" -o -name "*.cpp" | grep -v external | xargs awk 'length($0) > 100 {print FILENAME ":" NR ":" $0}' | head -20

echo "Linting checks completed."