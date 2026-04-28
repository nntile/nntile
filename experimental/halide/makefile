all: a.out

a.out: main.cc
	clang++ -std=c++17 -lHalide -o $@ $^

.PHONY: run
run: a.out
	./a.out

maxsumexp: maxsumexp.cc maxsumexp.a
	clang++ -std=c++20 -lHalide -o $@ $^

maxsumexp-bench: maxsumexp_bench.cc maxsumexp.a
	clang++ -std=c++20 -lHalide -lbenchmark \
		-I../../include -I../../build/include -o $@ $^ ../../build/libnntile.so
