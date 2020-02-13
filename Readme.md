# Explanation

This repo was created to reproduce an apparent bug in CuSparse's
sparse-dense multiply operation, `cusparseSpMM`.
However, someone from NVIDIA pointed out that I was using `uint64_t`
indices for my sparse matrices, which is not supported yet in
`cusparseSpMM`. When I switched to `uint32_t`, I started getting the
correct result.


# Build

```
$ make
```

# Run

```
$ ./a.out
```

# Dependencies

* CUDA 10.1
