#include <cuda_runtime.h>
#include <cusparse.h>
#include <iostream>

template<typename T>
void printCusparseDnMat(cusparseDnMatDescr_t dense_descr) {
  int64_t rows;
  int64_t cols;
  int64_t ld;
  T* values_dev;
  cudaDataType cuda_data_type;
  cusparseOrder_t order;
  cusparseDnMatGet(
    dense_descr,
    &rows,
    &cols,
    &ld,
    (void**)&values_dev,
    &cuda_data_type,
    &order
  );
  T* values_host = new T[rows*cols];
  cudaMemcpy(values_host, values_dev, rows*cols*sizeof(T), cudaMemcpyDeviceToHost);
  for (int64_t row = 0; row < rows; row++) {
    for (int64_t col = 0; col < cols; col++) {
      // Cusparse dense matrices are stored in column-major order
      std::cout << values_host[col*rows+row] << " ";
    }
    std::cout << std::endl;
  }
  std::cout << "  values: ";
  for (int64_t i = 0; i < rows*cols; i++) {
    std::cout << values_host[i] << " ";
  }
  std::cout << std::endl;
  std::cout << "  shape: " << rows << ", " << cols << std::endl;
  delete [] values_host;
}

template<typename T>
void printCusparseSpMat(cusparseSpMatDescr_t sparse_descr) {
  T* values_dev;
  int64_t* row_indices_dev;
  int64_t* col_indices_dev;
  int64_t rows;
  int64_t cols;
  int64_t nnz;
  cusparseIndexType_t idx_type;
  cusparseIndexBase_t idx_base;
  cudaDataType cuda_data_type;

  cusparseCooGet(
    sparse_descr,
    &rows,
    &cols,
    &nnz,
    (void**)&row_indices_dev,
    (void**)&col_indices_dev,
    (void**)&values_dev,
    &idx_type,
    &idx_base,
    &cuda_data_type
  );
  T* values_host = new T[nnz];
  int64_t* row_indices_host = new int64_t[nnz];
  int64_t* col_indices_host = new int64_t[nnz];
  cudaMemcpy(values_host, values_dev, nnz*sizeof(T), cudaMemcpyDeviceToHost);
  cudaMemcpy(row_indices_host, row_indices_dev, nnz*sizeof(int64_t), cudaMemcpyDeviceToHost);
  cudaMemcpy(col_indices_host, col_indices_dev, nnz*sizeof(int64_t), cudaMemcpyDeviceToHost);

  for (int64_t i = 0; i < nnz; i++) {
    std::cout << "(" << row_indices_host[i]
      << ", " << col_indices_host[i]
      << "): " << values_host[i] << std::endl;
  }
  std::cout << "  values: ";
  for (int64_t i = 0; i < nnz; i++) {
    std::cout << values_host[i] << " ";
  }
  std::cout << std::endl;
  std::cout << "  row_indices: ";
  for (int64_t i = 0; i < nnz; i++) {
    std::cout << row_indices_host[i] << " ";
  }
  std::cout << std::endl;
  std::cout << "  col_indices: ";
  for (int64_t i = 0; i < nnz; i++) {
    std::cout << col_indices_host[i] << " ";
  }
  std::cout << std::endl;
  std::cout << "  shape: " << rows << ", " << cols << std::endl;
  delete [] values_host;
  delete [] row_indices_host;
  delete [] col_indices_host;
}

void CHECK_CUSPARSE_ERROR(cusparseStatus_t status) {
    if (status != CUSPARSE_STATUS_SUCCESS) {
        std::cerr << "ERROR" << std::endl;
        exit(1);
    }
}

void do_cusparse_spmm3x3(
    int64_t a_nnz,
    double* a_values_host,
    int64_t* a_row_indices_host,
    int64_t* a_col_indices_host
) {
    // Create sparse matrix a
    int64_t a_rows = 3;
    int64_t a_cols = 3;

    double* a_values_dev;
    cudaMalloc(&a_values_dev,a_nnz*sizeof(double));

    int64_t* a_row_indices_dev;
    cudaMalloc(&a_row_indices_dev,a_nnz*sizeof(int64_t));

    int64_t* a_col_indices_dev;
    cudaMalloc(&a_col_indices_dev,a_nnz*sizeof(int64_t));

    cusparseSpMatDescr_t a_sparse_descr;

    CHECK_CUSPARSE_ERROR(cusparseCreateCoo(
        &a_sparse_descr,
        a_rows,
        a_cols,
        a_nnz,
        a_row_indices_dev,
        a_col_indices_dev,
        a_values_dev,
        CUSPARSE_INDEX_64I,
        CUSPARSE_INDEX_BASE_ZERO,
        CUDA_R_64F
    ));
    
    cudaMemcpy(a_values_dev, a_values_host, a_nnz*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(a_row_indices_dev, a_row_indices_host, a_nnz*sizeof(int64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(a_col_indices_dev, a_col_indices_host, a_nnz*sizeof(int64_t), cudaMemcpyHostToDevice);

    std::cout << "sparse matrix a:" << std::endl;
    printCusparseSpMat<double>(a_sparse_descr);


    // Create dense matrix b
    int64_t b_rows = 3;
    int64_t b_cols = 3;
    // cusparse is col major, so the following apparent matrix
    // is the transpose of what will actually be created
    double b_values_host[b_cols*b_rows] = {
        1, 4, 7,
        2, 5, 8,
        3, 6, 9
    };
    double* b_values_dev;
    cudaMalloc(&b_values_dev, b_rows*b_cols*sizeof(double));

    cusparseDnMatDescr_t b_dense_descr;

    CHECK_CUSPARSE_ERROR(cusparseCreateDnMat(
        &b_dense_descr,
        b_rows,
        b_cols,
        b_rows,
        b_values_dev,
        CUDA_R_64F,
        CUSPARSE_ORDER_COL
    ));
    cudaMemcpy(b_values_dev, b_values_host, b_rows*b_cols*sizeof(double), cudaMemcpyHostToDevice);

    std::cout << "dense matrix b:" << std::endl;
    printCusparseDnMat<double>(b_dense_descr);


    // Create matrix c

    int64_t c_rows = 3;
    int64_t c_cols = 3;
    double* c_values_dev;
    cudaMalloc(&c_values_dev, c_rows*c_cols*sizeof(double));

    cusparseDnMatDescr_t c_dense_descr;
    CHECK_CUSPARSE_ERROR(cusparseCreateDnMat(
        &c_dense_descr,
        c_rows,
        c_cols,
        c_rows,
        c_values_dev,
        CUDA_R_64F,
        CUSPARSE_ORDER_COL
    ));


    double alpha = 1;
    double beta = 0;

    cusparseHandle_t handle;
    CHECK_CUSPARSE_ERROR(cusparseCreate(&handle));

    size_t bufferSize;

    CHECK_CUSPARSE_ERROR(cusparseSpMM_bufferSize(
        handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        (void*)&alpha,
        a_sparse_descr,
        b_dense_descr,
        (void*)&beta,
        c_dense_descr,
        CUDA_R_64F,
        CUSPARSE_COOMM_ALG1,
        &bufferSize
    ));
    cudaDeviceSynchronize();

    void* buffer = nullptr;
    if (bufferSize > 0) {
        cudaMalloc(&buffer, bufferSize);
    }

    CHECK_CUSPARSE_ERROR(cusparseSpMM(
        handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        (void*)&alpha,
        a_sparse_descr,
        b_dense_descr,
        (void*)&beta,
        c_dense_descr,
        CUDA_R_64F,
        CUSPARSE_COOMM_ALG1,
        buffer
    ));

    printCusparseDnMat<double>(c_dense_descr);

    std::cout << std::endl;
    std::cout << std::endl;

    cudaFree(a_values_dev);
    cudaFree(a_row_indices_dev);
    cudaFree(a_col_indices_dev);
    if (bufferSize > 0) {
        cudaFree(buffer);
    }
    cudaFree(b_values_dev);
    cudaFree(c_values_dev);
    cusparseDestroySpMat(a_sparse_descr);
    cusparseDestroyDnMat(b_dense_descr);
    cusparseDestroyDnMat(c_dense_descr);
}

int main() {

    // Seems to always give the correct result if sparse
    // matrix only has 1 element
    int64_t nnz_0 = 1;
    double values_0[nnz_0] = {1};
    int64_t row_indices_0[nnz_0] = {1};
    int64_t col_indices_0[nnz_0] = {1};
    do_cusparse_spmm3x3(
        nnz_0,
        values_0,
        row_indices_0,
        col_indices_0
    );

    // This 2 element sparse matrix multiply gives the
    // correct result
    int64_t nnz_1 = 2;
    double values_1[nnz_1] = {1, 1};
    int64_t row_indices_1[nnz_1] = {1, 0};
    int64_t col_indices_1[nnz_1] = {1, 0};
    do_cusparse_spmm3x3(
        nnz_1,
        values_1,
        row_indices_1,
        col_indices_1
    );


    // However, simply reversing the order of the elements
    // from the previous sparse matrix gives an incorrect
    // result, which would seem to indicate that we need to
    // list the elements in backward order
    int64_t nnz_2 = 2;
    double values_2[nnz_2] = {1, 1};
    int64_t row_indices_2[nnz_2] = {0, 1};
    int64_t col_indices_2[nnz_2] = {0, 1};
    do_cusparse_spmm3x3(
        nnz_2,
        values_2,
        row_indices_2,
        col_indices_2
    );


    // But backward order fails with more than 2 elements
    int64_t nnz_3 = 3;
    double values_3[nnz_3] = {1, 1, 1};
    int64_t row_indices_3[nnz_3] = {2, 1, 0};
    int64_t col_indices_3[nnz_3] = {2, 1, 0};
    do_cusparse_spmm3x3(
        nnz_3,
        values_3,
        row_indices_3,
        col_indices_3
    );
}
