#include <cublas.h>
#include <iostream>
#include <math.h>
#include <string.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/functional.h>
#include <thrust/gather.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/partition.h>
#include <thrust/scatter.h>
#include <thrust/sequence.h>
#include <thrust/tuple.h>
#include <thrust/complex.h>

// Matrix Multiplication works as follows:
// 1. input the Arrays d_A, and
// 2.d_B that needs to be used
// 3. input the designated Array that shall be computed
// 4. set for the m colmuns of d_A
// 5. set for the n rows of d_B
// 6. set for the offset for elements in d_A that needs to be respected to
// stencil out the Matrix shape in row-major scheme EXAMPLE: <letter>o</letter>
// = 4 for a Matrix d_A that is 4x2
// 6. set the row-major offest for the Matrix d_B using <letter>p</letter>
// 7. set the variable in the for-loop to the deepest increment in the
// respective dimension
//  Template structure to pass to kernel

template <typename Iterator> class strided_range {
public:
  typedef typename thrust::iterator_difference<Iterator>::type difference_type;

  struct stride_functor
      : public thrust::unary_function<difference_type, difference_type> {
    difference_type stride;

    stride_functor(difference_type stride) : stride(stride) {}

    __host__ __device__ difference_type
    operator()(const difference_type &i) const {
      return stride * i;
    }
  };

  typedef typename thrust::counting_iterator<difference_type> CountingIterator;
  typedef typename thrust::transform_iterator<stride_functor, CountingIterator>
      TransformIterator;
  typedef typename thrust::permutation_iterator<Iterator, TransformIterator>
      PermutationIterator;

  // type of the strided_range iterator
  typedef PermutationIterator iterator;

  // construct strided_range for the range [first,last)
  strided_range(Iterator first, Iterator last, difference_type stride)
      : first(first), last(last), stride(stride) {}

  iterator begin(void) const {
    return PermutationIterator(
        first, TransformIterator(CountingIterator(0), stride_functor(stride)));
  }

  iterator end(void) const {
    return begin() + ((last - first) + (stride - 1)) / stride;
  }

protected:
  Iterator first;
  Iterator last;
  difference_type stride;
};

__global__ void MatrixMulKernel(float *d_A, float *d_B, float *d_C, int m,
                                int n, int o, int p) {

  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if ((row < p) && (col < p)) {
    float d_cValue = 0;
    {
      for (int i = 0; i < p; ++i) {
        d_cValue += d_A[row * o + i] * d_B[i * p + col];
      }
    }
    d_C[row * p + col] = d_cValue;
  }
}
struct diag_index : public thrust::unary_function<int, int> {
  diag_index(int rows) : rows(rows) {}

  __host__ __device__ int operator()(const int index) const {
    return (index + rows * (index % rows));
  }

  const int rows;
};

void diag(float *v_in, float *m_out) {

  int v_size = sizeof(v_in) / sizeof(float);
  thrust::device_vector<float> mat(v_size * v_size);
}

template <typename V>

void print_matrix(const V &A, int nr_rows_A, int nr_cols_A) {

  for (int i = 0; i < nr_rows_A; ++i) {
    for (int j = 0; j < nr_cols_A; ++j) {
      std::cout << A[j * nr_rows_A + i] << " ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}

template <typename T> struct KernelArray {
  T *_array;
  int _size;

  // constructor allows for implicit conversion
  KernelArray(thrust::device_vector<T> &dVec) {
    _array = thrust::raw_pointer_cast(&dVec[0]);
    _size = (int)dVec.size();
  }
};

__global__ void createVector(float *d_A, float *d_C, int m, int n) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if ((row < m) && (col < n)) {
    d_C[row * n + col] = 1;
  }
}

__global__ void setElem(float *d_A, int start, int end, float d_a) {
  for (int i = start; i < end; ++i) {
    d_A[i] = d_a;
  }
}

__global__ void diagM(float *d_A, float *d_B, int off) {
  for (int i = 0; i < off; ++i) {
    for (int j = 0; j < off; ++j) {
      if (j == i) {
        d_B[i] = d_A[j];
      }
    }
  }
}

float micrometers = 1;
float nanometers = 1e-3 * micrometers;
// WAVELENGTH AND MODE
float lam0 = 1.55 * micrometers; //
char MODE = 'H';

// SLAB WAVEGUIDE
float a = 1500 * nanometers;
float n1 = 1.0;
float n2 = 2.0;
float n3 = 1.5;

// GRID PARAMETERS
float nmax = n2;
float NRES = 20;    // 56.888990026/2;
float b = 3 * lam0; // 3

// NUMBER OF MODES TO CALCULATE
float NMODES = 4;

int m = 1;

float dx = lam0 / nmax / NRES;
float nx = ceil(a / dx);

float Sx = b + a + b;
float Nx = ceil(Sx / dx);

//<---------------------Execute-------------------->

dim3 threads(141, 141);
dim3 blocks(2, 2);

int main() {

  dx = a / nx;

  Sx = Nx * dx;
  Nx = ceil(Sx / dx);
  Sx = Nx * dx;

  int Nx2 = 2 * Nx;
  float dx2 = dx / 2;
  int size = Nx2 / 2;
  //  thrust::fill(xa.begin(), xa.end(),1);

  // CREATE X-AXIS
  thrust::device_vector<float> xa(Nx + 2); //+2

  // AXIS STARTS AT 1 to 284
  thrust::counting_iterator<float> iter(1);
  thrust::copy(iter, iter + xa.size(), xa.begin());

  // MULTIPLY AXIS-UNITS WITH THE STEPSIZE " dx "
  thrust::transform(xa.begin(), xa.end(), thrust::make_constant_iterator(dx),
                    xa.begin(), thrust::multiplies<float>());

  // INITIALIZE ARRAY WITH ONES
  thrust::device_vector<float> m_xa(Nx);
  thrust::fill(m_xa.begin(), m_xa.end(), 1);

  // CALCULATE MEAN OF THE THE STEP-SIZE-ARRAY SCALE
  float xmean = thrust::reduce(xa.begin(), xa.end()) / size;

  // SUBSTRACT THE MEAN FROM THE STEP-SIZE ARRAY
  using namespace thrust::placeholders;
  thrust::for_each(xa.begin(), xa.end(), _1 -= xmean);

  // CREATE MAGNETIC AND ELECTRIC ARRAY
  thrust::device_vector<float> ER2(Nx2);
  thrust::device_vector<float> UR2(Nx2);

  // DETERMINE THE ARRAY STARTING POSITIONS
  float nx1 = 1 + ceil(b / dx);
  float nx2 = 1 + round(b / dx2) - 1;

  // BUILD SLAB WAVEGUIDE
  // E-FIELD
  thrust::fill(ER2.begin(), ER2.begin() + (nx1), n1);
  thrust::fill(ER2.begin() + nx1, ER2.end() - nx2, n2);
  thrust::fill(ER2.end() - (nx2), ER2.end(), n3);

  // BUILD SLAB WAVEGUIDE
  // M-FIELD

  // EXTRACT YEE GRID ARRAYS ERxx: odd Array-Elements, ERyy & ERzz: even
  // Array-Elements
  typedef thrust::device_vector<float>::iterator Iterator;
  strided_range<Iterator> ERxx(ER2.begin() + 1, ER2.end(), 2);
  strided_range<Iterator> ERyy(ER2.begin(), ER2.end(), 2);
  strided_range<Iterator> ERzz(ER2.begin(), ER2.end(), 2);

  strided_range<Iterator> URxx(UR2.begin(), UR2.end(), 2);
  strided_range<Iterator> URyy(UR2.begin() + 1, UR2.end(), 2);
  strided_range<Iterator> URzz(UR2.begin() + 1, UR2.end(), 2);

  // CREATE VECTORS B.C. of TO BE ABLE TO TYPECAST PROPERPLY
  thrust::device_vector<float> dERxx(size);
  thrust::device_vector<float> dERyy(size);
  thrust::device_vector<float> dERzz(size);

  thrust::device_vector<float> dURxx(size);
  thrust::device_vector<float> dURyy(size);
  thrust::device_vector<float> dURzz(size);

  // COPY THE ITERATORS INTO VECTOR CONTAINERS
  thrust::copy(thrust::device, ERxx.begin(), ERxx.end(), dERxx.begin());
  thrust::copy(thrust::device, ERyy.begin(), ERyy.end(), dERyy.begin());
  thrust::copy(thrust::device, ERzz.begin(), ERzz.end(), dERzz.begin());

  thrust::copy(thrust::device, URxx.begin(), URxx.end(), dURxx.begin());
  thrust::copy(thrust::device, URyy.begin(), URyy.end(), dURyy.begin());
  thrust::copy(thrust::device, URzz.begin(), URzz.end(), dURzz.begin());

  thrust::device_vector<float> dERxx2(size * size);
  thrust::copy(thrust::device, ERxx.begin(), ERxx.end(), dERxx2.begin());

  float *d_C; // *d_A, *d_B;

  //   cudaMalloc(&d_A, size*sizeof(int));
  //   cudaMalloc(&d_B, size*sizeof(int));
  cudaMalloc(&d_C, size * size * sizeof(int));

  // BUILD DERIVATIVE MATRICES
  float k0 = 2 * M_PI / lam0;
  int NS[2] = {Nx, 1};
  float RES[2] = {dx, 1};
  int BC[2] = {0, 0};

  int d_Ns[2] = {Nx, 1};

  int Nx = d_Ns[0];
  int Ny = d_Ns[1];

  float dx = lam0 / nmax / NRES;
  float dy = RES[1];

  float kinc[2] = {0, 0};

  int M = Nx * Ny;

  //Zero Matrix
  //thrust::device_vector<float> d_Z(M * M);

  //print_matrix(d_Z, M, M);


  //CENTER DIAGONAL
  thrust::device_vector<float> d0(M);
  thrust::fill(d0.begin(), d0.end(), -1);

  //UPPER DIAGONAL
  thrust::device_vector<float> d1(M);
  thrust::fill(d1.begin(), d1.end(), 1);

  //CREATE A SPARSE MATRIX REPRESENTATION
  thrust::device_vector<float> DEX(2*size-1);
  thrust::device_vector<float> DHX(2*size-1);

  DHX = DEX;
  thrust::fill(DEX.begin(), DEX.end(), 1);
  thrust::fill(DEX.begin(), DEX.end(), -1);


  thrust::transform(DEX.begin(), DEX.end(),thrust::make_constant_iterator(dx), DEX.begin(), thrust::divides<float>());
  thrust::transform(DHX.begin(), DHX.end(),thrust::make_constant_iterator(dx), DHX.begin(), thrust::divides<float>());
  //
  //thrust::copy(dERzz.begin(), dERzz.end(), std::ostream_iterator<float>(std::cout, " "));

  // get the Determinante then do A= -(DEX/ERzz * DHX +ERyy)






  return 0;
}
