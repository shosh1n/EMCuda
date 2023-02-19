#include <algorithm>
#include <cublas.h>
#include <iostream>
#include <math.h>
#include <string.h>
#include <thrust/complex.h>
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
#include <thrust/fill.h>
#include <thrust/transform.h>
#include <thrust/scatter.h>
#include <thrust/sequence.h>
#include <thrust/tuple.h>


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

// SPARSE MATRIX DIVISION
struct sparseDivision
{
    __host__ __device__
    thrust::tuple<float,float> operator()(const thrust::tuple<float,float>& spMat, const thrust::tuple<float> &divi) const
    {
      auto out = thrust::make_tuple(thrust::get<0>(spMat)/thrust::get<0>(divi),thrust::get<1>(spMat)/thrust::get<0>(divi));
      return out;
    }

};

// SPARSE MATRIX MULTIPLICATION
__global__ void SpMM(float *d_A, float *d_B, float *d_C, int p, int *d_row_ptr, int *d_col_ptr)
{
    float transfer;
    int row = blockIdx.x * blockDim.x + threadIdx.x;


    if ((row < p))
        {

        float temp = 0;
        int offst = 0;

        if(row >0)
        {
            offst = 1;
        }

        int row_start = d_row_ptr[row];
        int row_end = d_row_ptr[row+1];

        for(int j = row_start; j < row_end; ++j)
        {
            temp += d_A[j-offst] * d_B[d_col_ptr[j-2*offst]];
        }
        d_C[row] = temp;
        }
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

int main() {

  dx = a / nx;

  Sx = Nx * dx;
  Nx = ceil(Sx / dx);
  Sx = Nx * dx;

  int Nx2 = 2 * Nx;
  float dx2 = dx / 2;
  int size = 281;
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
  thrust::fill(ER2.begin(), ER2.begin() + (nx1), n1*n1);
  thrust::fill(ER2.begin() + nx1, ER2.end() - nx2, n2*n2);
  thrust::fill(ER2.end() - (nx2), ER2.end(), n3*n3);

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
  thrust::host_vector<float> dERzz(size);

  thrust::device_vector<float> dURxx(size);
  thrust::device_vector<float> dURyy(size);
  thrust::device_vector<float> dURzz(size);

  // COPY THE ITERATORS INTO VECTOR CONTAINERS
  thrust::copy(thrust::device, ERxx.begin(), ERxx.end(), dERxx.begin());
  thrust::copy(thrust::device, ERyy.begin(), ERyy.end(), dERyy.begin());
  thrust::copy(thrust::host, ERzz.begin(), ERzz.end(), dERzz.begin());

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
  float NS[2] = {Nx, 1};
  float RES[2] = {dx, 1};
  int BC[2] = {0, 0};

  float d_Ns[2] = {Nx, 1};

  int Nx = d_Ns[0];
  int Ny = d_Ns[1];

  float dx = lam0 / nmax / NRES;
  float dy = RES[1];

  float kinc[2] = {0, 0};

  int M = Nx * Ny;


//BUILD DEX
  thrust::host_vector<float>mid_Diag(size);
  thrust::host_vector<float>top_Diag(size);

  thrust::fill(mid_Diag.begin(), mid_Diag.end(), -1);
  thrust::fill(top_Diag.begin(), top_Diag.end(), 1);

  //thrust::transform(sub_Diag.begin(), sub_Diag.end(), thrust::make_constant_iterator(dx), sub_Diag.begin(), thrust::divides<float>());
  thrust::transform(mid_Diag.begin(), mid_Diag.end(), thrust::make_constant_iterator(dx), mid_Diag.begin(), thrust::divides<float>());
  thrust::transform(top_Diag.begin(), top_Diag.end(), thrust::make_constant_iterator(dx), top_Diag.begin(), thrust::divides<float>());

//BUILD DHX
  thrust::host_vector<float>subDH_Diag(size);

  thrust::host_vector<float>midDH_Diag(size);

  thrust::fill(subDH_Diag.begin(), subDH_Diag.end(), -1);
  thrust::fill(midDH_Diag.begin(), midDH_Diag.end(), 1);

  thrust::transform(subDH_Diag.begin(), subDH_Diag.end(), thrust::make_constant_iterator(dx), subDH_Diag.begin(), thrust::divides<float>());
  thrust::transform(midDH_Diag.begin(), midDH_Diag.end(), thrust::make_constant_iterator(dx), midDH_Diag.begin(), thrust::divides<float>());


//PREPARE DIVISION  DEX/ERzz
//

  typedef thrust::host_vector<float>::iterator midDiag;
  typedef thrust::host_vector<float>::iterator topDiag;

  typedef thrust::tuple<midDiag,topDiag> IteratorTuple;
  typedef thrust::zip_iterator<IteratorTuple> ZipIterator;

  ZipIterator zipper(thrust::make_tuple(subDH_Diag.begin(), midDH_Diag.begin()));

  auto d_DEX_begin =  thrust::make_zip_iterator(thrust::make_tuple(&mid_Diag[0], &top_Diag[0]));
  auto d_DEX_end =  thrust::make_zip_iterator(thrust::make_tuple(&mid_Diag[size], &top_Diag[size]));

  auto d_ERzz_begin = thrust::make_zip_iterator(thrust::make_tuple(&dERzz[0]));
  auto d_ERzz_end   = thrust::make_zip_iterator(thrust::make_tuple(&dERzz[size]));

  thrust::host_vector<float> h_topResOut(size);
  thrust::host_vector<float> h_midResOut(size);

  thrust::zip_iterator<thrust::tuple<
  thrust::host_vector<float>::iterator,
  thrust::host_vector<float>::iterator>> zip_begin(thrust::make_tuple(mid_Diag.begin(), top_Diag.begin()));

  thrust::zip_iterator<thrust::tuple<
  thrust::host_vector<float>::iterator,
  thrust::host_vector<float>::iterator>> zip_end(thrust::make_tuple(mid_Diag.end(), top_Diag.end()));

  auto d_divRes_begin = thrust::make_zip_iterator(thrust::make_tuple(h_midResOut.begin(), h_topResOut.begin()));
  auto d_divRes_end = thrust::make_zip_iterator(thrust::make_tuple(h_midResOut.end(), h_topResOut.end()));

  //GET THE DIVISION FUNCTOR READY AND DIVIDE
  sparseDivision divByERzz;

  thrust::transform(zip_begin, zip_end, d_ERzz_begin, d_divRes_begin, divByERzz);

  thrust::host_vector<float> reduceZipRes;

  thrust::for_each(d_divRes_begin, d_divRes_end,
                   [&reduceZipRes] (const thrust::tuple<float, float>& tup)
                   {
                       reduceZipRes.push_back(thrust::get<0>(tup));
                       reduceZipRes.push_back(thrust::get<1>(tup));
                   });

  //DO THE MULTIPLICATION

  // 1. CREATE THE ARRAYS OUT OF TU
  // {
  // PLE STRUCTURE
  // 2. CREATE THE row_prt and col_ptr ARRAYS

  // 3. FEED INTO THE SPARSE MATRIX MULTIPLICATION

//sizeof(thrust::get<0>(spMat)




  thrust::host_vector<float> keyseq0(size*2);
  thrust::host_vector<float> keyseq1(size*2);

  thrust::sequence(thrust::host,keyseq0.begin(), keyseq0.end(), 1);
  thrust::sequence(thrust::host,keyseq1.begin(), keyseq1.end(), 1);


  typedef thrust::host_vector<float>::iterator Iter;
  strided_range<Iter>key0(keyseq0.begin(), keyseq0.end(), 2);
  strided_range<Iter>key1(keyseq1.begin()+1, keyseq1.end(), 2);


  thrust::host_vector<float> outkey1(size);
  thrust::device_vector<float> values1(size);

  thrust::host_vector<float> outkey2(size);
  thrust::device_vector<float> values2(size);

  thrust::copy(key0.begin(), key0.end(), outkey1.begin());
  thrust::copy(key1.begin(), key1.end(), outkey2.begin());




  for(int i = 0; i < size*2; ++i)
      {
          std::cout << reduceZipRes[i]  << " ";
      }
 // //thrust::copy(d_divRes.begin(), d_divRes.end(), std::ostream_iterator<float>(std::cout, " "));
 // thrust::for_each
 //std::cout << d.size()  << std::endl;
  return 0;
}
