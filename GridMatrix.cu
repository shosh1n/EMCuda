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
#include <vector>
#include <cusolverSp.h>
#include <cusparse.h>
#include "cusolver_utils.h"

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
struct sparseDiMatrixDivision
{
    __host__ __device__
    thrust::tuple<float,float> operator()(const thrust::tuple<float,float>& spMat, const thrust::tuple<float> &divi) const
    {
      auto out = thrust::make_tuple(thrust::get<0>(spMat)/thrust::get<0>(divi),thrust::get<1>(spMat)/thrust::get<0>(divi));
      return out;
    }

};

// SPARSE MATRIX MULTIPLICATION

__global__ void diMatMulToTriMat(float *d_A, float *d_B, float *d_C, int barrier, int* d_elem_scan, int *d_row_ptr, int *d_col_ptr)
{
int focus = blockIdx.x * blockDim.x + threadIdx.x;
    if ((focus < barrier))
        {

        float temp = 0;
        int k = 0;
        int d_elem_start = d_elem_scan[focus];
        int d_elem_end = d_elem_scan[focus+1];

        int elem = d_elem_end - d_elem_start;
        for(int i = 0; i < elem; i++)
        {
          //this is a temp fix because otherwise the wrong col-element gets pulled
          //A fix will be implemented within generalized upcoming linear algebra routines
          if(i==1)
          {
            temp += d_A[d_row_ptr[focus+i]] * d_B[d_col_ptr[focus+i+1]];
          }

         else
         {
            temp += d_A[d_row_ptr[focus+i]] * d_B[d_col_ptr[focus+i]];
         }
            //DEBUGGING
            //uncomment to see why the fix above is neccessary; the root of the problems lies within the combination of threading and loop-increment
            //printf(" thread: %i  row: %i  col: %i  inc: %i escan: %i temp: %f \n", focus,d_row_ptr[focus +i], d_col_ptr[focus +i], i, elem, temp );
        }
        d_C[focus] = temp;
        }
}

void triangularDiagAdd(float *diagonal, float* addMe, int diagSize )
{

    for(int diagIdx = 0; diagIdx < diagSize; ++diagIdx)
        {
                if(diagIdx % 3 == 0)
                        {
                            diagonal[diagIdx] =diagonal[diagIdx] + addMe[diagIdx/3];
                        }
        }
}

void stepThroughTriElements(int *elem_list, int size)
     {
        int offset = 2;
        int inc = 0;
        int i = 0;
        while (i < size)
        {

            if(i == 0)
                {
                    elem_list[i]=0;
                    ++i;
                }


            if(offset == 2)
            {
                inc +=2;
                elem_list[i] = inc;
                offset = 0;
                ++i;
            }

            else
            {
                ++inc;
                elem_list[i] =inc;
                ++offset;
                ++i;
            }
        }

     }

struct vectorInversion
{
    __host__ __device__
    float operator()(const float& vec) const
    {
      auto out = 1/vec;
      return out;
    }

};

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

//Sequence for Col-Elements in threading scheme
void creatSeqPtrToCols(int *cols, int size)
{
    int barrier = 1;
    int even = 2;
    int normal = 0;
    for(int i = 0;  i < size; ++i)
        {
            if((barrier) !=2 || (i == 0) ){
                cols[i] = normal;
                ++normal;
                ++barrier;
                }

            else if((barrier == 2) &&( i != 0))
            {
                    cols[i]  = even;
                    barrier=0;
                    even+=2;
            }
            }
}

//Sequence for Row-Elements in threading scheme
void creatSeqPtrToRows(int *rows, int size)
    {

    int inc = 0;
    int k = 0;
    while(k <size)
        {
            if((k!=0)&&(k%3==0))
                {
                rows[k] = inc-1;
                ++k;
                }
            rows[k]=inc;
            ++inc;
            ++k;
            }
    }

//<---------------------Execute-------------------->

int main() {

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

  dx = a / nx;

  float Sx = b + a + b;
  float Nx = ceil(Sx / dx);
   Sx = Nx * dx;

  int Nx2 = 2 * Nx;
  float dx2 = dx / 2;
  int size = Nx;
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
  thrust::fill(ER2.begin(), ER2.begin() + 2*(nx1-1), n1*n1);
  thrust::fill(ER2.begin() + 2*(nx1-1), ER2.end() - (nx2-2), n2*n2);
  thrust::fill(ER2.end() - (nx2-2), ER2.end(), n3*n3);

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
  thrust::host_vector<float> dURyy(size);
  thrust::device_vector<float> dURzz(size);

  // COPY THE ITERATORS INTO VECTOR CONTAINERS
  thrust::copy(thrust::device, ERxx.begin(), ERxx.end(), dERxx.begin());
  thrust::copy(thrust::device, ERyy.begin(), ERyy.end(), dERyy.begin());
  thrust::copy(thrust::host, ERzz.begin(), ERzz.end(), dERzz.begin());

  thrust::fill(dURyy.begin(), dURyy.end(), 1);
  thrust::copy(thrust::device, URxx.begin(), URxx.end(), dURxx.begin());
  //thrust::copy(thrust::device, URyy.begin(), URyy.end(), dURyy.begin());
  thrust::copy(thrust::device, URzz.begin(), URzz.end(), dURzz.begin());


  thrust::device_vector<float> dERxx2(size * size);
  thrust::copy(thrust::device, ERxx.begin(), ERxx.end(), dERxx2.begin());

  // BUILD DERIVATIVE MATRICES
  float k0 = 2 * M_PI / lam0;
  float NS[2] = {Nx, 1};
  float RES[2] = {dx, 1};
  int BC[2] = {0, 0};

  float d_Ns[2] = {Nx, 1};

   Nx = d_Ns[0];
  int Ny = d_Ns[1];

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

  thrust::host_vector<float> h_Bvec;

    thrust::for_each(d_DEX_begin, d_DEX_end,
                   [&h_Bvec] (const thrust::tuple<float, float>& tup)
                   {
                       h_Bvec.push_back(thrust::get<0>(tup));
                       h_Bvec.push_back(thrust::get<1>(tup));
                   });


  thrust::transform(h_Bvec.begin(), h_Bvec.end(), thrust::make_constant_iterator(-1), h_Bvec.begin(), thrust::multiplies<float>());
  thrust::device_vector<float> d_Bvec(562);
  thrust::copy(h_Bvec.begin(), h_Bvec.end()-1, d_Bvec.begin());




  auto d_ERzz_begin = thrust::make_zip_iterator(thrust::make_tuple(&dERzz[0]));
  auto d_ERzz_end   = thrust::make_zip_iterator(thrust::make_tuple(&dERzz[size]));

  thrust::host_vector<float> h_topResOut(size);
  thrust::host_vector<float> h_midResOut(size);

  thrust::zip_iterator<thrust::tuple<thrust::host_vector<float>::iterator,thrust::host_vector<float>::iterator>> zip_begin(thrust::make_tuple(mid_Diag.begin(), top_Diag.begin()));

  thrust::zip_iterator<thrust::tuple<thrust::host_vector<float>::iterator,thrust::host_vector<float>::iterator>> zip_end(thrust::make_tuple(mid_Diag.end(), top_Diag.end()));

  auto d_divRes_begin = thrust::make_zip_iterator(thrust::make_tuple(h_midResOut.begin(), h_topResOut.begin()));
  auto d_divRes_end = thrust::make_zip_iterator(thrust::make_tuple(h_midResOut.end(), h_topResOut.end()));

  //GET THE DIVISION FUNCTOR READY AND DIVIDE
  sparseDiMatrixDivision divByERzz;

  thrust::transform(zip_begin, zip_end, d_ERzz_begin, d_divRes_begin, divByERzz);

  thrust::host_vector<float> h_Avec;


  //thrust::transform(top_Diag.begin(), top_Diag.end(), thrust::make_constant_iterator(dx), top_Diag.begin(), thrust::divides<float>());



  thrust::for_each(d_divRes_begin, d_divRes_end,
                   [&h_Avec] (const thrust::tuple<float, float>& tup)
                   {
                       h_Avec.push_back(thrust::get<0>(tup));
                       h_Avec.push_back(thrust::get<1>(tup));
                   });


  thrust::device_vector<float> d_Avec(562);

  thrust::copy(h_Avec.begin(), h_Avec.end()-1, d_Avec.begin());





  //DO THE MULTIPLICATION

  // 1. CREATE THE ARRAYS OUT OF TU
  // {
  // PLE STRUCTURE
  // 2. CREATE THE row_prt and col_ptr ARRAYS

  // 3. FEED INTO THE SPARSE MATRIX MULTIPLICATION

//sizeof(thrust::get<0>(spMat)

  int* row_ptr;
  int* col_ptr;
  int* elem_scan;

  int ptr_size = 842;
  int ptr_size_buffer = 840;
  row_ptr   = (int *)malloc(ptr_size_buffer*sizeof(int));
  col_ptr   = (int *)malloc(ptr_size_buffer*sizeof(int));
  elem_scan = (int *)malloc(ptr_size_buffer*sizeof(int));

  float* d_C ;

  //float* d_A = thrust::raw_pointer_cast(d_Avec.data());
  //float* d_B = thrust::raw_pointer_cast(d_Bvec.data());
  //cudaMemcpy(&d_A, &h_A, ptr_size*sizeof(float), cudaMemcpyHostToDevice);

  stepThroughTriElements(elem_scan,ptr_size);
  creatSeqPtrToCols(col_ptr, ptr_size-1);
  creatSeqPtrToRows(row_ptr, ptr_size-1);
//
  int* d_col_ptr;
  int* d_row_ptr;
  int* d_elem_scan;
//
  cudaMalloc((void**) &d_col_ptr, ptr_size*sizeof(int));
  cudaMalloc((void**) &d_row_ptr, ptr_size*sizeof(int));
  cudaMalloc((void**) &d_elem_scan, ptr_size*sizeof(int));
//
  cudaMalloc((void**) &d_C, ptr_size*sizeof(float));
//
  cudaMemcpy(d_col_ptr, col_ptr, ptr_size*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_row_ptr, row_ptr, ptr_size*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_elem_scan, elem_scan, ptr_size*sizeof(int), cudaMemcpyHostToDevice);

  diMatMulToTriMat<<<1,ptr_size>>>(thrust::raw_pointer_cast(&d_Avec[0]),thrust::raw_pointer_cast(&d_Bvec[0]), d_C, ptr_size, d_elem_scan, d_row_ptr, d_col_ptr);
//
  float* h_C;
  h_C = (float *)malloc(ptr_size*sizeof(float));

  cudaMemcpy(h_C, d_C, ptr_size*sizeof(float), cudaMemcpyDeviceToHost);

  thrust::host_vector<float> hURyy(size);
  thrust::copy(dURyy.begin(), dURyy.end(), hURyy.begin());

  triangularDiagAdd(h_C, thrust::raw_pointer_cast(&hURyy[0]), ptr_size);


  for(int i = 0; i <= ptr_size_buffer; ++i)
      {
          std::cout << "IdX " << i << ": " << h_C[i] <<", ";
      }


  vectorInversion vecInv;

  thrust::host_vector <float> inv_dERzz(size);

  thrust::transform(dERzz.begin(), dERzz.end(), inv_dERzz.begin(), vecInv);
  //size_t n = sizeof(h_C)/sizeof(h_C[0]);


  //NOW ALL WHAT IS LEFT TO DO IS GET THE CUDA-SOLVER TO WORK
  //ARRAYS ARE READY LOCK' AND LOAD 'EM - ROCK 'N ROLL! <2023-02-27 Mon> shoshin
  //PS: ADD THE CUDA-SOLVER TO THE NIX-SHELL!

  cusolverSpHandle_t cusolverH = NULL;
  csrqrInfo_t info = NULL;
  cusparseMatDescr_t descrA  = NULL;
  cudaStream_t stream = NULL;

  //prepare Arrays for the solver to be loaded

  int *d_csrRowPtrA = nullptr;
  int *d_csrColIndA = nullptr;
  double *d_csrValA = nullptr;

  double *d_b = nullptr;
  double *d_x = nullptr;

  size_t size_qr = 0;
  size_t size_internal = 0;

  void *buffer_qr = nullptr;

  const int i_m = size;
  const int nnzA = 840;




  //std::cout << top_Diag.size() + mid_Diag.size();
  cudaFree(d_col_ptr);
  cudaFree(d_row_ptr);
  cudaFree(d_elem_scan);
  cudaFree(d_C);

  free(row_ptr);
  free(col_ptr);
  free(elem_scan);
 // //thrust::copy(d_divRes.begin(), d_divRes.end(), std::ostream_iterator<float>(std::cout, " "));
 // thrust::for_each
 //std::cout << d.size()  << std::endl;
  return 0;
}
