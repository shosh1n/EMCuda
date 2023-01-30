
#include <iostream>
#include <string.h>
#include <math.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/functional.h>
#include <thrust/gather.h>
#include <cublas.h>
#include <thrust/for_each.h>
#include <thrust/partition.h>
#include <thrust/execution_policy.h>





//Matrix Multiplication works as follows:
//1. input the Arrays d_A, and
//2.d_B that needs to be used
//3. input the designated Array that shall be computed
//4. set for the m colmuns of d_A
//5. set for the n rows of d_B
//6. set for the offset for elements in d_A that needs to be respected to stencil out the Matrix shape in row-major scheme
//EXAMPLE: <letter>o</letter> = 4 for a Matrix d_A that is 4x2
//6. set the row-major offest for the Matrix d_B using <letter>p</letter>
//7. set the variable in the for-loop to the deepest increment in the respective dimension
// Template structure to pass to kernel
template < typename T >
struct KernelArray
{
T* _array;
int _size;

// constructor allows for implicit conversion
KernelArray(thrust::device_vector<T>& dVec) {
    _array = thrust::raw_pointer_cast( &dVec[0] );
    _size  = ( int ) dVec.size();
}

};




template <typename Iterator>
class strided_range
{
    public:

    typedef typename thrust::iterator_difference<Iterator>::type difference_type;

    struct stride_functor : public thrust::unary_function<difference_type,difference_type>
    {
        difference_type stride;

        stride_functor(difference_type stride)
            : stride(stride) {}

        __host__ __device__
        difference_type operator()(const difference_type& i) const
        {
            return stride * i;
        }
    };

    typedef typename thrust::counting_iterator<difference_type>                   CountingIterator;
    typedef typename thrust::transform_iterator<stride_functor, CountingIterator> TransformIterator;
    typedef typename thrust::permutation_iterator<Iterator,TransformIterator>     PermutationIterator;

    // type of the strided_range iterator
    typedef PermutationIterator iterator;

    // construct strided_range for the range [first,last)
    strided_range(Iterator first, Iterator last, difference_type stride)
        : first(first), last(last), stride(stride) {}

    iterator begin(void) const
    {
        return PermutationIterator(first, TransformIterator(CountingIterator(0), stride_functor(stride)));
    }

    iterator end(void) const
    {
        return begin() + ((last - first) + (stride - 1)) / stride;
    }

    protected:
    Iterator first;
    Iterator last;
    difference_type stride;
};




__global__ void MatrixMulKernel(float* d_A, float *d_B, float *d_C, int m, int n, int o, int p)
    {

        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;

        if ((row < p) && (col < p))
            {
            float d_cValue = 0;
            {
                for (int i = 0; i < p; ++i )
                    {
                        d_cValue += d_A[row*o+i]*d_B[i*p+col];
                    }
            }
            d_C[row*p+col] = d_cValue;
            }
    }




void print_matrix(const float *A, int nr_rows_A, int nr_cols_A) {

     for(int i = 0; i < nr_rows_A; ++i){
         for(int j = 0; j < nr_cols_A; ++j){
             std::cout << A[j * nr_rows_A + i] << " ";
         }
         std::cout << std::endl;
     }
     std::cout << std::endl;
 }



__global__ void createVector(float* d_A, float* d_C, int m, int n)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if( (row < m ) && (col < n))
        {
            d_C[row*n+col] = 1;
        }
}

__global__ void setElem(float* d_A, int start, int end, float d_a)
    {
        for(int i = start; i < end; ++i )
            {
               d_A[i] =  d_a;
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
float Nx = ceil(Sx/dx);


//<---------------------Execute-------------------->

dim3 threads(150,150);
dim3 blocks(2,2);


int main()
{

    dx = a / nx;

    Sx = Nx * dx;
    Nx = ceil(Sx/dx);
    Sx = Nx * dx;

    float Nx2 = 2*Nx;
    float dx2 = dx/2;

  //  thrust::fill(xa.begin(), xa.end(),1);

    //CREATE X-AXIS
    thrust::device_vector<float>xa(Nx+2);

    //AXIS STARTS AT 1 to 284
    thrust::counting_iterator<float> iter(1);
    thrust::copy(iter, iter + xa.size(), xa.begin());

    //MULTIPLY AXIS-UNITS WITH THE STEPSIZE " dx "
    thrust::transform(xa.begin(), xa.end(), thrust::make_constant_iterator(dx), xa.begin(), thrust::multiplies<float>());

    //INITIALIZE ARRAY WITH ONES
    thrust::device_vector<float> m_xa(Nx) ;
    thrust::fill(m_xa.begin(), m_xa.end(),1);

    //CALCULATE MEAN OF THE THE STEP-SIZE-ARRAY SCALE
    float xmean = thrust::reduce(xa.begin(), xa.end())/281;

    //SUBSTRACT THE MEAN FROM THE STEP-SIZE ARRAY
    using namespace thrust::placeholders;
    thrust::for_each(xa.begin(), xa.end(), _1 -= xmean);

    //CREATE MAGNETIC AND ELECTRIC ARRAY
    thrust::device_vector<float>ER2(Nx2);
    thrust::device_vector<float>UR2(Nx2);

    //DETERMINE THE ARRAY STARTING POSITIONS
    float nx1 = 1 + ceil(b/dx);
    float nx2 = 1 + round(b/dx2)-1;


    //BUILD SLAB WAVEGUIDE
    thrust::fill(ER2.begin(), ER2.begin() + (nx1), n1 * n1);
    thrust::fill(ER2.begin() + nx1, ER2.end() - nx2, n2 * n2);
    thrust::fill(ER2.end() - (nx2), ER2.end(), n3 * n3);

    //EXTRACT YEE GRID ARRAYS

    typedef thrust::device_vector<float>::iterator Iterator;
    strided_range<Iterator> ERxx(ER2.begin()+1, ER2.end(), 2);
    strided_range<Iterator> ERyy(ER2.begin(), ER2.end(), 2);
    strided_range<Iterator> ERzz(ER2.begin(), ER2.end(), 2);

    thrust::device_vector<float> dERxx(Nx2*Nx2);
    thrust::copy(thrust::device, ERxx.begin(), ERxx.end(), dERxx.begin());

    thrust::device_vector<float> dERxx2(Nx2*Nx2);
    thrust::copy(thrust::device, ERxx.begin(), ERxx.end(), dERxx2.begin());

    float *d_C; // *d_A, *d_B;

 //   cudaMalloc(&d_A, size*sizeof(int));
 //   cudaMalloc(&d_B, size*sizeof(int));
      cudaMalloc(&d_C, Nx2*Nx2*sizeof(int));

 //   cudaMemcpy(d_A, h_A, size*sizeof(int), cudaMemcpyHostToDevice);
 //   cudaMemcpy(d_B, h_B, size*sizeof(int), cudaMemcpyHostToDevice);





    //1.) input:d_A 2.)input:d_B 3.)input:d_C 4.)cols d_A 5.)rows d_B 6.)deepest increment cols_A 7.)deepest increments rows rows_B

    MatrixMulKernel<<<threads,blocks>>>(thrust::raw_pointer_cast(&dERxx[0]), thrust::raw_pointer_cast(&dERxx2[0]), d_C, 1 , 1, 1, Nx2/2);

     float *h_C = (float *)malloc(Nx2*Nx2 * sizeof(int));

     cudaMemcpy(h_C, d_C, Nx2*Nx2 *sizeof(int), cudaMemcpyDeviceToHost);



     print_matrix(h_C, Nx2/2, Nx2/2);

 //   cudaFree(d_A);
 //   cudaFree(d_B);
    cudaFree(d_C);

    free(h_C);

    return 0;


}
