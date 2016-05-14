
extern "C"
__global__ void reverseVec(int n, float *a, float *b)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i<n)
    {
        b[n-1-i] = a[i];
    }
}
