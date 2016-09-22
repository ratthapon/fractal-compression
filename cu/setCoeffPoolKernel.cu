extern "C"
__global__ void setCoeffPoolKernel(
int nBatch,int rbs,int nDegree,int nDScale,

// arrays pointer
float *CA,
float *SA,
// pointer of array of pointer to pointer of array in arrays, nevermind i just stun you.
// p(i) = data(i + size(data))
float **CP,
float **SP
)
{
    int taskIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (taskIdx < nBatch)
    {
		// initialize domain arrays
		int nCoeff = ((nDegree - 1) * nDScale + 1);

		// pointing section
		CP[taskIdx] = (CA + taskIdx * nCoeff);
		SP[taskIdx] = (SA + taskIdx);
    }
}
