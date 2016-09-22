extern "C"
__global__ void setRangePoolKernel(
int nBatch,int rbs,int nDegree,int nDScale,

float *R, // array of range
// arrays pointer
float *RA,
float *BA,
float *EA,
// pointer of array of pointer to pointer of array in arrays, nevermind i just stun you.
// p(i) = data(i + size(data))
float **RP,
float **BP,
float **EP
)
{
    int taskIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (taskIdx < nBatch)
    {
		// initialize domain arrays
		int nCoeff = ((nDegree - 1) * nDScale + 1);
		
		// initialize range and error arrays
		int rpOffset = (taskIdx * rbs);
		for(int j = 0; j < rbs; j++){
			RA[rpOffset + j] = R[j];
			EA[rpOffset + j] = R[j];
		}

		// pointing section
		RP[taskIdx] = (RA + taskIdx * rbs);
		BP[taskIdx] = (BA + taskIdx * nCoeff);
		EP[taskIdx] = (EA + taskIdx * rbs);
    }
}
