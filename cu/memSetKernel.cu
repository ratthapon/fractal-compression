extern "C"
__global__ void memSetKernel(
int nBatch,int nRangeBlockSize,int nCoeff,int nDScale, int dbStopIdx,int dScale, float regularize,

float *data,float *dataRev, // array of data and reverse data
float *R, // array of range
// arrays pointer
float *DA, float *RA,
float *AA, float *BA,
float *IA, float *CA,
float *EA, float *SA
// pointer of array of pointer to pointer of array in arrays, nevermind i just stun you.
// p(i) = data(i + size(data))
,float **DP, float **RP,
float **AP, float **BP,
float **IP, float **CP,
float **EP, float **SP
)
{
    int taskIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (taskIdx < nBatch)
    {
		int dColStart = taskIdx % (nBatch/2);
		// initialize domain arrays
		int dpOffset = (taskIdx * nRangeBlockSize * nCoeff);
		for(int i = 0; i < nRangeBlockSize; i++){
			DA[dpOffset + i] = 1.0f; // power 0
		}
		for(int i = 0; i < nRangeBlockSize; i++){
			DA[dpOffset + i + nRangeBlockSize] = 0.0f; // power 1
		}
		// vec sumation
		if(taskIdx < (nBatch/2)){
			for(int i = 0; i < dScale; i++){
				for(int j = 0; j < nRangeBlockSize; j++){
					DA[dpOffset + nRangeBlockSize + j] = DA[dpOffset + nRangeBlockSize + j] + data[dColStart + j*dScale + i];
				}
			}
		}else{ // gen reverse domain
			for(int i = 0; i < dScale; i++){
				for(int j = 0; j < nRangeBlockSize; j++){
					DA[dpOffset + nRangeBlockSize + j] = DA[dpOffset + nRangeBlockSize + j] + dataRev[dColStart + j*dScale+ i];
				}
			}
		}
		// vec scalig
		for(int j = 0; j < nRangeBlockSize; j++){
			DA[dpOffset + nRangeBlockSize + j] = DA[dpOffset + nRangeBlockSize + j]/dScale;
		}
		// calculate next degree
		for(int j = 2; j < nCoeff; j++){
			int degreePad = (j*nRangeBlockSize);
			for(int i = 0; i < nRangeBlockSize; i++){
				DA[i + dpOffset + nRangeBlockSize + degreePad] = DA[j + dpOffset + nRangeBlockSize] * DA[j + dpOffset + nRangeBlockSize + degreePad - nRangeBlockSize] ; // power n>=2
			}
		}

		// initialize range and error arrays
		int rpOffset = (taskIdx * nRangeBlockSize);
		for(int j = 0; j < nRangeBlockSize; j++){
			RA[rpOffset + j] = R[j];
			EA[rpOffset + j] = R[j];
		}

		// initialize covariance matrix with regularization
		int apOffset = (taskIdx * nCoeff * nCoeff);
		for(int i = 0; i < nCoeff * nCoeff; i+= nCoeff+1){
			AA[apOffset + i] = regularize * regularize; // power 0
		}

		// pointing section
		DP[taskIdx] = (DA + taskIdx * nRangeBlockSize * nCoeff);
		RP[taskIdx] = (RA + taskIdx * nRangeBlockSize);
		AP[taskIdx] = (AA + taskIdx * nCoeff * nCoeff);
		BP[taskIdx] = (BA + taskIdx * nCoeff);
		IP[taskIdx] = (IA + taskIdx * nCoeff * nCoeff);
		CP[taskIdx] = (CA + taskIdx * nCoeff);
		EP[taskIdx] = (EA + taskIdx * nRangeBlockSize);
		SP[taskIdx] = (SA + taskIdx);
    }
}
