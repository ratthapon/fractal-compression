extern "C"
__global__ void memSetKernel(
int nBatch,int rbs,int nDegree,int nDScale, int dbStopIdx,int dBaseScale, float regularize,

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
		// initialize domain arrays
		int nCoeff = ((nDegree - 1) * nDScale + 1);
		int dpOffset = (taskIdx * rbs * nCoeff);
		for(int i = 0; i < rbs; i++){
			DA[dpOffset + i] = 1.0f; // power 0
		}
		for(int i = 0; i < rbs; i++){
			DA[dpOffset + i + rbs] = 0.0f; // power 1
		}
		
		int dStartIdx = taskIdx % (nBatch/2);
		for(int ds = 1; ds <= nDScale; ds++){
			// vec sumation
			int mapDStart = dStartIdx + (nDScale - ds) * (rbs/2);
			int dScale = dBaseScale * ds; // base_scale * current_scale
			for(int i = 0; i < dScale; i++){
				for(int j = 0; j < rbs; j++){
					if(taskIdx < (nBatch/2)){
						DA[dpOffset + rbs*ds + j] = DA[dpOffset + rbs*ds + j] + data[mapDStart + j*dScale + i];
					}else{ // gen reverse domain
						DA[dpOffset + rbs*ds + j] = DA[dpOffset + rbs*ds + j] + dataRev[mapDStart + j*dScale+ i];
					}
				}
			}
	
			// vec scalig
			for(int j = 0; j < rbs; j++){
				DA[dpOffset + rbs*ds + j] = DA[dpOffset + rbs*ds + j]/dScale;
			}
		}
		
		// calculate next degree
		for(int j = 2; j < nDegree; j++){
			int degreePad = (j * rbs * nDScale );
			for(int i = 0; i < rbs * nDScale; i++){
				DA[i + dpOffset + rbs + degreePad] = DA[j + dpOffset + rbs] * DA[j + dpOffset + rbs + degreePad - rbs] ; // power n>=2
			}
		}

		// initialize range and error arrays
		int rpOffset = (taskIdx * rbs);
		for(int j = 0; j < rbs; j++){
			RA[rpOffset + j] = R[j];
			EA[rpOffset + j] = R[j];
		}

		// initialize covariance matrix with regularization
		int apOffset = (taskIdx * nCoeff * nCoeff);
		for(int i = 0; i < nCoeff * nCoeff; i+= nCoeff+1){
			AA[apOffset + i] = regularize * regularize; // power 0
		}

		// pointing section
		DP[taskIdx] = (DA + taskIdx * rbs * nCoeff);
		RP[taskIdx] = (RA + taskIdx * rbs);
		AP[taskIdx] = (AA + taskIdx * nCoeff * nCoeff);
		BP[taskIdx] = (BA + taskIdx * nCoeff);
		IP[taskIdx] = (IA + taskIdx * nCoeff * nCoeff);
		CP[taskIdx] = (CA + taskIdx * nCoeff);
		EP[taskIdx] = (EA + taskIdx * rbs);
		SP[taskIdx] = (SA + taskIdx);
    }
}
