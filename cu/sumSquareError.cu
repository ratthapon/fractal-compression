
extern "C"
__global__ void sumSquareError
(int nBatch,int rbs, int nCoeff, float maxCoeff,
float *DA, float *CA, float *EA, float *SA)
{
	float FLT_MAX = 0x7f800000;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nBatch)
    {
		SA[i] = 0;
		if ((CA[i*2+1] > maxCoeff || CA[i*2+1] < -maxCoeff)
				&& nCoeff == 2) {
			SA[i] = FLT_MAX;
		} else{
			for(int j = 0; j < rbs ; j++){
				float fx = 0.0f;
				for(int k = 0 ; k < nCoeff ; k++){
					fx += DA[i*rbs*nCoeff + rbs*k + j] * CA[i*nCoeff + k];
				}
				float error = EA[i*rbs + j] - fx;
				//EA[i*rbs + j] = error; // store error value
				SA[i] += error*error; // sum square error
			}
		}
    }
}
