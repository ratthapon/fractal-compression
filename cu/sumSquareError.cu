
extern "C"
__global__ void sumSquareError
(int nBatch, int rbs, int rScale, int nCoeff,
	float *DA, float *CA, float *EA, float *SA)
	{
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i < nBatch)
		{
			const int daOffset = i * rbs * rScale * nCoeff;
			const int caOffset = i * nCoeff;
			const int eaOffset = i * rbs * rScale;

			SA[i] = 0;
			for(int j = 0; j < rbs * rScale ; j++){
				float fx = 0.0f;
				for(int k = 0 ; k < nCoeff ; k++){
					fx += DA[daOffset + rbs * rScale * k + j] * CA[caOffset + k];
				}
				float error = EA[eaOffset + j] - fx;
				SA[i] += error*error; // sum square error
			}
		}
	}
