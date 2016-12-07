
extern "C"
__global__ void limitCoeff
(int nBatch,int rbs, float maxCoeff,
	float *DA, float *RA, float *CA)
	{
		int taskIdx = blockIdx.x * blockDim.x + threadIdx.x;
		if (taskIdx < nBatch)
		{
			int i = taskIdx; // % (nBatch / 2);

			// support only 2 coefficients
			int nCoeff = 2;

			// locate arrays pointer
			int daOffset = i * rbs * nCoeff;
			int raOffset = i * rbs;
			int caOffset = i * nCoeff; // support only 2 coefficients

			// check if need to refit coefficients
			if (CA[caOffset + 1] > maxCoeff || CA[caOffset + 1] < -maxCoeff) {
				// set to maximum or minimum depend on sign
				if (CA[caOffset + 1] > maxCoeff) {
					CA[caOffset + 1] = maxCoeff;
				} else if (CA[caOffset + 1] < -maxCoeff) {
					CA[caOffset + 1] = -maxCoeff;
				}

				// refit coefficients
				float suma = 0.0f; // power 1 coeff
				float sumb = 0.0f; // power 0 coeff
				for(int j = 0; j<rbs ;j++){
					suma += DA[daOffset + rbs + j];
					sumb += RA[raOffset + j];
				}
				CA[caOffset] = (sumb - CA[caOffset + 1] * suma) / rbs;
			}
		}
	}
