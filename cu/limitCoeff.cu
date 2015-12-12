
extern "C"
__global__ void limitCoeff
(int nBatch,int rbs, float maxCoeff,
float *DA, float *RA, float *CA)
{
	int nCoeff = 2;
    int taskIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (taskIdx < nBatch)
    {
		int i = taskIdx; // % (nBatch / 2);
		if ((CA[i*2+1] > maxCoeff || CA[i*2+1] < -maxCoeff)
				&& nCoeff == 2) {
			if (CA[i*2+1] > maxCoeff) {
				CA[i*2+1] = maxCoeff;
			} else if (CA[i*2+1] < -maxCoeff) {
				CA[i*2+1] = -maxCoeff;
			}
			float suma = 0.0f;
			float sumb = 0.0f;
			for(int j = 0; j<rbs ;j++){
				suma += DA[i * rbs * nCoeff + j + rbs];
				sumb += RA[i * rbs + j];
			}
			CA[i*2] = (sumb - CA[i*2+1] * suma) / rbs;

		}
    }
}
