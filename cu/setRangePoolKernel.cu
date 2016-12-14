extern "C"
__global__ void setRangePoolKernel(
  int nBatch,int rbs,int nDegree,int nD,int rScale,

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
    int nCoeff = ((nDegree - 1) * nD + 1);

    // pointing section
    RP[taskIdx] = &RA[taskIdx * rbs * rScale];
    BP[taskIdx] = &BA[taskIdx * nCoeff];
    EP[taskIdx] = &EA[taskIdx * rbs];

    // initialize range and error arrays
    int raOffset = (taskIdx * rbs * rScale);
    for(int j = 0; j < rbs * rScale; j++){
      RA[raOffset + j] = R[j / rScale];
      EA[raOffset + j] = R[j / rScale];
    }
  }
}
