extern "C"
__global__ void setDomainPoolKernel(
  int nBatch,int rbs,int nDegree,int nD,int dScale, float regularize,

  float *data,float *dataRev, // array of data and reverse data
  // arrays pointer
  float *DA,
  float *AA,
  float *IA,
  // pointer of array of pointer to pointer of array in arrays, nevermind i just stun you.
  // p(i) = data(i + size(data))
  float **DP,
  float **AP,
  float **IP
)
{
  int taskIdx = blockIdx.x * blockDim.x + threadIdx.x;
  if (taskIdx < nBatch){
    // initialize domain arrays
    // array structure
    // DA = rbs-rows , 1 + D_1^1 + D_2^1 + D_3^1 + ... + D_ds^2
    // nCoeff is power of Domains. start from power 0
    // nDegree determine the number of degree (maximum degree)
    // nD determine the number of domain blocks
    // dScale determing the scale of domain size compare to rbs
    // dpOffset determine the number of bytes padded for each Array

    int nCoeff = ((nDegree - 1) * nD + 1);

    // pointing array of pointers to array
    const int daOffset = taskIdx * rbs * nCoeff;
    const int aaOffset = taskIdx * nCoeff * nCoeff;
    const int iaOffset = taskIdx * nCoeff * nCoeff;

    DP[taskIdx] = &DA[daOffset];
    AP[taskIdx] = &AA[aaOffset];
    IP[taskIdx] = &IA[iaOffset];

    // initialize covariance matrix with regularization
    for(int i = 0; i < nCoeff * nCoeff; i++){
      AA[aaOffset + i] = 0.0f;
    }
    for(int i = 0; i < nCoeff * nCoeff; i+= nCoeff+1){
      // set diagonal to regularization parameter
      AA[aaOffset + i] = regularize * regularize;
    }

    // initialize first column covariance matrix
    for(int i = 0; i < rbs; i++){
      DA[daOffset + i] = 1.0f; // power 0
    }

    int dIdx = taskIdx % (nBatch/2);

    // for each block number dn
    for(int dn = 1; dn <= nD; dn++){
      // set reference domain block
      int dSize = rbs * dScale;
      int dnIdx = dIdx + (dn - 1) * dSize; // * domian location factor

      int rowDA = rbs*dn; // number of row of DA

      // initialize column dn-th index
      for(int i = 0; i < rbs; i++){
        DA[daOffset + rowDA + i] = 0.0f; // power 1
      }

      // construct DA from domain blocks at power 1
      // copy elements
      for(int i = 0; i < rbs; i++){
        if(taskIdx < (nBatch/2)){
          DA[daOffset + rowDA + i] =
          DA[daOffset + rowDA + i] + data[dnIdx + i*dScale];
        }else{ // gen reverse domain
          DA[daOffset + rowDA + i] =
          DA[daOffset + rowDA + i] + dataRev[dnIdx + i*dScale];
        }
      }

      // handling if domain blocks are larger than rbs (by downsample)
      for(int ds = 1; ds < dScale; ds++){
        // vec sumation
        for(int i = 0; i < rbs; i++){
          if(taskIdx < (nBatch/2)){
            DA[daOffset + rowDA + i] =
            DA[daOffset  + rowDA + i] + data[dnIdx + ds + i*dScale];
          }else{ // gen reverse domain
            DA[daOffset + rowDA + i] =
            DA[daOffset  + rowDA + i] + dataRev[dnIdx + ds + i*dScale];
          }
        }
      }

      // vec scalig after resample
      for(int i = 0; i < rbs; i++){
        DA[daOffset + rowDA + i] = DA[daOffset + rowDA + i]/dScale;
      }

      // calculate next degree
      // for(int deg = 2; deg <= nDegree; deg++){
      //   int degPad = rowDA * nD * (deg - 2);
      //   int nextDegPad = rowDA * nD * (deg - 1);
      //   for(int i = 0; i < rbs * nD; i++){
      //     // power n>=2
      //     // D^n = D^1 * D^(n-1)
      //     DA[daOffset + nextDegPad + rbs + i] =
      //     DA[daOffset + rbs + i] * DA[daOffset + degPad + rbs + i] ;
      //   }
      // }
    }
  }
}
