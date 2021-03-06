extern "C"
__global__ void setDomainPoolKernel(
  int nBatch, int rbs, int nDegree, int nD, int dScale, int rScale,
  int expansion, int isCenAlign, float regularize,

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
    // nDegree determine the number of degree (maximum degree + 1)
    // nD determine the number of domain blocks
    // dScale determing the scale of domain size compare to rbs
    // dpOffset determine the number of bytes padded for each Array

    int nCoeff = ((nDegree - 1) * nD + 1);

    // pointing array of pointers to array
    const int daOffset = taskIdx * rbs * rScale * nCoeff;
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
    for(int i = 0; i < rbs * rScale; i++){
      DA[daOffset + i] = 1.0f; // power 0
    }

    int dIdx = taskIdx % (nBatch/2);

    // for each block number dn
    for(int dn = 1; dn <= nD; dn++){
      // set reference domain block
      // compute sumScale
      int sumScale = 0;
      for(int k = 1; k <= nD && k < dn; k++){
        sumScale += (int) powf( (float) dScale, (float) (1 + expansion * (k - 1))) ;
      }
      int dnScale = (int) powf( (float) dScale, (float) (1 + expansion * (dn - 1)));

      //int dnIdx = dIdx + rbs * sumScale; // * domian location factor
      int dnIdx = dIdx;
      if( isCenAlign == 0 ){
        // if left aligned
        dnIdx = dIdx + rbs * sumScale + 1;
      } else {
        // if center aligned
        dnIdx = dIdx + rbs/2 * (1 - dnScale) + 1;
      }

      int padDA = rbs * rScale * dn; // number of row of DA

      // initialize column dn-th index
      for(int i = 0; i < rbs * rScale; i++){
        DA[daOffset + padDA + i] = 0.0f; // power 1
      }

      // construct DA from domain blocks at power 1
      // copy elements
      for(int i = 0; i < rbs * rScale; i++){
        int datIdx = dnIdx + i*dnScale/rScale;
        if(datIdx >=0 && datIdx < (nBatch/2)){
          if(taskIdx < (nBatch/2)){
            DA[daOffset + padDA + i] =
            DA[daOffset + padDA + i] + data[datIdx];
          }else{ // gen reverse domain
            DA[daOffset + padDA + i] =
            DA[daOffset + padDA + i] + dataRev[datIdx];
          }
        }
      }

      // // handling if domain blocks are larger than rbs (by downsample)
      for(int ds = 1; ds < dnScale / rScale; ds++){
        // vec sumation
        for(int i = 0; i < rbs * rScale; i++){
          int datIdx = dnIdx + ds + i*dnScale/rScale;
          if(datIdx >=0 && datIdx < (nBatch/2)){
            if(taskIdx < (nBatch/2)){
              DA[daOffset + padDA + i] =
              DA[daOffset  + padDA + i] + data[datIdx];
            }else{ // gen reverse domain
              DA[daOffset + padDA + i] =
              DA[daOffset  + padDA + i] + dataRev[datIdx];
            }
          }
        }
      }

      // vec scalig after resample
      for(int i = 0; i < rbs * rScale; i++){
        DA[daOffset + padDA + i] = DA[daOffset + padDA + i] / (dnScale/rScale);
      }

      // calculate next degree
      for(int deg = 2; deg <= nDegree - 1; deg++){
        int degPad = rbs * nD * (deg - 2) + rbs * dn;
        int nextDegPad = rbs * nD * (deg - 1) + rbs * dn;
        for(int i = 0; i < rbs * rScale; i++){
          // power n>=2
          // D^n = D^1 * D^(n-1)
          DA[daOffset + nextDegPad + i] =
          DA[daOffset + rbs*dn + i] * DA[daOffset + degPad + i] ;
        }
      }
    }
  }
}
