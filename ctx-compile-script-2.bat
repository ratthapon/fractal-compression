nvcc -ptx cu/reverseVec.cu -o ./src/main/resources/reverseVec.ptx
nvcc -ptx cu/setDomainPoolKernel.cu -o ./src/main/resources/setDomainPoolKernel.ptx
nvcc -ptx cu/setRangePoolKernel.cu -o ./src/main/resources/setRangePoolKernel.ptx
nvcc -ptx cu/setCoeffPoolKernel.cu -o ./src/main/resources/setCoeffPoolKernel.ptx
nvcc -ptx cu/limitCoeff.cu -o ./src/main/resources/limitCoeff.ptx
nvcc -ptx cu/sumSquareError.cu -o ./src/main/resources/sumSquareError.ptx
pause > complete compile ctx