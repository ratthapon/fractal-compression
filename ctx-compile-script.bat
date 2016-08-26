nvcc -ptx cu/reverseVec.cu -o ./src/main/resources/reverseVec.ptx
nvcc -ptx cu/memSetKernel.cu -o ./src/main/resources/memSetKernel.ptx
nvcc -ptx cu/limitCoeff.cu -o ./src/main/resources/limitCoeff.ptx
nvcc -ptx cu/sumSquareError.cu -o ./src/main/resources/sumSquareError.ptx
pause > complete compile ctx