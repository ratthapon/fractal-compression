nvcc -ptx cu/reverseVec.cu -o ./target/reverseVec.ptx
nvcc -ptx cu/memSetKernel.cu -o ./target/memSetKernel.ptx
nvcc -ptx cu/limitCoeff.cu -o ./target/limitCoeff.ptx
nvcc -ptx cu/sumSquareError.cu -o ./target/sumSquareError.ptx
pause > complete compile ctx