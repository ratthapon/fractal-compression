package th.ac.kmitl.it.prip.fractal.compression.audio.gpu;

import static jcuda.driver.JCudaDriver.cuCtxCreate;
import static jcuda.driver.JCudaDriver.cuCtxSynchronize;
import static jcuda.driver.JCudaDriver.cuDeviceGet;
import static jcuda.driver.JCudaDriver.cuInit;
import static jcuda.jcublas.JCublas2.cublasSetStream;
import static jcuda.jcublas.cublasOperation.CUBLAS_OP_N;
import static jcuda.jcublas.cublasOperation.CUBLAS_OP_T;
import static jcuda.runtime.JCuda.cudaDeviceReset;
import static jcuda.runtime.JCuda.cudaMalloc;
import static jcuda.runtime.JCuda.cudaMemcpy;
import static jcuda.runtime.JCuda.cudaStreamCreate;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyDeviceToDevice;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyHostToDevice;

import java.util.logging.Level;
import java.util.logging.Logger;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUcontext;
import jcuda.driver.CUdevice;
import jcuda.driver.CUfunction;
import jcuda.driver.CUmodule;
import jcuda.driver.JCudaDriver;
import jcuda.jcublas.JCublas;
import jcuda.jcublas.JCublas2;
import jcuda.jcublas.cublasHandle;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaStream_t;
import th.ac.kmitl.it.prip.fractal.Parameters;
import th.ac.kmitl.it.prip.fractal.compression.audio.Compressor;

public class CUCompressor extends Compressor {
    private static final Logger LOGGER = Logger.getLogger(CUCompressor.class
            .getName());

    public CUCompressor(float[] inputAudioData, Parameters compressParameters) {
        super(inputAudioData, compressParameters);
        data = new float[inputAudioData.length];
        for (int i = 0; i < inputAudioData.length; i++) {
            data[i] = inputAudioData[i];
        }
    }

    @Override
    public double[][] process() {
        cudaDeviceReset();

        JCuda.setExceptionsEnabled(true);

        // Initialize the driver and create a context for the first device.
        cuInit(0);
        CUdevice device = new CUdevice();
        cuDeviceGet(device, 0);
        CUcontext context = new CUcontext();
        cuCtxCreate(context, 0, device);

        // Load the ptx file.
        CUmodule reverseVecModule = new CUmodule();
        JCudaDriver.cuModuleLoad(reverseVecModule, "reverseVec.ptx");

        CUmodule memSetModule = new CUmodule();
        JCudaDriver.cuModuleLoad(memSetModule, "memSetKernel.ptx");

        CUmodule limitCoeffModule = new CUmodule();
        JCudaDriver.cuModuleLoad(limitCoeffModule, "limitCoeff.ptx");

        CUmodule sumSquareErrorModule = new CUmodule();
        JCudaDriver.cuModuleLoad(sumSquareErrorModule, "sumSquareError.ptx");

        // Obtain a function pointer to the kernel function.
        CUfunction reverseVec = new CUfunction();
        JCudaDriver.cuModuleGetFunction(reverseVec, reverseVecModule,
                "reverseVec");

        CUfunction memSetKernel = new CUfunction();
        JCudaDriver.cuModuleGetFunction(memSetKernel, memSetModule,
                "memSetKernel");

        CUfunction limitCoeffKernel = new CUfunction();
        JCudaDriver.cuModuleGetFunction(limitCoeffKernel, limitCoeffModule,
                "limitCoeff");

        CUfunction sumSquareErrorKernel = new CUfunction();
        JCudaDriver.cuModuleGetFunction(sumSquareErrorKernel,
                sumSquareErrorModule, "sumSquareError");

        final int h2d = cudaMemcpyHostToDevice;
        final int nCoeff = parameters.getNCoeff();
        double[][] code = new double[nParts][nCoeff + 3];
        if (nCoeff <= 1) {
            String logMsg = "Invalid n coefficients. It should greater than 1.";
            LOGGER.log(Level.WARNING, logMsg);
            throw new IllegalArgumentException(logMsg);
        }

        initTime = System.currentTimeMillis();

        // timing
        long totalTimeTick = System.nanoTime();
        long allocateTimeTick;
        long batcTimeTick;
        long freeTimeTick;

        @SuppressWarnings("unused")
        long totalTime = 0;
        @SuppressWarnings("unused")
        long allocateTime = 0;
        long batcTime = 0;
        long freeTime = 0;

        // preparing device audio data
        Pointer deviceData = new Pointer();
        Pointer deviceDataRev = new Pointer();

        // init audio data power 0
        cudaMalloc(deviceData, Sizeof.FLOAT * nSamples);
        cudaMalloc(deviceDataRev, Sizeof.FLOAT * nSamples);
        cudaMemcpy(deviceData, Pointer.to(data), Sizeof.FLOAT * nSamples, h2d);

        // Set up the kernel parameters: A pointer to an array
        // of pointers which point to the actual values.
        Pointer reverseVecKernelParams = Pointer.to(
                Pointer.to(new int[]{nSamples}), Pointer.to(deviceData),
                Pointer.to(deviceDataRev));

        // Call the kernel function.
        int blockSizeX = 1024;
        int gridSizeX = (int) Math.ceil((double) nSamples / blockSizeX);
        JCudaDriver.cuLaunchKernel(reverseVec, gridSizeX, 1, 1, blockSizeX, 1,
                1, 0, null, reverseVecKernelParams, null);
        cuCtxSynchronize();

        cublasHandle cublasHandle = new cublasHandle();
        JCublas2.cublasCreate(cublasHandle);
        cudaStream_t stream = new cudaStream_t();
        cudaStreamCreate(stream);
        cublasSetStream(cublasHandle, stream);

        // pre allocate at maximum
        allocateTimeTick = System.nanoTime();
        int maxRBS = parameters.getMaxBlockSize();
        int minRBS = parameters.getMinBlockSize();
        int maxNBatch = ((nSamples - minRBS * parameters.getDomainScale()) / parameters
                .getDStep()) * 2;
        // arrays mem pointer
        Pointer dDArrays = new Pointer(); // domain
        Pointer dRArrays = new Pointer(); // range
        Pointer dAArrays = new Pointer(); // A
        Pointer dBArrays = new Pointer(); // B
        Pointer dIAArrays = new Pointer(); // inverse A
        Pointer dCArrays = new Pointer(); // coeff
        Pointer dEArrays = new Pointer(); // error
        Pointer dSSEArrays = new Pointer(); // sum square error
        Pointer dInfoArray = new Pointer(); // info array
        Pointer dR = new Pointer(); // range

        // large allocation
        cudaMalloc(dDArrays, maxNBatch * Sizeof.FLOAT * maxRBS * nCoeff);
        cudaMalloc(dRArrays, maxNBatch * Sizeof.FLOAT * maxRBS);
        cudaMalloc(dAArrays, maxNBatch * Sizeof.FLOAT * nCoeff * nCoeff);
        cudaMalloc(dBArrays, maxNBatch * Sizeof.FLOAT * nCoeff);
        cudaMalloc(dIAArrays, maxNBatch * Sizeof.FLOAT * nCoeff * nCoeff);
        cudaMalloc(dCArrays, maxNBatch * Sizeof.FLOAT * nCoeff);
        cudaMalloc(dEArrays, maxNBatch * Sizeof.FLOAT * maxRBS);
        cudaMalloc(dSSEArrays, maxNBatch * Sizeof.FLOAT);

        cudaMalloc(dInfoArray, maxNBatch * Sizeof.INT);
        cudaMalloc(dR, Sizeof.FLOAT * maxRBS);

        Pointer dDAP = new Pointer();
        Pointer dRAP = new Pointer();
        Pointer dAAP = new Pointer();
        Pointer dBAP = new Pointer();
        Pointer dIAAP = new Pointer();
        Pointer dCAP = new Pointer();
        Pointer dEAP = new Pointer();
        Pointer dSSEAP = new Pointer();

        cudaMalloc(dDAP, maxNBatch * Sizeof.POINTER);
        cudaMalloc(dRAP, maxNBatch * Sizeof.POINTER);
        cudaMalloc(dAAP, maxNBatch * Sizeof.POINTER);
        cudaMalloc(dBAP, maxNBatch * Sizeof.POINTER);
        cudaMalloc(dIAAP, maxNBatch * Sizeof.POINTER);
        cudaMalloc(dCAP, maxNBatch * Sizeof.POINTER);
        cudaMalloc(dEAP, maxNBatch * Sizeof.POINTER);
        cudaMalloc(dSSEAP, maxNBatch * Sizeof.POINTER);
        allocateTime = System.nanoTime() - allocateTimeTick;

        // each range block
        int rbIdx = 0;
        int prevRBS = 0; // speed trick, skip redundancy computation
        for (int fIdx = 0; fIdx < nParts; fIdx++) {
            // locate range block
            final int bColStart = rbIdx;
            final int rbs = (int) (parts[fIdx]); // range block size
            rbIdx = rbIdx + (int) parts[fIdx]; // cumulative for next range

            // set domain location
            int dbStartIdx = 0;
            int dbStopIdx = nSamples - rbs * parameters.getDomainScale() - 1;

            int nBatch = ((dbStopIdx - dbStartIdx + 1) / parameters.getDStep()) * 2;

            cudaMemcpy(dR, deviceData.withByteOffset(Sizeof.FLOAT * bColStart),
                    Sizeof.FLOAT * rbs, cudaMemcpyDeviceToDevice);

            // Set up the kernel parameters: A pointer to an array
            // of pointers which point to the actual values.
            Pointer memSetKernelParams = Pointer.to(
                    Pointer.to(new int[]{nBatch}),
                    Pointer.to(new int[]{rbs}),
                    Pointer.to(new int[]{nCoeff}),
                    Pointer.to(new int[]{dbStopIdx}),
                    Pointer.to(new int[]{parameters.getDomainScale()}),
                    Pointer.to(new float[]{parameters.getRegularize()}),
                    Pointer.to(deviceData),
                    Pointer.to(deviceDataRev),
                    Pointer.to(dR),
                    // pointer to arrays data
                    Pointer.to(dDArrays), Pointer.to(dRArrays),
                    Pointer.to(dAArrays), Pointer.to(dBArrays),
                    Pointer.to(dIAArrays), Pointer.to(dCArrays),
                    Pointer.to(dEArrays),
                    Pointer.to(dSSEArrays),
                    // pointer to arrays pointer
                    Pointer.to(dDAP), Pointer.to(dRAP), Pointer.to(dAAP),
                    Pointer.to(dBAP), Pointer.to(dIAAP), Pointer.to(dCAP),
                    Pointer.to(dEAP), Pointer.to(dSSEAP));

            // Call the kernel function.
            blockSizeX = 1024;
            gridSizeX = (int) Math.ceil((double) nBatch / blockSizeX);
            JCudaDriver.cuLaunchKernel(memSetKernel, gridSizeX, 1, 1,
                    blockSizeX, 1, 1, 0, null, memSetKernelParams, null);
            cuCtxSynchronize();

			// coefflimit kernel
			Pointer limitCoeffKernelParam = Pointer.to(
					Pointer.to(new int[] { nBatch }),
					Pointer.to(new int[] { rbs }),
					Pointer.to(new float[] { parameters.getCoeffLimit() }),
					Pointer.to(dDArrays), Pointer.to(dRArrays),
					Pointer.to(dCArrays));

            // sumSquareError kernel
            Pointer sumSquareErrorKernelParams = Pointer.to(
                    Pointer.to(new int[]{nBatch}),
                    Pointer.to(new int[]{rbs}),
                    Pointer.to(new int[]{nCoeff}),
                    Pointer.to(new float[]{parameters.getCoeffLimit()}),
                    Pointer.to(dDArrays),
                    Pointer.to(dCArrays), Pointer.to(dEArrays),
                    Pointer.to(dSSEArrays));

            // GPU batch operation
            JCuda.cudaStreamSynchronize(stream);
            batcTimeTick = System.nanoTime();
            try {

                if (prevRBS != rbs) { // skip redundancy computation

                    // compute X'X
                    JCublas2.cublasSgemmBatched(cublasHandle, CUBLAS_OP_T,
                            CUBLAS_OP_N, nCoeff, nCoeff, rbs,
                            Pointer.to(new float[]{1.0f}), dDAP, rbs, dDAP,
                            rbs, Pointer.to(new float[]{1.0f}), dAAP,
                            nCoeff, nBatch);
                    JCuda.cudaStreamSynchronize(stream);

                    // compute pseudo inverse X'X
                    JCublas2.cublasSmatinvBatched(cublasHandle, nCoeff, dAAP,
                            nCoeff, dIAAP, nCoeff, dInfoArray, nBatch);
                    JCuda.cudaStreamSynchronize(stream);

                }

                // compute X'Y
                JCublas2.cublasSgemmBatched(cublasHandle, CUBLAS_OP_T,
                        CUBLAS_OP_N, nCoeff, 1, rbs,
                        Pointer.to(new float[]{1.0f}), dDAP, rbs, dRAP, rbs,
                        Pointer.to(new float[]{0.0f}), dBAP, nCoeff, nBatch);
                JCuda.cudaStreamSynchronize(stream);

                // compute pinv(X'X)*(X'Y)
                JCublas2.cublasSgemmBatched(cublasHandle, CUBLAS_OP_N,
                        CUBLAS_OP_N, nCoeff, 1, nCoeff,
                        Pointer.to(new float[]{1.0f}), dIAAP, nCoeff, dBAP,
                        nCoeff, Pointer.to(new float[]{0.0f}), dCAP, nCoeff,
                        nBatch);
                JCuda.cudaStreamSynchronize(stream);

                // limit coeff
                // Call the kernel function.
                blockSizeX = 1024;
                gridSizeX = (int) Math.ceil((double) nBatch / blockSizeX);
                JCudaDriver.cuLaunchKernel(limitCoeffKernel, gridSizeX, 1, 1,
                        blockSizeX, 1, 1, 0, null, limitCoeffKernelParam, null);
                cuCtxSynchronize();
                JCuda.cudaStreamSynchronize(stream);

                // compute SumSquareErr = sum(E.^2)
                gridSizeX = (int) Math.ceil((double) nBatch / blockSizeX);
                JCudaDriver.cuLaunchKernel(sumSquareErrorKernel, gridSizeX, 1,
                        1, blockSizeX, 1, 1, 0, null,
                        sumSquareErrorKernelParams, null);
                cuCtxSynchronize();
                JCuda.cudaStreamSynchronize(stream);

            } catch (Exception e) {
                LOGGER.log(Level.WARNING, "cuBlass Error.");
                throw new IllegalStateException(e);
            }
            // find min sum square error idx
            int minSSEIdx = -1;
            minSSEIdx = JCublas.cublasIsamin(nBatch, dSSEArrays, 1);

            batcTime = batcTime + (System.nanoTime() - batcTimeTick);

            float[] codeBuffer = new float[nCoeff];
            cudaMemcpy(
                    Pointer.to(codeBuffer),
                    dCArrays.withByteOffset(Sizeof.FLOAT * nCoeff
                            * (minSSEIdx - 1)), Sizeof.FLOAT * nCoeff,
                    cudaMemcpyDeviceToHost);

            // store minimum value of self similarity
            for (int i = 0; i < nCoeff; i++) {
                code[fIdx][i] = codeBuffer[i];
            }

            // set domain index
            code[fIdx][nCoeff] = minSSEIdx;
            if (minSSEIdx <= nBatch / 2) {
                code[fIdx][nCoeff] = minSSEIdx;
            } else {
                code[fIdx][nCoeff] = -(nBatch - minSSEIdx + 2);
            }
            // set range block size
            code[fIdx][nCoeff + 1] = rbs;
            // range block size boundary
            code[fIdx][nCoeff + 2] = fIdx;

            JCuda.cudaStreamSynchronize(stream);

            JCuda.cudaFree(memSetKernelParams);
            JCuda.cudaFree(sumSquareErrorKernelParams);

            prevRBS = rbs;

            partProgress.addAndGet(1);
            samplesProgress.addAndGet(rbs);
        }
        completeTime = System.currentTimeMillis();
        freeTimeTick = System.nanoTime();
        JCuda.cudaFree(dR);

        JCuda.cudaFree(reverseVecKernelParams);
        JCuda.cudaFree(deviceData);
        JCuda.cudaFree(deviceDataRev);

        JCuda.cudaFree(dDArrays);
        JCuda.cudaFree(dRArrays);
        JCuda.cudaFree(dAArrays);
        JCuda.cudaFree(dBArrays);
        JCuda.cudaFree(dIAArrays);
        JCuda.cudaFree(dCArrays);
        JCuda.cudaFree(dEArrays);
        JCuda.cudaFree(dSSEArrays);
        JCuda.cudaFree(dInfoArray);

        JCuda.cudaFree(dDAP);
        JCuda.cudaFree(dRAP);
        JCuda.cudaFree(dAAP);
        JCuda.cudaFree(dBAP);
        JCuda.cudaFree(dIAAP);
        JCuda.cudaFree(dCAP);
        JCuda.cudaFree(dEAP);
        JCuda.cudaFree(dSSEAP);
        freeTime = freeTime + (System.nanoTime() - freeTimeTick);
        totalTime = System.nanoTime() - totalTimeTick;

        cudaDeviceReset();
        isDone = true;
        return code; // code of each file
    }
}
