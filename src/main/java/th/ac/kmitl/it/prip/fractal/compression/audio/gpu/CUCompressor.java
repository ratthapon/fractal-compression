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
	private static final Logger LOGGER = Logger.getLogger(CUCompressor.class.getName());

	// ptx modules
	private static CUmodule reverseVecModule;
	private static CUmodule setDomainPoolModule;
	private static CUmodule setRangePoolModule;
	private static CUmodule setCoeffPoolModule;
	private static CUmodule limitCoeffModule;
	private static CUmodule sumSquareErrorModule;

	// cuda kernel functions
	private static CUfunction reverseVec;
	private static CUfunction setDomainPoolKernel;
	private static CUfunction setRangePoolKernel;
	private static CUfunction setCoeffPoolKernel;
	private static CUfunction limitCoeffKernel;
	private static CUfunction sumSquareErrorKernel;

	// BLAS handler and stream handler
	private cudaStream_t stream;
	private cublasHandle cublasHandle;

	// Set up the kernel parameters: A pointer to an array
	// of pointers which point to the actual values.
	private Pointer setDomainPoolKernelParams;
	private Pointer setRangePoolKernelParams;
	private Pointer setCoeffPoolKernelParams;
	private Pointer reverseVecKernelParams;
	// coefflimit kernel
	private Pointer limitCoeffKernelParam;
	// sumSquareError kernel
	private Pointer sumSquareErrorKernelParams;

	// device data
	private Pointer deviceData;
	private Pointer deviceDataRev;

	public CUCompressor(float[] inputAudioData, Parameters compressParameters) throws IllegalStateException {
		super(inputAudioData, compressParameters);

		// Initialize the driver and create a context for the first device.
		initCudaDevice();

		// Load the ptx file.
		loadPTXModules();

		// Obtain a function pointer to the kernel function.
		loadCUKernelFunctions();

		// set compressor's input data
		data = new float[inputAudioData.length];
		for (int i = 0; i < inputAudioData.length; i++) {
			data[i] = inputAudioData[i];
		}
	}

	@Override
	public double[][] process() throws IllegalStateException {
		double[][] code;
		try {
			// reset device to free allocate memory
			try {
				cudaDeviceReset();
			} catch (Exception e) {
				LOGGER.log(Level.SEVERE, "cuBlass Error : Can not reset cuda device.");
				throw new IllegalStateException(e);
			}

			final int h2d = cudaMemcpyHostToDevice;
			final int nD = parameters.getND();
			final int nCoeff = ((parameters.getNCoeff() - 1) * nD + 1);
			code = new double[nParts][nCoeff + 3];
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

			int blockSizeX = 1024;
			int gridSizeX = (int) Math.ceil((double) nSamples / blockSizeX);

			// preparing device audio data
			setDeviceAudioData(h2d, blockSizeX, gridSizeX);

			// pre allocate at maximum
			allocateTimeTick = System.nanoTime();
			int maxRBS = parameters.getMaxBlockSize();
			int minRBS = parameters.getMinBlockSize();
			int maxNBatch = ((nSamples - minRBS * parameters.getDomainScale()) / parameters.getDStep()) * 2;
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
			batchCudaArraysAlloc(maxNBatch, dDArrays, dRArrays, dAArrays, dBArrays, dIAArrays, dCArrays, dEArrays,
					dSSEArrays);

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

			// allocate arrays pointers
			batchCudaArraysPointerAlloc(maxNBatch, dDAP, dRAP, dAAP, dBAP, dIAAP, dCAP, dEAP, dSSEAP);
			allocateTime = System.nanoTime() - allocateTimeTick;

			// range block size
			final int rbs = (int) (parameters.getMaxBlockSize());
			int prevRBS = 0; // speed trick, skip redundancy computation

			// set domain location
			int dbStartIdx = 0;
			int dbStopIdx = nSamples - rbs * parameters.getDomainScale() - 1;

			int nBatch = ((dbStopIdx - dbStartIdx + 1) / parameters.getDStep()) * 2;
			gridSizeX = (int) Math.ceil((double) nBatch / blockSizeX);

			// pre setting domain pool
			launchBatchInvGramianMatrix(nBatch, prevRBS, blockSizeX, gridSizeX, dDArrays, dAArrays, dIAArrays,
					dInfoArray, dDAP, dAAP, dIAAP);

			// each range block
			int rbIdx = 0;
			for (int fIdx = 0; fIdx < nParts; fIdx++) {
				// locate range block
				final int bColStart = rbIdx;
				rbIdx = rbIdx + (int) parts[fIdx]; // cumulative for next range
				blockSizeX = 1024;
				gridSizeX = (int) Math.ceil((double) nBatch / blockSizeX);

				setBatchPool(nD, blockSizeX, gridSizeX, dDArrays, dRArrays, dAArrays, dBArrays, dIAArrays, dCArrays,
						dEArrays, dSSEArrays, dR, dDAP, dRAP, dAAP, dBAP, dIAAP, dCAP, dEAP, dSSEAP, rbs, nBatch,
						bColStart);

				// GPU batch operation
				JCuda.cudaStreamSynchronize(stream);
				batcTimeTick = System.nanoTime();

				launchBatchMomentMatrix(nCoeff, dDAP, dRAP, dBAP, rbs, nBatch);

				launchBatchLeastSquare(nCoeff, dBAP, dIAAP, dCAP, nBatch);

				launchBatchLimitCoeff(nCoeff, blockSizeX, gridSizeX, dDArrays, dRArrays, dCArrays, rbs, nBatch);

				launchBatchComputeSSE(nCoeff, blockSizeX, gridSizeX, dDArrays, dCArrays, dEArrays, dSSEArrays, rbs,
						nBatch);

				batcTime = batcTime + (System.nanoTime() - batcTimeTick);

				getMinErrorCode(code, nCoeff, dCArrays, dSSEArrays, rbs, nBatch, fIdx);

				JCuda.cudaStreamSynchronize(stream);

				JCuda.cudaFree(setDomainPoolKernelParams);
				JCuda.cudaFree(setRangePoolKernelParams);
				JCuda.cudaFree(setCoeffPoolKernelParams);
				JCuda.cudaFree(sumSquareErrorKernelParams);

				prevRBS = rbs;

				partProgress.addAndGet(1);
				samplesProgress.addAndGet(rbs);
			}
			completeTime = System.currentTimeMillis();

			freeTimeTick = System.nanoTime();
			batchCudaMemFree(deviceData, deviceDataRev, reverseVecKernelParams, dDArrays, dRArrays, dAArrays, dBArrays,
					dIAArrays, dCArrays, dEArrays, dSSEArrays, dInfoArray, dR, dDAP, dRAP, dAAP, dBAP, dIAAP, dCAP,
					dEAP, dSSEAP);
			freeTime = freeTime + (System.nanoTime() - freeTimeTick);

			totalTime = System.nanoTime() - totalTimeTick;

			cudaDeviceReset();
			isDone = true;
		} catch (Exception e) {
			LOGGER.log(Level.SEVERE, "cuBlass Error : Can not process CUCompression.");
			throw new IllegalStateException(e);
		}
		return code; // code of each file
	}

	private void getMinErrorCode(double[][] code, final int nCoeff, Pointer dCArrays, Pointer dSSEArrays, final int rbs,
			int nBatch, int fIdx) {
		int minSSEIdx = -1;
		float[] sse = new float[] { Float.POSITIVE_INFINITY };
		try {
			// find min sum square error idx
			minSSEIdx = JCublas.cublasIsamin(nBatch, dSSEArrays, 1);
			cudaMemcpy(Pointer.to(sse), dSSEArrays.withByteOffset(Sizeof.FLOAT * (minSSEIdx - 1)), Sizeof.FLOAT,
					cudaMemcpyDeviceToHost);
		} catch (Exception e) {
			LOGGER.log(Level.WARNING, "cuBlass Error : Can not perform find minimum sum square error index.");
			throw new IllegalStateException(e);
		}

		float[] codeBuffer = new float[nCoeff];
		try {
			cudaMemcpy(Pointer.to(codeBuffer), dCArrays.withByteOffset(Sizeof.FLOAT * nCoeff * (minSSEIdx - 1)),
					Sizeof.FLOAT * nCoeff, cudaMemcpyDeviceToHost);
		} catch (Exception e) {
			LOGGER.log(Level.WARNING, "cuBlass Error : Can not gather coefficients result.");
			throw new IllegalStateException(e);
		}

		// store minimum value of self similarity
		code[fIdx] = composeCode(minSSEIdx, code, codeBuffer, nCoeff, rbs, nBatch, sse);
	}

	private void launchBatchComputeSSE(final int nCoeff, int blockSizeX, int gridSizeX, Pointer dDArrays,
			Pointer dCArrays, Pointer dEArrays, Pointer dSSEArrays, final int rbs, int nBatch) {
		try {
			// compute SumSquareErr = sum(E.^2)
			sumSquareErrorKernelParams = Pointer.to(Pointer.to(new int[] { nBatch }), Pointer.to(new int[] { rbs }),
					Pointer.to(new int[] { nCoeff }), Pointer.to(new float[] { parameters.getCoeffLimit() }),
					Pointer.to(dDArrays), Pointer.to(dCArrays), Pointer.to(dEArrays), Pointer.to(dSSEArrays));

			JCudaDriver.cuLaunchKernel(sumSquareErrorKernel, gridSizeX, 1, 1, blockSizeX, 1, 1, 0, null,
					sumSquareErrorKernelParams, null);
			cuCtxSynchronize();
			JCuda.cudaStreamSynchronize(stream);
		} catch (Exception e) {
			LOGGER.log(Level.WARNING, "cuBlass Error : Can not perform sum square error computation.");
			throw new IllegalStateException(e);
		}
	}

	private void launchBatchLimitCoeff(final int nCoeff, int blockSizeX, int gridSizeX, Pointer dDArrays,
			Pointer dRArrays, Pointer dCArrays, final int rbs, int nBatch) {
		try {
			// limit coeff
			if (nCoeff == 2 && parameters.getCoeffLimit() > 0) {
				limitCoeffKernelParam = Pointer.to(Pointer.to(new int[] { nBatch }), Pointer.to(new int[] { rbs }),
						Pointer.to(new float[] { parameters.getCoeffLimit() }), Pointer.to(dDArrays),
						Pointer.to(dRArrays), Pointer.to(dCArrays));

				// Call the kernel function.
				JCudaDriver.cuLaunchKernel(limitCoeffKernel, gridSizeX, 1, 1, blockSizeX, 1, 1, 0, null,
						limitCoeffKernelParam, null);
				cuCtxSynchronize();
				JCuda.cudaStreamSynchronize(stream);
			}
		} catch (Exception e) {
			LOGGER.log(Level.WARNING, "cuBlass Error : Can not perform limit coeff kernel.");
			throw new IllegalStateException(e);
		}
	}

	private void launchBatchLeastSquare(final int nCoeff, Pointer dBAP, Pointer dIAAP, Pointer dCAP, int nBatch) {
		try {
			// compute pinv(X'X)*(X'Y)
			JCublas2.cublasSgemmBatched(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, nCoeff, 1, nCoeff,
					Pointer.to(new float[] { 1.0f }), dIAAP, nCoeff, dBAP, nCoeff, Pointer.to(new float[] { 0.0f }),
					dCAP, nCoeff, nBatch);
			JCuda.cudaStreamSynchronize(stream);
		} catch (Exception e) {
			LOGGER.log(Level.WARNING, "cuBlass Error : Can not perform pinv(X'X)*(X'Y) computation.");
			throw new IllegalStateException(e);
		}
	}

	private void launchBatchMomentMatrix(final int nCoeff, Pointer dDAP, Pointer dRAP, Pointer dBAP, final int rbs,
			int nBatch) {
		try {
			// compute X'Y
			JCublas2.cublasSgemmBatched(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, nCoeff, 1, rbs,
					Pointer.to(new float[] { 1.0f }), dDAP, rbs, dRAP, rbs, Pointer.to(new float[] { 0.0f }), dBAP,
					nCoeff, nBatch);
			JCuda.cudaStreamSynchronize(stream);
		} catch (Exception e) {
			LOGGER.log(Level.WARNING, "cuBlass Error : Can not perform (X'Y) computation.");
			throw new IllegalStateException(e);
		}
	}

	private void setBatchPool(final int nDScale, int blockSizeX, int gridSizeX, Pointer dDArrays, Pointer dRArrays,
			Pointer dAArrays, Pointer dBArrays, Pointer dIAArrays, Pointer dCArrays, Pointer dEArrays,
			Pointer dSSEArrays, Pointer dR, Pointer dDAP, Pointer dRAP, Pointer dAAP, Pointer dBAP, Pointer dIAAP,
			Pointer dCAP, Pointer dEAP, Pointer dSSEAP, final int rbs, int nBatch, final int bColStart) {
		try {
			cudaMemcpy(dR, deviceData.withByteOffset(Sizeof.FLOAT * bColStart), Sizeof.FLOAT * rbs,
					cudaMemcpyDeviceToDevice);

			// set pool kernel parameters
			setSubPoolKernelParams(nBatch, rbs, nDScale, dR, dDArrays, dRArrays, dAArrays, dBArrays, dIAArrays,
					dCArrays, dEArrays, dSSEArrays, dDAP, dRAP, dAAP, dBAP, dIAAP, dCAP, dEAP, dSSEAP);

			// Call the kernel function.
			launchSubPoolKernel(blockSizeX, gridSizeX);

		} catch (Exception e) {
			LOGGER.log(Level.SEVERE, "cuBlass Error : Can not setup range blocks memory.");
			throw new IllegalStateException(e);
		}
	}

	private void launchBatchInvGramianMatrix(int nBatch, int prevRBS, int blockSizeX, int gridSizeX, Pointer dDArrays,
			Pointer dAArrays, Pointer dIAArrays, Pointer dInfoArray, Pointer dDAP, Pointer dAAP, Pointer dIAAP) {

		final int rbs = parameters.getMaxBlockSize();
		final int nCoeff = parameters.getNCoeff();
		final int nD = parameters.getND();
		final int dScale = parameters.getDomainScale();
		final int rScale = parameters.getRangeScale();
		final int expansion = parameters.getExpansion();
		final float regularize = parameters.getRegularize();
		int isAlign = 0;
		if (parameters.isCenAlign() == true) {
			isAlign = 1;
		}
		// launch domain pool setting
		setDomainPoolKernelParams = Pointer.to(Pointer.to(new int[] { nBatch }), Pointer.to(new int[] { rbs }),
				Pointer.to(new int[] { nCoeff }), Pointer.to(new int[] { nD }), Pointer.to(new int[] { dScale }),
				Pointer.to(new int[] { rScale }), Pointer.to(new int[] { expansion }),
				Pointer.to(new int[] { isAlign }), Pointer.to(new float[] { regularize }), Pointer.to(deviceData),
				Pointer.to(deviceDataRev),
				// pointer to arrays data
				Pointer.to(dDArrays), Pointer.to(dAArrays), Pointer.to(dIAArrays),
				// pointer to arrays pointer
				Pointer.to(dDAP), Pointer.to(dAAP), Pointer.to(dIAAP));
		try {
			JCudaDriver.cuLaunchKernel(setDomainPoolKernel, gridSizeX, 1, 1, blockSizeX, 1, 1, 0, null,
					setDomainPoolKernelParams, null);
			cuCtxSynchronize();

		} catch (Exception e) {
			LOGGER.log(Level.WARNING, "cuBlass Error : Can not perform set domains pool.");
			throw new IllegalStateException(e);
		}

		if (prevRBS != rbs) { // skip redundancy computation
			try {
				// compute X'X
				JCublas2.cublasSgemmBatched(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, nCoeff, nCoeff, rbs,
						Pointer.to(new float[] { 1.0f }), dDAP, rbs, dDAP, rbs, Pointer.to(new float[] { 1.0f }), dAAP,
						nCoeff, nBatch);
				JCuda.cudaStreamSynchronize(stream);
			} catch (Exception e) {
				LOGGER.log(Level.WARNING, "cuBlass Error : Can not perform covariance matrix (X'X) computation.");
				throw new IllegalStateException(e);
			}

			try {
				// compute pseudo inverse X'X
				JCublas2.cublasSmatinvBatched(cublasHandle, nCoeff, dAAP, nCoeff, dIAAP, nCoeff, dInfoArray, nBatch);
				JCuda.cudaStreamSynchronize(stream);
			} catch (Exception e) {
				LOGGER.log(Level.WARNING, "cuBlass Error : Can not perform inverse(X'X) computation.");
				throw new IllegalStateException(e);
			}
		}
	}

	private void launchSubPoolKernel(int blockSizeX, int gridSizeX) {
		try {
			JCudaDriver.cuLaunchKernel(setRangePoolKernel, gridSizeX, 1, 1, blockSizeX, 1, 1, 0, null,
					setRangePoolKernelParams, null);
			cuCtxSynchronize();
		} catch (Exception e) {
			LOGGER.log(Level.WARNING, "cuBlass Error : Can not perform set ranges pool.");
			throw new IllegalStateException(e);
		}
		try {
			JCudaDriver.cuLaunchKernel(setCoeffPoolKernel, gridSizeX, 1, 1, blockSizeX, 1, 1, 0, null,
					setCoeffPoolKernelParams, null);
			cuCtxSynchronize();
		} catch (Exception e) {
			LOGGER.log(Level.WARNING, "cuBlass Error : Can not perform set coefficients pool.");
			throw new IllegalStateException(e);
		}
	}

	private void setSubPoolKernelParams(int nBatch, final int rbs, final int nD, Pointer dR, Pointer dDArrays,
			Pointer dRArrays, Pointer dAArrays, Pointer dBArrays, Pointer dIAArrays, Pointer dCArrays, Pointer dEArrays,
			Pointer dSSEArrays, Pointer dDAP, Pointer dRAP, Pointer dAAP, Pointer dBAP, Pointer dIAAP, Pointer dCAP,
			Pointer dEAP, Pointer dSSEAP) {

		setRangePoolKernelParams = Pointer.to(Pointer.to(new int[] { nBatch }), Pointer.to(new int[] { rbs }),
				Pointer.to(new int[] { parameters.getNCoeff() }), Pointer.to(new int[] { nD }),
				Pointer.to(new int[] { parameters.getRangeScale() }), Pointer.to(dR),
				// pointer to arrays data
				Pointer.to(dRArrays), Pointer.to(dBArrays), Pointer.to(dEArrays),
				// pointer to arrays pointer
				Pointer.to(dRAP), Pointer.to(dBAP), Pointer.to(dEAP));

		setCoeffPoolKernelParams = Pointer.to(Pointer.to(new int[] { nBatch }), Pointer.to(new int[] { rbs }),
				Pointer.to(new int[] { parameters.getNCoeff() }), Pointer.to(new int[] { nD }),
				// pointer to arrays data
				Pointer.to(dCArrays), Pointer.to(dSSEArrays),
				// pointer to arrays pointer
				Pointer.to(dCAP), Pointer.to(dSSEAP));
	}

	private void setDeviceAudioData(final int h2d, int blockSizeX, int gridSizeX) {
		// preparing device audio data
		try {
			deviceData = new Pointer();
			deviceDataRev = new Pointer();

			// init audio data power 0
			cudaMalloc(deviceData, Sizeof.FLOAT * nSamples);
			cudaMalloc(deviceDataRev, Sizeof.FLOAT * nSamples);
			cudaMemcpy(deviceData, Pointer.to(data), Sizeof.FLOAT * nSamples, h2d);

			// Set up the kernel parameters: A pointer to an array
			// of pointers which point to the actual values.
			reverseVecKernelParams = Pointer.to(Pointer.to(new int[] { nSamples }), Pointer.to(deviceData),
					Pointer.to(deviceDataRev));

			// Call the kernel function.
			JCudaDriver.cuLaunchKernel(reverseVec, gridSizeX, 1, 1, blockSizeX, 1, 1, 0, null, reverseVecKernelParams,
					null);
			cuCtxSynchronize();
		} catch (Exception e) {
			LOGGER.log(Level.SEVERE, "cuBlass Error : Can not set input data and reverse data.");
			throw new IllegalStateException(e);
		}
	}

	private void initCudaDevice() throws IllegalStateException {
		// Initialize the driver and create a context for the first device.
		try {
			cudaDeviceReset();
			JCuda.setExceptionsEnabled(true);
			cuInit(0);
			CUdevice device = new CUdevice();
			cuDeviceGet(device, 0);
			CUcontext context = new CUcontext();
			cuCtxCreate(context, 0, device);
		} catch (Exception e) {
			LOGGER.log(Level.SEVERE, "cuBlass Error : Can not initialize cuda device or cuda context.");
			throw new IllegalStateException(e);
		}

		// set CuBLAS handler and CuStream
		try {
			cublasHandle = new cublasHandle();
			JCublas2.cublasCreate(cublasHandle);
			stream = new cudaStream_t();
			cudaStreamCreate(stream);
			cublasSetStream(cublasHandle, stream);
		} catch (Exception e) {
			LOGGER.log(Level.SEVERE, "cuBlass Error : Can not set cu stream handler.");
			throw new IllegalStateException(e);
		}
	}

	private void loadCUKernelFunctions() throws IllegalStateException {
		// Obtain a function pointer to the kernel function.
		try {
			reverseVec = new CUfunction();
			JCudaDriver.cuModuleGetFunction(reverseVec, reverseVecModule, "reverseVec");

			setDomainPoolKernel = new CUfunction();
			JCudaDriver.cuModuleGetFunction(setDomainPoolKernel, setDomainPoolModule, "setDomainPoolKernel");

			setRangePoolKernel = new CUfunction();
			JCudaDriver.cuModuleGetFunction(setRangePoolKernel, setRangePoolModule, "setRangePoolKernel");

			setCoeffPoolKernel = new CUfunction();
			JCudaDriver.cuModuleGetFunction(setCoeffPoolKernel, setCoeffPoolModule, "setCoeffPoolKernel");

			limitCoeffKernel = new CUfunction();
			JCudaDriver.cuModuleGetFunction(limitCoeffKernel, limitCoeffModule, "limitCoeff");

			sumSquareErrorKernel = new CUfunction();
			JCudaDriver.cuModuleGetFunction(sumSquareErrorKernel, sumSquareErrorModule, "sumSquareError");
		} catch (Exception e) {
			LOGGER.log(Level.SEVERE, "cuBlass Error : Can not load CU functions.");
			throw new IllegalStateException(e);
		}
	}

	private void loadPTXModules() throws IllegalStateException {
		// Load the ptx file.
		try {
			reverseVecModule = new CUmodule();
			JCudaDriver.cuModuleLoad(reverseVecModule, "classes/reverseVec.ptx");

			setDomainPoolModule = new CUmodule();
			JCudaDriver.cuModuleLoad(setDomainPoolModule, "classes/setDomainPoolKernel.ptx");

			setRangePoolModule = new CUmodule();
			JCudaDriver.cuModuleLoad(setRangePoolModule, "classes/setRangePoolKernel.ptx");

			setCoeffPoolModule = new CUmodule();
			JCudaDriver.cuModuleLoad(setCoeffPoolModule, "classes/setCoeffPoolKernel.ptx");

			limitCoeffModule = new CUmodule();
			JCudaDriver.cuModuleLoad(limitCoeffModule, "classes/limitCoeff.ptx");

			sumSquareErrorModule = new CUmodule();
			JCudaDriver.cuModuleLoad(sumSquareErrorModule, "classes/sumSquareError.ptx");
		} catch (Exception e) {
			LOGGER.log(Level.SEVERE, "cuBlass Error : Can not load ptx files.");
			throw new IllegalStateException(e);
		}
	}

	private double[] composeCode(int minSSEIdx, double[][] code, float[] codeBuffer, final int nCoeff, final int rbs,
			int nBatch, float[] sse) throws IllegalStateException {
		double[] codeChunk = new double[nCoeff + +3];
		for (int i = 0; i < nCoeff; i++) {
			codeChunk[i] = codeBuffer[i];
		}

		// set domain index
		codeChunk[nCoeff] = minSSEIdx;
		if (minSSEIdx <= nBatch / 2) {
			codeChunk[nCoeff] = minSSEIdx;
		} else {
			codeChunk[nCoeff] = -(nBatch - minSSEIdx + 2);
		}
		// set range block size
		codeChunk[nCoeff + 1] = rbs;
		// range block size boundary
		codeChunk[nCoeff + 2] = sse[0];
		return codeChunk;
	}

	private void batchCudaArraysPointerAlloc(int maxNBatch, Pointer dDAP, Pointer dRAP, Pointer dAAP, Pointer dBAP,
			Pointer dIAAP, Pointer dCAP, Pointer dEAP, Pointer dSSEAP) throws IllegalStateException {
		try {
			cudaMalloc(dDAP, maxNBatch * Sizeof.POINTER);
			cudaMalloc(dRAP, maxNBatch * Sizeof.POINTER);
			cudaMalloc(dAAP, maxNBatch * Sizeof.POINTER);
			cudaMalloc(dBAP, maxNBatch * Sizeof.POINTER);
			cudaMalloc(dIAAP, maxNBatch * Sizeof.POINTER);
			cudaMalloc(dCAP, maxNBatch * Sizeof.POINTER);
			cudaMalloc(dEAP, maxNBatch * Sizeof.POINTER);
			cudaMalloc(dSSEAP, maxNBatch * Sizeof.POINTER);
		} catch (Exception e) {
			LOGGER.log(Level.SEVERE, "cuBlass Error : Can not allocate cuda arrays pointers.");
			throw new IllegalStateException(e);
		}
	}

	private void batchCudaArraysAlloc(int maxNBatch, Pointer dDArrays, Pointer dRArrays, Pointer dAArrays,
			Pointer dBArrays, Pointer dIAArrays, Pointer dCArrays, Pointer dEArrays, Pointer dSSEArrays)
			throws IllegalStateException {
		int maxRBS = parameters.getMaxBlockSize();
		int rScale = parameters.getRangeScale();
		int nCoeff = parameters.getNCoeff();
		try {
			cudaMalloc(dDArrays, maxNBatch * Sizeof.FLOAT * maxRBS * rScale * nCoeff);
			cudaMalloc(dRArrays, maxNBatch * Sizeof.FLOAT * maxRBS * rScale);
			cudaMalloc(dAArrays, maxNBatch * Sizeof.FLOAT * nCoeff * nCoeff);
			cudaMalloc(dBArrays, maxNBatch * Sizeof.FLOAT * nCoeff);
			cudaMalloc(dIAArrays, maxNBatch * Sizeof.FLOAT * nCoeff * nCoeff);
			cudaMalloc(dCArrays, maxNBatch * Sizeof.FLOAT * nCoeff);
			cudaMalloc(dEArrays, maxNBatch * Sizeof.FLOAT * maxRBS * rScale);
			cudaMalloc(dSSEArrays, maxNBatch * Sizeof.FLOAT);
		} catch (Exception e) {
			LOGGER.log(Level.SEVERE, "cuBlass Error : Can not allocate cuda arrays.");
			throw new IllegalStateException(e);
		}
	}

	private void batchCudaMemFree(Pointer... cuPointers) throws IllegalStateException {
		try {
			// iterate over pointers list
			for (int i = 0; i < cuPointers.length; i++) {
				JCuda.cudaFree(cuPointers[i]);
			}
		} catch (Exception e) {
			LOGGER.log(Level.SEVERE, "cuBlass Error : Can not free cuda memories.");
			throw new IllegalStateException(e);
		}
	}
}
