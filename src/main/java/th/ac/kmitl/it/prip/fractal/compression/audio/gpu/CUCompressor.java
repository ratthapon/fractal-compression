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
	private static CUmodule memSetModule;
	private static CUmodule limitCoeffModule;
	private static CUmodule sumSquareErrorModule;

	// cuda kernel functions
	private static CUfunction reverseVec;
	private static CUfunction memSetKernel;
	private static CUfunction limitCoeffKernel;
	private static CUfunction sumSquareErrorKernel;

	// Set up the kernel parameters: A pointer to an array
	// of pointers which point to the actual values.
	Pointer memSetKernelParams;
	Pointer reverseVecKernelParams;
	// coefflimit kernel
	Pointer limitCoeffKernelParam;
	// sumSquareError kernel
	Pointer sumSquareErrorKernelParams;

	// device data
	Pointer deviceData;
	Pointer deviceDataRev;

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
			final int nDScale = parameters.getNDScale();
			final int nCoeff = ((parameters.getNCoeff() - 1) * nDScale + 1);
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

			cublasHandle cublasHandle;
			cudaStream_t stream;
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
			batchCudaArraysAlloc(nCoeff, maxRBS, maxNBatch, dDArrays, dRArrays, dAArrays, dBArrays, dIAArrays, dCArrays,
					dEArrays, dSSEArrays);

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

				try {
					cudaMemcpy(dR, deviceData.withByteOffset(Sizeof.FLOAT * bColStart), Sizeof.FLOAT * rbs,
							cudaMemcpyDeviceToDevice);

					memSetKernelParams = Pointer.to(Pointer.to(new int[] { nBatch }), Pointer.to(new int[] { rbs }),
							Pointer.to(new int[] { parameters.getNCoeff() }), Pointer.to(new int[] { nDScale }),
							Pointer.to(new int[] { dbStopIdx }), Pointer.to(new int[] { parameters.getDomainScale() }),
							Pointer.to(new float[] { parameters.getRegularize() }), Pointer.to(deviceData),
							Pointer.to(deviceDataRev), Pointer.to(dR),
							// pointer to arrays data
							Pointer.to(dDArrays), Pointer.to(dRArrays), Pointer.to(dAArrays), Pointer.to(dBArrays),
							Pointer.to(dIAArrays), Pointer.to(dCArrays), Pointer.to(dEArrays), Pointer.to(dSSEArrays),
							// pointer to arrays pointer
							Pointer.to(dDAP), Pointer.to(dRAP), Pointer.to(dAAP), Pointer.to(dBAP), Pointer.to(dIAAP),
							Pointer.to(dCAP), Pointer.to(dEAP), Pointer.to(dSSEAP));

					// Call the kernel function.
					blockSizeX = 1024;
					gridSizeX = (int) Math.ceil((double) nBatch / blockSizeX);
					JCudaDriver.cuLaunchKernel(memSetKernel, gridSizeX, 1, 1, blockSizeX, 1, 1, 0, null,
							memSetKernelParams, null);
					cuCtxSynchronize();

					limitCoeffKernelParam = Pointer.to(Pointer.to(new int[] { nBatch }), Pointer.to(new int[] { rbs }),
							Pointer.to(new float[] { parameters.getCoeffLimit() }), Pointer.to(dDArrays),
							Pointer.to(dRArrays), Pointer.to(dCArrays));

					sumSquareErrorKernelParams = Pointer.to(Pointer.to(new int[] { nBatch }),
							Pointer.to(new int[] { rbs }), Pointer.to(new int[] { nCoeff }),
							Pointer.to(new float[] { parameters.getCoeffLimit() }), Pointer.to(dDArrays),
							Pointer.to(dCArrays), Pointer.to(dEArrays), Pointer.to(dSSEArrays));
				} catch (Exception e) {
					LOGGER.log(Level.SEVERE, "cuBlass Error : Can not setup range blocks memory.");
					throw new IllegalStateException(e);
				}

				// GPU batch operation
				JCuda.cudaStreamSynchronize(stream);
				batcTimeTick = System.nanoTime();

				if (prevRBS != rbs) { // skip redundancy computation
					try {
						// compute X'X
						JCublas2.cublasSgemmBatched(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, nCoeff, nCoeff, rbs,
								Pointer.to(new float[] { 1.0f }), dDAP, rbs, dDAP, rbs,
								Pointer.to(new float[] { 1.0f }), dAAP, nCoeff, nBatch);
						JCuda.cudaStreamSynchronize(stream);
					} catch (Exception e) {
						LOGGER.log(Level.WARNING,
								"cuBlass Error : Can not perform covariance matrix (X'X) computation.");
						throw new IllegalStateException(e);
					}

					try {
						// compute pseudo inverse X'X
						JCublas2.cublasSmatinvBatched(cublasHandle, nCoeff, dAAP, nCoeff, dIAAP, nCoeff, dInfoArray,
								nBatch);
						JCuda.cudaStreamSynchronize(stream);
					} catch (Exception e) {
						LOGGER.log(Level.WARNING, "cuBlass Error : Can not perform inverse(X'X) computation.");
						throw new IllegalStateException(e);
					}

				}

				try {
					// compute X'Y
					JCublas2.cublasSgemmBatched(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, nCoeff, 1, rbs,
							Pointer.to(new float[] { 1.0f }), dDAP, rbs, dRAP, rbs, Pointer.to(new float[] { 0.0f }),
							dBAP, nCoeff, nBatch);
					JCuda.cudaStreamSynchronize(stream);
				} catch (Exception e) {
					LOGGER.log(Level.WARNING, "cuBlass Error : Can not perform (X'Y) computation.");
					throw new IllegalStateException(e);
				}

				try {
					// compute pinv(X'X)*(X'Y)
					JCublas2.cublasSgemmBatched(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, nCoeff, 1, nCoeff,
							Pointer.to(new float[] { 1.0f }), dIAAP, nCoeff, dBAP, nCoeff,
							Pointer.to(new float[] { 0.0f }), dCAP, nCoeff, nBatch);
					JCuda.cudaStreamSynchronize(stream);
				} catch (Exception e) {
					LOGGER.log(Level.WARNING, "cuBlass Error : Can not perform pinv(X'X)*(X'Y) computation.");
					throw new IllegalStateException(e);
				}

				try {
					// limit coeff
					// Call the kernel function.
					if (nCoeff == 2) {
						blockSizeX = 1024;
						gridSizeX = (int) Math.ceil((double) nBatch / blockSizeX);
						JCudaDriver.cuLaunchKernel(limitCoeffKernel, gridSizeX, 1, 1, blockSizeX, 1, 1, 0, null,
								limitCoeffKernelParam, null);
						cuCtxSynchronize();
						JCuda.cudaStreamSynchronize(stream);
					}
				} catch (Exception e) {
					LOGGER.log(Level.WARNING, "cuBlass Error : Can not perform limit coeff kernel.");
					throw new IllegalStateException(e);
				}

				try {
					// compute SumSquareErr = sum(E.^2)
					gridSizeX = (int) Math.ceil((double) nBatch / blockSizeX);
					JCudaDriver.cuLaunchKernel(sumSquareErrorKernel, gridSizeX, 1, 1, blockSizeX, 1, 1, 0, null,
							sumSquareErrorKernelParams, null);
					cuCtxSynchronize();
					JCuda.cudaStreamSynchronize(stream);
				} catch (Exception e) {
					LOGGER.log(Level.WARNING, "cuBlass Error : Can not perform sum square error computation.");
					throw new IllegalStateException(e);
				}

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

				batcTime = batcTime + (System.nanoTime() - batcTimeTick);

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

				JCuda.cudaStreamSynchronize(stream);

				JCuda.cudaFree(memSetKernelParams);
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
			JCudaDriver.cuLaunchKernel(reverseVec, gridSizeX, 1, 1, blockSizeX, 1, 1, 0, null,
					reverseVecKernelParams, null);
			cuCtxSynchronize();
		} catch (Exception e) {
			LOGGER.log(Level.SEVERE, "cuBlass Error : Can not set input data and reverse data.");
			throw new IllegalStateException(e);
		}
	}

	private void initCudaDevice() throws IllegalStateException {
		// Initialize the driver and create a context for the first device.
		try {
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
	}

	private void loadCUKernelFunctions() throws IllegalStateException {
		// Obtain a function pointer to the kernel function.
		try {
			reverseVec = new CUfunction();
			JCudaDriver.cuModuleGetFunction(reverseVec, reverseVecModule, "reverseVec");

			memSetKernel = new CUfunction();
			JCudaDriver.cuModuleGetFunction(memSetKernel, memSetModule, "memSetKernel");

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

			memSetModule = new CUmodule();
			JCudaDriver.cuModuleLoad(memSetModule, "classes/memSetKernel.ptx");

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

	private void batchCudaArraysAlloc(final int nCoeff, int maxRBS, int maxNBatch, Pointer dDArrays, Pointer dRArrays,
			Pointer dAArrays, Pointer dBArrays, Pointer dIAArrays, Pointer dCArrays, Pointer dEArrays,
			Pointer dSSEArrays) throws IllegalStateException {
		try {
			cudaMalloc(dDArrays, maxNBatch * Sizeof.FLOAT * maxRBS * nCoeff);
			cudaMalloc(dRArrays, maxNBatch * Sizeof.FLOAT * maxRBS);
			cudaMalloc(dAArrays, maxNBatch * Sizeof.FLOAT * nCoeff * nCoeff);
			cudaMalloc(dBArrays, maxNBatch * Sizeof.FLOAT * nCoeff);
			cudaMalloc(dIAArrays, maxNBatch * Sizeof.FLOAT * nCoeff * nCoeff);
			cudaMalloc(dCArrays, maxNBatch * Sizeof.FLOAT * nCoeff);
			cudaMalloc(dEArrays, maxNBatch * Sizeof.FLOAT * maxRBS);
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
