package th.ac.kmitl.it.prip.fractal;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.regex.Pattern;

public class Parameters {
	public static enum ProcessName {
		COMPRESS, DECOMPRESS, DISTRIBUTED_COMPRESS, DISTRIBUTED_DECOMPRESS,
	}

	// co-parameters
	private ProcessName processName = null;
	private String testName = null;
	private String infile = null;
	private String inPathPrefix = "";
	private int fromIdx = -1;
	private int toIdx = -1;
	private String outdir = null;
	private String inExtension = null;
	private String outExtension = null;
	private int maxParallelProcess = 1;
	private long progressReportRate = 0;
	private boolean skipIfExist = true;
	private boolean isHelp;
	private boolean validParams;
	private int overlap = 0;

	// compression parameters
	private boolean adaptivePartition = true;
	private int dStep = 1;
	private int nCoeff = 2;
	private float coeffLimit = 0.9f;
	private int minBlockSize = 4;
	private int maxBlockSize = 128;
	private float thresh = 1e-4f;
	private int domainScale = 2;
	private int frameLength = -1;
	private boolean usingCV = false;
	private boolean gpuEnable = false;
	private float regularize = 0.0f;

	// decompression parameters
	private double alpha = 1.0d;
	private int maxIteration = 15;
	private int samplingRate = 0;

	private Parameters() {
	}

	public Parameters(String[] args) {
		this.generateFrom(args);
	}

	private void setDefault() {
		if (infile == null || nCoeff < 2) {
			validParams = false;
			return;
		}
		if (fromIdx < 0) {
			fromIdx = 0;
		}
		if (toIdx < 0) {
			try {
				toIdx = Files.readAllLines(Paths.get(infile)).size();
			} catch (IOException e) {
				toIdx = fromIdx;
				e.printStackTrace();
			}
		}
		if (frameLength > 0 && frameLength < maxBlockSize * domainScale) {
			frameLength = maxBlockSize * domainScale;
		}
		if (outdir == null) {
			outdir = "..";
		}
		if (inExtension == null) {
			inExtension = "wav";
		}
		if (outExtension == null) {
			outExtension = "bin";
		}
		if (testName == null) {
			testName = Paths.get(infile).getFileName().toString()
					.replaceAll(".txt", "").replaceAll(".fileids", "");
			String codeDir = String
					.format("//%s_R%dto%d_T%s_AA%d_S%d_NCOEF%d_LIMCOEFF%1.1e//",
							testName, minBlockSize, maxBlockSize,
							String.format("%1.1e", thresh), domainScale, dStep,
							nCoeff, coeffLimit);
			outdir = Paths.get(outdir, codeDir).toAbsolutePath().toString();
		} else {
			String codeDir = String.format("//%s//", testName);
			outdir = Paths.get(outdir, codeDir).toAbsolutePath().toString();
		}
		validParams = true;
	}

	public Parameters generateFrom(String[] args) {
		new Parameters();
		for (String arg : args) {
			try {
				String[] params = arg.split(" ", 2);
				String argName = "";
				String argValue = "";
				if (params.length == 2) {
					argName = params[0];
					argValue = params[1];
				} else if (params.length == 1) {
					argName = params[0];
				}
				setParameter(argName, argValue);
			} catch (IndexOutOfBoundsException e) {
				if (arg.equalsIgnoreCase("help")) {
					setParameter("help", "");
				} else {
					// skip fault value
				}
			}
		}
		setDefault();
		return this;
	}

	protected void setParameter(String argName, String argValue) {
		switch (argName.toLowerCase()) {
		case "testname":
			testName = argValue;
			break;
		case "infile":
			// parse to standard
			infile = Paths.get(argValue).toAbsolutePath().toString();
			break;
		case "inpathprefix":
			inPathPrefix = Paths.get(argValue).toAbsolutePath().toString();
			break;
		case "outdir":
			outdir = Paths.get(argValue).toAbsolutePath().toString();
			break;
		case "inext":
			inExtension = argValue;
			break;
		case "outext":
			outExtension = argValue;
			break;
		case "minr":
		case "minpartsize":
			minBlockSize = Integer.parseInt(argValue);
			break;
		case "maxr":
		case "maxpartsize":
			maxBlockSize = Integer.parseInt(argValue);
			break;
		case "pthresh":
			thresh = Float.parseFloat(argValue);
			break;
		case "step":
			dStep = Integer.parseInt(argValue);
			break;
		case "domainscale":
		case "aa":
			domainScale = Integer.parseInt(argValue);
			break;
		case "maxprocess": // max parallel process
			maxParallelProcess = Integer.parseInt(argValue);
			break;
		case "h":
		case "help":
			isHelp = true;
			break;
		case "adaptive":
			adaptivePartition = Boolean.parseBoolean(argValue);
			break;
		case "usingcv":
		case "cv":
			usingCV = Boolean.parseBoolean(argValue);
			break;
		case "reportrate":
			progressReportRate = Long.parseLong(argValue);
			break;
		case "ncoeff":
			nCoeff = Integer.parseInt(argValue);
			break;
		case "coefflimit":
			coeffLimit = Float.parseFloat(argValue);
			break;
		case "alpha":
			alpha = Float.parseFloat(argValue);
			break;
		case "maxiter":
		case "maxiteration":
			maxIteration = Integer.parseInt(argValue);
			break;
		case "samplingrate":
		case "fs":
			samplingRate = Integer.parseInt(argValue);
			break;
		case "fromto":
			if (Pattern.matches("(\\d+)-(\\d+)", argValue)) {
				String[] values = argValue.split("-", 2);
				fromIdx = Integer.parseInt(values[0]) - 1;
				toIdx = Integer.parseInt(values[1]);
			}
			break;
		case "processname":
			processName = ProcessName.valueOf(argValue.toUpperCase());
			System.out.println(processName);
			break;
		case "skipifexist":
			skipIfExist = Boolean.parseBoolean(argValue);
			break;
		case "framelength":
			frameLength = Integer.parseInt(argValue);
			break;
		case "overlap":
			overlap = Integer.parseInt(argValue);
			break;
		case "gpuenable":
		case "enablegpu":
		case "gpu":
			gpuEnable = Boolean.parseBoolean(argValue);
			break;
		case "reguralize":
			regularize = Float.parseFloat(argValue);
			break;

		default:
			break;
		}
	}

	public String getTestName() {
		return testName;
	}

	public String getInfile() {
		return infile;
	}

	public String getOutdir() {
		return outdir;
	}

	public String getInExtension() {
		return inExtension;
	}

	public String getOutExtension() {
		return outExtension;
	}

	public int getMinBlockSize() {
		return minBlockSize;
	}

	public int getMaxBlockSize() {
		return maxBlockSize;
	}

	public float getThresh() {
		return thresh;
	}

	public int getDomainScale() {
		return domainScale;
	}

	public int getMaxParallelProcess() {
		return maxParallelProcess;
	}

	public boolean isAdaptivePartition() {
		return adaptivePartition;
	}

	public int getDStep() {
		return dStep;
	}

	public int getNCoeff() {
		return nCoeff;
	}

	public float getCoeffLimit() {
		return coeffLimit;
	}

	public boolean isValidParams() {
		return validParams;
	}

	public long getProgressReportRate() {
		return progressReportRate;
	}

	@Override
	public String toString() {
		StringBuilder builder = new StringBuilder();
		builder.append("Parameters \n[  processName = ");
		builder.append(processName);
		builder.append("\n  testName = ");
		builder.append(testName);
		builder.append("\n  infile = ");
		builder.append(infile);
		builder.append("\n  inPathPrefix = ");
		builder.append(inPathPrefix);
		builder.append("\n  fromIdx = ");
		builder.append(fromIdx);
		builder.append("\n  toIdx = ");
		builder.append(toIdx);
		builder.append("\n  outdir = ");
		builder.append(outdir);
		builder.append("\n  inExtension = ");
		builder.append(inExtension);
		builder.append("\n  outExtension = ");
		builder.append(outExtension);
		builder.append("\n  maxParallelProcess = ");
		builder.append(maxParallelProcess);
		builder.append("\n  progressReportRate = ");
		builder.append(progressReportRate);
		builder.append("\n  skipIfExist = ");
		builder.append(skipIfExist);
		builder.append("\n  isHelp = ");
		builder.append(isHelp);
		builder.append("\n  validParams = ");
		builder.append(validParams);
		builder.append("\n  adaptivePartition = ");
		builder.append(adaptivePartition);
		builder.append("\n  dStep = ");
		builder.append(dStep);
		builder.append("\n  nCoeff = ");
		builder.append(nCoeff);
		builder.append("\n  coeffLimit = ");
		builder.append(coeffLimit);
		builder.append("\n  minBlockSize = ");
		builder.append(minBlockSize);
		builder.append("\n  maxBlockSize = ");
		builder.append(maxBlockSize);
		builder.append("\n  thresh = ");
		builder.append(thresh);
		builder.append("\n  domainScale = ");
		builder.append(domainScale);
		builder.append("\n  frameLength = ");
		builder.append(frameLength);
		builder.append("\n  usingCV = ");
		builder.append(usingCV);
		builder.append("\n  alpha = ");
		builder.append(alpha);
		builder.append("\n  maxIteration = ");
		builder.append(maxIteration);
		builder.append("\n  samplingRate = ");
		builder.append(samplingRate);
		builder.append("  ]\n");
		return builder.toString();
	}

	public boolean isHelp() {
		return isHelp;
	}

	public String getInPathPrefix() {
		return inPathPrefix;
	}

	public double getAlpha() {
		return alpha;
	}

	public int getMaxIteration() {
		return maxIteration;
	}

	public int getSamplingRate() {
		return samplingRate;
	}

	public int getFromIdx() {
		return fromIdx;
	}

	public int getToIdx() {
		return toIdx;
	}

	public ProcessName getProcessName() {
		return processName;
	}

	public boolean isSkipIfExist() {
		return skipIfExist;
	}

	public int getFrameLength() {
		return frameLength;
	}

	public boolean isUsingCV() {
		return usingCV;
	}

	public int getOverlap() {
		return overlap;
	}

	public boolean isGpuEnable() {
		return gpuEnable;
	}

	public float getRegularize() {
		return regularize;
	}

}