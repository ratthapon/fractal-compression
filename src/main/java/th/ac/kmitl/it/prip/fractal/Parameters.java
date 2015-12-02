package th.ac.kmitl.it.prip.fractal;

import java.nio.file.Paths;

public class Parameters {

	private String testName = null;
	private String infile = null;
	private String inPathPrefix = "";
	private String outdir = null;
	private String inExtension = null;
	private String outExtension = null;
	private int minBlockSize = 4;
	private int maxBlockSize = 128;
	private float thresh = 1e-4f;
	private int domainScale = 2;
	private int maxParallelProcess = 1;
	private boolean adaptivePartition = true;
	private int dStep = 1;
	private int nCoeff = 2;
	private float coeffLimit = 0.9f;
	private boolean validParams;
	private long progressReportRate = 0;
	private double alpha = 1.0d;
	private int maxIteration = 15;
	private int samplingRate = 0;
	private boolean isHelp;

	private Parameters() {
	}

	public Parameters(String[] args) {
		new Parameters();
		this.generateFrom(args);
	}

	private void setDefault() {
		if (infile == null || nCoeff < 2) {
			validParams = false;
			return;
		}
		if (testName == null) {
			testName = Paths.get(infile).getFileName().toString()
					.replaceAll(".txt", "").replaceAll(".fileids", "");
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
		String codeDir = String
				.format("//FractalCode//%s_R%dto%d_T%s_AA%d_S%d_NCOEF%d_LIMCOEFF%1.1e//",
						testName, minBlockSize, maxBlockSize,
						String.format("%1.1e", thresh), domainScale, dStep,
						nCoeff, coeffLimit);
		outdir = Paths.get(outdir, codeDir).toAbsolutePath().toString();
		validParams = true;
	}

	public Parameters generateFrom(String[] args) {
		for (String arg : args) {
			try {
				String argName = arg.substring(0, arg.indexOf(" "));
				String argValue = arg.substring(arg.indexOf(" ") + 1,
						arg.length());
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
		case "AA":
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
		builder.append("Parameters \n[  testName = ");
		builder.append(testName);
		builder.append("\n  infile = ");
		builder.append(infile);
		builder.append("\n  outdir = ");
		builder.append(outdir);
		builder.append("\n  audioExtension = ");
		builder.append(inExtension);
		builder.append("\n  codeExtension = ");
		builder.append(outExtension);
		builder.append("\n  minBlockSize = ");
		builder.append(minBlockSize);
		builder.append("\n  maxBlockSize = ");
		builder.append(maxBlockSize);
		builder.append("\n  thresh = ");
		builder.append(thresh);
		builder.append("\n  domainScale = ");
		builder.append(domainScale);
		builder.append("\n  maxParallelProcess = ");
		builder.append(maxParallelProcess);
		builder.append("\n  adaptivePartition = ");
		builder.append(adaptivePartition);
		builder.append("\n  dStep = ");
		builder.append(dStep);
		builder.append("\n  nCoeff = ");
		builder.append(nCoeff);
		builder.append("\n  coeffLimit = ");
		builder.append(coeffLimit);
		builder.append("\n  validParams = ");
		builder.append(validParams);
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

}