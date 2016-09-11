package th.ac.kmitl.it.prip.fractal.dataset;

import java.io.BufferedReader;
import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.ShortBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Executors;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.stream.Collectors;

import javax.sound.sampled.AudioFileFormat;
import javax.sound.sampled.AudioFormat;
import javax.sound.sampled.AudioInputStream;
import javax.sound.sampled.AudioSystem;
import javax.sound.sampled.UnsupportedAudioFileException;

import com.jmatio.io.MatFileReader;
import com.jmatio.io.MatFileWriter;
import com.jmatio.types.MLArray;
import com.jmatio.types.MLDouble;

import th.ac.kmitl.it.prip.fractal.Parameters;

public class DataSetManager implements DataSetAPIv1 {
	private static final Logger LOGGER = Logger.getLogger(DataSetManager.class.getName());

	final static int SAMPLE_BYTE_SIZE = 2;

	private String[] nameList;
	private List<String> timing;
	private List<String> completed;
	private Parameters parameters;

	public DataSetManager(Parameters parameters) throws IOException {
		this.parameters = parameters;
		Paths.get(parameters.getOutdir()).toFile().mkdirs();

		this.nameList = getIdsNameList();

		Path timingLogPath = Paths.get(parameters.getOutdir(), "timelog.txt");
		if (timingLogPath.toFile().exists()) {
			timing = Files.readAllLines(timingLogPath);
		} else {
			timing = new ArrayList<>(nameList.length);
			for (int i = 0; i < nameList.length; i++) {
				timing.add("");
			}
		}

		Path completedLogPath = Paths.get(parameters.getOutdir(), "completedlog.txt");
		if (completedLogPath.toFile().exists()) {
			completed = Files.readAllLines(completedLogPath);
		} else {
			completed = new ArrayList<>(nameList.length);
			for (int i = 0; i < nameList.length; i++) {
				completed.add("");
			}
		}
	}

	/**
	 * @param fileIds
	 *            input file that store paths of audio data
	 * @return audio paths
	 * @throws IOException
	 */
	private List<String> getIdsPathList() throws IOException {
		List<String> pathsList = new ArrayList<String>();
		try {
			String[] idsArrays = new String[nameList.length];
			idsArrays = removeExtension(nameList);
			for (int i = 0; i < idsArrays.length; i++) {
				if (parameters.getInPathPrefix().length() > 0) {
					pathsList.add(Paths.get(parameters.getInPathPrefix() + "\\" + idsArrays[i]).toString());
				} else {
					pathsList.add(Paths.get(idsArrays[i]).toString());
				}
			}
		} catch (IOException e) {
			LOGGER.log(Level.SEVERE, e.getMessage());
			throw e;
		}
		return pathsList;
	}

	private String[] getIdsNameList() throws IOException {
		String[] idsArrays = new String[0];
		try {
			List<String> idsList = Files.readAllLines(Paths.get(parameters.getInfile()));
			idsArrays = new String[idsList.size()];
			idsList.toArray(idsArrays);
			idsArrays = removeExtension(idsArrays);
			for (int i = 0; i < idsArrays.length; i++) {
				idsArrays[i] = Paths.get(idsArrays[i]).toString();
				LOGGER.log(Level.INFO, idsArrays[i]);
			}
		} catch (IOException e) {
			LOGGER.log(Level.SEVERE, e.getMessage());
			throw e;
		}
		return idsArrays;
	}

	@SuppressWarnings("unused")
	private String removeExtension(String fileName) throws IOException {
		List<String> extList = new ArrayList<String>();
		String newFileName = fileName;
		try {
			InputStream in = DataSetManager.class.getResourceAsStream("supportextension.txt");
			BufferedReader br = new BufferedReader(new InputStreamReader(in));
			extList = br.lines().collect(Collectors.toList());
			for (String ext : extList) {
				newFileName = fileName.replace(ext, "");
			}
			in.close();
			br.close();

		} catch (IOException e) {
			LOGGER.log(Level.SEVERE, e.getMessage());
			throw e;
		}
		return newFileName;
	}

	private String[] removeExtension(String[] fileName) throws IOException {
		List<String> extList = new ArrayList<String>();
		try {
			InputStream in = DataSetManager.class.getResourceAsStream("supportextension.txt");
			BufferedReader br = new BufferedReader(new InputStreamReader(in));
			extList = br.lines().collect(Collectors.toList());
			in.close();
			br.close();
			for (int i = 0; i < fileName.length; i++) {
				for (String ext : extList) {
					fileName[i] = fileName[i].replace(ext, "");
				}
			}
		} catch (IOException e) {
			LOGGER.log(Level.SEVERE, e.getMessage());
			throw e;
		}
		return fileName;
	}

	private float[] audioread(String fileName, String extension) throws UnsupportedAudioFileException, IOException {

		float[] audioArrays = null;

		try {
			switch (extension) {
			case "wav":
			case "":
				audioArrays = wavToDat(fileName + "." + extension);
				break;
			case "raw":
				audioArrays = rawToDat(fileName + "." + extension);
				break;
			default:
				break;
			}
		} catch (UnsupportedAudioFileException e) {
			LOGGER.log(Level.SEVERE, e.getMessage());
			throw e;
		} catch (IOException e) {
			LOGGER.log(Level.SEVERE, e.getMessage());
			throw e;
		}

		return audioArrays;
	}

	private double[][] codesread(String fileName, String extension) throws FileNotFoundException, IOException {
		double[][] codes = null;
		try {
			switch (extension) {
			case "mat":
			case "":
				codes = matToCodes(fileName + "." + extension);
				break;
			case "bin":
				break;
			default:
				break;
			}
		} catch (FileNotFoundException e) {
			LOGGER.log(Level.SEVERE, e.getMessage());
			throw e;
		} catch (IOException e) {
			LOGGER.log(Level.SEVERE, e.getMessage());
			throw e;
		}

		return codes;
	}

	private void writecode(String fileName, double[][] codeData, String codeExtention) throws IOException {
		try {
			switch (codeExtention) {
			case "mat":
				writeToMat(fileName + "." + codeExtention, codeData);
				break;

			default:
				break;
			}
		} catch (IOException e) {
			LOGGER.log(Level.SEVERE, e.getMessage());
			throw e;
		}
	}

	private void writeaudio(String fileName, double[] audioData, String audioExtension, int sampleRate)
			throws IOException {
		try {
			Paths.get(fileName).getParent().toFile().mkdirs();
			switch (audioExtension) {
			case "wav":
				writeToWav(fileName + "." + audioExtension, audioData, sampleRate);
				break;
			case "raw":
				writeToRaw(fileName + "." + audioExtension, audioData);
				break;

			default:
				break;
			}
		} catch (IOException e) {
			LOGGER.log(Level.SEVERE, e.getMessage());
			throw e;
		}
	}

	private float[] rawToDat(String fileName) throws IOException {
		float[] audioData = null;
		try {
			FileInputStream fis = new FileInputStream(fileName);
			byte[] buffer = new byte[fis.available()];
			fis.read(buffer);
			ShortBuffer ib = ByteBuffer.wrap(buffer).order(ByteOrder.LITTLE_ENDIAN).asShortBuffer(); //
			short[] shortBuffer = new short[ib.capacity()];
			ib.get(shortBuffer);
			audioData = new float[shortBuffer.length];
			for (int i = 0; i < shortBuffer.length; i++) {
				audioData[i] = (float) shortBuffer[i];
			}

			fis.close();
		} catch (IOException e) {
			LOGGER.log(Level.SEVERE, e.getMessage());
			throw e;
		}
		return audioData;
	}

	private float[] wavToDat(String fileName) throws UnsupportedAudioFileException, IOException {
		float[] audioData = null;
		File file;
		AudioInputStream ais;
		int bytesPerFrame = 0;
		int audioLength = 0;
		try {
			file = Paths.get(fileName).toFile();
			ais = AudioSystem.getAudioInputStream(file);

			bytesPerFrame = ais.getFormat().getFrameSize();
			audioLength = (int) ais.getFrameLength();
			audioData = new float[audioLength];
			if (bytesPerFrame == AudioSystem.NOT_SPECIFIED) {
				// some audio formats may have unspecified frame size
				// in that case we may read any amount of bytes
				bytesPerFrame = 1;
			}
			// Set an arbitrary buffer size of 1024 frames.
			byte[] audioBytes = new byte[bytesPerFrame];
			// Try to read numBytes bytes from the file.
			int frameIdx = 0;
			while (frameIdx < audioLength) {
				ais.read(audioBytes);
				double leftChannel = ((audioBytes[0] & 0xFF) | (audioBytes[1] << 8)) / 32768.0;
				if (leftChannel < 1) {
					audioData[frameIdx] = (float) (leftChannel * Math.pow(2, 15));
				}
				frameIdx += 1;
			}
			ais.close();
		} catch (UnsupportedAudioFileException | IOException e) {
			LOGGER.log(Level.SEVERE, e.getMessage());
			throw e;
		}
		return audioData;
	}

	private double[][] matToCodes(String fileName) throws IOException, FileNotFoundException {
		double[][] codes = null;
		try {
			MatFileReader mfr = new MatFileReader(fileName);
			codes = ((MLDouble) mfr.getMLArray("f")).getArray();
		} catch (FileNotFoundException e) {
			LOGGER.log(Level.SEVERE, e.getMessage());
			throw e;
		} catch (IOException e) {
			LOGGER.log(Level.SEVERE, e.getMessage());
			throw e;
		}
		return codes;
	}

	@SuppressWarnings("unused")
	private double[][] binToCodes(String fileName) {
		LOGGER.log(Level.WARNING, "Can not open " + fileName + ". This api does not implement yet.");
		return new double[0][0];
	}

	private void writeToRaw(String fileName, double[] audioData) throws IOException {
		try {

			short[] shortBuffer = new short[audioData.length];
			for (int i = 0; i < shortBuffer.length; i++) {
				shortBuffer[i] = (short) audioData[i];
			}
			ByteBuffer buffer = ByteBuffer.allocate(shortBuffer.length * SAMPLE_BYTE_SIZE)
					.order(ByteOrder.LITTLE_ENDIAN);
			buffer.asShortBuffer().put(shortBuffer);
			Files.write(Paths.get(fileName), buffer.array());
		} catch (IOException e) {
			LOGGER.log(Level.SEVERE, e.getMessage());
			throw e;
		}
	}

	private void writeToMat(String fileName, double[][] codeData) throws IOException {
		String varName = "f";
		MLDouble mlDouble = new MLDouble(varName, codeData);

		// write array to file
		List<MLArray> list = new ArrayList<MLArray>();
		list.add(mlDouble);

		// write arrays to file
		try {
			File outputDir = Paths.get(fileName).getParent().toFile();
			if (!outputDir.exists()) {
				outputDir.mkdirs();
			}
			new MatFileWriter(fileName, list);
		} catch (IOException e) {
			LOGGER.log(Level.SEVERE, e.getMessage());
			throw e;
		}
	}

	private void writeToWav(String fileName, double[] audioData, int sampleRate) throws IOException {
		final boolean bigEndian = false;
		final int nBit = 16;
		final int nCH = 1;
		try {
			byte[] byteBuffer = new byte[audioData.length * 2];
			short[] valueBuffer = new short[audioData.length];
			for (int i = 0; i < valueBuffer.length; i++) {
				valueBuffer[i] = (short) ((audioData[i] / Math.pow(2, 15)) * Short.MAX_VALUE);
			}
			ByteBuffer.wrap(byteBuffer).order(ByteOrder.LITTLE_ENDIAN).asShortBuffer().put(valueBuffer);
			ByteArrayInputStream bais = new ByteArrayInputStream(byteBuffer);
			AudioFormat format = new AudioFormat((float) sampleRate, nBit, nCH, true, bigEndian);
			AudioInputStream ais = new AudioInputStream(bais, format, byteBuffer.length);
			AudioSystem.write(ais, AudioFileFormat.Type.WAVE, Paths.get(fileName).toFile());
			ais.close();
		} catch (IOException e) {
			LOGGER.log(Level.SEVERE, e.getMessage());
			throw e;
		}
	}

	@Override
	public float[] readAudio(int index) throws UnsupportedAudioFileException, IOException {
		return audioread(Paths.get(parameters.getInPathPrefix(), nameList[index]).toString(),
				parameters.getInExtension());
	}

	@Override
	public boolean writeAudio(int index, double[] audioData) throws IOException {
		writeaudio(Paths.get(parameters.getOutdir(), nameList[index]).toString(), audioData,
				parameters.getOutExtension(), parameters.getSamplingRate());
		return true;
	}

	@Override
	public double[][] readCode(int index) throws FileNotFoundException, IOException {
		return codesread(Paths.get(parameters.getInPathPrefix(), nameList[index]).toString(),
				parameters.getInExtension());
	}

	@Override
	public boolean writeCode(int index, double[][] codeData) throws IOException {
		writecode(Paths.get(parameters.getOutdir(), nameList[index]).toString(), codeData,
				parameters.getOutExtension());
		return true;
	}

	@Override
	public boolean updateTimingLog(int index, long time) throws IOException {
		timing.set(index, new String(Long.toString(time)));
		Files.write(Paths.get(parameters.getOutdir(), "\\timelog.txt"), timing);
		return true;
	}

	@Override
	public boolean updateCompletedLog(int index, String logString) throws IOException {
		completed.set(index, new String(logString));
		Files.write(Paths.get(parameters.getOutdir(), "\\completedlog.txt"), completed);
		return true;
	}

	@Override
	public boolean writeInfo() throws IOException {
		List<String> info = new ArrayList<>();
		try {
			info.add("Java " + System.getProperty("java.version"));
			info.add("Version " + Executors.class.getPackage().getImplementationVersion());
			info.add("Date " + LocalDateTime.now());
			info.add(parameters.toString());
			Files.write(Paths.get(parameters.getOutdir(), "info.txt"), info);
		} catch (IOException e) {
			LOGGER.log(Level.SEVERE, e.getMessage());
			throw e;
		}
		return true;
	}

	@Override
	public boolean writeParameters() throws IOException {
		List<String> parameter = new ArrayList<>();
		try {
			parameter.add(parameters.toString());
			Files.write(Paths.get(parameters.getOutdir(), "\\parameters.txt"), parameter);
		} catch (IOException e) {
			LOGGER.log(Level.SEVERE, e.getMessage());
			throw e;
		}
		return true;
	}

	@Override
	public boolean writeOutputPaths() throws IOException {
		List<String> pathsList = new ArrayList<String>();
		try {
			String[] idsArrays = new String[nameList.length];
			idsArrays = removeExtension(nameList);
			for (int i = 0; i < idsArrays.length; i++) {
				if (parameters.getInPathPrefix().length() > 0) {
					pathsList.add(Paths.get(parameters.getInPathPrefix(), idsArrays[i]).toString());
				} else {
					pathsList.add(Paths.get(idsArrays[i]).toString());
				}
			}
			Files.write(Paths.get(parameters.getOutdir(), "\\pathslist.txt"), pathsList);
		} catch (IOException e) {
			LOGGER.log(Level.SEVERE, e.getMessage());
			throw e;
		}
		return true;
	}

	@Override
	public int estimateNumSamples() throws IOException {
		int numSample = 0;
		for (int i = 0; i < nameList.length; i++) {
			Path outputPath = Paths.get(parameters.getInPathPrefix(), nameList[i] + "." + parameters.getInExtension());
			FileInputStream fis = new FileInputStream(outputPath.toString());
			numSample = numSample + fis.available() / SAMPLE_BYTE_SIZE;
			fis.close();
		}
		return numSample;
	}

	@Override
	public int getSize() {
		return nameList.length;
	}

	@Override
	public boolean existOutput(int index) {
		Path outputPath = Paths.get(parameters.getOutdir(), nameList[index] + "." + parameters.getOutExtension());
		File file = new File(outputPath.toString());
		return file.exists();
	}

	@Override
	public String getName(int index) {
		return new String(nameList[index]);
	}
}
