package th.ac.kmitl.it.prip.fractal;

import java.io.BufferedReader;
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
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.stream.Collectors;

import javax.sound.sampled.AudioInputStream;
import javax.sound.sampled.AudioSystem;
import javax.sound.sampled.UnsupportedAudioFileException;

import com.jmatio.io.MatFileReader;
import com.jmatio.io.MatFileWriter;
import com.jmatio.types.MLArray;
import com.jmatio.types.MLDouble;

public class DataHandler {
	private static final Logger LOGGER = Logger.getLogger(DataHandler.class
			.getName());

	/**
	 * @param fileIds
	 *            input file that store paths of audio data
	 * @return audio paths
	 */
	public static String[] getIdsPathList(Parameters parameters) {
		try {
			List<String> idsList = Files.readAllLines(Paths.get(parameters
					.getInfile()));
			String[] idsArrays = new String[idsList.size()];
			idsList.toArray(idsArrays);
			idsArrays = removeExtension(idsArrays);
			for (int i = 0; i < idsArrays.length; i++) {
				if (parameters.getInPathPrefix().length() > 0) {
					idsArrays[i] = Paths.get(
							parameters.getInPathPrefix() + "\\" + idsArrays[i])
							.toString();
				} else {
					idsArrays[i] = Paths.get(idsArrays[i]).toString();
				}

			}
			return idsArrays;
		} catch (IOException e) {
			LOGGER.log(Level.SEVERE, e.getMessage());
			return null;
		}
	}

	public static String[] getIdsNameList(Parameters parameters) {
		try {
			List<String> idsList = Files.readAllLines(Paths.get(parameters
					.getInfile()));
			String[] idsArrays = new String[idsList.size()];
			idsList.toArray(idsArrays);
			idsArrays = removeExtension(idsArrays);
			for (int i = 0; i < idsArrays.length; i++) {
				idsArrays[i] = Paths.get(idsArrays[i]).toString();
				LOGGER.log(Level.INFO, idsArrays[i]);
			}
			return idsArrays;
		} catch (IOException e) {
			LOGGER.log(Level.SEVERE, e.getMessage());
			return null;
		}
	}

	@SuppressWarnings("unused")
	private static String removeExtension(String fileName) {
		List<String> extList = new ArrayList<String>();
		try {
			InputStream in = DataHandler.class
					.getResourceAsStream("supportextension.txt");
			BufferedReader br = new BufferedReader(new InputStreamReader(in));
			extList = br.lines().collect(Collectors.toList());
			for (String ext : extList) {
				fileName = fileName.replace(ext, "");
			}
			in.close();
			br.close();

		} catch (IOException e) {
			LOGGER.log(Level.SEVERE, e.getMessage());
		}
		return fileName;
	}

	private static String[] removeExtension(String[] fileName) {
		List<String> extList = new ArrayList<String>();
		try {
			InputStream in = DataHandler.class
					.getResourceAsStream("supportextension.txt");
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
		}
		return fileName;
	}

	public static float[] audioread(String fileName, String extension) {

		float[] audioArrays = null;

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

		return audioArrays;
	}

	public static double[][] codesread(String fileName, String extension) {
		double[][] codes = null;
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

		return codes;
	}

	public static void writecode(String fileName, double[][] codeData,
			String codeExtention) {
		switch (codeExtention) {
		case "mat":
			writeToMat(fileName + "." + codeExtention, codeData);
			break;

		default:
			break;
		}
	}

	public static void writeaudio(String fileName, double[] audioData,
			String audioExtension) {
		switch (audioExtension) {
		case "raw":
			writeToRaw(fileName + "." + audioExtension, audioData);
			break;

		default:
			break;
		}
	}

	private static float[] rawToDat(String fileName) {
		float[] audioData = null;
		try {
			FileInputStream fis = new FileInputStream(fileName);
			byte[] buffer = new byte[fis.available()];
			fis.read(buffer);
			ShortBuffer ib = ByteBuffer.wrap(buffer)
					.order(ByteOrder.LITTLE_ENDIAN).asShortBuffer(); //
			short[] shortBuffer = new short[ib.capacity()];
			ib.get(shortBuffer);
			audioData = new float[shortBuffer.length];
			for (int i = 0; i < shortBuffer.length; i++) {
				audioData[i] = (float) shortBuffer[i];
			}

			fis.close();
		} catch (Exception e) {
			LOGGER.log(Level.SEVERE, e.getMessage());
		}
		return audioData;
	}

	private static float[] wavToDat(String fileName) {
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
					audioData[frameIdx] = (float) (((leftChannel)) * Math.pow(
							2, 15));
				}
				frameIdx += 1;
			}
			ais.close();
		} catch (UnsupportedAudioFileException | IOException e) {
			LOGGER.log(Level.SEVERE, e.getMessage());
		}
		return audioData;
	}

	private static double[][] matToCodes(String fileName) {
		double[][] codes = null;
		try {
			MatFileReader mfr = new MatFileReader(fileName);
			codes = ((MLDouble) mfr.getMLArray("f")).getArray();
		} catch (FileNotFoundException e) {
			LOGGER.log(Level.SEVERE, e.getMessage());
		} catch (IOException e) {
			LOGGER.log(Level.SEVERE, e.getMessage());
		}
		return codes;
	}

	@SuppressWarnings("unused")
	private static double[][] binToCodes(String fileName) {
		return null;
	}

	private static void writeToRaw(String fileName, double[] audioData) {
		try {
			final int RAW_N_BYTES = 2;
			short[] shortBuffer = new short[audioData.length];
			for (int i = 0; i < shortBuffer.length; i++) {
				shortBuffer[i] = (short) audioData[i];
			}
			ByteBuffer buffer = ByteBuffer.allocate(
					shortBuffer.length * RAW_N_BYTES).order(
					ByteOrder.LITTLE_ENDIAN);
			buffer.asShortBuffer().put(shortBuffer);
			Files.write(Paths.get(fileName), buffer.array());
		} catch (IOException e) {
			LOGGER.log(Level.SEVERE, e.getMessage());
		}
	}

	private static void writeToMat(String fileName, double[][] codeData) {
		String varName = "f";
		MLDouble mlDouble = new MLDouble(varName, codeData);

		// write array to file
		ArrayList<MLArray> list = new ArrayList<MLArray>();
		list.add(mlDouble);

		// write arrays to file
		try {
			new MatFileWriter(fileName, list);
		} catch (IOException e) {
			LOGGER.log(Level.SEVERE, e.getMessage());
		}

	}

}
