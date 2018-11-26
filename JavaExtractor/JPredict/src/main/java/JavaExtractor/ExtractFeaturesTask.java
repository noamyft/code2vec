package JavaExtractor;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;

import org.apache.commons.lang3.StringUtils;

import com.github.javaparser.ParseException;
import com.github.javaparser.ast.CompilationUnit;

import JavaExtractor.Common.CommandLineValues;
import JavaExtractor.Common.Common;
import JavaExtractor.Common.MethodContent;
import JavaExtractor.FeaturesEntities.ProgramFeatures;

public class ExtractFeaturesTask implements Callable<Void> {
	CommandLineValues m_CommandLineValues;
	CompilationUnit m_CompilationUnit;
	String code = null;
	Path filePath;

	public ExtractFeaturesTask(CommandLineValues commandLineValues, Path path) {
		m_CommandLineValues = commandLineValues;
		this.filePath = path;
		try {
			this.code = new String(Files.readAllBytes(path));
		} catch (IOException e) {
			e.printStackTrace();
			this.code = Common.EmptyString;
		}
	}

	@Override
	public Void call() throws Exception {
		//System.err.println("Extracting file: " + filePath);
		processFile();
		//System.err.println("Done with file: " + filePath);
		return null;
	}

	public void processFile() {
		ArrayList<ProgramFeatures> features;
		try {
			features = extractSingleFile();
		} catch (ParseException | IOException e) {
			e.printStackTrace();
			return;
		}
		if (features == null) {
			return;
		}
		
                System.out.println();
                //TODO : change comments here to switch betwenn normal mode and method-name mode
		String toPrint= featuresToString(features);
//                String toPrint = namesToString(features, this.filePath.toString());
		if (toPrint.length() > 0) {
			System.out.println(toPrint);				
		}
	}

	public ArrayList<ProgramFeatures> extractSingleFile() throws ParseException, IOException {
		FeatureExtractor featureExtractor = new FeatureExtractor(m_CommandLineValues, code);

                //TODO : change comments here to switch betwenn normal mode and method-name mode
		ArrayList<ProgramFeatures> features = featureExtractor.extractFeatures();
//                ArrayList<MethodContent> features = featureExtractor.extractFeaturesMethods();

		m_CompilationUnit = featureExtractor.getParsedFile();

		return features;
	}

	public String featuresToString(ArrayList<ProgramFeatures> features) {
		if (features == null || features.isEmpty()) {
			return Common.EmptyString;
		}

		List<String> methodsOutputs = new ArrayList<>();

		for (ProgramFeatures singleMethodfeatures : features) {
			StringBuilder builder = new StringBuilder();
			
			String toPrint = Common.EmptyString;
			toPrint = singleMethodfeatures.toString();
			if (m_CommandLineValues.PrettyPrint) {
				toPrint = toPrint.replace(" ", "\n\t");
			}
			builder.append(toPrint);
			

			methodsOutputs.add(builder.toString());

		}
		return StringUtils.join(methodsOutputs, "\n");
	}
        
        public String namesToString(ArrayList<MethodContent> features, String path) {
		if (features == null || features.isEmpty()) {
			return Common.EmptyString;
		}

		List<String> methodsOutputs = new ArrayList<>();

		for (MethodContent singleMethodfeatures : features) {
			StringBuilder builder = new StringBuilder();
			
			String toPrint = Common.EmptyString;
			toPrint = singleMethodfeatures.getName();
			builder.append(toPrint);
                        builder.append(",");
                        builder.append(path);

			methodsOutputs.add(builder.toString());

		}
		return StringUtils.join(methodsOutputs, "\n");
	}
}
