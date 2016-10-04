import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
//import weka.classifiers.trees.SimpleCart;
import weka.classifiers.Evaluation;
import java.util.Random;

public class Main {
    public static void main(String[] args) throws Exception {
        DataSource source = new DataSource("../forrestTemp/datasets/play_tennis.arff");
        Instances data = source.getDataSet();
        // setting class attribute if the data format does not provide this information
        // For example, the XRFF format saves the class attribute information as well
        if (data.classIndex() == -1)
            data.setClassIndex(data.numAttributes() - 1);

        // builds classifier
        String[] options = {"-S", "1", "-M", "2.0", "-N", "5", "-C", "1.0"};
        SimpleCart tree = new SimpleCart();
        tree.setOptions(options);
        Evaluation eval = new Evaluation(data);
        eval.crossValidateModel(tree, data, 10, new Random(1));

        System.out.println(
                "Correctly classified instances (mean):" + String.valueOf(
                        eval.correct() / (eval.correct() + eval.incorrect())
                )
        );
    }
}
