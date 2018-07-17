package moa.classifiers.meta;

import com.github.javacliparser.IntOption;
import com.github.javacliparser.MultiChoiceOption;
import com.yahoo.labs.samoa.instances.*;
import moa.classifiers.AbstractClassifier;
import moa.classifiers.Classifier;
import moa.classifiers.MultiClassClassifier;
import moa.core.FastVector;
import moa.core.Measurement;
import moa.core.Utils;
import moa.options.ClassOption;
import moa.streams.InstanceStream;

public class RandomProjectionEnsemble extends AbstractClassifier implements MultiClassClassifier {

    private final int PROJECTION_BERNOULLI  = 0;
    private final int PROJECTION_ACHLIOPTAS = 1;
    private final int PROJECTION_GAUSSIAN   = 2;

    public IntOption ensembleSizeOption = new IntOption("ensembleSize", 'e',
            "Number of experts in the ensemble.", 10, 1, 10000);

    public IntOption projectionSizeOption = new IntOption("projectionSize", 's',
            "Number of features in the projected space.", 10, 1, 10000);

    public MultiChoiceOption projectionOption = new MultiChoiceOption("projection", 'p',
            "Type of projection to be performed.",
            new String[]{"Bernoulli", "Achlioptas", "Gaussian"},
            new String[]{"Bernoulli", "Achlioptas", "Gaussian"},
            0);

    public ClassOption baseLearnerOption = new ClassOption("baseLearner",
            'b',
            "Base learner to be used",
            MultiClassClassifier.class,
            "bayes.NaiveBayes");

    protected Classifier ensemble[];
    protected Projection projections[];
    protected InstancesHeader originalHeader;
    protected InstancesHeader projectedHeader;

    @Override
    public double[] getVotesForInstance(Instance inst) {
        if(this.ensemble == null) init(inst);
        double votes[] = new double[inst.numClasses()];
        for(int i = 0 ; i < ensemble.length; i++){
            Instance projected = projections[i].project(inst);
            double v[] = ensemble[i].getVotesForInstance(projected);
            votes[Utils.maxIndex(v)]++;
        }
        if(Utils.sum(votes) > 0) Utils.normalize(votes);
        return votes;
    }

    @Override
    public void resetLearningImpl() {
        this.ensemble        = null;
        this.originalHeader  = null;
        this.projectedHeader = null;
    }

    @Override
    public void trainOnInstanceImpl(Instance inst) {
        for(int i = 0 ; i < ensemble.length; i++){
            Instance projected = projections[i].project(inst);
            ensemble[i].trainOnInstance(projected);
        }
    }

    @Override
    protected Measurement[] getModelMeasurementsImpl() {
        return null;
    }

    @Override
    public void getModelDescription(StringBuilder out, int indent) {}

    @Override
    public boolean isRandomizable() {
        return true;
    }

    private void init(Instance instnc) {
        this.originalHeader = (InstancesHeader) instnc.dataset();
        this.ensemble = new Classifier[ensembleSizeOption.getValue()];
        this.projections = new Projection[ensembleSizeOption.getValue()];
        initProjectedHeader(originalHeader, projectionSizeOption.getValue());

        for(int i = 0; i < ensemble.length; i++){
            ensemble[i] = ((Classifier) getPreparedClassOption(this.baseLearnerOption)).copy();
            ensemble[i].prepareForUse();
            ensemble[i].setModelContext(projectedHeader);
            projections[i] = new Projection(projectionOption.getChosenIndex());
        }
        int debug = 1;
    }

    private void initProjectedHeader(InstancesHeader originalHeader, int lengthProjectedSpace) {
        FastVector<Attribute> atts = new FastVector<>();
        for(int i = 0; i < lengthProjectedSpace; i++){
            Attribute att = new Attribute("proj" + (i + 1));
            atts.addElement(att);
        }
        atts.addElement(new Attribute("class",
                originalHeader.classAttribute().getAttributeValues()));
        this.projectedHeader = new InstancesHeader(new Instances(getCLICreationString(InstanceStream.class),
                atts, 0));
        this.projectedHeader.setClassIndex(this.projectedHeader.numAttributes() - 1);
    }

    class Projection{
        private int projectionType;
        private double w[][] = null;

        public Projection(int projectionType){
            this.projectionType = projectionType;
        }

        public Instance project(Instance instnc){
            int projectionLength = projectionSizeOption.getValue();

            // Initializes the weights
            if (w == null){
                w = new double[projectionLength][instnc.numAttributes() - 1];
                for (int i = 0; i < w.length; i++) {
                    for (int j = 0; j < w[i].length; j++) {
                        double r;
                        if (projectionType == PROJECTION_BERNOULLI) {
                            r = classifierRandom.nextDouble();
                            w[i][j] = r < 0.5 ? -1 : +1;
                        } else if (projectionType == PROJECTION_ACHLIOPTAS) {
                            r = classifierRandom.nextDouble();
                            if (r <= 1.0 / 6.0) w[i][j] = -Math.sqrt(3);
                            else if (r <= 2.0 / 6.0) w[i][j] = Math.sqrt(3);
                            else w[i][j] = 0.0;
                        } else if (projectionType == PROJECTION_GAUSSIAN) {
                            r = classifierRandom.nextGaussian();
                            w[i][j] = r;
                        }
                    }
                }
            }

            double vals[] = new double[projectionLength + 1];

            // copies the values in the original instance with the exception of the class
            double o[] = getDoubleArrayWithoutClass(instnc);
            for (int i = 0; i < projectionLength; i++){
                double v = 0;
                for(int k = 0; k < w[i].length; k++){
                    v += w[i][k] * o[k];
                }
                vals[i] = (1.0 / Math.sqrt(projectionLength)) * v;
            }

            Instance projected = new DenseInstance(1.0, vals);
            projected.setDataset(projectedHeader);
            projected.setClassValue(instnc.classValue());
            return projected;
        }

        private double[] getDoubleArrayWithoutClass(Instance instnc) {
            double a[] = new double[instnc.numAttributes() - 1];
            int pos = 0;
            for(int i = 0; i < instnc.numAttributes(); i++){
                if(i != instnc.classIndex()){
                    a[pos] = instnc.value(i);
                    pos++;
                }
            }
            return a;
        }
    }
}
