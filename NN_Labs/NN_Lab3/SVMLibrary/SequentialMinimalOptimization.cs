using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SVMLibrary
{
    public class SequentialMinimalOptimization
    {
        private static Random random = new Random();

        public double Complexity { get; set; }
        public bool UseComplexityHeuristic { get; set; }
        public double Epsilon { get; set; }
        public double Tolerance { get; set; }

        private double[][] inputs;
        private int[] outputs;

        private SupportVectorMachine machine;
        private IFunction function;
        private double[] alpha;
        private double bias;

        private double[] errors;

        public SequentialMinimalOptimization(SupportVectorMachine machine, Data[] data)
        {
            this.machine = machine;

            CoreSVM csvm = machine as CoreSVM;
            function = (csvm != null) ? csvm.Function : new LinearFunction();

            Complexity = 1.0;
            Tolerance = 0.001;
            Epsilon = 0.001;

            inputs = new double[data.Length][];
            outputs = new int[data.Length];

            for (int i = 0; i < data.Length; i++)
            {
                inputs[i] = new double[data[i].Inputs.Length];
                for (int j = 0; j < data[i].Inputs.Length; j++)
                {
                    inputs[i][j] = data[i].Inputs[j];
                }
                outputs[i] = (int)data[i].Expected;
            }
        }

        public double[] Run()
        {
            int N = inputs.Length;

            if (UseComplexityHeuristic)
                Complexity = ComputeComplexity();

            this.alpha = new double[N];

            this.errors = new double[N];

            int numChanged = 0;
            int examineAll = 1;

            List<double> changes = new List<double>();

            while (numChanged > 0 || examineAll > 0)
            {
                numChanged = 0;
                if (examineAll > 0)
                {
                    for (int i = 0; i < N; i++)
                        numChanged += ExamineExample(i);
                }
                else
                {
                    for (int i = 0; i < N; i++)
                        if (alpha[i] != 0 && alpha[i] != Complexity)
                            numChanged += ExamineExample(i);
                }

                if (examineAll == 1)
                    examineAll = 0;
                else if (numChanged == 0)
                    examineAll = 1;

                changes.Add(numChanged);
            }


            List<int> indices = new List<int>();
            for (int i = 0; i < N; i++)
            {
                if (alpha[i] > 0) indices.Add(i);
            }

            int vectors = indices.Count;
            machine.SupportVectors = new double[vectors][];
            machine.Weights = new double[vectors];
            for (int i = 0; i < vectors; i++)
            {
                int j = indices[i];
                machine.SupportVectors[i] = inputs[j];
                machine.Weights[i] = alpha[j] * outputs[j];
            }
            machine.Threshold = -bias;


            return changes.ToArray();
        }

        private int ExamineExample(int i2)
        {
            double[] p2 = inputs[i2];
            double y2 = outputs[i2];
            double alph2 = alpha[i2];

            double e2 = (alph2 > 0 && alph2 < Complexity) ? errors[i2] : Compute(p2) - y2;

            double r2 = y2 * e2;


            if (!(r2 < -Tolerance && alph2 < Complexity) && !(r2 > Tolerance && alph2 > 0))
                return 0;


            int i1 = -1; double max = 0;
            for (int i = 0; i < inputs.Length; i++)
            {
                if (alpha[i] > 0 && alpha[i] < Complexity)
                {
                    double error1 = errors[i];
                    double aux = System.Math.Abs(e2 - error1);

                    if (aux > max)
                    {
                        max = aux;
                        i1 = i;
                    }
                }
            }

            if (i1 >= 0 && Step(i1, i2)) return 1;


            int start = random.Next(inputs.Length);
            for (i1 = start; i1 < inputs.Length; i1++)
            {
                if (alpha[i1] > 0 && alpha[i1] < Complexity)
                    if (Step(i1, i2)) return 1;
            }
            for (i1 = 0; i1 < start; i1++)
            {
                if (alpha[i1] > 0 && alpha[i1] < Complexity)
                    if (Step(i1, i2)) return 1;
            }


            start = random.Next(inputs.Length);
            for (i1 = start; i1 < inputs.Length; i1++)
            {
                if (Step(i1, i2)) return 1;
            }
            for (i1 = 0; i1 < start; i1++)
            {
                if (Step(i1, i2)) return 1;
            }


            return 0;
        }

        private bool Step(int i1, int i2)
        {
            if (i1 == i2) return false;

            double[] p1 = inputs[i1];
            double alph1 = alpha[i1];
            double y1 = outputs[i1];

            double e1 = (alph1 > 0 && alph1 < Complexity) ? errors[i1] : Compute(p1) - y1;

            double[] p2 = inputs[i2];
            double alph2 = alpha[i2];
            double y2 = outputs[i2];

            double e2 = (alph2 > 0 && alph2 < Complexity) ? errors[i2] : Compute(p2) - y2;


            double s = y1 * y2;


            double L, H;
            if (y1 != y2)
            {
                L = Math.Max(0, alph2 - alph1);
                H = Math.Min(Complexity, Complexity + alph2 - alph1);
            }
            else
            {
                L = Math.Max(0, alph2 + alph1 - Complexity);
                H = Math.Min(Complexity, alph2 + alph1);
            }

            if (L == H) return false;

            double k11, k22, k12, eta;
            k11 = function.Function(p1, p1);
            k12 = function.Function(p1, p2);
            k22 = function.Function(p2, p2);
            eta = k11 + k22 - 2.0 * k12;

            double a1, a2;

            if (eta > 0)
            {
                a2 = alph2 - y2 * (e2 - e1) / eta;

                if (a2 < L) a2 = L;
                else if (a2 > H) a2 = H;
            }
            else
            {
                double L1 = alph1 + s * (alph2 - L);
                double H1 = alph1 + s * (alph2 - H);
                double f1 = y1 * (e1 + bias) - alph1 * k11 - s * alph2 * k12;
                double f2 = y2 * (e2 + bias) - alph2 * k22 - s * alph1 * k12;
                double Lobj = -0.5 * L1 * L1 * k11 - 0.5 * L * L * k22 - s * L * L1 * k12 - L1 * f1 - L * f2;
                double Hobj = -0.5 * H1 * H1 * k11 - 0.5 * H * H * k22 - s * H * H1 * k12 - H1 * f1 - H * f2;

                if (Lobj > Hobj + Epsilon) a2 = L;
                else if (Lobj < Hobj - Epsilon) a2 = H;
                else a2 = alph2;
            }

            if (Math.Abs(a2 - alph2) < Epsilon * (a2 + alph2 + Epsilon))
                return false;

            a1 = alph1 + s * (alph2 - a2);

            if (a1 < 0)
            {
                a2 += s * a1;
                a1 = 0;
            }
            else if (a1 > Complexity)
            {
                double d = a1 - Complexity;
                a2 += s * d;
                a1 = Complexity;
            }


            double b1 = 0, b2 = 0;
            double new_b = 0, delta_b;

            if (a1 > 0 && a1 < Complexity)
            {
                new_b = e1 + y1 * (a1 - alph1) * k11 + y2 * (a2 - alph2) * k12 + bias;
            }
            else
            {
                if (a2 > 0 && a2 < Complexity)
                {
                    new_b = e2 + y1 * (a1 - alph1) * k12 + y2 * (a2 - alph2) * k22 + bias;
                }
                else
                {
                    b1 = e1 + y1 * (a1 - alph1) * k11 + y2 * (a2 - alph2) * k12 + bias;
                    b2 = e2 + y1 * (a1 - alph1) * k12 + y2 * (a2 - alph2) * k22 + bias;
                    new_b = (b1 + b2) / 2;
                }
            }

            delta_b = new_b - bias;
            bias = new_b;


            double t1 = y1 * (a1 - alph1);
            double t2 = y2 * (a2 - alph2);

            for (int i = 0; i < inputs.Length; i++)
            {
                if (0 < alpha[i] && alpha[i] < Complexity)
                {
                    double[] point = inputs[i];
                    errors[i] +=
                            t1 * function.Function(p1, point) +
                            t2 * function.Function(p2, point) -
                            delta_b;
                }
            }

            errors[i1] = 0f;
            errors[i2] = 0f;


            alpha[i1] = a1;
            alpha[i2] = a2;


            return true;
        }

        public double ComputeError(double[][] inputs, int[] expectedOutputs)
        {
            int count = 0;
            for (int i = 0; i < inputs.Length; i++)
            {
                if(Math.Sign(Compute(inputs[i])) != Math.Sign(expectedOutputs[i]))
                {
                    count++;
                }
            }

            return (double)count / inputs.Length;
        }

        private double Compute(double[] point)
        {
            double sum = -bias;
            for(int i = 0; i < inputs.Length; i++)
            {
                if(alpha[i] > 0)
                {
                    sum += alpha[i] * outputs[i] * function.Function(inputs[i], point);
                }
            }
            return sum;
        }

        private double ComputeComplexity()
        {
            double sum = 0; 
            for (int i = 0; i < inputs.Length; i++)
                sum += function.Function(inputs[i], inputs[i]);
            return inputs.Length / sum;
        }
    }
}
