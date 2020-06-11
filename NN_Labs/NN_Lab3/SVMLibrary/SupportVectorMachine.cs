using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SVMLibrary
{
    public class SupportVectorMachine
    {
        public int Inputs { get; private set; }
        public double[][] SupportVectors { get; set; }
        public double[] Weights { get; set; }
        public double Threshold { get; set; }

        public SupportVectorMachine(int inputsNum)
        {
            Inputs = inputsNum;
        }

        public virtual double Compute(double[] input)
        {
            double sum = Threshold;
            for(int i = 0; i < SupportVectors.Length; i++)
            {
                double p = 0;
                for(int j = 0; j < input.Length; j++)
                {
                    p += SupportVectors[i][j] * input[j];
                }
                sum += Weights[i] * p;
            }
            return sum;
        }

        public double[] Compute(Data[] data)
        {
            double[] output = new double[data.Length];
            double[][] inputs = new double[data.Length][];

            for(int i = 0; i < data.Length; i++)
            {
                inputs[i] = new double[data[i].Inputs.Length];
                for (int j = 0; j < data[i].Inputs.Length; j++)
                {
                    inputs[i][j] = data[i].Inputs[j];
                }
            }

            for (int i = 0; i < inputs.Length; i++)
            {
                output[i] = Compute(inputs[i]);
            }
            return output;
        }
    }

}
