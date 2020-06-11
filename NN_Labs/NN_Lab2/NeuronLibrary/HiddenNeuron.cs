using System;
using System.Linq;

namespace NeuronLibrary
{
    public class HiddenNeuron : Neuron
    {
        public static Random rnd = new Random();
        public double Error { get; protected set; }
        public double[] Weights;
        public Neuron[] Neurons;
        public HiddenNeuron(params Neuron[] neurons)
        {
            Neurons = neurons;
            Weights = new double[Neurons.Length];
            InitWeights();
        }

        public void CalcOutput()
        {
            double summator = 0;
            for (int i = 0; i < Neurons.Length; i++)
            {
                summator += Neurons[i].OutputValue * Weights[i];
            }
            OutputValue = 1 / (1 + Math.Exp(-summator));
        }

        public void CalcError(params Neuron[] neurons)
        {
            double tempError = 0;
            for (int i = 0; i < neurons.Length; i++)
            {
                if (((HiddenNeuron)neurons[i]).Neurons.Contains(this))
                {
                    tempError += ((HiddenNeuron)neurons[i]).Error * ((HiddenNeuron)neurons[i]).Weights[((HiddenNeuron)neurons[i]).Neurons.ToList().IndexOf(this)];
                }
                else
                {
                    throw new ArgumentException("Wrong neuron layer");
                }
            }
            Error = tempError;
        }

        private void InitWeights()
        {
            for (int i = 0; i < Weights.Length; i++)
            {
                Weights[i] = rnd.NextDouble() * (0.5 + 0.5) - 0.5;
            }
        }

        public void ModifyWeights(double speed)
        {
            for (int i = 0; i < Weights.Length; i++)
            {
                Weights[i] = Weights[i] + speed * Error * Neurons[i].OutputValue * (1 / (1 + Math.Exp(-Neurons[i].OutputValue))) * (1 - (1 / (1 + Math.Exp(-Neurons[i].OutputValue))));
            }
        }

        public void HoffCorrection(double error, double speed)
        {
            for (int i = 0; i < Neurons.Length; i++)
            {
                Weights[i] = Weights[i] + Neurons[i].OutputValue * error * speed;
            }
        }
    }
}
