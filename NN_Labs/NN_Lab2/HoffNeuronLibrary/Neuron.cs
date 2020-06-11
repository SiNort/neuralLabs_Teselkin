using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace HoffNeuronLibrary
{
    public class Neuron
    {
        public double Error { get; set; }
        public double Value { get; set; }
        public double Weight { get; set; }
        public Neuron[] Neurons { get; set; }

        public Neuron()
        { 
            Weight = 1;
        }

        public Neuron(Neuron[] neurons)
        {
            Neurons = neurons;
            Weight = 0;
        }

        public void SetValue(double value)
        {
            Value = value;
        }

        public void CalcError(int expValue)
        {
            Error = expValue - Value;
        }

        public void CalcOutput()
        {
            double summator = 0;
            for (int i = 0; i < Neurons.Length; i++)
            {
                summator += Neurons[i].Value * Neurons[i].Weight;
            }
            Value = 1 / (1 + Math.Exp(-summator));
        }

        public void HoffCorrection(double error)
        {
            Weight = Weight + Value * 0.5 * error;
        }
    }
}
