using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuronLibrary
{
    public class OutputNeuron : HiddenNeuron
    {
        public OutputNeuron(params Neuron[] neurons) : base(neurons)
        {
            Error = 1;
        }
        public void CalcError(double expectedOutput)
        {
            Error = expectedOutput - OutputValue;
        }
    }
}
