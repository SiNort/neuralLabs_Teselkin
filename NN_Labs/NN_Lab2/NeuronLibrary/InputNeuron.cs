using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuronLibrary
{
    public class InputNeuron : Neuron
    {
        public void SetStartValue(double startValue)
        {
            OutputValue = startValue;
        }
    }
}
