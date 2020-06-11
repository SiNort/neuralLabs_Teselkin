using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuronLibrary
{
    public class NNHoff
    {
        public delegate void AccountStateHandler(string message);
        AccountStateHandler _del;

        public void RegisterHandler(AccountStateHandler del)
        {
            _del = del;
        }

        private double speed = 0.5;
        private double biods = 1;
        private Neuron[] neurons = new Neuron[7];
        private int[] x1 = new int[] { 1, 1, 0, 0 };
        private int[] x2 = new int[] { 0, 1, 1, 0 };
        private int[] y = new int[] { 1, 1, 1, 1 };

        public NNHoff()
        {
            neurons[0] = new InputNeuron();
            neurons[1] = new InputNeuron();

            neurons[2] = new BaseNeuron(2);
            neurons[3] = new BaseNeuron(2);
            neurons[4] = new BaseNeuron(2);
            neurons[5] = new BaseNeuron(2);
            neurons[6] = new BaseNeuron(2);
        }

        private void Learn(int iterations)
        {
            for(int i = 0; i < iterations; i++)
            {
                for(int j = 0; j < 4; j++)
                {
                    neurons[0].OutputSynapse.Value = x1[j];
                    neurons[1].OutputSynapse.Value = x2[j];

                    _del($"");
                }
            }
        }
    }
}
