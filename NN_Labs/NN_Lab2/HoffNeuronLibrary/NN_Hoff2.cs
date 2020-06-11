using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace HoffNeuronLibrary
{
    public class NN_Hoff2
    {
        public int[] Result;
        public int[][] TrainData;
        public int[][] TestData;

        public Neuron[] InputLayer = new Neuron[25];
        public Neuron[] HiddenLayer1 = new Neuron[8];
        public Neuron[] HiddenLayer2 = new Neuron[8];
        public Neuron OutputNeuron;

        public delegate void AccountStateHandler(string message);
        AccountStateHandler _del;

        public void RegisterHandler(AccountStateHandler del)
        {
            _del = del;
        }

        public NN_Hoff2(int[][] trainData, int[][] testData, int[] result)
        {
            Result = result;
            TrainData = trainData;
            TestData = testData;
            InitLayers();
        }

        private void InitLayers()
        {
            for(int i = 0; i < InputLayer.Length; i++)
            {
                InputLayer[i] = new Neuron();
            }

            for(int i = 0; i < HiddenLayer1.Length; i++)
            {
                HiddenLayer1[i] = new Neuron(InputLayer);
            }

            for (int i = 0; i < HiddenLayer2.Length; i++)
            {
                HiddenLayer2[i] = new Neuron(HiddenLayer1);
            }
            OutputNeuron = new Neuron(HiddenLayer2);
        }

        public void Learn()
        {
            for (int f = 0; f < 50; f++)
            {
                for (int iteration = 0; iteration < TrainData.Length; iteration++)
                {
                    for (int i = 0; i < InputLayer.Length; i++)
                    {
                        InputLayer[i].SetValue(TrainData[iteration][i]);
                    }

                    for (int i = 0; i < HiddenLayer1.Length; i++)
                    {
                        HiddenLayer1[i].CalcOutput();
                    }
                    for (int i = 0; i < HiddenLayer2.Length; i++)
                    {
                        HiddenLayer2[i].CalcOutput();
                    }
                    OutputNeuron.CalcOutput();
                    OutputNeuron.CalcError(Result[iteration]);
                    _del("Value: " + OutputNeuron.Value + " Error: " + OutputNeuron.Error);

                    for (int i = 0; i < HiddenLayer2.Length; i++)
                    {
                        HiddenLayer2[i].HoffCorrection(OutputNeuron.Error);
                    }

                    for (int i = 0; i < HiddenLayer1.Length; i++)
                    {
                        HiddenLayer1[i].HoffCorrection(OutputNeuron.Error);
                    }

                    for (int i = 0; i < InputLayer.Length; i++)
                    {
                        InputLayer[i].HoffCorrection(OutputNeuron.Error);
                    }
                }
            }
        }
    }
}
