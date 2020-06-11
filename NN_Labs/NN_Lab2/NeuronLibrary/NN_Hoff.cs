using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuronLibrary
{
    public class NN_Hoff
    {
        public double speed = 0.1;
        public int[] Result;
        public int[][] TrainData;
        public int[][] TestData;
        public delegate void AccountStateHandler(string message);
        AccountStateHandler _del;

        public void RegisterHandler(AccountStateHandler del)
        {
            _del = del;
        }

        public InputNeuron[] inputLayer = new InputNeuron[25];
        public HiddenNeuron[] hiddenLayer1 = new HiddenNeuron[8];
        public HiddenNeuron[] hiddenLayer2 = new HiddenNeuron[8];
        public OutputNeuron outputNeuron = new OutputNeuron();

        public NN_Hoff(int[][] trainData, int[][] testData, int[] result)
        {
            Result = result;
            TrainData = trainData;
            TestData = testData;
            InitLayers();
        }

        public void InitLayers()
        {
            for (int i = 0; i < inputLayer.Length; i++)
            {
                inputLayer[i] = new InputNeuron();
            }

            for (int i = 0; i < hiddenLayer1.Length; i++)
            {
                hiddenLayer1[i] = new HiddenNeuron(inputLayer);
            }

            for (int i = 0; i < hiddenLayer2.Length; i++)
            {
                hiddenLayer2[i] = new HiddenNeuron(hiddenLayer1);
            }
            outputNeuron = new OutputNeuron(hiddenLayer2);
        }

        public double[] Learn(double accurancy)
        {
            List<double> result = new List<double>();
            while (Math.Abs(outputNeuron.Error) > accurancy)
            {
                for(int iteration = 0; iteration < TrainData.Length; iteration++)
                {
                    for (int i = 0; i < inputLayer.Length; i++)
                    {
                        inputLayer[i].SetStartValue(TrainData[iteration][i]);
                    }
                    for (int i = 0; i < hiddenLayer1.Length; i++)
                    {
                        hiddenLayer1[i].CalcOutput();
                    }
                    for (int i = 0; i < hiddenLayer2.Length; i++)
                    {
                        hiddenLayer2[i].CalcOutput();
                    }
                    outputNeuron.CalcOutput();
                    outputNeuron.CalcError(Result[iteration]);
                    if (Math.Abs(outputNeuron.Error) > accurancy) //(Math.Abs(outputNeuron.Error) > accurancy)
                    {
                        for (int i = 0; i < hiddenLayer2.Length; i++)
                        {
                            hiddenLayer2[i].CalcError(outputNeuron);
                        }

                        for (int i = 0; i < hiddenLayer1.Length; i++)
                        {
                            hiddenLayer1[i].CalcError(hiddenLayer2);
                        }

                        for (int i = 0; i < hiddenLayer1.Length; i++)
                        {
                            hiddenLayer1[i].ModifyWeights(speed);
                        }

                        for (int i = 0; i < hiddenLayer2.Length; i++)
                        {
                            hiddenLayer2[i].ModifyWeights(speed);
                        }
                        outputNeuron.ModifyWeights(speed);
                    }
                    else
                        break;
                }
                result.Add(outputNeuron.Error);
            }
            return result.ToArray();
        }

        public double[] LearnHoff()
        {
            List<double> result = new List<double>();
            for (int f = 0; f < 500; f++)
            {
                for (int iteration = 0; iteration < TrainData.Length; iteration++)
                {
                    for (int i = 0; i < inputLayer.Length; i++)
                    {
                        inputLayer[i].SetStartValue(TrainData[iteration][i]);
                    }

                    for (int i = 0; i < hiddenLayer1.Length; i++)
                    {
                        hiddenLayer1[i].CalcOutput();
                    }
                    for (int i = 0; i < hiddenLayer2.Length; i++)
                    {
                        hiddenLayer2[i].CalcOutput();
                    }
                    outputNeuron.CalcOutput();
                    outputNeuron.CalcError(Result[iteration]);

                    outputNeuron.HoffCorrection(outputNeuron.Error, speed);

                    for (int i = 0; i < hiddenLayer2.Length; i++)
                    {
                        hiddenLayer2[i].HoffCorrection(outputNeuron.Error, speed);
                    }

                    for (int i = 0; i < hiddenLayer1.Length; i++)
                    {
                        hiddenLayer1[i].HoffCorrection(outputNeuron.Error, speed);
                    }
                }
                result.Add(outputNeuron.Error);
            }
            return result.ToArray();
        }

        public double GetResult(int[] data)
        {
            for (int i = 0; i < inputLayer.Length; i++)
            {
                inputLayer[i].SetStartValue(data[i]);
            }

            for (int i = 0; i < hiddenLayer1.Length; i++)
            {
                hiddenLayer1[i].CalcOutput();
            }

            for (int i = 0; i < hiddenLayer2.Length; i++)
            {
                hiddenLayer2[i].CalcOutput();
            }
            outputNeuron.CalcOutput();
            return outputNeuron.OutputValue;
        }

        public double[] Test()
        {
            double[] result = new double[TestData.Length];
            for (int iteration = 0; iteration < TestData.Length; iteration++)
            {
                for (int i = 0; i < inputLayer.Length; i++)
                {
                    inputLayer[i].SetStartValue(TestData[iteration][i]);
                }

                for (int i = 0; i < hiddenLayer1.Length; i++)
                {
                    hiddenLayer1[i].CalcOutput();
                }

                for (int i = 0; i < hiddenLayer2.Length; i++)
                {
                    hiddenLayer2[i].CalcOutput();
                }
                outputNeuron.CalcOutput();
                outputNeuron.CalcError(Result[iteration]);
                result[iteration] = outputNeuron.OutputValue;
            }
            return result;
        }
    }
}
