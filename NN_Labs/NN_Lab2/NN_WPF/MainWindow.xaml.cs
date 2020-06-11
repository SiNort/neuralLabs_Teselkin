using NeuronLibrary;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using System.Windows.Shapes;

namespace NN_WPF
{
    /// <summary>
    /// Логика взаимодействия для MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        int[][] trainData;
        int[][] testData;
        int[] result;
        NN_Hoff backProp;
        NN_Hoff hoff;
        public MainWindow()
        {
            InitializeComponent();
            trainData = ReadFile(@"C:\Users\saymo\source\repos\NN_train.txt");
            testData = ReadFile(@"C:\Users\saymo\source\repos\NN_test.txt");
            result = ReadFile(@"C:\Users\saymo\source\repos\NN_results.txt")[0];
            backProp = new NN_Hoff(trainData, testData, result);
            DisplayGraphic(backProp.Learn(0.01), graphicCanvas);
            hoff = new NN_Hoff(trainData, testData, result);
            DisplayGraphic(hoff.LearnHoff(), graphicCanvas1);
            //hoff.RegisterHandler(DisplayData);
            DisplayTests();
        }

        private void DisplayTests()
        {
            double[] hoffTestData = hoff.Test();
            double[] propTestData = backProp.Test();
            for(int i = 0; i < testsArea.Children.Count; i++)
            {
                for(int j = 0; j < ((Grid)testsArea.Children[i]).Children.Count; j++)
                {
                    if(j < 25)
                    {
                        if(testData[i][j] == 1)
                        {
                            ((Rectangle)((Grid)testsArea.Children[i]).Children[j]).Fill = new SolidColorBrush(Colors.Black);
                        }
                    }
                    else
                    {
                        if(j == 25)
                        {
                            if(propTestData[i] > 0.9)
                            {
                                ((Label)((Grid)testsArea.Children[i]).Children[j]).Content = "9";
                            }
                            else
                            {
                                ((Label)((Grid)testsArea.Children[i]).Children[j]).Content = "0";
                            }
                        }
                        else
                        {
                            if (hoffTestData[i] > 0.6)
                            {
                                ((Label)((Grid)testsArea.Children[i]).Children[j]).Content = "9";
                            }
                            else
                            {
                                ((Label)((Grid)testsArea.Children[i]).Children[j]).Content = "0";
                            }
                        }
                    }
                }
            }
        }
        
        private int[][] ReadFile(string path)
        {
            List<int[]> res = new List<int[]>();
            using (StreamReader sr = new StreamReader(path, System.Text.Encoding.Default))
            {
                string line;
                while ((line = sr.ReadLine()) != null)
                {
                    res.Add(line.Split(',').Select(n => Convert.ToInt32(n)).ToArray());
                }
            }
            return res.ToArray();
        }

        private void DisplayGraphic(double[] data, Canvas canvas)
        {
            int i = 0;
            canvas.Children.Clear();
            foreach (var item in data)
            {
                Rectangle rectangle = new Rectangle();
                rectangle.Fill = new SolidColorBrush(Colors.LightGray);
                rectangle.Height = Math.Abs(item) * 80;
                rectangle.Width = 3;
                rectangle.Margin = new Thickness(rectangle.Width * i, canvas.Height - rectangle.Height, 0, 0);
                canvas.Children.Add(rectangle);
                i++;
            }
        }

        private void Result_Button(object sender, RoutedEventArgs e)
        {
            int[] startData = new int[25];
            for(int i = 0; i < userInput.Children.Count; i++)
            {
                if(((Button)userInput.Children[i]).Background == Brushes.Black)
                {
                    startData[i] = 1;
                }
                else
                {
                    startData[i] = 0;
                }
            }

            if (hoff.GetResult(startData) > 0.6)
            {
                hoffResult.Text = "9";
            }
            else
            {
                hoffResult.Text = "0";
            }

            if (backProp.GetResult(startData) > 0.9)
            {
                backpropResult.Text = "9";
            }
            else
            {
                backpropResult.Text = "0";
            }

            //hoffResult.Text = hoff.GetResult(startData).ToString();
            //backpropResult.Text = backProp.GetResult(startData).ToString();
        }

        private void ChangeCollor_Click(object sender, RoutedEventArgs e)
        {
            if(((Button)sender).Background == Brushes.Black)
            {
                ((Button)sender).Background = Brushes.LightGray;
            }
            else
            {
                ((Button)sender).Background = Brushes.Black;
            }
        }

        private void Relearch_Click(object sender, RoutedEventArgs e)
        {
            backProp = new NN_Hoff(trainData, testData, result);
            DisplayGraphic(backProp.Learn(0.01), graphicCanvas);
            hoff = new NN_Hoff(trainData, testData, result);
            DisplayGraphic(hoff.LearnHoff(), graphicCanvas1);
            DisplayTests();
        }
    }
}
