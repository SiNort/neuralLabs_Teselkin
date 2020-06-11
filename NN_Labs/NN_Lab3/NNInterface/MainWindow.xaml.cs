using SVMLibrary;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;

namespace NNInterface
{
    /// <summary>
    /// Логика взаимодействия для MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        List<Data> data = new List<Data>();
        Pen pen = new Pen(Brushes.Black, 2);
        CoreSVM machine;
        double[] statistic;

        public MainWindow()
        {
            InitializeComponent();
            AddData();
            Learn();
        }

        private void Button_Click(object sender, RoutedEventArgs e)
        {
            Learn();
        }

        private void Learn()
        {
            machine = new CoreSVM(new PolynomicalFunction(2), 25);
            var learn = new SequentialMinimalOptimization(machine, data.ToArray());
            statistic = learn.Run();
            DisplayGraphic();
        }

        private void DisplayGraphic()
        {
            int i = 0;
            graphic.Children.Clear();
            foreach (var item in statistic)
            {
                Rectangle rectangle = new Rectangle();
                rectangle.Fill = new SolidColorBrush(Colors.LightGray);
                rectangle.Height = Math.Abs(item) * 5;
                rectangle.Width = 40;
                rectangle.Margin = new Thickness(rectangle.Width * i, graphic.Height - rectangle.Height, 0, 0);
                graphic.Children.Add(rectangle);
                i++;
            }
        }

        private double[][] ReadFile(string path)
        {
            List<double[]> res = new List<double[]>();
            using (StreamReader sr = new StreamReader(path, System.Text.Encoding.Default))
            {
                string line;
                while ((line = sr.ReadLine()) != null)
                {
                    res.Add(line.Split(',').Select(n => Convert.ToDouble(n)).ToArray());
                }
            }
            return res.ToArray();
        }

        private void ChangeCollor_Click(object sender, RoutedEventArgs e)
        {
            if (((Button)sender).Background == Brushes.Black)
            {
                ((Button)sender).Background = Brushes.LightGray;
            }
            else
            {
                ((Button)sender).Background = Brushes.Black;
            }
        }

        private void Result_Button(object sender, RoutedEventArgs e)
        {
            double[] startData = new double[25];
            for (int i = 0; i < userInput.Children.Count; i++)
            {
                if (((Button)userInput.Children[i]).Background == Brushes.Black)
                {
                    startData[i] = 1;
                }
                else
                {
                    startData[i] = 0;
                }
            }
            Data data = new Data(0, startData);
            double output = machine.Compute(data.Inputs);
            string res = output > 0 ? "9" : "0";
            result.Text = res;
        }

        private void AddData()
        {
            double[][] trainData = ReadFile(@"C:\Users\saymo\source\repos\NN_train.txt");
            double[] result = ReadFile(@"C:\Users\saymo\source\repos\NN_results.txt")[0];

            for(int i = 0; i < trainData.Length; i++)
            {
                if(result[i] == 1)
                {
                    data.Add(new Data(1, trainData[i]));
                }
                else
                {
                    data.Add(new Data(-1, trainData[i]));
                }
            }
        }
    }
}
