using SVMLibrary;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace NN_Lab3
{
    public partial class Form1 : Form
    {
        List<Data> data = new List<Data>();
        Pen pen = new Pen(Color.Black);
        CoreSVM machine;
        double[] statistic;

        public Form1()
        {
            InitializeComponent();
        }

        private void button1_Click(object sender, EventArgs e)
        {
            if(data.Count == 0) AddData();
            machine = new CoreSVM(new PolynomicalFunction(2), 25);
            var learn = new SequentialMinimalOptimization(machine, data.ToArray());
            statistic = learn.Run();
            pictureBox1.Paint += PictureBox1_Paint;
        }

        private void PictureBox1_Paint(object sender, PaintEventArgs e)
        {
            int x = 0;
            for (int i = 0; i <= statistic.Length; i++)
            {
                e.Graphics.DrawLine(pen, x, 10 * (float)statistic[i], x + 10, (float)statistic[i + 1]);
                x += 10;
            }
        }

        private void Panel1_Paint(object sender, PaintEventArgs e)
        {
            int x = 0;
            for (int i = 0; i <= statistic.Length; i++)
            {
                e.Graphics.DrawLine(pen, x, 10 * (float)statistic[i], x + 10, (float)statistic[i + 1]);
                x += 10;
            }
        }

        private void AddData()
        {
            data.Add(new Data(1, new double[] {
                0,0,1,1,1,
                0,0,1,0,1,
                0,0,1,1,1,
                0,0,0,0,1,
                0,0,1,1,1
            }));
            data.Add(new Data(1, new double[] {
                0,0,1,0,0,
                0,1,0,1,0,
                0,0,1,1,0,
                0,0,0,1,0,
                0,0,0,1,0
            }));
            data.Add(new Data(1, new double[] {
                1,1,1,0,0,
                1,0,1,0,0,
                1,1,1,0,0,
                0,1,0,0,0,
                1,0,0,0,0
            }));
            data.Add(new Data(1, new double[] {
                0,1,1,1,0,
                0,1,0,1,0,
                0,1,1,1,0,
                0,0,0,1,0,
                0,0,1,1,0
            }));
            data.Add(new Data(1, new double[] {
                0,0,0,1,1,
                0,0,1,0,1,
                0,0,0,1,1,
                0,0,0,0,1,
                0,0,0,0,1
            }));
            data.Add(new Data(1, new double[] {
                0,1,0,0,0,
                1,0,1,0,0,
                0,1,1,0,0,
                0,1,0,0,0,
                1,0,0,0,0
            }));
            data.Add(new Data(-1, new double[] {
                0,0,1,0,0,
                0,1,0,1,0,
                0,1,0,1,0,
                0,1,0,1,0,
                0,0,1,0,0
            }));
            data.Add(new Data(-1, new double[] {
                0,0,1,1,1,
                0,0,1,0,1,
                0,0,1,0,1,
                0,0,1,0,1,
                0,0,1,1,1
            }));
            data.Add(new Data(-1, new double[] {
                0,0,0,0,0,
                0,1,0,0,0,
                1,0,1,0,0,
                1,0,1,0,0,
                0,1,0,0,0
            }));
            data.Add(new Data(-1, new double[] {
                0,0,0,0,0,
                1,1,0,0,0,
                1,0,1,0,0,
                1,0,1,0,0,
                0,1,1,0,0
            }));
            data.Add(new Data(-1, new double[] {
                0,0,0,0,0,
                0,0,0,1,0,
                0,0,1,0,1,
                0,0,1,0,1,
                0,0,0,1,0
            }));
            data.Add(new Data(-1, new double[] {
                0,0,0,0,0,
                0,1,1,1,0,
                0,1,0,1,0,
                0,1,1,0,0,
                0,0,0,0,0
            }));
        }
    }
}
