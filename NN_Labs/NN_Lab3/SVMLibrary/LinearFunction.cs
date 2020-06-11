using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SVMLibrary
{
    public class LinearFunction : IFunction
    {
        public double Constant { get; set; }

        public LinearFunction(double constant = 0)
        {
            Constant = constant;
        }

        public double Function(double[] x, double[] y)
        {
            double sum = Constant;
            for(int i = 0; i < y.Length; i++)
            {
                sum += x[i] * y[i];
            }
            return sum;
        }
    }
}
