using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SVMLibrary
{
    public class Data
    {
        public double Expected { get; private set; }
        public double[] Inputs{ get; private set; }

        public Data(double expected, double[] inputs)
        {
            Expected = expected;
            Inputs = inputs;
        }
    }
}
