using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SVMLibrary
{
    public class CoreSVM : SupportVectorMachine
    {
        public IFunction Function { get; set; }
        public CoreSVM(IFunction function, int inputs) : base(inputs)
        {
            Function = function;
        }

        public override double Compute(double[] inputs)
        {
            double sum = Threshold;
            
            for(int i = 0; i < SupportVectors.Length; i++)
            {
                sum += Weights[i] * Function.Function(SupportVectors[i], inputs);
            }

            return sum;
        }
    }
}
