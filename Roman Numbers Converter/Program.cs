using System.Linq;
using System.Collections.Generic;
using System.Text;

namespace Roman_Numbers_Converter
{
    internal class Program
    {
        static void Main(string[] args)
        {
            //romanNumbers a = romanNumbers.XL;
            //Console.WriteLine($"Integral value of {a} is {(int)a}");  // output: Integral value of Autumn is 2

            //var b = (romanNumbers)1;
            //Console.WriteLine(b);  // output: Summer

            //var c = (romanNumbers)4;
            //var t = (romanNumbers)2 | (romanNumbers)5;

            //Console.WriteLine(t);  // output: 4

            //Console.WriteLine((int)b+(int)c);

            //var values = Enum.GetValues(typeof(romanNumbers));
            Console.WriteLine(ToRoman(1990));
            Console.WriteLine(FromRoman("MM"));



        }

        enum romanNumbers{
            I = 1,
            IV = 4,
            V = 5,
            IX = 9,
            X = 10,
            XL = 40,
            L = 50,
            XC = 90,
            C = 100,
            CD = 400,
            D = 500,
            CM = 900,
            M = 1000
                

        }
        public static string ToRoman(int n)
        {
            var values = Enum.GetValues(typeof(romanNumbers));
            StringBuilder rN = new StringBuilder();


            for (int i = values.Length-1; i >= 0 ;)
            {
                romanNumbers currentValue = (romanNumbers)values.GetValue(i);
                if (n - (int)currentValue >= 0)
                {
                    n -= (int)currentValue;
                    rN.Append(currentValue.ToString());
                    //Console.WriteLine((int)currentValue);
                }
                else i--;
            }
            return rN.ToString();
        }

        public static int FromRoman(string romanNumeral)
        {
            int number = 0;
            var values = Enum.GetValues(typeof(romanNumbers));
            for(int i = 0; i < romanNumeral.Length; )
            {
                if (i + 1 < romanNumeral.Length && Enum.TryParse<romanNumbers>(string.Concat(romanNumeral[i], romanNumeral[i + 1]), out romanNumbers result)) i = i + 2;
                else 
                { 
                    Enum.TryParse<romanNumbers>(string.Concat(romanNumeral[i]), out  result);
                    i++;
                }
                    number += (int)result;
            }
            return number;
        }
    }


}