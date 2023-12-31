using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text.RegularExpressions;
using System.Threading.Channels;


//HOMEWORK
//1.Find the greatest common divisor of two positive integers.
//The inputs x and y are always greater or equal to 1, so the greatest common divisor will always be an integer that is also greater or equal to 1.
//2. The sum of the primes below or equal to 10 is 2 + 3 + 5 + 7 = 17. Find the sum of all the primes below or equal to the number passed in.
//EXTRA:
//Do using recursion
namespace Homework5
{
    class Program
    {
        public static int Persistence(int n) => n < 10 ? 0 : 1 + Persistence($"{n}".Aggregate(1, (a, b) => a * (b - 48)));
        public static int Persistence(long n)
        {

            return n < 10 ? 0 : 1 + Persistence($"{n}".Aggregate(1, (a, b) => a * (b - 48)));
        }

        public static int FindEvenIndex(int[] arr)
        {
            //foreach(int i in arr){
            for (int i = 0; i < arr.Length; i++) {

                int left, right;
                left = arr.Take(i).Aggregate(0, (a, b) => a + b);
                right = arr.Skip(i + 1).Aggregate(0, (a, b) => a + b);
                Console.WriteLine("Left side = " + left + " right side = " + right);
                // if (arr.Take(i).Aggregate(0, (a, b) => a + b) == arr.Skip(i+1).Aggregate(0,(a,b) => a+b))
                //   return i;

                // arr.Take(i).Aggregate(0, (a, b) => a + b) == arr.Skip(i + 1).Aggregate(0, (a, b) => a + b) ? return i :

            }
            return -1;
        }




        //public static int MaxSequence(int[] arr)
        //{
        //    //TODO : create code
        //    int maxSum = 0, sum = 0, prevSum = 0;
        //    for (int i = 0; i < arr.Length; i++)
        //    {
        //        for (int j = i; j <= arr.Length; j++)
        //        {
        //            sum = arr.Skip(i).Take(j-i).Sum();
        //            if(sum > maxSum) maxSum = sum;
        //            arr.Skip(i).Take(j - i).ToList().ForEach(c => Console.WriteLine(c));
        //            Console.WriteLine(sum +"  "+ maxSum);
        //            //if (sum < prevSum) { }
        //        }

        //    }
        //    return maxSum <= 0 ? 0 : maxSum;
        //}

        public static int MaxSequence(int[] arr)
        {
            int max = 0, res = 0, sum = 0;
            foreach (var item in arr)
            {
                sum += item;
                max = sum > max ? max : sum;

                res = res > sum - max ? res : sum - max;

                if (sum > max) max = max;
                else max = sum;

                if (res > sum - max) res = res;
                else res = sum - max;

            }
            return res;
        }


        public static int MaxSequence(int[] arr)
        {
            int max = 0, current = 0;
            foreach (int item in arr)
            {
                current += item;
                if (item > current) current = item;
                if (current > max) max = current;

            }

            return max;

        }

        //public static IEnumerable<T> UniqueInOrder<T>(IEnumerable<T> iterable)
        //{
        //    //string chars = iterable.ToString();   
        //    List<char> sequence = new List<char>();
        //    //List<T> chars = (List<T>)iterable;

        //    if (iterable.Count() == 0) return (IEnumerable<T>)sequence;
        //    sequence.Add(iterable.ToString()[0]);
        //    //iterable.ToString();
        //    for (int i = 1; i < iterable.Count(); i++)
        //    {
        //        if (iterable.ToString()[i] != iterable.ToString()[i - 1])
        //        {
        //            sequence.Add(iterable.ToString()[i]);
        //            Console.WriteLine(string.Concat(sequence));
        //        }
        //    }
        //    //foreach(var i in iterable){
        //    return (IEnumerable<T>)sequence;
        //}

        public static IEnumerable<T> UniqueInOrder<T>(IEnumerable<T> iterable)
        {
            List<T> sequence = new List<T>();
            Console.WriteLine("!!!" + string.Concat(iterable));
            if (iterable.Count() == 0) return sequence;
            sequence.Add(iterable.FirstOrDefault());

            for (int i = 1; i < iterable.Count(); i++)
            {
                if (!iterable.ElementAt(i).Equals(iterable.ElementAt(i - 1))) 
                {
                    sequence.Add(iterable.ElementAt(i));
                    Console.WriteLine(string.Concat(sequence));
                }
            }
            return sequence;
        }



        static void Main(string[] args)
        {
            //UniqueInOrder(new List<int> { 1, 2, 2 });
            Console.WriteLine("Max Sequence: " + MaxSequence(new[] { -2, 1, -3, 4, -1, 2, 1, -5, 4 }));

            //string h = new List<int> { 1, 3, 5 }.ToString();

            Console.WriteLine(UniqueInOrder(new String("fdussjjgd").ToString()));
            Console.WriteLine(UniqueInOrder(new List<int> { 1,2,2,3,3}));
            //Console.WriteLine(UniqueInOrder(new List<string> { "abc", null, null, "bcc" }));



            Console.WriteLine("It Is:" + FindEvenIndex(new [] { 1, 100, 50, -51, 1, 1 }));

            Console.WriteLine(Persistence(0l));
            Console.WriteLine(Persistence(5l));
            Console.WriteLine(Persistence(25));
            Console.WriteLine(Persistence(999));



            string input = "dajdhazsjdhaiughzz";
            //Num = string.Concat(Num.OrderByDescending(x => x));

                if (input.ToLower().Count( x=> x == 'o') == input.ToLower().Count(x => x == 'x'))
                            Console.WriteLine("Yes");
                else Console.WriteLine( "No");

            string n = "4";
            
            Regex reg = new Regex("[^a-m]");
            Console.WriteLine(input.Count() +"  "+reg.Matches(input).Count());



            $"{n}".Aggregate(1,(a,b)=>a*b);



        long total = 1, it = 0;
            // your code
            //if (n < 10) return 0;
            while (true)
            {
                total = 1;
                foreach (var digit in n.ToString())
                {
                    total = total * Convert.ToInt64(digit.ToString());
                }
                it++;
                if (total < 10) { Console.WriteLine(it.ToString()); break; }
                n = total.ToString();

            }

            //long n = 999;
            foreach (var digit in n)
            {

            }

            //input = input.ToLower();
            //var a = if (input.Count(x => x == 'o') == input.Count(x => x == 'x')) ;
            //bool d = input.Count(x => x == 'o') == input.Count(x => x == 'x');
            //  Console.WriteLine( );


            int A = 0, B = 0, C = 0;
            int d = 1;
            try {
                Console.WriteLine("Find the greatest common divisor of two positive integers:");
                Console.WriteLine("Input A (Positive Integer only)"); A = int.Parse(Console.ReadLine().ToString());
                Console.WriteLine("Input B (Positive Integer only)"); B = int.Parse(Console.ReadLine().ToString());
                Console.WriteLine("Find the sum of all the primes below or equal to the number passed in:");
                Console.WriteLine("Input number to summarize primes below or equal to (Positive Inreger only)"); C = int.Parse(Console.ReadLine().ToString());
            }
            catch (FormatException) {
                Console.WriteLine("Your value is unexceptable. GOODBAY!");
                Environment.Exit(0); }
            if (A == 0 || B == 0) { Console.WriteLine("Zero enter"); Environment.Exit(0); }

            //Vprava 1
            //Rekursiinyi poshuk naibilshogo spilnogo dilnyka
            Divsion(A, B);
            void Divsion(int x, int y)
            {
                if (x % 2 == 0 && y % 2 == 0) { d *= 2; Divsion(x / 2, y / 2); }
                else if (x % 3 == 0 && y % 3 == 0) { d *= 3; Divsion(x / 3, y / 3); }
                else if (x % 5 == 0 && y % 5 == 0) { d *= 5; Divsion(x / 5, y / 5); }
                else if (x % 7 == 0 && y % 7 == 0) { d *= 7; ; Divsion(x / 7, y / 7); }
            }
            Console.WriteLine("\nDivider =" + d);


            //Vprava 2
            List<int> L = new List<int>();
            for (int i = 2; i <= C; i++) L.Add(i);
            Primes(ref L, 2);
            int Sum = 0;



            //Rekusiinyi poshuk prostyh chysel
            void Primes(ref List<int> numbers, int d1)  {
                for(int i = numbers.Count-1; i >=2; i--)
                    if (numbers[i] % d1 == 0 && numbers[i] != d1) numbers.RemoveAt(i);
                d1 += 1;
                if (d1 <= C / 2) Primes (ref numbers, d1);//Rekursiya
            }

            foreach (var l in L) Sum += l;
            Console.WriteLine("Sum of primes =" + Sum);

        }
    }
}
