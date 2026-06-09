#include <iostream>
#include <windows.h>
#include <functional>
#include <algorithm>
#include <random>

using namespace std;

long long int mysum;

double func1(double a[], long long n) { // 平凡算法
    double sum = 0;
    for (int i = 0; i < n; i+=4) {
        sum += a[i];
        sum += a[i+1];
        sum += a[i+2];
        sum += a[i+3];
    }
    return sum;
}

double func2(double a[], long long n) {  // 两路链式
    double sum1 = 0, sum2 = 0;
    for (int i = 0; i < n; i += 4) {
        sum1 += a[i];
        sum1+= a[i+1];
        sum2 += a[i + 2];
        sum2 += a[i+3];
    }
    return sum1 + sum2;
}


double func3(double a[], long long halfN) {
    //long long halfN = n / 2;
    double sum1 = 0, sum2 = 0;
    for (int i = 0; i < halfN; i+=4) {
        sum1 += a[i];
        sum1+= a[i+1];
        sum1 += a[i + 2];
        sum1 += a[i+3];
    }
    for (int i = 0; i < halfN; i+=4) {
        sum2 += a[halfN+i];
        sum2+= a[halfN+i+1];
        sum2 += a[halfN+i + 2];
        sum2 += a[halfN+i+3];
    }
    return sum1 + sum2;
}

double func4(double a[], long long n) { // 递归
    long long m = n;
    double sum = 0;
    for (m = n; m > 1; m /= 2) {
        for (int i = 0; i < m / 2; i+=4) {
            a[i] = a[i*2] + a[i*2+1];
            a[i+1] = a[i*2+2] + a[i*2+3];
            a[i+2] = a[i*2+4] + a[i*2+5];
            a[i+3] = a[i*2+6] + a[i*2+7];
        }
    }
    sum = a[0];
    return sum;
}

double func5(double a[], long long n) {
    double sum = 0;
    for (int i = 0; i < n; i+=4) {
        sum += a[i] + a[i+1] + a[i+2] + a[i+3];
    }
    return sum;

}


int main()
{
    const long long int n = 67108864;
    int halfN = n / 2;
    double* a = new double[n];

    for (int i = 0; i < n; i++) {
        a[i] = i;
    }

    LARGE_INTEGER freq, head, tail,temp;

    int count = 0;
    double time = 0;
//    double* backup = new double[n];
//    memcpy(backup, a, n * sizeof(double));  // 备份
    double rst = 0;

    for (long long int i = 2; i <= n; i *= 2)
    {
        count = 0;
        QueryPerformanceFrequency(&freq);
        QueryPerformanceCounter(&head);

//        while (count < n / i * 5) {
//            memcpy(a, backup, n * sizeof(double));
//            count++;
//        }
//        QueryPerformanceCounter(&temp);
//        double overhead = (temp.QuadPart - head.QuadPart) * 1000.0 / freq.QuadPart;

        count = 0;
        long long int halfi = i / 2;

        QueryPerformanceCounter(&head);

        volatile double total_rst = 0;  // 避免编译器把循环优化掉

        while (count< n/i*5)
        {
//            memcpy(a, backup, n * sizeof(double));
//            rst = func1(a, i);
//            rst = func2(a, i);
//            rst = func3(a, halfi);
//            rst = func4(a, i);
            rst = func5(a, i);
            total_rst += rst;
            count++;
        }
        QueryPerformanceCounter(&tail);

        time = (tail.QuadPart - head.QuadPart) * 1000.0 / freq.QuadPart;

        cout << "规模:" << i <<" 结果:"<< rst << " count:" << count << " 总时间:" << time << " 单次时间:" << time / count << endl;
        cout<<total_rst<<endl;
    }

    return 0;
}
