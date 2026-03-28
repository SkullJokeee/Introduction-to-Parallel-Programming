#include <iostream>
#include <windows.h>
#include <functional>
#include <algorithm>
#include <random>

using namespace std;

long long int mysum;

double func1(double a[], long long n) { // 平凡算法
    double sum = 0;
    for (int i = 0; i < n; i++) {
        sum += a[i];
    }
    return sum;
}

double func2(double a[], long long n) {  // 两路链式
    double sum1 = 0, sum2 = 0;
    for (int i = 0; i < n; i += 2) {
        sum1 += a[i];
        sum2 += a[i + 1];
    }
    return sum1 + sum2;
}


double func3(double a[], long long halfN) {
    //long long halfN = n / 2;
    double sum1 = 0, sum2 = 0;
    for (int i = 0; i < halfN; i++) {
        sum1 += a[i];
    }
    for (int i = 0; i < halfN; i++) {
        sum2 += a[halfN + i];
    }
    return sum1 + sum2;
}

double func4(double a[], long long n) { // 递归
    long long m = n;
    double sum = 0;
    for (m = n; m > 1; m /= 2) {
        for (int i = 0; i < m / 2; i++) {
            a[i] = a[i * 2] + a[i * 2 + 1];
        }
    }
    sum = a[0];
    return sum;
}

int main()
{
    const long long int n = 33554432;
    int halfN = n / 2;
    int n2 = 1;
    double* a = new double[n];

    for (int i = 0; i < n; i++) {
        a[i] = i;
    }

    LARGE_INTEGER freq, head, tail;

    int count = 0;
    double time = 0;
    double* backup = new double[n];
    memcpy(backup, a, n * sizeof(double));  // 备份

    for (long long int i = 2; i <= n; i *= 2)
    {
        count = 0;
        long long int halfi = i / 2;

        memcpy(a, backup, n * sizeof(double));

        QueryPerformanceFrequency(&freq);
        QueryPerformanceCounter(&head);

        //cout << a[0] << endl;

        while (count< n/i)
        {
            //func1(a, i);
            //func2(a, i);
            //func3(a, halfi);
            func4(a, i);
            count++;
        }
        QueryPerformanceCounter(&tail);

        time = (tail.QuadPart - head.QuadPart) * 1000.0 / freq.QuadPart;

        cout << "规模:" << i << " count:" << count << " 总时间:" << time << " 单次时间:" << time / count << endl;

    }

    return 0;
}
