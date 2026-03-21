//#include <iostream>
//#include <windows.h>
//
//using namespace std;
//
//int main()
//{
//    const int n = 500;
//
//    double a[n][n];
//    double b[n];
//    double sum[n];
//
//
//    for (int i = 0; i < n; i++) {
//        for (int j = 0; j < n; j++) {
//            a[i][j] = i + j;
//        }
//        b[i] = i;
//    }
//
//    LARGE_INTEGER freq, head, tail1, tail2;
//    QueryPerformanceFrequency(&freq);
//    QueryPerformanceCounter(&head);
//
//    int n2 = 100;
//
//    // 平凡算法
//    for(int k=0;k<n2;k++){
//        for (int i = 0; i < n; i++) {
//            sum[i] = 0;
//            for (int j = 0; j < n; j++) {
//                sum[i] += a[j][i] * b[j];
//            }
//        }
//    }
//
//    QueryPerformanceCounter(&tail1);
//
//    // 优化算法
//    for(int k=0;k<n2;k++){
//        for (int i = 0; i < n; i++) {
//            sum[i] = 0;
//        }
//        for (int j = 0; j < n; j++) {
//            for (int i = 0; i < n; i++) {
//                sum[i] += a[j][i] * b[j];
//            }
//        }
//    }
//
//    QueryPerformanceCounter(&tail2);
//
//    double total = 0.0;
//    for (int i = 0; i < n; i++) {
//        total += sum[i];
//    }
//
//    cout<<total<<endl;
//
//    cout << (tail1.QuadPart - head.QuadPart) * 1000.0 / freq.QuadPart << endl;
//
//    cout << (tail2.QuadPart - tail1.QuadPart) * 1000.0 / freq.QuadPart << endl;
//
//    return 0;
//}
