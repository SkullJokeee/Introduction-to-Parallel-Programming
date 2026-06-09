#include <iostream>
#include <windows.h>

using namespace std;

const int n = 5000;
const int n2 = 100;

double a[n][n];
double b[n];
double sum[n];

int main()
{

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            a[i][j] = i + j;
        }
        b[i] = i;
    }

    LARGE_INTEGER freq, head, tail;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&head);


int count = 0;

for (int k = 200; k <= n; k +=200)
    {
        count = 0;
        QueryPerformanceCounter(&head);

        while(count<=(n-k)/50+3){
            for (int i = 0; i < k; i++) {
                sum[i] = 0;
                for (int j = 0; j < k; j++) {
                    sum[i] += a[j][i] * b[j];
                }
            }
            count++;
        }

//        while(count<=(n-k)/50+3){
//            for (int i = 0; i < k; i++)
//                sum[i] = 0;
//            for (int j = 0; j < k; j++) {
//                for (int i = 0; i < k; i++) {
//                    sum[i] += a[j][i] * b[j];
//                }
//            }
//            count++;
//        }

        QueryPerformanceCounter(&tail);

        double time = (double)(tail.QuadPart - head.QuadPart) * 1000.0 / freq.QuadPart;

        cout << "规模:" << k << " count:" << count << " 总时间:" << time << " 单次时间:" << time / count << endl;

    }

    double total = 0.0;
    for (int i = 0; i < n; i++) {
        total += sum[i];
    }

    cout<<total<<endl;

    return 0;
}
