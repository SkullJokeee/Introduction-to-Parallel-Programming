#include <iostream>
#include <windows.h>

using namespace std;

int main()
{
	const int n = 8192;
	int halfN = n/2;
	int m=n;
	int halfM = m/2;
	double a[n];
	float sum = 0,sum1=0.0, sum2=0.0,sum3=0.0;
	for (int i = 0; i < n; i++) {
		a[i] = i;
	}

	LARGE_INTEGER freq, head, tail1, tail2,tail3;
	QueryPerformanceFrequency(&freq);
	QueryPerformanceCounter(&head);

	for (int i = 0; i < n; i++) {
		sum += a[i];
	}

	QueryPerformanceCounter(&tail1);

	for (int i = 0; i < halfN; i++) {
		sum1 += a[i];
		sum2 +=a[halfN+i];
	}

	sum1 += sum2;

	QueryPerformanceCounter(&tail2);

    while(m>1){
        for(int i=0;i<halfM;i++){
            a[i]+=a[m-i-1];
        }
        m/=2;
        halfM = m/2;
    }

    sum3 = a[0];

	QueryPerformanceCounter(&tail3);

	cout << sum << " " << sum1 << " " << sum2 << " "<<sum3<<endl;

	cout << (tail1.QuadPart - head.QuadPart) * 1000.0 / freq.QuadPart << endl;
	cout << (tail2.QuadPart - tail1.QuadPart) * 1000.0 / freq.QuadPart << endl;
	cout << (tail3.QuadPart - tail2.QuadPart) * 1000.0 / freq.QuadPart << endl;


	return 0;
}
