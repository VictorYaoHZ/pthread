#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include<iostream>
#include <pmmintrin.h>
#include<immintrin.h>
using namespace std;
#define N 2048//��ĸ���
#define K 128//����ĸ���
#define E 256//Ԫ�صĸ���

typedef struct
{
	float elements[E];
}Point;

int center[N];  ///  �ж�ÿ���������ĸ��� center[k]=p������k��pointλ�ڵ�p����

Point point[N] = {
	{1,1,1,1},
	{1,2,1,1},
	{2,1,2,1},
	{2,2,1,1},
	{50,49,49,50},
	{50,51,51,50},
	{51,49,51,49},
	{100,99,98,101},
	{100,101,100,100},
	{101,99,98,100},
	{102,99,101,100},
	{98,101,101,98}
};

Point mean[K];  ///  ����ÿ���ص����ĵ�
void printPointInfo(int index)
{
	cout << "�� :(";
	cout << point[index].elements[0];
	for (int i = 1; i < E; i++)
	{
		cout << "," << point[index].elements[i];
	}
	cout << ") �ھ���" << center[index] + 1 << "��" << endl;
}
void printCenterInfo(int index)
{
	cout << "����" << index + 1 << "�������ĵ���:(";
	cout << mean[index].elements[0];
	for (int i = 1; i < E; i++)
	{
		cout << "," << mean[index].elements[i];
	}
	cout << ")" << endl;
}
float getDistance(Point point1, Point point2)//����ŷ�Ͼ���
{
	float d = 0.0;
	for (int i = 0; i < E; i++)
	{
		d += (point1.elements[i] - point2.elements[i]) * (point1.elements[i] - point2.elements[i]);
	}
	d = sqrt(d);
	return d;
}

/// ����ÿ���ص����ĵ�
void getMean(int center[N])
{
	Point tep;
	int i, j, count = 0;
	for (i = 0; i < K; ++i)
	{//k�Ǿ����������һѭ����ÿ��ѭ�����һ���������ģ����Ϊi
		count = 0;

		for (int m = 0; m < E; ++m)
		{
			tep.elements[m] = 0;
		}

		for (j = 0; j < N; ++j)
		{
			if (center[j] == i)
			{
				count++;
				for (int m = 0; m < E; m++)
				{
					tep.elements[m] += point[j].elements[m];
				}

			}
		}
		for (int m = 0; m < E; m++)
		{
			tep.elements[m] /= count;
		}
		mean[i] = tep;
	}
	/*
	for (i = 0; i < K; ++i)
	{
		printCenterInfo(i);
		//printf("The new center point of %d is : \t( %f, %f )\n", i + 1, mean[i].x, mean[i].y);
	}
	*/

}


void getMean1(int center[N])
{
	Point tep[K];
	int i, j;
	int count[K] = { 0 };
	{//k�Ǿ����������һѭ����ÿ��ѭ�����һ���������ģ����Ϊi

		for (int m = 0; m < K; ++m)
		{
			for (int n = 0; n < E; ++n)
			{
				tep[m].elements[n] = 0;
			}

		}


		/*
		for (int m = 0; m < K; ++m)
		{
			__m128 t1;
			t1 = _mm_setzero_ps();
			for (int n = E - 4; n >= 0; n -= 4)
			{
				_mm_store_ps(tep[m].elements, t1);
			}
			for (int n = (E % 4) - 1; n >= 0; --n)
			{
				tep[m].elements[n] = 0;
			}

		}
		*/
		for (j = 0; j < N; ++j)
		{
			__m128 t1, t2, sum;
			Point* temp2 = &point[j];
			Point* temp = &tep[center[j]];//����Ƶ�����ʣ�����ָ��
			count[center[j]]++;
			for (int m = E - 4; m >= 0; m -= 4)
			{
				t1 = _mm_loadu_ps(temp->elements + m);
				t2 = _mm_loadu_ps(temp2->elements + m);
				t1 = _mm_add_ps(t1, t2);
				_mm_storeu_ps(temp->elements + m, t1);
			}
			for (int m = (E % 4) - 1; m >= 0; --m)
			{
				temp->elements[m] += temp2->elements[m];

			}

		}

		/*
		for (j = 0; j < N; ++j)
		{
			count[center[j]]++;
			for (int m = 0;m<E;m++)
			{
				tep[center[j]].elements[m] += point[j].elements[m];

			}

		}
		*/

		for (int i = 0; i < K; i++)
		{
			for (int m = 0; m < E; ++m)
			{
				mean[i].elements[m] = tep[i].elements[m] / count[i];
			}

		}

	}
}

/// ����ƽ������
float getE()
{
	int i, j, k;
	float cnt = 0.0, sum = 0.0;
	for (i = 0; i < K; ++i)
	{
		for (j = 0; j < N; ++j)
		{

			if (i == center[j])
			{
				//cnt = (point[j].x - mean[i].x) * (point[j].x - mean[i].x) + (point[j].y - mean[i].y) * (point[j].y - mean[i].y);
				for (k = 0; k < E; k++)
				{
					cnt += (point[j].elements[k] - mean[i].elements[k]) * (point[j].elements[k] - mean[i].elements[k]);
				}
				sum += cnt;
				cnt=0;
			}
		}
	}
	return sum;
}



float getE1()
{
	int i, j, k;
	float cnt = 0.0, sum = 0.0;
	float temp1[4];
        __m128 res;
        res = _mm_setzero_ps();
		for (j = 0; j < N; ++j)
		{
			//cnt = (point[j].x - mean[i].x) * (point[j].x - mean[i].x) + (point[j].y - mean[i].y) * (point[j].y - mean[i].y);
			for (int m = E - 4; m >= 0; m -= 4)
			{

				__m128 t1, t2;
				t1 = _mm_loadu_ps(point[j].elements + m);
				t2 = _mm_loadu_ps(mean[center[j]].elements + m);
				t2 = _mm_sub_ps(t1, t2);
				res = _mm_add_ps(res, _mm_mul_ps(t2, t2));
			}

			for (int m = (E % 4) - 1; m >= 0; --m)
			{
				cnt += (point[j].elements[m] - mean[center[j]].elements[m]) * (point[j].elements[m] - mean[center[j]].elements[m]);

			}

			sum += cnt;
            cnt=0;
		}

		_mm_storeu_ps(temp1, res);

        cnt = temp1[0]+temp1[1]+temp1[2]+temp1[3];
        sum += cnt;
	return sum;
}

/// ��N�������
void cluster()
{
	int i, j, q;
	float min;
	float distance[N][K];
	for (i = 0; i < N; ++i)
	{
		min = (float)INT_MAX;
		for (j = 0; j < K; ++j)
		{
			//distance[i][j] = getDistance(point[i], mean[j]);

			float d = 0.0;
			for (int k = 0; k < E; k++)
			{
				d += (point[i].elements[k] - mean[j].elements[k]) * (point[i].elements[k] - mean[j].elements[k]);
			}
			//d = sqrt(d);
			if(d<min)
            {
                min = d;
                center[i]=j;
            }
			//distance[i][j] = d;
			/// printf("%f\n", distance[i][j]);  /// �����������Զ���ÿ������3�����ĵ�֮��ľ���
		}

		//printPointInfo(i);
	}
	//printf("-----------------------------\n");
}

void cluster1()
{
	int i, j, q;
	float min;




	for (i = 0; i < N; ++i)
	{
	    float d = 0.0;
	    float distance[K] = {0.0};
        float temp1[4];
        min = (float)INT_MAX;
	    for (j = 0; j < K; ++j)//����ÿ�����ĵ㣬��¼���뵽distance����
		{
		    d=0;
		    __m128 res;
            res = _mm_setzero_ps();
		    //min = (float)INT_MAX;
            for (int k = E-4; k >= 0 ; k-=4)
            {
                //distance[i][j] = getDistance(point[i], mean[j]);
                __m128 t1, t2;
                t1 = _mm_loadu_ps(point[i].elements + k);
                t2 = _mm_loadu_ps(mean[j].elements + k);
                t2 = _mm_sub_ps(t1, t2);
                t2 = _mm_mul_ps(t2, t2);
                res = _mm_add_ps(res,t2);

            }
            _mm_storeu_ps(temp1,res);
            for (int k = (E % 4) - 1; k >= 0; --k)
            {
                d+=(point[i].elements[k] - mean[j].elements[k]) * (point[i].elements[k] - mean[j].elements[k]);
            }
            d+=temp1[0]+temp1[1]+temp1[2]+temp1[3];
             if(d<min)
            {
                min = d;
                center[i] = j;
            }
		}




		//printPointInfo(i);
	}

	//printf("-----------------------------\n");
}

void initPointSet()
{
	srand((unsigned int)time(NULL));
	for (int i = 0; i < N; ++i)
	{
		for (int j = 0; j < E; ++j)
		{
			point[i].elements[j] = rand() % 25;
		}
	}
}
bool checkFlag(int* flag, int j)//flag��û��jʱ������
{
	for (int i = 0; i < K; i++)
	{
		if (flag[i] == j)
		{
			return false;
		}
	}
	return true;
}
void initCenter()
{
	int i, j, n = 0;
	int flag[K];
	for (i = 0; i < K; i++)
	{
		flag[i] = -1;
	}
	srand((unsigned int)time(NULL));
	for (i = 0; i < K; ++i)
	{
		while (true)
		{
			j = rand() % N;
			if (checkFlag(flag, j))
			{
				mean[i] = point[j];
				flag[i] = j;
				break;
			}

		}




		for (int e = 0; e < E; ++e)
		{
			mean[i].elements[e] = point[j].elements[e];
		}


	}
}
void doit()
{
	int i, j, n = 0;
	float temp1;
	float temp2, t;
	//printf("----------Data sets----------\n");
	/*
	* for (i = 0; i < N; ++i)
	{
		printPointInfo(i);
	}
	*/

	printf("-----------------------------\n");

	//����ѡ��ǰʱ��Ϊ�����
	initCenter();

	/*
	* mean[0].x = point[0].x;      /// ��ʼ��k�����ĵ�
mean[0].y = point[0].y;

mean[1].x = point[3].x;
mean[1].y = point[3].y;

mean[2].x = point[6].x;
mean[2].y = point[6].y;
	*/

	//��ʱ��ʼ
	clock_t start, finish;
	long double time_getE = 0, time_getMean = 0, time_cluster = 0;

	start = clock();
	cluster();
	finish = clock();
	time_cluster += (double(finish) - double(start)) / double(CLOCKS_PER_SEC);

	start = clock();
	temp1 = getE();
	finish = clock();
	time_getE += (double(finish) - double(start)) / double(CLOCKS_PER_SEC);

	n++;                   ///  n�����γ����յĴ����˶��ٴ�

	//printf("The E1 is: %f\n\n", temp1);

	start = clock();
	getMean(center);
	finish = clock();
	time_getMean += (double(finish) - double(start)) / double(CLOCKS_PER_SEC);

	start = clock();
	cluster();
	finish = clock();
	time_cluster += (double(finish) - double(start)) / double(CLOCKS_PER_SEC);

	start = clock();
	temp2 = getE();
	finish = clock();
	time_getE += (double(finish) - double(start)) / double(CLOCKS_PER_SEC);
	n++;



	//printf("The E2 is: %f\n\n", temp2);

	while (fabs(temp2 - temp1) != 0)   ///  �Ƚ�����ƽ����� �ж��Ƿ���ȣ�����ȼ�������
	{
		temp1 = temp2;

		start = clock();
		getMean(center);
		finish = clock();
		time_getMean += (double(finish) - double(start)) / double(CLOCKS_PER_SEC);


		start = clock();
		cluster();
		finish = clock();
		time_cluster += (double(finish) - double(start)) / double(CLOCKS_PER_SEC);

		start = clock();
		temp2 = getE();
		finish = clock();
		time_getE += (double(finish) - double(start)) / double(CLOCKS_PER_SEC);
		n++;
		//printf("The E%d is: %f\n", n, temp2);
	}
	printf("The E is: %f\n", temp2);
	printf("The total number of cluster is: %d\n\n", n);  /// ͳ�Ƴ���������
	/*
	for (i = 0; i < N; ++i)
	{
		printPointInfo(i);
	}
	for (j = 0; j < K; j++)
	{
		printCenterInfo(j);
	}
	*/



	//��ʱ����
	cout << "total time of getE is" << time_getE/n << endl;
	cout << "total time of getMean is" << time_getMean/n << endl;
	cout << "total time of cluster is" << time_cluster/n << endl;
}
void doit1()
{
	int i, j, n = 0;
	float temp1;
	float temp2, t;

	//printf("----------Data sets----------\n");
	/*
	* for (i = 0; i < N; ++i)
	{
		printPointInfo(i);
	}
	*/

	printf("-----------------------------\n");

	//����ѡ��ǰʱ��Ϊ�����
	initCenter();

	/*
	* mean[0].x = point[0].x;      /// ��ʼ��k�����ĵ�
mean[0].y = point[0].y;

mean[1].x = point[3].x;
mean[1].y = point[3].y;

mean[2].x = point[6].x;
mean[2].y = point[6].y;
	*/

	//��ʱ��ʼ
	clock_t start, finish;
	long double time_getE = 0, time_getMean = 0, time_cluster = 0;

	start = clock();
	cluster1();
	finish = clock();
	time_cluster += (double(finish) - double(start)) / double(CLOCKS_PER_SEC);

	start = clock();
	temp1 = getE1();
	finish = clock();
	time_getE += (double(finish) - double(start)) / double(CLOCKS_PER_SEC);

	n++;                   ///  n�����γ����յĴ����˶��ٴ�

	//printf("The E1 is: %f\n\n", temp1);

	start = clock();
	getMean1(center);
	finish = clock();
	time_getMean += (double(finish) - double(start)) / double(CLOCKS_PER_SEC);

	start = clock();
	cluster1();
	finish = clock();
	time_cluster += (double(finish) - double(start)) / double(CLOCKS_PER_SEC);

	start = clock();
	temp2 = getE1();
	finish = clock();
	time_getE += (double(finish) - double(start)) / double(CLOCKS_PER_SEC);
	n++;



	//printf("The E2 is: %f\n\n", temp2);

	while (fabs(temp2 - temp1) != 0)   ///  �Ƚ�����ƽ����� �ж��Ƿ���ȣ�����ȼ�������
	{
		temp1 = temp2;

		start = clock();
		getMean1(center);
		finish = clock();
		time_getMean += (double(finish) - double(start)) / double(CLOCKS_PER_SEC);


		start = clock();
		cluster1();
		finish = clock();
		time_cluster += (double(finish) - double(start)) / double(CLOCKS_PER_SEC);

		start = clock();
		temp2 = getE1();
		finish = clock();
		time_getE += (double(finish) - double(start)) / double(CLOCKS_PER_SEC);
		n++;
		//printf("The E%d is: %f\n", n, temp2);
	}
	printf("The E is: %f\n", temp2);
	printf("The total number of cluster is: %d\n\n", n);  /// ͳ�Ƴ���������


/*
	for (i = 0; i < N; ++i)
	{
		printPointInfo(i);
	}
	for (j = 0; j < K; j++)
	{
		printCenterInfo(j);
	}
*/





	//��ʱ����
	cout << "total time of getE is" << time_getE/n << endl;
	cout << "total time of getMean is" << time_getMean/n << endl;
	cout << "total time of cluster is" << time_cluster/n << endl;
}
int main()
{
	initPointSet();
	doit();
	doit1();
    doit();
	doit1();
    doit();
	doit1();
    doit();
	doit1();
    doit();
	doit1();

	system("pause");
	return 0;
}
