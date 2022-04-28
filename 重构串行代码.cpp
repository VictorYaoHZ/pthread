#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <iostream>
#include <pmmintrin.h>
#include <immintrin.h>
#include <pthread.h>
#include<limits.h>
using namespace std;
#define N 12//点的个数
#define K 3//聚类的个数
#define E 4 //元素的个数
#define NUM_THREAD 4
typedef struct
{
	float elements[E];
}Point;
Point tep[K];
Point mean[K];  ///  保存每个簇的中心点
int count1[K];
int center[N];  ///  判断每个点属于哪个簇 center[k]=p，即第k个point位于第p聚类

pthread_barrier_t	barrier1;
//Point point[N];
float minn;

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


void printPointInfo(int index)
{
	cout << "点 :(";
	cout << point[index].elements[0];
	for (int i = 1; i < E; i++)
	{
		cout << "," << point[index].elements[i];
	}
	cout << ") 在聚类" << center[index] + 1 << "中" << endl;
}
void printCenterInfo(int index)
{
	cout << "聚类" << index + 1 << "的新中心点是:(";
	cout << mean[index].elements[0];
	for (int i = 1; i < E; i++)
	{
		cout << "," << mean[index].elements[i];
	}
	cout << ")" << endl;
}
float getDistance(Point point1, Point point2)//计算欧氏距离
{
	float d = 0.0;
	for (int i = 0; i < E; i++)
	{
		d += (point1.elements[i] - point2.elements[i]) * (point1.elements[i] - point2.elements[i]);
	}
	d = sqrt(d);
	return d;
}
//--------------------------
void kmeans()
{
    //================================================
    //重置count1与tep
	for (int i = 0; i < K; i++) count1[i] = 0;
    for (int m = 0; m < K; ++m)
    {
        for (int n = 0; n < E; ++n)
        {
            tep[m].elements[n] = 0;
        }
    }
    //================================================
    //根据上一轮聚类的结果，重新计算聚类中心，原getMean函数
    //累加
    for (int j = 0; j < N; ++j)
    {
        count1[center[j]]++;
        for (int m = 0; m < E; m++)
        {
            tep[center[j]].elements[m] += point[j].elements[m] * 100000;
        }
    }
    //求平均
    for (int i = 0; i < K; i++)
    {
        for (int m = 0; m < E; ++m)
        {
            mean[i].elements[m] = tep[i].elements[m] / (count1[i] * 100000);
        }
    }
    //==========================================
    //根据新的聚类中心，重新进行聚类，原cluster函数
	for (int i = 0; i < N; ++i)
	{
		minn = (float)INT_MAX;
		for (int j = 0; j < K; ++j)
		{
			float d = 0.0;
			for (int k = 0; k < E; k++)
			{
				d += (point[i].elements[k] - mean[j].elements[k]) * (point[i].elements[k] - mean[j].elements[k]);
			}
			if (d < minn)
			{
				minn = d;
				center[i] = j;
			}
		}
		//printPointInfo(i);
	}
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
bool checkFlag(int* flag, int j)//flag中没有j时返回真
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
void cluster()
{

	float min;
	for (int i = 0; i < N; ++i)
	{
		min = (float)INT_MAX;
		for (int j = 0; j < K; ++j)
		{
			//distance[i][j] = getDistance(point[i], mean[j]);

			float d = 0.0;
			for (int k = 0; k < E; k++)
			{
				d += (point[i].elements[k] - mean[j].elements[k]) * (point[i].elements[k] - mean[j].elements[k]);
			}
			//d = sqrt(d);
			//***********************************************
			// 这里发现不需要用一个distance数组来进行记录，最后再顺序比较数组内数据
			// 只需要用一个临时变量来代替，顺序比较更新欧氏距离最小的聚类中心点即可
			//***********************************************
			if (d < min)
			{
				min = d;
				center[i] = j;
			}
			//distance[i][j] = d;
			/// printf("%f\n", distance[i][j]);  /// 可以用来测试对于每个点与3个中心点之间的距离
		}

		//printPointInfo(i);
	}
	//printf("-----------------------------\n");
}

void kmeans_helper()
{
	int n = 0;
	initCenter();
	cluster();
	n++;
	while (n < 20) { kmeans(); n++; }

	for (int i = 0; i < N; ++i)
	{
		printPointInfo(i);
	}
	for (int j = 0; j < K; j++)
	{
		printCenterInfo(j);
	}
}

int main()
{
	//initPointSet();
	kmeans_helper();


	system("pause");
	return 0;
}
