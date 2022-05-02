#include <stdio.h>
#include <windows.h>
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
#define N 2048//点的个数
#define K 8//聚类的个数
#define E 3//元素的个数
#define NUM_THREAD 8
typedef struct
{
    float elements[E];
}Point;
Point tep[K];
Point mean[K];  ///  保存每个簇的中心点
int count1[K];
int center[N];  ///  判断每个点属于哪个簇 center[k]=p，即第k个point位于第p聚类

float temp1_for_kmeanspthread = 0.0;//为pthread实现的kmeans函数给一个全局变量用于记录平方误差
float temp2_for_kmeanspthread = 0.0;
int n_for_kmeanspthread = 0;//同样地，记录迭代次数n
pthread_barrier_t	barrier1;
Point point[N];
float minn;
/*
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
*/

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
pthread_barrier_t barrier_reset;
pthread_barrier_t barrier_sum;
pthread_barrier_t barrier_avr;
pthread_barrier_t barrier_cluster;
pthread_barrier_t barrier_getE;
pthread_mutex_t amutex = PTHREAD_MUTEX_INITIALIZER;
typedef struct {
    int t_id;
}threadParam_t;
void* threadFunc(void* param)
{
    threadParam_t* p = (threadParam_t*)param;
    int tid = p->t_id;
    int worknum = N / NUM_THREAD;
    int begin_index = tid * worknum;
    int end_index = (tid + 1) * worknum;
    int worknum2 = E / NUM_THREAD;
    int begin_index2 = tid * worknum2;
    int end_index2 = (tid + 1) * worknum2;
    __m128 res, t1, t2;
    if (tid == NUM_THREAD - 1)
    {
        //最后一个工作任务
        end_index = N;
        end_index2 = E;
    }
    for (int n = 1; n < 20; ++n) {//这里的20是最大迭代次数
        n_for_kmeanspthread = n;
        for (int i = 0; i < K; i++) count1[i] = 0;
        for (int m = 0; m < K; ++m)
        {
            for (int n = begin_index2; n < end_index2; ++n)
            {
                tep[m].elements[n] = 0;
            }
        }



        pthread_barrier_wait(&barrier_reset);
        //===================================================

        for (int j = begin_index; j < end_index; ++j)
        {
            //__m128 t1, t2;
            Point* temp2 = &point[j];
            Point* temp = &tep[center[j]];//避免频繁访问，保存指针
            count1[center[j]]++;
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
        pthread_barrier_wait(&barrier_sum);
        //=================================================

        for (int i = 0; i < K; i++)
        {
            for (int m = begin_index2; m < end_index2; ++m)
            {
                mean[i].elements[m] = tep[i].elements[m] / (count1[i] * 100000);
            }
        }

        pthread_barrier_wait(&barrier_avr);
        //================================================
        for (int i = begin_index; i < end_index; ++i)
        {
            float d = 0.0;
            float temp1[4];
            minn = (float)INT_MAX;
            for (int j = 0; j < K; ++j)//对于每个中心点，记录距离到distance数组
            {
                d = 0;
                //__m128 res;
                res = _mm_setzero_ps();
                //min = (float)INT_MAX;
                for (int k = E - 4; k >= 0; k -= 4)
                {
                    //distance[i][j] = getDistance(point[i], mean[j]);
                    //__m128 t1, t2;
                    t1 = _mm_loadu_ps(point[i].elements + k);
                    t2 = _mm_loadu_ps(mean[j].elements + k);
                    t2 = _mm_sub_ps(t1, t2);
                    t2 = _mm_mul_ps(t2, t2);
                    res = _mm_add_ps(res, t2);

                }
                _mm_storeu_ps(temp1, res);
                for (int k = (E % 4) - 1; k >= 0; --k)
                {
                    d += (point[i].elements[k] - mean[j].elements[k]) * (point[i].elements[k] - mean[j].elements[k]);
                }
                d += temp1[0] + temp1[1] + temp1[2] + temp1[3];
                if (d < minn)
                {
                    minn = d;
                    center[i] = j;
                }
            }

        }
        pthread_barrier_wait(&barrier_cluster);
        //================================================================
        //计算聚类后的平方误差
        float cnt = 0.0,sum_inter=0.0;
        float temp1[4];
        __m128 res;
        res = _mm_setzero_ps();
		for (int j = begin_index; j < end_index; ++j)
		{
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
			    pthread_mutex_lock(&amutex);
				cnt = (point[j].elements[m] - mean[center[j]].elements[m]) * (point[j].elements[m] - mean[center[j]].elements[m]);
			    pthread_mutex_unlock(&amutex);
			}

			sum_inter += cnt;
            cnt=0;
		}
		_mm_storeu_ps(temp1, res);
        cnt = temp1[0]+temp1[1]+temp1[2]+temp1[3];
        sum_inter+=cnt;

        temp2_for_kmeanspthread += sum_inter;

        //pthread_barrier_wait(&barrier_getE);
        //=================================================
        //判断前后两次
        /*
        if(fabs(temp2_for_kmeanspthread-temp1_for_kmeanspthread)<100)
        {
            break;
        }
        else
        {
            temp1_for_kmeanspthread = temp2_for_kmeanspthread;
        }
        */
    }
}
float kmeans()
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
            tep[center[j]].elements[m] += point[j].elements[m];
        }
    }
    //求平均
    for (int i = 0; i < K; i++)
    {
        for (int m = 0; m < E; ++m)
        {
            mean[i].elements[m] = tep[i].elements[m] / (count1[i]);
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
    //==============================================
    //计算并返回平方误差
    float cnt = 0.0, sum = 0.0;
    for (int j = 0; j < N; ++j)
    {
        for (int k = 0; k < E; k++)
        {
            cnt += (point[j].elements[k] - mean[center[j]].elements[k]) * (point[j].elements[k] - mean[center[j]].elements[k]);
        }
        sum += cnt;
        cnt=0;
    }
	return sum;
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
    int i, j = 0;
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
void kmeans_helper()
{
    int n = 0;
    float tempE1 = 0.0,tempE2=0.0;
    initCenter();
    cluster();
    tempE1 = getE();
    n++;
    long long head, tail, freq;
    double time = 0.0;
    double total_time = 0.0;
    while (n < 100) {
        QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
        QueryPerformanceCounter((LARGE_INTEGER*)&head);
        tempE2 = kmeans();
        QueryPerformanceCounter((LARGE_INTEGER*)&tail);
        time = (tail - head) * 1000.0 / freq;
        total_time += time;
        //cout<<"common time:"<<time<<endl;
        if(fabs(tempE1-tempE2)<=10)
        {
            break;
        }
        tempE1 = tempE2;
        n++;
    }
    cout << "=================================" << endl;
    cout << "total time of common kmeans alg:" << total_time << endl;
    cout << "times of cluster"<<n<<endl;
    cout << "average time of common kmeans alg:" <<total_time/n<<endl;
    cout << "the square error(E) is:"<<tempE2<<endl;
    /*
    for (int i = 0; i < N; ++i)
    {
        printPointInfo(i);
    }
    for (int j = 0; j < K; j++)
    {
        printCenterInfo(j);
    }
    */
}
void kmeans_pthread()
{
    n_for_kmeanspthread = 0;
    pthread_barrier_init(&barrier_reset, NULL, NUM_THREAD);
    pthread_barrier_init(&barrier_sum, NULL, NUM_THREAD);
    pthread_barrier_init(&barrier_avr, NULL, NUM_THREAD);
    pthread_barrier_init(&barrier_cluster, NULL, NUM_THREAD);
    pthread_barrier_init(&barrier_getE, NULL, NUM_THREAD);
    pthread_t thread_handles[NUM_THREAD];
    threadParam_t threadParam[NUM_THREAD];
    pthread_mutex_init(&amutex,NULL);
    for (int tid = 0; tid < NUM_THREAD; tid++)
    {
        threadParam[tid].t_id = tid;
    }
    for (int tid = 0; tid < NUM_THREAD; tid++)
    {
        pthread_create(&thread_handles[tid], NULL, threadFunc, (void*)(threadParam + tid));
    }
    for (int tid = 0; tid < NUM_THREAD; tid++)
    {
        pthread_join(thread_handles[tid], NULL);
    }
    pthread_barrier_destroy(&barrier_reset);
    pthread_barrier_destroy(&barrier_sum);
    pthread_barrier_destroy(&barrier_avr);
    pthread_barrier_destroy(&barrier_cluster);
    pthread_barrier_destroy(&barrier_getE);
    pthread_mutex_destroy(&amutex);
}
void kmeans_helper_pthread()
{
    int n = 0;
    initCenter();
    cluster();
    temp1_for_kmeanspthread = getE();
    n++;
    long long head, tail, freq;
    double time = 0.0;
    double total_time = 0.0;
    //while (n < 20) {
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
    QueryPerformanceCounter((LARGE_INTEGER*)&head);
    kmeans_pthread();
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
    time = (tail - head) * 1000.0 / freq;
    total_time += time;
    // cout<<"pthread time:"<<time<<endl;
    n++;
    //}
    cout << "=================================" << endl;
    cout << "total time of kmeans alg in pthread:" << total_time << endl;
    cout << "=================================" << endl;
    cout<<n_for_kmeanspthread<<endl;
    cout<<temp2_for_kmeanspthread<<endl;
    /*
    for (int i = 0; i < N; ++i)
    {
        printPointInfo(i);
    }
    for (int j = 0; j < K; j++)
    {
        printCenterInfo(j);
    }
    */
}
int main()
{
    initPointSet();
    kmeans_helper();
    //initPointSet();
    kmeans_helper_pthread();
    system("pause");
    return 0;
}
