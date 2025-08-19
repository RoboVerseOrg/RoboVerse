#include "cpu/vision.h"


void knn_cpu(float* ref_dev, int ref_width, float* query_dev, int query_width,
    int height, int k, float* dist_dev, long* ind_dev, long* ind_buf)
{
    // Compute all the distances
    for(int query_idx = 0;query_idx<query_width;query_idx++)
    {
        for(int ref_idx = 0;ref_idx < ref_width;ref_idx++)
        {
            dist_dev[query_idx * ref_width + ref_idx] = 0;
            for(int hi=0;hi<height;hi++)
                dist_dev[query_idx * ref_width + ref_idx] += (ref_dev[hi * ref_width + ref_idx] - query_dev[hi * query_width + query_idx]) * (ref_dev[hi * ref_width + ref_idx] - query_dev[hi * query_width + query_idx]);
        }
    }

    float temp_value;
    long temp_idx;
    // sort the distance and get the index
    for(int query_idx = 0;query_idx<query_width;query_idx++)
    {
        for(int i = 0;i < ref_width;i++)
        {
            ind_buf[i] = i+1;
        }
        for(int i = 0;i < ref_width;i++)
            for(int j = 0;j < ref_width - i - 1;j++)
            {
                if(dist_dev[query_idx * ref_width + j] > dist_dev[query_idx * ref_width + j + 1])
                {
                    temp_value = dist_dev[query_idx * ref_width + j];
                    dist_dev[query_idx * ref_width + j] = dist_dev[query_idx * ref_width + j + 1];
                    dist_dev[query_idx * ref_width + j + 1] = temp_value;
                    temp_idx = ind_buf[j];
                    ind_buf[j] = ind_buf[j + 1];
                    ind_buf[j + 1] = temp_idx;
                }

            }

        for(int i = 0;i < k;i++)
            ind_dev[query_idx + i * query_width] = ind_buf[i];
        #if DEBUG
        for(int i = 0;i < ref_width;i++)
            printf("%d, ", ind_buf[i]);
        printf("\n");
        #endif

    }





}