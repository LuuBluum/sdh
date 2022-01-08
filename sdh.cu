#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>


#define BOX_SIZE	23000 /* size of the data box on one dimension            */

/* descriptors for single atom in the tree */
typedef struct atomdesc {
	double x_pos;
	double y_pos;
	double z_pos;
} atom;

typedef struct hist_entry{
	//float min;
	//float max;
	unsigned long long d_cnt;   /* need a long long type as the count might be huge */
} bucket;

typedef struct hist_entry_small{
	//float min;
	//float max;
	unsigned int d_cnt;   /* internal smaller bucket for improving algorithm */
} bucket_s;


bucket * histogram;		/* list of all buckets in the histogram   */
bucket * old_histogram; /* first histogram after second PDH runs  */
long long	PDH_acnt;	/* total number of data points            */
int num_buckets;		/* total number of buckets in the histogram */
double   PDH_res;		/* value of w                             */
atom * atom_list;		/* list of all data points                */
int BLOCK_SIZE;

/* These are for an old way of tracking time */
struct timezone Idunno;	
struct timeval startTime, endTime;

/* 
	distance of two points in the atom_list 
*/
double p2p_distance(int ind1, int ind2) {
	
	double x1 = atom_list[ind1].x_pos;
	double x2 = atom_list[ind2].x_pos;
	double y1 = atom_list[ind1].y_pos;
	double y2 = atom_list[ind2].y_pos;
	double z1 = atom_list[ind1].z_pos;
	double z2 = atom_list[ind2].z_pos;
		
	return sqrt((x1 - x2)*(x1-x2) + (y1 - y2)*(y1 - y2) + (z1 - z2)*(z1 - z2));
}

/* 
	distance of two points in the device atom list
*/
__device__ double p2p_distance_device(int ind1, int ind2, atom* d_atom_list) {
	
	double x1 = d_atom_list[ind1].x_pos;
	double x2 = d_atom_list[ind2].x_pos;
	double y1 = d_atom_list[ind1].y_pos;
	double y2 = d_atom_list[ind2].y_pos;
	double z1 = d_atom_list[ind1].z_pos;
	double z2 = d_atom_list[ind2].z_pos;
		
	return sqrt((x1 - x2)*(x1-x2) + (y1 - y2)*(y1 - y2) + (z1 - z2)*(z1 - z2));
}
__device__ double atom_p2p_distance_device(atom a1, int ind2, atom* d_atom_list) {
	
	double x1 = a1.x_pos;
	double x2 = d_atom_list[ind2].x_pos;
	double y1 = a1.y_pos;
	double y2 = d_atom_list[ind2].y_pos;
	double z1 = a1.z_pos;
	double z2 = d_atom_list[ind2].z_pos;
		
	return sqrt((x1 - x2)*(x1-x2) + (y1 - y2)*(y1 - y2) + (z1 - z2)*(z1 - z2));
}

/* 
	brute-force SDH solution in a single CPU thread 
*/
int PDH_baseline() {
	int i, j, h_pos;
	double dist;
	
	for(i = 0; i < PDH_acnt; i++) {
		for(j = i+1; j < PDH_acnt; j++) {
			dist = p2p_distance(i,j);
			h_pos = (int) (dist / PDH_res);
			histogram[h_pos].d_cnt++;
		} 
	}
	return 0;
}



/*
	CUDA SDH solutions
*/

__global__ void PDH_CUDA_kernel1(atom * d_atom_list, bucket * d_histogram, long long d_PDH_acnt, double d_PDH_res)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x, j, h_pos;
	double dist;
	if (i < d_PDH_acnt)
	{
		for(j = i+1; j < d_PDH_acnt; j++) {
			dist = p2p_distance_device(i,j,d_atom_list);
			h_pos = (int) (dist / d_PDH_res);
			atomicAdd(&(d_histogram[h_pos].d_cnt),1);
		}
	}
}
__global__ void PDH_CUDA_kernel2(atom * d_atom_list, bucket * d_histogram, long long d_PDH_acnt, double d_PDH_res)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x, j, k, h_pos;
	extern __shared__ atom s_atom_list[];
	double dist;
	atom local_entry;
	if (i < d_PDH_acnt)
		local_entry = d_atom_list[i];
	for(j = blockIdx.x+1; j < gridDim.x; ++j)
	{
		if(threadIdx.x+j*blockDim.x < d_PDH_acnt)
			s_atom_list[threadIdx.x] = d_atom_list[threadIdx.x+j*blockDim.x];
		__syncthreads();
		for(k = 0; k < blockDim.x; ++k)
		{
			if(k+j*blockDim.x < d_PDH_acnt)
			{
				dist = atom_p2p_distance_device(local_entry,k,s_atom_list);
				h_pos = (int) (dist / d_PDH_res);
				atomicAdd(&(d_histogram[h_pos].d_cnt),1);
			}
		}
		__syncthreads();	
	}
	if (i < d_PDH_acnt)
		s_atom_list[threadIdx.x] = local_entry;
	__syncthreads();
	
	for(j = threadIdx.x+1; j < blockDim.x; ++j)
	{
		if(j + blockIdx.x * blockDim.x < d_PDH_acnt)
		{
			dist = atom_p2p_distance_device(local_entry,j,s_atom_list);
			h_pos = (int) (dist / d_PDH_res);
			atomicAdd(&(d_histogram[h_pos].d_cnt),1);
		}
	}
}
__global__ void PDH_CUDA_kernel3(atom * d_atom_list, bucket * d_histogram, long long d_PDH_acnt, double d_PDH_res, int d_num_buckets)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x, j, h_pos;
	extern __shared__ bucket s_histogram[];
	double dist;
	bucket null_bucket;
	null_bucket.d_cnt = 0;
	for(j = 0; j < d_num_buckets; j+=blockDim.x)
		if(threadIdx.x + j < d_num_buckets)
			s_histogram[j+threadIdx.x] = null_bucket;
	__syncthreads();
	if (i < d_PDH_acnt)
	{
		for(j = i+1; j < d_PDH_acnt; j++) {
			dist = p2p_distance_device(i,j,d_atom_list);
			h_pos = (int) (dist / d_PDH_res);
			atomicAdd(&(s_histogram[h_pos].d_cnt),1);
		}
	}
	__syncthreads();
	for(j = 0; j < d_num_buckets; j+=blockDim.x)
		if(threadIdx.x + j < d_num_buckets)
			atomicAdd(&(d_histogram[j+threadIdx.x].d_cnt),s_histogram[j+threadIdx.x].d_cnt);
}


__global__ void PDH_CUDA_kernel4(atom * d_atom_list, bucket * d_histogram, long long d_PDH_acnt, double d_PDH_res, int d_num_buckets)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x, j, k, h_pos;
	extern __shared__ atom s_base_array[];
	atom* s_atom_list = s_base_array;
	bucket* s_histogram = (bucket*)&s_atom_list[blockDim.x];
	double dist;
	atom local_entry;
	bucket null_bucket;
	null_bucket.d_cnt = 0;
	for(j = 0; j < d_num_buckets; j+=blockDim.x)
		if(threadIdx.x + j < d_num_buckets)
			s_histogram[j+threadIdx.x] = null_bucket;
	__syncthreads();
	if (i < d_PDH_acnt)
		local_entry = d_atom_list[i];
	for(j = blockIdx.x+1; j < gridDim.x; ++j)
	{
		if(threadIdx.x+j*blockDim.x < d_PDH_acnt)
			s_atom_list[threadIdx.x] = d_atom_list[threadIdx.x+j*blockDim.x];
		__syncthreads();
		for(k = 0; k < blockDim.x; ++k)
		{
			if(k+j*blockDim.x < d_PDH_acnt)
			{
				dist = atom_p2p_distance_device(local_entry,k,s_atom_list);
				h_pos = (int) (dist / d_PDH_res);
				atomicAdd(&(s_histogram[h_pos].d_cnt),1);
			}
		}
		__syncthreads();	
	}
	if (i < d_PDH_acnt)
		s_atom_list[threadIdx.x] = local_entry;
	__syncthreads();
	
	for(j = threadIdx.x+1; j < blockDim.x; ++j)
	{
		if(j + blockIdx.x * blockDim.x < d_PDH_acnt)
		{
			dist = atom_p2p_distance_device(local_entry,j,s_atom_list);
			h_pos = (int) (dist / d_PDH_res);
			atomicAdd(&(s_histogram[h_pos].d_cnt),1);
		}
	}
	__syncthreads();
	for(j = 0; j < d_num_buckets; j+=blockDim.x)
		if(threadIdx.x + j < d_num_buckets)
			atomicAdd(&(d_histogram[j+threadIdx.x].d_cnt),s_histogram[j+threadIdx.x].d_cnt);
}

__global__ void PDH_CUDA_kernel5(atom * d_atom_list, bucket * d_histogram, long long d_PDH_acnt, double d_PDH_res, int d_num_buckets)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x, j, k, h_pos, offset, c;
	extern __shared__ atom s_base_array[];
	atom* s_atom_list = s_base_array;
	bucket_s* s_histogram = (bucket_s*)&s_atom_list[blockDim.x];
	double dist;
	atom local_entry;
	bucket_s null_bucket;
	null_bucket.d_cnt = 0;
	for(j = 0; j < d_num_buckets; j+=blockDim.x)
		if(threadIdx.x + j < d_num_buckets)
			s_histogram[j+threadIdx.x] = null_bucket;
	__syncthreads();
	if (i < d_PDH_acnt)
		local_entry = d_atom_list[i];
	offset = 1;
	if(gridDim.x % 2 == 0)
		if(blockIdx.x >= gridDim.x/2)
			offset = 0;
	for(j = (blockIdx.x+1)%gridDim.x, c = 1; c < gridDim.x/2 + offset; j=(j+1)%gridDim.x, ++c)
	{
		if(threadIdx.x+j*blockDim.x < d_PDH_acnt)
			s_atom_list[threadIdx.x] = d_atom_list[threadIdx.x+j*blockDim.x];
		__syncthreads();
		for(k = 0; k < blockDim.x; ++k)
		{
			if(k+j*blockDim.x < d_PDH_acnt && i < d_PDH_acnt)
			{
				dist = atom_p2p_distance_device(local_entry,k,s_atom_list);
				h_pos = (int) (dist / d_PDH_res);
				atomicAdd(&(s_histogram[h_pos].d_cnt),1);
			}
		}
		__syncthreads();
	}
	if (i < d_PDH_acnt)
		s_atom_list[threadIdx.x] = local_entry;
	__syncthreads();
	
	for(j = threadIdx.x+1; j < blockDim.x; ++j)
	{
		if(j + blockIdx.x * blockDim.x < d_PDH_acnt)
		{
			dist = atom_p2p_distance_device(local_entry,j,s_atom_list);
			h_pos = (int) (dist / d_PDH_res);
			atomicAdd(&(s_histogram[h_pos].d_cnt),1);
		}
	}
	__syncthreads();
	for(j = 0; j < d_num_buckets; j+=blockDim.x)
		if(threadIdx.x + j < d_num_buckets)
			atomicAdd(&(d_histogram[j+threadIdx.x].d_cnt),s_histogram[j+threadIdx.x].d_cnt);
}

__host__ int PDH_CUDA1()
{
	size_t size = PDH_acnt * sizeof(atom);
	size_t size_histogram = num_buckets * sizeof(bucket);
	atom * d_atom_list;
	bucket * d_histogram;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start,0);

	cudaMalloc((void**) &d_atom_list, size);
	cudaMemcpy(d_atom_list, atom_list, size, cudaMemcpyHostToDevice);

	cudaMalloc((void**) &d_histogram, size_histogram);
	cudaMemset(d_histogram, 0, size_histogram);

	dim3 DimGrid(PDH_acnt/BLOCK_SIZE, 1, 1);
	if (PDH_acnt % BLOCK_SIZE) DimGrid.x++;
	dim3 DimBlock(BLOCK_SIZE, 1, 1);

	PDH_CUDA_kernel1<<<DimGrid,DimBlock>>>(d_atom_list, d_histogram, PDH_acnt, PDH_res);
	cudaMemcpy(histogram, d_histogram, size_histogram, cudaMemcpyDeviceToHost);
	cudaEventRecord(stop,0);

	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("Running time for GPU version 1: %0.5f ms\n", elapsedTime);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	cudaFree(d_atom_list);
	cudaFree(d_histogram);

	return 0;
}



__host__ int PDH_CUDA2()
{
	size_t size = PDH_acnt * sizeof(atom);
	size_t size_histogram = num_buckets * sizeof(bucket);
	atom * d_atom_list;
	bucket * d_histogram;

	cudaMalloc((void**) &d_atom_list, size);
	cudaMemcpy(d_atom_list, atom_list, size, cudaMemcpyHostToDevice);

	cudaMalloc((void**) &d_histogram, size_histogram);
	cudaMemset(d_histogram, 0, size_histogram);

	dim3 DimGrid(PDH_acnt/BLOCK_SIZE, 1, 1);
	if (PDH_acnt % BLOCK_SIZE) DimGrid.x++;
	dim3 DimBlock(BLOCK_SIZE, 1, 1);
	size_t shared_memory_size = sizeof(atom)*BLOCK_SIZE;

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start,0);
	PDH_CUDA_kernel2<<<DimGrid,DimBlock,shared_memory_size>>>(d_atom_list, d_histogram, PDH_acnt, PDH_res);
	cudaEventRecord(stop,0);

	cudaMemcpy(histogram, d_histogram, size_histogram, cudaMemcpyDeviceToHost);
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("Running time for GPU version 2: %0.5f ms\n", elapsedTime);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	cudaFree(d_atom_list);
	cudaFree(d_histogram);

	return 0;
}



__host__ int PDH_CUDA3()
{
	size_t size = PDH_acnt * sizeof(atom);
	size_t size_histogram = num_buckets * sizeof(bucket);
	atom * d_atom_list;
	bucket * d_histogram;

	cudaMalloc((void**) &d_atom_list, size);
	cudaMemcpy(d_atom_list, atom_list, size, cudaMemcpyHostToDevice);

	cudaMalloc((void**) &d_histogram, size_histogram);
	cudaMemset(d_histogram, 0, size_histogram);

	dim3 DimGrid(PDH_acnt/BLOCK_SIZE, 1, 1);
	if (PDH_acnt % BLOCK_SIZE) DimGrid.x++;
	dim3 DimBlock(BLOCK_SIZE, 1, 1);
	size_t shared_memory_size = sizeof(bucket)*num_buckets;

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start,0);
	PDH_CUDA_kernel3<<<DimGrid,DimBlock,shared_memory_size>>>(d_atom_list, d_histogram, PDH_acnt, PDH_res, num_buckets);
	cudaEventRecord(stop,0);

	cudaMemcpy(histogram, d_histogram, size_histogram, cudaMemcpyDeviceToHost);
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("Running time for GPU version 3: %0.5f ms\n", elapsedTime);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	cudaFree(d_atom_list);
	cudaFree(d_histogram);

	return 0;
}



__host__ int PDH_CUDA4()
{
	size_t size = PDH_acnt * sizeof(atom);
	size_t size_histogram = num_buckets * sizeof(bucket);
	atom * d_atom_list;
	bucket * d_histogram;

	cudaMalloc((void**) &d_atom_list, size);
	cudaMemcpy(d_atom_list, atom_list, size, cudaMemcpyHostToDevice);

	cudaMalloc((void**) &d_histogram, size_histogram);
	cudaMemset(d_histogram, 0, size_histogram);

	dim3 DimGrid(PDH_acnt/BLOCK_SIZE, 1, 1);
	if (PDH_acnt % BLOCK_SIZE) DimGrid.x++;
	dim3 DimBlock(BLOCK_SIZE, 1, 1);
	size_t shared_memory_tile_size = sizeof(atom)*BLOCK_SIZE;
	size_t shared_memory_histogram_size = sizeof(bucket)*num_buckets;
	size_t shared_memory_size = shared_memory_tile_size + shared_memory_histogram_size;

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start,0);
	PDH_CUDA_kernel4<<<DimGrid,DimBlock,shared_memory_size>>>(d_atom_list, d_histogram, PDH_acnt, PDH_res, num_buckets);
	cudaEventRecord(stop,0);

	cudaMemcpy(histogram, d_histogram, size_histogram, cudaMemcpyDeviceToHost);
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("Running time for GPU version 4: %0.5f ms\n", elapsedTime);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	cudaFree(d_atom_list);
	cudaFree(d_histogram);

	return 0;
}


__host__ float PDH_CUDA5()
{
	size_t size = PDH_acnt * sizeof(atom);
	size_t size_histogram = num_buckets * sizeof(bucket);
	atom * d_atom_list;
	bucket * d_histogram;

	cudaMalloc((void**) &d_atom_list, size);
	cudaMemcpy(d_atom_list, atom_list, size, cudaMemcpyHostToDevice);

	cudaMalloc((void**) &d_histogram, size_histogram);
	cudaMemset(d_histogram, 0, size_histogram);

	dim3 DimGrid(PDH_acnt/BLOCK_SIZE, 1, 1);
	if (PDH_acnt % BLOCK_SIZE) DimGrid.x++;
	dim3 DimBlock(BLOCK_SIZE, 1, 1);
	size_t shared_memory_tile_size = sizeof(atom)*BLOCK_SIZE;
	size_t shared_memory_histogram_size = sizeof(bucket_s)*num_buckets;
	size_t shared_memory_size = shared_memory_tile_size + shared_memory_histogram_size;

	/* Run time measurement start */
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start,0);
	PDH_CUDA_kernel5<<<DimGrid,DimBlock,shared_memory_size>>>(d_atom_list, d_histogram, PDH_acnt, PDH_res, num_buckets);
	cudaEventRecord(stop,0);

	cudaMemcpy(histogram, d_histogram, size_histogram, cudaMemcpyDeviceToHost);
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	cudaFree(d_atom_list);
	cudaFree(d_histogram);

	return elapsedTime;
}




/* 
	set a checkpoint and show the (natural) running time in seconds 
*/
double report_running_time() {
	long sec_diff, usec_diff;
	gettimeofday(&endTime, &Idunno);
	sec_diff = endTime.tv_sec - startTime.tv_sec;
	usec_diff= endTime.tv_usec-startTime.tv_usec;
	if(usec_diff < 0) {
		sec_diff --;
		usec_diff += 1000000;
	}
	printf("Running time for CPU version: %ld.%06ld\n", sec_diff, usec_diff);
	return (double)(sec_diff*1.0 + usec_diff/1000000.0);
}


/* 
	print the counts in all buckets of the histogram 
*/
void output_histogram(){
	int i; 
	long long total_cnt = 0;
	for(i=0; i< num_buckets; i++) {
		if(i%5 == 0) /* we print 5 buckets in a row */
			printf("\n%02d: ", i);
		printf("%15lld ", histogram[i].d_cnt);
		total_cnt += histogram[i].d_cnt;
	  	/* we also want to make sure the total distance count is correct */
		if(i == num_buckets - 1)	
			printf("\n T:%lld \n", total_cnt);
		else printf("| ");
	}
}


/* 
	print the counts in all buckets of the difference between the CPU histogram and the GPU histogram
*/
void output_histogram_difference()
{
	int i; 
	long long total_cnt = 0;
	for(i=0; i< num_buckets; i++) {
		if(i%5 == 0) /* we print 5 buckets in a row */
			printf("\n%02d: ", i);
		printf("%15lld ", old_histogram[i].d_cnt - histogram[i].d_cnt);
		total_cnt += old_histogram[i].d_cnt - histogram[i].d_cnt;
	  	/* we also want to make sure the total distance count is correct */
		if(i == num_buckets - 1)	
			printf("\n T:%lld \n", total_cnt);
		else printf("| ");
	}
}


int main(int argc, char **argv)
{
	int i;

	PDH_acnt = atoi(argv[1]);
	PDH_res	 = atof(argv[2]);
	BLOCK_SIZE = atof(argv[3]);
//printf("args are %d and %f\n", PDH_acnt, PDH_res);

	num_buckets = (int)(BOX_SIZE * 1.732 / PDH_res) + 1;
	histogram = (bucket *)malloc(sizeof(bucket)*num_buckets);
	memset(histogram, 0, sizeof(bucket)*num_buckets);
	old_histogram = (bucket *)malloc(sizeof(bucket)*num_buckets);

	atom_list = (atom *)malloc(sizeof(atom)*PDH_acnt);

	
	srand(1);
	/* generate data following a uniform distribution */
	for(i = 0;  i < PDH_acnt; i++) {
		atom_list[i].x_pos = ((double)(rand()) / RAND_MAX) * BOX_SIZE;
		atom_list[i].y_pos = ((double)(rand()) / RAND_MAX) * BOX_SIZE;
		atom_list[i].z_pos = ((double)(rand()) / RAND_MAX) * BOX_SIZE;
	}
	/* start counting time */
	gettimeofday(&startTime, &Idunno);
	
	/* call CPU single thread version to compute the histogram */
	PDH_baseline();
	
	/* check the total running time */ 
	report_running_time();
	
	/* print out the histogram */
	output_histogram();

	/* store old histogram */
	memcpy(old_histogram, histogram, sizeof(bucket)*num_buckets);
	memset(histogram, 0, sizeof(bucket)*num_buckets);
	
	/* call CUDA version to compute the histogram */
	PDH_CUDA1();
	PDH_CUDA2();
	PDH_CUDA3();
	PDH_CUDA4();
	PDH_CUDA5();
	float elapsedTime = PDH_CUDA5();

	output_histogram();
	output_histogram_difference();
	
	printf("Total Running Time of Kernel: %0.6f sec\n", elapsedTime/1000);

	free(histogram);
	free(old_histogram);
	free(atom_list);
	
	return 0;
}


