#include "gpuvar.h"
#include "gpufun.h"


__global__ __inline__ void calculateVec3Len(float* vec, float* len, int vecNum)
{
	unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadid >= vecNum) return;

	const float x = vec[threadid * 3];
	const float y = vec[threadid * 3 + 1];
	const float z = vec[threadid * 3 + 2];
	// 优化：使用单精度
	len[threadid] = sqrtf(x * x + y * y + z * z);
}

//计算初始状态
int runcalculateST(float damping, float dt) {
	//每个block中的线程数
	int  threadNum = 128;
	//每个grid中的block数(为了保证)
	int blockNum = (tetVertNum_d + threadNum - 1) / threadNum;

	//并行计算
	calculateST << <blockNum, threadNum >> > (tetVertPos_d, tetVertVelocity_d, 
		tetVertExternForce_d, 
		tetVertPos_old_d, tetVertPos_prev_d, tetVertPos_last_d, 
		tetVertFixed_d, 
		tetVertNum_d, gravityX_d, gravityY_d, gravityZ_d, damping, dt);
	//cudaDeviceSynchronize();//cuda中核函数的执行都是异步的，加上这一步保证核函数完全执行，或者加上memcpy(cudamemcpy是同步的)
	printCudaError("runcalculateST");
	return 0;
}
// 优化：寄存器缓存 + 内存合并，978ms -> 918ms
__global__ void calculateST(float* positions, float* velocity, float* externForce,
	float* old_positions, float* prev_positions, float* last_Positions, float* fixed,
	int vertexNum, float gravityX, float gravityY, float gravityZ, float damping, float dt)
{
	const int threadid = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadid >= vertexNum) return;

	const int indexX = threadid * 3;
	const int indexY = threadid * 3 + 1;
	const int indexZ = threadid * 3 + 2;

	float positionsX = positions[indexX];
	float positionsY = positions[indexY];
	float positionsZ = positions[indexZ];

	last_Positions[indexX] = positionsX;
	last_Positions[indexY] = positionsY;
	last_Positions[indexZ] = positionsZ;

	float fixflag = fixed[threadid] < 1e8f ? 1 : 0;
	float velocityX = velocity[indexX];
	float velocityY = velocity[indexY];
	float velocityZ = velocity[indexZ];
	velocityX = (velocityX * damping + dt * (gravityX + externForce[indexX])) * fixflag;
	velocityY = (velocityY * damping + dt * (gravityY + externForce[indexY])) * fixflag;
	velocityZ = (velocityZ * damping + dt * (gravityZ + externForce[indexZ])) * fixflag;

	velocity[indexX] = velocityX;
	velocity[indexY] = velocityY;
	velocity[indexZ] = velocityZ;
	// 更新位置
	// st
	prev_positions[indexX] = old_positions[indexX] = positions[indexX] = positionsX + velocityX * dt;
	prev_positions[indexY] = old_positions[indexY] = positions[indexY] = positionsY + velocityY * dt;
	prev_positions[indexZ] = old_positions[indexZ] = positions[indexZ] = positionsZ + velocityZ * dt;

	// 外力清零
	externForce[indexX] = externForce[indexY] = externForce[indexZ] = 0.0f;
}

//清空碰撞标记，和碰撞项的对角元素
int runClearCollision() {
	cudaMemset(tetVertisCollide_d, 0, tetVertNum_d * sizeof(unsigned char));
	
	cudaMemset(tetVertCollisionForce_d, 0.0f, tetVertNum_d * 3 * sizeof(float));
	cudaMemset(tetVertCollisionDiag_d, 0.0f, tetVertNum_d * 3 * sizeof(float));
	cudaMemset(tetVertInsertionDepth_d, 0.0f, tetVertNum_d * sizeof(float));

	printCudaError("runClearCollision");
	return 0;
}

int runClearForce()
{
	cudaMemset(tetVertForce_d, 0.0f, tetVertNum_d * 3 * sizeof(float));
	printCudaError("runClearForce");
	return 0;
}

int runCalculateTetEdgeSpringConstraint()
{
	int threadNum = 128;
	int blockNum = (tetSpringNum_d + threadNum - 1) / threadNum;
	//printf("tetSpringNum_d:%d\n", tetSpringNum_d);
	calculateTetEdgeSpringConstraint << <blockNum, threadNum >> > (
		tetVertPos_d,
		tetVertForce_d,
		tetSpringStiffness_d, tetSpringOrgLen_d, tetSpringIndex_d,
		tetSpringNum_d);
	//cudaDeviceSynchronize();
	printCudaError("runCalculateTetEdgeSpringConstraint");
	return 0;
}
// 优化：计算次数，396ms->135ms
__global__ void calculateTetEdgeSpringConstraint(
	float* positions, 
	float* force, 
	float* springStiffness, float* springOrigin, int * springIndex, 
	int springNum)
{
	const int threadid = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadid >= springNum) return;

	const int vIndex0 = springIndex[threadid * 2] * 3;
	const int vIndex1 = springIndex[threadid * 2 + 1] * 3;

	float d[3];
	for (int i = 0;i < 3;i++)
	{
		d[i] = positions[vIndex0 + i] - positions[vIndex1 + i];
	}

	float increment = (springOrigin[threadid] / sqrtf(d[0] * d[0] + d[1] * d[1] + d[2] * d[2])) - 1.0;
	if (increment < 0.0) return;

	const float k_scale = springStiffness[threadid] * increment;

	for (int i = 0;i < 3;i++)
	{
		increment = k_scale * d[i];
		atomicAdd(&force[vIndex0 + i], increment);
		atomicAdd(&force[vIndex1 + i], -increment);
	}
}

int runcalculateIF() {
	// 优化：低threadNum
	int  threadNum = 32;
	int blockNum = (tetNum_d + threadNum - 1) / threadNum;
	//并行计算
	calculateIF << <blockNum, threadNum >> > (tetVertPos_d, tetIndex_d,
		tetInvD3x3_d, tetInvD3x4_d,
		tetVertForce_d, tetVolume_d, tetActive_d,
		tetNum_d, tetStiffness_d);

	cudaStream_t stream;
	cudaStreamCreate(&stream);
	blockNum = (tetVertNum_d + threadNum - 1) / threadNum;
	calculateVec3Len << <blockNum, threadNum, 0, stream>> > (tetVertForce_d, tetVertForceLen_d, tetVertNum_d);
	cudaStreamDestroy(stream);
	//cudaDeviceSynchronize();
	printCudaError("runcalculateIF");
	return 0;
}

///计算每个顶点的restpos约束
int runcalculateRestPos() {
	int  threadNum = 128;
	int blockNum = (tetNum_d + threadNum - 1) / threadNum;
	calculateRestPosStiffness << <blockNum, threadNum >> > (
		toolPositionAndDirection_d, toolCollideFlag_d, tetVertPos_d, tetVertisCollide_d, tetVertRestStiffness_d, 1, tetVertNum_d
		);
	calculateRestPos << <blockNum, threadNum >> > (
		tetVertPos_d, tetVertRestPos_d, 
		tetVertCollisionForce_d, tetVertCollisionDiag_d, 
		tetVertRestStiffness_d, tetVertNum_d);
	//calculateRestPosCombined << <blockNum, threadNum >> > (
	//	toolPositionAndDirection_d,   // ballPos
	//	toolCollideFlag_d,            // toolCollideFlag
	//	tetVertPos_d,                 // positions
	//	tetVertisCollide_d,           // isCollide
	//	tetVertRestPos_d,             // rest_positions
	//	tetVertCollisionForce_d,      // force
	//	tetVertCollisionDiag_d,       // collisionDiag
	//	tetVertRestStiffness_d,       // restStiffness
	//	1,                            // toolNum
	//	tetVertNum_d                  // vertexNum
	//);

	//cudaDeviceSynchronize();
	printCudaError("runcalculateRestPos");
	return 0;
 }
// 优化：加速计算，97ms -> 90ms
__global__ void calculateRestPosStiffness(float* ballPos, unsigned char* toolCollideFlag, float* positions, unsigned char* isCollide, float* reststiffness, int toolNum, int vertexNum) {
	const int threadid = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadid >= vertexNum) return;
	//根据与工具的距离或者碰撞信息，计算restpos刚度系数
	const float maxStiffness = 200.0;
	if (toolCollideFlag[threadid] == 0) //未与工具发生碰撞
	{
		reststiffness[threadid] = maxStiffness;
		return;
	}
	else if (isCollide[threadid]) //与工具发生碰撞 + 按压点、夹取点：和工具直接碰撞的顶点
	{
		reststiffness[threadid] = 0.0;
		return;
	}
	float distance = 1e9 + 7;  //计算顶点到两个工具最近的距离
	const float p[3] = { positions[threadid * 3], positions[threadid * 3 + 1], positions[threadid * 3 + 2] };
	for (int i = 0; i < toolNum * 3; i += 3)
	{
		float dir[3] = { ballPos[i] - p[0], ballPos[i + 1] - p[1], ballPos[i + 2] - p[2] };
		float distSq = dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2];
		if (distSq < distance) {
			distance = distSq;
		}
	}
	//非碰撞点，根据顶点到工具的距离计算不同的刚度系数
	reststiffness[threadid] = 0.5 * maxStiffness * (sqrtf(distance) - 0.5);
#ifdef OUTPUT_INFO
	if (threadid == LOOK_THREAD)
		printf("calculateRestStiffness isCollide:%d, stiffness:%f\n", isCollide[LOOK_THREAD], reststiffness[LOOK_THREAD]);
#endif
}
// 优化：简化算法，17ms->15ms
__global__ void calculateRestPosStiffnessWithMesh_part(
	float* ballPos, float  ballRadius,
	unsigned char* toolCollideFlag, float* positions,
	unsigned char* isCollide, float* meshStiffness,
	int toolNum, int* sortedTetVertIndices, int startIdx, int activeElementNum)
{
	const int threadid = blockIdx.x * blockDim.x + threadIdx.x;
	//根据与工具的距离或者碰撞信息，计算restpos刚度系数
	if (threadid >= activeElementNum) return;

	const int tetVertIdx = sortedTetVertIndices[startIdx + threadid];
	const int indexX = tetVertIdx * 3;
	const int indexY = tetVertIdx * 3 + 1;
	const int indexZ = tetVertIdx * 3 + 2;

	float disSq = 1e9 + 7;  //计算顶点到两个工具最近的距离
	const float p[3] = { positions[indexX], positions[indexY], positions[indexZ] };
	for (int i = 0; i < toolNum; i++)
	{
		float dir[3] = { ballPos[0] - p[0], ballPos[1] - p[1], ballPos[2] - p[2] };
		float d = dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2];
		if (d < disSq) disSq = d;
	}
	//夹取点，和工具直接碰撞的顶点
	//如果没有和工具发生碰撞，布料不对四面体产生约束力
	meshStiffness[tetVertIdx] = 0.0;
	const float maxStiffness = 1000;
	for (int i = 0; i < toolNum; i++)
	{
		if (toolCollideFlag[i] > 0) //与工具发生碰撞
		{
			switch (isCollide[tetVertIdx])
			{
			case 1: //按压点，和工具直接发生碰撞的顶点
				meshStiffness[tetVertIdx] = maxStiffness;
				break;
			case 0: //非碰撞点，根据顶点到工具的距离计算不同的刚度系数
				float k = 1 / (1 + exp(10 * sqrtf(disSq) - 5));
				meshStiffness[threadid] = k * maxStiffness;
				break;
			}
			return;
		}
	}
#ifdef OUTPUT_INFO
	//if (threadid == LOOK_THREAD)
		//printf("calculateRestStiffness isCollide:%d, stiffness:%f\n", isCollide[LOOK_THREAD], reststiffness[LOOK_THREAD]);
#endif
}
// 优化：618ms->53ms
__global__ void calculateRestPosStiffnessWithMesh(float* ballPos, unsigned char* toolCollideFlag, float* positions, unsigned char* isCollide, float* meshStiffness, int toolNum, int vertexNum) 
{
	const int threadid = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadid >= vertexNum) return;

	float maxStiffness = 50000, distance = 1e9 + 7;//计算顶点到两个工具最近的距离
	const int indexX = threadid * 3;
	const int indexY = threadid * 3 + 1;
	const int indexZ = threadid * 3 + 2;
	const float p[3] = { positions[indexX], positions[indexY], positions[indexZ] };
	for (int i = 0; i < toolNum; i++)
	{
		float dir[3] = { ballPos[0] - p[0], ballPos[1] - p[1], ballPos[2] - p[2] };
		float d = dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2];
		if (d < distance) distance = d;
	}
	//如果没有和工具发生碰撞，布料不对四面体产生约束力
	//夹取点，和工具直接碰撞的顶点
	meshStiffness[threadid] = 0.0;
	for (int i = 0; i < toolNum; i++)
	{
		if (toolCollideFlag[i] > 0) //与工具发生碰撞
		{
			switch (isCollide[threadid])
			{
			case 1: //按压点，和工具直接发生碰撞的顶点
				meshStiffness[threadid] = maxStiffness;
				break;
			case 0://非碰撞点，根据顶点到工具的距离计算不同的刚度系数
				float k = 1 / (1 + exp(10 * sqrtf(distance) - 5));
				meshStiffness[threadid] = k * maxStiffness;
				break;
			}
			return;
		}
	}
#ifdef OUTPUT_INFO
	//if (threadid == LOOK_THREAD)
		//printf("calculateRestStiffness isCollide:%d, stiffness:%f\n", isCollide[LOOK_THREAD], reststiffness[LOOK_THREAD]);
#endif
}
__global__ void calculateRestPosCombined(
	float* ballPos,
	unsigned char* toolCollideFlag,
	float* positions,
	unsigned char* isCollide,
	float* rest_positions,
	float* force,
	float* collisionDiag,
	float* restStiffness,
	int toolNum,
	int vertexNum)
{
	const int threadid = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadid >= vertexNum) return;

	const int offset = threadid * 3;
	// Step 1: 计算刚度系数（原 calculateRestPosStiffness 的逻辑）
	const float maxStiffness = 200.0f;
	float stiffness;

	if (toolCollideFlag[threadid] == 0) {
		stiffness = maxStiffness;
	}
	else if (isCollide[threadid]) {
		stiffness = 0.0f;
	}
	else {
		float distance = 1e9f;
		const float p[3] = {
			positions[offset],
			positions[offset + 1],
			positions[offset + 2]
		};
		for (int i = 0; i < toolNum * 3; i += 3) {
			float dx = ballPos[i] - p[0];
			float dy = ballPos[i + 1] - p[1];
			float dz = ballPos[i + 2] - p[2];
			float distSq = dx * dx + dy * dy + dz * dz;
			if (distSq < distance) {
				distance = distSq;
			}
		}
		stiffness = 0.5f * maxStiffness * (sqrtf(distance) - 0.5f);
	}

	// Step 2: 应用刚度到 force 和 diag（原 calculateRestPos 的逻辑）
	atomicAdd(&collisionDiag[offset], stiffness);
	atomicAdd(&collisionDiag[offset + 1], stiffness);
	atomicAdd(&collisionDiag[offset + 2], stiffness);

	atomicAdd(&force[offset], (rest_positions[offset] - positions[offset]) * stiffness);
	atomicAdd(&force[offset + 1], (rest_positions[offset + 1] - positions[offset + 1]) * stiffness);
	atomicAdd(&force[offset + 2], (rest_positions[offset + 2] - positions[offset + 2]) * stiffness);

	// 可选：保存计算出的刚度值到 restStiffness 数组中
	restStiffness[threadid] = stiffness;

#ifdef OUTPUT_INFO
	if (threadid == LOOK_THREAD) {
		printf("CombinedKernel: stiffness=%f\n", stiffness);
	}
#endif
}
// 优化：算法简化，748ms->266ms
__global__ void calculateRestPos(float* positions, float* rest_positions, float* force, float* collisionDiag, float* restStiffness, int vertexNum) {
	const int threadid = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadid >= vertexNum) return;

	const int offset = threadid * 3;
	const float stiffness = restStiffness[threadid];
	// 优化：交叉计算充分利用流水线
	atomicAdd(&collisionDiag[offset], stiffness);
	atomicAdd(&force[offset], (rest_positions[offset] - positions[offset]) * stiffness);
	atomicAdd(&collisionDiag[offset + 1], stiffness);
	atomicAdd(&force[offset + 1], (rest_positions[offset + 1] - positions[offset + 1]) * stiffness);
	atomicAdd(&collisionDiag[offset + 2], stiffness);
	atomicAdd(&force[offset + 2], (rest_positions[offset + 2] - positions[offset + 2]) * stiffness);
}
// 优化：载入寄存器
__global__ void calculateRestPos_part(float* positions, float* rest_positions, float* force, float* collisionDiag, float* restStiffness, 
	int* sortedTetVertIndices, int offset, int activeElement)
{
	const int threadid = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadid > activeElement) return;

	int tetVertIdx = sortedTetVertIndices[threadid+offset];

	//计算受力
	const float tempx = rest_positions[3 * tetVertIdx + 0] - positions[3 * tetVertIdx + 0];
	const float tempy = rest_positions[3 * tetVertIdx + 1] - positions[3 * tetVertIdx + 1];
	const float tempz = rest_positions[3 * tetVertIdx + 2] - positions[3 * tetVertIdx + 2];

	const float restStiffness_tetVertIdx = restStiffness[tetVertIdx];
	atomicAdd(force + tetVertIdx * 3 + 0, tempx * restStiffness_tetVertIdx);
	atomicAdd(force + tetVertIdx * 3 + 1, tempy * restStiffness_tetVertIdx);
	atomicAdd(force + tetVertIdx * 3 + 2, tempz * restStiffness_tetVertIdx);

	atomicAdd(collisionDiag + tetVertIdx * 3 + 0, restStiffness_tetVertIdx);
	atomicAdd(collisionDiag + tetVertIdx * 3 + 1, restStiffness_tetVertIdx);
	atomicAdd(collisionDiag + tetVertIdx * 3 + 2, restStiffness_tetVertIdx);
}
// 优化：循环展开，223ms->69ms
__device__ void MatrixSubstract_3_D(float* A, float* B, float* R)						//R=A-B
{
	R[0] = A[0] - B[0];R[1] = A[1] - B[1];R[2] = A[2] - B[2];
	R[3] = A[3] - B[3];R[4] = A[4] - B[4];R[5] = A[5] - B[5];
	R[6] = A[6] - B[6];R[7] = A[7] - B[7];R[8] = A[8] - B[8];
}
// 优化：加载进寄存器，216ms->54ms
__device__ void MatrixProduct_3_D(const float* A, const float* B, float* R)				//R=A*B
{
	// Load A into registers (row-wise)
	const float a0 = A[0], a1 = A[1], a2 = A[2];
	const float a3 = A[3], a4 = A[4], a5 = A[5];
	const float a6 = A[6], a7 = A[7], a8 = A[8];

	// Load B into registers (column-wise)
	const float b0 = B[0], b3 = B[3], b6 = B[6]; // Column 0
	const float b1 = B[1], b4 = B[4], b7 = B[7]; // Column 1
	const float b2 = B[2], b5 = B[5], b8 = B[8]; // Column 2

	// Compute R = A * B (row-wise multiplication)
	R[0] = a0 * b0 + a1 * b3 + a2 * b6;
	R[1] = a0 * b1 + a1 * b4 + a2 * b7;
	R[2] = a0 * b2 + a1 * b5 + a2 * b8;
	R[3] = a3 * b0 + a4 * b3 + a5 * b6;
	R[4] = a3 * b1 + a4 * b4 + a5 * b7;
	R[5] = a3 * b2 + a4 * b5 + a5 * b8;
	R[6] = a6 * b0 + a7 * b3 + a8 * b6;
	R[7] = a6 * b1 + a7 * b4 + a8 * b7;
	R[8] = a6 * b2 + a7 * b5 + a8 * b8;
}
// 优化：加载进寄存器
__device__ void MatrixProduct_3x3x4(const float* A, const float* B, float* R)				//R=A*B
{
	// 1. 显式加载 A 的所有元素到寄存器（3x3 矩阵）
	const float a00 = A[0], a01 = A[1], a02 = A[2];
	const float a10 = A[3], a11 = A[4], a12 = A[5];
	const float a20 = A[6], a21 = A[7], a22 = A[8];
	// 2. 显式加载 B 的所有元素到寄存器（3x4 矩阵）
	const float b00 = B[0], b01 = B[1], b02 = B[2], b03 = B[3];
	const float b10 = B[4], b11 = B[5], b12 = B[6], b13 = B[7];
	const float b20 = B[8], b21 = B[9], b22 = B[10], b23 = B[11];
	// 3. 计算 R = A * B（3x3 * 3x4 -> 3x4）
	R[0] = a00 * b00 + a01 * b10 + a02 * b20;
	R[1] = a00 * b01 + a01 * b11 + a02 * b21;
	R[2] = a00 * b02 + a01 * b12 + a02 * b22;
	R[3] = a00 * b03 + a01 * b13 + a02 * b23;
	R[4] = a10 * b00 + a11 * b10 + a12 * b20;
	R[5] = a10 * b01 + a11 * b11 + a12 * b21;
	R[6] = a10 * b02 + a11 * b12 + a12 * b22;
	R[7] = a10 * b03 + a11 * b13 + a12 * b23;
	R[8] = a20 * b00 + a21 * b10 + a22 * b20;
	R[9] = a20 * b01 + a21 * b11 + a22 * b21;
	R[10] = a20 * b02 + a21 * b12 + a22 * b22;
	R[11] = a20 * b03 + a21 * b13 + a22 * b23;
}
// 优化：将a加载进寄存器
__device__ __inline__ void MatrixProduct_D(float* A, float* B, float* R, int nx, int ny, int nz)	//R=A*B
{
	memset(R, 0, sizeof(float) * nx * nz);
	for (int k = 0; k < ny; k++)
		for (int i = 0; i < nx; i++)
		{
			float a = A[i * ny + k];
			for (int j = 0; j < nz; j++)
					R[i * nz + j] += a * B[k * nz + j];
		}
}
// 优化：寄存器优化
__global__ void calculateIF(float* positions, int* tetIndex,
	float* tetInvD3x3, float* tetInvD3x4,
	float* force, float* tetVolumn, bool* active,
	int tetNum, float* volumnStiffness)
{
	const int threadid = blockIdx.x * blockDim.x + threadIdx.x;
	// 优化：合并if判断
	if (threadid >= tetNum || !active[threadid]) return;

	//获取当前四面体的变形系数
	//volumnStiffness = tetStiffness_d[threadid];

	//计算每个四面体初始化的shape矩阵的逆
	const int vIndex0 = tetIndex[threadid * 4];
	const int vIndex1 = tetIndex[threadid * 4 + 1];
	const int vIndex2 = tetIndex[threadid * 4 + 2];
	const int vIndex3 = tetIndex[threadid * 4 + 3];

	const int vIndex00 = vIndex0 * 3, vIndex01 = vIndex0 * 3 + 1, vIndex02 = vIndex0 * 3 + 2;
	const int vIndex10 = vIndex1 * 3, vIndex11 = vIndex1 * 3 + 1, vIndex12 = vIndex1 * 3 + 2;
	const int vIndex20 = vIndex2 * 3, vIndex21 = vIndex2 * 3 + 1, vIndex22 = vIndex2 * 3 + 2;
	const int vIndex30 = vIndex3 * 3, vIndex31 = vIndex3 * 3 + 1, vIndex32 = vIndex3 * 3 + 2;
	//先计算shape矩阵
	const float pos00 = positions[vIndex00], pos01 = positions[vIndex01], pos02 = positions[vIndex02];
	const float D[9] = {
		positions[vIndex10] - pos00, positions[vIndex20] - pos00, positions[vIndex30] - pos00,
		positions[vIndex11] - pos01, positions[vIndex21] - pos01, positions[vIndex31] - pos01,
		positions[vIndex12] - pos02, positions[vIndex22] - pos02, positions[vIndex32] - pos02
	};
	//计算形变梯度F
	float F[9], R[9], temp[12];
	MatrixProduct_3_D(D, &tetInvD3x3[threadid * 9], F);
	//从F中分解出R
	GetRotation_D((float(*)[3])F, (float(*)[3])R);//转化为数组指针，即对应二维数组的形参要求
	//梯度减去
	#pragma unroll
	for (int i = 0;i < 9;i++) R[i] -= F[i];

	MatrixProduct_3x3x4(R, &tetInvD3x4[threadid * 12], temp);
	//对应的四个点的xyz分量
	//这里应该需要原子操作
	const float tetVolumn_volumnStiffness = tetVolumn[threadid] * volumnStiffness[threadid];
	atomicAdd(&force[vIndex00], temp[0] * tetVolumn_volumnStiffness);
	atomicAdd(&force[vIndex01], temp[4] * tetVolumn_volumnStiffness);
	atomicAdd(&force[vIndex02], temp[8] * tetVolumn_volumnStiffness);
	atomicAdd(&force[vIndex10], temp[1] * tetVolumn_volumnStiffness);
	atomicAdd(&force[vIndex11], temp[5] * tetVolumn_volumnStiffness);
	atomicAdd(&force[vIndex12], temp[9] * tetVolumn_volumnStiffness);
	atomicAdd(&force[vIndex20], temp[2] * tetVolumn_volumnStiffness);
	atomicAdd(&force[vIndex21], temp[6] * tetVolumn_volumnStiffness);
	atomicAdd(&force[vIndex22], temp[10] * tetVolumn_volumnStiffness);
	atomicAdd(&force[vIndex30], temp[3] * tetVolumn_volumnStiffness);
	atomicAdd(&force[vIndex31], temp[7] * tetVolumn_volumnStiffness);
	atomicAdd(&force[vIndex32], temp[11] * tetVolumn_volumnStiffness);
}

__global__ void calculateIF_part(float* positions, int* tetIndex,
	float* tetInvD3x3, float* tetInvD3x4,
	float* force, float* tetVolumn, float* volumnStiffness, 
	int * sortedTetIdx, int offset, int activeElementNum)
{
	unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadid >= activeElementNum) return;

	//获取当前四面体的变形系数
	//volumnStiffness = tetStiffness_d[threadid];
#ifdef OUTPUT_INFO
	if (threadid == 0)
		printf("calculateIF startIdx:%d, activeTetNum:%d\n", threadid + offset, activeElementNum);
#endif
	unsigned int tetIdx = sortedTetIdx[threadid+offset];
	//计算每个四面体初始化的shape矩阵的逆
	int vIndex0 = tetIndex[tetIdx * 4 + 0];
	int vIndex1 = tetIndex[tetIdx * 4 + 1];
	int vIndex2 = tetIndex[tetIdx * 4 + 2];
	int vIndex3 = tetIndex[tetIdx * 4 + 3];

	//先计算shape矩阵
	float D[9];
	D[0] = positions[vIndex1 * 3 + 0] - positions[vIndex0 * 3 + 0];
	D[1] = positions[vIndex2 * 3 + 0] - positions[vIndex0 * 3 + 0];
	D[2] = positions[vIndex3 * 3 + 0] - positions[vIndex0 * 3 + 0];
	D[3] = positions[vIndex1 * 3 + 1] - positions[vIndex0 * 3 + 1];
	D[4] = positions[vIndex2 * 3 + 1] - positions[vIndex0 * 3 + 1];
	D[5] = positions[vIndex3 * 3 + 1] - positions[vIndex0 * 3 + 1];
	D[6] = positions[vIndex1 * 3 + 2] - positions[vIndex0 * 3 + 2];
	D[7] = positions[vIndex2 * 3 + 2] - positions[vIndex0 * 3 + 2];
	D[8] = positions[vIndex3 * 3 + 2] - positions[vIndex0 * 3 + 2];

	//计算形变梯度F
	float F[9];
	float* B = &tetInvD3x3[tetIdx * 9];
	MatrixProduct_3_D(D, &tetInvD3x3[tetIdx * 9], F);

	//从F中分解出R（直接搬运，这个算法太复杂了）
	float R[9];
	GetRotation_D((float(*)[3])F, (float(*)[3])R);//转化为数组指针，即对应二维数组的形参要求

	MatrixSubstract_3_D(R, F, R);
	//for (int i = 0; i < 9; i++)	
	//	R[i] = R[i] - F[i];

	float temp[12];
	memset(temp, 0, sizeof(float) * 12);
	MatrixProduct_D(R, &tetInvD3x4[tetIdx * 12], temp, 3, 3, 4);

	//对应的四个点的xyz分量
	//这里应该需要原子操作
	atomicAdd(force + vIndex0 * 3 + 0, temp[0] * tetVolumn[tetIdx] * volumnStiffness[tetIdx]);
	atomicAdd(force + vIndex0 * 3 + 1, temp[4] * tetVolumn[tetIdx] * volumnStiffness[tetIdx]);
	atomicAdd(force + vIndex0 * 3 + 2, temp[8] * tetVolumn[tetIdx] * volumnStiffness[tetIdx]);

	atomicAdd(force + vIndex1 * 3 + 0, temp[1] * tetVolumn[tetIdx] * volumnStiffness[tetIdx]);
	atomicAdd(force + vIndex1 * 3 + 1, temp[5] * tetVolumn[tetIdx] * volumnStiffness[tetIdx]);
	atomicAdd(force + vIndex1 * 3 + 2, temp[9] * tetVolumn[tetIdx] * volumnStiffness[tetIdx]);

	atomicAdd(force + vIndex2 * 3 + 0, temp[2] * tetVolumn[tetIdx] * volumnStiffness[tetIdx]);
	atomicAdd(force + vIndex2 * 3 + 1, temp[6] * tetVolumn[tetIdx] * volumnStiffness[tetIdx]);
	atomicAdd(force + vIndex2 * 3 + 2, temp[10] * tetVolumn[tetIdx] * volumnStiffness[tetIdx]);

	atomicAdd(force + vIndex3 * 3 + 0, temp[3] * tetVolumn[tetIdx] * volumnStiffness[tetIdx]);
	atomicAdd(force + vIndex3 * 3 + 1, temp[7] * tetVolumn[tetIdx] * volumnStiffness[tetIdx]);
	atomicAdd(force + vIndex3 * 3 + 2, temp[11] * tetVolumn[tetIdx] * volumnStiffness[tetIdx]);
#ifdef OUTPUT_INFO
	if (vIndex0 == LOOK_THREAD)
		printf("calculateIF tetVertForce_d in calculateIF: %f %f %f\n", force[vIndex0 * 3 + 0], force[vIndex0 * 3 + 1], force[vIndex0 * 3 + 2]);
	//if (vIndex0 == 0)
	//{
	//	printf("calculateIF threadid: %d v0_temp[%f %f %f]\n",
	//		threadid, temp[0], temp[4], temp[8]);
	//	if (isnan(temp[0]) || isnan(temp[4]) || isnan(temp[8]))
	//	{
	//		unsigned int t = threadid * 12;
	//		printf("threadid: %d\n nan occured in v0, R[ %f %f %f %f %f %f %f %f %f]\n\tF[ %f %f %f %f %f %f %f %f %f]\n\tD[ %f %f %f %f %f %f %f %f %f]\n",
	//			threadid,
	//			R[0], R[1], R[2], R[3], R[4], R[5], R[6], R[7], R[8],
	//			F[0], F[1], F[2], F[3], F[4], F[5], F[6], F[7], F[8],
	//			D[0], D[1], D[2], D[3], D[4], D[5], D[6], D[7], D[8]);
	//		printf("threadid: %d\ntetInvD3x4[ %f %f %f %f \n\t%f %f %f %f\n\t %f %f %f %f]\n",
	//			threadid,
	//			tetInvD3x4[t + 0], tetInvD3x4[t + 1], tetInvD3x4[t + 2], tetInvD3x4[t + 3],
	//			tetInvD3x4[t + 4], tetInvD3x4[t + 5], tetInvD3x4[t + 6], tetInvD3x4[t + 7],
	//			tetInvD3x4[t + 8], tetInvD3x4[t + 9], tetInvD3x4[t + 10], tetInvD3x4[t + 11]);
	//	}
	//}
	//if (vIndex1 == 0)
	//{
	//	printf("calculateIF v1_temp[%f %f %f]\n",
	//		temp[1], temp[5], temp[9]);
	//	if (isnan(temp[1]) || isnan(temp[5]) || isnan(temp[9]))
	//	{
	//		printf("nan occured in v1, R[ %f %f %f %f %f %f %f %f %f]\n\tF[ %f %f %f %f %f %f %f %f %f]\n\tD[ %f %f %f %f %f %f %f %f %f]\n",
	//			R[0], R[1], R[2], R[3], R[4], R[5], R[6], R[7], R[8],
	//			F[0], F[1], F[2], F[3], F[4], F[5], F[6], F[7], F[8],
	//			D[0], D[1], D[2], D[3], D[4], D[5], D[6], D[7], D[8]);
	//	}
	//}
	//if (vIndex2 == 0)
	//{
	//	printf("calculateIF v2_temp[%f %f %f]\n",
	//		temp[2], temp[6], temp[10]);
	//	if (isnan(temp[2]) || isnan(temp[6]) || isnan(temp[10]))
	//	{
	//		printf("nan occured in v2, R[ %f %f %f %f %f %f %f %f %f]\n\tF[ %f %f %f %f %f %f %f %f %f]\n\tD[ %f %f %f %f %f %f %f %f %f]\n",
	//			R[0], R[1], R[2], R[3], R[4], R[5], R[6], R[7], R[8],
	//			F[0], F[1], F[2], F[3], F[4], F[5], F[6], F[7], F[8],
	//			D[0], D[1], D[2], D[3], D[4], D[5], D[6], D[7], D[8]);
	//	}
	//}
	//if (vIndex3 == 0)
	//{
	//	printf("calculateIF v3_temp[%f %f %f]\n",
	//		temp[3], temp[7], temp[11]);
	//	if (isnan(temp[3]) || isnan(temp[7]) || isnan(temp[11]))
	//	{
	//		printf("nan occured in v3, R[ %f %f %f %f %f %f %f %f %f]\n\tF[ %f %f %f %f %f %f %f %f %f]\n\tD[ %f %f %f %f %f %f %f %f %f]\n",
	//			R[0], R[1], R[2], R[3], R[4], R[5], R[6], R[7], R[8],
	//			F[0], F[1], F[2], F[3], F[4], F[5], F[6], F[7], F[8],
	//			D[0], D[1], D[2], D[3], D[4], D[5], D[6], D[7], D[8]);
	//	}
	//}
#endif
}
int runcalculateRestPosForceWithMeshPos(float toolRadius)
{
	int threadNum = 128;
	int blockNum = (tetVertNum_d + threadNum - 1) / threadNum;

	calculateRestPosStiffnessWithMesh << <blockNum, threadNum >> > (
		toolPositionAndDirection_d, toolCollideFlag_d,
		tetVertPos_d, tetVertisCollide_d, 
		tetVertfromTriStiffness_d, cylinderNum_d, tetVertNum_d);

	calculateRestPos << <blockNum, threadNum >> > (
		tetVertPos_d, tetVertRestPos_d,
		tetVertCollisionForce_d, tetVertCollisionDiag_d,
		tetVertfromTriStiffness_d, tetVertNum_d);
	//cudaDeviceSynchronize();
	printCudaError("runcalculateRestPosForceWithMeshPos");
	return 0;
}

__global__ void calculateRestPosForceWithMeshPos(
	float* positions, int* skeletonMesh,
	float* force, float* collisionDiag,
	float* meshPositions, unsigned char* isCollide,
	float* meshStiffness, int vertexNum)
{
	unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadid >= vertexNum) return;

	int tri_Idx = skeletonMesh[threadid];
	if (tri_Idx == -1) return; // 没有与四面体顶点绑定的表面布料点不参与计算。
	if (isCollide[threadid] == 0) return;

	float deltaPos[3];
	deltaPos[0] = meshPositions[3 * tri_Idx + 0] - positions[3 * threadid + 0];
	deltaPos[1] = meshPositions[3 * tri_Idx + 1] - positions[3 * threadid + 1];
	deltaPos[2] = meshPositions[3 * tri_Idx + 2] - positions[3 * threadid + 2];
	float d = sqrt(deltaPos[0] * deltaPos[0] + deltaPos[1] * deltaPos[1] + deltaPos[2] * deltaPos[2]);
#ifdef OUTPUT_INFO
	if (threadid == LOOK_THREAD)
	{
		printf("calculateRestPosForceWithMeshPos thread:%d deltaPos[%f %f %f]\n", threadid, deltaPos[0], deltaPos[1], deltaPos[2]);
	}
#endif
	if (d < 1e-9)
		return;
	float dir[3] = { deltaPos[0] / d, deltaPos[1] / d,deltaPos[2] / d };
	

	float forcex = deltaPos[0] * meshStiffness[threadid];
	float forcey = deltaPos[1] * meshStiffness[threadid];
	float forcez = deltaPos[2] * meshStiffness[threadid];
	force[threadid * 3 + 0] += forcex;
	force[threadid * 3 + 1] += forcey;
	force[threadid * 3 + 2] += forcez;

	collisionDiag[threadid * 3 + 0] += meshStiffness[threadid];
	collisionDiag[threadid * 3 + 1] += meshStiffness[threadid];
	collisionDiag[threadid * 3 + 2] += meshStiffness[threadid];

}

//计算position
__global__ void calculatePOS(float* positions, float* force, float* fixed, float* mass,
	float* next_positions, float* prev_positions, float* old_positions,
	float* volumnDiag, float* collisionDiag, float* collisionForce,
	int vertexNum, float dt, float omega)
{
	unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadid >= vertexNum) return;
	int indexX = threadid * 3 + 0;
	int indexY = threadid * 3 + 1;
	int indexZ = threadid * 3 + 2;

	float diagConstant = (mass[threadid] + fixed[threadid]) / (dt * dt);
	float forceX = force[indexX], forceY = force[indexY], forceZ = force[indexZ];
	float forceLen = sqrtf(forceX * forceX + forceY * forceY + forceZ * forceZ);

	//计算每个点的shape match产生的约束部分，因为之前是按照每个四面体计算的，现在要摊到每个顶点上
	float elementX = forceX + collisionForce[indexX];
	float elementY = forceY + collisionForce[indexY];
	float elementZ = forceZ + collisionForce[indexZ];

	float positionX = positions[indexX];
	float positionY = positions[indexY];
	float positionZ = positions[indexZ];

	//相当于先按重力运动，每次再在收重力的效果上再修正
	float volumnDiag_diagConstant = volumnDiag[threadid] + diagConstant;
	float nextnext_positionsX = (diagConstant * (old_positions[indexX] - positionX) + elementX) / (collisionDiag[indexX] + volumnDiag_diagConstant) + positionX;
	float nextnext_positionsY = (diagConstant * (old_positions[indexY] - positionY) + elementY) / (collisionDiag[indexY] + volumnDiag_diagConstant) + positionY;
	float nextnext_positionsZ = (diagConstant * (old_positions[indexZ] - positionZ) + elementZ) / (collisionDiag[indexZ] + volumnDiag_diagConstant) + positionZ;

	//under-relaxation 和 切比雪夫迭代
	nextnext_positionsX = (nextnext_positionsX - positionX) * 0.6 + positionX;
	nextnext_positionsY = (nextnext_positionsY - positionY) * 0.6 + positionY;
	nextnext_positionsZ = (nextnext_positionsZ - positionZ) * 0.6 + positionZ;

	// omega定义：omega = 4 / (4 - rho*rho*omega);
	float prev_positionsX = prev_positions[indexX];
	float prev_positionsY = prev_positions[indexY];
	float prev_positionsZ = prev_positions[indexZ];
	nextnext_positionsX = omega * (nextnext_positionsX - prev_positionsX) + prev_positionsX;
	nextnext_positionsY = omega * (nextnext_positionsY - prev_positionsY) + prev_positionsY;
	nextnext_positionsZ = omega * (nextnext_positionsZ - prev_positionsZ) + prev_positionsZ;

	prev_positions[indexX] = positionX;
	prev_positions[indexY] = positionY;
	prev_positions[indexZ] = positionZ;

	positions[indexX] = next_positions[indexX] = nextnext_positionsX;
	positions[indexY] = next_positions[indexY] = nextnext_positionsY;
	positions[indexZ] = next_positions[indexZ] = nextnext_positionsZ;
}

//计算更新位置
int runcalculatePOS(float omega, float dt) {
	int  threadNum = 128;
	int blockNum = (tetVertNum_d + threadNum - 1) / threadNum;
	//并行计算
	calculatePOS << <blockNum, threadNum >> > (tetVertPos_d, tetVertForce_d,
		tetVertFixed_d, tetVertMass_d,
		tetVertPos_next_d, tetVertPos_prev_d, tetVertPos_old_d,
		tetVolumeDiag_d, tetVertCollisionDiag_d, tetVertCollisionForce_d,
		tetVertNum_d, dt, omega);
	//cudaDeviceSynchronize();
	printCudaError("runcalculatePOS");
	return 0;
}

//计算position
__global__ void calculatePOS(float* positions, float* force, float* fixed, float* mass,
	float* next_positions, float* prev_positions, float* old_positions,
	float* volumnDiag, float* collisionDiag, float* collisionForce,
	int* sortedIndices, int offset, int activeElementNum, float dt, float omega)
{
	const int threadid = blockIdx.x * blockDim.x + threadIdx.x;
	const int vertIdx = sortedIndices[offset + threadid];
	if (threadid >= activeElementNum || vertIdx == GRABED_TETIDX) return;

	int indexX = vertIdx * 3 + 0;
	int indexY = vertIdx * 3 + 1;
	int indexZ = vertIdx * 3 + 2;

	float diagConstant = (mass[threadid] + fixed[threadid]) / (dt * dt);
	float forceX = force[indexX], forceY = force[indexY], forceZ = force[indexZ];
	float forceLen = sqrtf(forceX * forceX + forceY * forceY + forceZ * forceZ);

	//计算每个点的shape match产生的约束部分，因为之前是按照每个四面体计算的，现在要摊到每个顶点上
	float elementX = forceX + collisionForce[indexX];
	float elementY = forceY + collisionForce[indexY];
	float elementZ = forceZ + collisionForce[indexZ];

	float positionX = positions[indexX];
	float positionY = positions[indexY];
	float positionZ = positions[indexZ];

	//相当于先按重力运动，每次再在收重力的效果上再修正
	float volumnDiag_diagConstant = volumnDiag[threadid] + diagConstant;
	float nextnext_positionsX = (diagConstant * (old_positions[indexX] - positionX) + elementX) / (collisionDiag[indexX] + volumnDiag_diagConstant) + positionX;
	float nextnext_positionsY = (diagConstant * (old_positions[indexY] - positionY) + elementY) / (collisionDiag[indexY] + volumnDiag_diagConstant) + positionY;
	float nextnext_positionsZ = (diagConstant * (old_positions[indexZ] - positionZ) + elementZ) / (collisionDiag[indexZ] + volumnDiag_diagConstant) + positionZ;

	//under-relaxation 和 切比雪夫迭代
	nextnext_positionsX = (nextnext_positionsX - positionX) * 0.6 + positionX;
	nextnext_positionsY = (nextnext_positionsY - positionY) * 0.6 + positionY;
	nextnext_positionsZ = (nextnext_positionsZ - positionZ) * 0.6 + positionZ;

	// omega定义：omega = 4 / (4 - rho*rho*omega);
	float prev_positionsX = prev_positions[indexX];
	float prev_positionsY = prev_positions[indexY];
	float prev_positionsZ = prev_positions[indexZ];
	nextnext_positionsX = omega * (nextnext_positionsX - prev_positionsX) + prev_positionsX;
	nextnext_positionsY = omega * (nextnext_positionsY - prev_positionsY) + prev_positionsY;
	nextnext_positionsZ = omega * (nextnext_positionsZ - prev_positionsZ) + prev_positionsZ;

	prev_positions[indexX] = positionX;
	prev_positions[indexY] = positionY;
	prev_positions[indexZ] = positionZ;

	positions[indexX] = next_positions[indexX] = nextnext_positionsX;
	positions[indexY] = next_positions[indexY] = nextnext_positionsY;
	positions[indexZ] = next_positions[indexZ] = nextnext_positionsZ;
}

int runcalculateV(float dt) {
	int  threadNum = 128;
	int blockNum = (tetVertNum_d + threadNum - 1) / threadNum;
	//并行计算
	calculateV << <blockNum, threadNum >> > (tetVertPos_d, tetVertVelocity_d, tetVertPos_last_d, tetVertNum_d, dt);

	//cudaDeviceSynchronize();
	printCudaError("runcalculateV");
	return 0;

}

//计算速度更新
__global__ void calculateV(float* positions, float* velocity, float* last_positions, int vertexNum, float dt) {
	unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadid >= vertexNum) return;
	velocity[threadid * 3 + 0] = (positions[threadid * 3 + 0] - last_positions[threadid * 3 + 0]) / dt;
	velocity[threadid * 3 + 1] = (positions[threadid * 3 + 1] - last_positions[threadid * 3 + 1]) / dt;
	velocity[threadid * 3 + 2] = (positions[threadid * 3 + 2] - last_positions[threadid * 3 + 2]) / dt;
}

__global__ void calculateV(float* positions, float* velocity, float* last_positions, int* sortedIndices, int offset, int activeElementNum, float dt) {
	unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadid >= activeElementNum) return;

	int vertIdx = sortedIndices[threadid + offset];
	velocity[vertIdx * 3 + 0] = (positions[vertIdx * 3 + 0] - last_positions[vertIdx * 3 + 0]) / dt;
	velocity[vertIdx * 3 + 1] = (positions[vertIdx * 3 + 1] - last_positions[vertIdx * 3 + 1]) / dt;
	velocity[vertIdx * 3 + 2] = (positions[vertIdx * 3 + 2] - last_positions[vertIdx * 3 + 2]) / dt;
}
int runUpdateInnerTetVertDDir()
{
	int threadNum = 128;
	int blockNum = (tetVertNum_d + threadNum - 1) / threadNum;

	updateInnerTetVertDirectDirection << <blockNum, threadNum >> > (tetVertPos_d,
		tetVertBindingTetVertIndices_d, tetVertBindingTetVertWeight_d,
		tetVertNonPenetrationDir_d, tetVertNum_d);
	// no need to sychronize
	printCudaError("updateInnerTetVertDDir");
	return 0;
}
int runUpdateSurfaceTetVertDDir()
{
	int threadNum = 128;
	int blockNum = (triVertOrgNum_d + threadNum - 1) / threadNum;
	updateSurfaceTetVertDirectDirection << <blockNum, threadNum >> > (
		onSurfaceTetVertIndices_d,
		tetVert2TriVertMapping_d, triVertNorm_d,
		tetVertNonPenetrationDir_d,
		tetVertPos_d, triVertPos_d,
		triVertOrgNum_d);
	printCudaError("updateSurfaceTetVertDDir");
	return 0;
}
int runNormalizeDDir()
{
	int threadNum = 128;
	int blockNum = (tetVertNum_d + threadNum - 1) / threadNum;
	normalizeDDir << <blockNum, threadNum >> > (tetVertNonPenetrationDir_d, tetVertNum_d);
	printCudaError("NormalizeTetVertDDir");
	return 0;
}
int runUpdateTetVertDirectDirection()
{
	runUpdateInnerTetVertDDir();
	runUpdateSurfaceTetVertDDir();
	//cudaDeviceSynchronize();

	runNormalizeDDir();
	//cudaDeviceSynchronize();
	printCudaError("UpdateTetVertDirectDirection");
	return 0;
}

__global__ void normalizeDDir(float* dDir, int pointNum)
{
	unsigned int threadid = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadid >= pointNum) return;
	
	int idxX = threadid * 3 + 0;
	int idxY = threadid * 3 + 1;
	int idxZ = threadid * 3 + 2;
	float l = sqrt(dDir[idxX] * dDir[idxX] + dDir[idxY] * dDir[idxY] + dDir[idxZ] * dDir[idxZ]);
	if(l<1e-7)
	{
		//printf("threadid %d, dDirLen=0\n", threadid);
		dDir[idxX] = 1;
		dDir[idxY] = 0;
		dDir[idxZ] = 0;
	}
	else
	{
		dDir[idxX] /= l;
		dDir[idxY] /= l;
		dDir[idxZ] /= l;
	}
	//if (threadid < 10)
	//{
	//	printf("threadid %d, DDir [%f %f %f]\n", threadid, dDir[idxX], dDir[idxY], dDir[idxZ]);
	//}
}
__device__ __inline__ void GetRotation_D(float F[3][3], float R[3][3])
{
	float C[3][3], C2[3][3];
	// 优化：计算 C = F^T * F，利用对称性减少计算
	C[0][0] = F[0][0] * F[0][0] + F[1][0] * F[1][0] + F[2][0] * F[2][0];
	C[0][1] = F[0][0] * F[0][1] + F[1][0] * F[1][1] + F[2][0] * F[2][1];
	C[0][2] = F[0][0] * F[0][2] + F[1][0] * F[1][2] + F[2][0] * F[2][2];
	C[1][1] = F[0][1] * F[0][1] + F[1][1] * F[1][1] + F[2][1] * F[2][1];
	C[1][2] = F[0][1] * F[0][2] + F[1][1] * F[1][2] + F[2][1] * F[2][2];
	C[2][2] = F[0][2] * F[0][2] + F[1][2] * F[1][2] + F[2][2] * F[2][2];
	C[1][0] = C[0][1];C[2][0] = C[0][2];C[2][1] = C[1][2];
	// 优化：计算 C2 = C * C^T（因C对称，等价于C * C）
	C2[0][0] = C[0][0] * C[0][0] + C[0][1] * C[0][1] + C[0][2] * C[0][2];
	C2[0][1] = C[0][0] * C[1][0] + C[0][1] * C[1][1] + C[0][2] * C[1][2];
	C2[0][2] = C[0][0] * C[2][0] + C[0][1] * C[2][1] + C[0][2] * C[2][2];
	C2[1][1] = C[1][0] * C[1][0] + C[1][1] * C[1][1] + C[1][2] * C[1][2];
	C2[1][2] = C[1][0] * C[2][0] + C[1][1] * C[2][1] + C[1][2] * C[2][2];
	C2[2][2] = C[2][0] * C[2][0] + C[2][1] * C[2][1] + C[2][2] * C[2][2];
	C2[1][0] = C2[0][1];C2[2][0] = C2[0][2];C2[2][1] = C2[1][2];

	float det = (F[0][0] * F[1][1] - F[0][1] * F[1][0]) * F[2][2] +
				(F[1][0] * F[0][2] - F[0][0] * F[1][2]) * F[2][1] +
				(F[0][1] * F[1][2] - F[0][2] * F[1][1]) * F[2][0];

	float I_c = C[0][0] + C[1][1] + C[2][2], I_c2 = I_c * I_c;
	float II_c = 0.5 * (I_c2 - C2[0][0] - C2[1][1] - C2[2][2]);
	float k = I_c2 - 3 * II_c;
	float III_c = det * det;
	float l = I_c * (I_c2 - 4.5 * II_c) + 13.5 * III_c;
	float k_root = sqrtf(k);
	float value = l / (k * k_root);
	value = fmaxf(-1.0f, fminf(1.0f, value));
	float phi = acosf(value);
	float lambda2 = (I_c + 2 * k_root * cosf(phi / 3)) / 3.0;
	float lambda = sqrtf(lambda2);

	float III_u = det;
	float I_u = lambda + sqrtf(-lambda2 + I_c + 2 * III_u / lambda);
	float I_u2 = I_u * I_u;
	float II_u = (I_u2 - I_c) * 0.5;

	float inv_rate = 1 / (I_u * II_u - III_u),
		  factor = I_u * III_u * inv_rate;

	float U[3][3] = {
		factor, 0, 0,
		0, factor, 0,
		0, 0, factor
	};

	factor = (I_u * I_u - II_u) * inv_rate;
	U[0][0] += factor * C[0][0] - inv_rate * C2[0][0];
	U[0][1] += factor * C[0][1] - inv_rate * C2[0][1];
	U[1][1] += factor * C[1][1] - inv_rate * C2[1][1];
	U[0][2] += factor * C[0][2] - inv_rate * C2[0][2];
	U[1][2] += factor * C[1][2] - inv_rate * C2[1][2];
	U[2][2] += factor * C[2][2] - inv_rate * C2[2][2];
	U[1][0] = U[0][1]; U[2][0] = U[0][2]; U[2][1] = U[1][2];

	inv_rate = 1 / III_u;
	factor = II_u * inv_rate;
	float inv_U[3][3] = {
		factor, 0, 0,
		0, factor, 0,
		0, 0, factor
	};
	factor = -I_u * inv_rate;
	inv_U[0][0] += factor * U[0][0] + inv_rate * C[0][0];
	inv_U[0][1] += factor * U[0][1] + inv_rate * C[0][1];
	inv_U[0][2] += factor * U[0][2] + inv_rate * C[0][2];
	inv_U[1][1] += factor * U[1][1] + inv_rate * C[1][1];
	inv_U[1][2] += factor * U[1][2] + inv_rate * C[1][2];
	inv_U[2][2] += factor * U[2][2] + inv_rate * C[2][2];
	inv_U[1][0] = inv_U[0][1];inv_U[2][0] = inv_U[0][2];inv_U[2][1] = inv_U[1][2];

	if (k < 1e-10f)
	{
		for (int i = 0;i < 3;i++)
			for (int j = 0;j < 3;j++)
				inv_U[i][j] = 0;
		inv_U[0][0] = inv_U[1][1] = inv_U[2][2] = 1 / sqrtf(I_c / 3);
	}

	R[0][0] = F[0][0] * inv_U[0][0] + F[0][1] * inv_U[1][0] + F[0][2] * inv_U[2][0];
	R[0][1] = F[0][0] * inv_U[0][1] + F[0][1] * inv_U[1][1] + F[0][2] * inv_U[2][1];
	R[0][2] = F[0][0] * inv_U[0][2] + F[0][1] * inv_U[1][2] + F[0][2] * inv_U[2][2];
	R[1][0] = F[1][0] * inv_U[0][0] + F[1][1] * inv_U[1][0] + F[1][2] * inv_U[2][0];
	R[1][1] = F[1][0] * inv_U[0][1] + F[1][1] * inv_U[1][1] + F[1][2] * inv_U[2][1];
	R[1][2] = F[1][0] * inv_U[0][2] + F[1][1] * inv_U[1][2] + F[1][2] * inv_U[2][2];
	R[2][0] = F[2][0] * inv_U[0][0] + F[2][1] * inv_U[1][0] + F[2][2] * inv_U[2][0];
	R[2][1] = F[2][0] * inv_U[0][1] + F[2][1] * inv_U[1][1] + F[2][2] * inv_U[2][1];
	R[2][2] = F[2][0] * inv_U[0][2] + F[2][1] * inv_U[1][2] + F[2][2] * inv_U[2][2];
	if (det <= 0) {
		R[0][0] = 1;R[0][1] = 0;R[0][2] = 0;
		R[1][0] = 0;R[1][1] = 1;R[1][2] = 0;
		R[2][0] = 0;R[2][1] = 0;R[2][2] = 1;
	}
}
