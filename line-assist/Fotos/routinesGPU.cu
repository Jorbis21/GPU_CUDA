#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>


#include "routinesGPU.h"

__global__ void kernelBlur(uint8_t *im, float *NR, int height, int width){
	int i = (blockIdx.x*blockDim.x + threadIdx.x) + 2;
	int j = (blockIdx.y*blockDim.y + threadIdx.y) + 2;
	if(i < height-2 && j < width-2){
		NR[i*width+j] =
				 (2.0*im[(i-2)*width+(j-2)] +  4.0*im[(i-2)*width+(j-1)] +  5.0*im[(i-2)*width+(j)] +  4.0*im[(i-2)*width+(j+1)] + 2.0*im[(i-2)*width+(j+2)]
				+ 4.0*im[(i-1)*width+(j-2)] +  9.0*im[(i-1)*width+(j-1)] + 12.0*im[(i-1)*width+(j)] +  9.0*im[(i-1)*width+(j+1)] + 4.0*im[(i-1)*width+(j+2)]
				+ 5.0*im[(i  )*width+(j-2)] + 12.0*im[(i  )*width+(j-1)] + 15.0*im[(i  )*width+(j)] + 12.0*im[(i  )*width+(j+1)] + 5.0*im[(i  )*width+(j+2)]
				+ 4.0*im[(i+1)*width+(j-2)] +  9.0*im[(i+1)*width+(j-1)] + 12.0*im[(i+1)*width+(j)] +  9.0*im[(i+1)*width+(j+1)] + 4.0*im[(i+1)*width+(j+2)]
				+ 2.0*im[(i+2)*width+(j-2)] +  4.0*im[(i+2)*width+(j-1)] +  5.0*im[(i+2)*width+(j)] +  4.0*im[(i+2)*width+(j+1)] + 2.0*im[(i+2)*width+(j+2)])
				/159.0;
	}
}

__global__ void kernelGradient(float *NR, float *Gx, float *Gy, float *phi, float *G, int height, int width){
	float PI = 3.141593;
	int i = (blockIdx.x*blockDim.x + threadIdx.x) + 2;
	int j = (blockIdx.y*blockDim.y + threadIdx.y) + 2;
	if(i < height-2 && j < width-2){
		Gx[i*width+j] = 
				 (1.0*NR[(i-2)*width+(j-2)] +  2.0*NR[(i-2)*width+(j-1)] +  (-2.0)*NR[(i-2)*width+(j+1)] + (-1.0)*NR[(i-2)*width+(j+2)]
				+ 4.0*NR[(i-1)*width+(j-2)] +  8.0*NR[(i-1)*width+(j-1)] +  (-8.0)*NR[(i-1)*width+(j+1)] + (-4.0)*NR[(i-1)*width+(j+2)]
				+ 6.0*NR[(i  )*width+(j-2)] + 12.0*NR[(i  )*width+(j-1)] + (-12.0)*NR[(i  )*width+(j+1)] + (-6.0)*NR[(i  )*width+(j+2)]
				+ 4.0*NR[(i+1)*width+(j-2)] +  8.0*NR[(i+1)*width+(j-1)] +  (-8.0)*NR[(i+1)*width+(j+1)] + (-4.0)*NR[(i+1)*width+(j+2)]
				+ 1.0*NR[(i+2)*width+(j-2)] +  2.0*NR[(i+2)*width+(j-1)] +  (-2.0)*NR[(i+2)*width+(j+1)] + (-1.0)*NR[(i+2)*width+(j+2)]);

		Gy[i*width+j] = 
				 ((-1.0)*NR[(i-2)*width+(j-2)] + (-4.0)*NR[(i-2)*width+(j-1)] +  (-6.0)*NR[(i-2)*width+(j)] + (-4.0)*NR[(i-2)*width+(j+1)] + (-1.0)*NR[(i-2)*width+(j+2)]
				+ (-2.0)*NR[(i-1)*width+(j-2)] + (-8.0)*NR[(i-1)*width+(j-1)] + (-12.0)*NR[(i-1)*width+(j)] + (-8.0)*NR[(i-1)*width+(j+1)] + (-2.0)*NR[(i-1)*width+(j+2)]
				+    2.0*NR[(i+1)*width+(j-2)] +    8.0*NR[(i+1)*width+(j-1)] +    12.0*NR[(i+1)*width+(j)] +    8.0*NR[(i+1)*width+(j+1)] +    2.0*NR[(i+1)*width+(j+2)]
				+    1.0*NR[(i+2)*width+(j-2)] +    4.0*NR[(i+2)*width+(j-1)] +     6.0*NR[(i+2)*width+(j)] +    4.0*NR[(i+2)*width+(j+1)] +    1.0*NR[(i+2)*width+(j+2)]);

		G[i*width+j]   = sqrtf((Gx[i*width+j]*Gx[i*width+j])+(Gy[i*width+j]*Gy[i*width+j]));
		phi[i*width+j] = atan2f(fabs(Gy[i*width+j]),fabs(Gx[i*width+j]));

		if(fabs(phi[i*width+j])<=PI/8 )
			phi[i*width+j] = 0;
		else if (fabs(phi[i*width+j])<= 3*(PI/8))
			phi[i*width+j] = 45;
		else if (fabs(phi[i*width+j]) <= 5*(PI/8))
			phi[i*width+j] = 90;
		else if (fabs(phi[i*width+j]) <= 7*(PI/8))
			phi[i*width+j] = 135;
		else phi[i*width+j] = 0;
	}
}

__global__ void kernelEdge(float *G, uint8_t *pedge, float *phi, int height, int width){
	int i = (blockIdx.x*blockDim.x + threadIdx.x) + 3;
	int j = (blockIdx.y*blockDim.y + threadIdx.y) + 3;
	if(i < height-3 && j < width-3){
		pedge[i*width+j] = 0;
			if(phi[i*width+j] == 0){
				if(G[i*width+j]>G[i*width+j+1] && G[i*width+j]>G[i*width+j-1]) //edge is in N-S
					pedge[i*width+j] = 1;

			} else if(phi[i*width+j] == 45) {
				if(G[i*width+j]>G[(i+1)*width+j+1] && G[i*width+j]>G[(i-1)*width+j-1]) // edge is in NW-SE
					pedge[i*width+j] = 1;

			} else if(phi[i*width+j] == 90) {
				if(G[i*width+j]>G[(i+1)*width+j] && G[i*width+j]>G[(i-1)*width+j]) //edge is in E-W
					pedge[i*width+j] = 1;

			} else if(phi[i*width+j] == 135) {
				if(G[i*width+j]>G[(i+1)*width+j-1] && G[i*width+j]>G[(i-1)*width+j+1]) // edge is in NE-SW
					pedge[i*width+j] = 1;
			}
	}
}

__global__ void kernelThresholding(uint8_t *pedge, float *G,uint8_t *image_out, float level, int height, int width){
	float lowthres = level/2;
	float hithres = 2*(level);
	int ii, jj;
	int i = (blockIdx.x*blockDim.x + threadIdx.x) + 3;
	int j = (blockIdx.y*blockDim.y + threadIdx.y) + 3;
	if(i < height-3 && j < width-3){
		image_out[i*width+j] = 0;
			if(G[i*width+j]>hithres && pedge[i*width+j])
				image_out[i*width+j] = 255;
			else if(pedge[i*width+j] && G[i*width+j]>=lowthres && G[i*width+j]<hithres)
				// check neighbours 3x3
				for (ii=-1;ii<=1; ii++)
					for (jj=-1;jj<=1; jj++)
						if (G[(i+ii)*width+j+jj]>hithres)
							image_out[i*width+j] = 255;
	}
}

void canny(uint8_t *im, uint8_t *image_out,
	float *NR, float *G, float *phi, float *Gx, float *Gy, uint8_t *pedge,
	float level,
	int height, int width)
{	
	float *d_NR, *d_G, *d_phi, *d_Gx, *d_Gy;
	uint8_t *d_im, *d_pedge, *d_image_out;
	//Carga de memoria al kernel blur
	cudaMalloc((void**)&d_im, height*width*sizeof(uint8_t));
	cudaMalloc((void**)&d_NR, height*width*sizeof(float));
	cudaMemcpy(d_im, im, height*width*sizeof(uint8_t), cudaMemcpyHostToDevice);
	//Llamada al kernel blur
	dim3 dimBlockB(32,32,1); // numero de hilos por bloque
	dim3 dimGridB(ceil(height/32.0), ceil(width/32.0),1); // numero de bloques
	kernelBlur<<<dimGridB, dimBlockB>>>(d_im, d_NR, height, width);
	cudaFree(d_im);
	//Fin kernel blur
	//Carga de memoria al kernel gradient
	cudaMalloc((void**)&d_Gx, height*width*sizeof(float));
	cudaMalloc((void**)&d_Gy, height*width*sizeof(float));
	cudaMalloc((void**)&d_phi, height*width*sizeof(float));
	cudaMalloc((void**)&d_G, height*width*sizeof(float));
	//Llamada al kernel gradient
	dim3 dimBlockG(32,32,1); // numero de hilos por bloque
	dim3 dimGridG(ceil(height/32.0), ceil(width/32.0),1); // numero de bloques
	kernelGradient<<<dimGridG, dimBlockG>>>(d_NR, d_Gx, d_Gy, d_phi, d_G, height, width);
	cudaFree(d_Gx); cudaFree(d_Gy); cudaFree(d_NR);
	//Fin kernel gradient
	//Carga de memoria al kernel edge
	cudaMalloc((void**)&d_pedge, height*width*sizeof(uint8_t));
	//Llamada al kernel edge
	dim3 dimBlockE(32,32,1); // numero de hilos por bloque
	dim3 dimGridE( ceil(height/32.0), ceil(width/32.0), 1); // numero de bloques
	kernelEdge<<<dimGridE, dimBlockE>>>(d_G, d_pedge, d_phi, height, width);
	cudaFree(d_phi); 
	//Fin kernel edge
	//Carga de memoria al kernel Thresholding
	cudaMalloc((void**)&d_image_out, height*width*sizeof(uint8_t));
	//Llamada al kernel Thresholding
	dim3 dimBlockT(32,32,1); // numero de hilos por bloque
	dim3 dimGridT(ceil(height/32.0), ceil(width/32.0), 1); // numero de bloques
	kernelThresholding<<<dimGridT, dimBlockT>>>(d_pedge, d_G, d_image_out, level, height, width);
	cudaMemcpy(image_out, d_image_out, height*width*sizeof(uint8_t), cudaMemcpyDeviceToHost);
	cudaFree(d_pedge); cudaFree(d_G); cudaFree(d_image_out);
	//Fin kernel Thresholding
}

__global__ void kernelAccInit(uint32_t *accumulators, int accu_width, int accu_height)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int j = blockIdx.y*blockDim.y + threadIdx.y;
	if (i<accu_height && j<accu_width)
		accumulators[i*accu_width+j] = 0;
}
__global__ void kernelHoughTransform(uint8_t *im, int width, int height, uint32_t *accumulators, float hough_h, float *sin_table, float *cos_table, float center_x, float center_y){
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int j = blockIdx.y*blockDim.y + threadIdx.y;
	int theta;
	if(i < height && j < width)
		if( im[ (i*width) + j] > 250 ) // Pixel is edge  
				{  
					for(theta=0;theta<180;theta++)  
					{  
						float rho = ( ((float)j - center_x) * cos_table[theta]) + (((float)i - center_y) * sin_table[theta]);
						atomicAdd(&accumulators[ (int)((round(rho + hough_h) * 180.0)) + theta], 1);
					} 
				} 	

}

void houghtransform(uint8_t *im, int width, int height, uint32_t *accumulators, int accu_width, int accu_height, 
	float *sin_table, float *cos_table)
{
	uint8_t *d_im;
	uint32_t *d_accumulators;

	float *d_sin_table, *d_cos_table, hough_h, center_x = width/2.0, center_y = height/2.0;
	hough_h = ((sqrt(2.0) * (float)(height>width?height:width)) / 2.0);

	cudaMalloc((void**)&d_accumulators, accu_height*accu_width*sizeof(uint32_t));
	dim3 dimBlockA(32,32,1); // numero de hilos por bloque
	dim3 dimGridA(ceil(accu_height/32.0), ceil(accu_width/32.0),1); // numero de bloques
	kernelAccInit<<<dimGridA, dimBlockA>>>(d_accumulators, accu_width, accu_height);


	cudaMalloc((void**)&d_im, height*width*sizeof(uint8_t));
	cudaMalloc((void**)&d_sin_table, 180*sizeof(float));
	cudaMalloc((void**)&d_cos_table, 180*sizeof(float));
	cudaMemcpy(d_sin_table, sin_table, 180*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_cos_table, cos_table, 180*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_im, im, height*width*sizeof(uint8_t), cudaMemcpyHostToDevice);
	
	dim3 dimBlockH(32,32,1); // numero de hilos por bloque
	dim3 dimGridH(ceil(height/32.0), ceil(width/32.0),1); // numero de bloques
	kernelHoughTransform<<<dimGridH, dimBlockH>>>(d_im, width, height, d_accumulators, hough_h, d_sin_table, d_cos_table, center_x, center_y);
	cudaMemcpy(accumulators, d_accumulators, accu_height*accu_width*sizeof(uint32_t), cudaMemcpyDeviceToHost);
	cudaFree(d_im); cudaFree(d_accumulators); cudaFree(d_sin_table); cudaFree(d_cos_table);
}

__global__ void kernelGetLines(int threshold, uint32_t *accumulators, int accu_width, int accu_height, int width, int height, 
	float *sin_table, float *cos_table,
	int *x1_lines, int *y1_lines, int *x2_lines, int *y2_lines, int *lines){
	int rho = blockIdx.x*blockDim.x + threadIdx.x;
	int theta = blockIdx.y*blockDim.y + threadIdx.y;
	int ii, jj;
	uint32_t max;
	if(rho < accu_height && theta < accu_width){
		if(accumulators[(rho*accu_width) + theta] >= threshold)  
			{  
				//Is this point a local maxima (9x9)  
				max = accumulators[(rho*accu_width) + theta]; 
				for(int ii=-4;ii<=4;ii++)  
				{  
					for(int jj=-4;jj<=4;jj++)  
					{  
						if( (ii+rho>=0 && ii+rho<accu_height) && (jj+theta>=0 && jj+theta<accu_width) )  
						{  
							if( accumulators[((rho+ii)*accu_width) + (theta+jj)] > max )  
							{
								max = accumulators[((rho+ii)*accu_width) + (theta+jj)];
							}  
						}  
					}  
				}  

				if(max == accumulators[(rho*accu_width) + theta]) //local maxima
				{
					int x1, y1, x2, y2;  
					x1 = y1 = x2 = y2 = 0;  

					if(theta >= 45 && theta <= 135)  
					{
						if (theta>90) {
							//y = (r - x cos(t)) / sin(t)  
							x1 = width/2;  
							y1 = ((float)(rho-(accu_height/2)) - ((x1 - (width/2) ) * cos_table[theta])) / sin_table[theta] + (height / 2);
							x2 = width;  
							y2 = ((float)(rho-(accu_height/2)) - ((x2 - (width/2) ) * cos_table[theta])) / sin_table[theta] + (height / 2);  
						} else {
							//y = (r - x cos(t)) / sin(t)  
							x1 = 0;  
							y1 = ((float)(rho-(accu_height/2)) - ((x1 - (width/2) ) * cos_table[theta])) / sin_table[theta] + (height / 2);
							x2 = width*2/5;  
							y2 = ((float)(rho-(accu_height/2)) - ((x2 - (width/2) ) * cos_table[theta])) / sin_table[theta] + (height / 2); 
						}
					} else {
						//x = (r - y sin(t)) / cos(t);  
						y1 = 0;  
						x1 = ((float)(rho-(accu_height/2)) - ((y1 - (height/2) ) * sin_table[theta])) / cos_table[theta] + (width / 2);  
						y2 = height;  
						x2 = ((float)(rho-(accu_height/2)) - ((y2 - (height/2) ) * sin_table[theta])) / cos_table[theta] + (width / 2);  
					}
					x1_lines[*lines] = x1;
					y1_lines[*lines] = y1;
					x2_lines[*lines] = x2;
					y2_lines[*lines] = y2;
					(*lines)++;
				}
			}
	}
}

void getlines(int threshold, uint32_t *accumulators, int accu_width, int accu_height, int width, int height, 
	float *sin_table, float *cos_table,
	int *x1_lines, int *y1_lines, int *x2_lines, int *y2_lines, int *lines)
{
	uint32_t *d_accumulators;
	float *d_sin_table, *d_cos_table;
	int *d_x1_lines, *d_y1_lines, *d_x2_lines, *d_y2_lines, *d_lines;
	cudaMalloc((void**)&d_accumulators, accu_height*accu_width*sizeof(uint32_t));
	cudaMalloc((void**)&d_sin_table, 180*sizeof(float));
	cudaMalloc((void**)&d_cos_table, 180*sizeof(float));
	cudaMalloc((void**)&d_x1_lines, 10*sizeof(int));
	cudaMalloc((void**)&d_y1_lines, 10*sizeof(int));
	cudaMalloc((void**)&d_x2_lines, 10*sizeof(int));
	cudaMalloc((void**)&d_y2_lines, 10*sizeof(int));
	cudaMalloc((void**)&d_lines, sizeof(int));
	cudaMemcpy(d_accumulators, accumulators, accu_height*accu_width*sizeof(uint32_t), cudaMemcpyHostToDevice);
	cudaMemcpy(d_sin_table, sin_table, 180*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_cos_table, cos_table, 180*sizeof(float), cudaMemcpyHostToDevice);
	dim3 dimBlockL(32,32,1); // numero de hilos por bloque
	dim3 dimGridL(ceil(accu_height/32.0), ceil(accu_width/32.0),1); // numero de bloques
	kernelGetLines<<<dimGridL, dimBlockL>>>(threshold, d_accumulators, accu_width, accu_height, width, height, d_sin_table, d_cos_table, d_x1_lines, d_y1_lines, d_x2_lines, d_y2_lines, d_lines);
	cudaMemcpy(x1_lines, d_x1_lines, 10*sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(y1_lines, d_y1_lines, 10*sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(x2_lines, d_x2_lines, 10*sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(y2_lines, d_y2_lines, 10*sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(lines, d_lines, sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(d_accumulators); cudaFree(d_sin_table); cudaFree(d_cos_table);
	cudaFree(d_x1_lines); cudaFree(d_y1_lines); cudaFree(d_x2_lines); cudaFree(d_y2_lines); cudaFree(d_lines);
}

//hacer uso de la memoria compartida si es necesario para reusar los datos
void line_asist_GPU(uint8_t *im, int height, int width,
	uint8_t *imEdge, float *NR, float *G, float *phi, float *Gx, float *Gy, uint8_t *pedge,
	float *sin_table, float *cos_table, 
	uint32_t *accum, int accu_height, int accu_width,
	int *x1, int *y1, int *x2, int *y2, int *nlines)
{
	int threshold;
	canny(im, imEdge, NR, G, phi, Gx, Gy, pedge, 1000.0f, height, width);
	houghtransform(imEdge, width, height, accum, accu_width, accu_height, sin_table, cos_table); //GPU

	if (width>height) threshold = width/6;
	else threshold = height/6;


	getlines(threshold, accum, accu_width, accu_height, width, height, 
		sin_table, cos_table,
		x1, y1, x2, y2, nlines);
}
