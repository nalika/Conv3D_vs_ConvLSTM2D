# Conv3D_vs_ConvLSTM2D
Computing total number of parameters in Conv3D and ConvLSTM2D


##	1. Number of parameters of a Conv3D			
								
	Let:							
		
 Wc = Number of weights of the Conv Layer.		
 Bc = Number of biases of the Conv Layer.		
 Pc = Number of parameters of the Conv Layer.		
 K = Size (width) of kernels used in the Conv Layer.		
 N = Number of kernels.					
 C = Number of channels of the input image.
								
Then, the total number of parameters of a Conv will be:
  Pc=Wc+Bc, where Wc = K^2xCxN and Bc = N
	
In our case, the conv op takes 3 input frames at a time (Note: batch size is 4, so it will stride in time domain by 1)
So, we have K = 3x3, C=(R, G,B)x3 = 3x3, and N = 16, resulting in total number of parameters to be 1312.
								
		K =	3x3=	9				
		C= 	3x3=	9				
		N=	feature maps	16				
	Thus,							
		Wc = K^2xCx N	1296				
		Bc=N = 16				
		Pc=Wc+Bc = 1312				
								
								
								
##	2. Number of parameters of a ConvLSTM2D		
								
General LSTM have four gates (input,output,forget,cell gate). In ConvLSTM also same gate present but they where not perform element-wise multiplication, perform Convolution operation with LSTM equation.		
								
ParamConvLSTM2D = [K × K × (Cin + Cout) × Cout+Cout] × 4. These parameters are shared among timesteps
								
								
In this, example:	K=	3, Cin=	3, Cout=	16, and #of gates (I,O, F, C)=	4				
Thus, the ParamConvLSTM2D = [3x3 x(3+16) x 16 + 16]x4 = 11008				
								
								
Ref: 
1. https://www.learnopencv.com/number-of-parameters-and-tensor-sizes-in-convolutional-neural-network/
2. Attention in Convolutional LSTM for Gesture Recognition, 32nd Conference on Neural Information Processing Systems (NeurIPS 2018), Montréal, Canada.	

