import numpy as np

class controller:
    def __init__(self,input_size,hiddensizes,output,std=2):
        """
        GA class takes in the input size (so flattened image size)
        hiddensizes should be an array containining n layers and each index being the layer size
        output does exactly what you think
        """
        self.w=[]
        self.b=[]
        last_layer=input_size
        self.gene_size=input_size
        self.geno=np.array([]) #gather the genotype for flat mutation
        self.hiddensizes=hiddensizes
        for i in range(len(self.hiddensizes)):
            self.w.append(np.random.normal(0,std,(last_layer,hiddensizes[i])))
            self.b.append(np.random.normal(0,std,(hiddensizes[i])))
            self.gene_size = self.gene_size + last_layer*hiddensizes[i] + hiddensizes[i] #gene size calculation for later
            last_layer=hiddensizes[i] #keep moving through network
            self.geno = np.concatenate([self.geno,self.w[i].flatten(),self.b[i].flatten()])
        self.w.append(np.random.normal(0,std,(last_layer,output)))
        self.b.append(np.random.normal(0,std,(output)))
        self.geno = np.concatenate([self.geno,self.w[-1].flatten(),self.b[-1].flatten()])
        self.gene_size = len(self.geno)  #gene size calculation for later
        self.std=std
        #form a neural network ^
    def mutate(self,rate=0.2): #random change of mutating different genes throughout the network
        probailities=np.random.random(self.gene_size)
        self.geno[np.where(probailities<rate)]+=np.random.normal(0,self.std,self.geno[np.where(probailities<rate)].shape)
        self.geno[self.geno<-16]=-16
        self.geno[self.geno>16]=16
        idx=0
        for i in range(len(self.w)):
            size=self.w[i].flatten().shape[0]
            self.w[i]=self.geno[idx:idx+size].reshape(self.w[i].shape)
            idx+=size
            size=self.b[i].flatten().shape[0]
            self.b[i]=self.geno[idx:idx+size].reshape(self.b[i].shape)
            idx+=size
    def activation(self,x): #can replace with any activation function
        return 1/(1 + np.exp(-x))
    def step(self,x):
        #take in parameter x which is the input data, this can be linear data or an image
        x=x.flatten()
        x=x.reshape((1,x.shape[0]))
        for i in range(len(self.w)-1):
            x=self.activation(np.dot(x,self.w[i]) + self.b[i])
        return (np.dot(x,self.w[-1]) + self.b[-1])[0]
    def sex(self,geno1,geno2,prob_winning=0.6):
        probabilities=np.random.random(self.gene_size)
        geno2.geno[np.where(probabilities<prob_winning)]=geno1.geno[np.where(probabilities<prob_winning)]
        return geno2
    

class controller_LRF(controller):
    def step(self,x):
        #take in parameter x which is the input data, this can be linear data or an image
        x=x.flatten()
        x=x.reshape((1,x.shape[0]))
        for i in range(len(self.w)-1):
            x=self.activation(np.dot(x,self.w[i]) + self.b[i])
        return np.argmax((np.dot(x,self.w[-1]) + self.b[-1])[0])
    
import numpy as np
import scipy.signal  # for convolution without needing torch

class controllerCNN:
    def __init__(self, input_shape, hidden_size, output_size, kernel_sizes=[[5,3],[3,3],[3,3]], num_kernels=[4,2,1], std=0.1):
        """
        CNN Controller that can be evolved.
        
        input_shape: (height, width)
        kernel_sizes: list of kernel sizes (e.g., [3, 3])
        num_kernels: list of number of kernels per layer (e.g., [8, 16])
        output_size: size of final output vector
        std: standard deviation for initial random weights
        """
        self.input_shape = input_shape
        self.kernel_sizes = kernel_sizes
        self.num_kernels = num_kernels
        self.output_size = output_size
        self.std = std

        self.kernels = []
        self.biases = []

        self.geno = np.array([])

        in_channels = 1  #assuming grayscale input; you can adapt for RGB later
        for ksize, nkernels in zip(kernel_sizes, num_kernels):
            kernel_shape = (nkernels, in_channels, ksize[0], ksize[1])
            bias_shape = (nkernels,)
            kernels = np.random.normal(0, std, kernel_shape).astype(np.float32)
            biases = np.random.normal(0, std, bias_shape).astype(np.float32)

            self.kernels.append(kernels)
            self.biases.append(biases)

            self.geno = np.concatenate([self.geno, kernels.flatten(), biases.flatten()])
            in_channels = nkernels  # for next layer
        self.kernel_vals=len(self.geno)
        # Fully connected layer from conv outputs to output
        # Flatten size needs to be estimated. Assume input shrinks by (kernel_size - 1) each conv layer without padding
        h, w = input_shape
        for k in kernel_sizes:
            h -= (k[0] - 1)
            w -= (k[1] - 1)
        flattened_size = h * w * in_channels
        self.w = [np.random.normal(0, std, (flattened_size, hidden_size)).astype(np.float32), np.random.normal(0, std, (hidden_size, output_size)).astype(np.float32)]
        self.b = [np.random.normal(0, std, (hidden_size,)).astype(np.float32),np.random.normal(0, std, (output_size,)).astype(np.float32)]
        self.geno = np.concatenate([self.geno, self.w[0].flatten(), self.b[0].flatten(), self.w[1].flatten(),self.b[1].flatten()])
        self.gene_size = len(self.geno)
    def mutate(self,rate=0.2): #random change of mutating different genes throughout the network
        probailities=np.random.random(self.gene_size)
        self.geno[np.where(probailities<rate)]+=np.random.normal(0,self.std,self.geno[np.where(probailities<rate)].shape).astype(np.float32)
        self.geno[self.geno<-5]=-5
        self.geno[self.geno>5]=5


        idx=self.kernel_vals
        for i in range(len(self.w)):
            size=self.w[i].flatten().shape[0]
            self.w[i]=self.geno[idx:idx+size].reshape(self.w[i].shape)
            idx+=size
            size=self.b[i].flatten().shape[0]
            self.b[i]=self.geno[idx:idx+size].reshape(self.b[i].shape)
            idx+=size

    def activation(self, x):
        return np.tanh(x)

    def step(self, x):
        x = x.copy()
        pointer = 0
        x = x.squeeze() 
        in_data = x[np.newaxis, :, :]  #add channel dimension

        for layer_idx, (ksize, nkernels) in enumerate(zip(self.kernel_sizes, self.num_kernels)):
            kernels_shape = (nkernels, in_data.shape[0], ksize[0], ksize[1])
            num_kernel_params = np.prod(kernels_shape)
            kernels = self.geno[pointer:pointer+num_kernel_params].reshape(kernels_shape)
            pointer += num_kernel_params

            biases_shape = (nkernels,)
            biases = self.geno[pointer:pointer+nkernels]
            pointer += nkernels

            #convolve
            out_data = []
            for i in range(nkernels):
                summed = np.zeros((
                    in_data.shape[1] - ksize[0] + 1,
                    in_data.shape[2] - ksize[1] + 1
                ))
                for j in range(in_data.shape[0]):  #over input channels
                    summed += scipy.signal.correlate2d(in_data[j], kernels[i, j], mode='valid')
                summed += biases[i]
                out_data.append(self.activation(summed))
            in_data = np.array(out_data)

        #flatten and fully connect
        flat = in_data.flatten()
        #pass through
        x=self.relu(np.dot(flat,self.w[0])+self.b[0])
        output=np.dot(x,self.w[1])+self.b[1]
        return output
    def relu(self,x):
        return np.maximum(0, x)
    def sex(self,geno1,geno2,prob_winning=0.6):
        probabilities=np.random.random(self.gene_size)
        geno2.geno[np.where(probabilities<prob_winning)]=geno1.geno[np.where(probabilities<prob_winning)]
        return geno2

class controllerCNN_LRF(controllerCNN):
    def step(self, x):
        x = x.copy()
        pointer = 0
        x = x.squeeze() 
        in_data = x[np.newaxis, :, :]  #add channel dimension

        for layer_idx, (ksize, nkernels) in enumerate(zip(self.kernel_sizes, self.num_kernels)):
            kernels_shape = (nkernels, in_data.shape[0], ksize[0], ksize[1])
            num_kernel_params = np.prod(kernels_shape)
            kernels = self.geno[pointer:pointer+num_kernel_params].reshape(kernels_shape)
            pointer += num_kernel_params

            biases_shape = (nkernels,)
            biases = self.geno[pointer:pointer+nkernels]
            pointer += nkernels

            #convolve
            out_data = []
            for i in range(nkernels):
                summed = np.zeros((
                    in_data.shape[1] - ksize[0] + 1,
                    in_data.shape[2] - ksize[1] + 1
                ))
                for j in range(in_data.shape[0]):  #over input channels
                    summed += scipy.signal.correlate2d(in_data[j], kernels[i, j], mode='valid')
                summed += biases[i]
                out_data.append(self.activation(summed))
            in_data = np.array(out_data)

        #flatten and fully connect
        flat = in_data.flatten()
        #pass through
        x=self.relu(np.dot(flat,self.w[0])+self.b[0])
        output=np.dot(x,self.w[1])+self.b[1]
        return np.argmax(output)
    
if __name__=="__main__":
    #control=controller(100,[10,10],2)
    control=controllerCNN_LRF((40,8),512,3)
    control.step(np.random.random((1,40,8,1)))
    print("Mutating")
    control.mutate()
    out=control.step(np.random.random((1,40,8,1)))
    print(out)