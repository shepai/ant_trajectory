import numpy as np
import random 
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
        self.reform()
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
    def reform(self):
        idx=0
        for i in range(len(self.w)):
            size=self.w[i].flatten().shape[0]
            self.w[i]=self.geno[idx:idx+size].reshape(self.w[i].shape)
            idx+=size
            size=self.b[i].flatten().shape[0]
            self.b[i]=self.geno[idx:idx+size].reshape(self.b[i].shape)
            idx+=size

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
        self.reform()
        print("mutated")

    def activation(self, x):
        return np.tanh(x)

    def forward(self, x):
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
        x=flat
        for i in range(len(self.w)):
            x=self.relu(np.dot(x,self.w[i])+self.b[i])
        output=x
        return output
    def step(self,x):
        return self.forward(x)
    def relu(self,x):
        return np.maximum(0, x)
    def sex(self,geno1,geno2,prob_winning=0.6):
        probabilities=np.random.random(self.gene_size)
        geno2.geno[np.where(probabilities<prob_winning)]=geno1.geno[np.where(probabilities<prob_winning)]
        return geno2
    def insert_layer(self,weights,bias,n):
        self.w.insert(n,weights)
        self.b.insert(n,bias)
        idx=self.kernel_vals
        assert n>0 and n<len(self.w),"no weight there"
        for i in range(len(self.w)):
            if i==n: #if it is the layer
                self.geno=np.concatenate([self.geno[:idx],weights.flatten(),self.geno[idx:]])
                self.geno=np.concatenate([self.geno[:idx+len(weights.flatten())],bias.flatten(),self.geno[idx+len(weights.flatten()):]])
            idx+=len(self.w[i].flatten())+len(self.b[i].flatten())
        self.reform()
    def delete_layer(self):
        if len(self.w)>2: #can delete
            n=random.choice([i for i in range(1,len(self.w)-1)])
            del self.w[n]
            del self.b[n]
            #remove geno part
            idx=self.kernel_vals
            for i in range(len(self.w)):
                if i==n: #if it is the layer
                    self.geno = np.concatenate((self.geno[:idx], self.geno[idx+len(self.w[i].flatten()):]))
                    self.geno = np.concatenate((self.geno[:idx], self.geno[idx+len(self.b[i].flatten()):]))
                idx+=len(self.w[i].flatten())+len(self.b[i].flatten())
        self.reform()
    def reform(self):
        idx=0
        in_channels = 1 
        for ksize, nkernels in zip(self.kernel_sizes, self.num_kernels):
            kernel_shape = (nkernels, in_channels, ksize[0], ksize[1])
            bias_shape = (nkernels,)
            geno_k=self.geno[idx:idx+(nkernels* in_channels* ksize[0]* ksize[1])]
            kernels = geno_k.reshape(kernel_shape).astype(np.float32)
            idx=idx+(nkernels* in_channels* ksize[0]* ksize[1])
            geno_k=self.geno[idx:idx+nkernels]
            biases = geno_k.reshape(bias_shape).astype(np.float32)

        idx=self.kernel_vals
        for i in range(len(self.w)):
            size=self.w[i].flatten().shape[0]
            self.w[i]=self.geno[idx:idx+size].reshape(self.w[i].shape)
            idx+=size
            size=self.b[i].flatten().shape[0]
            self.b[i]=self.geno[idx:idx+size].reshape(self.b[i].shape)
            idx+=size
    def show(self):
        print("NETWORK ARCH")
        for ksize, nkernels in zip(self.kernel_sizes, self.num_kernels):
            print("\tCNN layer",ksize,nkernels)
        for i in range(len(self.w)):
            print("\tLinear layer",self.w[i].shape)
            print("\tBias",self.b[i].shape)
class controllerCNN_LRF(controllerCNN):
    def step(self, x):
        output=self.forward(x)
        return np.argmax(output)
    
if __name__=="__main__":
    #control=controller(100,[10,10],2)
    control=controllerCNN_LRF((40,8),512,3)
    control.step(np.random.random((1,40,8,1)))
    print("Mutating")
    control.mutate()
    out=control.step(np.random.random((1,40,8,1)))
    layer=[np.random.normal(1,5,(512,512)),np.random.normal(1,5,(512,))]
    control.insert_layer(layer[0],layer[1],1)
    control.show()
    layer=[np.random.normal(1,5,(512,512)),np.random.normal(1,5,(512,))]
    control.insert_layer(layer[0],layer[1],1)
    control.show()
    control.delete_layer()
    control.show()
    control.delete_layer()
    control.show()
    control.delete_layer()
    control.show()