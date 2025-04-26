import numpy as np

class controller:
    def __init__(self,input_size,hiddensizes,output,std=5):
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
        for i in range(len(hiddensizes)):
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
    def activation(self,x): #can replace with any activation function
        return np.tanh(x)
    def step(self,x):
        #take in parameter x which is the input data, this can be linear data or an image
        x=x.flatten()
        for i in range(len(self.w)-1):
            x=self.activation(np.dot(x,self.w[i]) + self.b[i])
        return np.dot(x,self.w[-1]) + self.b[-1]
    def sex(self,geno1,geno2,prob_winning=0.6):
        probabilities=np.random.random(self.gene_size)
        geno2.geno[np.where(probabilities<prob_winning)]=geno1.geno[np.where(probabilities<prob_winning)]
        return geno2
    
import numpy as np
import scipy.signal  # for convolution without needing torch

class controllerCNN:
    def __init__(self, input_shape, kernel_sizes, num_kernels, output_size, std=0.1):
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
            kernel_shape = (nkernels, in_channels, ksize, ksize)
            bias_shape = (nkernels,)
            kernels = np.random.normal(0, std, kernel_shape)
            biases = np.random.normal(0, std, bias_shape)

            self.kernels.append(kernels)
            self.biases.append(biases)

            self.geno = np.concatenate([self.geno, kernels.flatten(), biases.flatten()])
            in_channels = nkernels  # for next layer

        # Fully connected layer from conv outputs to output
        # Flatten size needs to be estimated. Assume input shrinks by (kernel_size - 1) each conv layer without padding
        h, w = input_shape
        for k in kernel_sizes:
            h -= (k - 1)
            w -= (k - 1)
        flattened_size = h * w * in_channels

        self.fc_w = np.random.normal(0, std, (flattened_size, output_size))
        self.fc_b = np.random.normal(0, std, (output_size,))
        self.geno = np.concatenate([self.geno, self.fc_w.flatten(), self.fc_b.flatten()])

        self.gene_size = len(self.geno)

    def mutate(self, rate=0.2):
        probabilities = np.random.random(self.gene_size)
        self.geno[np.where(probabilities < rate)] += np.random.normal(0, self.std, self.geno[np.where(probabilities < rate)].shape)

    def activation(self, x):
        return np.tanh(x)

    def step(self, x):
        x = x.copy()
        pointer = 0
        in_data = x[np.newaxis, :, :]  #add channel dimension

        for layer_idx, (ksize, nkernels) in enumerate(zip(self.kernel_sizes, self.num_kernels)):
            kernels_shape = (nkernels, in_data.shape[0], ksize, ksize)
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
                    in_data.shape[1] - ksize + 1,
                    in_data.shape[2] - ksize + 1
                ))
                for j in range(in_data.shape[0]):  #over input channels
                    summed += scipy.signal.correlate2d(in_data[j], kernels[i, j], mode='valid')
                summed += biases[i]
                out_data.append(self.activation(summed))
            in_data = np.array(out_data)

        #flatten and fully connect
        flat = in_data.flatten()

        fc_weight_size = self.fc_w.size
        fc_weights = self.geno[pointer:pointer+fc_weight_size].reshape(self.fc_w.shape)
        pointer += fc_weight_size

        fc_biases = self.geno[pointer:pointer+self.fc_b.size]

        output = np.dot(flat, fc_weights) + fc_biases
        return output

    def sex(self, parent1, parent2, prob_winning=0.6):
        probabilities = np.random.random(self.gene_size)
        parent2.geno[np.where(probabilities < prob_winning)] = parent1.geno[np.where(probabilities < prob_winning)]
        return parent2
