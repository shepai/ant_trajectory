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