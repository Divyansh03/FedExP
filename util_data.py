from util_libs import *


def getDirichletData(y, n, alpha, num_c):
        n_nets = n
        K = num_c

        labelList_true = y


        min_size = 0
        N = len(labelList_true)
        rnd = 0

        net_dataidx_map = {}

        p_client = np.zeros((n,K))

        for i in range(n):
          p_client[i] = np.random.dirichlet(np.repeat(alpha,K))

    
        
        idx_batch = [[] for _ in range(n)]
        
        m = int(N/n)
        
        for k in range(K):
            idx_k = np.where(labelList_true == k)[0]

            np.random.shuffle(idx_k)

            proportions = p_client[:,k]

            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]

        for j in range(n_nets):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]

        net_cls_counts = {}

        for net_i, dataidx in net_dataidx_map.items():
            unq, unq_cnt = np.unique(labelList_true[dataidx], return_counts=True)
            tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
            net_cls_counts[net_i] = tmp

        local_sizes = []
        for i in range(n_nets):
            local_sizes.append(len(net_dataidx_map[i]))
        local_sizes = np.array(local_sizes)
        weights = local_sizes / np.sum(local_sizes)

        print('Data statistics: %s' % str(net_cls_counts))
        print('Data ratio: %s' % str(weights))

        return idx_batch
    
def getDirichletData_equal(y, n, alpha, num_c):
        n_nets = n
        K = num_c

        labelList_true = y


        min_size = 0
        N = len(labelList_true)
        rnd = 0

        net_dataidx_map = {}

        p_client = np.zeros((n,K))

        for i in range(n):
          p_client[i] = np.random.dirichlet(np.repeat(alpha,K))
            
        p_client_cdf = np.cumsum(p_client, axis=1)
      
        
        idx_batch = [[] for _ in range(n)]
        
        m = int(N/n)
        
        
        idx_labels = [np.where(labelList_true==k)[0] for k in range(K)]

        
        idx_counter = [0 for k in range(K)]
        total_cnt = 0
        
        
        while(total_cnt<m*n):
                
            curr_clnt = np.random.randint(n)
            
            if (len(idx_batch[curr_clnt])>=m):
                continue

            
            total_cnt += 1
            curr_prior = p_client_cdf[curr_clnt]
                
            while True:
                cls_label = np.argmax(np.random.uniform() <= curr_prior)
                if (idx_counter[cls_label] >= len(idx_labels[cls_label])):
                    continue

                idx_batch[curr_clnt].append(idx_labels[cls_label][idx_counter[cls_label]])
                idx_counter[cls_label] += 1

                break

        for j in range(n_nets):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]

        net_cls_counts = {}

        for net_i, dataidx in net_dataidx_map.items():
            unq, unq_cnt = np.unique(labelList_true[dataidx], return_counts=True)
            tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
            net_cls_counts[net_i] = tmp

        local_sizes = []
        for i in range(n_nets):
            local_sizes.append(len(net_dataidx_map[i]))
        local_sizes = np.array(local_sizes)
        weights = local_sizes / np.sum(local_sizes)

        print('Data statistics: %s' % str(net_cls_counts))
        print('Data ratio: %s' % str(weights))

        return idx_batch
    
    

def get_dataset(datatype, n_client, n_c, alpha, partition_equal=True):

    trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262])])

    trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    trans_fashionmnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])


    if(datatype=='CIFAR10' or datatype=='CIFAR100' or datatype=='MNIST' or datatype =='FashionMNIST' or datatype == 'CINIC10'):
    
        if(datatype=='CIFAR10'):

            dataset_train_global = datasets.CIFAR10('./data/cifar10', train=True, download=True, transform=trans_cifar)
            dataset_test_global = datasets.CIFAR10('./data/cifar10', train=False, download=True, transform=trans_cifar)
                
        if(datatype=='CINIC10'):
            cinic_mean = [0.47889522, 0.47227842, 0.43047404]
            cinic_std = [0.24205776, 0.23828046, 0.25874835]
            dataset_train_global = datasets.ImageFolder('cinic_train',transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=cinic_mean,std=cinic_std)]))
            dataset_test_global = datasets.ImageFolder('cinic_test',transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=cinic_mean,std=cinic_std)]))

        elif(datatype=='CIFAR100'):

            dataset_train_global = datasets.CIFAR100('./data/cifar100', train=True, download=True, transform=trans_cifar)
            dataset_test_global = datasets.CIFAR100('./data/cifar100', train=False, download=True, transform=trans_cifar)

        elif(datatype=='MNIST'):

            dataset_train_global = datasets.MNIST('./data/mnist', train=True, download=True, transform=trans_mist)
            dataset_test_global = datasets.MNIST('./data/mnist', train=False, download=True, transform=trans_mnist)

        elif(datatype=='FashionMNIST'):


            dataset_train_global = datasets.FashionMNIST('./data/fashionmnist', train=True, download=True, transform=trans_fashionmist)
            dataset_test_global = datasets.FashionMNIST('./data/fashionmnist', train=False, download=True, transform=trans_fashionmnist)


        

        train_loader = DataLoader(dataset_train_global, batch_size=len(dataset_train_global))
        test_loader  = DataLoader(dataset_test_global, batch_size=len(dataset_test_global))

        X_train = next(iter(train_loader))[0].numpy()
        Y_train = next(iter(train_loader))[1].numpy()

        X_test = next(iter(test_loader))[0].numpy()
        Y_test = next(iter(test_loader))[1].numpy()


        if(partition_equal == True):
            inds = getDirichletData_equal(Y_train, n_client, alpha, n_c)
        else:
            inds = getDirichletData(Y_train, n_client, alpha, n_c)


        dataset_train=[]
        dataset_test = []

        len_test = int(len(X_test)/n_client)


        for (i,ind) in enumerate(inds):


            ind = inds[i]
            
            x = X_train[ind]
            y = Y_train[ind]
                

            x_test = X_test[i*len_test:(i+1)*len_test]
            y_test = Y_test[i*len_test:(i+1)*len_test]
            

            n_i = len(ind)

            x_train = torch.Tensor(x[0:n_i])
            y_train = torch.LongTensor(y[0:n_i])

            x_test = torch.Tensor(x_test)
            y_test = torch.LongTensor(y_test)


            print ("Client ", i, " Training examples-" , len(x_train), " Test examples-", len(x_test))

            dataset_train_torch = TensorDataset(x_train,y_train)
            dataset_test_torch = TensorDataset(x_test,y_test)

            dataset_train.append(dataset_train_torch)
            dataset_test.append(dataset_test_torch)
    

    if(datatype=='EMNIST'):

      
      with ZipFile('emnist_dataset_umifa.npy.zip', 'r') as f:
        f.extractall()


      emnist_data = np.load('emnist_dataset_umifa.npy', allow_pickle= True).item()
      dataset_train_emnist = emnist_data['dataset_train']
      dataset_test_emnist = emnist_data['dataset_test']
      dict_users_emnist = emnist_data['dict_users']
      emnist_clients = list(dict_users_emnist.keys())

      x_train = dataset_train_emnist[:][0]
      y_train = dataset_train_emnist[:][1]
      x_train = x_train[:,None]

      dataset_train_emnist_new = TensorDataset(x_train,y_train)

      x_test = dataset_test_emnist[:][0]
      y_test = dataset_test_emnist[:][1]
      x_test = x_test[:,None]

      dataset_test_emnist_new = TensorDataset(x_test,y_test)

      dataset_test_global = dataset_test_emnist_new

      dataset_train=[]
      dataset_test = []

      n = len(emnist_clients)

      len_test = int(len(dataset_test_emnist)/n)


      ctr = 0

      for i in range(n):
          
        ind = dict_users_emnist[i]

        x_train = dataset_train_emnist_new[ind][0]
        y_train = dataset_train_emnist_new[ind][1]

        x_test = dataset_test_emnist_new[i*len_test:(i+1)*len_test][0]
        y_test = dataset_test_emnist_new[i*len_test:(i+1)*len_test][1]

        

        n_i = len(ind)

        dataset_train_torch = TensorDataset(x_train,y_train)
        dataset_test_torch = TensorDataset(x_test,y_test)


        dataset_train.append(dataset_train_torch)
        dataset_test.append(dataset_test_torch)


    return dataset_train, dataset_test_global
    







    

    








