
##################################
### Main: Server side optimization and testing
#################################
# Initiate the NN

from sys import float_info
from util_data import *
from util_models import *
from util_general import *




parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, required=True)
parser.add_argument('--algorithm', type=str, required=True)
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--num_clients', type=int, required=True)
parser.add_argument('--num_participating_clients', type=int, required=True)
parser.add_argument('--num_rounds', type=int, required=True)
parser.add_argument('--alpha', type=float, required=True)

args_required = parser.parse_args()



seed = args_required.seed
dataset = args_required.dataset
algorithm = args_required.algorithm
model = args_required.model
num_clients = args_required.num_clients
num_participating_clients = args_required.num_participating_clients
num_rounds = args_required.num_rounds
alpha = args_required.alpha

print_every_test = 5
print_every_train = 5



filename = "results_"+str(seed)+"_"+algorithm+"_"+dataset+"_"+model+"_"+str(num_clients)+"_"+str(num_participating_clients)+"_"+str(num_rounds)+"_"+str(alpha)
filename_txt = filename + ".txt"


if(dataset=='CIFAR100'):
  n_c = 100
elif (dataset == 'EMNIST'):
  n_c = 62
else: n_c = 10








np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True

dataset_train, dataset_test_global = get_dataset(dataset, num_clients, n_c, alpha, True)



dict_results = {} ###dictionary to store results for all algorithms


###Default training parameters for all algorithms

args={
"bs":50,   ###batch size
"cp":20,   ### number of local steps
"device":'cuda',
"rounds":num_rounds, 
"num_clients": num_clients,
"num_participating_clients":num_participating_clients
}




net_glob_org = get_model(model,n_c).to(args['device'])




algs = [algorithm]


decay=  0.998
max_norm = 10
use_gradient_clipping = True
weight_decay = 0.0001

if(dataset=='CIFAR10'):
  eta_l_fedavg = 0.01
  eta_l_fedexp = 0.01
  eta_l_scaffold = 0.01
  eta_l_scaffold_exp = 0.01
  eta_l_fedadagrad = 0.01
  eta_l_fedprox = 0.01
  eta_l_fedprox_exp = 0.01
  eta_l_fedadam = 0.01
  eta_l_fedavgm = 0.01
  eta_l_fedavgm_exp = 0.01

  eta_g_fedavg = 1
  eta_g_scaffold = 1
  eta_g_fedprox = 1
  eta_g_fedadagrad = 0.1
  eta_g_fedadam = 0.1
  eta_g_fedavgm = 1
 
  

  epsilon_fedexp = 0.001
  epsilon_scaffold_exp = 0.001
  epsilon_fedprox_exp = 0.001
  epsilon_fedavgm_exp = 0.001
  
elif(dataset=='CINIC10'):
  eta_l_fedavg = 0.01
  eta_l_fedexp = 0.01
  eta_l_scaffold = 0.01
  eta_l_scaffold_exp = 0.01
  eta_l_fedadagrad = 0.01
  eta_l_fedprox = 0.01
  eta_l_fedprox_exp = 0.01
  eta_l_fedadam = 0.01
  eta_l_fedavgm = 0.01
  eta_l_fedavgm_exp = 0.01

  eta_g_fedavg = 1
  eta_g_scaffold = 1
  eta_g_fedprox = 1
  eta_g_fedadagrad = 0.1
  eta_g_fedadam = 0.1
  eta_g_fedavgm = 1
  
 


  epsilon_fedexp =  0.001
  epsilon_scaffold_exp = 0.001
  epsilon_fedprox_exp = 0.001
  epsilon_fedavgm_exp = 0.001

elif(dataset=='CIFAR100'):
  eta_l_fedavg = 0.01
  eta_l_fedexp = 0.01
  eta_l_scaffold = 0.01
  eta_l_scaffold_exp = 0.01
  eta_l_fedadagrad = 0.01
  eta_l_fedprox = 0.01
  eta_l_fedprox_exp = 0.01
  eta_l_fedadam = 0.01
  eta_l_fedavgm = 0.01
  eta_l_fedavgm_exp = 0.01

  eta_g_fedavg = 1
  eta_g_scaffold = 1
  eta_g_fedadagrad = 0.1
  eta_g_fedadam = 0.1
  eta_g_fedprox = 1
  eta_g_fedavgm = 1
  

  epsilon_fedexp = 0.001
  epsilon_scaffold_exp = 0.001
  epsilon_fedprox_exp = 0.001
  epsilon_fedavgm_exp = 0.001


elif(dataset=='EMNIST'):
  eta_l_fedavg = 0.1
  eta_l_fedexp = 0.1
  eta_l_scaffold = 0.1
  eta_l_scaffold_exp = 0.1
  eta_l_fedadagrad = 0.1
  eta_l_fedprox = 0.1
  eta_l_fedprox_exp = 0.1
  eta_l_fedadam = 0.1
  eta_l_fedavgm = 0.316
  eta_l_fedavgm_exp = 0.316

  eta_g_fedavg = 1
  eta_g_scaffold = 1
  eta_g_fedadagrad = 0.316
  eta_g_fedadam = 0.316
  eta_g_fedprox = 1
  eta_g_fedavgm = 1
 

  epsilon_fedexp = 0.1
  epsilon_scaffold_exp = 0.1
  epsilon_fedprox_exp = 0.1
  epsilon_fedavgm_exp = 0.1


epsilon_fedadagrad = 0.01
epsilon_fedadam = 0.01
  
if(dataset=='EMNIST'):
    epsilon_fedadagrad = 0.0316
    epsilon_fedadam = 0.0316
    


mu_fedprox = 0

if(dataset=='CIFAR10'):
  mu_fedprox = 0.1
  
if(dataset=='CINIC10'):
  mu_fedprox = 1
  
if(dataset=='EMNIST'):
  mu_fedprox = 0.001
 
if (dataset=='CIFAR100'):
  mu_fedprox = 0.001

 





eta_l_algs = {'fedavgm(exp)': eta_l_fedavgm_exp, 'fedavgm': eta_l_fedavgm,'fedadam':eta_l_fedadam, 'fedprox':eta_l_fedprox, 'fedprox(exp)': eta_l_fedprox_exp, 'fedavg': eta_l_fedavg, 'fedadagrad': eta_l_fedadagrad, 'fedexp': eta_l_fedexp, 'scaffold': eta_l_scaffold, 'scaffold(exp)': eta_l_scaffold_exp}

eta_g_algs = {'fedavgm(exp)': 'adaptive', 'fedavgm': eta_g_fedavgm,'fedadam':eta_g_fedadam, 'fedprox':eta_g_fedprox, 'fedprox(exp)': 'adaptive', 'fedavg':eta_g_fedavg, 'fedadagrad': eta_g_fedadagrad, 'fedexp': 'adaptive', 'scaffold': eta_g_scaffold, 'scaffold(exp)': 'adaptive'}

epsilon_algs = {'fedavgm(exp)': epsilon_fedavgm_exp, 'fedavgm': 0,'fedadam': 0, 'fedprox':0, 'fedprox(exp)':epsilon_fedprox_exp, 'fedavg': 0, 'fedadagrad':0, 'fedexp':epsilon_fedexp, 'scaffold': 0, 'scaffold(exp)': epsilon_scaffold_exp}

mu_algs = {'fedavgm(exp)': 0, 'fedavgm': 0, 'fedadam':0, 'fedprox': mu_fedprox, 'fedprox(exp)': mu_fedprox, 'fedavg':0, 'fedadagrad':0, 'fedexp':0, 'scaffold':0, 'scaffold(exp)':0}



n = len(dataset_train)
print ("No. of clients", n)

p = np.zeros((n))

for i in range(n):
  p[i] = len(dataset_train[i])
             
p = p/np.sum(p)




for alg in algs:


    dict_results[alg] = {}
    
    filename_model_alg = alg + "_" + filename+".pt"

    d = parameters_to_vector(net_glob_org.parameters()).numel()


    net_glob = copy.deepcopy(net_glob_org)


    net_glob.train()


    w_glob = net_glob.state_dict()

    loss_t = []

    train_loss_algo_tmp = []
    train_acc_algo_tmp = []
    test_loss_algo_tmp = []
    test_acc_algo_tmp = []
    eta_g_tmp = []

    
    grad_mom = torch.zeros(d).to(args['device'])
    
    mem_mat = torch.zeros((n, d)).to(args['device'])  ###needed for scaffold
    
    w_vec_estimate = torch.zeros(d).to(args['device'])
    
    delta = torch.zeros(d).to(args['device'])

    grad_norm_avg_running = 0

    
    
    local_lr = eta_l_algs[alg]
    global_lr = eta_g_algs[alg]
    epsilon = epsilon_algs[alg]
    mu = mu_algs[alg]

    
    for t in range(0,args['rounds']+1):
        

        print ("Algo ", alg, " Round No. " , t)

        local_lr = decay*local_lr
        epsilon = decay*decay*epsilon

        args_hyperparameters = {'mu': mu, 'eta_l':local_lr, 'decay': decay, 'weight_decay': weight_decay, 'eta_g': global_lr, 'use_gradient_clipping': use_gradient_clipping, 'max_norm': max_norm, 'epsilon': epsilon, 'use_augmentation':True}
        
        
        if(dataset=='CIFAR10' or dataset=='CIFAR100' or dataset=='CINIC10'):
          args_hyperparameters['use_augmentation'] = True
        else:
          args_hyperparameters['use_augmentation'] = False
      
        S = args['num_participating_clients']

        ind = np.random.choice(n,S,replace=False)



        grad_avg = torch.zeros(d).to(args['device'])

        c = torch.zeros((d,)).to(args['device'])
       

        w_init = parameters_to_vector(net_glob.parameters()).to(args['device'])

        grad_norm_sum = 0
        
        p_sum = 0

        
        if(alg=='scaffold' or alg=='scaffold(exp)'):
            for i in range(n):
                c = c+ p[i]*mem_mat[i]

            

        for i in ind:


            grad = get_grad(copy.deepcopy(net_glob),args, args_hyperparameters, dataset_train[i], alg, i,  mem_mat, c)

            grad_norm_sum += p[i]*torch.linalg.norm(grad)**2

            grad_avg = grad_avg + p[i]*grad
            
            p_sum += p[i]


        

        with torch.no_grad():



            grad_avg = grad_avg/p_sum
            
            grad_norm_avg = grad_norm_sum/p_sum

            eta_g = args_hyperparameters['eta_g']

            grad_norm_avg_running = grad_norm_avg +0.9*0.5*grad_norm_avg_running

            
            
            if(alg=='fedavgm' or alg=='fedavgm(ep)'):

              grad_avg = grad_avg + 0.9*grad_mom
                
              grad_mom = grad_avg


            if(alg=='fedadagrad'):
              
              delta = delta + grad_avg**2
                
              grad_avg = grad_avg/(torch.sqrt(delta+epsilon_fedadagrad))

            
            if(alg=='fedadam'):

              grad_avg = 0.1*grad_avg + 0.9*grad_mom
              grad_mom = grad_avg

              delta = 0.01*grad_avg**2 + 0.99*delta

              grad_avg_normalized = grad_avg/(0.1)
              delta_normalized = delta/(0.01)

              grad_avg = (grad_avg_normalized/torch.sqrt(delta_normalized + epsilon_fedadam))
            
            
            

            grad_avg_norm = torch.linalg.norm(grad_avg)**2

            if(eta_g == 'adaptive'):

              if(alg!='fedavgm(exp)'):
                eta_g = (0.5*grad_norm_avg/(grad_avg_norm + S*epsilon)).cpu()
              else:
                eta_g = (0.5*grad_norm_avg_running/(grad_avg_norm + S*epsilon)).cpu()


              if(alg!='fedavgm(exp)'):
                eta_g = max(1,eta_g)

            eta_g_tmp.append(eta_g)

            w_vec_prev = w_vec_estimate
            
            w_vec_estimate =  parameters_to_vector(net_glob.parameters()) + eta_g*grad_avg

            if(t>0):
              w_vec_avg = (w_vec_estimate+w_vec_prev)/2
            else:
              w_vec_avg = w_vec_estimate


            vector_to_parameters(w_vec_estimate,net_glob.parameters())
        
        
        
        net_eval = copy.deepcopy(net_glob)

        if(alg=='fedexp' or alg=='scaffold(exp)' or alg=='fedprox(exp)' or alg=='fedavgm(exp)'):
          vector_to_parameters(w_vec_avg, net_eval.parameters())

        
        if(t%print_every_test==0):


          if(t%print_every_train==0):
              
              sum_loss_train = 0
              sum_acc_train = 0
              for i in range(n):
                test_acc_i, test_loss_i = test_img(net_eval, dataset_train[i],args)

                sum_loss_train += test_loss_i
                sum_acc_train += test_acc_i

              sum_loss_train = sum_loss_train/n
              sum_acc_train = sum_acc_train/n



              print ("Training Loss ", sum_loss_train, "Training Accuracy ", sum_acc_train)
        
              train_loss_algo_tmp.append(sum_loss_train)
              train_acc_algo_tmp.append(sum_acc_train)


          
          sum_loss_test = 0
          sum_acc_test = 0

          
          test_acc_i, test_loss_i = test_img(net_eval, dataset_test_global,args)
          sum_loss_test = test_loss_i
          sum_acc_test = test_acc_i


          print ("Test Loss", sum_loss_test, "Test Accuracy ", sum_acc_test)

          

          test_loss_algo_tmp.append(sum_loss_test)
          test_acc_algo_tmp.append(sum_acc_test)

          dict_results[alg][alg+"_training_loss"] = train_loss_algo_tmp
          dict_results[alg][alg+"_training_accuracy"] = train_acc_algo_tmp
          dict_results[alg][alg+"_test_loss"] = test_loss_algo_tmp
          dict_results[alg][alg+"_testing_accuracy"] = test_acc_algo_tmp
          dict_results[alg][alg+"_global_learning_rate"] = eta_g_tmp


          torch.save(net_glob, filename_model_alg)


          with open(filename_txt, 'w') as f:    
            for i in dict_results.keys():
              for key, value in dict_results[i].items():
                f.write(key+" ")
                f.write(str(value))
                f.write("\n")





