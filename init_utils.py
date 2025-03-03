import torch
from data_utils import generate_Cifar, generate_Mnist, generate_FashionMnist
from opacus.accountants.utils import get_noise_multiplier
from opacus.validators import ModuleValidator
import timm
import os
from datetime import datetime
from model_utils import LinearModel, CNN5

def base_parse_args(parser):
    # Task arguments
    parser.add_argument('--cuda', default=0, type=int, help='cuda device')
    parser.add_argument('--tag', default = '', type=str, help='log file tag')
    parser.add_argument('--log_path', default = './log', type=str, help='log file path')
    parser.add_argument('--log_freq', default=-1, type=int, help='log frequency during training')
    parser.add_argument('--load_path', default=None, type=str, help='load checkpoint if specified')
    parser.add_argument('--save_path', default=None, type=str, help='save checkpoint is specified')
    parser.add_argument('--save_freq', default=999, type=int, help='checkpoint saving frequency')

    parser.add_argument('--data', default='cifar10', type=str, help='dataset (cifar10, cifar100)')
    parser.add_argument('--bs', default=256, type=int, help='batch size')
    parser.add_argument('--mnbs', default=32, type=int, help='mini batch size')
    parser.add_argument('--model', default = 'vit_small_patch16_224', type=str, help='trained model')
    parser.add_argument('--pretrained', action='store_true', help='use pre-trained weights')

    # Algorithm parameters
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate list')
    parser.add_argument('--beta', default=0.999, type=float, help='beta for adam')
    parser.add_argument('--epoch', default=3, type=int,help='number of public epochs')

    # DP parameters
    parser.add_argument('--epsilon', default=8, type=float, help='dp privacy, must be larger than 0, used when noise is not specified')
    parser.add_argument('--clipping_norm', default=-1,  type=float, help='clipping style, <=0: automatic, >0: Abadi')
    parser.add_argument('--clipping_style', default='all-layer', type=str, help='clipping style, all-layer, layer-wise, param-wise')
    return parser

def task_init(args):
    device = torch.device('cuda:'+str(args.cuda)) if torch.cuda.is_available() else torch.device('cpu')
    if args.clipping_norm <=0:
        args.clipping_fn = 'automatic'
        args.clipping_norm = 1
    else:
        args.clipping_fn = 'Abadi'
    model = None
    if args.data == 'cifar10':
        num_classes = 10
        train_dl, test_dl = generate_Cifar(args.mnbs, args.data, args.model)
        sample_size = 50000
    elif args.data == 'cifar100':
        num_classes = 100
        train_dl, test_dl = generate_Cifar(args.mnbs, args.data, args.model)
        sample_size = 50000
    elif args.data == 'mnist':
        model = LinearModel(28*28, 10)
        train_dl, test_dl = generate_Mnist(args.mnbs, args.data)
        sample_size = 60000
    elif args.data == 'fashion-mnist':
        model = LinearModel(28*28, 10) 
        train_dl, test_dl = generate_FashionMnist(args.mnbs, args.data)
        sample_size = 60000 
    if model is None:
        if args.model != 'cnn5':
            model = timm.create_model(args.model, pretrained=args.pretrained, num_classes = num_classes)
        else:
            model = CNN5(num_classes = num_classes, normalization= True)
    model = ModuleValidator.fix(model)
    model.to(device)
    if args.load_path is not None:
        checkpoint = torch.load(args.model_path, map_location='cuda')
        model.load_state_dict(checkpoint['model'], strict = True)

    noise = get_noise_multiplier(target_delta=1.0/(sample_size)**1.1, target_epsilon=args.epsilon, sample_rate=args.bs/sample_size, epochs=args.epoch)
    acc_step = args.bs//args.mnbs
    
    return train_dl, test_dl, model, device, sample_size, acc_step, noise

def logger_init(args, noise, steps_per_epoch, type = 'file'):
    if not os.path.isdir(args.log_path):
        os.makedirs(args.log_path)
    log_file_path = '%s/%s'%(args.log_path,args.tag)
    log_file = file_logger(log_file_path, 2, ["acc","loss"], steps_per_epoch, heading = "Data=%s, Model=%s, E=%d, B=%d, lr=%-.6f, sigma=%-.6f, coef=%s"%(args.data, args.model, args.epoch, args.bs, args.lr, noise, args.coef_file))
    return log_file
    

class file_logger():
    def __init__(self, path, time_num, item_list, steps_per_epoch, heading = None):
        head = ['time_'+str(i) for i in range(time_num)]
        head_str = ','.join(head)+','+','.join(item_list)
        now = datetime.now()
        dt_string = now.strftime("%d%m%Y_%H_%M_%S")
        self.train_path = path+'_train'+dt_string+'.csv'
        self.test_path = path+'_test'+dt_string+'.csv'
        self.epoch_per_step = 1.0/steps_per_epoch
        self.time_num = time_num
        self.item_length = len(item_list)
        with open(self.train_path,'a') as fp:
            if heading is not None:
                print(heading, file=fp)
            print(head_str, file=fp)
        with open(self.test_path,'a') as fp:
            if heading is not None:
                print(heading, file=fp)
            print(head_str, file=fp)
    
    def update(self, time_list, item_list):
        if len(time_list)!=self.time_num:
            raise RuntimeError('incorrect log time information')
        
        if time_list[1] == -1:
            # test log
            log_info = str(time_list[0])+','+','.join(map(str,item_list))
            with open(self.test_path,'a') as fp:
                print(log_info, file=fp)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            # train log
            log_info = str(time_list[0]+time_list[1]*self.epoch_per_step)+','+','.join(map(str,item_list))
            with open(self.train_path,'a') as fp:
                print(log_info, file=fp)
        

