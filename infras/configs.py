
class Config(object):
        
    def parse(self, kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)
        #
        
        print('=================================')
        print('*', self.config_name)
        print('---------------------------------')
        
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('_'):
                print('-', k, ':', getattr(self, k))
        print('=================================')
        
        
    def __str__(self,):
        
        buff = ""
        buff += '=================================\n'
        buff += ('*'+self.config_name+'\n')
        buff += '---------------------------------\n'
        
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('_'):
                buff += ('-' + str(k) + ':' + str(getattr(self, k))+'\n')
            #
        #
        buff += '=================================\n'
        
        return buff    
    
    
class ExpConfig(Config):

    domain = None
    heuristic = None
    mode = None
    
    nfree = 5
    nplay = 200
    nroll = 100

    num_adam=1
    int_adam=1000
    lr_adam=1e-3
    
    num_lbfgs=1
    int_lbfgs=10000
    lr_lbfgs=1e-1

#     nfree = 2
#     nplay = 3
#     nroll = 3
    
#     num_adam=3
#     int_adam=50
#     lr_adam=1e-3
    
#     num_lbfgs=2
#     int_lbfgs=100
#     lr_lbfgs=1e-1
    
    device = 'cpu'
    
    verbose=True
    
    def __init__(self,):
        super(ExpConfig, self).__init__()
        self.config_name = 'XPINN-MAB-Configs'
        


        
        
        
        
     

        