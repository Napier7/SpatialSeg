import importlib


class dict2obj(object):
    def __init__(self, d):
        self.__dict__['d'] = d
 
    def __getattr__(self, key):
        value = self.__dict__['d'][key]
        if type(value) == type({}):
            return dict2obj(value)
        return value


#   Read a config file by path and return its Object. 
def set_cfg_from_file(cfg_path):
    spec = importlib.util.spec_from_file_location('cfg_file', cfg_path)
    cfg_file = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg_file)
    cfg_dic = cfg_file.cfg
    cfg_obj = dict2obj(cfg_dic)
    return cfg_obj, cfg_dic
         



