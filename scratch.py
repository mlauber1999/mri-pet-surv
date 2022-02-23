#/usr/bin/python
import numpy as np

def read_reqs(f_name):
    with open(f_name,'r') as fi:
        new_fi = fi.readlines()
    return list(map(lambda x: x[:-1], new_fi))

def strip_version(version_list):
    return list(map(lambda x: x.split('==')[0],version_list))

def _retrieve_version(module_list, module_name):
    match = list(map(lambda x: x.find(module_name), module_list))
    match = [x for (x,y) in zip(module_list, match) if y is not -1]
    return match

def compare_versions(modules1, modules2, modules_noversion):
    diff_versions = []
    for module in modules_noversion:
        module1_v = _retrieve_version(modules1, module)
        module2_v = _retrieve_version(modules2, module)
        if module1_v != module2_v:
            print(module1_v, module2_v)

_all = read_reqs('mri_pet_pipreqs_all.txt')

_cnn = read_reqs('pytorch_pipreqs_cnn.txt')
_vit = read_reqs('pytorch_pipreqs.txt')

_torch = np.union1d(_vit, _cnn)

_torch_noversion = strip_version(_torch)
_mripet_noversion = strip_version(_all)

overlapping = np.intersect1d(_torch_noversion, _mripet_noversion)

diff_versions = compare_versions(_torch, _all, overlapping)