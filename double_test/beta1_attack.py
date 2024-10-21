#!/usr/bin/env python
# coding=utf-8
import sys
sys.path.append('..')

import os, json
import torch
from torch import nn
from model_zoo import ModelZoo
from dataset_zoo import DatasetZoo
from configure import *
import pdb
from attack_utils import save_one_img
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
import pickle
from torchattacks import attacks
from torchattacks.tools.seed import random_seed

def main():
    mzoo = ModelZoo()
    dzoo = DatasetZoo()
    with open(attack_book, 'r') as fp:
        attack_targets = json.load(fp)

    # for mixup in [0, 1, 2, 3, 4, 5, 6]:
    # for mixup in [2, 3, 4, 5, 6]:
    # for mixup in [6, 5]:
    # for mixup in [4]:
    # for mixup, beta2 in zip([0.9, 0.8, 0.7, 0.6, 0.5], [0.99, 0.89, 0.79, 0.69, 0.59]):
    for mixup, beta2 in zip([0.4, 0.3, 0.2, 0.1, 0.0], [0.49, 0.39, 0.29, 0.19, 0.1]):
        print('mixup num:', mixup)
        for mname in model_names:
        # for mname in ['wide_resnet50_2']:
            print('model {} generates adversarial examples...'.format(mname))
    
            adv_output_dir = os.path.join(beta1_output_path, str(mixup), mname)
            if not os.path.exists(adv_output_dir):
                os.makedirs(adv_output_dir)
    
            for (dname, dpath), (fbname, fbpath) in zip(victim_datasets, feature_libraries):
    
                adv_output_dir = os.path.join(beta1_output_path, str(mixup), mname, dname)
                if not os.path.exists(adv_output_dir):
                    os.makedirs(adv_output_dir)
    
                print('1. dataset {} is attacked...'.format(dname)) 
                ds = dzoo.load_dataset(dname, dpath)
                label_space = list(ds.class_to_idx.values())
    
                model = mzoo.pick_model(mname)
                feature_model, decision_model = mzoo.default_split(mname)
                model = model.cuda()
                model.eval()
                if feature_model is not None:
                    feature_model = feature_model.cuda()
                    decision_model = decision_model.cuda()
                    feature_model.eval()
                    decision_model.eval()
    
                for i, (attack_name, attack_args) in enumerate(baseline_attack_methods.items()):
                    random_seed()
    
                    # skip some methods
                    # if attack_name in ['MI-FGSM', 'NI-FGSM', 'AA', 'TAA', 'BSI-FGSM', 'VMI-FGSM', 'VNI-FGSM', 'BAP', 'SINI-FGSM', 'DI-FGSM']:
                    # if attack_name in ['AA', 'TAA', 'BSI-FGSM']:
                    if attack_name not in ['BAP']:
                        continue
    
                    if attack_name in ['AA', 'TAA']:
                        feature_library = {}
                        print('1.* building feature library for {}...'.format(dname))
    
                        fb_ds = dzoo.load_dataset(fbname, os.path.join(fbpath, mname))
                        k = 0
                        debug = False 
                        for (fs, lab) in tqdm(fb_ds):
                            if lab in feature_library:
                                feature_library[lab].append(fs.unsqueeze(0))
                            else:
                                feature_library[lab] = [fs.unsqueeze(0)]
    
                            k = k + 1
                            if k > 100 and debug:
                                break
                        for lab in feature_library.keys():
                            feature_library[lab] = torch.cat(feature_library[lab])
                    elif attack_name in ['BSI-FGSM']:
                        saliency_layer_name = mzoo.get_saliency_layer(mname)
                        saliency_layer = getattr(model[1], saliency_layer_name)
    
                    adv_output_dir = os.path.join(beta1_output_path, str(mixup), mname, dname, attack_name)
                    if not os.path.exists(adv_output_dir):
                        os.makedirs(adv_output_dir)
        
                    print('2.{} attack method {} is attacking...'.format(i, attack_name))
                    if attack_name == 'DI-FGSM':
                        alpha = attack_args['eps'] / attack_args['max_iter']
                        attack = attacks.difgsm.DIFGSM(model, eps=attack_args['eps'],
                                steps=attack_args['max_iter'],
                                decay=attack_args['decay_factor'],
                                alpha=alpha,
                                diversity_prob=attack_args['diversity_prob'])
                        # targeted
                        attack.set_mode_targeted_by_label()
                    elif attack_name == 'MI-FGSM':
                        alpha = attack_args['eps'] / attack_args['max_iter']
                        attack = attacks.mifgsm.MIFGSM(model, eps=attack_args['eps'],
                                steps=attack_args['max_iter'],
                                decay=attack_args['decay_factor'],
                                alpha=alpha)
                        # targeted
                        attack.set_mode_targeted_by_label()
                    elif attack_name == 'VMI-FGSM':
                        alpha = attack_args['eps'] / attack_args['max_iter']
                        attack = attacks.vmifgsm.VMIFGSM(model, eps=attack_args['eps'],
                                steps=attack_args['max_iter'],
                                decay=attack_args['decay_factor'],
                                alpha=alpha)
                        # targeted
                        attack.set_mode_targeted_by_label()
                    elif attack_name == 'VNI-FGSM':
                        alpha = attack_args['eps'] / attack_args['max_iter']
                        attack = attacks.vnifgsm.VNIFGSM(model, eps=attack_args['eps'],
                                steps=attack_args['max_iter'],
                                decay=attack_args['decay_factor'],
                                alpha=alpha)
                        # targeted
                        attack.set_mode_targeted_by_label()
                    elif attack_name == 'NI-FGSM':
                        alpha = attack_args['eps'] / attack_args['max_iter']
                        attack = attacks.nifgsm.NIFGSM(model, eps=attack_args['eps'],
                                steps=attack_args['max_iter'],
                                decay=attack_args['decay_factor'],
                                alpha=alpha)
                        # targeted
                        attack.set_mode_targeted_by_label()
                    elif attack_name == 'SINI-FGSM':
                        alpha = attack_args['eps'] / attack_args['max_iter']
                        attack = attacks.sinifgsm.SINIFGSM(model, eps=attack_args['eps'],
                                steps=attack_args['max_iter'],
                                decay=attack_args['decay_factor'],
                                alpha=alpha)
                        # targeted
                        attack.set_mode_targeted_by_label()
                    elif attack_name == 'AA':
                        alpha = attack_args['eps'] / attack_args['max_iter']
                        attack = attacks.aa.ActivationAttack(feature_model, feature_library,
                                eps=attack_args['eps'],
                                steps=attack_args['max_iter'],
                                alpha=alpha)
                        # targeted
                        attack.set_mode_targeted_by_label()
                    elif attack_name == 'TAA':
                        alpha = attack_args['eps'] / attack_args['max_iter']
                        attack = attacks.taa.TAA(feature_model, feature_library,
                                eps=attack_args['eps'],
                                steps=attack_args['max_iter'],
                                alpha=alpha)
                        # targeted
                        attack.set_mode_targeted_by_label()
                    elif attack_name == 'BSI-FGSM':
                        alpha = attack_args['eps'] / attack_args['max_iter']
                        attack = attacks.bsifgsm.BSIFGSM(model, saliency_layer,
                                saliency_map_ratio=attack_args['ratio'],
                                eps=attack_args['eps'],
                                steps=attack_args['max_iter'],
                                beta1=attack_args['beta1'],
                                beta2=attack_args['beta2'],
                                delta=attack_args['delta'],
                                alpha=alpha)
                        # targeted
                        attack.set_mode_targeted_by_label()
                    elif attack_name == 'BAP':
                        alpha = 1.0 
                        # alpha = lr 
                        attack = attacks.bap.BAP(model,
                                eps=attack_args['eps'],
                                steps=attack_args['max_iter'],
                                alpha=alpha,
                                decay=attack_args['decay_factor'],
                                n=10, mixup_num=3, mixup_weight=0.4, mixup_ratio=0.7, beta1=mixup, beta2=beta2)
                        #        n=attack_args['n'])
                        # targeted
                        attack.set_mode_targeted_by_label()
                    else:
                        # raise 'Invalid attack method!!!'
                        continue
    
                    # begin to attack
                    adv_confidences = {} 
                    for (feature, label), (fname, _) in tqdm(zip(ds, ds.imgs)):
                        feature = feature.unsqueeze(0).cuda()
                        source = torch.LongTensor([label]).cuda()
    
                        fname_basename = os.path.basename(fname)
                        (_, target) = attack_targets[fname_basename]
                        target = torch.LongTensor([target]).cuda()
                        adv_output_file = os.path.join(adv_output_dir, fname_basename)
    
                        adv_feature = attack(feature, target, source_labels=source) 
                        save_one_img(adv_feature.detach().cpu(), adv_output_file)
    
                        adv_confidence = F.softmax(model(adv_feature), dim=1)
                        adv_confidences[fname_basename] = adv_confidence.detach().cpu().numpy()
    
                    adv_output_confidence = os.path.join(adv_output_dir, 'confidence.npy')
                    with open(adv_output_confidence, 'wb') as fp:
                        pickle.dump(adv_confidences, fp)

if __name__ == '__main__':
    main()
