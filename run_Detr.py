import argparse
import os

from detr import detr


#create arguments depending on the run


name="run1"
parser1 = argparse.ArgumentParser('Set transformer detector', add_help=False)
parser1.add_argument("-f", "--file", type=str, required=False)
parser1.add_argument('--name',default=name,action="store", required=False)                    
parser1.add_argument('--lr', default=5e-5, type=float,action="store", required=False)
parser1.add_argument('--lr_backbone', default=5e-6, type=float,action="store", required=False)
parser1.add_argument('--patience', default=20, type=float,action="store", required=False)
parser1.add_argument('--cost_class', default=2, type=float,action="store", required=False)
parser1.add_argument('--cost_bbox', default=5, type=float,action="store", required=False)
parser1.add_argument('--cost_giou', default=2, type=float,action="store", required=False)

args1=parser1.parse_args()

n=detr(args1)
