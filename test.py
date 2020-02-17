import os
import sys

gpus = '0'
if sys.argv[1] == '1':
    test_dir         = '/home/ld/Documents/dataset/CUFS/test/photo'
    test_gt_dir      = '/home/ld/Documents/dataset/CUFS/test/sketch'
    result_dir       = './result/CUFS'
    #test_weight_path = './pretrain_model/cufs-epochs-026-meanshift30-G.pth'
    test_weight_path = './weight/epochs-029-G.pth'
elif sys.argv[1] == '2':
    test_dir         = '/home/ld/Documents/dataset/CUFSF/test/photo'
    test_gt_dir      = '/home/ld/Documents/dataset/CUFSF/test/sketch'
    result_dir       = './result/CUFSF'
    test_weight_path = './pretrain_model/cufsf-epochs-019-meanshift30-G.pth'
elif sys.argv[1] == '3':
    test_dir         = '/home/ld/Documents/dataset/CUHKstudent/test/photo'
    test_gt_dir      = '/home/ld/Documents/dataset/CUHKstudent/test/sketch'
    result_dir       = './result/CUHK_student'
    test_weight_path = './pretrain_model/cufs-epochs-026-meanshift30-G.pth'
elif sys.argv[1] == '4':
    test_dir         = './data/vgg_test/'
    test_gt_dir      = 'none' 
    result_dir       = './result/VGG'
    test_weight_path = './pretrain_model/vgg-epochs-003-G.pth'
elif sys.argv[1] == '5':
    test_dir         = './data/mydata/'
    test_gt_dir      = 'none'
    result_dir       = './result/myResult'
    test_weight_path = './pretrain_model/cufs-epochs-026-meanshift30-G.pth'

param            = [
        '--gpus {}'.format(gpus),
        '--test-dir {}'.format(test_dir),
        '--test-gt-dir {}'.format(test_gt_dir),
        '--result-dir {}'.format(result_dir),
        '--test-weight-path {}'.format(test_weight_path),
        ]

os.system('python face2sketch_wild.py eval {}'.format(" ".join(param)))




