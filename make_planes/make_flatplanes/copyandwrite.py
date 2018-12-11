#! /usr/bin/env python
# coding=utf-8
import os
import shutil
import time
import sys
reload(sys)
sys.setdefaultencoding('utf-8')


def copy_and_rename(fpath_input, fpath_output):
    for i in range(7481):
        #if os.path.splitext(file)[1] == ".jpg":
        oldname = fpath_input
        newname_1 = os.path.join(fpath_output,
                                 "%06d"%i + ".txt")
        #os.rename(oldname, newname)
        shutil.copyfile(oldname, newname_1)


if __name__ == '__main__':
    print('start ...')
    t1 = time.time() * 1000
    fpath_input = "/home/jackqian/Kitti/object/training/make_planes/0.txt"
    fpath_output = "/home/jackqian/Kitti/object/training/planes_flat1.73/"
    copy_and_rename(fpath_input, fpath_output)
    t2 = time.time() * 1000
    print('take time:' + str(t2 - t1) + 'ms')
    print('end.')
