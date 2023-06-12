#!/bin/bash
mkdir train && mv ILSVRC2012_img_train.tar train/ && cd train

# step 2
# 解压 train压缩包并删除train压缩包
# tar -xvf ILSVRC2012_img_train.tar && rm -f ILSVRC2012_img_train.tar

# step 3
# 解压1000个类别压缩包并创建对应的子文件。
# find . -name "*.tar" | while read NAME ; do mkdir -p "${NAME%.tar}"; tar -xvf "${NAME}" -C "${NAME%.tar}"; rm -f "${NAME}"; done
