#!/bin/sh
# 示例变量列表，需替换为实际变量名
#copy fg as anal fisrt,then run this bash
common_vars="MU,OM_TMP,OM_U,OM_V,OM_S,P,PH,QVAPOR,THM,U,V,W"
ncks -A -v $common_vars ${updated_file} ${base_file}
