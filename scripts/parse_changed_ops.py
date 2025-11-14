#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import os
import sys
import re

NEW_OPS_PATH = [
    "math",
    "conversion"
    # 添加更多算子路径
]

class OperatorChangeInfo:
    def __init__(self, changed_operators=None, operator_file_map=None):
        self.changed_operators = [] if changed_operators is None else changed_operators
        self.operator_file_map = {} if operator_file_map is None else operator_file_map

    def print_change_info(self):
        print("=========================================================================\n")
        print("changed ops info\n")
        print("-------------------------------------------------------------------------\n")
        print("ops changed: \n%s" % "\n".join(self.changed_operators))
        print("-------------------------------------------------------------------------\n")

def extract_operator_name(file_path):
    clean_path = file_path.lstrip('/')
    path_parts = clean_path.split('/')

    if len(path_parts) >= 2 :
        domain = path_parts[0]
        operator_name = path_parts[1]

        if domain in NEW_OPS_PATH :
            return operator_name

def get_operator_change_info_from_ci(changed_file_info_from_ci):
    """
      get operator change info from ci, ci will write `git diff > /or_filelist.txt`
      :param changed_file_info_from_ci: git diff result file from ci
      :return: None or OperatorChangeInf
      """
    or_file_path = os.path.realpath(changed_file_info_from_ci)
    if not os.path.exists(or_file_path):
        print("[ERROR] change file is not exist, can not get file change info in this pull request.")
        return None
    with open(or_file_path) as or_f:
        lines = or_f.readlines()
        changed_operators = set()
        operator_file_map = {}

        for line in lines:
            line = line.strip()
            ext = os.path.splitext(line)[-1].lower()
            if ext in (".md",):
                continue

            operator_name = extract_operator_name(line)

            if operator_name:
                changed_operators.add(operator_name)
                if operator_name not in operator_file_map:
                    operator_file_map[operator_name] = []
                operator_file_map[operator_name].append(line)

    return OperatorChangeInfo(changed_operators=list(changed_operators), operator_file_map = operator_file_map)


def get_change_ops_list(changed_file_info_from_ci):
    ops_change_info = get_operator_change_info_from_ci(changed_file_info_from_ci)
    if not ops_change_info:
        print("[INFO] not found ops change info, run all c++.")
        return None

    ops_change_info.print_change_info

    return ";".join(ops_change_info.changed_operators)


if __name__ == '__main__':
    print(get_change_ops_list(sys.argv[1]))
