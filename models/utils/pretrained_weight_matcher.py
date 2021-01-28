
special_cases = {
    "conv1_w": "C1.C1.0.weight",
    "conv1_b": "C1.C1.0.bias"
}

suffix = {
    "_beta": ".bias",
    "_running_mean": ".running_mean",
    "_running_var": ".running_var",
    "_gamma": ".weight",
    "_w": ".weight"
}

prefix = {
    "res_conv1_bn": "C1.C1.1",

    "res2_0_branch1_bn": "C2.0.shortcut.1",
    "res2_0_branch1": "C2.0.shortcut.0",

    "res2_0_branch2a_bn": "C2.0.bn1",
    "res2_0_branch2a": "C2.0.conv1",

    "res2_0_branch2b_bn": "C2.0.bn2",
    "res2_0_branch2b": "C2.0.conv2",

    "res2_0_branch2c_bn": "C2.0.bn3",
    "res2_0_branch2c": "C2.0.conv3",

    "res2_1_branch2a_bn": "C2.1.bn1",
    "res2_1_branch2a": "C2.1.conv1",

    "res2_1_branch2b_bn": "C2.1.bn2",
    "res2_1_branch2b": "C2.1.conv2",

    "res2_1_branch2c_bn": "C2.1.bn3",
    "res2_1_branch2c": "C2.1.conv3",

    "res2_2_branch2a_bn": "C2.2.bn1",
    "res2_2_branch2a": "C2.2.conv1",

    "res2_2_branch2b_bn": "C2.2.bn2",
    "res2_2_branch2b": "C2.2.conv2",

    "res2_2_branch2c_bn": "C2.2.bn3",
    "res2_2_branch2c": "C2.2.conv3",

    "res3_0_branch1_bn": "C3.0.shortcut.1",
    "res3_0_branch1": "C3.0.shortcut.0",

    "res3_0_branch2a_bn": "C3.0.bn1",
    "res3_0_branch2a": "C3.0.conv1",

    "res3_0_branch2b_bn": "C3.0.bn2",
    "res3_0_branch2b": "C3.0.conv2",

    "res3_0_branch2c_bn": "C3.0.bn3",
    "res3_0_branch2c": "C3.0.conv3",

    "res3_1_branch2a_bn": "C3.1.bn1",
    "res3_1_branch2a": "C3.1.conv1",

    "res3_1_branch2b_bn": "C3.1.bn2",
    "res3_1_branch2b": "C3.1.conv2",

    "res3_1_branch2c_bn": "C3.1.bn3",
    "res3_1_branch2c": "C3.1.conv3",

    "res3_2_branch2a_bn": "C3.2.bn1",
    "res3_2_branch2a": "C3.2.conv1",

    "res3_2_branch2b_bn": "C3.2.bn2",
    "res3_2_branch2b": "C3.2.conv2",

    "res3_2_branch2c_bn": "C3.2.bn3",
    "res3_2_branch2c": "C3.2.conv3",

    "res3_3_branch2a_bn": "C3.3.bn1",
    "res3_3_branch2a": "C3.3.conv1",

    "res3_3_branch2b_bn": "C3.3.bn2",
    "res3_3_branch2b": "C3.3.conv2",

    "res3_3_branch2c_bn": "C3.3.bn3",
    "res3_3_branch2c": "C3.3.conv3",

    "res4_0_branch1_bn": "C4.0.shortcut.1",
    "res4_0_branch1": "C4.0.shortcut.0",

    "res4_0_branch2a_bn": "C4.0.bn1",
    "res4_0_branch2a": "C4.0.conv1",

    "res4_0_branch2b_bn": "C4.0.bn2",
    "res4_0_branch2b": "C4.0.conv2",

    "res4_0_branch2c_bn": "C4.0.bn3",
    "res4_0_branch2c": "C4.0.conv3",

    "res4_1_branch2a_bn": "C4.1.bn1",
    "res4_1_branch2a": "C4.1.conv1",

    "res4_1_branch2b_bn": "C4.1.bn2",
    "res4_1_branch2b": "C4.1.conv2",

    "res4_1_branch2c_bn": "C4.1.bn3",
    "res4_1_branch2c": "C4.1.conv3",

    "res4_2_branch2a_bn": "C4.2.bn1",
    "res4_2_branch2a": "C4.2.conv1",

    "res4_2_branch2b_bn": "C4.2.bn2",
    "res4_2_branch2b": "C4.2.conv2",

    "res4_2_branch2c_bn": "C4.2.bn3",
    "res4_2_branch2c": "C4.2.conv3",

    "res4_3_branch2a_bn": "C4.3.bn1",
    "res4_3_branch2a": "C4.3.conv1",

    "res4_3_branch2b_bn": "C4.3.bn2",
    "res4_3_branch2b": "C4.3.conv2",

    "res4_3_branch2c_bn": "C4.3.bn3",
    "res4_3_branch2c": "C4.3.conv3",

    "res4_4_branch2a_bn": "C4.4.bn1",
    "res4_4_branch2a": "C4.4.conv1",

    "res4_4_branch2b_bn": "C4.4.bn2",
    "res4_4_branch2b": "C4.4.conv2",

    "res4_4_branch2c_bn": "C4.4.bn3",
    "res4_4_branch2c": "C4.4.conv3",

    "res4_5_branch2a_bn": "C4.5.bn1",
    "res4_5_branch2a": "C4.5.conv1",

    "res4_5_branch2b_bn": "C4.5.bn2",
    "res4_5_branch2b": "C4.5.conv2",

    "res4_5_branch2c_bn": "C4.5.bn3",
    "res4_5_branch2c": "C4.5.conv3",

    "res5_0_branch1_bn": "C5.0.shortcut.1",
    "res5_0_branch1": "C5.0.shortcut.0",

    "res5_0_branch2a_bn": "C5.0.bn1",
    "res5_0_branch2a": "C5.0.conv1",

    "res5_0_branch2b_bn": "C5.0.bn2",
    "res5_0_branch2b": "C5.0.conv2",

    "res5_0_branch2c_bn": "C5.0.bn3",
    "res5_0_branch2c": "C5.0.conv3",

    "res5_1_branch2a_bn": "C5.1.bn1",
    "res5_1_branch2a": "C5.1.conv1",

    "res5_1_branch2b_bn": "C5.1.bn2",
    "res5_1_branch2b": "C5.1.conv2",

    "res5_1_branch2c_bn": "C5.1.bn3",
    "res5_1_branch2c": "C5.1.conv3",

    "res5_2_branch2a_bn": "C5.2.bn1",
    "res5_2_branch2a": "C5.2.conv1",

    "res5_2_branch2b_bn": "C5.2.bn2",
    "res5_2_branch2b": "C5.2.conv2",

    "res5_2_branch2c_bn": "C5.2.bn3",
    "res5_2_branch2c": "C5.2.conv3",
}


def matcher(name):
    if name in special_cases:
        return special_cases[name]
    
    pref = []

    for key in prefix.keys():
        if name.startswith(key):
            pref.append(key)
    
    if len(pref) == 1:
        real_prefix = pref[0]

    elif len(pref) == 2:
        real_prefix = pref[0] if len(pref[0]) > len(pref[1]) else pref[1]


    suff = None   
    for key in suffix.keys():
        if name.endswith(key):
            suff = key

    if len(pref) == 0 or suff is None:
        return None
    
    return prefix[real_prefix] + suffix[suff]
