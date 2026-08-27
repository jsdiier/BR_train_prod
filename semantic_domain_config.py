STOKEN_SEQUENCE_CONFIG = {
    "user_click_seq": list(range(32841, 32861)), # 声明 slot=663, depend=u_clk_sess
    "user_pay_seq": list(range(32861, 32881)), # 声明 slot=664, depend=u_pay_sess
    "user_12h_click_seq": list(range(32901, 32926)), # 声明 slot=830, depend=u_12h_click_cates
    "search_long_pay_seq": list(range(33000, 33050)), # 声明 slot=1519 + 1520, depend=search_long_pay_shop_seq + pay_shop_catel3_sideinfo_seq
    "search_long_click_seq": list(range(33200, 33250)), # 声明 slot=1523 + 1524, depend=search_long_clk_shop_seq + clk_shop_catel3_sideinfo_seq
    "search_long_query_seq": list(range(33400, 33450)), # 声明 slot=1527, depend=search_long_query_catel3_seq
}
STOKEN_EVENT_SIDE_INFO = {
    "user_click_seq": "shop_id + time_gap、behavior_type=click、shop_category、price_band、position/mask",
    "user_pay_seq": "shop_id + time_gap、behavior_type=pay、shop_category、price_band、position/mask",
    "user_12h_click_seq": "category_id + time_gap、behavior_type=12h_click、position/mask",
    "search_long_pay_seq": "shop_id + category_l3、time_gap、behavior_type=search_pay、position/mask",
    "search_long_click_seq": "shop_id + category_l3、time_gap、behavior_type=search_click、position/mask",
    "search_long_query_seq": "category_l3 + time_gap、behavior_type=search_query、query_weight、position/mask",
}
STOKEN_ALIGNED_SIDE_SLOTS = {
    "search_long_pay_seq.category_l3": list(range(33050, 33100)),
    "search_long_click_seq.category_l3": list(range(33250, 33300)),
}
# 有完整事件时间：timestamp-aware merge + sequence-type；否则序列间插入 5 个可学习 [SEP]。

scene = [39, 40, 41, 85]
shop_stat_cvr = [1, 2, 3, 4, 5, 6, 7, 8]
shop_price = [9, 10, 27, 28, 31, 32]
shop_sales = [33, 34, 35, 36]
delivery = [
    15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
    25, 29, 30, 719,
]
shop_rating = [
    51, 52, 53, 54, 55, 56, 57, 58, 86, 87,
    717, 718, 762, 763, 764,
]
shop_traffic_show = [59, 62]
shop_traffic_click = [60, 63]
shop_traffic_pay = [61, 64]
shop_catalog_tags = [48, 67, 75, 76, 77, 78, 79, 88, 902, 904]
shop_profile = [
    42, 43, 44, 45, 46, 49, 50, 68, 80, 81,
    83, 84, 1044,
]
shop_hourly_show = [576, 583, 587, 594, 598, 605, 609]
shop_hourly_click = [577, 584, 588, 595, 599, 606, 610]
shop_hourly_cart = [574, 578, 585, 589, 596, 600, 607, 611]
shop_hourly_pay = [575, 579, 586, 590, 597, 601, 608, 612]
shop_hourly_cross = [
    580, 581, 582, 591, 592, 593, 602, 603, 604, 613,
    614, 615,
]
user_profile = [65, 66, 82, 761]
user_exposure_stat = [
    220, 221, 230, 231, 240, 241, 250, 251, 528, 532,
    539, 543, 550, 554, 561, 565, 853, 854, 863, 867,
]
user_click_stat = [
    222, 223, 232, 233, 242, 243, 252, 253, 529, 533,
    540, 544, 551, 555, 562, 566, 855, 856, 864, 868,
    1062, 1063, 1064, 1065, 1066, 1067, 1068, 1069, 1070, 1071,
    1072, 1073, 1074, 1075, 1076, 1077, 1078, 1079, 1080, 1081,
    1082, 1174, 1175, 1178, 1179, 1182, 1183, 1266, 1267,
    1270, 1271,
]
user_cart_stat = [530, 534, 541, 545, 552, 556, 563, 567, 865, 869]
user_pay_stat = [
    224, 225, 234, 235, 244, 245, 254, 255, 531, 535,
    542, 546, 553, 557, 564, 568, 857, 858, 866, 870,
    1090, 1091, 1092, 1093, 1094, 1095, 1096, 1097, 1098, 1099,
    1100, 1101, 1102, 1103, 1104, 1105, 1106, 1107, 1108, 1109,
    1110, 1176, 1177, 1180, 1181, 1184, 1185, 1268, 1269,
    1272, 1273,
]
user_funnel_cross = [
    228, 229, 238, 239, 248, 249, 258, 259, 536, 537,
    538, 547, 548, 549, 558, 559, 560, 569, 570, 571,
    861, 862, 871, 872, 873,
]
user_value_stat = [226, 227, 236, 237, 246, 247, 256, 257, 859, 860]
user_history_seq = [93, 94, 96, 97, 98, 100, 101, 102, 104]
user_order_history = [
    37, 874, 875, 876, 877, 878, 879, 880, 881, 882,
    883, 884, 885, 886, 887, 888, 889, 890, 891, 892,
    893, 1045,
]
user_history_prefer = [
    69, 70, 71, 72, 73, 74, 90, 91, 92, 895,
    897, 899, 900, 906, 907, 908, 910, 911,
]
recent_12h = [
    831, 832, 834, 836, 837, 840, 841, 843, 845, 846,
    847, 848,
]
user_act_d30 = [
    722, 723, 724, 725, 726, 727, 728, 729, 730, 731,
    732, 733, 734, 735, 736, 737, 738, 739, 740,
]
user_recency_app = [
    218, 219, 913, 914, 915, 916, 917, 918, 921, 922,
    923, 924, 925, 926, 927, 928, 929, 930, 931, 932,
    933, 934,
]
user_cart_recency = [919, 920]
user_order_pattern = [
    943, 944, 945, 946, 947, 948, 949, 950, 951, 952,
    953, 954, 955, 956, 957, 958, 959, 960, 961, 962,
    963, 964, 965,
]
user_brand_category_history = [
    936, 938, 940, 942, 966, 967, 968, 969, 970, 971,
    972, 973, 974, 975, 977, 979,
]
user_brand_pref = [
    1421, 1422, 1423, 1424, 1425, 1426, 1427, 1428, 1429, 1430,
    1431, 1432, 1433, 1434, 1435, 1436, 1437, 1438, 1439, 1440,
    1441, 1442, 1443, 1444, 1445, 1446, 1447, 1448, 1449, 1450,
    1451, 1452, 1453, 1454, 1455, 1456, 1457, 1458, 1459, 1460,
    1461, 1462, 1463, 1464, 1465, 1466, 1467, 1468,
]
user_hotdish_pref = [
    1469, 1470, 1471, 1472, 1473, 1474, 1475, 1476, 1477, 1478,
    1479, 1480, 1481, 1482, 1483, 1484, 1485, 1486, 1487, 1488,
    1489, 1490, 1491, 1492, 1493, 1494, 1495, 1496, 1497, 1498,
    1499, 1500, 1501, 1502, 1503, 1504, 1505, 1506, 1507, 1508,
    1509, 1510, 1511, 1512, 1513, 1514, 1515, 1516,
]
period_price_score_60d = [
    1118, 1119, 1120, 1121, 1122, 1123, 1124, 1125, 1126, 1127,
    1128, 1129, 1130, 1131, 1132, 1133, 1134, 1135, 1136, 1137,
    1138, 1139, 1140, 1141, 1142, 1143, 1144, 1145, 1146, 1147,
    1148, 1149, 1150, 1151, 1152, 1153, 1154, 1155, 1156, 1157,
    1158, 1159, 1160, 1161, 1162, 1163, 1164, 1165, 1166,
]
weekday_behavior_click = [1186, 1187]
weekday_behavior_pay = [1188, 1189]
weekday_shop_price = [
    1202, 1203, 1204, 1205, 1206, 1207, 1208, 1209, 1210, 1211,
    1212, 1213,
]
weekday_shop_click = [
    1214, 1215, 1216, 1217, 1218, 1219, 1220, 1221, 1222, 1223,
    1224, 1225, 1226, 1227, 1228, 1229, 1230, 1231, 1232, 1233,
    1234, 1235, 1236, 1237,
]
weekday_shop_pay = [
    1238, 1239, 1240, 1241, 1242, 1243, 1250, 1251, 1252, 1253,
    1254, 1255, 1256, 1257, 1258, 1259, 1260, 1261,
]
weekday_shop_score = [1262, 1263, 1264, 1265]
app_shop_price = [
    1244, 1245, 1246, 1247, 1248, 1249, 1294, 1295, 1296, 1297,
    1298, 1299,
]
app_shop_click = [
    1306, 1307, 1308, 1309, 1310, 1311, 1318, 1319, 1320, 1321,
    1322, 1323,
]
app_shop_pay = [
    1330, 1331, 1332, 1333, 1334, 1335, 1342, 1343, 1344, 1345,
    1346, 1347,
]
app_shop_score = [1354, 1355, 1356, 1357]
user_shop_pref = [11, 12, 13, 14, 38]
user_shop_show = [260, 265, 273, 278, 286, 291, 299, 304]
user_shop_click = [261, 266, 274, 279, 287, 292, 300, 305]
user_shop_cart = [262, 267, 275, 280, 288, 293, 301, 306]
user_shop_pay = [
    263, 264, 268, 269, 276, 277, 281, 282, 289, 290,
    294, 295, 302, 303, 307, 308, 318, 319, 320,
]
user_shop_funnel_cross = [
    270, 271, 272, 283, 284, 285, 296, 297, 298, 309,
    310, 311,
]
user_shop_value_match = [
    312, 313, 314, 315, 316, 317, 705, 706, 707, 708,
    709, 710, 711, 712, 1016, 1017,
]
user_shop_hourly_show = [616, 620, 627, 631, 638, 642, 649, 653]
user_shop_hourly_click = [617, 621, 628, 632, 639, 643, 650, 654]
user_shop_hourly_cart = [618, 622, 629, 633, 640, 644, 651, 655]
user_shop_hourly_pay = [619, 623, 630, 634, 641, 645, 652, 656]
user_shop_hourly_cross = [
    624, 625, 626, 635, 636, 637, 646, 647, 648, 657,
    658, 659,
]
geo_grid_user_show = [
    328, 329, 334, 335, 353, 354, 359, 360, 378, 379,
    384, 385, 403, 404, 409, 410,
]
geo_grid_user_click = [
    330, 331, 336, 337, 355, 356, 361, 362, 380, 381,
    386, 387, 405, 406, 411, 412,
]
geo_grid_user_cart = [
    332, 333, 338, 339, 357, 358, 363, 364, 382, 383,
    388, 389, 407, 408, 413, 414,
]
geo_grid_user_pay = [
    340, 341, 342, 343, 344, 345, 352, 365, 366, 367,
    368, 369, 370, 377, 390, 391, 392, 393, 394, 395,
    402, 415, 416, 417, 418, 419, 420, 427,
]
geo_grid_user_cross = [
    346, 347, 348, 349, 350, 351, 371, 372, 373, 374,
    375, 376, 396, 397, 398, 399, 400, 401, 421, 422,
    423, 424, 425, 426,
]
geo_grid_shop_show = [
    428, 429, 434, 435, 453, 454, 459, 460, 478, 479,
    484, 485, 503, 504, 509, 510,
]
geo_grid_shop_click = [
    430, 431, 436, 437, 455, 456, 461, 462, 480, 481,
    486, 487, 505, 506, 511, 512,
]
geo_grid_shop_cart = [
    432, 433, 438, 439, 457, 458, 463, 464, 482, 483,
    488, 489, 507, 508, 513, 514,
]
geo_grid_shop_pay = [
    440, 441, 442, 443, 444, 445, 452, 465, 466, 467,
    468, 469, 470, 477, 490, 491, 492, 493, 494, 495,
    502, 515, 516, 517, 518, 519, 520, 527,
]
geo_grid_shop_cross = [
    446, 447, 448, 449, 450, 451, 471, 472, 473, 474,
    475, 476, 496, 497, 498, 499, 500, 501, 521, 522,
    523, 524, 525, 526,
]
realtime_show = [769, 773]
realtime_click = [770, 774]
realtime_cart = [771, 775]
realtime_pay = [772, 776]
realtime_cross = [777, 778, 779]
ifood = [1414, 1415, 1416, 1417, 1418, 1419, 1420]
ride_channel = [713, 714, 716]
geo_code = [1517, 1518]
special_item = [159, 160, 161, 162]
free_delivery = [163, 164, 167]
buy_gift = [
    180, 181, 182, 183, 184, 185, 186, 187, 188, 189,
    190, 191, 192,
]
coupon = [
    194, 196, 198, 199, 200, 201, 202, 203, 204, 205,
    206, 207, 208, 209, 210, 211, 741, 742, 743, 744,
    745, 746, 747, 748, 749, 750, 751, 752, 753, 754,
    755, 756, 757, 758, 759, 760,
]
coupon_spend = [
    795, 796, 797, 798, 799, 800, 801, 802, 803, 804,
    805, 806, 807, 808, 809, 810, 811, 812, 813, 814,
    815, 816, 817, 818, 819, 820, 821, 822, 823, 824,
    825, 826, 827, 828,
]
candidate_match = [
    851, 852, 903, 905, 1046, 1047, 1048, 1049, 1050, 1051,
    1052, 1053, 1054, 1055, 1056, 1057, 1058, 1059, 1060, 1061,
    1196, 1197, 1198, 1199, 1200, 1201, 1288, 1289, 1290, 1291,
    1292, 1293,
]
period_shop_cross = [
    692, 693, 694, 695, 696, 697, 698, 699, 700, 702,
    703, 704,
]
weekday_shop_cross = [
    980, 981, 982, 983, 984, 985, 986, 987, 988, 989,
    990, 991, 993, 994, 995,
]
legacy_unassigned = [
    156, 158,
    1358, 1360, 1362, 1364, 1366, 1368, 1370, 1372, 1374,
    1376, 1378, 1380, 1382, 1384, 1386, 1388, 1390, 1392,
    1394, 1396, 1398, 1400, 1402, 1404, 1406, 1408, 1410,
    1413,
]
device_shop_cross = [665, 667, 669, 670, 671, 673, 677]
device_user_cross = [
    1021, 1022, 1023, 1024, 1025, 1026, 1027, 1028, 1029, 1030,
    1031, 1032, 1033, 1034, 1035, 1036, 1037,
]

ONETRANS_NS_TOKEN_ORDER = [
    "scene", "shop_stat_cvr", "shop_price", "shop_sales", "delivery",
    "shop_rating", "shop_traffic_show", "shop_traffic_click", "shop_traffic_pay", "shop_catalog_tags",
    "shop_profile", "shop_hourly_show", "shop_hourly_click", "shop_hourly_cart", "shop_hourly_pay",
    "shop_hourly_cross", "user_profile", "user_exposure_stat", "user_click_stat", "user_cart_stat",
    "user_pay_stat", "user_funnel_cross", "user_value_stat", "user_history_seq", "user_order_history",
    "user_history_prefer", "recent_12h", "user_act_d30", "user_recency_app", "user_cart_recency",
    "user_order_pattern", "user_brand_category_history", "user_brand_pref", "user_hotdish_pref", "period_price_score_60d",
    "weekday_behavior_click", "weekday_behavior_pay", "weekday_shop_price", "weekday_shop_click", "weekday_shop_pay",
    "weekday_shop_score", "app_shop_price", "app_shop_click", "app_shop_pay", "app_shop_score",
    "user_shop_pref", "user_shop_show", "user_shop_click", "user_shop_cart", "user_shop_pay",
    "user_shop_funnel_cross", "user_shop_value_match", "user_shop_hourly_show", "user_shop_hourly_click", "user_shop_hourly_cart",
    "user_shop_hourly_pay", "user_shop_hourly_cross", "geo_grid_user_show", "geo_grid_user_click", "geo_grid_user_cart",
    "geo_grid_user_pay", "geo_grid_user_cross", "geo_grid_shop_show", "geo_grid_shop_click", "geo_grid_shop_cart",
    "geo_grid_shop_pay", "geo_grid_shop_cross", "realtime_show", "realtime_click", "realtime_cart",
    "realtime_pay", "realtime_cross", "ifood", "ride_channel", "geo_code",
    "special_item", "free_delivery", "buy_gift", "coupon", "coupon_spend",
    "candidate_match", "period_shop_cross", "weekday_shop_cross", "legacy_unassigned", "device_shop_cross",
    "device_user_cross",
]

SEQUENCE_TOKEN_ORDER = [
    "user_click_seq",
    "user_pay_seq",
    "user_12h_click_seq",
    "search_long_pay_seq",
    "search_long_click_seq",
    "search_long_query_seq",
]

DOMAIN_SLOT_GROUPS = {
    name: list(globals()[name]) for name in ONETRANS_NS_TOKEN_ORDER
}


def validate_semantic_domain_schema():
    if len(ONETRANS_NS_TOKEN_ORDER) != 86:
        raise ValueError("ordinary semantic domain count must be 86")
    if len(SEQUENCE_TOKEN_ORDER) != 6:
        raise ValueError("sequence semantic domain count must be 6")
    if set(SEQUENCE_TOKEN_ORDER) != set(STOKEN_SEQUENCE_CONFIG):
        raise ValueError("sequence token order/config mismatch")
    if len(set(ONETRANS_NS_TOKEN_ORDER)) != len(ONETRANS_NS_TOKEN_ORDER):
        raise ValueError("duplicate ordinary semantic domain name")

    owner = {}
    for domain_name in ONETRANS_NS_TOKEN_ORDER:
        slots = DOMAIN_SLOT_GROUPS[domain_name]
        if not slots:
            raise ValueError("empty ordinary semantic domain: %s" % domain_name)
        for slot_id in slots:
            if slot_id in owner:
                raise ValueError(
                    "slot %d belongs to both %s and %s" %
                    (slot_id, owner[slot_id], domain_name))
            owner[slot_id] = domain_name

    if set(legacy_unassigned) != set(DOMAIN_SLOT_GROUPS['legacy_unassigned']):
        raise ValueError("legacy_unassigned coverage mismatch")
    if set(range(1530, 1550)).intersection(owner):
        raise ValueError("slots 1530..1549 are not part of the baseline slot set")
    return owner


ORDINARY_SLOT_OWNER = validate_semantic_domain_schema()
ORDINARY_SLOT_IDS = sorted(ORDINARY_SLOT_OWNER)
DOMAIN_TOKEN_COUNTS = {
    name: max(1, (len(DOMAIN_SLOT_GROUPS[name]) + 7) // 8)
    for name in ONETRANS_NS_TOKEN_ORDER
}
ORDINARY_TOKEN_COUNT = sum(DOMAIN_TOKEN_COUNTS.values())
SEMANTIC_TOKEN_COUNT = ORDINARY_TOKEN_COUNT + len(SEQUENCE_TOKEN_ORDER)

if ORDINARY_TOKEN_COUNT != 187:
    raise ValueError(
        "ordinary semantic token count must be 187, got %d" %
        ORDINARY_TOKEN_COUNT)
if SEMANTIC_TOKEN_COUNT != 193:
    raise ValueError(
        "total semantic token count must be 193, got %d" %
        SEMANTIC_TOKEN_COUNT)
