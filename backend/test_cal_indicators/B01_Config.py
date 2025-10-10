# -*- encoding: utf-8 -*-
"""
@File: B01_Config.py
@Modify Time: 2025/7/17 10:41
@Author: Kevin-Chen
@Descriptions:
"""
from A04_CalFunc import *

# 指标数据上下限
max_outliers = 0 + 99999999.99
min_outliers = 0 - 99999999.99

# YS频率 支持的区间列表
p_list = [
    'CY',  # 今年以来
    'CC',  # 成立以来
    '1W', '1M', '3M', '6M', '1Y', '2Y', '3Y', '5Y',  # 滚动区间
    'Y1', 'Y2', 'Y3', 'Y4', 'Y5', 'Y6', 'Y7', 'Y8', 'Y9', 'Y10'  # 自然区间
]
# W频率 支持的区间列表
w_p_list = [
    'CY',  # 今年以来
    'CC',  # 成立以来
    '1M', '3M', '6M', '1Y', '2Y', '3Y', '5Y',  # 滚动区间
    'Y1', 'Y2', 'Y3', 'Y4', 'Y5', 'Y6', 'Y7', 'Y8', 'Y9', 'Y10'  # 自然区间
]
# M频率 支持的区间列表
m_p_list = [
    'CY',  # 今年以来
    'CC',  # 成立以来
    '3M', '6M', '1Y', '2Y', '3Y', '5Y',  # 滚动区间
    'Y1', 'Y2', 'Y3', 'Y4', 'Y5', 'Y6', 'Y7', 'Y8', 'Y9', 'Y10'  # 自然区间
]

# YS频率 要求计算的指标字典信息
indicator_dic = {
    # ------------- 收益率相关指标 -------------
    'TotalReturn': {
        'name': '累计收益',
        'index_order': 'B',  # 排名方向: B越大排名越靠前, S越小排名越靠前, 空字符串表示不排名
        'index_unit': '%',  # 指标的单位
        'index_process': '*100',  # 指标展示前的处理方式
        'max_outliers': max_outliers,  # 异常值-极大值-不排名
        'min_outliers': min_outliers,  # 异常值-极小值-不排名
        'spec_num': None,  # 特殊值
        'spec_means': None,  # 特殊值的含义
        'spec_rank': None,  # 特殊值是否排名 (Y:参与排名, N:不参与排名)
        'spec_show': None,  # 特殊值的前端展示
        'table_name': '',
        'periods': p_list
    },
    'AnnualReturn': {
        'name': '年化收益率',
        'index_order': 'B',  # 排名方向: B越大排名越靠前, S越小排名越靠前
        'index_unit': '%',  # 指标的单位
        'index_process': '*100',  # 指标展示前的处理方式
        'max_outliers': max_outliers,  # 异常值-极大值-不排名
        'min_outliers': min_outliers,  # 异常值-极小值-不排名
        'spec_num': None,  # 特殊值
        'spec_means': None,  # 特殊值的含义
        'spec_rank': None,  # 特殊值是否排名 (Y:参与排名, N:不参与排名)
        'spec_show': None,  # 特殊值的前端展示
        'table_name': '',
        'periods': p_list
    },
    'RelaRe': {
        'name': '相对累计收益(主动收益),超过累计基准收益率的值',
        'index_order': 'B',  # 排名方向: B越大排名越靠前, S越小排名越靠前
        'index_unit': '%',  # 指标的单位
        'index_process': '*100',  # 指标展示前的处理方式
        'max_outliers': max_outliers,  # 异常值-极大值-不排名
        'min_outliers': min_outliers,  # 异常值-极小值-不排名
        'spec_num': None,  # 特殊值
        'spec_means': None,  # 特殊值的含义
        'spec_rank': None,  # 特殊值是否排名 (Y:参与排名, N:不参与排名)
        'spec_show': None,  # 特殊值的前端展示
        'table_name': '',
        'periods': p_list
    },
    'RelaReAnn': {
        'name': '相对年化收益(年化主动收益),超过年化基准收益率的值',
        'index_order': 'B',  # 排名方向: B越大排名越靠前, S越小排名越靠前
        'index_unit': '%',  # 指标的单位
        'index_process': '*100',  # 指标展示前的处理方式
        'max_outliers': max_outliers,  # 异常值-极大值-不排名
        'min_outliers': min_outliers,  # 异常值-极小值-不排名
        'spec_num': None,  # 特殊值
        'spec_means': None,  # 特殊值的含义
        'spec_rank': None,  # 特殊值是否排名 (Y:参与排名, N:不参与排名)
        'spec_show': None,  # 特殊值的前端展示
        'table_name': '',
        'periods': p_list
    },
    'TotalReturn_bm': {
        'name': '累计基准收益率',
        'index_order': 'B',  # 排名方向: B越大排名越靠前, S越小排名越靠前
        'index_unit': '%',  # 指标的单位
        'index_process': '*100',  # 指标展示前的处理方式
        'max_outliers': max_outliers,  # 异常值-极大值-不排名
        'min_outliers': min_outliers,  # 异常值-极小值-不排名
        'spec_num': None,  # 特殊值
        'spec_means': None,  # 特殊值的含义
        'spec_rank': None,  # 特殊值是否排名 (Y:参与排名, N:不参与排名)
        'spec_show': None,  # 特殊值的前端展示
        'table_name': '',
        'periods': []
    },
    'AnnualReturn_bm': {
        'name': '基准年化收益率',
        'index_order': 'B',  # 排名方向: B越大排名越靠前, S越小排名越靠前
        'index_unit': '%',  # 指标的单位
        'index_process': '*100',  # 指标展示前的处理方式
        'max_outliers': max_outliers,  # 异常值-极大值-不排名
        'min_outliers': min_outliers,  # 异常值-极小值-不排名
        'spec_num': None,  # 特殊值
        'spec_means': None,  # 特殊值的含义
        'spec_rank': None,  # 特殊值是否排名 (Y:参与排名, N:不参与排名)
        'spec_show': None,  # 特殊值的前端展示
        'table_name': '',
        'periods': []
    },
    'TotalReturn_rf': {
        'name': '无风险累计基准收益率',
        'index_order': 'B',  # 排名方向: B越大排名越靠前, S越小排名越靠前
        'index_unit': '%',  # 指标的单位
        'index_process': '*100',  # 指标展示前的处理方式
        'max_outliers': max_outliers,  # 异常值-极大值-不排名
        'min_outliers': min_outliers,  # 异常值-极小值-不排名
        'spec_num': None,  # 特殊值
        'spec_means': None,  # 特殊值的含义
        'spec_rank': None,  # 特殊值是否排名 (Y:参与排名, N:不参与排名)
        'spec_show': None,  # 特殊值的前端展示
        'table_name': '',
        'periods': []
    },
    'MeanReturn': {
        'name': '区间内的平均收益率',
        'index_order': 'B',  # 排名方向: B越大排名越靠前, S越小排名越靠前, 空字符串表示不排名
        'index_unit': '%',  # 指标的单位
        'index_process': '*100',  # 指标展示前的处理方式
        'max_outliers': max_outliers,  # 异常值-极大值-不排名
        'min_outliers': min_outliers,  # 异常值-极小值-不排名
        'spec_num': None,  # 特殊值
        'spec_means': None,  # 特殊值的含义
        'spec_rank': None,  # 特殊值是否排名 (Y:参与排名, N:不参与排名)
        'spec_show': None,  # 特殊值的前端展示
        'table_name': '',
        'periods': []
    },
    'MeanReturn_bm': {
        'name': '区间内基准的平均收益率',
        'index_order': 'B',  # 排名方向: B越大排名越靠前, S越小排名越靠前, 空字符串表示不排名
        'index_unit': '%',  # 指标的单位
        'index_process': '*100',  # 指标展示前的处理方式
        'max_outliers': max_outliers,  # 异常值-极大值-不排名
        'min_outliers': min_outliers,  # 异常值-极小值-不排名
        'spec_num': None,  # 特殊值
        'spec_means': None,  # 特殊值的含义
        'spec_rank': None,  # 特殊值是否排名 (Y:参与排名, N:不参与排名)
        'spec_show': None,  # 特殊值的前端展示
        'table_name': '',
        'periods': []
    },

    # ------------- 胜率相关指标 -------------
    'AbsoluteOdds': {
        'name': '日频绝对胜率',
        'index_order': 'B',  # 排名方向: B越大排名越靠前, S越小排名越靠前
        'index_unit': '%',  # 指标的单位
        'index_process': '*100',  # 指标展示前的处理方式
        'max_outliers': max_outliers,  # 异常值-极大值-不排名
        'min_outliers': min_outliers,  # 异常值-极小值-不排名
        'spec_num': None,  # 特殊值
        'spec_means': None,  # 特殊值的含义
        'spec_rank': None,  # 特殊值是否排名 (Y:参与排名, N:不参与排名)
        'spec_show': None,  # 特殊值的前端展示
        'table_name': '',
        'periods': p_list
    },
    # 'AbsoluteOddsWeekly': {
    #     'name': '周频绝对胜率',
    #     'index_order': 'B',  # 排名方向: B越大排名越靠前, S越小排名越靠前
    #     'index_unit': '%',  # 指标的单位
    #     'index_process': '*100',  # 指标展示前的处理方式
    #     'max_outliers': max_outliers,  # 异常值-极大值-不排名
    #     'min_outliers': min_outliers,  # 异常值-极小值-不排名
    #     'spec_num': None,  # 特殊值
    #     'spec_means': None,  # 特殊值的含义
    #     'spec_rank': None,  # 特殊值是否排名 (Y:参与排名, N:不参与排名)
    #     'spec_show': None,  # 特殊值的前端展示
    #     'table_name': '',
    #     'periods': p_list
    # },
    # 'AbsoluteOddsQuart': {
    #     'name': '季频绝对胜率',
    #     'index_order': 'B',  # 排名方向: B越大排名越靠前, S越小排名越靠前
    #     'index_unit': '%',  # 指标的单位
    #     'index_process': '*100',  # 指标展示前的处理方式
    #     'max_outliers': max_outliers,  # 异常值-极大值-不排名
    #     'min_outliers': min_outliers,  # 异常值-极小值-不排名
    #     'spec_num': None,  # 特殊值
    #     'spec_means': None,  # 特殊值的含义
    #     'spec_rank': None,  # 特殊值是否排名 (Y:参与排名, N:不参与排名)
    #     'spec_show': None,  # 特殊值的前端展示
    #     'table_name': '',
    #     'periods': p_list
    # },

    # ------------- 波动率相关指标 -------------
    'prodStd': {
        'name': '基金标准差',
        'index_order': 'S',  # 排名方向: B越大排名越靠前, S越小排名越靠前
        'index_unit': '%',  # 指标的单位
        'index_process': '*100',  # 指标展示前的处理方式
        'max_outliers': max_outliers,  # 异常值-极大值-不排名
        'min_outliers': min_outliers,  # 异常值-极小值-不排名
        'spec_num': None,  # 特殊值
        'spec_means': None,  # 特殊值的含义
        'spec_rank': None,  # 特殊值是否排名 (Y:参与排名, N:不参与排名)
        'spec_show': None,  # 特殊值的前端展示
        'table_name': '',
        'periods': p_list
    },
    'prodAnnualStd': {
        'name': '基金年化标准差',
        'index_order': 'S',  # 排名方向: B越大排名越靠前, S越小排名越靠前
        'index_unit': '%',  # 指标的单位
        'index_process': '*100',  # 指标展示前的处理方式
        'max_outliers': max_outliers,  # 异常值-极大值-不排名
        'min_outliers': min_outliers,  # 异常值-极小值-不排名
        'spec_num': None,  # 特殊值
        'spec_means': None,  # 特殊值的含义
        'spec_rank': None,  # 特殊值是否排名 (Y:参与排名, N:不参与排名)
        'spec_show': None,  # 特殊值的前端展示
        'table_name': '',
        'periods': p_list
    },
    'DownsideStd': {
        'name': '下行年化波动率',
        'index_order': 'S',  # 排名方向: B越大排名越靠前, S越小排名越靠前
        'index_unit': '%',  # 指标的单位
        'index_process': '*100',  # 指标展示前的处理方式
        'max_outliers': max_outliers,  # 异常值-极大值-不排名
        'min_outliers': min_outliers,  # 异常值-极小值-不排名
        'spec_num': None,  # 特殊值
        'spec_means': None,  # 特殊值的含义
        'spec_rank': None,  # 特殊值是否排名 (Y:参与排名, N:不参与排名)
        'spec_show': None,  # 特殊值的前端展示
        'table_name': '',
        'periods': p_list
    },
    'DownsideStd_not_ann': {
        'name': '下行波动率(非年化)',
        'index_order': 'S',  # 排名方向: B越大排名越靠前, S越小排名越靠前
        'index_unit': '%',  # 指标的单位
        'index_process': '*100',  # 指标展示前的处理方式
        'max_outliers': max_outliers,  # 异常值-极大值-不排名
        'min_outliers': min_outliers,  # 异常值-极小值-不排名
        'spec_num': None,  # 特殊值
        'spec_means': None,  # 特殊值的含义
        'spec_rank': None,  # 特殊值是否排名 (Y:参与排名, N:不参与排名)
        'spec_show': None,  # 特殊值的前端展示
        'table_name': '',
        'periods': p_list
    },
    'ActiveRisk': {
        'name': '跟踪误差/主动风险',
        'index_order': 'S',  # 排名方向: B越大排名越靠前, S越小排名越靠前
        'index_unit': '%',  # 指标的单位
        'index_process': '*100',  # 指标展示前的处理方式
        'max_outliers': max_outliers,  # 异常值-极大值-不排名
        'min_outliers': min_outliers,  # 异常值-极小值-不排名
        'spec_num': None,  # 特殊值
        'spec_means': None,  # 特殊值的含义
        'spec_rank': None,  # 特殊值是否排名 (Y:参与排名, N:不参与排名)
        'spec_show': None,  # 特殊值的前端展示
        'table_name': '',
        'periods': p_list
    },
    'AnnualActiveRisk': {
        'name': '年化跟踪误差/年化主动风险',
        'index_order': 'S',  # 排名方向: B越大排名越靠前, S越小排名越靠前
        'index_unit': '%',  # 指标的单位
        'index_process': '*100',  # 指标展示前的处理方式
        'max_outliers': max_outliers,  # 异常值-极大值-不排名
        'min_outliers': min_outliers,  # 异常值-极小值-不排名
        'spec_num': None,  # 特殊值
        'spec_means': None,  # 特殊值的含义
        'spec_rank': None,  # 特殊值是否排名 (Y:参与排名, N:不参与排名)
        'spec_show': None,  # 特殊值的前端展示
        'table_name': '',
        'periods': p_list
    },

    # ------------- 最大回撤相关指标 -------------
    'prodMddr': {  # 取绝对值,以正数保存
        'name': '最大回撤',
        'index_order': 'S',  # 排名方向: B越大排名越靠前, S越小排名越靠前
        'index_unit': '%',  # 指标的单位
        'index_process': '*100',  # 指标展示前的处理方式
        'max_outliers': max_outliers,  # 异常值-极大值-不排名
        'min_outliers': min_outliers,  # 异常值-极小值-不排名
        'spec_num': 0,  # 特殊值
        'spec_means': '无回撤',  # 特殊值的含义
        'spec_rank': 'Y',  # 特殊值是否排名 (Y:参与排名, N:不参与排名)
        'spec_show': '无回撤',  # 特殊值的前端展示
        'table_name': '',
        'periods': p_list
    },
    'MddrRecoverTime': {
        'name': '最大回撤恢复时长',
        'index_order': 'S',  # 排名方向: B越大排名越靠前, S越小排名越靠前
        'index_unit': '天',  # 指标的单位
        'index_process': '-',  # 指标展示前的处理方式
        'max_outliers': max_outliers,  # 异常值-极大值-不排名
        'min_outliers': min_outliers,  # 异常值-极小值-不排名
        'spec_num': 1000000,  # 特殊值
        'spec_means': '回撤未修复',  # 特殊值的含义
        'spec_rank': 'N',  # 特殊值是否排名 (Y:参与排名, N:不参与排名)
        'spec_show': '回撤未修复',  # 特殊值的前端展示
        'table_name': '',
        'periods': p_list
    },

    # ------------- 投资性价比相关指标 -------------
    'sharpe_not_ann': {
        'name': '夏普比率',
        'index_order': 'B',  # 排名方向: B越大排名越靠前, S越小排名越靠前
        'index_unit': '',  # 指标的单位
        'index_process': '-',  # 指标展示前的处理方式
        'max_outliers': max_outliers,  # 异常值-极大值-不排名
        'min_outliers': min_outliers,  # 异常值-极小值-不排名
        'spec_num': None,  # 特殊值
        'spec_means': None,  # 特殊值的含义
        'spec_rank': None,  # 特殊值是否排名 (Y:参与排名, N:不参与排名)
        'spec_show': None,  # 特殊值的前端展示
        'table_name': '',
        'periods': p_list
    },
    'sharpe': {
        'name': '年化夏普比率',
        'index_order': 'B',  # 排名方向: B越大排名越靠前, S越小排名越靠前
        'index_unit': '',  # 指标的单位
        'index_process': '-',  # 指标展示前的处理方式
        'max_outliers': max_outliers,  # 异常值-极大值-不排名
        'min_outliers': min_outliers,  # 异常值-极小值-不排名
        'spec_num': None,  # 特殊值
        'spec_means': None,  # 特殊值的含义
        'spec_rank': None,  # 特殊值是否排名 (Y:参与排名, N:不参与排名)
        'spec_show': None,  # 特殊值的前端展示
        'table_name': '',
        'periods': p_list
    },
    'Calmar_not_ann': {
        'name': '卡玛比率',
        'index_order': 'B',  # 排名方向: B越大排名越靠前, S越小排名越靠前
        'index_unit': '',  # 指标的单位
        'index_process': '-',  # 指标展示前的处理方式
        'max_outliers': max_outliers,  # 异常值-极大值-不排名
        'min_outliers': min_outliers,  # 异常值-极小值-不排名
        'spec_num': None,  # 特殊值
        'spec_means': None,  # 特殊值的含义
        'spec_rank': None,  # 特殊值是否排名 (Y:参与排名, N:不参与排名)
        'spec_show': None,  # 特殊值的前端展示
        'table_name': '',
        'periods': p_list
    },
    'Calmar': {
        'name': '年化卡玛比率',
        'index_order': 'B',  # 排名方向: B越大排名越靠前, S越小排名越靠前
        'index_unit': '',  # 指标的单位
        'index_process': '-',  # 指标展示前的处理方式
        'max_outliers': max_outliers,  # 异常值-极大值-不排名
        'min_outliers': min_outliers,  # 异常值-极小值-不排名
        'spec_num': None,  # 特殊值
        'spec_means': None,  # 特殊值的含义
        'spec_rank': None,  # 特殊值是否排名 (Y:参与排名, N:不参与排名)
        'spec_show': None,  # 特殊值的前端展示
        'table_name': '',
        'periods': p_list
    },
    'InfoRatio_not_ann': {
        'name': '信息比率',
        'index_order': 'B',  # 排名方向: B越大排名越靠前, S越小排名越靠前
        'index_unit': '',  # 指标的单位
        'index_process': '-',  # 指标展示前的处理方式
        'max_outliers': max_outliers,  # 异常值-极大值-不排名
        'min_outliers': min_outliers,  # 异常值-极小值-不排名
        'spec_num': None,  # 特殊值
        'spec_means': None,  # 特殊值的含义
        'spec_rank': None,  # 特殊值是否排名 (Y:参与排名, N:不参与排名)
        'spec_show': None,  # 特殊值的前端展示
        'table_name': '',
        'periods': p_list
    },
    'InfoRatio': {
        'name': '年化信息比率',
        'index_order': 'B',  # 排名方向: B越大排名越靠前, S越小排名越靠前
        'index_unit': '',  # 指标的单位
        'index_process': '-',  # 指标展示前的处理方式
        'max_outliers': max_outliers,  # 异常值-极大值-不排名
        'min_outliers': min_outliers,  # 异常值-极小值-不排名
        'spec_num': None,  # 特殊值
        'spec_means': None,  # 特殊值的含义
        'spec_rank': None,  # 特殊值是否排名 (Y:参与排名, N:不参与排名)
        'spec_show': None,  # 特殊值的前端展示
        'table_name': '',
        'periods': p_list
    },

    # ------------- 最大涨幅相关指标 -------------
    'LargestContinueRaisingRate': {
        'name': '最大涨幅幅度',
        'index_order': 'B',  # 排名方向: B越大排名越靠前, S越小排名越靠前
        'index_unit': '%',  # 指标的单位
        'index_process': '*100',  # 指标展示前的处理方式
        'max_outliers': max_outliers,  # 异常值-极大值-不排名
        'min_outliers': min_outliers,  # 异常值-极小值-不排名
        'spec_num': None,  # 特殊值
        'spec_means': None,  # 特殊值的含义
        'spec_rank': None,  # 特殊值是否排名 (Y:参与排名, N:不参与排名)
        'spec_show': None,  # 特殊值的前端展示
        'table_name': '',
        'periods': p_list
    },
    'LargestContinueRaisingStart': {
        'name': '最大涨幅开始日期',
        'index_order': '',  # 排名方向: B越大排名越靠前, S越小排名越靠前
        'index_unit': 'yyyymmdd',  # 指标的单位
        'index_process': '转日期',  # 指标展示前的处理方式
        'max_outliers': max_outliers,  # 异常值-极大值-不排名
        'min_outliers': min_outliers,  # 异常值-极小值-不排名
        'spec_num': None,  # 特殊值
        'spec_means': None,  # 特殊值的含义
        'spec_rank': None,  # 特殊值是否排名 (Y:参与排名, N:不参与排名)
        'spec_show': None,  # 特殊值的前端展示
        'table_name': '',
        'periods': p_list
    },
    'LargestContinueRaisingEnd': {
        'name': '最大涨幅结束日期',
        'index_order': '',  # 排名方向: B越大排名越靠前, S越小排名越靠前
        'index_unit': 'yyyymmdd',  # 指标的单位
        'index_process': '转日期',  # 指标展示前的处理方式
        'max_outliers': max_outliers,  # 异常值-极大值-不排名
        'min_outliers': min_outliers,  # 异常值-极小值-不排名
        'spec_num': None,  # 特殊值
        'spec_means': None,  # 特殊值的含义
        'spec_rank': None,  # 特殊值是否排名 (Y:参与排名, N:不参与排名)
        'spec_show': None,  # 特殊值的前端展示
        'table_name': '',
        'periods': p_list
    },
    'LargestContinueRaisingDays': {
        'name': '最大涨幅天数',
        'index_order': 'B',  # 排名方向: B越大排名越靠前, S越小排名越靠前
        'index_unit': '天',  # 指标的单位
        'index_process': '-',  # 指标展示前的处理方式
        'max_outliers': max_outliers,  # 异常值-极大值-不排名
        'min_outliers': min_outliers,  # 异常值-极小值-不排名
        'spec_num': None,  # 特殊值
        'spec_means': None,  # 特殊值的含义
        'spec_rank': None,  # 特殊值是否排名 (Y:参与排名, N:不参与排名)
        'spec_show': None,  # 特殊值的前端展示
        'table_name': '',
        'periods': p_list
    },

    # ------------- 最大跌幅相关指标 -------------
    'LargestContinueLossingRate': {  # 该指标未取绝对值
        'name': '最大跌幅幅度',
        'index_order': 'B',  # 排名方向: B越大排名越靠前, S越小排名越靠前
        'index_unit': '%',  # 指标的单位
        'index_process': '*100',  # 指标展示前的处理方式
        'max_outliers': max_outliers,  # 异常值-极大值-不排名
        'min_outliers': min_outliers,  # 异常值-极小值-不排名
        'spec_num': None,  # 特殊值
        'spec_means': None,  # 特殊值的含义
        'spec_rank': None,  # 特殊值是否排名 (Y:参与排名, N:不参与排名)
        'spec_show': None,  # 特殊值的前端展示
        'table_name': '',
        'periods': p_list
    },
    'LargestContinueLossingStart': {
        'name': '最大跌幅开始日期',
        'index_order': '',  # 排名方向: B越大排名越靠前, S越小排名越靠前
        'index_unit': 'yyyymmdd',  # 指标的单位
        'index_process': '转日期',  # 指标展示前的处理方式
        'max_outliers': max_outliers,  # 异常值-极大值-不排名
        'min_outliers': min_outliers,  # 异常值-极小值-不排名
        'spec_num': None,  # 特殊值
        'spec_means': None,  # 特殊值的含义
        'spec_rank': None,  # 特殊值是否排名 (Y:参与排名, N:不参与排名)
        'spec_show': None,  # 特殊值的前端展示
        'table_name': '',
        'periods': p_list
    },
    'LargestContinueLossingEnd': {
        'name': '最大跌幅结束日期',
        'index_order': '',  # 排名方向: B越大排名越靠前, S越小排名越靠前
        'index_unit': 'yyyymmdd',  # 指标的单位
        'index_process': '转日期',  # 指标展示前的处理方式
        'max_outliers': max_outliers,  # 异常值-极大值-不排名
        'min_outliers': min_outliers,  # 异常值-极小值-不排名
        'spec_num': None,  # 特殊值
        'spec_means': None,  # 特殊值的含义
        'spec_rank': None,  # 特殊值是否排名 (Y:参与排名, N:不参与排名)
        'spec_show': None,  # 特殊值的前端展示
        'table_name': '',
        'periods': p_list
    },
    'LargestContinueLossingDays': {
        'name': '最大跌幅天数',
        'index_order': 'S',  # 排名方向: B越大排名越靠前, S越小排名越靠前
        'index_unit': 'yyyymmdd',  # 指标的单位
        'index_process': '-',  # 指标展示前的处理方式
        'max_outliers': max_outliers,  # 异常值-极大值-不排名
        'min_outliers': min_outliers,  # 异常值-极小值-不排名
        'spec_num': None,  # 特殊值
        'spec_means': None,  # 特殊值的含义
        'spec_rank': None,  # 特殊值是否排名 (Y:参与排名, N:不参与排名)
        'spec_show': None,  # 特殊值的前端展示
        'table_name': '',
        'periods': p_list
    },

    # ------------- Alpha 和 Beta -------------
    'alpha_not_ann': {
        'name': '詹森alpha',
        'index_order': 'B',  # 排名方向: B越大排名越靠前, S越小排名越靠前
        'index_unit': '%',  # 指标的单位
        'index_process': '*100',  # 指标展示前的处理方式
        'max_outliers': max_outliers,  # 异常值-极大值-不排名
        'min_outliers': min_outliers,  # 异常值-极小值-不排名
        'spec_num': None,  # 特殊值
        'spec_means': None,  # 特殊值的含义
        'spec_rank': None,  # 特殊值是否排名 (Y:参与排名, N:不参与排名)
        'spec_show': None,  # 特殊值的前端展示
        'table_name': '',
        'periods': p_list
    },
    'alpha': {
        'name': '年化詹森alpha',
        'index_order': 'B',  # 排名方向: B越大排名越靠前, S越小排名越靠前
        'index_unit': '%',  # 指标的单位
        'index_process': '*100',  # 指标展示前的处理方式
        'max_outliers': max_outliers,  # 异常值-极大值-不排名
        'min_outliers': min_outliers,  # 异常值-极小值-不排名
        'spec_num': None,  # 特殊值
        'spec_means': None,  # 特殊值的含义
        'spec_rank': None,  # 特殊值是否排名 (Y:参与排名, N:不参与排名)
        'spec_show': None,  # 特殊值的前端展示
        'table_name': '',
        'periods': p_list
    },
    'beta': {
        'name': '市场风险beta',
        'index_order': 'S',  # 排名方向: B越大排名越靠前, S越小排名越靠前
        'index_unit': '',  # 指标的单位
        'index_process': '-',  # 指标展示前的处理方式
        'max_outliers': max_outliers,  # 异常值-极大值-不排名
        'min_outliers': min_outliers,  # 异常值-极小值-不排名
        'spec_num': None,  # 特殊值
        'spec_means': None,  # 特殊值的含义
        'spec_rank': None,  # 特殊值是否排名 (Y:参与排名, N:不参与排名)
        'spec_show': None,  # 特殊值的前端展示
        'table_name': '',
        'periods': p_list
    },
}
# W频率 要求计算的指标字典信息
w_indicator_dic = {
    # ------------- 收益率相关指标 -------------
    'TotalReturn': {
        'name': '累计收益',
        'index_order': 'B',  # 排名方向: B越大排名越靠前, S越小排名越靠前, 空字符串表示不排名
        'index_unit': '%',  # 指标的单位
        'index_process': '*100',  # 指标展示前的处理方式
        'max_outliers': max_outliers,  # 异常值-极大值-不排名
        'min_outliers': min_outliers,  # 异常值-极小值-不排名
        'spec_num': None,  # 特殊值
        'spec_means': None,  # 特殊值的含义
        'spec_rank': None,  # 特殊值是否排名 (Y:参与排名, N:不参与排名)
        'spec_show': None,  # 特殊值的前端展示
        'table_name': '',
        'periods': w_p_list
    },
    'AnnualReturn': {
        'name': '年化收益率',
        'index_order': 'B',  # 排名方向: B越大排名越靠前, S越小排名越靠前
        'index_unit': '%',  # 指标的单位
        'index_process': '*100',  # 指标展示前的处理方式
        'max_outliers': max_outliers,  # 异常值-极大值-不排名
        'min_outliers': min_outliers,  # 异常值-极小值-不排名
        'spec_num': None,  # 特殊值
        'spec_means': None,  # 特殊值的含义
        'spec_rank': None,  # 特殊值是否排名 (Y:参与排名, N:不参与排名)
        'spec_show': None,  # 特殊值的前端展示
        'table_name': '',
        'periods': w_p_list
    },
    'RelaRe': {
        'name': '相对累计收益(主动收益),超过累计基准收益率的值',
        'index_order': 'B',  # 排名方向: B越大排名越靠前, S越小排名越靠前
        'index_unit': '%',  # 指标的单位
        'index_process': '*100',  # 指标展示前的处理方式
        'max_outliers': max_outliers,  # 异常值-极大值-不排名
        'min_outliers': min_outliers,  # 异常值-极小值-不排名
        'spec_num': None,  # 特殊值
        'spec_means': None,  # 特殊值的含义
        'spec_rank': None,  # 特殊值是否排名 (Y:参与排名, N:不参与排名)
        'spec_show': None,  # 特殊值的前端展示
        'table_name': '',
        'periods': w_p_list
    },
    'RelaReAnn': {
        'name': '相对年化收益(年化主动收益),超过年化基准收益率的值',
        'index_order': 'B',  # 排名方向: B越大排名越靠前, S越小排名越靠前
        'index_unit': '%',  # 指标的单位
        'index_process': '*100',  # 指标展示前的处理方式
        'max_outliers': max_outliers,  # 异常值-极大值-不排名
        'min_outliers': min_outliers,  # 异常值-极小值-不排名
        'spec_num': None,  # 特殊值
        'spec_means': None,  # 特殊值的含义
        'spec_rank': None,  # 特殊值是否排名 (Y:参与排名, N:不参与排名)
        'spec_show': None,  # 特殊值的前端展示
        'table_name': '',
        'periods': w_p_list
    },
    'TotalReturn_bm': {
        'name': '累计基准收益率',
        'index_order': 'B',  # 排名方向: B越大排名越靠前, S越小排名越靠前
        'index_unit': '%',  # 指标的单位
        'index_process': '*100',  # 指标展示前的处理方式
        'max_outliers': max_outliers,  # 异常值-极大值-不排名
        'min_outliers': min_outliers,  # 异常值-极小值-不排名
        'spec_num': None,  # 特殊值
        'spec_means': None,  # 特殊值的含义
        'spec_rank': None,  # 特殊值是否排名 (Y:参与排名, N:不参与排名)
        'spec_show': None,  # 特殊值的前端展示
        'table_name': '',
        'periods': []
    },
    'AnnualReturn_bm': {
        'name': '基准年化收益率',
        'index_order': 'B',  # 排名方向: B越大排名越靠前, S越小排名越靠前
        'index_unit': '%',  # 指标的单位
        'index_process': '*100',  # 指标展示前的处理方式
        'max_outliers': max_outliers,  # 异常值-极大值-不排名
        'min_outliers': min_outliers,  # 异常值-极小值-不排名
        'spec_num': None,  # 特殊值
        'spec_means': None,  # 特殊值的含义
        'spec_rank': None,  # 特殊值是否排名 (Y:参与排名, N:不参与排名)
        'spec_show': None,  # 特殊值的前端展示
        'table_name': '',
        'periods': []
    },
    'TotalReturn_rf': {
        'name': '无风险累计基准收益率',
        'index_order': 'B',  # 排名方向: B越大排名越靠前, S越小排名越靠前
        'index_unit': '%',  # 指标的单位
        'index_process': '*100',  # 指标展示前的处理方式
        'max_outliers': max_outliers,  # 异常值-极大值-不排名
        'min_outliers': min_outliers,  # 异常值-极小值-不排名
        'spec_num': None,  # 特殊值
        'spec_means': None,  # 特殊值的含义
        'spec_rank': None,  # 特殊值是否排名 (Y:参与排名, N:不参与排名)
        'spec_show': None,  # 特殊值的前端展示
        'table_name': '',
        'periods': []
    },
    'MeanReturn': {
        'name': '区间内的平均收益率',
        'index_order': 'B',  # 排名方向: B越大排名越靠前, S越小排名越靠前, 空字符串表示不排名
        'index_unit': '%',  # 指标的单位
        'index_process': '*100',  # 指标展示前的处理方式
        'max_outliers': max_outliers,  # 异常值-极大值-不排名
        'min_outliers': min_outliers,  # 异常值-极小值-不排名
        'spec_num': None,  # 特殊值
        'spec_means': None,  # 特殊值的含义
        'spec_rank': None,  # 特殊值是否排名 (Y:参与排名, N:不参与排名)
        'spec_show': None,  # 特殊值的前端展示
        'table_name': '',
        'periods': []
    },
    'MeanReturn_bm': {
        'name': '区间内基准的平均收益率',
        'index_order': 'B',  # 排名方向: B越大排名越靠前, S越小排名越靠前, 空字符串表示不排名
        'index_unit': '%',  # 指标的单位
        'index_process': '*100',  # 指标展示前的处理方式
        'max_outliers': max_outliers,  # 异常值-极大值-不排名
        'min_outliers': min_outliers,  # 异常值-极小值-不排名
        'spec_num': None,  # 特殊值
        'spec_means': None,  # 特殊值的含义
        'spec_rank': None,  # 特殊值是否排名 (Y:参与排名, N:不参与排名)
        'spec_show': None,  # 特殊值的前端展示
        'table_name': '',
        'periods': []
    },

    # ------------- 胜率相关指标 -------------
    # 'AbsoluteOdds': {
    #     'name': '日频绝对胜率',
    #     'index_order': 'B',  # 排名方向: B越大排名越靠前, S越小排名越靠前
    #     'index_unit': '%',  # 指标的单位
    #     'index_process': '*100',  # 指标展示前的处理方式
    #     'max_outliers': max_outliers,  # 异常值-极大值-不排名
    #     'min_outliers': min_outliers,  # 异常值-极小值-不排名
    #     'spec_num': None,  # 特殊值
    #     'spec_means': None,  # 特殊值的含义
    #     'spec_rank': None,  # 特殊值是否排名 (Y:参与排名, N:不参与排名)
    #     'spec_show': None,  # 特殊值的前端展示
    #     'table_name': '',
    #     'periods': w_p_list
    # },
    'AbsoluteOddsWeekly': {
        'name': '周频绝对胜率',
        'index_order': 'B',  # 排名方向: B越大排名越靠前, S越小排名越靠前
        'index_unit': '%',  # 指标的单位
        'index_process': '*100',  # 指标展示前的处理方式
        'max_outliers': max_outliers,  # 异常值-极大值-不排名
        'min_outliers': min_outliers,  # 异常值-极小值-不排名
        'spec_num': None,  # 特殊值
        'spec_means': None,  # 特殊值的含义
        'spec_rank': None,  # 特殊值是否排名 (Y:参与排名, N:不参与排名)
        'spec_show': None,  # 特殊值的前端展示
        'table_name': '',
        'periods': w_p_list
    },
    # 'AbsoluteOddsQuart': {
    #     'name': '季频绝对胜率',
    #     'index_order': 'B',  # 排名方向: B越大排名越靠前, S越小排名越靠前
    #     'index_unit': '%',  # 指标的单位
    #     'index_process': '*100',  # 指标展示前的处理方式
    #     'max_outliers': max_outliers,  # 异常值-极大值-不排名
    #     'min_outliers': min_outliers,  # 异常值-极小值-不排名
    #     'spec_num': None,  # 特殊值
    #     'spec_means': None,  # 特殊值的含义
    #     'spec_rank': None,  # 特殊值是否排名 (Y:参与排名, N:不参与排名)
    #     'spec_show': None,  # 特殊值的前端展示
    #     'table_name': '',
    #     'periods': p_list
    # },

    # ------------- 波动率相关指标 -------------
    'prodStd': {
        'name': '基金标准差',
        'index_order': 'S',  # 排名方向: B越大排名越靠前, S越小排名越靠前
        'index_unit': '%',  # 指标的单位
        'index_process': '*100',  # 指标展示前的处理方式
        'max_outliers': max_outliers,  # 异常值-极大值-不排名
        'min_outliers': min_outliers,  # 异常值-极小值-不排名
        'spec_num': None,  # 特殊值
        'spec_means': None,  # 特殊值的含义
        'spec_rank': None,  # 特殊值是否排名 (Y:参与排名, N:不参与排名)
        'spec_show': None,  # 特殊值的前端展示
        'table_name': '',
        'periods': w_p_list
    },
    'prodAnnualStd': {
        'name': '基金年化标准差',
        'index_order': 'S',  # 排名方向: B越大排名越靠前, S越小排名越靠前
        'index_unit': '%',  # 指标的单位
        'index_process': '*100',  # 指标展示前的处理方式
        'max_outliers': max_outliers,  # 异常值-极大值-不排名
        'min_outliers': min_outliers,  # 异常值-极小值-不排名
        'spec_num': None,  # 特殊值
        'spec_means': None,  # 特殊值的含义
        'spec_rank': None,  # 特殊值是否排名 (Y:参与排名, N:不参与排名)
        'spec_show': None,  # 特殊值的前端展示
        'table_name': '',
        'periods': w_p_list
    },
    'DownsideStd': {
        'name': '下行年化波动率',
        'index_order': 'S',  # 排名方向: B越大排名越靠前, S越小排名越靠前
        'index_unit': '%',  # 指标的单位
        'index_process': '*100',  # 指标展示前的处理方式
        'max_outliers': max_outliers,  # 异常值-极大值-不排名
        'min_outliers': min_outliers,  # 异常值-极小值-不排名
        'spec_num': None,  # 特殊值
        'spec_means': None,  # 特殊值的含义
        'spec_rank': None,  # 特殊值是否排名 (Y:参与排名, N:不参与排名)
        'spec_show': None,  # 特殊值的前端展示
        'table_name': '',
        'periods': w_p_list
    },
    'DownsideStd_not_ann': {
        'name': '下行波动率(非年化)',
        'index_order': 'S',  # 排名方向: B越大排名越靠前, S越小排名越靠前
        'index_unit': '%',  # 指标的单位
        'index_process': '*100',  # 指标展示前的处理方式
        'max_outliers': max_outliers,  # 异常值-极大值-不排名
        'min_outliers': min_outliers,  # 异常值-极小值-不排名
        'spec_num': None,  # 特殊值
        'spec_means': None,  # 特殊值的含义
        'spec_rank': None,  # 特殊值是否排名 (Y:参与排名, N:不参与排名)
        'spec_show': None,  # 特殊值的前端展示
        'table_name': '',
        'periods': w_p_list
    },
    'ActiveRisk': {
        'name': '跟踪误差/主动风险',
        'index_order': 'S',  # 排名方向: B越大排名越靠前, S越小排名越靠前
        'index_unit': '%',  # 指标的单位
        'index_process': '*100',  # 指标展示前的处理方式
        'max_outliers': max_outliers,  # 异常值-极大值-不排名
        'min_outliers': min_outliers,  # 异常值-极小值-不排名
        'spec_num': None,  # 特殊值
        'spec_means': None,  # 特殊值的含义
        'spec_rank': None,  # 特殊值是否排名 (Y:参与排名, N:不参与排名)
        'spec_show': None,  # 特殊值的前端展示
        'table_name': '',
        'periods': w_p_list
    },
    'AnnualActiveRisk': {
        'name': '年化跟踪误差/年化主动风险',
        'index_order': 'S',  # 排名方向: B越大排名越靠前, S越小排名越靠前
        'index_unit': '%',  # 指标的单位
        'index_process': '*100',  # 指标展示前的处理方式
        'max_outliers': max_outliers,  # 异常值-极大值-不排名
        'min_outliers': min_outliers,  # 异常值-极小值-不排名
        'spec_num': None,  # 特殊值
        'spec_means': None,  # 特殊值的含义
        'spec_rank': None,  # 特殊值是否排名 (Y:参与排名, N:不参与排名)
        'spec_show': None,  # 特殊值的前端展示
        'table_name': '',
        'periods': w_p_list
    },

    # ------------- 最大回撤相关指标 -------------
    'prodMddr': {  # 取绝对值,以正数保存
        'name': '最大回撤',
        'index_order': 'S',  # 排名方向: B越大排名越靠前, S越小排名越靠前
        'index_unit': '%',  # 指标的单位
        'index_process': '*100',  # 指标展示前的处理方式
        'max_outliers': max_outliers,  # 异常值-极大值-不排名
        'min_outliers': min_outliers,  # 异常值-极小值-不排名
        'spec_num': 0,  # 特殊值
        'spec_means': '无回撤',  # 特殊值的含义
        'spec_rank': 'Y',  # 特殊值是否排名 (Y:参与排名, N:不参与排名)
        'spec_show': '无回撤',  # 特殊值的前端展示
        'table_name': '',
        'periods': w_p_list
    },
    'MddrRecoverTime': {
        'name': '最大回撤恢复时长',
        'index_order': 'S',  # 排名方向: B越大排名越靠前, S越小排名越靠前
        'index_unit': '天',  # 指标的单位
        'index_process': '-',  # 指标展示前的处理方式
        'max_outliers': max_outliers,  # 异常值-极大值-不排名
        'min_outliers': min_outliers,  # 异常值-极小值-不排名
        'spec_num': 1000000,  # 特殊值
        'spec_means': '回撤未修复',  # 特殊值的含义
        'spec_rank': 'N',  # 特殊值是否排名 (Y:参与排名, N:不参与排名)
        'spec_show': '回撤未修复',  # 特殊值的前端展示
        'table_name': '',
        'periods': w_p_list
    },

    # ------------- 投资性价比相关指标 -------------
    'sharpe_not_ann': {
        'name': '夏普比率',
        'index_order': 'B',  # 排名方向: B越大排名越靠前, S越小排名越靠前
        'index_unit': '',  # 指标的单位
        'index_process': '-',  # 指标展示前的处理方式
        'max_outliers': max_outliers,  # 异常值-极大值-不排名
        'min_outliers': min_outliers,  # 异常值-极小值-不排名
        'spec_num': None,  # 特殊值
        'spec_means': None,  # 特殊值的含义
        'spec_rank': None,  # 特殊值是否排名 (Y:参与排名, N:不参与排名)
        'spec_show': None,  # 特殊值的前端展示
        'table_name': '',
        'periods': w_p_list
    },
    'sharpe': {
        'name': '年化夏普比率',
        'index_order': 'B',  # 排名方向: B越大排名越靠前, S越小排名越靠前
        'index_unit': '',  # 指标的单位
        'index_process': '-',  # 指标展示前的处理方式
        'max_outliers': max_outliers,  # 异常值-极大值-不排名
        'min_outliers': min_outliers,  # 异常值-极小值-不排名
        'spec_num': None,  # 特殊值
        'spec_means': None,  # 特殊值的含义
        'spec_rank': None,  # 特殊值是否排名 (Y:参与排名, N:不参与排名)
        'spec_show': None,  # 特殊值的前端展示
        'table_name': '',
        'periods': w_p_list
    },
    'Calmar_not_ann': {
        'name': '卡玛比率',
        'index_order': 'B',  # 排名方向: B越大排名越靠前, S越小排名越靠前
        'index_unit': '',  # 指标的单位
        'index_process': '-',  # 指标展示前的处理方式
        'max_outliers': max_outliers,  # 异常值-极大值-不排名
        'min_outliers': min_outliers,  # 异常值-极小值-不排名
        'spec_num': None,  # 特殊值
        'spec_means': None,  # 特殊值的含义
        'spec_rank': None,  # 特殊值是否排名 (Y:参与排名, N:不参与排名)
        'spec_show': None,  # 特殊值的前端展示
        'table_name': '',
        'periods': w_p_list
    },
    'Calmar': {
        'name': '年化卡玛比率',
        'index_order': 'B',  # 排名方向: B越大排名越靠前, S越小排名越靠前
        'index_unit': '',  # 指标的单位
        'index_process': '-',  # 指标展示前的处理方式
        'max_outliers': max_outliers,  # 异常值-极大值-不排名
        'min_outliers': min_outliers,  # 异常值-极小值-不排名
        'spec_num': None,  # 特殊值
        'spec_means': None,  # 特殊值的含义
        'spec_rank': None,  # 特殊值是否排名 (Y:参与排名, N:不参与排名)
        'spec_show': None,  # 特殊值的前端展示
        'table_name': '',
        'periods': w_p_list
    },
    'InfoRatio_not_ann': {
        'name': '信息比率',
        'index_order': 'B',  # 排名方向: B越大排名越靠前, S越小排名越靠前
        'index_unit': '',  # 指标的单位
        'index_process': '-',  # 指标展示前的处理方式
        'max_outliers': max_outliers,  # 异常值-极大值-不排名
        'min_outliers': min_outliers,  # 异常值-极小值-不排名
        'spec_num': None,  # 特殊值
        'spec_means': None,  # 特殊值的含义
        'spec_rank': None,  # 特殊值是否排名 (Y:参与排名, N:不参与排名)
        'spec_show': None,  # 特殊值的前端展示
        'table_name': '',
        'periods': w_p_list
    },
    'InfoRatio': {
        'name': '年化信息比率',
        'index_order': 'B',  # 排名方向: B越大排名越靠前, S越小排名越靠前
        'index_unit': '',  # 指标的单位
        'index_process': '-',  # 指标展示前的处理方式
        'max_outliers': max_outliers,  # 异常值-极大值-不排名
        'min_outliers': min_outliers,  # 异常值-极小值-不排名
        'spec_num': None,  # 特殊值
        'spec_means': None,  # 特殊值的含义
        'spec_rank': None,  # 特殊值是否排名 (Y:参与排名, N:不参与排名)
        'spec_show': None,  # 特殊值的前端展示
        'table_name': '',
        'periods': w_p_list
    },

    # ------------- 最大涨幅相关指标 -------------
    'LargestContinueRaisingRate': {
        'name': '最大涨幅幅度',
        'index_order': 'B',  # 排名方向: B越大排名越靠前, S越小排名越靠前
        'index_unit': '%',  # 指标的单位
        'index_process': '*100',  # 指标展示前的处理方式
        'max_outliers': max_outliers,  # 异常值-极大值-不排名
        'min_outliers': min_outliers,  # 异常值-极小值-不排名
        'spec_num': None,  # 特殊值
        'spec_means': None,  # 特殊值的含义
        'spec_rank': None,  # 特殊值是否排名 (Y:参与排名, N:不参与排名)
        'spec_show': None,  # 特殊值的前端展示
        'table_name': '',
        'periods': w_p_list
    },
    'LargestContinueRaisingStart': {
        'name': '最大涨幅开始日期',
        'index_order': '',  # 排名方向: B越大排名越靠前, S越小排名越靠前
        'index_unit': 'yyyymmdd',  # 指标的单位
        'index_process': '转日期',  # 指标展示前的处理方式
        'max_outliers': max_outliers,  # 异常值-极大值-不排名
        'min_outliers': min_outliers,  # 异常值-极小值-不排名
        'spec_num': None,  # 特殊值
        'spec_means': None,  # 特殊值的含义
        'spec_rank': None,  # 特殊值是否排名 (Y:参与排名, N:不参与排名)
        'spec_show': None,  # 特殊值的前端展示
        'table_name': '',
        'periods': w_p_list
    },
    'LargestContinueRaisingEnd': {
        'name': '最大涨幅结束日期',
        'index_order': '',  # 排名方向: B越大排名越靠前, S越小排名越靠前
        'index_unit': 'yyyymmdd',  # 指标的单位
        'index_process': '转日期',  # 指标展示前的处理方式
        'max_outliers': max_outliers,  # 异常值-极大值-不排名
        'min_outliers': min_outliers,  # 异常值-极小值-不排名
        'spec_num': None,  # 特殊值
        'spec_means': None,  # 特殊值的含义
        'spec_rank': None,  # 特殊值是否排名 (Y:参与排名, N:不参与排名)
        'spec_show': None,  # 特殊值的前端展示
        'table_name': '',
        'periods': w_p_list
    },
    'LargestContinueRaisingDays': {
        'name': '最大涨幅天数',
        'index_order': 'B',  # 排名方向: B越大排名越靠前, S越小排名越靠前
        'index_unit': '天',  # 指标的单位
        'index_process': '-',  # 指标展示前的处理方式
        'max_outliers': max_outliers,  # 异常值-极大值-不排名
        'min_outliers': min_outliers,  # 异常值-极小值-不排名
        'spec_num': None,  # 特殊值
        'spec_means': None,  # 特殊值的含义
        'spec_rank': None,  # 特殊值是否排名 (Y:参与排名, N:不参与排名)
        'spec_show': None,  # 特殊值的前端展示
        'table_name': '',
        'periods': w_p_list
    },

    # ------------- 最大跌幅相关指标 -------------
    'LargestContinueLossingRate': {  # 该指标未取绝对值
        'name': '最大跌幅幅度',
        'index_order': 'B',  # 排名方向: B越大排名越靠前, S越小排名越靠前
        'index_unit': '%',  # 指标的单位
        'index_process': '*100',  # 指标展示前的处理方式
        'max_outliers': max_outliers,  # 异常值-极大值-不排名
        'min_outliers': min_outliers,  # 异常值-极小值-不排名
        'spec_num': None,  # 特殊值
        'spec_means': None,  # 特殊值的含义
        'spec_rank': None,  # 特殊值是否排名 (Y:参与排名, N:不参与排名)
        'spec_show': None,  # 特殊值的前端展示
        'table_name': '',
        'periods': w_p_list
    },
    'LargestContinueLossingStart': {
        'name': '最大跌幅开始日期',
        'index_order': '',  # 排名方向: B越大排名越靠前, S越小排名越靠前
        'index_unit': 'yyyymmdd',  # 指标的单位
        'index_process': '转日期',  # 指标展示前的处理方式
        'max_outliers': max_outliers,  # 异常值-极大值-不排名
        'min_outliers': min_outliers,  # 异常值-极小值-不排名
        'spec_num': None,  # 特殊值
        'spec_means': None,  # 特殊值的含义
        'spec_rank': None,  # 特殊值是否排名 (Y:参与排名, N:不参与排名)
        'spec_show': None,  # 特殊值的前端展示
        'table_name': '',
        'periods': w_p_list
    },
    'LargestContinueLossingEnd': {
        'name': '最大跌幅结束日期',
        'index_order': '',  # 排名方向: B越大排名越靠前, S越小排名越靠前
        'index_unit': 'yyyymmdd',  # 指标的单位
        'index_process': '转日期',  # 指标展示前的处理方式
        'max_outliers': max_outliers,  # 异常值-极大值-不排名
        'min_outliers': min_outliers,  # 异常值-极小值-不排名
        'spec_num': None,  # 特殊值
        'spec_means': None,  # 特殊值的含义
        'spec_rank': None,  # 特殊值是否排名 (Y:参与排名, N:不参与排名)
        'spec_show': None,  # 特殊值的前端展示
        'table_name': '',
        'periods': w_p_list
    },
    'LargestContinueLossingDays': {
        'name': '最大跌幅天数',
        'index_order': 'S',  # 排名方向: B越大排名越靠前, S越小排名越靠前
        'index_unit': 'yyyymmdd',  # 指标的单位
        'index_process': '-',  # 指标展示前的处理方式
        'max_outliers': max_outliers,  # 异常值-极大值-不排名
        'min_outliers': min_outliers,  # 异常值-极小值-不排名
        'spec_num': None,  # 特殊值
        'spec_means': None,  # 特殊值的含义
        'spec_rank': None,  # 特殊值是否排名 (Y:参与排名, N:不参与排名)
        'spec_show': None,  # 特殊值的前端展示
        'table_name': '',
        'periods': w_p_list
    },

    # ------------- Alpha 和 Beta -------------
    'alpha_not_ann': {
        'name': '詹森alpha',
        'index_order': 'B',  # 排名方向: B越大排名越靠前, S越小排名越靠前
        'index_unit': '%',  # 指标的单位
        'index_process': '*100',  # 指标展示前的处理方式
        'max_outliers': max_outliers,  # 异常值-极大值-不排名
        'min_outliers': min_outliers,  # 异常值-极小值-不排名
        'spec_num': None,  # 特殊值
        'spec_means': None,  # 特殊值的含义
        'spec_rank': None,  # 特殊值是否排名 (Y:参与排名, N:不参与排名)
        'spec_show': None,  # 特殊值的前端展示
        'table_name': '',
        'periods': w_p_list
    },
    'alpha': {
        'name': '年化詹森alpha',
        'index_order': 'B',  # 排名方向: B越大排名越靠前, S越小排名越靠前
        'index_unit': '%',  # 指标的单位
        'index_process': '*100',  # 指标展示前的处理方式
        'max_outliers': max_outliers,  # 异常值-极大值-不排名
        'min_outliers': min_outliers,  # 异常值-极小值-不排名
        'spec_num': None,  # 特殊值
        'spec_means': None,  # 特殊值的含义
        'spec_rank': None,  # 特殊值是否排名 (Y:参与排名, N:不参与排名)
        'spec_show': None,  # 特殊值的前端展示
        'table_name': '',
        'periods': w_p_list
    },
    'beta': {
        'name': '市场风险beta',
        'index_order': 'S',  # 排名方向: B越大排名越靠前, S越小排名越靠前
        'index_unit': '',  # 指标的单位
        'index_process': '-',  # 指标展示前的处理方式
        'max_outliers': max_outliers,  # 异常值-极大值-不排名
        'min_outliers': min_outliers,  # 异常值-极小值-不排名
        'spec_num': None,  # 特殊值
        'spec_means': None,  # 特殊值的含义
        'spec_rank': None,  # 特殊值是否排名 (Y:参与排名, N:不参与排名)
        'spec_show': None,  # 特殊值的前端展示
        'table_name': '',
        'periods': w_p_list
    },
}
# M频率 要求计算的指标字典信息
m_indicator_dic = {
    # ------------- 收益率相关指标 -------------
    'TotalReturn': {
        'name': '累计收益',
        'index_order': 'B',  # 排名方向: B越大排名越靠前, S越小排名越靠前, 空字符串表示不排名
        'index_unit': '%',  # 指标的单位
        'index_process': '*100',  # 指标展示前的处理方式
        'max_outliers': max_outliers,  # 异常值-极大值-不排名
        'min_outliers': min_outliers,  # 异常值-极小值-不排名
        'spec_num': None,  # 特殊值
        'spec_means': None,  # 特殊值的含义
        'spec_rank': None,  # 特殊值是否排名 (Y:参与排名, N:不参与排名)
        'spec_show': None,  # 特殊值的前端展示
        'table_name': '',
        'periods': m_p_list
    },
    'AnnualReturn': {
        'name': '年化收益率',
        'index_order': 'B',  # 排名方向: B越大排名越靠前, S越小排名越靠前
        'index_unit': '%',  # 指标的单位
        'index_process': '*100',  # 指标展示前的处理方式
        'max_outliers': max_outliers,  # 异常值-极大值-不排名
        'min_outliers': min_outliers,  # 异常值-极小值-不排名
        'spec_num': None,  # 特殊值
        'spec_means': None,  # 特殊值的含义
        'spec_rank': None,  # 特殊值是否排名 (Y:参与排名, N:不参与排名)
        'spec_show': None,  # 特殊值的前端展示
        'table_name': '',
        'periods': m_p_list
    },
    'RelaRe': {
        'name': '相对累计收益(主动收益),超过累计基准收益率的值',
        'index_order': 'B',  # 排名方向: B越大排名越靠前, S越小排名越靠前
        'index_unit': '%',  # 指标的单位
        'index_process': '*100',  # 指标展示前的处理方式
        'max_outliers': max_outliers,  # 异常值-极大值-不排名
        'min_outliers': min_outliers,  # 异常值-极小值-不排名
        'spec_num': None,  # 特殊值
        'spec_means': None,  # 特殊值的含义
        'spec_rank': None,  # 特殊值是否排名 (Y:参与排名, N:不参与排名)
        'spec_show': None,  # 特殊值的前端展示
        'table_name': '',
        'periods': m_p_list
    },
    'RelaReAnn': {
        'name': '相对年化收益(年化主动收益),超过年化基准收益率的值',
        'index_order': 'B',  # 排名方向: B越大排名越靠前, S越小排名越靠前
        'index_unit': '%',  # 指标的单位
        'index_process': '*100',  # 指标展示前的处理方式
        'max_outliers': max_outliers,  # 异常值-极大值-不排名
        'min_outliers': min_outliers,  # 异常值-极小值-不排名
        'spec_num': None,  # 特殊值
        'spec_means': None,  # 特殊值的含义
        'spec_rank': None,  # 特殊值是否排名 (Y:参与排名, N:不参与排名)
        'spec_show': None,  # 特殊值的前端展示
        'table_name': '',
        'periods': m_p_list
    },
    'TotalReturn_bm': {
        'name': '累计基准收益率',
        'index_order': 'B',  # 排名方向: B越大排名越靠前, S越小排名越靠前
        'index_unit': '%',  # 指标的单位
        'index_process': '*100',  # 指标展示前的处理方式
        'max_outliers': max_outliers,  # 异常值-极大值-不排名
        'min_outliers': min_outliers,  # 异常值-极小值-不排名
        'spec_num': None,  # 特殊值
        'spec_means': None,  # 特殊值的含义
        'spec_rank': None,  # 特殊值是否排名 (Y:参与排名, N:不参与排名)
        'spec_show': None,  # 特殊值的前端展示
        'table_name': '',
        'periods': []
    },
    'AnnualReturn_bm': {
        'name': '基准年化收益率',
        'index_order': 'B',  # 排名方向: B越大排名越靠前, S越小排名越靠前
        'index_unit': '%',  # 指标的单位
        'index_process': '*100',  # 指标展示前的处理方式
        'max_outliers': max_outliers,  # 异常值-极大值-不排名
        'min_outliers': min_outliers,  # 异常值-极小值-不排名
        'spec_num': None,  # 特殊值
        'spec_means': None,  # 特殊值的含义
        'spec_rank': None,  # 特殊值是否排名 (Y:参与排名, N:不参与排名)
        'spec_show': None,  # 特殊值的前端展示
        'table_name': '',
        'periods': []
    },
    'TotalReturn_rf': {
        'name': '无风险累计基准收益率',
        'index_order': 'B',  # 排名方向: B越大排名越靠前, S越小排名越靠前
        'index_unit': '%',  # 指标的单位
        'index_process': '*100',  # 指标展示前的处理方式
        'max_outliers': max_outliers,  # 异常值-极大值-不排名
        'min_outliers': min_outliers,  # 异常值-极小值-不排名
        'spec_num': None,  # 特殊值
        'spec_means': None,  # 特殊值的含义
        'spec_rank': None,  # 特殊值是否排名 (Y:参与排名, N:不参与排名)
        'spec_show': None,  # 特殊值的前端展示
        'table_name': '',
        'periods': []
    },
    'MeanReturn': {
        'name': '区间内的平均收益率',
        'index_order': 'B',  # 排名方向: B越大排名越靠前, S越小排名越靠前, 空字符串表示不排名
        'index_unit': '%',  # 指标的单位
        'index_process': '*100',  # 指标展示前的处理方式
        'max_outliers': max_outliers,  # 异常值-极大值-不排名
        'min_outliers': min_outliers,  # 异常值-极小值-不排名
        'spec_num': None,  # 特殊值
        'spec_means': None,  # 特殊值的含义
        'spec_rank': None,  # 特殊值是否排名 (Y:参与排名, N:不参与排名)
        'spec_show': None,  # 特殊值的前端展示
        'table_name': '',
        'periods': []
    },
    'MeanReturn_bm': {
        'name': '区间内基准的平均收益率',
        'index_order': 'B',  # 排名方向: B越大排名越靠前, S越小排名越靠前, 空字符串表示不排名
        'index_unit': '%',  # 指标的单位
        'index_process': '*100',  # 指标展示前的处理方式
        'max_outliers': max_outliers,  # 异常值-极大值-不排名
        'min_outliers': min_outliers,  # 异常值-极小值-不排名
        'spec_num': None,  # 特殊值
        'spec_means': None,  # 特殊值的含义
        'spec_rank': None,  # 特殊值是否排名 (Y:参与排名, N:不参与排名)
        'spec_show': None,  # 特殊值的前端展示
        'table_name': '',
        'periods': []
    },

    # # ------------- 胜率相关指标 -------------
    # 'AbsoluteOdds': {
    #     'name': '日频绝对胜率',
    #     'index_order': 'B',  # 排名方向: B越大排名越靠前, S越小排名越靠前
    #     'index_unit': '%',  # 指标的单位
    #     'index_process': '*100',  # 指标展示前的处理方式
    #     'max_outliers': max_outliers,  # 异常值-极大值-不排名
    #     'min_outliers': min_outliers,  # 异常值-极小值-不排名
    #     'spec_num': None,  # 特殊值
    #     'spec_means': None,  # 特殊值的含义
    #     'spec_rank': None,  # 特殊值是否排名 (Y:参与排名, N:不参与排名)
    #     'spec_show': None,  # 特殊值的前端展示
    #     'table_name': '',
    #     'periods': p_list
    # },
    # 'AbsoluteOddsWeekly': {
    #     'name': '周频绝对胜率',
    #     'index_order': 'B',  # 排名方向: B越大排名越靠前, S越小排名越靠前
    #     'index_unit': '%',  # 指标的单位
    #     'index_process': '*100',  # 指标展示前的处理方式
    #     'max_outliers': max_outliers,  # 异常值-极大值-不排名
    #     'min_outliers': min_outliers,  # 异常值-极小值-不排名
    #     'spec_num': None,  # 特殊值
    #     'spec_means': None,  # 特殊值的含义
    #     'spec_rank': None,  # 特殊值是否排名 (Y:参与排名, N:不参与排名)
    #     'spec_show': None,  # 特殊值的前端展示
    #     'table_name': '',
    #     'periods': p_list
    # },
    # 'AbsoluteOddsQuart': {
    #     'name': '季频绝对胜率',
    #     'index_order': 'B',  # 排名方向: B越大排名越靠前, S越小排名越靠前
    #     'index_unit': '%',  # 指标的单位
    #     'index_process': '*100',  # 指标展示前的处理方式
    #     'max_outliers': max_outliers,  # 异常值-极大值-不排名
    #     'min_outliers': min_outliers,  # 异常值-极小值-不排名
    #     'spec_num': None,  # 特殊值
    #     'spec_means': None,  # 特殊值的含义
    #     'spec_rank': None,  # 特殊值是否排名 (Y:参与排名, N:不参与排名)
    #     'spec_show': None,  # 特殊值的前端展示
    #     'table_name': '',
    #     'periods': p_list
    # },

    # ------------- 波动率相关指标 -------------
    'prodStd': {
        'name': '基金标准差',
        'index_order': 'S',  # 排名方向: B越大排名越靠前, S越小排名越靠前
        'index_unit': '%',  # 指标的单位
        'index_process': '*100',  # 指标展示前的处理方式
        'max_outliers': max_outliers,  # 异常值-极大值-不排名
        'min_outliers': min_outliers,  # 异常值-极小值-不排名
        'spec_num': None,  # 特殊值
        'spec_means': None,  # 特殊值的含义
        'spec_rank': None,  # 特殊值是否排名 (Y:参与排名, N:不参与排名)
        'spec_show': None,  # 特殊值的前端展示
        'table_name': '',
        'periods': m_p_list
    },
    'prodAnnualStd': {
        'name': '基金年化标准差',
        'index_order': 'S',  # 排名方向: B越大排名越靠前, S越小排名越靠前
        'index_unit': '%',  # 指标的单位
        'index_process': '*100',  # 指标展示前的处理方式
        'max_outliers': max_outliers,  # 异常值-极大值-不排名
        'min_outliers': min_outliers,  # 异常值-极小值-不排名
        'spec_num': None,  # 特殊值
        'spec_means': None,  # 特殊值的含义
        'spec_rank': None,  # 特殊值是否排名 (Y:参与排名, N:不参与排名)
        'spec_show': None,  # 特殊值的前端展示
        'table_name': '',
        'periods': m_p_list
    },
    'DownsideStd': {
        'name': '下行年化波动率',
        'index_order': 'S',  # 排名方向: B越大排名越靠前, S越小排名越靠前
        'index_unit': '%',  # 指标的单位
        'index_process': '*100',  # 指标展示前的处理方式
        'max_outliers': max_outliers,  # 异常值-极大值-不排名
        'min_outliers': min_outliers,  # 异常值-极小值-不排名
        'spec_num': None,  # 特殊值
        'spec_means': None,  # 特殊值的含义
        'spec_rank': None,  # 特殊值是否排名 (Y:参与排名, N:不参与排名)
        'spec_show': None,  # 特殊值的前端展示
        'table_name': '',
        'periods': m_p_list
    },
    'DownsideStd_not_ann': {
        'name': '下行波动率(非年化)',
        'index_order': 'S',  # 排名方向: B越大排名越靠前, S越小排名越靠前
        'index_unit': '%',  # 指标的单位
        'index_process': '*100',  # 指标展示前的处理方式
        'max_outliers': max_outliers,  # 异常值-极大值-不排名
        'min_outliers': min_outliers,  # 异常值-极小值-不排名
        'spec_num': None,  # 特殊值
        'spec_means': None,  # 特殊值的含义
        'spec_rank': None,  # 特殊值是否排名 (Y:参与排名, N:不参与排名)
        'spec_show': None,  # 特殊值的前端展示
        'table_name': '',
        'periods': m_p_list
    },
    'ActiveRisk': {
        'name': '跟踪误差/主动风险',
        'index_order': 'S',  # 排名方向: B越大排名越靠前, S越小排名越靠前
        'index_unit': '%',  # 指标的单位
        'index_process': '*100',  # 指标展示前的处理方式
        'max_outliers': max_outliers,  # 异常值-极大值-不排名
        'min_outliers': min_outliers,  # 异常值-极小值-不排名
        'spec_num': None,  # 特殊值
        'spec_means': None,  # 特殊值的含义
        'spec_rank': None,  # 特殊值是否排名 (Y:参与排名, N:不参与排名)
        'spec_show': None,  # 特殊值的前端展示
        'table_name': '',
        'periods': m_p_list
    },
    'AnnualActiveRisk': {
        'name': '年化跟踪误差/年化主动风险',
        'index_order': 'S',  # 排名方向: B越大排名越靠前, S越小排名越靠前
        'index_unit': '%',  # 指标的单位
        'index_process': '*100',  # 指标展示前的处理方式
        'max_outliers': max_outliers,  # 异常值-极大值-不排名
        'min_outliers': min_outliers,  # 异常值-极小值-不排名
        'spec_num': None,  # 特殊值
        'spec_means': None,  # 特殊值的含义
        'spec_rank': None,  # 特殊值是否排名 (Y:参与排名, N:不参与排名)
        'spec_show': None,  # 特殊值的前端展示
        'table_name': '',
        'periods': m_p_list
    },

    # ------------- 最大回撤相关指标 -------------
    'prodMddr': {  # 取绝对值,以正数保存
        'name': '最大回撤',
        'index_order': 'S',  # 排名方向: B越大排名越靠前, S越小排名越靠前
        'index_unit': '%',  # 指标的单位
        'index_process': '*100',  # 指标展示前的处理方式
        'max_outliers': max_outliers,  # 异常值-极大值-不排名
        'min_outliers': min_outliers,  # 异常值-极小值-不排名
        'spec_num': 0,  # 特殊值
        'spec_means': '无回撤',  # 特殊值的含义
        'spec_rank': 'Y',  # 特殊值是否排名 (Y:参与排名, N:不参与排名)
        'spec_show': '无回撤',  # 特殊值的前端展示
        'table_name': '',
        'periods': m_p_list
    },
    'MddrRecoverTime': {
        'name': '最大回撤恢复时长',
        'index_order': 'S',  # 排名方向: B越大排名越靠前, S越小排名越靠前
        'index_unit': '天',  # 指标的单位
        'index_process': '-',  # 指标展示前的处理方式
        'max_outliers': max_outliers,  # 异常值-极大值-不排名
        'min_outliers': min_outliers,  # 异常值-极小值-不排名
        'spec_num': 1000000,  # 特殊值
        'spec_means': '回撤未修复',  # 特殊值的含义
        'spec_rank': 'N',  # 特殊值是否排名 (Y:参与排名, N:不参与排名)
        'spec_show': '回撤未修复',  # 特殊值的前端展示
        'table_name': '',
        'periods': m_p_list
    },

    # ------------- 投资性价比相关指标 -------------
    'sharpe_not_ann': {
        'name': '夏普比率',
        'index_order': 'B',  # 排名方向: B越大排名越靠前, S越小排名越靠前
        'index_unit': '',  # 指标的单位
        'index_process': '-',  # 指标展示前的处理方式
        'max_outliers': max_outliers,  # 异常值-极大值-不排名
        'min_outliers': min_outliers,  # 异常值-极小值-不排名
        'spec_num': None,  # 特殊值
        'spec_means': None,  # 特殊值的含义
        'spec_rank': None,  # 特殊值是否排名 (Y:参与排名, N:不参与排名)
        'spec_show': None,  # 特殊值的前端展示
        'table_name': '',
        'periods': m_p_list
    },
    'sharpe': {
        'name': '年化夏普比率',
        'index_order': 'B',  # 排名方向: B越大排名越靠前, S越小排名越靠前
        'index_unit': '',  # 指标的单位
        'index_process': '-',  # 指标展示前的处理方式
        'max_outliers': max_outliers,  # 异常值-极大值-不排名
        'min_outliers': min_outliers,  # 异常值-极小值-不排名
        'spec_num': None,  # 特殊值
        'spec_means': None,  # 特殊值的含义
        'spec_rank': None,  # 特殊值是否排名 (Y:参与排名, N:不参与排名)
        'spec_show': None,  # 特殊值的前端展示
        'table_name': '',
        'periods': m_p_list
    },
    'Calmar_not_ann': {
        'name': '卡玛比率',
        'index_order': 'B',  # 排名方向: B越大排名越靠前, S越小排名越靠前
        'index_unit': '',  # 指标的单位
        'index_process': '-',  # 指标展示前的处理方式
        'max_outliers': max_outliers,  # 异常值-极大值-不排名
        'min_outliers': min_outliers,  # 异常值-极小值-不排名
        'spec_num': None,  # 特殊值
        'spec_means': None,  # 特殊值的含义
        'spec_rank': None,  # 特殊值是否排名 (Y:参与排名, N:不参与排名)
        'spec_show': None,  # 特殊值的前端展示
        'table_name': '',
        'periods': m_p_list
    },
    'Calmar': {
        'name': '年化卡玛比率',
        'index_order': 'B',  # 排名方向: B越大排名越靠前, S越小排名越靠前
        'index_unit': '',  # 指标的单位
        'index_process': '-',  # 指标展示前的处理方式
        'max_outliers': max_outliers,  # 异常值-极大值-不排名
        'min_outliers': min_outliers,  # 异常值-极小值-不排名
        'spec_num': None,  # 特殊值
        'spec_means': None,  # 特殊值的含义
        'spec_rank': None,  # 特殊值是否排名 (Y:参与排名, N:不参与排名)
        'spec_show': None,  # 特殊值的前端展示
        'table_name': '',
        'periods': m_p_list
    },
    'InfoRatio_not_ann': {
        'name': '信息比率',
        'index_order': 'B',  # 排名方向: B越大排名越靠前, S越小排名越靠前
        'index_unit': '',  # 指标的单位
        'index_process': '-',  # 指标展示前的处理方式
        'max_outliers': max_outliers,  # 异常值-极大值-不排名
        'min_outliers': min_outliers,  # 异常值-极小值-不排名
        'spec_num': None,  # 特殊值
        'spec_means': None,  # 特殊值的含义
        'spec_rank': None,  # 特殊值是否排名 (Y:参与排名, N:不参与排名)
        'spec_show': None,  # 特殊值的前端展示
        'table_name': '',
        'periods': m_p_list
    },
    'InfoRatio': {
        'name': '年化信息比率',
        'index_order': 'B',  # 排名方向: B越大排名越靠前, S越小排名越靠前
        'index_unit': '',  # 指标的单位
        'index_process': '-',  # 指标展示前的处理方式
        'max_outliers': max_outliers,  # 异常值-极大值-不排名
        'min_outliers': min_outliers,  # 异常值-极小值-不排名
        'spec_num': None,  # 特殊值
        'spec_means': None,  # 特殊值的含义
        'spec_rank': None,  # 特殊值是否排名 (Y:参与排名, N:不参与排名)
        'spec_show': None,  # 特殊值的前端展示
        'table_name': '',
        'periods': m_p_list
    },

    # ------------- 最大涨幅相关指标 -------------
    'LargestContinueRaisingRate': {
        'name': '最大涨幅幅度',
        'index_order': 'B',  # 排名方向: B越大排名越靠前, S越小排名越靠前
        'index_unit': '%',  # 指标的单位
        'index_process': '*100',  # 指标展示前的处理方式
        'max_outliers': max_outliers,  # 异常值-极大值-不排名
        'min_outliers': min_outliers,  # 异常值-极小值-不排名
        'spec_num': None,  # 特殊值
        'spec_means': None,  # 特殊值的含义
        'spec_rank': None,  # 特殊值是否排名 (Y:参与排名, N:不参与排名)
        'spec_show': None,  # 特殊值的前端展示
        'table_name': '',
        'periods': m_p_list
    },
    'LargestContinueRaisingStart': {
        'name': '最大涨幅开始日期',
        'index_order': '',  # 排名方向: B越大排名越靠前, S越小排名越靠前
        'index_unit': 'yyyymmdd',  # 指标的单位
        'index_process': '转日期',  # 指标展示前的处理方式
        'max_outliers': max_outliers,  # 异常值-极大值-不排名
        'min_outliers': min_outliers,  # 异常值-极小值-不排名
        'spec_num': None,  # 特殊值
        'spec_means': None,  # 特殊值的含义
        'spec_rank': None,  # 特殊值是否排名 (Y:参与排名, N:不参与排名)
        'spec_show': None,  # 特殊值的前端展示
        'table_name': '',
        'periods': m_p_list
    },
    'LargestContinueRaisingEnd': {
        'name': '最大涨幅结束日期',
        'index_order': '',  # 排名方向: B越大排名越靠前, S越小排名越靠前
        'index_unit': 'yyyymmdd',  # 指标的单位
        'index_process': '转日期',  # 指标展示前的处理方式
        'max_outliers': max_outliers,  # 异常值-极大值-不排名
        'min_outliers': min_outliers,  # 异常值-极小值-不排名
        'spec_num': None,  # 特殊值
        'spec_means': None,  # 特殊值的含义
        'spec_rank': None,  # 特殊值是否排名 (Y:参与排名, N:不参与排名)
        'spec_show': None,  # 特殊值的前端展示
        'table_name': '',
        'periods': m_p_list
    },
    'LargestContinueRaisingDays': {
        'name': '最大涨幅天数',
        'index_order': 'B',  # 排名方向: B越大排名越靠前, S越小排名越靠前
        'index_unit': '天',  # 指标的单位
        'index_process': '-',  # 指标展示前的处理方式
        'max_outliers': max_outliers,  # 异常值-极大值-不排名
        'min_outliers': min_outliers,  # 异常值-极小值-不排名
        'spec_num': None,  # 特殊值
        'spec_means': None,  # 特殊值的含义
        'spec_rank': None,  # 特殊值是否排名 (Y:参与排名, N:不参与排名)
        'spec_show': None,  # 特殊值的前端展示
        'table_name': '',
        'periods': m_p_list
    },

    # ------------- 最大跌幅相关指标 -------------
    'LargestContinueLossingRate': {  # 该指标未取绝对值
        'name': '最大跌幅幅度',
        'index_order': 'B',  # 排名方向: B越大排名越靠前, S越小排名越靠前
        'index_unit': '%',  # 指标的单位
        'index_process': '*100',  # 指标展示前的处理方式
        'max_outliers': max_outliers,  # 异常值-极大值-不排名
        'min_outliers': min_outliers,  # 异常值-极小值-不排名
        'spec_num': None,  # 特殊值
        'spec_means': None,  # 特殊值的含义
        'spec_rank': None,  # 特殊值是否排名 (Y:参与排名, N:不参与排名)
        'spec_show': None,  # 特殊值的前端展示
        'table_name': '',
        'periods': m_p_list
    },
    'LargestContinueLossingStart': {
        'name': '最大跌幅开始日期',
        'index_order': '',  # 排名方向: B越大排名越靠前, S越小排名越靠前
        'index_unit': 'yyyymmdd',  # 指标的单位
        'index_process': '转日期',  # 指标展示前的处理方式
        'max_outliers': max_outliers,  # 异常值-极大值-不排名
        'min_outliers': min_outliers,  # 异常值-极小值-不排名
        'spec_num': None,  # 特殊值
        'spec_means': None,  # 特殊值的含义
        'spec_rank': None,  # 特殊值是否排名 (Y:参与排名, N:不参与排名)
        'spec_show': None,  # 特殊值的前端展示
        'table_name': '',
        'periods': m_p_list
    },
    'LargestContinueLossingEnd': {
        'name': '最大跌幅结束日期',
        'index_order': '',  # 排名方向: B越大排名越靠前, S越小排名越靠前
        'index_unit': 'yyyymmdd',  # 指标的单位
        'index_process': '转日期',  # 指标展示前的处理方式
        'max_outliers': max_outliers,  # 异常值-极大值-不排名
        'min_outliers': min_outliers,  # 异常值-极小值-不排名
        'spec_num': None,  # 特殊值
        'spec_means': None,  # 特殊值的含义
        'spec_rank': None,  # 特殊值是否排名 (Y:参与排名, N:不参与排名)
        'spec_show': None,  # 特殊值的前端展示
        'table_name': '',
        'periods': m_p_list
    },
    'LargestContinueLossingDays': {
        'name': '最大跌幅天数',
        'index_order': 'S',  # 排名方向: B越大排名越靠前, S越小排名越靠前
        'index_unit': 'yyyymmdd',  # 指标的单位
        'index_process': '-',  # 指标展示前的处理方式
        'max_outliers': max_outliers,  # 异常值-极大值-不排名
        'min_outliers': min_outliers,  # 异常值-极小值-不排名
        'spec_num': None,  # 特殊值
        'spec_means': None,  # 特殊值的含义
        'spec_rank': None,  # 特殊值是否排名 (Y:参与排名, N:不参与排名)
        'spec_show': None,  # 特殊值的前端展示
        'table_name': '',
        'periods': m_p_list
    },

    # ------------- Alpha 和 Beta -------------
    'alpha_not_ann': {
        'name': '詹森alpha',
        'index_order': 'B',  # 排名方向: B越大排名越靠前, S越小排名越靠前
        'index_unit': '%',  # 指标的单位
        'index_process': '*100',  # 指标展示前的处理方式
        'max_outliers': max_outliers,  # 异常值-极大值-不排名
        'min_outliers': min_outliers,  # 异常值-极小值-不排名
        'spec_num': None,  # 特殊值
        'spec_means': None,  # 特殊值的含义
        'spec_rank': None,  # 特殊值是否排名 (Y:参与排名, N:不参与排名)
        'spec_show': None,  # 特殊值的前端展示
        'table_name': '',
        'periods': m_p_list
    },
    'alpha': {
        'name': '年化詹森alpha',
        'index_order': 'B',  # 排名方向: B越大排名越靠前, S越小排名越靠前
        'index_unit': '%',  # 指标的单位
        'index_process': '*100',  # 指标展示前的处理方式
        'max_outliers': max_outliers,  # 异常值-极大值-不排名
        'min_outliers': min_outliers,  # 异常值-极小值-不排名
        'spec_num': None,  # 特殊值
        'spec_means': None,  # 特殊值的含义
        'spec_rank': None,  # 特殊值是否排名 (Y:参与排名, N:不参与排名)
        'spec_show': None,  # 特殊值的前端展示
        'table_name': '',
        'periods': m_p_list
    },
    'beta': {
        'name': '市场风险beta',
        'index_order': 'S',  # 排名方向: B越大排名越靠前, S越小排名越靠前
        'index_unit': '',  # 指标的单位
        'index_process': '-',  # 指标展示前的处理方式
        'max_outliers': max_outliers,  # 异常值-极大值-不排名
        'min_outliers': min_outliers,  # 异常值-极小值-不排名
        'spec_num': None,  # 特殊值
        'spec_means': None,  # 特殊值的含义
        'spec_rank': None,  # 特殊值是否排名 (Y:参与排名, N:不参与排名)
        'spec_show': None,  # 特殊值的前端展示
        'table_name': '',
        'periods': m_p_list
    },
}


def get_indicator_registry(frequency: str):
    """
    根据频率动态生成指标注册表，使用正确的年化因子。

    Args:
        frequency (str): 'YS', 'W', or 'M'

    Returns:
        dict: The indicator registry with correct annualization factors.
    """
    freq_upper = frequency.upper()
    if freq_upper == 'YS':
        ann_return_base = 365
        ann_std_base = 252
    elif freq_upper == 'W':
        ann_return_base = 52
        ann_std_base = 52
    elif freq_upper == 'M':
        ann_return_base = 12
        ann_std_base = 12
    else:
        # 默认为YS的配置
        ann_return_base = 365
        ann_std_base = 252

    # 指标类型, 对应函数, 以及上游依赖
    indicator_registry = {
        # --- 收益率指标 ---
        'TotalReturn': {
            'kind': 'segment_reduce',
            'source': ['pct_array'],
            'depends_on': [],
            'func': ind_total_return,
        },
        'AnnualReturn': {
            'kind': 'vector_transform',
            'depends_on': ['TotalReturn'],
            'func': calc_annualized_return,
            'annual_factor': ann_return_base,
            'day_count_source': 'n_days'  # 使用区间内自然日天数计算年化
        },
        'TotalReturn_bm': {
            'kind': 'segment_reduce',
            'source': ['bench_array'],
            'depends_on': [],
            'func': ind_total_return,
        },
        'AnnualReturn_bm': {
            'kind': 'vector_transform',
            'depends_on': ['TotalReturn_bm'],
            'func': calc_bm_annualized_return,
            'annual_factor': ann_return_base,
            'day_count_source': 'n_days'  # 使用区间内自然日天数计算年化
        },
        'TotalReturn_rf': {
            'kind': 'segment_reduce',
            'source': ['rf_array'],
            'depends_on': [],
            'func': ind_total_return,
        },
        'RelaRe': {
            'kind': 'vector_transform',
            'depends_on': ['TotalReturn', 'TotalReturn_bm'],
            'func': cal_relative_return,
        },
        'RelaReAnn': {
            'kind': 'vector_transform',
            'depends_on': ['AnnualReturn', 'AnnualReturn_bm'],
            'func': cal_relative_ann_return,
        },
        'MeanReturn': {
            'kind': 'segment_reduce',
            'source': ['pct_array'],
            'func': ind_mean_return,
        },
        'MeanReturn_bm': {
            'kind': 'segment_reduce',
            'source': ['bench_array'],
            'func': ind_mean_return,
        },

        # --- 胜率指标 ---
        'AbsoluteOdds': {
            'kind': 'segment_reduce',
            'source': ['pct_array'],
            'depends_on': [],
            'func': ind_abs_odds,
            'day_count_source': 'n_pts'  # 使用区间内收益率数据点位数量作为分母进行计算
        },

        # --- 波动率指标 ---
        'prodStd': {
            'kind': 'segment_reduce',
            'source': ['pct_array'],
            'depends_on': [],
            'func': ind_std,
        },
        'prodAnnualStd': {
            'kind': 'vector_transform',
            'depends_on': ['prodStd'],
            'func': cal_annualized_std,
            'annual_factor': ann_std_base,
        },
        'DownsideStd': {
            'kind': 'vector_transform',
            'source': [],
            'depends_on': ['DownsideStd_not_ann'],
            'func': cal_downside_std,
            'annual_factor': ann_std_base,
        },
        'DownsideStd_not_ann': {
            'kind': 'segment_reduce',
            'source': ['pct_array', 'rf_array'],
            'depends_on': [],
            'func': ind_downside_std,
        },

        # --- 最大回撤相关 ---
        'prodMddr': {
            'kind': 'segment_reduce',
            'source': ['pct_array'],
            'func': ind_mdd_and_recovery,
            'outputs': ['prodMddr', 'MddrRecoverTime'],
            'day_count_source': 'n_pts'
        },
        'MddrRecoverTime': {
            'kind': 'segment_reduce',
            'source': ['pct_array'],
            'func': ind_mdd_and_recovery,
            'outputs': ['prodMddr', 'MddrRecoverTime'],
            'day_count_source': 'n_pts'
        },

        # --- 投资性价比 ---
        'sharpe_not_ann': {
            'kind': 'vector_transform',
            'source': [],
            'depends_on': ['TotalReturn', 'prodStd', 'TotalReturn_rf'],
            'func': cal_sharpe_ratio_not_ann,
        },
        'sharpe': {
            'kind': 'vector_transform',
            'depends_on': ['AnnualReturn', 'prodAnnualStd'],
            'func': cal_sharpe_ratio,
            'ann_rf': 0.015,  # 年化无风险收益率
        },
        'Calmar_not_ann': {
            'kind': 'vector_transform',
            'depends_on': ['TotalReturn', 'prodMddr'],
            'func': cal_calmar_ratio_not_ann,
        },
        'Calmar': {
            'kind': 'vector_transform',
            'depends_on': ['AnnualReturn', 'prodMddr'],
            'func': cal_calmar_ratio,
        },
        'ActiveRisk': {
            'kind': 'segment_reduce',
            'source': ['pct_array', 'bench_array'],
            'func': ind_tracking_error,
        },
        'AnnualActiveRisk': {
            'kind': 'vector_transform',
            'depends_on': ['ActiveRisk'],
            'func': cal_tracking_error,
            'annual_factor': ann_std_base,
        },
        'InfoRatio_not_ann': {
            'kind': 'vector_transform',
            'depends_on': ['RelaRe', 'ActiveRisk'],
            'func': cal_info_ratio_not_ann,
        },
        'InfoRatio': {
            'kind': 'vector_transform',
            'depends_on': ['RelaReAnn', 'AnnualActiveRisk'],
            'func': cal_info_ratio,
        },

        # --- Alpha 和 Beta ---
        'beta': {
            'kind': 'segment_reduce',
            'source': ['pct_array', 'bench_array'],
            'func': ind_beta,
        },
        'alpha_not_ann': {
            'kind': 'vector_transform',
            'depends_on': ['MeanReturn', 'MeanReturn_bm', 'beta'],
            'func': cal_alpha_not_ann,
        },
        'alpha': {
            'kind': 'vector_transform',
            'depends_on': ['alpha_not_ann'],
            'func': cal_alpha,
            'annual_factor': ann_return_base,
            'day_count_source': 'n_days'  # 使用区间内自然日天数计算年化
        },

        # --- 最大涨跌幅相关指标 ---
        'LargestContinueRaisingRate': {
            'kind': 'segment_reduce',
            'source': ['pct_array'],
            'depends_on': [],
            'outputs': ['LargestContinueRaisingRate', 'LargestContinueRaisingStart',
                        'LargestContinueRaisingEnd', 'LargestContinueRaisingDays'],
            'date_outputs': ['LargestContinueRaisingStart', 'LargestContinueRaisingEnd'],  # 显式声明日期类型的输出字段
            'func': ind_largest_continue_raising,
            'day_count_source': 't_days',
        },
        'LargestContinueRaisingStart': {
            'kind': 'segment_reduce',
            'source': ['pct_array'],
            'depends_on': [],
            'outputs': ['LargestContinueRaisingRate', 'LargestContinueRaisingStart',
                        'LargestContinueRaisingEnd', 'LargestContinueRaisingDays'],
            'date_outputs': ['LargestContinueRaisingStart', 'LargestContinueRaisingEnd'],  # 显式声明日期类型的输出字段
            'func': ind_largest_continue_raising,
            'day_count_source': 't_days',
        },
        'LargestContinueRaisingEnd': {
            'kind': 'segment_reduce',
            'source': ['pct_array'],
            'depends_on': [],
            'outputs': ['LargestContinueRaisingRate', 'LargestContinueRaisingStart',
                        'LargestContinueRaisingEnd', 'LargestContinueRaisingDays'],
            'date_outputs': ['LargestContinueRaisingStart', 'LargestContinueRaisingEnd'],  # 显式声明日期类型的输出字段
            'func': ind_largest_continue_raising,
            'day_count_source': 't_days',
        },
        'LargestContinueRaisingDays': {
            'kind': 'segment_reduce',
            'source': ['pct_array'],
            'depends_on': [],
            'outputs': ['LargestContinueRaisingRate', 'LargestContinueRaisingStart',
                        'LargestContinueRaisingEnd', 'LargestContinueRaisingDays'],
            'date_outputs': ['LargestContinueRaisingStart', 'LargestContinueRaisingEnd'],  # 显式声明日期类型的输出字段
            'func': ind_largest_continue_raising,
            'day_count_source': 't_days',
        },
        'LargestContinueLossingRate': {
            'kind': 'segment_reduce',
            'source': ['pct_array'],
            'outputs': ['LargestContinueLossingRate', 'LargestContinueLossingStart',
                        'LargestContinueLossingEnd', 'LargestContinueLossingDays'],
            'date_outputs': ['LargestContinueLossingStart', 'LargestContinueLossingEnd'],
            'depends_on': [],
            'func': ind_largest_continue_falling,
        },
        'LargestContinueLossingStart': {
            'kind': 'segment_reduce',
            'source': ['pct_array'],
            'outputs': ['LargestContinueLossingRate', 'LargestContinueLossingStart',
                        'LargestContinueLossingEnd', 'LargestContinueLossingDays'],
            'date_outputs': ['LargestContinueLossingStart', 'LargestContinueLossingEnd'],
            'depends_on': [],
            'func': ind_largest_continue_falling,
        },
        'LargestContinueLossingEnd': {
            'kind': 'segment_reduce',
            'source': ['pct_array'],
            'outputs': ['LargestContinueLossingRate', 'LargestContinueLossingStart',
                        'LargestContinueLossingEnd', 'LargestContinueLossingDays'],
            'date_outputs': ['LargestContinueLossingStart', 'LargestContinueLossingEnd'],
            'depends_on': [],
            'func': ind_largest_continue_falling,
        },
        'LargestContinueLossingDays': {
            'kind': 'segment_reduce',
            'source': ['pct_array'],
            'outputs': ['LargestContinueLossingRate', 'LargestContinueLossingStart',
                        'LargestContinueLossingEnd', 'LargestContinueLossingDays'],
            'date_outputs': ['LargestContinueLossingStart', 'LargestContinueLossingEnd'],
            'depends_on': [],
            'func': ind_largest_continue_falling,
            'day_count_source': 'n_pts'
        },
    }
    return indicator_registry
