def pinyin_2_hanzi(pinyinList):
    from Pinyin2Hanzi import DefaultDagParams
    from Pinyin2Hanzi import dag

    dagParams = DefaultDagParams()
    # path_num：候选值，可设置一个或多个
    result = dag(dagParams, ['我'], path_num=100, log=True)
    for item in result:
        # socre = item.score # 得分
        res = item.path  # 转换结果
        print(res)


if __name__ == '__main__':
    lists = ['ni', 'hao', 'ya']
    pinyin_2_hanzi(lists)

# 输出结果
['你', '好呀']
['你好', '压']
['你好', '亚']
['你好', '雅']
['你好', '牙']
['你好', '涯']
['你好', '呀']
['你好', '丫']
['你好', '讶']
['你好', '押']