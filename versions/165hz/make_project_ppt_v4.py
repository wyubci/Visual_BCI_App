import os
import win32com.client as win32


OUT = r"C:\Users\adam\Desktop\基于脑机接口与实时视觉反馈的交互系统_立项报告_优化深化版.pptx"
IMG_DIR = os.path.abspath("申报书_images")


def rgb(r, g, b):
    return int(r) + int(g) * 256 + int(b) * 65536


def img(name):
    return os.path.join(IMG_DIR, name)


def add_text(slide, content, x, y, w, h, size=20, color=(255, 255, 255), bold=False, align=1):
    box = slide.Shapes.AddTextbox(1, x, y, w, h)
    box.TextFrame.MarginLeft = 8
    box.TextFrame.MarginRight = 8
    box.TextFrame.MarginTop = 4
    box.TextFrame.MarginBottom = 4
    tr = box.TextFrame.TextRange
    tr.Text = content
    tr.Font.Name = "Microsoft YaHei"
    tr.Font.Size = size
    tr.Font.Bold = -1 if bold else 0
    tr.Font.Color.RGB = rgb(*color)
    tr.ParagraphFormat.Alignment = align
    return box


def add_rect(slide, x, y, w, h, fill=(12, 45, 82), line=(0, 232, 255), transparency=0.08):
    shape = slide.Shapes.AddShape(1, x, y, w, h)
    shape.Fill.ForeColor.RGB = rgb(*fill)
    shape.Fill.Transparency = transparency
    shape.Line.ForeColor.RGB = rgb(*line)
    shape.Line.Transparency = 0.15
    shape.Line.Weight = 1.15
    return shape


def add_bg(slide, prs):
    slide.FollowMasterBackground = False
    slide.Background.Fill.ForeColor.RGB = rgb(3, 20, 52)
    for i, col in enumerate([(0, 58, 105), (0, 101, 145), (0, 210, 235)]):
        band = slide.Shapes.AddShape(1, -70, 416 + i * 27, prs.PageSetup.SlideWidth + 140, 120)
        band.Fill.ForeColor.RGB = rgb(*col)
        band.Fill.Transparency = 0.62 + i * 0.08
        band.Line.Visible = 0
    rail = slide.Shapes.AddLine(36, 28, 36, 508)
    rail.Line.ForeColor.RGB = rgb(0, 240, 255)
    rail.Line.Transparency = 0.30
    rail.Line.Weight = 1.2
    add_text(slide, "WYU · BCI", 795, 25, 120, 24, 13, (228, 248, 255), True, 2)


def add_header(slide, section, page_no=None):
    add_text(slide, "项目内容", 54, 31, 155, 38, 29, (255, 255, 255), True)
    line = slide.Shapes.AddLine(218, 54, 292, 54)
    line.Line.ForeColor.RGB = rgb(255, 255, 255)
    line.Line.Weight = 2.1
    add_text(slide, section, 306, 34, 420, 32, 23, (255, 255, 255), True)
    if page_no:
        add_text(slide, f"{page_no:02d}", 870, 470, 48, 30, 18, (0, 240, 255), True, 2)


def new_slide(prs, section=None, page_no=None):
    s = prs.Slides.Add(prs.Slides.Count + 1, 12)
    add_bg(s, prs)
    if section:
        add_header(s, section, page_no)
    return s


def add_picture_fit(slide, path, x, y, w, h, pad=6, panel=True):
    if panel:
        add_rect(slide, x, y, w, h, fill=(246, 250, 255), line=(0, 224, 255), transparency=0)
    pic = slide.Shapes.AddPicture(os.path.abspath(path), 0, -1, x + pad, y + pad, w - 2 * pad, h - 2 * pad)
    try:
        pic.LockAspectRatio = -1
    except Exception:
        pass
    pic.Width = w - 2 * pad
    if pic.Height > h - 2 * pad:
        pic.Height = h - 2 * pad
    pic.Left = x + (w - pic.Width) / 2
    pic.Top = y + (h - pic.Height) / 2
    return pic


def add_card(slide, x, y, w, h, title, body, accent=(0, 240, 255), body_size=16):
    add_rect(slide, x, y, w, h)
    add_text(slide, title, x + 12, y + 10, w - 24, 28, 21, accent, True)
    add_text(slide, body, x + 14, y + 48, w - 28, h - 56, body_size, (238, 246, 255))


def add_plan_box(slide, x, y, w, h, period, body):
    add_rect(slide, x, y, w, h)
    add_text(slide, period, x + 10, y + 12, w - 20, 30, 20, (0, 240, 255), True, 2)
    add_text(slide, body, x + 18, y + 55, w - 36, h - 62, 16, (245, 250, 255), False, 2)


def add_tag(slide, label, x, y, w, h, color=(0, 240, 255)):
    r = slide.Shapes.AddShape(5, x, y, w, h)
    r.Fill.ForeColor.RGB = rgb(4, 38, 72)
    r.Fill.Transparency = 0.08
    r.Line.ForeColor.RGB = rgb(*color)
    r.Line.Weight = 1.2
    add_text(slide, label, x + 8, y + 6, w - 16, h - 10, 15, (255, 255, 255), True, 2)


def add_arrow(slide, x1, y1, x2, y2, color=(0, 240, 255)):
    line = slide.Shapes.AddLine(x1, y1, x2, y2)
    line.Line.ForeColor.RGB = rgb(*color)
    line.Line.Weight = 2.2


def paragraph_slide(prs, section, title, blocks, page_no, red_line=None):
    s = new_slide(prs, section, page_no)
    add_text(s, title, 72, 112, 810, 40, 28, (0, 246, 255), True)
    y = 178
    for head, body in blocks:
        add_text(s, head, 80, y, 760, 30, 22, (0, 246, 255), True)
        add_text(s, body, 110, y + 38, 760, 70, 20, (244, 248, 255))
        y += 128
    if red_line:
        add_text(s, red_line, 115, 436, 730, 36, 23, (255, 80, 80), True, 2)
    return s


def build():
    app = win32.Dispatch("KWPP.Application")
    try:
        app.Visible = True
    except Exception:
        pass
    if os.path.exists(OUT):
        try:
            os.remove(OUT)
        except OSError:
            pass
    prs = app.Presentations.Add()
    prs.PageSetup.SlideWidth = 960
    prs.PageSetup.SlideHeight = 540

    # 1 cover
    s = new_slide(prs)
    add_text(s, "基于脑机接口与实时视觉反馈的", 110, 155, 740, 58, 42, (0, 246, 255), True, 2)
    add_text(s, "交互系统", 310, 222, 340, 58, 42, (0, 246, 255), True, 2)
    add_text(s, "大学生创新训练计划项目立项报告", 250, 312, 460, 36, 24, (255, 255, 255), True, 2)
    add_text(s, "负责人：张连成    指导教师：王洪涛教授\n电子与信息工程学院 · 计算机科学与技术", 280, 374, 400, 54, 16, (232, 244, 255), False, 2)

    # 2 outline
    s = new_slide(prs, "汇报提纲", 2)
    outline = [
        ("01", "项目背景与需求", "脑机接口从抽象矩阵走向真实视觉场景"),
        ("02", "项目目的与意义", "面向助残康复、远程作业与智能交互"),
        ("03", "研究内容与关键问题", "时序对齐、样本治理、TDCA、门控决策"),
        ("04", "实施方案与技术路线", "动态刺激范式与在线闭环系统"),
        ("05", "基础条件与成果计划", "平台基础、实施进度与预期交付"),
    ]
    for i, (no, title, desc) in enumerate(outline):
        y = 112 + i * 70
        add_text(s, no, 115, y, 60, 34, 27, (0, 246, 255), True, 2)
        add_text(s, title, 205, y, 250, 32, 24, (255, 255, 255), True)
        add_text(s, desc, 480, y + 4, 370, 28, 17, (220, 236, 250))
        add_arrow(s, 172, y + 17, 198, y + 17)

    # 3 pure text background
    paragraph_slide(
        prs,
        "项目背景",
        "脑机接口技术正从“可识别”走向“可交互”",
        [
            ("脑机接口的应用基础",
             "BCI 通过脑电信号直接建立大脑与外部设备之间的通信通道。SSVEP 范式具有信噪比高、训练成本低、响应速度快等优势，是非侵入式 BCI 中较接近实用化的方向。"),
            ("传统系统的核心局限",
             "现有系统多采用固定位置、固定频率的静态闪烁块，用户需要先理解频率、按钮与动作之间的抽象映射，真实目标与控制指令之间存在割裂。"),
        ],
        3,
        "问题本质：交互对象不在真实场景中，用户体验和任务效率受限。"
    )

    # 4 background with visual solution
    s = new_slide(prs, "需求分析", 4)
    add_card(s, 70, 126, 250, 220, "传统 BCI", "固定矩阵\n抽象按钮\n用户需记忆映射\n真实目标与控制动作割裂", accent=(255, 85, 85), body_size=18)
    add_arrow(s, 350, 236, 438, 236)
    add_picture_fit(s, img("image5.jpeg"), 468, 102, 410, 290, pad=6)
    add_text(s, "本项目方案", 590, 410, 170, 28, 24, (0, 246, 255), True, 2)
    add_text(s, "把频率刺激叠加到摄像头画面中的真实目标上，形成“看见目标，即可选择目标”的交互方式。",
             450, 445, 430, 36, 18, (255, 255, 255), False, 2)

    # 5 pure text significance
    paragraph_slide(
        prs,
        "项目意义",
        "项目价值不仅在算法精度，更在真实场景闭环能力",
        [
            ("理论意义",
             "将 TDCA 放入摄像头实时目标驱动的动态视觉情境中，分析目标移动、背景变化、光照波动与注视点变化对 SSVEP 特征的影响，扩展算法适用边界。"),
            ("实践意义",
             "面向助残辅具、高危作业、智能座舱等场景，用户可直接注视真实对象触发控制，减少抽象指令学习负担，提高系统可用性与自然性。"),
        ],
        5,
        "目标：让 BCI 从“实验室识别”走向“任务级交互”。"
    )

    # 6 objectives
    s = new_slide(prs, "研究目标", 6)
    add_text(s, "围绕“实时、准确、稳定、可复现”四个目标展开", 105, 112, 750, 34, 27, (0, 246, 255), True, 2)
    goals = [
        ("实时", "短窗识别\n低延迟响应"),
        ("准确", "TDCA判别\n多子带融合"),
        ("稳定", "置信门控\n投票冷却"),
        ("可复现", "日志追踪\n参数留痕"),
    ]
    for i, (g, body) in enumerate(goals):
        x = 115 + i * 190
        add_card(s, x, 205, 150, 120, g, body, body_size=17)
        if i < 3:
            add_arrow(s, x + 155, 265, x + 184, 265)
    add_text(s, "最终目标：构建可在线运行、可闭环验证、可扩展应用的 TDCA-SSVEP 系统。",
             110, 410, 740, 36, 23, (255, 80, 80), True, 2)

    # 7 pure text research content
    paragraph_slide(
        prs,
        "研究内容",
        "围绕四个模块形成完整技术链路",
        [
            ("链路一致性与样本治理",
             "统一刺激起点、接收窗口与分析窗口，建立离线训练和在线解码一致的切窗口径；同时完成去直流、标准化、异常截断与无效窗剔除。"),
            ("TDCA 短窗解码与门控决策",
             "利用 TDCA 的判别投影能力提升短窗识别效果，再通过置信度、连续投票和冷却窗口抑制误触发，使分类结果能够稳定转化为控制指令。"),
        ],
        7,
        "优化对象：执行正确率、误触发率、漏触发率与端到端响应时延。"
    )

    # 8 key problems
    s = new_slide(prs, "关键问题", 8)
    add_picture_fit(s, img("image4.jpeg"), 65, 115, 520, 275, pad=5)
    add_text(s, "拟解决的关键问题", 635, 115, 250, 32, 25, (0, 246, 255), True)
    problems = [
        "时序错位导致样本不可分",
        "短窗信息不足与噪声放大",
        "误触发与漏触发并存",
        "训练口径与在线口径不一致",
        "工程链路耦合过高",
    ]
    y = 170
    for p in problems:
        add_text(s, "•", 635, y, 22, 24, 20, (0, 240, 255), True)
        add_text(s, p, 662, y, 250, 26, 17, (245, 250, 255))
        y += 40
    add_text(s, "解决抓手：统一时间戳 + 有效窗治理 + 门控决策", 135, 426, 690, 34, 22, (255, 80, 80), True, 2)

    # 9 implementation route
    s = new_slide(prs, "总体闭环", 9)
    add_picture_fit(s, img("image3.png"), 70, 105, 805, 275, pad=8)
    tags = [("采集接入", 98), ("缓存对齐", 252), ("TDCA识别", 406), ("门控决策", 560), ("设备执行", 714)]
    for label, x in tags:
        add_tag(s, label, x, 410, 120, 42, (255, 160, 70) if label == "门控决策" else (0, 240, 255))

    # 10 dynamic visual stimulation
    s = new_slide(prs, "动态刺激范式", 10)
    add_picture_fit(s, img("image5.jpeg"), 68, 100, 520, 335, pad=6)
    add_text(s, "动态摄像头范式", 625, 115, 250, 32, 25, (0, 246, 255), True)
    points = [
        "YOLO 轻量模型检测候选目标",
        "DeepSORT / ID 关联保持目标身份",
        "在目标边界叠加频率编码闪烁",
        "频率-标签-动作一一映射",
        "目标丢失时执行重分配策略",
    ]
    y = 170
    for p in points:
        add_text(s, "•", 620, y, 22, 24, 20, (0, 240, 255), True)
        add_text(s, p, 648, y, 250, 26, 17, (245, 250, 255))
        y += 39
    add_text(s, "用户注视真实目标，系统输出对应控制指令。", 100, 455, 760, 30, 22, (255, 255, 255), True, 2)

    # 11 data processing
    s = new_slide(prs, "数据处理", 11)
    add_picture_fit(s, img("image6.jpeg"), 66, 96, 365, 382, pad=6)
    add_text(s, "处理流水线", 475, 108, 220, 32, 25, (0, 246, 255), True)
    flow = [
        "事件对齐与有效窗截取",
        "去直流、标准化、幅值约束",
        "滤波器组分解多频段信息",
        "时间延迟扩展增强短窗特征",
        "模板相关评分与子带加权融合",
    ]
    y = 166
    for p in flow:
        add_text(s, "•", 475, y, 22, 24, 20, (0, 240, 255), True)
        add_text(s, p, 505, y, 340, 26, 18, (245, 250, 255))
        y += 45
    add_text(s, "原则：拒绝脏样本，保证训练与在线输入一致。", 475, 420, 365, 30, 21, (255, 80, 80), True, 2)

    # 12 model and decision
    s = new_slide(prs, "模型与决策", 12)
    add_picture_fit(s, img("image7.jpeg"), 72, 112, 405, 185, pad=5)
    add_picture_fit(s, img("image9.jpeg"), 72, 320, 405, 145, pad=5)
    add_text(s, "TDCA 负责“分得开”", 525, 120, 300, 30, 24, (0, 246, 255), True)
    add_text(s, "通过判别投影最大化类间差异、压缩类内波动，在短窗条件下提升类别可分性。", 525, 160, 315, 64, 17)
    add_text(s, "门控策略负责“不误触”", 525, 270, 300, 30, 24, (0, 246, 255), True)
    add_text(s, "用 Top1-Top2 分数差计算置信度，再叠加连续投票与冷却窗口，降低单次噪声带来的错误执行。", 525, 310, 315, 78, 17)
    add_tag(s, "准确率", 530, 430, 90, 36)
    add_tag(s, "误触发率", 645, 430, 105, 36, (255, 160, 70))
    add_tag(s, "响应时延", 775, 430, 105, 36)

    # 13 basis
    s = new_slide(prs, "基础条件", 13)
    add_picture_fit(s, img("image13.png"), 58, 105, 430, 288, pad=0)
    add_picture_fit(s, img("image11.png"), 525, 105, 290, 128, pad=4)
    add_picture_fit(s, img("image12.png"), 525, 255, 290, 128, pad=4)
    add_card(s, 72, 418, 225, 66, "技术基础", "已具备实时 EEG 接入、TDCA/CCA/FBCCA 对照链路。", body_size=14)
    add_card(s, 330, 418, 225, 66, "平台基础", "ROSMaster 小车、神舞 EEG 设备与刺激终端已就绪。", body_size=14)
    add_card(s, 588, 418, 225, 66, "指导基础", "导师团队具备脑机接口、模式识别与混合智能研究积累。", body_size=14)

    # 14 plan
    s = new_slide(prs, "计划成果", 14)
    add_text(s, "实施计划", 72, 112, 150, 30, 26, (0, 246, 255), True)
    plan = [
        ("2026.04-06", "范式设计\n采集联调"),
        ("2026.07-10", "算法优化\n参数消融"),
        ("2026.11-\n2027.03", "动态场景\n闭环实验"),
        ("2027.04-06", "成果整理\n结题答辩"),
    ]
    for i, (t, body) in enumerate(plan):
        x = 86 + i * 210
        add_plan_box(s, x, 165, 172, 116, t, body)
        if i < 3:
            add_arrow(s, x + 176, 223, x + 205, 223)
    add_text(s, "预期成果", 72, 332, 150, 30, 26, (0, 246, 255), True)
    add_tag(s, "创新训练报告", 95, 385, 160, 44)
    add_tag(s, "软件作品", 300, 385, 130, 44)
    add_tag(s, "实验数据与日志", 475, 385, 170, 44)
    add_tag(s, "竞赛 / 论文成果", 690, 385, 170, 44)
    add_text(s, "汇报完毕  恳请指导", 250, 470, 460, 42, 34, (255, 255, 255), True, 2)

    prs.SaveAs(OUT)
    prs.Close()
    try:
        app.Quit()
    except Exception:
        pass
    print(OUT)


if __name__ == "__main__":
    build()
